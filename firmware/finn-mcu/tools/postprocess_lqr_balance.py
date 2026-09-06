#!/usr/bin/env python3
"""Analyze a captured Finn LQR balance run and emit sim-correcting derived data.

A closed-loop balance run is a poor system-identification experiment and this
script is built around admitting that. The controller actively suppresses the
motion that would excite the plant, so most regressors arrive collinear. Every
estimate below therefore carries an observability verdict, and a parameter whose
excitation was inadequate is reported as not identified rather than fitted
anyway. See hard rules 6 and 7 in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEASUREMENTS = REPO_ROOT / "sim" / "config" / "finn_measurements.yaml"
DEFAULT_SEEDED_HEADER = (
    REPO_ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance" / "lqr_seeded_config.h"
)

EXPECTED_SCHEMA = "lqr_v2"
CONTROL_PERIOD_US = 10000.0
GRAVITY_M_S2 = 9.80665

# A parameter needs both enough samples and enough spread in its regressor before
# a fit means anything. These thresholds are what separate "identified" from
# "the controller held it still and the fit is reading noise".
MIN_STEADY_SAMPLES = 50
MIN_FIT_SAMPLES = 100
MAX_CONDITION_NUMBER = 30.0
MIN_FIT_R2 = 0.5
MIN_TORQUE_RANGE_NM = 0.05
MIN_PITCH_RANGE_RAD = 0.01
STEADY_PITCH_RATE_RAD_S = 0.20
STEADY_FORWARD_VEL_M_S = 0.05
SETTLE_S = 0.5
MAX_LAG_TICKS = 25


class LqrBalancePostprocessError(Exception):
    """Expected failure with a concise user-facing message."""


def portable_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _round(value: float | None, ndigits: int = 8) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, ndigits)


# ---------------------------------------------------------------- run loading


@dataclass
class Run:
    schema: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    events: list[dict[str, str]] = field(default_factory=list)
    checks: list[dict[str, str]] = field(default_factory=list)
    malformed_rows: int = 0
    repeated_headers: int = 0


def read_run(run_dir: Path) -> Run:
    telemetry = run_dir / "telemetry.csv"
    if not telemetry.is_file():
        raise LqrBalancePostprocessError(f"missing telemetry: {portable_path(telemetry)}")

    schema: str | None = None
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    malformed = 0
    repeated_headers = 0

    for raw in telemetry.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("schema,"):
            schema = line.split(",", 1)[1].strip()
            continue
        if not line.startswith("data,"):
            continue
        fields = line.split(",")[1:]
        if fields and fields[0] == "t_us":
            if columns:
                repeated_headers += 1
            columns = fields
            continue
        if not columns or len(fields) != len(columns):
            malformed += 1
            continue
        rows.append(_coerce_row(dict(zip(columns, fields, strict=True))))

    if schema is None:
        raise LqrBalancePostprocessError(f"no schema line in {portable_path(telemetry)}")
    if schema != EXPECTED_SCHEMA:
        raise LqrBalancePostprocessError(
            f"schema {schema!r} is not {EXPECTED_SCHEMA!r}; this run came from different firmware"
        )
    if not rows:
        raise LqrBalancePostprocessError(f"no telemetry rows in {portable_path(telemetry)}")

    events, checks = read_events(run_dir / "events.log")
    return Run(schema, columns, rows, events, checks, malformed, repeated_headers)


TEXT_COLUMNS = frozenset({"state", "phase", "model_sha256", "fault_reason"})


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in row.items():
        if key in TEXT_COLUMNS:
            coerced[key] = value
            continue
        try:
            coerced[key] = float(value)
        except ValueError:
            coerced[key] = math.nan
    return coerced


def read_events(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    events: list[dict[str, str]] = []
    checks: list[dict[str, str]] = []
    if not path.is_file():
        return events, checks
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        parts = line.split(",")
        if line.startswith("event,") and len(parts) >= 5:
            events.append(
                {"t_us": parts[1], "event": parts[2], "state": parts[3], "detail": parts[4]}
            )
        elif line.startswith("check,") and len(parts) >= 7:
            checks.append(
                {
                    "t_us": parts[1],
                    "name": parts[2],
                    "result": parts[3],
                    "measured": parts[4],
                    "limit": parts[5],
                    "detail": parts[6],
                }
            )
    return events, checks


def load_measurements(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LqrBalancePostprocessError(f"missing measurements: {portable_path(path)}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def measurement_value(data: dict[str, Any], *keys: str) -> float | None:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    if isinstance(node, dict):
        node = node.get("value")
    try:
        number = float(node)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_seeded_header(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for name, raw in re.findall(r"constexpr\s+\w+\s+(k\w+)\s*=\s*([^;]+);", text):
        cleaned = raw.strip().rstrip("fUL")
        try:
            values[name] = float(cleaned)
        except ValueError:
            continue
    return values


# ------------------------------------------------------------------ integrity


def rotation_vector_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often a fused quaternion actually arrived.

    `imu_ok` is `isImuAlive()` and reads true on any report, so a dead rotation
    vector hides behind a healthy gyro stream. `isImuFresh()` gates arming on the
    quaternion alone, so this is the number that decides whether Finn can arm.
    """

    if len(rows) < 2:
        return {"arrivals": 0, "rate_hz": None, "max_age_ms": None}
    arrivals: list[float] = []
    for row in rows:
        stamp_us = row.get("t_us", math.nan) - row.get("imu_age_us", math.nan)
        if not math.isfinite(stamp_us):
            continue
        stamp_s = stamp_us / 1e6
        if not arrivals or abs(stamp_s - arrivals[-1]) > 0.05:
            arrivals.append(stamp_s)
    span_s = (rows[-1]["t_us"] - rows[0]["t_us"]) / 1e6
    ages = [r.get("imu_age_ms", math.nan) for r in rows]
    finite_ages = [a for a in ages if math.isfinite(a)]
    return {
        "arrivals": len(arrivals),
        "rate_hz": _round(len(arrivals) / span_s, 4) if span_s > 0 else None,
        "max_age_ms": _round(max(finite_ages), 1) if finite_ages else None,
    }


def _spread(values: np.ndarray | None) -> dict[str, float | None]:
    """Mean, max, and jitter, or all-None when the phase produced no samples."""
    if values is None or not values.size:
        return {"mean": None, "max": None, "jitter_us": None}
    return {
        "mean": _round(float(np.mean(values)), 2),
        "max": _round(float(np.max(values)), 2),
        "jitter_us": _round(float(np.std(values)), 2),
    }


def audit_integrity(run: Run) -> dict[str, Any]:
    """Was this run recorded well enough to trust anything derived from it?"""

    times_us = np.array([r["t_us"] for r in run.rows], dtype=float)
    dt_ms = np.diff(times_us) / 1000.0 if len(times_us) >= 2 else np.array([])
    numeric = [c for c in run.columns if c not in TEXT_COLUMNS]
    nonfinite = {
        column: int(sum(1 for r in run.rows if not math.isfinite(r.get(column, math.nan))))
        for column in numeric
    }
    nonfinite = {k: v for k, v in nonfinite.items() if v}

    balance = phase_rows(run, "balance")
    control_dt = np.array([r["control_dt_us"] for r in balance], dtype=float) if balance else None
    tick_us = np.array([r["tick_duration_us"] for r in balance], dtype=float) if balance else None
    faults = [
        r
        for r in run.rows
        if r.get("left_fault", 0)
        or r.get("right_fault", 0)
        or r.get("fault_reason", "none") not in ("", "none")
    ]
    check_results = [c["result"] for c in run.checks]

    return {
        "schema": run.schema,
        "columns": len(run.columns),
        "rows": len(run.rows),
        "malformed_rows": run.malformed_rows,
        "repeated_headers": run.repeated_headers,
        "nonfinite_columns": nonfinite,
        "duration_s": _round((times_us[-1] - times_us[0]) / 1e6, 6) if len(times_us) >= 2 else 0.0,
        "telemetry_dt_ms": {
            "mean": _round(float(np.mean(dt_ms)), 4) if dt_ms.size else None,
            "max": _round(float(np.max(dt_ms)), 4) if dt_ms.size else None,
        },
        "control_dt_us": _spread(control_dt),
        "tick_duration_us": _spread(tick_us),
        "fault_rows": len(faults),
        "imu_fresh_fraction": _round(
            float(np.mean([1.0 if r.get("imu_ok", 0) else 0.0 for r in run.rows])), 4
        ),
        "imu_resets": _round(max((r.get("imu_resets", 0.0) for r in run.rows), default=0.0), 0),
        "rotation_vector": rotation_vector_rate(run.rows),
        "saturated_fraction": _round(
            float(np.mean([1.0 if r.get("saturated", 0) else 0.0 for r in balance])), 4
        )
        if balance
        else None,
        "preflight_checks": {
            "pass": check_results.count("pass"),
            "fail": check_results.count("fail"),
            "skip": check_results.count("skip"),
            "failed": [c["name"] for c in run.checks if c["result"] == "fail"],
            "skipped": [c["name"] for c in run.checks if c["result"] == "skip"],
        },
    }


def phase_rows(run: Run, phase: str) -> list[dict[str, Any]]:
    return [r for r in run.rows if r.get("phase") == phase]


def column(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.array([r.get(name, math.nan) for r in rows], dtype=float)


# ------------------------------------------------------------- identification


def _verdict(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **extra}


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope, intercept, and R^2 for y = slope*x + intercept."""

    design = np.column_stack([x, np.ones_like(x)])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(solution[0]), float(solution[1])
    residual = y - (slope * x + intercept)
    total = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 0.0
    return slope, intercept, r2


def steady_mask(rows: list[dict[str, Any]]) -> np.ndarray:
    """Samples where the robot is genuinely holding station, not still settling."""

    times_s = (column(rows, "t_us") - column(rows, "t_us")[0]) / 1e6
    return (
        (times_s >= SETTLE_S)
        & (np.abs(column(rows, "pitch_rate_rad_s")) < STEADY_PITCH_RATE_RAD_S)
        & (np.abs(column(rows, "forward_vel_m_s")) < STEADY_FORWARD_VEL_M_S)
    )


def identify_balance_trim(
    balance: list[dict[str, Any]], mass_kg: float | None, com_height_m: float | None
) -> dict[str, Any]:
    """A persistent steady wheel torque is the model's missing static imbalance.

    If the seeded trim pitch were right, a station-keeping robot would need no
    mean torque. What it does need measures how far the real balance point sits
    from the modelled one, which is a direct correction to the COM the model was
    built with.
    """

    if not balance:
        return _verdict("insufficient_data", "no balance phase rows")
    if mass_kg is None or com_height_m is None:
        return _verdict("insufficient_data", "measurements lack robot mass or COM height")

    mask = steady_mask(balance)
    samples = int(np.count_nonzero(mask))
    if samples < MIN_STEADY_SAMPLES:
        return _verdict(
            "insufficient_data",
            f"only {samples} steady samples, need {MIN_STEADY_SAMPLES}",
            steady_samples=samples,
        )

    tau = column(balance, "balance_tau_nm")[mask]
    pitch = column(balance, "pitch_rad")[mask]
    target = column(balance, "target_pitch_rad")[mask]
    tau_mean = float(np.mean(tau))
    tau_std = float(np.std(tau))
    standard_error = tau_std / math.sqrt(samples) if samples else math.inf
    pitch_error_mean = float(np.mean(pitch - target))

    # A mean torque inside its own noise is not a bias, it is a still robot.
    if abs(tau_mean) <= 2.0 * standard_error:
        return _verdict(
            "no_correction_needed",
            "mean steady torque is not distinguishable from zero",
            mean_torque_nm=_round(tau_mean, 6),
            torque_standard_error_nm=_round(standard_error, 6),
            steady_samples=samples,
        )

    gravity_torque_nm = mass_kg * GRAVITY_M_S2 * com_height_m
    ratio = 2.0 * abs(tau_mean) / gravity_torque_nm
    if ratio >= 1.0:
        return _verdict(
            "implausible",
            "steady torque exceeds the gravity torque of the modelled COM",
            mean_torque_nm=_round(tau_mean, 6),
            gravity_torque_nm=_round(gravity_torque_nm, 6),
        )

    offset_rad = math.asin(ratio) * (1.0 if pitch_error_mean >= 0.0 else -1.0)
    return _verdict(
        "identified",
        "steady wheel torque converted through the modelled gravity torque",
        steady_samples=samples,
        mean_torque_nm=_round(tau_mean, 6),
        torque_std_nm=_round(tau_std, 6),
        mean_pitch_error_rad=_round(pitch_error_mean, 6),
        gravity_torque_nm=_round(gravity_torque_nm, 6),
        trim_offset_rad=_round(offset_rad, 6),
        trim_offset_deg=_round(math.degrees(offset_rad), 4),
        suggested_target_pitch_rad=_round(float(np.mean(target)) + offset_rad, 8),
        suggested_com_fore_aft_shift_m=_round(com_height_m * math.tan(offset_rad), 8),
        direction_from="sign of the mean steady pitch error",
    )


def identify_pitch_plant(balance: list[dict[str, Any]]) -> dict[str, Any]:
    """Regress pitch acceleration on lean and wheel torque.

    This is the estimate a closed-loop run is least able to support: the
    controller makes torque a deterministic function of lean, so the two
    regressors arrive nearly collinear and the fit is reading noise. The
    condition number below is the gate, and it is expected to fail on a quiet
    station-keeping trial.
    """

    if len(balance) < MIN_FIT_SAMPLES:
        return _verdict("insufficient_data", f"{len(balance)} rows, need {MIN_FIT_SAMPLES}")

    times_s = column(balance, "t_us") / 1e6
    pitch = column(balance, "pitch_rad")
    pitch_rate = column(balance, "pitch_rate_rad_s")
    tau_total = 2.0 * column(balance, "balance_tau_nm")

    # Differentiate the gyro once rather than the angle twice.
    pitch_accel = np.gradient(pitch_rate, times_s)
    design = np.column_stack([np.sin(pitch), tau_total])
    finite = np.all(np.isfinite(design), axis=1) & np.isfinite(pitch_accel)
    design, pitch_accel = design[finite], pitch_accel[finite]
    if design.shape[0] < MIN_FIT_SAMPLES:
        return _verdict("insufficient_data", "too few finite samples after differentiation")

    pitch_range = float(np.ptp(pitch[finite]))
    torque_range = float(np.ptp(tau_total[finite]))
    # Scale the columns to unit norm first. sin(pitch) runs about 0.01 while total
    # torque runs about 0.5, so a raw condition number measures the unit mismatch
    # rather than the collinearity this gate exists to catch.
    norms = np.linalg.norm(design, axis=0)
    condition = float(np.linalg.cond(design / norms)) if np.all(norms > 0.0) else float("inf")
    correlation = (
        float(np.corrcoef(design[:, 0], design[:, 1])[0, 1])
        if np.std(design[:, 0]) > 0.0 and np.std(design[:, 1]) > 0.0
        else float("nan")
    )

    if pitch_range < MIN_PITCH_RANGE_RAD or torque_range < MIN_TORQUE_RANGE_NM:
        return _verdict(
            "not_excited",
            "the run never moved far enough to separate lean from torque",
            pitch_range_rad=_round(pitch_range, 6),
            torque_range_nm=_round(torque_range, 6),
            condition_number=_round(condition, 3),
        )
    if condition > MAX_CONDITION_NUMBER:
        return _verdict(
            "not_excited",
            "lean and torque are collinear in closed loop; add an external disturbance",
            condition_number=_round(condition, 3),
            condition_limit=MAX_CONDITION_NUMBER,
            regressor_correlation=_round(correlation, 4),
            pitch_range_rad=_round(pitch_range, 6),
            torque_range_nm=_round(torque_range, 6),
        )

    solution, *_ = np.linalg.lstsq(design, pitch_accel, rcond=None)
    predicted = design @ solution
    total = float(np.sum((pitch_accel - float(np.mean(pitch_accel))) ** 2))
    r2 = 1.0 - float(np.sum((pitch_accel - predicted) ** 2)) / total if total > 0 else 0.0
    if r2 < MIN_FIT_R2:
        return _verdict(
            "poor_fit",
            "regression explains too little of the pitch acceleration",
            r2=_round(r2, 4),
            condition_number=_round(condition, 3),
        )

    gravity_over_inertia = float(solution[0])
    torque_over_inertia = float(solution[1])
    return _verdict(
        "identified",
        "pitch acceleration regressed on lean and total wheel torque",
        samples=int(design.shape[0]),
        condition_number=_round(condition, 3),
        regressor_correlation=_round(correlation, 4),
        r2=_round(r2, 4),
        mgl_over_inertia_s2=_round(gravity_over_inertia, 6),
        inverse_pitch_inertia_per_kg_m2=_round(torque_over_inertia, 6),
        effective_pitch_inertia_kg_m2=_round(1.0 / torque_over_inertia, 6)
        if torque_over_inertia
        else None,
        pitch_range_rad=_round(pitch_range, 6),
        torque_range_nm=_round(torque_range, 6),
    )


def identify_actuator_tracking(balance: list[dict[str, Any]]) -> dict[str, Any]:
    """Commanded versus measured moteus torque, per wheel, under real load."""

    if len(balance) < MIN_FIT_SAMPLES:
        short = _verdict("insufficient_data", f"{len(balance)} rows, need {MIN_FIT_SAMPLES}")
        return {"left": short, "right": dict(short)}

    wheels: dict[str, Any] = {}
    for side in ("left", "right"):
        commanded = column(balance, f"{side}_cmd_nm")
        measured = column(balance, f"{side}_torque_nm")
        finite = np.isfinite(commanded) & np.isfinite(measured)
        commanded, measured = commanded[finite], measured[finite]
        if commanded.size < MIN_FIT_SAMPLES:
            wheels[side] = _verdict("insufficient_data", "too few finite torque samples")
            continue
        command_range = float(np.ptp(commanded))
        if command_range < MIN_TORQUE_RANGE_NM:
            wheels[side] = _verdict(
                "not_excited",
                "commanded torque barely moved",
                command_range_nm=_round(command_range, 6),
            )
            continue
        gain, offset, r2 = linear_fit(commanded, measured)
        wheels[side] = _verdict(
            "identified" if r2 >= MIN_FIT_R2 else "poor_fit",
            "measured torque regressed on commanded torque",
            samples=int(commanded.size),
            gain=_round(gain, 6),
            offset_nm=_round(offset, 6),
            r2=_round(r2, 4),
            command_range_nm=_round(command_range, 6),
        )
    return wheels


def identify_loop_latency(balance: list[dict[str, Any]]) -> dict[str, Any]:
    """Lag between commanding a torque and seeing the robot answer."""

    if len(balance) < MIN_FIT_SAMPLES:
        short = _verdict("insufficient_data", f"{len(balance)} rows, need {MIN_FIT_SAMPLES}")
        return {
            "command_to_measured_torque": short,
            "command_to_pitch_acceleration": dict(short),
        }

    times_s = column(balance, "t_us") / 1e6
    commanded = column(balance, "balance_tau_nm")
    measured = 0.5 * (column(balance, "left_torque_nm") + column(balance, "right_torque_nm"))
    pitch_accel = np.gradient(column(balance, "pitch_rate_rad_s"), times_s)
    dt_s = float(np.median(np.diff(times_s))) if times_s.size >= 2 else CONTROL_PERIOD_US / 1e6

    def best_lag(reference: np.ndarray, response: np.ndarray) -> dict[str, Any]:
        finite = np.isfinite(reference) & np.isfinite(response)
        a, b = reference[finite], response[finite]
        if a.size < MIN_FIT_SAMPLES or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
            return _verdict("insufficient_data", "no variance to correlate")
        a = (a - float(np.mean(a))) / float(np.std(a))
        b = (b - float(np.mean(b))) / float(np.std(b))
        best_ticks, best_score = 0, -2.0
        for lag in range(MAX_LAG_TICKS + 1):
            shifted = b[lag:] if lag else b
            head = a[: shifted.size]
            if head.size < MIN_FIT_SAMPLES:
                break
            score = float(np.mean(head * shifted))
            if score > best_score:
                best_ticks, best_score = lag, score
        return _verdict(
            "identified" if abs(best_score) >= 0.3 else "weak_correlation",
            "peak normalized cross-correlation over integer control ticks",
            lag_ticks=best_ticks,
            lag_ms=_round(best_ticks * dt_s * 1000.0, 3),
            correlation=_round(best_score, 4),
        )

    return {
        "command_to_measured_torque": best_lag(commanded, measured),
        "command_to_pitch_acceleration": best_lag(commanded, pitch_accel),
    }


def identify_sensor_noise(preflight: list[dict[str, Any]]) -> dict[str, Any]:
    """Noise floor from the preflight window, where the robot is held still.

    This is the one estimate a balance run makes cleanly, because the operator
    holding Finn motionless is exactly the condition it needs.
    """

    if len(preflight) < MIN_STEADY_SAMPLES:
        return _verdict(
            "insufficient_data",
            f"{len(preflight)} preflight rows, need {MIN_STEADY_SAMPLES}",
        )
    noise: dict[str, Any] = {"samples": len(preflight)}
    for name in (
        "imu_gyro_x_rad_s",
        "imu_gyro_y_rad_s",
        "imu_gyro_z_rad_s",
        "imu_linear_accel_x_m_s2",
        "imu_linear_accel_y_m_s2",
        "imu_linear_accel_z_m_s2",
    ):
        values = column(preflight, name)
        values = values[np.isfinite(values)]
        if values.size:
            noise[name] = {
                "mean": _round(float(np.mean(values)), 6),
                "std": _round(float(np.std(values)), 6),
            }
    noise["status"] = "identified"
    noise["reason"] = "standard deviation while held stationary with motors stopped"
    return noise


NOT_IDENTIFIED = [
    "tire_ground_slide_friction",
    "tire_ground_rolling_friction",
    "contact_solref",
    "contact_solimp",
    "wheel_track_width_m",
    "yaw_plant_and_steering_gain",
    "wheel_armature_and_damping",
    "absolute_world_trajectory",
]


def recommend_next_change(identification: dict[str, Any]) -> dict[str, Any]:
    """Name one model change, because coupled families cannot be fitted together.

    Ranked by how well this run could actually see each one, not by how large the
    correction looks.
    """

    trim = identification["balance_trim"]
    if trim.get("status") == "identified":
        return {
            "parameter_family": "balance trim / COM fore-aft",
            "action": (
                f"shift robot.com_fore_aft_m by {trim['suggested_com_fore_aft_shift_m']} m, "
                "rebuild the seeded model, and re-export the gain header"
            ),
            "evidence": (
                f"{trim['steady_samples']} steady samples held a mean wheel torque of "
                f"{trim['mean_torque_nm']} N*m, worth {trim['trim_offset_deg']} degrees of trim"
            ),
        }
    actuator = identification["actuator_tracking"]
    tracked = [
        side
        for side, item in actuator.items()
        if isinstance(item, dict) and item.get("status") == "identified"
    ]
    off_by = [side for side in tracked if abs(float(actuator[side]["gain"]) - 1.0) > 0.1]
    if off_by:
        gains = ", ".join(f"{side}={actuator[side]['gain']}" for side in off_by)
        return {
            "parameter_family": "actuator torque gain",
            "action": "correct the wheel actuator gear/gain in the seeded model build",
            "evidence": f"measured/commanded torque slope away from unity: {gains}",
        }
    noise = identification["sensor_noise"]
    if noise.get("status") == "identified":
        return {
            "parameter_family": "IMU sensor noise",
            "action": "set the sim gyro and accelerometer noise from the measured floor",
            "evidence": f"stationary standard deviations over {noise['samples']} preflight samples",
        }
    return {
        "parameter_family": None,
        "action": (
            "change nothing yet. Run a trial with an external disturbance so lean and "
            "torque stop being collinear, then re-run this analysis"
        ),
        "evidence": "no parameter family reached its observability threshold on this run",
    }


def analyze_run(run_dir: Path, measurements_path: Path, seeded_header: Path) -> dict[str, Any]:
    run = read_run(run_dir)
    measurements = load_measurements(measurements_path)
    header = read_seeded_header(seeded_header)

    mass_kg = measurement_value(measurements, "robot", "mass_kg")
    com_height_m = measurement_value(measurements, "robot", "com_height_m")
    com_fore_aft_m = measurement_value(measurements, "robot", "com_fore_aft_m")
    pitch_inertia = measurement_value(
        measurements, "robot", "cad_inertia_tensor_whole_robot", "Iyy_kg_m2"
    )

    balance = phase_rows(run, "balance")
    preflight = phase_rows(run, "preflight")
    integrity = audit_integrity(run)

    identification = {
        "balance_trim": identify_balance_trim(balance, mass_kg, com_height_m),
        "pitch_plant": identify_pitch_plant(balance),
        "actuator_tracking": identify_actuator_tracking(balance),
        "loop_latency": identify_loop_latency(balance),
        "sensor_noise": identify_sensor_noise(preflight),
    }

    model_hashes = sorted({str(r.get("model_sha256", "")) for r in run.rows} - {""})
    fault_reasons = sorted({str(r.get("fault_reason", "none")) for r in run.rows} - {"none", ""})

    return {
        "schema_version": 1,
        "source": "finn_mcu_lqr_balance",
        "run": {
            "path": portable_path(run_dir),
            "telemetry_schema": run.schema,
            "rows": len(run.rows),
            "balance_rows": len(balance),
            "preflight_rows": len(preflight),
            "model_sha256": model_hashes[0] if len(model_hashes) == 1 else model_hashes,
            "fault_reasons": fault_reasons,
            "events": len(run.events),
        },
        "integrity": integrity,
        "inputs": {
            "measurements": portable_path(measurements_path),
            "seeded_header": portable_path(seeded_header),
            "robot_mass_kg": mass_kg,
            "com_height_m": com_height_m,
            "com_fore_aft_m": com_fore_aft_m,
            "cad_pitch_inertia_kg_m2": pitch_inertia,
            "target_pitch_rad": header.get("kTargetPitchRad"),
            "gain_pitch": header.get("kGainPitch"),
            "gain_pitch_rate": header.get("kGainPitchRate"),
            "gain_forward_vel": header.get("kGainForwardVel"),
        },
        "identification": identification,
        "recommended_next_change": recommend_next_change(identification),
        "not_identified": NOT_IDENTIFIED,
    }


# --------------------------------------------------------------------- output


def write_derived_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _status_line(name: str, item: dict[str, Any]) -> str:
    return f"- `{name}`: **{item.get('status', 'unknown')}** - {item.get('reason', '')}"


def write_report(path: Path, derived: dict[str, Any]) -> None:
    run = derived["run"]
    integrity = derived["integrity"]
    identification = derived["identification"]
    checks = integrity["preflight_checks"]
    lines = [
        "# Finn LQR balance run",
        "",
        f"Run: `{run['path']}`",
        f"Schema: `{run['telemetry_schema']}`  Model: `{run['model_sha256']}`",
        f"Rows: {run['rows']} total, {run['balance_rows']} balancing, "
        f"{run['preflight_rows']} in preflight",
        "",
        "## Recording integrity",
        "",
        f"- duration: {integrity['duration_s']} s",
        f"- malformed rows: {integrity['malformed_rows']}",
        f"- non-finite columns: {integrity['nonfinite_columns'] or 'none'}",
        f"- control dt: mean {integrity['control_dt_us']['mean']} us, "
        f"max {integrity['control_dt_us']['max']} us, "
        f"jitter {integrity['control_dt_us']['jitter_us']} us",
        f"- control tick duration: max {integrity['tick_duration_us']['max']} us",
        f"- IMU alive fraction: {integrity['imu_fresh_fraction']}",
        f"- IMU resets: {integrity['imu_resets']}",
        f"- rotation vector: {integrity['rotation_vector']['arrivals']} arrivals, "
        f"{integrity['rotation_vector']['rate_hz']} Hz, "
        f"worst age {integrity['rotation_vector']['max_age_ms']} ms",
        f"- torque saturation fraction: {integrity['saturated_fraction']}",
        f"- fault rows: {integrity['fault_rows']}  reasons: {run['fault_reasons'] or 'none'}",
        f"- preflight checks: {checks['pass']} pass, {checks['fail']} fail, {checks['skip']} skip",
    ]
    if checks["failed"]:
        lines.append(f"- failed checks: {', '.join(checks['failed'])}")
    if checks["skipped"]:
        lines.append(f"- skipped checks: {', '.join(checks['skipped'])}")

    lines.extend(["", "## Identification", ""])
    for name in ("balance_trim", "pitch_plant", "sensor_noise"):
        lines.append(_status_line(name, identification[name]))
    for side, item in identification["actuator_tracking"].items():
        lines.append(_status_line(f"actuator_tracking.{side}", item))
    for name, item in identification["loop_latency"].items():
        lines.append(_status_line(f"loop_latency.{name}", item))

    trim = identification["balance_trim"]
    if trim.get("status") == "identified":
        lines.extend(
            [
                "",
                "### Balance trim",
                "",
                f"- mean steady wheel torque: {trim['mean_torque_nm']} N*m "
                f"(std {trim['torque_std_nm']}, n={trim['steady_samples']})",
                f"- implied trim offset: {trim['trim_offset_rad']} rad "
                f"({trim['trim_offset_deg']} deg)",
                f"- suggested target pitch: {trim['suggested_target_pitch_rad']} rad",
                f"- suggested COM fore-aft shift: {trim['suggested_com_fore_aft_shift_m']} m",
            ]
        )

    recommended = derived["recommended_next_change"]
    lines.extend(
        [
            "",
            "## Recommended next model change",
            "",
            f"- family: {recommended['parameter_family'] or 'none yet'}",
            f"- action: {recommended['action']}",
            f"- evidence: {recommended['evidence']}",
            "",
            "Change one coupled parameter family at a time, regenerate the seeded model and",
            "the gain header, then capture another short trial.",
            "",
            "## Not identified by a closed-loop balance run",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in derived["not_identified"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--seeded-header", type=Path, default=DEFAULT_SEEDED_HEADER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        derived = analyze_run(args.run_dir, args.measurements, args.seeded_header)
    except LqrBalancePostprocessError as error:
        print(f"error: {error}")
        return 1

    out_dir = args.run_dir / "postprocess"
    out_dir.mkdir(exist_ok=True)
    derived_path = out_dir / "derived.yaml"
    report_path = out_dir / "report.md"
    write_derived_yaml(derived_path, derived)
    write_report(report_path, derived)

    recommended = derived["recommended_next_change"]
    print(f"Wrote {report_path} and {derived_path} ({derived['run']['rows']} rows)")
    print(f"Recommended next change: {recommended['parameter_family'] or 'none yet'}")
    print(f"  {recommended['action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
