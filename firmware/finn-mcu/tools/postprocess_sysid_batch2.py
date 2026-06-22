#!/usr/bin/env python3
"""Analyze a captured Batch 2 loaded ground-contact sysid run."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RAD_PER_REV = 2.0 * math.pi
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEASUREMENTS = REPO_ROOT / "sim" / "config" / "finn_measurements.yaml"
ROBOT_FORWARD_ACCEL_FIELD = "imu_linear_accel_z_m_s2"

NUMERIC_FIELDS = {
    "t_us",
    "phase_index",
    "armed",
    "control_tick_us",
    "segment_elapsed_ms",
    "left_cmd_nm",
    "right_cmd_nm",
    "left_mode",
    "left_pos_rev",
    "left_vel_rev_s",
    "left_torque_nm",
    "left_voltage_v",
    "left_temp_c",
    "left_fault",
    "right_mode",
    "right_pos_rev",
    "right_vel_rev_s",
    "right_torque_nm",
    "right_voltage_v",
    "right_temp_c",
    "right_fault",
    "imu_ok",
    "imu_age_ms",
    "imu_qr",
    "imu_qi",
    "imu_qj",
    "imu_qk",
    "imu_accuracy_rad",
    "imu_gyro_x_rad_s",
    "imu_gyro_y_rad_s",
    "imu_gyro_z_rad_s",
    "imu_linear_accel_x_m_s2",
    "imu_linear_accel_y_m_s2",
    "imu_linear_accel_z_m_s2",
    "pitch_rad",
    "pitch_rate_rad_s",
    "yaw_rad",
    "yaw_rate_rad_s",
    "left_pos_delta_rev",
    "right_pos_delta_rev",
    "avg_wheel_pos_rev",
    "diff_wheel_pos_rev",
    "pitch_limit",
    "speed_limit",
    "travel_limit",
}


@dataclass(frozen=True)
class LossFit:
    sample_count: int
    linear_damping_s: float | None
    friction_accel_m_s2: float | None
    per_wheel_torque_nm: float | None
    rmse_m_s: float | None
    confidence: str
    bound_active: dict[str, bool]
    notes: list[str]


def read_schema(telemetry: Path) -> str | None:
    if not telemetry.exists():
        return None
    for line in telemetry.read_text().splitlines():
        if line.startswith("schema,"):
            parts = line.split(",", 1)
            return parts[1] if len(parts) == 2 else None
    return None


def read_batch2_rows(telemetry: Path) -> list[dict[str, str]]:
    if not telemetry.exists():
        return []
    data_lines = [line for line in telemetry.read_text().splitlines() if line.startswith("data,")]
    header_index = next(
        (index for index, line in enumerate(data_lines) if line.startswith("data,t_us,")),
        None,
    )
    if header_index is None:
        return []
    return list(csv.DictReader(data_lines[header_index:]))


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("event,"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        event: dict[str, Any] = {
            "t_us": _to_float(parts[1]),
            "event": parts[2],
            "state": parts[3],
            "detail": ",".join(parts[4:]) if len(parts) > 4 else "",
            "fields": parts[4:],
        }
        if parts[2] == "segment_start" and len(parts) >= 5:
            event["phase"] = parts[4]
        if parts[2] == "segment_end":
            if len(parts) >= 7:
                event["phase"] = parts[4]
                event["reason"] = parts[5]
                event["elapsed_ms"] = _to_float(parts[6])
            else:
                parsed = dict(re.findall(r"([a-zA-Z_]+)=([^,]+)", event["detail"]))
                event.update(parsed)
                if "elapsed_ms" in event:
                    event["elapsed_ms"] = _to_float(str(event["elapsed_ms"]))
        events.append(event)
    return events


def rows_as_numeric(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            item[key] = _to_float(value) if key in NUMERIC_FIELDS else value
        converted.append(item)
    return converted


def _to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_or_none(value: float | None, ndigits: int = 8) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), ndigits)


def _safe_int(value: object) -> int:
    try:
        f = float(value)  # type: ignore[arg-type]
        return int(f) if math.isfinite(f) else 0
    except (TypeError, ValueError):
        return 0


def _range(values: list[float], ndigits: int = 6) -> list[float | None]:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return [None, None]
    return [_finite_or_none(min(finite), ndigits), _finite_or_none(max(finite), ndigits)]


def _group_by_phase(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    result: list[tuple[str, list[dict[str, Any]]]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phase = str(row.get("phase", ""))
        if phase not in groups:
            groups[phase] = []
            result.append((phase, groups[phase]))
        groups[phase].append(row)
    return result


def _duration_s(group: list[dict[str, Any]]) -> float:
    if len(group) < 2:
        return 0.0
    return max(0.0, (float(group[-1]["t_us"]) - float(group[0]["t_us"])) / 1_000_000.0)


def _series(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.array([float(row.get(field, math.nan)) for row in rows], dtype=float)


def _finite_series(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = _series(rows, field)
    return values[np.isfinite(values)]


def _measurement_value(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    node: Any = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, dict):
        node = node.get("value")
    value = _to_float(node)
    return float(value) if math.isfinite(value) else None


def load_measurements(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    return loaded if isinstance(loaded, dict) else {}


def physical_inputs(
    measurements: dict[str, Any],
    *,
    mass_kg: float | None = None,
    gantry_mass_kg: float | None = None,
    loaded_wheel_radius_m: float | None = None,
    wheel_track_width_m: float | None = None,
    com_height_m: float | None = None,
    com_fore_aft_m: float | None = None,
    pitch_inertia_kg_m2: float | None = None,
) -> dict[str, Any]:
    left_radius = _measurement_value(measurements, ("wheels", "left", "radius_m"))
    right_radius = _measurement_value(measurements, ("wheels", "right", "radius_m"))
    radii = [value for value in (left_radius, right_radius) if value and value > 0]
    radius = loaded_wheel_radius_m or (float(np.mean(radii)) if radii else None)
    com_height = (
        com_height_m
        or _measurement_value(measurements, ("robot", "com_height_m"))
        or _measurement_value(measurements, ("robot", "com", "z_m"))
    )
    com_fore_aft = (
        com_fore_aft_m
        or _measurement_value(measurements, ("robot", "com_fore_aft_m"))
        or _measurement_value(measurements, ("robot", "com", "x_m"))
    )
    robot_mass = mass_kg or _measurement_value(measurements, ("robot", "mass_kg"))
    gantry_mass = gantry_mass_kg or _measurement_value(measurements, ("gantry", "mass_kg"))
    return {
        "robot_mass_kg": robot_mass,
        "gantry_mass_kg": gantry_mass,
        "loaded_wheel_radius_m": radius,
        "wheel_track_width_m": wheel_track_width_m
        or _measurement_value(measurements, ("robot", "wheel_track_width_m")),
        "com_height_m": com_height,
        "com_fore_aft_m": com_fore_aft,
        "pitch_inertia_kg_m2": pitch_inertia_kg_m2
        or _measurement_value(measurements, ("robot", "pitch_inertia_kg_m2")),
    }


def actuator_tracking(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    cmd = _series(rows, f"{side}_cmd_nm")
    torque = _series(rows, f"{side}_torque_nm")
    mask = np.isfinite(cmd) & np.isfinite(torque) & (np.abs(cmd) > 1e-5)
    if len(cmd) >= 5:
        stable = np.ones(len(cmd), dtype=bool)
        for lag in range(1, 4):
            stable[lag:] &= np.abs(cmd[lag:] - cmd[:-lag]) <= 1e-9
            stable[:lag] = False
        if int(np.sum(mask & stable)) >= 3:
            mask &= stable
    count = int(np.sum(mask))
    if count < 3:
        return {"sample_count": count, "gain": None, "bias_nm": None, "rmse_nm": None}
    design = np.column_stack([cmd[mask], np.ones(count)])
    gain, bias = np.linalg.lstsq(design, torque[mask], rcond=None)[0]
    residual = design @ np.array([gain, bias]) - torque[mask]
    return {
        "sample_count": count,
        "gain": _finite_or_none(float(gain), 6),
        "bias_nm": _finite_or_none(float(bias), 6),
        "rmse_nm": _finite_or_none(float(np.sqrt(np.mean(residual * residual))), 6),
    }


def cross_correlation_delay_s(
    rows: list[dict[str, Any]],
    input_field: str,
    output_field: str,
    max_lag_s: float = 0.5,
) -> dict[str, Any]:
    if len(rows) < 8:
        return {"sample_count": len(rows), "delay_s": None, "confidence": "insufficient"}
    t = _series(rows, "t_us") / 1_000_000.0
    x = _series(rows, input_field)
    y = _series(rows, output_field)
    finite = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y = t[finite], x[finite], y[finite]
    if len(t) < 8:
        return {"sample_count": len(t), "delay_s": None, "confidence": "insufficient"}
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0:
        return {"sample_count": len(t), "delay_s": None, "confidence": "insufficient"}

    # Use derivative-like changes so command steps align with delayed responses.
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dx -= float(np.mean(dx))
    dy -= float(np.mean(dy))
    if float(np.std(dx)) <= 1e-9 or float(np.std(dy)) <= 1e-9:
        return {"sample_count": len(t), "delay_s": None, "confidence": "insufficient"}

    max_lag = min(round(max_lag_s / dt), len(t) - 2)
    best_lag = 0
    best_score = -math.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = dx[: -lag or None]
            b = dy[lag:]
        else:
            a = dx[-lag:]
            b = dy[:lag]
        if len(a) < 4:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            continue
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    confidence = "measured" if best_score >= 0.6 else "provisional"
    return {
        "sample_count": len(t),
        "delay_s": _finite_or_none(best_lag * dt, 5),
        "correlation": _finite_or_none(best_score, 5),
        "confidence": confidence,
    }


def delay_estimates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    avg_cmd_rows = []
    diff_cmd_rows = []
    for row in rows:
        item = dict(row)
        item["avg_cmd_nm"] = 0.5 * (
            float(row.get("left_cmd_nm", 0.0)) + float(row.get("right_cmd_nm", 0.0))
        )
        item["avg_torque_nm"] = 0.5 * (
            float(row.get("left_torque_nm", 0.0)) + float(row.get("right_torque_nm", 0.0))
        )
        item["avg_wheel_vel_rev_s"] = 0.5 * (
            float(row.get("left_vel_rev_s", 0.0)) + float(row.get("right_vel_rev_s", 0.0))
        )
        item["diff_cmd_nm"] = 0.5 * (
            float(row.get("left_cmd_nm", 0.0)) - float(row.get("right_cmd_nm", 0.0))
        )
        avg_cmd_rows.append(item)
        diff_cmd_rows.append(item)
    return {
        "command_to_measured_torque": {
            "left": cross_correlation_delay_s(rows, "left_cmd_nm", "left_torque_nm"),
            "right": cross_correlation_delay_s(rows, "right_cmd_nm", "right_torque_nm"),
            "average": cross_correlation_delay_s(avg_cmd_rows, "avg_cmd_nm", "avg_torque_nm"),
        },
        "command_to_wheel_velocity": {
            "average": cross_correlation_delay_s(avg_cmd_rows, "avg_cmd_nm", "avg_wheel_vel_rev_s"),
        },
        "command_to_imu": {
            "linear_accel_forward": cross_correlation_delay_s(
                avg_cmd_rows, "avg_cmd_nm", ROBOT_FORWARD_ACCEL_FIELD
            ),
            "yaw_rate": cross_correlation_delay_s(diff_cmd_rows, "diff_cmd_nm", "yaw_rate_rad_s"),
        },
    }


def fit_loaded_coastdown_loss(
    rows: list[dict[str, Any]],
    radius_m: float | None,
    mass_kg: float | None,
) -> LossFit:
    segments = [
        group
        for phase, group in _group_by_phase(rows)
        if phase.startswith("coastdown_average_") and "settle" not in phase and len(group) >= 8
    ]
    sample_count = sum(len(group) for group in segments)
    if sample_count < 20 or not radius_m or radius_m <= 0:
        return LossFit(
            sample_count,
            None,
            None,
            None,
            None,
            "insufficient",
            {"linear_damping": False, "friction_accel": False},
            ["too_few_coastdown_samples" if sample_count < 20 else "missing_loaded_radius_m"],
        )

    velocities: list[float] = []
    accelerations: list[float] = []
    for group in segments:
        t = _series(group, "t_us") / 1_000_000.0
        avg_rev_s = 0.5 * (_series(group, "left_vel_rev_s") + _series(group, "right_vel_rev_s"))
        v = avg_rev_s * RAD_PER_REV * radius_m
        finite = np.isfinite(t) & np.isfinite(v)
        t, v = t[finite], v[finite]
        if len(t) < 8 or float(np.max(np.abs(v))) < 0.01:
            continue
        accel = np.gradient(v, t)
        mask = np.abs(v) >= 0.01
        velocities.extend(v[mask].tolist())
        accelerations.extend(accel[mask].tolist())

    if len(velocities) < 20:
        return LossFit(
            len(velocities),
            None,
            None,
            None,
            None,
            "insufficient",
            {"linear_damping": False, "friction_accel": False},
            ["too_few_moving_coastdown_samples"],
        )

    v = np.array(velocities, dtype=float)
    a = np.array(accelerations, dtype=float)
    design = np.column_stack([-v, -np.sign(v)])
    coeff, *_ = np.linalg.lstsq(design, a, rcond=None)
    damping = max(0.0, float(coeff[0]))
    friction_accel = max(0.0, float(coeff[1]))
    predicted = design @ np.array([damping, friction_accel])
    rmse = float(np.sqrt(np.mean((predicted - a) ** 2)))
    per_wheel_torque = None
    if mass_kg and mass_kg > 0:
        per_wheel_torque = 0.5 * mass_kg * friction_accel * radius_m
    confidence = "measured" if len(velocities) >= 80 and rmse < 0.5 else "provisional"
    return LossFit(
        len(velocities),
        damping,
        friction_accel,
        per_wheel_torque,
        rmse,
        confidence,
        {"linear_damping": damping <= 1e-9, "friction_accel": friction_accel <= 1e-9},
        [],
    )


def tire_traction_lower_bounds(
    rows: list[dict[str, Any]],
    radius_m: float | None,
    track_width_m: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "straight_accel_m_s2": None,
        "straight_mu_lower_bound": None,
        "yaw_accel_rad_s2": None,
        "source": "onboard_imu_and_wheel_odometry_lower_bound",
    }
    if rows:
        forward_accel = np.abs(_finite_series(rows, ROBOT_FORWARD_ACCEL_FIELD))
        if len(forward_accel):
            max_accel = float(np.percentile(forward_accel, 95))
            result["straight_accel_m_s2"] = _finite_or_none(max_accel, 6)
            result["straight_mu_lower_bound"] = _finite_or_none(max_accel / 9.80665, 6)
    if radius_m and track_width_m and track_width_m > 0:
        yaw_rates = []
        times = []
        for row in rows:
            left = float(row.get("left_vel_rev_s", math.nan))
            right = float(row.get("right_vel_rev_s", math.nan))
            t = float(row.get("t_us", math.nan)) / 1_000_000.0
            if math.isfinite(left) and math.isfinite(right) and math.isfinite(t):
                yaw_rates.append((right - left) * RAD_PER_REV * radius_m / track_width_m)
                times.append(t)
        if len(yaw_rates) >= 4:
            yaw_accel = np.gradient(np.array(yaw_rates), np.array(times))
            result["yaw_accel_rad_s2"] = _finite_or_none(
                float(np.percentile(np.abs(yaw_accel), 95)), 6
            )
    return result


def yaw_response(
    rows: list[dict[str, Any]],
    radius_m: float | None,
    track_width_m: float | None,
) -> dict[str, Any]:
    if not radius_m or radius_m <= 0:
        return {
            "sample_count": 0,
            "effective_track_width_m": None,
            "track_width_correction": None,
            "confidence": "insufficient",
            "notes": ["missing_loaded_radius_m"],
        }
    samples = []
    ratios = []
    for phase, group in _group_by_phase(rows):
        if not phase.startswith("yaw_"):
            continue
        for row in group:
            left = float(row.get("left_vel_rev_s", math.nan))
            right = float(row.get("right_vel_rev_s", math.nan))
            imu_yaw = float(row.get("yaw_rate_rad_s", math.nan))
            wheel_delta = (right - left) * RAD_PER_REV * radius_m
            if math.isfinite(wheel_delta) and math.isfinite(imu_yaw) and abs(imu_yaw) > 0.02:
                samples.append(wheel_delta)
                ratios.append(wheel_delta / imu_yaw)
    if len(ratios) < 8:
        return {
            "sample_count": len(ratios),
            "effective_track_width_m": None,
            "track_width_correction": None,
            "confidence": "insufficient",
            "notes": ["too_few_differential_yaw_samples"],
        }
    eff_track = float(np.median(np.abs(ratios)))
    correction = eff_track / track_width_m if track_width_m and track_width_m > 0 else None
    return {
        "sample_count": len(ratios),
        "effective_track_width_m": _finite_or_none(eff_track, 6),
        "track_width_correction": _finite_or_none(correction, 6),
        "confidence": "measured" if len(ratios) >= 40 else "provisional",
        "notes": [],
    }


def stationary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        phase = str(row.get("phase", "")).lower()
        is_stationary_phase = phase in ("initial_stationary_noise", "final_stationary_noise")
        stopped = (
            abs(float(row.get("left_cmd_nm", 0.0))) <= 1e-5
            and abs(float(row.get("right_cmd_nm", 0.0))) <= 1e-5
            and abs(float(row.get("left_vel_rev_s", 0.0))) < 0.02
            and abs(float(row.get("right_vel_rev_s", 0.0))) < 0.02
        )
        if is_stationary_phase and stopped:
            result.append(row)
    return result


def noise_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stationary = stationary_rows(rows)

    def stats(field: str) -> dict[str, Any]:
        values = _finite_series(stationary, field)
        if len(values) == 0:
            return {"sample_count": 0, "mean": None, "std": None, "p95_abs": None}
        return {
            "sample_count": len(values),
            "mean": _finite_or_none(float(np.mean(values)), 8),
            "std": _finite_or_none(float(np.std(values)), 8),
            "p95_abs": _finite_or_none(float(np.percentile(np.abs(values), 95)), 8),
        }

    return {
        "stationary_row_count": len(stationary),
        "gyro_rad_s": {
            "x": stats("imu_gyro_x_rad_s"),
            "y": stats("imu_gyro_y_rad_s"),
            "z": stats("imu_gyro_z_rad_s"),
        },
        "linear_accel_m_s2": {
            "x": stats("imu_linear_accel_x_m_s2"),
            "y": stats("imu_linear_accel_y_m_s2"),
            "z": stats("imu_linear_accel_z_m_s2"),
        },
        "pitch_rad": stats("pitch_rad"),
        "yaw_rad": stats("yaw_rad"),
        "left_wheel_velocity_rev_s": stats("left_vel_rev_s"),
        "right_wheel_velocity_rev_s": stats("right_vel_rev_s"),
        "left_torque_nm": stats("left_torque_nm"),
        "right_torque_nm": stats("right_torque_nm"),
    }


def segment_summaries(
    rows: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    end_reason = {
        str(event.get("phase")): str(event.get("reason"))
        for event in events
        if event.get("event") == "segment_end" and event.get("phase")
    }
    summaries = []
    for phase, group in _group_by_phase(rows):
        summaries.append(
            {
                "phase": phase,
                "rows": len(group),
                "duration_s": _finite_or_none(_duration_s(group), 6),
                "end_reason": end_reason.get(phase),
                "left_cmd_mean_nm": _finite_or_none(
                    float(np.mean(_series(group, "left_cmd_nm"))), 6
                ),
                "right_cmd_mean_nm": _finite_or_none(
                    float(np.mean(_series(group, "right_cmd_nm"))), 6
                ),
                "left_vel_peak_rev_s": _finite_or_none(
                    float(np.max(np.abs(_series(group, "left_vel_rev_s")))), 6
                ),
                "right_vel_peak_rev_s": _finite_or_none(
                    float(np.max(np.abs(_series(group, "right_vel_rev_s")))), 6
                ),
                "pitch_peak_rad": _finite_or_none(
                    float(np.max(np.abs(_series(group, "pitch_rad")))), 6
                ),
            }
        )
    return summaries


def voltage_temp_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "left_voltage_v": _range([float(r.get("left_voltage_v", math.nan)) for r in rows], 4),
        "right_voltage_v": _range([float(r.get("right_voltage_v", math.nan)) for r in rows], 4),
        "left_temp_c": _range([float(r.get("left_temp_c", math.nan)) for r in rows], 4),
        "right_temp_c": _range([float(r.get("right_temp_c", math.nan)) for r in rows], 4),
        "min_voltage_sag_v": _finite_or_none(
            min(
                [
                    value
                    for value in [
                        *[float(r.get("left_voltage_v", math.nan)) for r in rows],
                        *[float(r.get("right_voltage_v", math.nan)) for r in rows],
                    ]
                    if math.isfinite(value)
                ],
                default=math.nan,
            ),
            4,
        ),
    }


def event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(event.get("event")) for event in events))


def extract_batch1_priors(priors: dict[str, Any] | None) -> dict[str, Any]:
    """Pull per-wheel motor/actuator parameters out of a batch 1 derived YAML dict.

    Returns a dict keyed by side ("left"/"right"), each containing float-or-None
    values ready for downstream arithmetic.  Missing or non-finite values are None.
    """
    if not priors:
        return {}

    def _prov(node: Any) -> float | None:
        value = node.get("value") if isinstance(node, dict) else node
        f = _to_float(value)
        return f if math.isfinite(f) else None

    result: dict[str, Any] = {}
    for side in ("left", "right"):
        wheel = priors.get("wheels", {}).get(side, {})
        sim = wheel.get("suggested_sim", {})
        diag = wheel.get("diagnostics", {})
        tracking = diag.get("actuator_tracking", {})
        delay = diag.get("response_delay", {})
        cs_node = sim.get("command_sign", {})
        cs_value = cs_node.get("value") if isinstance(cs_node, dict) else cs_node
        result[side] = {
            "frictionloss_nm": _prov(sim.get("frictionloss")),
            "damping_nm_s_per_rad": _prov(sim.get("damping")),
            "armature_kg_m2": _prov(sim.get("armature")),
            "torque_limit_nm": _prov(sim.get("torque_limit_nm")),
            "command_sign": int(cs_value) if cs_value in (-1, 1, "-1", "1") else None,
            "total_wheel_inertia_kg_m2": _prov(diag.get("fitted_total_wheel_inertia_kg_m2")),
            "actuator_gain": _prov(tracking.get("gain")),
            "actuator_bias_nm": _prov(tracking.get("bias_nm")),
            "response_delay_s": _prov(delay.get("median_s")),
        }
    return result


def estimate_loaded_radius(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate loaded wheel radius from the kinematic ratio a_imu / alpha_wheel.

    Pure-rolling constraint: a_linear = r * alpha_wheel.  We compute both from
    telemetry during straight dynamic segments and take the median ratio.
    Plausible range for Finn's wheels: 30 mm - 120 mm.
    """
    ratios: list[float] = []
    for phase, group in _group_by_phase(rows):
        if not (phase.startswith("straight_") or phase.startswith("coast_spinup")):
            continue
        if "settle" in phase:
            continue
        t = _series(group, "t_us") / 1_000_000.0
        a_imu = _series(group, ROBOT_FORWARD_ACCEL_FIELD)
        omega_avg = (
            0.5
            * (_series(group, "left_vel_rev_s") + _series(group, "right_vel_rev_s"))
            * RAD_PER_REV
        )
        finite = np.isfinite(t) & np.isfinite(a_imu) & np.isfinite(omega_avg)
        if int(np.sum(finite)) < 6:
            continue
        t_f, a_f, om_f = t[finite], a_imu[finite], omega_avg[finite]
        alpha = np.gradient(om_f, t_f)
        mask = np.abs(alpha) > 0.5  # rad/s² — ignore near-constant-velocity windows
        if int(np.sum(mask)) < 3:
            continue
        r_samples = np.abs(a_f[mask] / alpha[mask])
        valid = r_samples[(r_samples > 0.03) & (r_samples < 0.12)]
        ratios.extend(valid.tolist())

    if len(ratios) < 10:
        return {
            "sample_count": len(ratios),
            "radius_m": None,
            "radius_std_m": None,
            "confidence": "insufficient",
            "notes": ["too_few_kinematic_ratio_samples"],
        }
    r_arr = np.array(ratios, dtype=float)
    r_med = float(np.median(r_arr))
    r_std = float(np.std(r_arr))
    confidence = "measured" if len(ratios) >= 40 and r_std < 0.005 else "provisional"
    return {
        "sample_count": len(ratios),
        "radius_m": _finite_or_none(r_med, 6),
        "radius_std_m": _finite_or_none(r_std, 6),
        "confidence": confidence,
        "notes": ["kinematic_ratio_imu_accel_over_wheel_angular_accel"],
    }


def decompose_losses(
    loss_fit: LossFit,
    b1_left: dict[str, Any],
    b1_right: dict[str, Any],
    radius_m: float | None,
    mass_kg: float | None,
    left_inertia_kg_m2: float | None,
    right_inertia_kg_m2: float | None,
) -> dict[str, Any]:
    """Split the loaded coastdown loss into motor-bearing and tire-rolling components.

    The loaded coastdown fit gives total linear deceleration coefficients.
    Batch 1 measured the motor contributions off-ground.  With wheel radius and
    robot mass we can convert motor (N·m) values to equivalent linear (m/s²) values
    and subtract them, leaving the tire-only contribution.

    Returns None for all outputs when prerequisite data is missing.
    """
    _insufficient: dict[str, Any] = {
        "motor_friction_accel_m_s2": None,
        "motor_damping_s": None,
        "tire_friction_accel_m_s2": None,
        "tire_damping_s": None,
        "rolling_resistance_coeff": None,
        "motor_fraction_of_total_friction": None,
        "confidence": "insufficient",
        "notes": [],
    }
    if not radius_m or radius_m <= 0 or not mass_kg or mass_kg <= 0:
        return {**_insufficient, "notes": ["missing_loaded_radius_or_mass"]}
    if loss_fit.friction_accel_m_s2 is None or loss_fit.linear_damping_s is None:
        return {**_insufficient, "notes": ["coastdown_fit_insufficient"]}

    f_left = b1_left.get("frictionloss_nm")
    f_right = b1_right.get("frictionloss_nm")
    b_left = b1_left.get("damping_nm_s_per_rad")
    b_right = b1_right.get("damping_nm_s_per_rad")
    if any(v is None for v in (f_left, f_right, b_left, b_right)):
        return {**_insufficient, "notes": ["batch1_motor_parameters_unavailable"]}

    j_left = left_inertia_kg_m2 or 0.0
    j_right = right_inertia_kg_m2 or 0.0
    if not math.isfinite(j_left):
        j_left = 0.0
    if not math.isfinite(j_right):
        j_right = 0.0

    m_eff = mass_kg + (j_left + j_right) / (radius_m**2)
    avg_friction_nm = 0.5 * (float(f_left) + float(f_right))
    avg_damping = 0.5 * (float(b_left) + float(b_right))

    # Convert per-wheel rotational losses to whole-robot linear deceleration
    motor_friction_accel = 2.0 * avg_friction_nm / (radius_m * m_eff)
    motor_damping_s = 2.0 * avg_damping / (radius_m**2 * m_eff)

    total_friction = float(loss_fit.friction_accel_m_s2)
    total_damping = float(loss_fit.linear_damping_s)
    tire_friction = max(0.0, total_friction - motor_friction_accel)
    tire_damping = max(0.0, total_damping - motor_damping_s)
    rolling_coeff = tire_friction / 9.80665
    motor_frac = motor_friction_accel / total_friction if total_friction > 1e-9 else None

    return {
        "motor_friction_accel_m_s2": _finite_or_none(motor_friction_accel, 8),
        "motor_damping_s": _finite_or_none(motor_damping_s, 8),
        "tire_friction_accel_m_s2": _finite_or_none(tire_friction, 8),
        "tire_damping_s": _finite_or_none(tire_damping, 8),
        "rolling_resistance_coeff": _finite_or_none(rolling_coeff, 8),
        "motor_fraction_of_total_friction": _finite_or_none(motor_frac, 4),
        "confidence": loss_fit.confidence,
        "notes": [],
    }


def actuator_cross_check(
    b2_tracking: dict[str, Any],
    b1_side: dict[str, Any],
) -> dict[str, Any]:
    """Compare batch 2 actuator gain/bias against batch 1 off-ground values.

    Gain should be ~1 in both batches (motor torque ≈ commanded torque).
    A delta > 0.1 signals a motor calibration drift or temperature effect worth
    investigating before batch 3.
    """
    b1_gain = b1_side.get("actuator_gain")
    b1_bias = b1_side.get("actuator_bias_nm")
    b2_gain = _to_float(b2_tracking.get("gain"))
    b2_bias = _to_float(b2_tracking.get("bias_nm"))

    if b1_gain is None or not math.isfinite(b2_gain):
        return {
            "status": "insufficient_data",
            "batch1_gain": b1_gain,
            "batch2_gain": _finite_or_none(b2_gain, 6),
            "gain_delta": None,
            "bias_delta_nm": None,
        }

    gain_delta = b2_gain - float(b1_gain)
    b1_bias_f = float(b1_bias) if b1_bias is not None else 0.0
    b2_bias_f = b2_bias if math.isfinite(b2_bias) else 0.0
    bias_delta = b2_bias_f - b1_bias_f
    status = "consistent" if abs(gain_delta) < 0.1 and abs(bias_delta) < 0.02 else "diverged"
    return {
        "status": status,
        "batch1_gain": _finite_or_none(float(b1_gain), 6),
        "batch2_gain": _finite_or_none(b2_gain, 6),
        "gain_delta": _finite_or_none(gain_delta, 5),
        "bias_delta_nm": _finite_or_none(bias_delta, 5),
    }


def mujoco_params(
    b1: dict[str, Any],
    radius_est: dict[str, Any],
    loss_decomp: dict[str, Any],
    yaw: dict[str, Any],
    traction: dict[str, Any],
    phys: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the MuJoCo-ready parameter set from batch 1 priors and batch 2 results.

    This is the primary output that feeds fill_measurements.py → finn_measurements.yaml
    → postprocess_mujoco.py → finn.sim.xml.  Parameters that require batch 3 to
    identify (solref, solimp) are explicitly set to None here.
    """
    radius_m = radius_est.get("radius_m") or phys.get("loaded_wheel_radius_m")
    open_params = []
    if radius_m is None:
        open_params.append("loaded_wheel_radius_m")
    if loss_decomp.get("rolling_resistance_coeff") is None:
        open_params.append("rolling_resistance_coeff")
    open_params.extend(["contact_solref", "contact_solimp"])

    actuators: dict[str, Any] = {}
    for side in ("left", "right"):
        s = b1.get(side, {})
        actuators[side] = {
            "frictionloss_nm": _finite_or_none(_to_float(s.get("frictionloss_nm")), 8),
            "damping_nm_s_per_rad": _finite_or_none(_to_float(s.get("damping_nm_s_per_rad")), 8),
            "armature_kg_m2": _finite_or_none(_to_float(s.get("armature_kg_m2")), 8),
            "command_sign": s.get("command_sign"),
            "torque_limit_nm": _finite_or_none(_to_float(s.get("torque_limit_nm")), 6),
            "source": "batch1_off_ground_sysid",
        }

    radius_source = (
        "batch2_imu_wheel_kinematic_ratio"
        if radius_est.get("radius_m") is not None
        else "finn_measurements_yaml"
    )

    return {
        "actuators": actuators,
        "geometry": {
            "loaded_wheel_radius_m": {
                "value": radius_m,
                "confidence": radius_est.get("confidence", "insufficient"),
                "source": radius_source,
            },
            "effective_track_width_m": {
                "value": yaw.get("effective_track_width_m"),
                "confidence": yaw.get("confidence", "insufficient"),
                "source": "batch2_differential_yaw_response",
            },
        },
        "contact": {
            "slide_friction_mu_lower_bound": traction.get("straight_mu_lower_bound"),
            "rolling_resistance_coeff": loss_decomp.get("rolling_resistance_coeff"),
            "tire_friction_accel_m_s2": loss_decomp.get("tire_friction_accel_m_s2"),
            "solref": None,
            "solimp": None,
            "confidence": loss_decomp.get("confidence", "insufficient"),
            "notes": ["solref_and_solimp_to_be_identified_in_batch3"],
        },
        "readiness": {
            "batch3_can_proceed": all(
                [
                    radius_m is not None,
                    yaw.get("effective_track_width_m") is not None,
                    b1.get("left", {}).get("frictionloss_nm") is not None,
                ]
            ),
            "open_params_for_batch3": open_params,
        },
    }


def lqr_readiness(
    warnings: list[str], rows: list[dict[str, Any]], phys: dict[str, Any], schema: str | None
) -> dict[str, Any]:
    checklist = {
        "has_batch2_schema": schema == "batch2_v1",
        "has_rows": len(rows) > 0,
        "has_stationary_noise": len(stationary_rows(rows)) >= 100,
        "has_loaded_radius": bool(phys.get("loaded_wheel_radius_m")),
        "has_track_width": bool(phys.get("wheel_track_width_m")),
        "no_failsafe": "failsafe_events_present" not in warnings
        and "fault_rows_present" not in warnings,
        "no_safety_limit": not any("safety_limit" in warning for warning in warnings),
    }
    return {
        "pass": all(checklist.values()),
        "checklist": checklist,
        "next_step": "attempt_tethered_lqr_replay"
        if all(checklist.values())
        else "collect_or_fix_batch2_data",
    }


def analyze_run(
    run_dir: Path,
    measurements_path: Path = DEFAULT_MEASUREMENTS,
    batch1_derived: Path | None = None,
    command_signs: dict[str, int] | None = None,
    physical_overrides: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    telemetry = run_dir / "telemetry.csv"
    rows = rows_as_numeric(read_batch2_rows(telemetry))
    schema = read_schema(telemetry)
    events = read_events(run_dir / "events.log")
    measurements = load_measurements(measurements_path)
    phys = physical_inputs(measurements, **(physical_overrides or {}))
    priors = (
        yaml.safe_load(batch1_derived.read_text())
        if batch1_derived and batch1_derived.exists()
        else None
    )
    b1_priors = extract_batch1_priors(priors)
    if command_signs is None:
        cs_left = b1_priors.get("left", {}).get("command_sign")
        cs_right = b1_priors.get("right", {}).get("command_sign")
        if cs_left in (-1, 1) and cs_right in (-1, 1):
            command_signs = {"left": int(cs_left), "right": int(cs_right)}
        else:
            command_signs = {"left": -1, "right": 1}

    robot_mass_kg = phys.get("robot_mass_kg")
    gantry_mass_kg = phys.get("gantry_mass_kg")
    if robot_mass_kg and gantry_mass_kg and gantry_mass_kg > 0:
        system_mass_kg = robot_mass_kg + gantry_mass_kg
    else:
        system_mass_kg = robot_mass_kg

    warnings: list[str] = []
    warnings.append("imu_axis_mapping_uses_sensor_x_pitch_y_yaw_z_forward_verify_signs")
    if gantry_mass_kg and gantry_mass_kg > 0:
        ratio = gantry_mass_kg / robot_mass_kg if robot_mass_kg else 0.0
        warnings.append(
            f"gantry_present_mass_{gantry_mass_kg:.3f}_kg_ratio_{ratio:.2f}_"
            "coastdown_uses_system_mass_rolling_resistance_includes_caster_friction"
        )
    if schema != "batch2_v1":
        warnings.append(f"unexpected_or_missing_schema_{schema}")
    if not rows:
        warnings.append("no_telemetry_rows")
    missing_physical = [key for key, value in phys.items() if value is None and key != "gantry_mass_kg"]
    warnings.extend(f"missing_physical_input_{key}" for key in missing_physical)

    duration = ((rows[-1]["t_us"] - rows[0]["t_us"]) / 1_000_000.0) if len(rows) >= 2 else 0.0
    dts = np.diff(_series(rows, "t_us")) / 1000.0 if len(rows) >= 2 else np.array([])
    faults = [
        row
        for row in rows
        if int(row.get("left_fault", 0) or 0) != 0
        or int(row.get("right_fault", 0) or 0) != 0
        or str(row.get("fault_reason", "none")) not in ("", "none")
    ]
    if faults:
        warnings.append("fault_rows_present")
    if any(event.get("event") == "failsafe" for event in events):
        warnings.append("failsafe_events_present")
    for event in events:
        if event.get("event") == "segment_end" and event.get("reason") == "safety_limit":
            warnings.append(f"safety_limit_{event.get('phase')}")

    loss_fit = fit_loaded_coastdown_loss(
        rows, phys.get("loaded_wheel_radius_m"), system_mass_kg
    )
    yaw = yaw_response(rows, phys.get("loaded_wheel_radius_m"), phys.get("wheel_track_width_m"))
    noise = noise_stats(rows)
    delays = delay_estimates(rows)
    summaries = segment_summaries(rows, events)
    voltage_temp = voltage_temp_summary(rows)
    traction = tire_traction_lower_bounds(
        rows, phys.get("loaded_wheel_radius_m"), phys.get("wheel_track_width_m")
    )
    radius_est = estimate_loaded_radius(rows)
    radius_for_decomp = radius_est.get("radius_m") or phys.get("loaded_wheel_radius_m")
    loss_decomp = decompose_losses(
        loss_fit,
        b1_priors.get("left", {}),
        b1_priors.get("right", {}),
        radius_for_decomp,
        system_mass_kg,
        b1_priors.get("left", {}).get("total_wheel_inertia_kg_m2"),
        b1_priors.get("right", {}).get("total_wheel_inertia_kg_m2"),
    )
    b2_tracking_left = actuator_tracking(rows, "left")
    b2_tracking_right = actuator_tracking(rows, "right")
    cross_check_left = actuator_cross_check(b2_tracking_left, b1_priors.get("left", {}))
    cross_check_right = actuator_cross_check(b2_tracking_right, b1_priors.get("right", {}))
    mujoco = mujoco_params(b1_priors, radius_est, loss_decomp, yaw, traction, phys)

    validated_loaded_torque = max(
        [abs(float(row.get("left_cmd_nm", 0.0))) for row in rows]
        + [abs(float(row.get("right_cmd_nm", 0.0))) for row in rows],
        default=0.0,
    )

    derived: dict[str, Any] = {
        "schema_version": 1,
        "source": "finn_mcu_batch2_loaded_ground_sysid",
        "metadata": {
            "run_path": str(run_dir),
            "telemetry_schema": schema,
            "rows": len(rows),
            "duration_s": _finite_or_none(duration, 6),
            "sample_rate": {
                "dt_ms_mean": _finite_or_none(float(np.mean(dts)) if len(dts) else None, 6),
                "dt_ms_min": _finite_or_none(float(np.min(dts)) if len(dts) else None, 6),
                "dt_ms_max": _finite_or_none(float(np.max(dts)) if len(dts) else None, 6),
            },
            "event_counts": event_counts(events),
            "fault_counts": {
                "fault_rows": len(faults),
                "failsafe_events": sum(1 for event in events if event.get("event") == "failsafe"),
            },
            "voltage_temp_ranges": voltage_temp,
        },
        "inputs": {
            "measurements": str(measurements_path),
            "physical": phys,
            "batch1_derived": str(batch1_derived) if batch1_derived else None,
            "batch1_priors_available": bool(b1_priors),
            "batch1_priors": b1_priors if b1_priors else None,
        },
        "suggested_sim": {
            "wheels": {
                "command_signs": command_signs,
                "validated_loaded_torque_nm": _finite_or_none(validated_loaded_torque, 6),
                "actuator_tracking": {
                    "left": b2_tracking_left,
                    "right": b2_tracking_right,
                },
                "loaded_radius_estimate": radius_est,
                "loaded_loss_decomposition": loss_decomp,
                "loaded_loss_correction": {
                    "linear_damping_s": _finite_or_none(loss_fit.linear_damping_s, 8),
                    "friction_accel_m_s2": _finite_or_none(loss_fit.friction_accel_m_s2, 8),
                    "per_wheel_torque_nm": _finite_or_none(loss_fit.per_wheel_torque_nm, 8),
                    "source": "loaded_coastdown_output_error",
                    "confidence": loss_fit.confidence,
                    "bound_active": loss_fit.bound_active,
                },
                "left_right_asymmetry": {
                    "mean_torque_delta_nm": _finite_or_none(
                        float(
                            np.mean(
                                _series(rows, "left_torque_nm") - _series(rows, "right_torque_nm")
                            )
                        )
                        if rows
                        else None,
                        8,
                    ),
                    "mean_velocity_delta_rev_s": _finite_or_none(
                        float(
                            np.mean(
                                _series(rows, "left_vel_rev_s") - _series(rows, "right_vel_rev_s")
                            )
                        )
                        if rows
                        else None,
                        8,
                    ),
                },
                "yaw_response": yaw,
            },
            "contact": {
                "tire_friction_lower_bounds": traction,
                "rolling_loss_estimates": {
                    "friction_accel_m_s2": _finite_or_none(loss_fit.friction_accel_m_s2, 8),
                    "per_wheel_torque_nm": _finite_or_none(loss_fit.per_wheel_torque_nm, 8),
                    "confidence": loss_fit.confidence,
                },
                "contact_compliance": None,
                "notes": ["do_not_emit_confident_solref_or_solimp_from_onboard_only_batch2"],
            },
            "sensors": noise,
            "delays": delays,
        },
        "diagnostics": {
            "segment_summaries": summaries,
            "warnings": warnings,
            "fit_confidence": {
                "loaded_loss": loss_fit.confidence,
                "yaw_response": yaw["confidence"],
            },
            "residuals": {
                "loaded_loss_rmse_m_s": _finite_or_none(loss_fit.rmse_m_s, 8),
            },
            "saturation_flags": {
                "pitch_limit_rows": int(sum(_safe_int(row.get("pitch_limit", 0)) for row in rows)),
                "speed_limit_rows": int(sum(_safe_int(row.get("speed_limit", 0)) for row in rows)),
                "travel_limit_rows": int(sum(_safe_int(row.get("travel_limit", 0)) for row in rows)),
                "max_command_near_hard_cap": validated_loaded_torque >= 0.249,
            },
            "actuator_cross_check": {
                "left": cross_check_left,
                "right": cross_check_right,
            },
        },
        "mujoco_params": mujoco,
        "lqr_readiness": {},
        "warnings": warnings,
        "do_not_infer": [
            "global_pose_without_external_ground_truth",
            "absolute_floor_slip_without_external_pose_or_video",
            "contact_solref",
            "contact_solimp",
            "contact_compliance",
            "full_body_inertia_from_onboard_batch2_only",
            "signed_robot_frame_pitch_yaw_forward_accel_until_imu_axis_signs_confirmed",
        ],
    }
    derived["lqr_readiness"] = lqr_readiness(warnings, rows, phys, schema)
    return derived


def write_derived_yaml(path: Path, derived: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")


def write_segment_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        path.write_text("", encoding="utf-8")
        return
    fields = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def write_report(path: Path, derived: dict[str, Any]) -> None:
    sim = derived["suggested_sim"]
    wheel = sim["wheels"]
    contact = sim["contact"]
    loaded_loss = wheel["loaded_loss_correction"]
    yaw = wheel["yaw_response"]
    traction = contact["tire_friction_lower_bounds"]
    loss_decomp = wheel.get("loaded_loss_decomposition", {})
    radius_est = wheel.get("loaded_radius_estimate", {})
    mujoco = derived.get("mujoco_params", {})
    mujoco_geo = mujoco.get("geometry", {})
    mujoco_contact = mujoco.get("contact", {})
    mujoco_readiness = mujoco.get("readiness", {})

    lines = [
        "# Batch 2 loaded ground-contact sysid",
        "",
        "## Warnings",
        "",
    ]
    warnings = derived.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")

    lines.extend(
        [
            "",
            "## Summary Values For Sim",
            "",
            "| Value | Estimate | Confidence |",
            "| --- | ---: | --- |",
            f"| validated loaded torque Nm | {wheel['validated_loaded_torque_nm']} | measured |",
            "| loaded friction accel m/s^2 (total) | "
            f"{loaded_loss['friction_accel_m_s2']} | {loaded_loss['confidence']} |",
            "| per-wheel rolling loss torque Nm (total) | "
            f"{loaded_loss['per_wheel_torque_nm']} | {loaded_loss['confidence']} |",
            "| tire-only friction accel m/s^2 | "
            + f"{loss_decomp.get('tire_friction_accel_m_s2')}"
            + f" | {loss_decomp.get('confidence', 'insufficient')} |",
            "| rolling resistance coeff | "
            + f"{loss_decomp.get('rolling_resistance_coeff')}"
            + f" | {loss_decomp.get('confidence', 'insufficient')} |",
            "| loaded wheel radius m | "
            f"{radius_est.get('radius_m')} | {radius_est.get('confidence', 'insufficient')} |",
            f"| effective track width m | {yaw['effective_track_width_m']} | {yaw['confidence']} |",
            f"| straight mu lower bound | {traction['straight_mu_lower_bound']} | lower_bound |",
        ]
    )

    lines.extend(
        [
            "",
            "## Actuator Tracking",
            "",
            "| Side | batch2 gain | batch1 gain | delta | status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    cross_checks = derived.get("diagnostics", {}).get("actuator_cross_check", {})
    for side in ("left", "right"):
        tracking = wheel["actuator_tracking"][side]
        cc = cross_checks.get(side, {})
        lines.append(
            f"| {side} | {tracking['gain']} | {cc.get('batch1_gain')} | "
            f"{cc.get('gain_delta')} | {cc.get('status', 'n/a')} |"
        )

    lines.extend(
        [
            "",
            "## MuJoCo Parameters",
            "",
            "### Actuators (from batch 1)",
            "",
            "| Side | frictionloss Nm | damping Nm*s/rad | armature kg*m2 | sign | limit Nm |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for side in ("left", "right"):
        act = mujoco.get("actuators", {}).get(side, {})
        lines.append(
            f"| {side} | {act.get('frictionloss_nm')} | {act.get('damping_nm_s_per_rad')}"
            f" | {act.get('armature_kg_m2')} | {act.get('command_sign')}"
            f" | {act.get('torque_limit_nm')} |"
        )

    lines.extend(
        [
            "",
            "### Geometry (from batch 2)",
            "",
            "- loaded_wheel_radius_m: "
            + f"{mujoco_geo.get('loaded_wheel_radius_m', {}).get('value')}"
            + f" ({mujoco_geo.get('loaded_wheel_radius_m', {}).get('confidence', 'insufficient')})",
            "- effective_track_width_m: "
            + f"{mujoco_geo.get('effective_track_width_m', {}).get('value')}"
            + " ("
            + f"{mujoco_geo.get('effective_track_width_m', {}).get('confidence', 'insufficient')})",
            "",
            "### Contact (batch 2 lower bounds; solref/solimp deferred to batch 3)",
            "",
            "- slide_friction_mu_lower_bound: "
            f"{mujoco_contact.get('slide_friction_mu_lower_bound')}",
            f"- rolling_resistance_coeff: {mujoco_contact.get('rolling_resistance_coeff')}",
            f"- solref: {mujoco_contact.get('solref')}",
            f"- solimp: {mujoco_contact.get('solimp')}",
            "",
            f"Batch 3 can proceed: `{mujoco_readiness.get('batch3_can_proceed')}`",
            "Open params for batch 3: "
            + ", ".join(mujoco_readiness.get("open_params_for_batch3") or []),
        ]
    )

    lines.extend(
        [
            "",
            "## LQR Readiness",
            "",
            f"Pass: `{derived['lqr_readiness']['pass']}`",
        ]
    )
    for key, value in derived["lqr_readiness"]["checklist"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Do Not Infer", ""])
    lines.extend(f"- {item}" for item in derived["do_not_infer"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--batch1-derived", type=Path)
    parser.add_argument("--left-command-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--right-command-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument(
        "--gantry-mass-kg",
        type=float,
        help="Moving mass of the safety gantry in kg. Added to robot mass for coastdown "
        "force balance. Read from measurements YAML (gantry.mass_kg) if omitted.",
    )
    parser.add_argument("--loaded-wheel-radius-m", type=float)
    parser.add_argument("--wheel-track-width-m", type=float)
    parser.add_argument("--com-height-m", type=float)
    parser.add_argument("--com-fore-aft-m", type=float)
    parser.add_argument("--pitch-inertia-kg-m2", type=float)
    args = parser.parse_args()

    out_dir = args.run_dir / "postprocess"
    out_dir.mkdir(exist_ok=True)
    derived = analyze_run(
        args.run_dir,
        measurements_path=args.measurements,
        batch1_derived=args.batch1_derived,
        command_signs={"left": args.left_command_sign, "right": args.right_command_sign},
        physical_overrides={
            "mass_kg": args.mass_kg,
            "gantry_mass_kg": args.gantry_mass_kg,
            "loaded_wheel_radius_m": args.loaded_wheel_radius_m,
            "wheel_track_width_m": args.wheel_track_width_m,
            "com_height_m": args.com_height_m,
            "com_fore_aft_m": args.com_fore_aft_m,
            "pitch_inertia_kg_m2": args.pitch_inertia_kg_m2,
        },
    )
    write_derived_yaml(out_dir / "derived.yaml", derived)
    write_report(out_dir / "report.md", derived)
    write_segment_summary_csv(
        out_dir / "segment_summary.csv", derived["diagnostics"]["segment_summaries"]
    )
    print(
        f"Wrote {out_dir / 'report.md'}, {out_dir / 'derived.yaml'}, "
        f"and {out_dir / 'segment_summary.csv'} ({derived['metadata']['rows']} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
