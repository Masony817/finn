#!/usr/bin/env python3
"""Analyze a captured Batch 1 wheel sysid run and emit sim-derived YAML."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import least_squares, lsq_linear

from finn.telemetry import read_events as read_events
from finn.telemetry import read_rows
from finn.telemetry import read_schema as read_schema

# Batch 1 runs end at power-off, so a final truncated row is normal, not an error.
read_batch1_rows = partial(read_rows, strict=False)

MOTION_VELOCITY_REV_S = 0.02
MOTION_POSITION_REV = 0.01
SUSTAINED_MOTION_POSITION_REV = 0.06
SUSTAINED_MOTION_MIN_SAMPLES = 3
MIN_DYNAMIC_SPEED_RAD_S = 0.05
RAD_PER_REV = 2.0 * math.pi
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_XML = REPO_ROOT / "sim" / "model" / "finn" / "finn_robot.xml"
DEFAULT_MEASUREMENTS = REPO_ROOT / "sim" / "config" / "finn_measurements.yaml"


def portable_path(path: Path | None) -> str | None:
    """Prefer portable repository-relative paths in generated artifacts."""
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


NUMERIC_FIELDS = {
    "t_us",
    "phase_index",
    "armed",
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
}


@dataclass(frozen=True)
class PhaseStats:
    phase: str
    side: str | None
    direction: str | None
    command_nm: float
    mean_torque_nm: float
    duration_s: float
    max_abs_velocity_rev_s: float
    max_abs_velocity_rad_s: float
    position_delta_rev: float
    moving: bool
    sample_count: int
    first_motion: bool = False
    sustained_motion: bool = False


@dataclass(frozen=True)
class BreakawayInterval:
    low_nm: float | None
    high_nm: float | None
    estimate_nm: float | None
    samples: int
    anomalous: bool
    notes: list[str]


@dataclass(frozen=True)
class DynamicFit:
    armature: float | None
    damping: float | None
    frictionloss: float | None
    rmse_nm: float | None
    sample_count: int
    confidence: str
    notes: list[str]


@dataclass(frozen=True)
class CoastdownFit:
    damping_per_inertia: float | None
    friction_per_inertia: float | None
    rmse_rad_s: float | None
    sample_count: int
    confidence: str
    bound_active: dict[str, bool]
    notes: list[str]


@dataclass(frozen=True)
class PoweredInertiaFit:
    inertia_kg_m2: float | None
    rmse_rad_s: float | None
    sample_count: int
    confidence: str
    bound_active: bool
    notes: list[str]


def rows_as_numeric(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in NUMERIC_FIELDS:
                item[key] = _to_float(value)
            else:
                item[key] = value
        converted.append(item)
    return converted


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite_or_none(value: float | None, ndigits: int = 8) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), ndigits)


def _parse_vec(value: str | None, expected: int) -> list[float]:
    if not value:
        return []
    parts = [float(part) for part in value.split()]
    return parts if len(parts) == expected else []


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


def physical_inputs(measurements: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_mass_kg": _measurement_value(measurements, ("robot", "mass_kg")),
        "robot_mass_no_battery_kg": _measurement_value(
            measurements, ("robot", "mass_no_battery_kg")
        ),
        "loaded_wheel_radius_m": _mean_present(
            [
                _measurement_value(measurements, ("wheels", "left", "radius_m")),
                _measurement_value(measurements, ("wheels", "right", "radius_m")),
            ]
        ),
        "wheel_track_width_m": _measurement_value(measurements, ("robot", "wheel_track_width_m")),
        "left_wheel_mass_kg": _measurement_value(measurements, ("wheels", "left", "mass_kg")),
        "right_wheel_mass_kg": _measurement_value(measurements, ("wheels", "right", "mass_kg")),
        "left_gear_ratio": _measurement_value(measurements, ("wheels", "left", "gear_ratio")),
        "right_gear_ratio": _measurement_value(measurements, ("wheels", "right", "gear_ratio")),
        "com_height_m": _measurement_value(measurements, ("robot", "com_height_m"))
        or _measurement_value(measurements, ("robot", "com", "z_m")),
        "com_fore_aft_m": _measurement_value(measurements, ("robot", "com_fore_aft_m"))
        or _measurement_value(measurements, ("robot", "com", "x_m")),
    }


def _mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and value > 0.0]
    return float(np.mean(present)) if present else None


def _quat_to_matrix(quat: list[float]) -> np.ndarray:
    if len(quat) != 4:
        return np.eye(3)
    w, x, y, z = quat
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def cad_axial_inertias_from_robot_xml(robot_xml: Path) -> dict[str, float]:
    if not robot_xml.exists():
        return {}
    root = ET.parse(robot_xml).getroot()
    result: dict[str, float] = {}
    for side in ("left", "right"):
        body_name = f"{side}_wheel"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        joint = body.find(f"joint[@name='{body_name}']")
        if joint is None:
            joint = body.find("joint")
        inertial = body.find("inertial")
        if joint is None or inertial is None:
            continue
        axis = np.array(_parse_vec(joint.get("axis"), 3) or [0.0, 0.0, 1.0], dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 0.0:
            continue
        axis = axis / axis_norm
        full = _parse_vec(inertial.get("fullinertia"), 6)
        if len(full) == 6:
            ixx, iyy, izz, ixy, ixz, iyz = full
            inertia = np.array(
                [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
                dtype=float,
            )
        else:
            diag = _parse_vec(inertial.get("diaginertia"), 3)
            if len(diag) != 3:
                continue
            inertia = np.diag(diag)
        quat = _parse_vec(inertial.get("quat"), 4)
        if quat:
            rotation = _quat_to_matrix(quat)
            inertia = rotation @ inertia @ rotation.T
        result[side] = float(axis @ inertia @ axis)
    return result


def _group_by_phase(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: list[tuple[str, list[dict[str, Any]]]] = []
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        phase = str(row.get("phase", ""))
        if phase not in by_name:
            by_name[phase] = []
            grouped.append((phase, by_name[phase]))
        by_name[phase].append(row)
    return grouped


def _side_from_phase(phase: str) -> str | None:
    if phase.startswith("left_") or "_left_" in phase:
        return "left"
    if phase.startswith("right_") or "_right_" in phase:
        return "right"
    return None


def _active_side(row: dict[str, Any]) -> str | None:
    left = abs(float(row.get("left_cmd_nm", 0.0)))
    right = abs(float(row.get("right_cmd_nm", 0.0)))
    if left > 1e-5 and left >= right:
        return "left"
    if right > 1e-5:
        return "right"
    return _side_from_phase(str(row.get("phase", "")))


def _direction(value: float) -> str | None:
    if value > 1e-5:
        return "pos"
    if value < -1e-5:
        return "neg"
    return None


def segment_end_reasons(events: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(event["phase"]): str(event["reason"])
        for event in events
        if event.get("event") == "segment_end" and event.get("phase") and event.get("reason")
    }


def _has_sustained_motion(velocity: np.ndarray, position_delta: float) -> bool:
    if abs(position_delta) >= SUSTAINED_MOTION_POSITION_REV:
        return True
    above = np.abs(velocity) >= MOTION_VELOCITY_REV_S
    run = 0
    for item in above:
        if item:
            run += 1
            if run >= SUSTAINED_MOTION_MIN_SAMPLES:
                return True
        else:
            run = 0
    return False


def phase_stats(
    rows: list[dict[str, Any]],
    end_reasons: dict[str, str] | None = None,
) -> list[PhaseStats]:
    end_reasons = end_reasons or {}
    stats: list[PhaseStats] = []
    for phase, group in _group_by_phase(rows):
        if not group:
            continue
        side = _active_side(group[0])
        if side is None:
            stats.append(
                PhaseStats(
                    phase=phase,
                    side=None,
                    direction=None,
                    command_nm=0.0,
                    mean_torque_nm=0.0,
                    duration_s=_duration_s(group),
                    max_abs_velocity_rev_s=0.0,
                    max_abs_velocity_rad_s=0.0,
                    position_delta_rev=0.0,
                    moving=False,
                    sample_count=len(group),
                    first_motion=False,
                    sustained_motion=False,
                )
            )
            continue

        command = float(np.mean([r[f"{side}_cmd_nm"] for r in group]))
        velocity = np.array([float(r[f"{side}_vel_rev_s"]) for r in group])
        position = np.array([float(r[f"{side}_pos_rev"]) for r in group])
        torque = np.array([float(r[f"{side}_torque_nm"]) for r in group])
        position_delta = float(position[-1] - position[0]) if len(position) else 0.0
        max_abs_velocity = float(np.max(np.abs(velocity))) if len(velocity) else 0.0
        moving = (
            max_abs_velocity >= MOTION_VELOCITY_REV_S
            or abs(position_delta) >= MOTION_POSITION_REV
            or end_reasons.get(phase) == "motion_reached"
        )
        sustained_motion = _has_sustained_motion(velocity, position_delta)
        stats.append(
            PhaseStats(
                phase=phase,
                side=side,
                direction=_direction(command),
                command_nm=command,
                mean_torque_nm=float(np.mean(torque)) if len(torque) else 0.0,
                duration_s=_duration_s(group),
                max_abs_velocity_rev_s=max_abs_velocity,
                max_abs_velocity_rad_s=max_abs_velocity * RAD_PER_REV,
                position_delta_rev=position_delta,
                moving=moving,
                sample_count=len(group),
                first_motion=moving,
                sustained_motion=sustained_motion,
            )
        )
    return stats


def _duration_s(group: list[dict[str, Any]]) -> float:
    if len(group) < 2:
        return 0.0
    return (float(group[-1]["t_us"]) - float(group[0]["t_us"])) / 1_000_000.0


def detect_breakaway_intervals(stats: list[PhaseStats]) -> dict[str, dict[str, BreakawayInterval]]:
    result: dict[str, dict[str, BreakawayInterval]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for direction in ("pos", "neg"):
            candidates = [
                s
                for s in stats
                if s.side == side
                and s.direction == direction
                and _is_breakaway_phase(s.phase)
                and abs(s.command_nm) > 0.0
            ]
            candidates.sort(key=lambda item: abs(item.command_nm))
            no_motion = [abs(s.command_nm) for s in candidates if not s.moving]
            motion = [abs(s.command_nm) for s in candidates if s.moving]
            low = max(no_motion) if no_motion else None
            high = min(motion) if motion else None
            notes: list[str] = []
            anomalous = False
            estimate = None
            if candidates and not motion:
                anomalous = True
                notes.append("no_motion_at_highest_tested_torque")
            if low is not None and high is not None:
                estimate = (low + high) / 2.0
            elif high is not None:
                low = 0.0
                estimate = high / 2.0
                notes.append("motion_seen_at_lowest_tested_torque")
            elif low is not None:
                notes.append("breakaway_above_tested_range")
            else:
                notes.append("no_breakaway_samples")
            result[side][direction] = BreakawayInterval(
                low_nm=low,
                high_nm=high,
                estimate_nm=estimate,
                samples=len(candidates),
                anomalous=anomalous,
                notes=notes,
            )
    return result


def summarize_breakaway_diagnostics(stats: list[PhaseStats], side: str) -> dict[str, Any]:
    candidates = [
        item
        for item in stats
        if item.side == side
        and item.direction in ("pos", "neg")
        and _is_breakaway_phase(item.phase)
    ]
    first_motion_torques = [abs(item.command_nm) for item in candidates if item.first_motion]
    sustained_motion_torques = [
        abs(item.command_nm) for item in candidates if item.sustained_motion
    ]
    floor_source = sustained_motion_torques or first_motion_torques
    if floor_source:
        floor = min(floor_source)
        scatter = max(floor_source) - min(floor_source)
        amplitude = scatter / 2.0
    else:
        floor = None
        scatter = None
        amplitude = None
    return {
        "friction_floor_nm": _finite_or_none(floor, 6),
        "breakaway_scatter_nm": _finite_or_none(scatter, 6),
        "cogging_amplitude_nm": _finite_or_none(amplitude, 6),
        "first_motion_sample_count": len(first_motion_torques),
        "sustained_motion_sample_count": len(sustained_motion_torques),
        "source": "sustained_motion" if sustained_motion_torques else "first_motion_fallback",
    }


def _is_breakaway_phase(phase: str) -> bool:
    lower = phase.lower()
    if "breakaway" in lower or "deadband" in lower:
        return True
    return "step" in lower and "spinup" not in lower and "dynamic" not in lower


def fit_wheel_dynamics(rows: list[dict[str, Any]], side: str) -> DynamicFit:
    samples_alpha: list[float] = []
    samples_omega: list[float] = []
    samples_tau: list[float] = []
    notes: list[str] = []

    for phase, group in _group_by_phase(rows):
        if len(group) < 6:
            continue
        if _active_side(group[0]) != side:
            continue
        if not _is_dynamic_phase(phase):
            continue

        t = np.array([float(r["t_us"]) / 1_000_000.0 for r in group], dtype=float)
        omega = np.array([float(r[f"{side}_vel_rev_s"]) * RAD_PER_REV for r in group], dtype=float)
        tau = np.array([float(r[f"{side}_torque_nm"]) for r in group], dtype=float)
        if len(omega) >= 7:
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            smooth = np.convolve(omega, kernel, mode="same")
            trim = kernel_size // 2
            t = t[trim:-trim]
            omega = smooth[trim:-trim]
            tau = tau[trim:-trim]
        if len(omega) < 3:
            continue
        alpha = np.gradient(omega, t)
        mask = np.abs(omega) >= MIN_DYNAMIC_SPEED_RAD_S
        samples_alpha.extend(alpha[mask].tolist())
        samples_omega.extend(omega[mask].tolist())
        samples_tau.extend(tau[mask].tolist())

    if len(samples_tau) < 20:
        return DynamicFit(
            None, None, None, None, len(samples_tau), "insufficient", ["too_few_samples"]
        )

    alpha = np.array(samples_alpha, dtype=float)
    omega = np.array(samples_omega, dtype=float)
    tau = np.array(samples_tau, dtype=float)
    design = np.column_stack([alpha, omega, np.sign(omega)])
    fit = lsq_linear(design, tau, bounds=(0.0, np.inf), lsmr_tol="auto")
    coeffs = fit.x
    residual = design @ coeffs - tau
    rmse = float(np.sqrt(np.mean(residual * residual)))
    confidence = "measured"
    if len(samples_tau) < 80 or rmse > 0.05:
        confidence = "provisional"
    if rmse > 0.05:
        notes.append("high_torque_fit_rmse")

    return DynamicFit(
        armature=float(coeffs[0]),
        damping=float(coeffs[1]),
        frictionloss=float(coeffs[2]),
        rmse_nm=rmse,
        sample_count=len(samples_tau),
        confidence=confidence,
        notes=notes,
    )


def _fit_segments(
    rows: list[dict[str, Any]], side: str, kind: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for phase, group in _group_by_phase(rows):
        if len(group) < 6:
            continue
        if _active_side(group[0]) != side:
            continue
        lower = phase.lower()
        if kind == "coast":
            if "coast" not in lower or "spinup" in lower or "settle" in lower:
                continue
            group = [r for r in group if abs(float(r.get(f"{side}_cmd_nm", 0.0))) <= 1e-5]
        elif kind == "powered":
            if "coastdown" in lower and "spinup" not in lower:
                continue
            if max(abs(float(r.get(f"{side}_cmd_nm", 0.0))) for r in group) <= 1e-5:
                continue
            if not any(token in lower for token in ("dynamic", "spinup", "pulse", "step")):
                continue
        else:
            continue

        t = np.array([float(r["t_us"]) / 1_000_000.0 for r in group], dtype=float)
        omega = np.array([float(r[f"{side}_vel_rev_s"]) * RAD_PER_REV for r in group], dtype=float)
        tau = np.array([float(r[f"{side}_torque_nm"]) for r in group], dtype=float)
        finite = np.isfinite(t) & np.isfinite(omega) & np.isfinite(tau)
        t = t[finite]
        omega = omega[finite]
        tau = tau[finite]
        if len(t) < 6 or float(np.max(np.abs(omega))) < MIN_DYNAMIC_SPEED_RAD_S:
            continue
        segments.append((t - t[0], omega, tau))
    return segments


def _dry_friction_sign(omega: float, tau: float = 0.0) -> float:
    if abs(omega) >= MIN_DYNAMIC_SPEED_RAD_S:
        return math.copysign(1.0, omega)
    if abs(tau) > 1e-6:
        return math.copysign(1.0, tau)
    return 0.0


def _simulate_coast_segment(
    t: np.ndarray, omega0: float, damping_per_j: float, friction_per_j: float
) -> np.ndarray:
    predicted = np.empty_like(t)
    predicted[0] = omega0
    for index in range(1, len(t)):
        dt = max(float(t[index] - t[index - 1]), 0.0)
        sign = _dry_friction_sign(float(predicted[index - 1]))
        predicted[index] = (
            predicted[index - 1]
            + (-damping_per_j * predicted[index - 1] - friction_per_j * sign) * dt
        )
    return predicted


def _simulate_powered_segment(
    t: np.ndarray,
    omega0: float,
    tau: np.ndarray,
    inertia: float,
    damping_per_j: float,
    friction_per_j: float,
) -> np.ndarray:
    predicted = np.empty_like(t)
    predicted[0] = omega0
    for index in range(1, len(t)):
        dt = max(float(t[index] - t[index - 1]), 0.0)
        sign = _dry_friction_sign(float(predicted[index - 1]), float(tau[index - 1]))
        predicted[index] = (
            predicted[index - 1]
            + (
                float(tau[index - 1]) / inertia
                - damping_per_j * predicted[index - 1]
                - friction_per_j * sign
            )
            * dt
        )
    return predicted


def fit_coastdown_output_error(rows: list[dict[str, Any]], side: str) -> CoastdownFit:
    segments = _fit_segments(rows, side, "coast")
    sample_count = sum(len(omega) for _t, omega, _tau in segments)
    if sample_count < 20:
        return CoastdownFit(
            None,
            None,
            None,
            sample_count,
            "insufficient",
            {"damping": False, "frictionloss": False},
            ["too_few_coastdown_samples"],
        )

    def residual(params: np.ndarray) -> np.ndarray:
        damping_per_j, friction_per_j = params
        parts = [
            _simulate_coast_segment(t, float(omega[0]), damping_per_j, friction_per_j) - omega
            for t, omega, _tau in segments
        ]
        return np.concatenate(parts)

    fit = least_squares(residual, x0=np.array([0.5, 2.0]), bounds=(0.0, np.inf))
    rmse = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    confidence = "measured"
    notes: list[str] = []
    if sample_count < 80 or rmse > 0.5 or not fit.success:
        confidence = "provisional"
    if rmse > 0.5:
        notes.append("high_velocity_fit_rmse")
    if not fit.success:
        notes.append("optimizer_not_successful")
    return CoastdownFit(
        damping_per_inertia=float(fit.x[0]),
        friction_per_inertia=float(fit.x[1]),
        rmse_rad_s=rmse,
        sample_count=sample_count,
        confidence=confidence,
        bound_active={
            "damping": bool(fit.x[0] <= 1e-9),
            "frictionloss": bool(fit.x[1] <= 1e-9),
        },
        notes=notes,
    )


def fit_powered_inertia_output_error(
    rows: list[dict[str, Any]],
    side: str,
    damping_per_j: float | None,
    friction_per_j: float | None,
    initial_inertia_kg_m2: float | None = None,
) -> PoweredInertiaFit:
    segments = _fit_segments(rows, side, "powered")
    sample_count = sum(len(omega) for _t, omega, _tau in segments)
    if sample_count < 20:
        return PoweredInertiaFit(
            None,
            None,
            sample_count,
            "insufficient",
            False,
            ["too_few_powered_samples"],
        )

    damping_per_j = damping_per_j or 0.0
    friction_per_j = friction_per_j or 0.0
    initial = (
        initial_inertia_kg_m2 if initial_inertia_kg_m2 and initial_inertia_kg_m2 > 0 else 0.005
    )
    x0 = np.array([initial])

    def residual(params: np.ndarray) -> np.ndarray:
        inertia = max(float(params[0]), 1e-9)
        parts = [
            _simulate_powered_segment(
                t, float(omega[0]), tau, inertia, damping_per_j, friction_per_j
            )
            - omega
            for t, omega, tau in segments
        ]
        return np.concatenate(parts)

    fit = least_squares(residual, x0=x0, bounds=(1e-7, 0.2))
    rmse = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    confidence = "measured"
    notes: list[str] = []
    if sample_count < 80 or rmse > 0.75 or not fit.success:
        confidence = "provisional"
    if rmse > 0.75:
        notes.append("high_velocity_fit_rmse")
    if not fit.success:
        notes.append("optimizer_not_successful")
    return PoweredInertiaFit(
        inertia_kg_m2=float(fit.x[0]),
        rmse_rad_s=rmse,
        sample_count=sample_count,
        confidence=confidence,
        bound_active=bool(fit.x[0] <= 1.01e-7),
        notes=notes,
    )


def _is_dynamic_phase(phase: str) -> bool:
    lower = phase.lower()
    return any(token in lower for token in ("dynamic", "pulse", "spinup", "coast", "step"))


def actuator_fit(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    cmd = np.array([float(r[f"{side}_cmd_nm"]) for r in rows], dtype=float)
    tau = np.array([float(r[f"{side}_torque_nm"]) for r in rows], dtype=float)
    mask = np.abs(cmd) > 1e-5
    if int(np.sum(mask)) < 3:
        return {"sample_count": int(np.sum(mask)), "gain": None, "bias_nm": None, "rmse_nm": None}
    x = np.column_stack([cmd[mask], np.ones(int(np.sum(mask)))])
    gain, bias = np.linalg.lstsq(x, tau[mask], rcond=None)[0]
    residual = x @ np.array([gain, bias]) - tau[mask]
    return {
        "sample_count": int(np.sum(mask)),
        "gain": _finite_or_none(float(gain), 6),
        "bias_nm": _finite_or_none(float(bias), 6),
        "rmse_nm": _finite_or_none(float(np.sqrt(np.mean(residual * residual))), 6),
    }


def response_delay_s(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    delays: list[float] = []
    for _phase, group in _group_by_phase(rows):
        if not group or _active_side(group[0]) != side:
            continue
        if abs(float(group[0][f"{side}_cmd_nm"])) <= 1e-5:
            continue
        start_t = float(group[0]["t_us"])
        start_pos = float(group[0][f"{side}_pos_rev"])
        for row in group:
            vel = abs(float(row[f"{side}_vel_rev_s"]))
            delta = abs(float(row[f"{side}_pos_rev"]) - start_pos)
            if vel >= MOTION_VELOCITY_REV_S or delta >= MOTION_POSITION_REV:
                delays.append((float(row["t_us"]) - start_t) / 1_000_000.0)
                break
    if not delays:
        return {"sample_count": 0, "median_s": None, "min_s": None, "max_s": None}
    return {
        "sample_count": len(delays),
        "median_s": _finite_or_none(float(np.median(delays)), 5),
        "min_s": _finite_or_none(float(np.min(delays)), 5),
        "max_s": _finite_or_none(float(np.max(delays)), 5),
    }


def velocity_noise(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    values = [
        abs(float(r[f"{side}_vel_rev_s"]))
        for r in rows
        if abs(float(r.get(f"{side}_cmd_nm", 0.0))) <= 1e-5
        and any(token in str(r.get("phase", "")).lower() for token in ("idle", "stop", "settle"))
    ]
    if not values:
        return {"sample_count": 0, "p95_rev_s": None, "max_rev_s": None}
    arr = np.array(values, dtype=float)
    return {
        "sample_count": len(arr),
        "p95_rev_s": _finite_or_none(float(np.percentile(arr, 95)), 8),
        "max_rev_s": _finite_or_none(float(np.max(arr)), 8),
    }


def numeric_range(values: list[float], ndigits: int) -> list[float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [None, None]
    return [_finite_or_none(min(finite), ndigits), _finite_or_none(max(finite), ndigits)]


def analyze_run(
    run_dir: Path,
    hard_torque_limit_nm: float,
    command_signs: dict[str, int],
    robot_xml: Path | None = DEFAULT_ROBOT_XML,
    measurements_path: Path = DEFAULT_MEASUREMENTS,
    cad_axial_inertia_overrides: dict[str, float | None] | None = None,
    kt_metadata: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    telemetry = run_dir / "telemetry.csv"
    rows = rows_as_numeric(read_batch1_rows(telemetry))
    schema = read_schema(telemetry)
    events = read_events(run_dir / "events.log")
    end_reasons = segment_end_reasons(events)
    stats = phase_stats(rows, end_reasons)
    breakaway = detect_breakaway_intervals(stats)
    breakaway_summaries = {
        "left": summarize_breakaway_diagnostics(stats, "left"),
        "right": summarize_breakaway_diagnostics(stats, "right"),
    }
    measurements = load_measurements(measurements_path)
    phys = physical_inputs(measurements)
    cad_axial_inertias = cad_axial_inertias_from_robot_xml(robot_xml) if robot_xml else {}
    cad_axial_inertia_overrides = cad_axial_inertia_overrides or {}
    kt_metadata = kt_metadata or {}

    warnings: list[str] = []
    duration = ((rows[-1]["t_us"] - rows[0]["t_us"]) / 1_000_000.0) if len(rows) >= 2 else 0.0
    if len(rows) >= 2:
        dts = np.diff(np.array([r["t_us"] for r in rows], dtype=float)) / 1000.0
    else:
        dts = np.array([])
    faults = [
        r
        for r in rows
        if int(r.get("left_fault", 0)) != 0
        or int(r.get("right_fault", 0)) != 0
        or str(r.get("fault_reason", "none")) not in ("", "none")
    ]

    derived: dict[str, Any] = {
        "schema_version": 1,
        "source": "finn_mcu_batch1_sysid",
        "run": {
            "path": str(run_dir),
            "telemetry_schema": schema,
            "rows": len(rows),
            "duration_s": _finite_or_none(duration, 6),
            "phase_count": len({r.get("phase") for r in rows}),
            "event_count": len(events),
        },
        "sampling": {
            "dt_ms_mean": _finite_or_none(float(np.mean(dts)) if len(dts) else None, 6),
            "dt_ms_min": _finite_or_none(float(np.min(dts)) if len(dts) else None, 6),
            "dt_ms_max": _finite_or_none(float(np.max(dts)) if len(dts) else None, 6),
        },
        "health": {
            "fault_rows": len(faults),
            "imu_fresh_rows": int(sum(1 for r in rows if int(r.get("imu_ok", 0)) == 1)),
            "imu_rows": len(rows),
        },
        "input_models": {
            "robot_xml": portable_path(robot_xml),
            "measurements": portable_path(measurements_path),
            "physical": phys,
        },
        "wheels": {},
        "warnings": warnings,
        "not_identified": [
            "tire_ground_slide_friction",
            "tire_ground_torsional_friction",
            "tire_ground_rolling_friction",
            "contact_solref",
            "contact_solimp",
            "loaded_wheel_radius_m",
            "wheel_mass_kg",
            "full_robot_inertia",
        ],
    }

    for side in ("left", "right"):
        gradient_crosscheck = fit_wheel_dynamics(rows, side)
        coast_fit = fit_coastdown_output_error(rows, side)
        cad_inertia = cad_axial_inertia_overrides.get(side)
        if cad_inertia is None:
            cad_inertia = cad_axial_inertias.get(side)
        powered_fit = fit_powered_inertia_output_error(
            rows,
            side,
            coast_fit.damping_per_inertia,
            coast_fit.friction_per_inertia,
            initial_inertia_kg_m2=cad_inertia,
        )
        intervals = breakaway[side]
        breakaway_summary = breakaway_summaries[side]
        powered_inertia_usable = (
            powered_fit.inertia_kg_m2 is not None
            and powered_fit.confidence == "measured"
            and not powered_fit.bound_active
        )
        inertia_for_losses = powered_fit.inertia_kg_m2 if powered_inertia_usable else cad_inertia
        if not powered_inertia_usable and cad_inertia is not None:
            warnings.append(f"{side}_powered_inertia_weak_using_cad_inertia_for_losses")

        if (
            coast_fit.damping_per_inertia is not None
            and coast_fit.friction_per_inertia is not None
            and inertia_for_losses is not None
        ):
            damping = coast_fit.damping_per_inertia * inertia_for_losses
            friction = coast_fit.friction_per_inertia * inertia_for_losses
            friction_source = "coastdown"
            friction_confidence = coast_fit.confidence
            damping_confidence = coast_fit.confidence
        else:
            damping = gradient_crosscheck.damping
            friction = breakaway_summary["friction_floor_nm"]
            friction_source = "staircase_fallback"
            friction_confidence = "provisional" if friction is not None else "insufficient"
            damping_confidence = gradient_crosscheck.confidence

        fitted_total_inertia = powered_fit.inertia_kg_m2
        inertia_for_armature = fitted_total_inertia if powered_inertia_usable else cad_inertia
        if inertia_for_armature is not None and cad_inertia is not None and cad_inertia > 0.0:
            armature_raw = inertia_for_armature - cad_inertia
            armature = max(0.0, armature_raw)
            if fitted_total_inertia is not None:
                armature_vs_cad_pct = 100.0 * (fitted_total_inertia - cad_inertia) / cad_inertia
            else:
                armature_vs_cad_pct = 0.0
            if armature_vs_cad_pct is not None and abs(armature_vs_cad_pct) > 15.0:
                warnings.append(f"{side}_fitted_total_inertia_differs_from_cad_gt_15pct")
        elif inertia_for_armature is not None:
            armature = inertia_for_armature
            armature_vs_cad_pct = None
            warnings.append(f"{side}_cad_axial_inertia_unavailable_armature_not_subtracted")
        else:
            armature = None
            armature_vs_cad_pct = None
        for direction, interval in intervals.items():
            if interval.anomalous:
                warnings.append(f"{side}_{direction}_no_motion_at_highest_tested_breakaway_torque")

        commands = [abs(float(r[f"{side}_cmd_nm"])) for r in rows]
        voltages = [float(r[f"{side}_voltage_v"]) for r in rows]
        temps = [float(r[f"{side}_temp_c"]) for r in rows]
        velocities = [float(r[f"{side}_vel_rev_s"]) for r in rows]
        torques = [float(r[f"{side}_torque_nm"]) for r in rows]
        derived["wheels"][side] = {
            "suggested_sim": {
                "command_sign": {
                    "value": command_signs[side],
                    "unit": "sign",
                    "source": "operator_video_default",
                    "confidence": "provisional",
                },
                "torque_limit_nm": {
                    "value": _finite_or_none(hard_torque_limit_nm, 6),
                    "unit": "N*m",
                    "source": "firmware_hard_cap",
                    "confidence": "measured",
                },
                "validated_torque_nm": {
                    "value": _finite_or_none(max(commands) if commands else 0.0, 6),
                    "unit": "N*m",
                    "source": "max_abs_command_in_log",
                    "confidence": "measured",
                },
                "frictionloss": {
                    "value": _finite_or_none(friction, 6),
                    "unit": "N*m",
                    "source": friction_source,
                    "confidence": friction_confidence,
                },
                "friction_source": {
                    "value": friction_source,
                    "unit": "source",
                    "source": "postprocess_fit_selection",
                    "confidence": friction_confidence,
                },
                "damping": {
                    "value": _finite_or_none(damping, 8),
                    "unit": "N*m*s/rad",
                    "source": (
                        "coastdown_output_error"
                        if friction_source == "coastdown"
                        else "gradient_crosscheck"
                    ),
                    "confidence": damping_confidence,
                },
                "armature": {
                    "value": _finite_or_none(armature, 8),
                    "unit": "kg*m^2",
                    "source": "powered_output_error_minus_cad_axial_inertia",
                    "confidence": powered_fit.confidence,
                },
                "armature_vs_cad_pct": {
                    "value": _finite_or_none(armature_vs_cad_pct, 3),
                    "unit": "%",
                    "source": "powered_output_error_vs_robot_xml_inertial",
                    "confidence": powered_fit.confidence,
                },
                "bound_active": {
                    "frictionloss": coast_fit.bound_active.get("frictionloss", False)
                    if friction_source == "coastdown"
                    else False,
                    "damping": coast_fit.bound_active.get("damping", False)
                    if friction_source == "coastdown"
                    else False,
                    "armature": bool(
                        powered_fit.bound_active
                        or (
                            fitted_total_inertia is not None
                            and cad_inertia is not None
                            and (not powered_inertia_usable or fitted_total_inertia <= cad_inertia)
                        )
                    ),
                },
            },
            "diagnostics": {
                "breakaway": {
                    **{
                        direction: {
                            "low_no_motion_nm": _finite_or_none(interval.low_nm, 6),
                            "high_motion_nm": _finite_or_none(interval.high_nm, 6),
                            "estimate_nm": _finite_or_none(interval.estimate_nm, 6),
                            "sample_count": interval.samples,
                            "anomalous": interval.anomalous,
                            "notes": interval.notes,
                        }
                        for direction, interval in intervals.items()
                    },
                    **breakaway_summary,
                },
                "cad_axial_wheel_inertia_kg_m2": _finite_or_none(cad_inertia, 10),
                "fitted_total_wheel_inertia_kg_m2": _finite_or_none(fitted_total_inertia, 10),
                "coastdown_output_error_fit": {
                    "damping_per_inertia": _finite_or_none(coast_fit.damping_per_inertia, 8),
                    "friction_per_inertia": _finite_or_none(coast_fit.friction_per_inertia, 8),
                    "sample_count": coast_fit.sample_count,
                    "rmse_rad_s": _finite_or_none(coast_fit.rmse_rad_s, 6),
                    "confidence": coast_fit.confidence,
                    "bound_active": coast_fit.bound_active,
                    "notes": coast_fit.notes,
                },
                "powered_output_error_fit": {
                    "inertia_kg_m2": _finite_or_none(powered_fit.inertia_kg_m2, 10),
                    "sample_count": powered_fit.sample_count,
                    "rmse_rad_s": _finite_or_none(powered_fit.rmse_rad_s, 6),
                    "confidence": powered_fit.confidence,
                    "bound_active": powered_fit.bound_active,
                    "torque_signal_source": "moteus_measured_torque_nm",
                    "kt_nm_per_amp": _finite_or_none(kt_metadata.get(side), 8),
                    "notes": powered_fit.notes,
                },
                "gradient_crosscheck": {
                    "armature": _finite_or_none(gradient_crosscheck.armature, 8),
                    "damping": _finite_or_none(gradient_crosscheck.damping, 8),
                    "frictionloss": _finite_or_none(gradient_crosscheck.frictionloss, 8),
                    "sample_count": gradient_crosscheck.sample_count,
                    "rmse_nm": _finite_or_none(gradient_crosscheck.rmse_nm, 6),
                    "confidence": gradient_crosscheck.confidence,
                    "notes": gradient_crosscheck.notes,
                },
                "dynamic_fit": {
                    "sample_count": gradient_crosscheck.sample_count,
                    "rmse_nm": _finite_or_none(gradient_crosscheck.rmse_nm, 6),
                    "notes": ["legacy_alias_for_gradient_crosscheck", *gradient_crosscheck.notes],
                },
                "actuator_tracking": actuator_fit(rows, side),
                "response_delay": response_delay_s(rows, side),
                "velocity_noise": velocity_noise(rows, side),
                "ranges": {
                    "voltage_v": numeric_range(voltages, 4),
                    "temp_c": numeric_range(temps, 4),
                    "velocity_rev_s": numeric_range(velocities, 6),
                    "measured_torque_nm": numeric_range(torques, 6),
                },
            },
        }

    if faults:
        warnings.append("fault_rows_present")
    if not rows:
        warnings.append("no_telemetry_rows")
    return derived


def write_derived_yaml(out_path: Path, data: dict[str, Any]) -> None:
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_report(path: Path, derived: dict[str, Any]) -> None:
    lines = [
        "# Batch 1 sysid derived sim data",
        "",
        f"Run: `{derived['run']['path']}`",
        f"Telemetry schema: `{derived['run']['telemetry_schema']}`",
        f"Rows: {derived['run']['rows']}",
        f"Duration: {derived['run']['duration_s']} s",
        "",
        "## Suggested MuJoCo wheel values",
        "",
        "| Wheel | command_sign | torque_limit_nm | validated_torque_nm | "
        "frictionloss | damping | armature |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for side, item in derived["wheels"].items():
        sim = item["suggested_sim"]
        lines.append(
            (
                "| {side} | {sign} | {limit} | {validated} | {friction} | {damping} | {armature} |"
            ).format(
                side=side,
                sign=sim["command_sign"]["value"],
                limit=sim["torque_limit_nm"]["value"],
                validated=sim["validated_torque_nm"]["value"],
                friction=sim["frictionloss"]["value"],
                damping=sim["damping"]["value"],
                armature=sim["armature"]["value"],
            )
        )
    physical = derived.get("input_models", {}).get("physical", {})
    lines.extend(
        [
            "",
            "## Physical inputs",
            "",
            f"- measurements: `{derived.get('input_models', {}).get('measurements')}`",
            f"- robot_mass_kg: `{physical.get('robot_mass_kg')}`",
            f"- loaded_wheel_radius_m: `{physical.get('loaded_wheel_radius_m')}`",
            f"- wheel_track_width_m: `{physical.get('wheel_track_width_m')}`",
            f"- com_height_m: `{physical.get('com_height_m')}`",
            f"- com_fore_aft_m: `{physical.get('com_fore_aft_m')}`",
        ]
    )
    lines.extend(["", "## Warnings", ""])
    warnings = derived.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Fit provenance", ""])
    for side, item in derived["wheels"].items():
        sim = item["suggested_sim"]
        diagnostics = item["diagnostics"]
        breakaway = diagnostics["breakaway"]
        lines.extend(
            [
                f"### {side}",
                "",
                f"- friction source: `{sim['friction_source']['value']}`",
                f"- CAD axial inertia: `{diagnostics['cad_axial_wheel_inertia_kg_m2']}` kg*m^2",
                "- fitted total wheel inertia: "
                f"`{diagnostics['fitted_total_wheel_inertia_kg_m2']}` kg*m^2",
                f"- armature vs CAD: `{sim['armature_vs_cad_pct']['value']}` %",
                f"- cogging amplitude: `{breakaway['cogging_amplitude_nm']}` N*m",
                f"- breakaway scatter: `{breakaway['breakaway_scatter_nm']}` N*m",
                "",
            ]
        )
    lines.extend(["", "## Not identified by this off-ground test", ""])
    lines.extend(f"- {item}" for item in derived["not_identified"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hard-torque-limit-nm", type=float, default=0.25)
    parser.add_argument("--left-command-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--right-command-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--left-cad-axial-inertia-kg-m2", type=float)
    parser.add_argument("--right-cad-axial-inertia-kg-m2", type=float)
    parser.add_argument("--left-kt-nm-per-amp", type=float)
    parser.add_argument("--right-kt-nm-per-amp", type=float)
    args = parser.parse_args()

    out_dir = args.run_dir / "postprocess"
    out_dir.mkdir(exist_ok=True)
    report = out_dir / "report.md"
    derived = out_dir / "derived.yaml"
    data = analyze_run(
        args.run_dir,
        hard_torque_limit_nm=args.hard_torque_limit_nm,
        command_signs={"left": args.left_command_sign, "right": args.right_command_sign},
        robot_xml=args.robot_xml,
        measurements_path=args.measurements,
        cad_axial_inertia_overrides={
            "left": args.left_cad_axial_inertia_kg_m2,
            "right": args.right_cad_axial_inertia_kg_m2,
        },
        kt_metadata={
            "left": args.left_kt_nm_per_amp,
            "right": args.right_kt_nm_per_amp,
        },
    )
    write_derived_yaml(derived, data)
    write_report(report, data)
    print(f"Wrote {report} and {derived} ({data['run']['rows']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
