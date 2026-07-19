#!/usr/bin/env python3
"""Run a first robot-shaped LQR controller on the generated Finn MuJoCo model.

This script is intentionally written as both a runnable validation tool and a
teaching artifact.  The controller avoids MuJoCo-only state shortcuts in the
feedback law: it estimates the same kind of state the robot can estimate from
IMU attitude/rate plus sign-corrected wheel odometry, then maps a single
balance torque into equal left/right wheel torque commands.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "sim/generated/seeded/latest/finn.seeded.sim.xml"
DEFAULT_OUT_ROOT = REPO_ROOT / "logs/lqr_sim"


def portable_path(path: Path) -> str:
    """Prefer portable repository-relative paths in generated reports."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


# Keep this order fixed.  The LQR gain columns are only understandable if the
# state vector has one canonical order everywhere in the script.
#
# This first controller deliberately leaves wheel position out of the LQR state.
# With only the common balance torque channel, position is an outer-loop problem;
# including it here creates an uncontrollable integrator in the generated model.
STATE_NAMES = ("pitch_rad", "pitch_rate_rad_s", "forward_vel_m_s")


class LqrSimError(Exception):
    """Expected failure with a concise user-facing message."""


@dataclass(frozen=True)
class SimConfig:
    """Simulation and controller timing.

    The robot will eventually run this logic at a fixed control period.  MuJoCo
    can integrate faster than that, so each controller tick advances several
    internal MuJoCo steps while holding the torque command constant.
    """

    control_dt_s: float
    duration_s: float
    initial_pitch_rad: float
    target_pitch_rad: float
    target_forward_vel_m_s: float
    position_hold_kp_s: float
    max_position_correction_m_s: float
    linearization_torque_eps_nm: float
    fall_pitch_rad: float
    pitch_axis: int
    pitch_sign: float
    forward_sign: float


@dataclass(frozen=True)
class ModelHandles:
    """Compiled MuJoCo addresses for names this controller depends on."""

    left_actuator_id: int
    right_actuator_id: int
    left_wheel_qposadr: int
    right_wheel_qposadr: int
    left_wheel_dofadr: int
    right_wheel_dofadr: int
    imu_quat_adr: int
    imu_quat_dim: int
    imu_gyro_adr: int
    imu_gyro_dim: int
    wheel_left_pos_adr: int
    wheel_left_vel_adr: int
    wheel_right_pos_adr: int
    wheel_right_vel_adr: int
    wheel_radius_m: float
    torque_limit_nm: float


@dataclass
class StateEstimator:
    """Robot-shaped estimator used by the LQR loop.

    In sim we could read perfect root position and body angular velocity from
    qpos/qvel.  Do not do that in the controller.  The real robot will not have
    MuJoCo qpos; it will have an IMU and motor encoders.  This estimator mirrors
    that boundary:

    - pitch comes from the calibrated IMU quaternion;
    - pitch rate comes from the IMU gyro;
    - forward velocity comes from sign-corrected wheel odometry.
    """

    model: mujoco.MjModel
    handles: ModelHandles
    config: SimConfig
    neutral_imu_quat: np.ndarray | None = None
    neutral_forward_pos_m: float = 0.0

    def calibrate(self, data: mujoco.MjData) -> None:
        """Capture the upright reference, like zeroing the robot on boot."""

        self.neutral_imu_quat = self.imu_quat(data)
        self.neutral_forward_pos_m = self.forward_pos_m(data)

    def state(self, data: mujoco.MjData) -> np.ndarray:
        """Return x = [pitch, pitch_rate, forward_vel]."""

        if self.neutral_imu_quat is None:
            raise LqrSimError("StateEstimator.calibrate() must be called before state().")

        current_quat = self.imu_quat(data)

        # MuJoCo framequat is a world orientation quaternion in wxyz order.  We
        # subtract the calibrated upright orientation in the IMU/site frame:
        # relative = inverse(neutral) * current.
        #
        # For Finn's current IMU site, this makes robot pitch land on local IMU
        # axis X, matching the firmware convention from Batch 2 telemetry.
        relative_quat = quat_mul(quat_conj(self.neutral_imu_quat), current_quat)
        pitch_rotvec = quat_to_rotvec(relative_quat)
        pitch_rad = self.config.pitch_sign * pitch_rotvec[self.config.pitch_axis]

        gyro = self.imu_gyro(data)
        pitch_rate_rad_s = self.config.pitch_sign * gyro[self.config.pitch_axis]

        forward_vel_m_s = self.forward_vel_m_s(data)

        return np.array(
            [pitch_rad, pitch_rate_rad_s, forward_vel_m_s],
            dtype=float,
        )

    def relative_forward_pos_m(self, data: mujoco.MjData) -> float:
        return self.forward_pos_m(data) - self.neutral_forward_pos_m

    def imu_quat(self, data: mujoco.MjData) -> np.ndarray:
        adr = self.handles.imu_quat_adr
        dim = self.handles.imu_quat_dim
        return np.array(data.sensordata[adr : adr + dim], dtype=float)

    def imu_gyro(self, data: mujoco.MjData) -> np.ndarray:
        adr = self.handles.imu_gyro_adr
        dim = self.handles.imu_gyro_dim
        return np.array(data.sensordata[adr : adr + dim], dtype=float)

    def forward_pos_m(self, data: mujoco.MjData) -> float:
        left = float(data.sensordata[self.handles.wheel_left_pos_adr])
        right = float(data.sensordata[self.handles.wheel_right_pos_adr])
        return (
            self.config.forward_sign
            * signed_forward_wheel_rad(left, right)
            * (self.handles.wheel_radius_m)
        )

    def forward_vel_m_s(self, data: mujoco.MjData) -> float:
        left = float(data.sensordata[self.handles.wheel_left_vel_adr])
        right = float(data.sensordata[self.handles.wheel_right_vel_adr])
        return (
            self.config.forward_sign
            * signed_forward_wheel_rad(left, right)
            * (self.handles.wheel_radius_m)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design and validate a first LQR balance controller in Finn MuJoCo sim."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--control-dt-s", type=float, default=0.01)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--initial-pitch-rad", type=float, default=0.03)
    parser.add_argument(
        "--target-pitch-rad",
        type=float,
        help=(
            "Pitch setpoint in radians. By default it is derived from the seeded "
            "model's axle-to-COM offset so zero forward velocity is a valid balance point."
        ),
    )
    parser.add_argument("--target-forward-vel-m-s", type=float, default=0.0)
    parser.add_argument(
        "--position-hold-kp-s",
        type=float,
        default=0.15,
        help=(
            "Outer-loop position gain in 1/s. It adjusts the LQR velocity target "
            "to return toward the commanded path; set to 0 to disable position hold."
        ),
    )
    parser.add_argument(
        "--max-position-correction-m-s",
        type=float,
        default=0.15,
        help="Maximum velocity correction contributed by the outer position loop.",
    )
    parser.add_argument("--fall-pitch-rad", type=float, default=0.45)
    parser.add_argument(
        "--linearization-torque-eps-nm",
        type=float,
        default=0.18,
        help=(
            "Finite common-wheel torque used to identify B.  This is intentionally "
            "above tiny epsilon because the seeded model includes dry wheel friction."
        ),
    )
    parser.add_argument(
        "--pitch-axis",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="IMU local axis used as robot pitch. Finn's Batch 2 convention is X=0.",
    )
    parser.add_argument("--pitch-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument("--forward-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    parser.add_argument(
        "--q-diag",
        type=float,
        nargs=3,
        default=(160.0, 16.0, 2.0),
        metavar=("PITCH", "PITCH_RATE", "VEL"),
        help="LQR state penalties in STATE_NAMES order.",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.6,
        help="LQR common torque penalty. Larger values make the controller gentler.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "Open a live MuJoCo viewer and pace the closed-loop rollout in real time. "
            "On macOS, launch this script with mjpython."
        ),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except LqrSimError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"wrote LQR validation bundle: {result['out_dir']}")
    print(f"status: {result['status']}")
    lqr_result = result["lqr"]
    if not isinstance(lqr_result, dict):
        raise LqrSimError("internal result is missing lqr metadata")
    print(f"gain: {np.array(lqr_result['gain']).round(5).tolist()}")
    print(
        "metrics: "
        f"max_abs_pitch_rad={result['metrics']['max_abs_pitch_rad']:.5f}, "
        f"final_abs_pitch_rad={result['metrics']['final_abs_pitch_rad']:.5f}, "
        f"final_forward_pos_m={result['metrics']['final_forward_pos_m']:.5f}, "
        f"saturation_fraction={result['metrics']['saturation_fraction']:.3f}"
    )
    return 0 if result["status"] == "pass" else 1


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.model.exists():
        raise LqrSimError(f"missing model XML: {args.model}")

    out_dir = args.out_dir or DEFAULT_OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    handles = inspect_model(model)
    target_pitch_rad = (
        estimate_balance_trim_pitch_rad(model)
        if args.target_pitch_rad is None
        else float(args.target_pitch_rad)
    )

    config = SimConfig(
        control_dt_s=args.control_dt_s,
        duration_s=args.duration_s,
        initial_pitch_rad=args.initial_pitch_rad,
        target_pitch_rad=target_pitch_rad,
        target_forward_vel_m_s=args.target_forward_vel_m_s,
        position_hold_kp_s=args.position_hold_kp_s,
        max_position_correction_m_s=args.max_position_correction_m_s,
        linearization_torque_eps_nm=args.linearization_torque_eps_nm,
        fall_pitch_rad=args.fall_pitch_rad,
        pitch_axis=args.pitch_axis,
        pitch_sign=args.pitch_sign,
        forward_sign=args.forward_sign,
    )

    validate_timing(model, config)
    validate_linearization_torque(config, handles)

    estimator = calibrated_estimator(model, handles, config)

    # Linearization produces the discrete-time model:
    #
    #   x[k+1] = A x[k] + B u[k]
    #
    # where u is one scalar: the common balance torque sent equally to both
    # wheel motors.  This is the same mixing shape you would put on the robot:
    # balance torque first, yaw/differential torque later.
    a_matrix, b_matrix = linearize_balance_dynamics(model, handles, estimator, config)
    q_cost = np.diag(np.array(args.q_diag, dtype=float))
    r_cost = np.array([[float(args.r)]], dtype=float)
    gain = discrete_lqr(a_matrix, b_matrix, q_cost, r_cost)
    closed_loop_eigs = np.linalg.eigvals(a_matrix - b_matrix @ gain)

    rows, metrics = run_closed_loop(
        model,
        handles,
        estimator,
        config,
        gain,
        show_viewer=args.viewer,
    )
    status = "pass" if metrics["pass"] else "failed"

    csv_path = out_dir / "timeseries.csv"
    report_path = out_dir / "report.json"
    write_timeseries(csv_path, rows)

    result: dict[str, object] = {
        "status": status,
        "out_dir": portable_path(out_dir),
        "model": portable_path(args.model),
        "state_names": list(STATE_NAMES),
        "control": {
            "name": "common_balance_torque_nm",
            "mapping": {
                "motor_left_wheel": "tau_balance",
                "motor_right_wheel": "tau_balance",
            },
            "torque_limit_nm": handles.torque_limit_nm,
            "control_dt_s": config.control_dt_s,
            "target_pitch_rad": config.target_pitch_rad,
            "target_forward_vel_m_s": config.target_forward_vel_m_s,
            "position_hold_kp_s": config.position_hold_kp_s,
            "max_position_correction_m_s": config.max_position_correction_m_s,
        },
        "linearization": {
            "torque_eps_nm": config.linearization_torque_eps_nm,
            "a_matrix": a_matrix.tolist(),
            "b_matrix": b_matrix.tolist(),
            "closed_loop_eigs": [
                {"real": float(value.real), "imag": float(value.imag)} for value in closed_loop_eigs
            ],
        },
        "lqr": {
            "q_diag": list(args.q_diag),
            "r": float(args.r),
            "gain": gain.tolist(),
        },
        "metrics": metrics,
        "notes": [
            (
                "The inner LQR regulates pitch, pitch rate, and forward velocity. "
                "A slower outer loop adjusts the velocity target to hold wheel position."
            ),
            (
                "A marginal closed-loop eigenvalue near 1.0 is expected here because "
                "absolute wheel position remains an outer-loop state."
            ),
        ],
        "artifacts": {
            "timeseries_csv": portable_path(csv_path),
            "report_json": portable_path(report_path),
        },
        "scope": (
            "Sim-only LQR validation using robot-shaped IMU and wheel-odometry state. "
            "This does not prove sim-to-real transfer without Batch 3 world-pose replay."
        ),
    }

    if not args.no_plot:
        plot_path = out_dir / "timeseries.png"
        if write_plot(plot_path, rows):
            result["artifacts"]["timeseries_png"] = portable_path(plot_path)  # type: ignore[index]

    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def inspect_model(model: mujoco.MjModel) -> ModelHandles:
    """Resolve names once so later code fails loudly on model/config drift."""

    left_actuator_id = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_left_wheel")
    right_actuator_id = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right_wheel")

    left_joint_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel")
    right_joint_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel")

    imu_quat_adr, imu_quat_dim = sensor_adr_dim(model, "imu_quat")
    imu_gyro_adr, imu_gyro_dim = sensor_adr_dim(model, "imu_gyro")
    wheel_left_pos_adr, _ = sensor_adr_dim(model, "wheel_left_pos")
    wheel_left_vel_adr, _ = sensor_adr_dim(model, "wheel_left_vel")
    wheel_right_pos_adr, _ = sensor_adr_dim(model, "wheel_right_pos")
    wheel_right_vel_adr, _ = sensor_adr_dim(model, "wheel_right_vel")

    left_range = model.actuator_ctrlrange[left_actuator_id]
    right_range = model.actuator_ctrlrange[right_actuator_id]
    if not np.allclose(left_range, right_range):
        raise LqrSimError(
            f"left/right actuator ranges differ: left={left_range}, right={right_range}"
        )
    if not math.isclose(abs(float(left_range[0])), abs(float(left_range[1])), rel_tol=1e-6):
        raise LqrSimError(f"expected symmetric actuator range, got {left_range}")

    return ModelHandles(
        left_actuator_id=left_actuator_id,
        right_actuator_id=right_actuator_id,
        left_wheel_qposadr=int(model.jnt_qposadr[left_joint_id]),
        right_wheel_qposadr=int(model.jnt_qposadr[right_joint_id]),
        left_wheel_dofadr=int(model.jnt_dofadr[left_joint_id]),
        right_wheel_dofadr=int(model.jnt_dofadr[right_joint_id]),
        imu_quat_adr=imu_quat_adr,
        imu_quat_dim=imu_quat_dim,
        imu_gyro_adr=imu_gyro_adr,
        imu_gyro_dim=imu_gyro_dim,
        wheel_left_pos_adr=wheel_left_pos_adr,
        wheel_left_vel_adr=wheel_left_vel_adr,
        wheel_right_pos_adr=wheel_right_pos_adr,
        wheel_right_vel_adr=wheel_right_vel_adr,
        wheel_radius_m=wheel_radius(model),
        torque_limit_nm=abs(float(left_range[1])),
    )


def estimate_balance_trim_pitch_rad(model: mujoco.MjModel) -> float:
    """Return the pitch that places the whole-robot COM above the wheel axle.

    Finn's CAD-derived COM is slightly behind the axle at zero pitch. Asking the
    velocity LQR to hold zero pitch therefore forces the wheels to keep moving
    underneath that offset COM. Rotating the axle-to-COM vector until its
    forward component is zero gives the stationary balance trim.
    """

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    left_site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_wheel_center")
    right_site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_wheel_center")

    axle_pos = 0.5 * (data.site_xpos[left_site_id] + data.site_xpos[right_site_id])
    axle_to_com = data.subtree_com[base_body_id] - axle_pos
    vertical_offset_m = float(axle_to_com[2])
    if vertical_offset_m <= 0.0:
        raise LqrSimError("cannot derive balance trim: whole-robot COM is not above the wheel axle")

    return math.atan2(-float(axle_to_com[0]), vertical_offset_m)


def require_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise LqrSimError(f"model is missing expected MuJoCo object: {name}")
    return int(obj_id)


def sensor_adr_dim(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    sensor_id = require_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id])


def wheel_radius(model: mujoco.MjModel) -> float:
    geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_tire_collision")
    return float(model.geom_size[geom_id][0])


def validate_timing(model: mujoco.MjModel, config: SimConfig) -> None:
    if config.control_dt_s <= 0.0:
        raise LqrSimError("--control-dt-s must be positive")
    if config.duration_s <= 0.0:
        raise LqrSimError("--duration-s must be positive")
    if config.position_hold_kp_s < 0.0:
        raise LqrSimError("--position-hold-kp-s must be nonnegative")
    if config.max_position_correction_m_s < 0.0:
        raise LqrSimError("--max-position-correction-m-s must be nonnegative")
    steps = config.control_dt_s / float(model.opt.timestep)
    if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
        raise LqrSimError(
            f"control_dt_s={config.control_dt_s} is not an integer multiple of "
            f"MuJoCo timestep={model.opt.timestep}"
        )


def validate_linearization_torque(config: SimConfig, handles: ModelHandles) -> None:
    eps = abs(config.linearization_torque_eps_nm)
    if eps <= 0.0:
        raise LqrSimError("--linearization-torque-eps-nm must be positive")
    if eps > handles.torque_limit_nm:
        raise LqrSimError(
            f"linearization torque {eps} exceeds actuator limit {handles.torque_limit_nm}"
        )


def calibrated_estimator(
    model: mujoco.MjModel, handles: ModelHandles, config: SimConfig
) -> StateEstimator:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    estimator = StateEstimator(model=model, handles=handles, config=config)
    estimator.calibrate(data)
    return estimator


def linearize_balance_dynamics(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference the one-tick dynamics around upright.

    A textbook LQR starts from analytical equations.  For Finn's generated model,
    the safer first pass is to ask MuJoCo for the local dynamics of the exact XML
    you will simulate.  The perturbations below are deliberately small for state
    and finite for torque because the seeded wheel joints include dry friction.
    """

    state_eps = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    zero_state = np.zeros(len(STATE_NAMES), dtype=float)

    a_matrix = np.zeros((len(STATE_NAMES), len(STATE_NAMES)), dtype=float)
    for column, eps in enumerate(state_eps):
        plus = zero_state.copy()
        minus = zero_state.copy()
        plus[column] = eps
        minus[column] = -eps
        f_plus = simulate_one_control_tick(model, handles, estimator, config, plus, 0.0)
        f_minus = simulate_one_control_tick(model, handles, estimator, config, minus, 0.0)
        a_matrix[:, column] = (f_plus - f_minus) / (2.0 * eps)

    torque_eps = config.linearization_torque_eps_nm
    f_plus = simulate_one_control_tick(model, handles, estimator, config, zero_state, torque_eps)
    f_minus = simulate_one_control_tick(model, handles, estimator, config, zero_state, -torque_eps)
    b_matrix = ((f_plus - f_minus) / (2.0 * torque_eps)).reshape(-1, 1)

    if np.linalg.norm(b_matrix) < 1e-9:
        raise LqrSimError(
            "linearized input matrix is near zero; increase --linearization-torque-eps-nm "
            "or revisit wheel friction/contact parameters"
        )

    return a_matrix, b_matrix


def simulate_one_control_tick(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
    initial_state: np.ndarray,
    tau_balance_nm: float,
) -> np.ndarray:
    data = mujoco.MjData(model)
    set_reduced_state(model, data, handles, config, initial_state)
    apply_balance_torque(data, handles, tau_balance_nm)
    step_control_tick(model, data, config)
    return estimator.state(data)


def set_reduced_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    config: SimConfig,
    state: np.ndarray,
) -> None:
    """Set a robot-readable initial condition.

    This function is allowed to touch MuJoCo qpos/qvel because it prepares a
    simulation experiment.  The controller itself never sees these internals.
    """

    pitch_rad, pitch_rate_rad_s, forward_vel_m_s = state
    forward_pos_m = 0.0

    mujoco.mj_resetData(model, data)

    # The freejoint position keeps the chassis above the wheels.  We move x with
    # wheel odometry so the visual/root pose and encoder pose start consistent.
    data.qpos[0] = forward_pos_m

    # A world-Y rotation appears as local IMU X pitch after subtracting the
    # neutral IMU orientation.  This was verified against the generated model.
    data.qpos[3:7] = axis_angle_quat(axis=1, angle=float(pitch_rad))

    # Freejoint qvel order is [vx, vy, vz, wx, wy, wz].  World-Y angular velocity
    # maps to the current IMU pitch-rate axis for Finn's mounted IMU frame.
    data.qvel[0] = forward_vel_m_s
    data.qvel[4] = pitch_rate_rad_s

    wheel_rad = forward_pos_m / handles.wheel_radius_m
    wheel_rad_s = forward_vel_m_s / handles.wheel_radius_m

    # Sign-corrected forward convention: right wheel positive and left wheel
    # negative correspond to forward motion for the current generated model.
    data.qpos[handles.left_wheel_qposadr] = -wheel_rad
    data.qpos[handles.right_wheel_qposadr] = wheel_rad
    data.qvel[handles.left_wheel_dofadr] = -wheel_rad_s
    data.qvel[handles.right_wheel_dofadr] = wheel_rad_s

    mujoco.mj_forward(model, data)


def discrete_lqr(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    q_cost: np.ndarray,
    r_cost: np.ndarray,
) -> np.ndarray:
    """Solve u = -Kx for the discrete system x[k+1] = A x[k] + B u[k]."""

    p_matrix = solve_discrete_are(a_matrix, b_matrix, q_cost, r_cost)
    lhs = b_matrix.T @ p_matrix @ b_matrix + r_cost
    rhs = b_matrix.T @ p_matrix @ a_matrix
    return np.linalg.solve(lhs, rhs)


def run_closed_loop(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
    gain: np.ndarray,
    *,
    show_viewer: bool = False,
) -> tuple[list[dict[str, float]], dict[str, float | bool]]:
    data = mujoco.MjData(model)
    initial_state = np.array([config.initial_pitch_rad, 0.0, 0.0], dtype=float)
    set_reduced_state(model, data, handles, config, initial_state)

    rows: list[dict[str, float]] = []
    control_steps = round(config.duration_s / config.control_dt_s)
    saturated_count = 0
    finite = True
    fell = False
    stopped_by_viewer = False

    viewer_context = mujoco.viewer.launch_passive(model, data) if show_viewer else nullcontext(None)
    with viewer_context as viewer:
        wall_start_s = time.perf_counter()
        if viewer is not None:
            viewer.sync()

        for tick in range(control_steps + 1):
            if tick > 0 and viewer is not None and not viewer.is_running():
                stopped_by_viewer = True
                break

            time_s = tick * config.control_dt_s
            state = estimator.state(data)
            forward_pos_m = estimator.relative_forward_pos_m(data)
            target_forward_pos_m = config.target_forward_vel_m_s * time_s
            position_error_m = forward_pos_m - target_forward_pos_m
            position_velocity_correction_m_s = clamp(
                -config.position_hold_kp_s * position_error_m,
                -config.max_position_correction_m_s,
                config.max_position_correction_m_s,
            )
            effective_target_forward_vel_m_s = (
                config.target_forward_vel_m_s + position_velocity_correction_m_s
            )
            target_state = np.array(
                [config.target_pitch_rad, 0.0, effective_target_forward_vel_m_s],
                dtype=float,
            )
            error = state - target_state

            # This is the entire LQR controller you would port to the robot once the
            # estimator is real: read state, subtract target, multiply by K, clip to
            # the known torque envelope, then send torque commands.
            raw_tau = float((-gain @ error.reshape(-1, 1)).item())
            tau = clamp(raw_tau, -handles.torque_limit_nm, handles.torque_limit_nm)
            saturated = not math.isclose(raw_tau, tau, rel_tol=0.0, abs_tol=1e-12)
            saturated_count += int(saturated)

            rows.append(
                {
                    "time_s": time_s,
                    "pitch_rad": float(state[0]),
                    "pitch_rate_rad_s": float(state[1]),
                    "forward_pos_m": forward_pos_m,
                    "target_forward_pos_m": target_forward_pos_m,
                    "forward_vel_m_s": float(state[2]),
                    "target_forward_vel_m_s": effective_target_forward_vel_m_s,
                    "tau_balance_raw_nm": raw_tau,
                    "tau_balance_nm": tau,
                    "left_cmd_nm": tau,
                    "right_cmd_nm": tau,
                    "saturated": float(saturated),
                }
            )

            finite = finite and np.all(np.isfinite(state)) and math.isfinite(tau)
            fell = fell or abs(float(state[0])) > config.fall_pitch_rad
            if tick == control_steps or not finite or fell:
                break

            apply_balance_torque(data, handles, tau)
            step_control_tick(model, data, config)

            if viewer is not None:
                viewer.sync()
                target_wall_s = wall_start_s + (tick + 1) * config.control_dt_s
                remaining_s = target_wall_s - time.perf_counter()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)

    max_abs_pitch = max(abs(row["pitch_rad"]) for row in rows)
    final_abs_pitch = abs(rows[-1]["pitch_rad"])
    max_abs_position_error = max(
        abs(row["forward_pos_m"] - row["target_forward_pos_m"]) for row in rows
    )
    final_abs_position_error = abs(rows[-1]["forward_pos_m"] - rows[-1]["target_forward_pos_m"])
    saturation_fraction = saturated_count / max(1, len(rows))
    position_hold_passed = config.position_hold_kp_s == 0.0 or (
        max_abs_position_error < 0.25 and final_abs_position_error < 0.10
    )

    passed = (
        finite
        and not fell
        and final_abs_pitch < 0.08
        and max_abs_pitch < config.fall_pitch_rad
        and position_hold_passed
        and saturation_fraction < 0.80
    )
    return rows, {
        "pass": passed,
        "finite": finite,
        "fell": fell,
        "stopped_by_viewer": stopped_by_viewer,
        "max_abs_pitch_rad": max_abs_pitch,
        "final_abs_pitch_rad": final_abs_pitch,
        "final_forward_pos_m": rows[-1]["forward_pos_m"],
        "final_forward_vel_m_s": rows[-1]["forward_vel_m_s"],
        "max_abs_position_error_m": max_abs_position_error,
        "final_abs_position_error_m": final_abs_position_error,
        "saturation_fraction": saturation_fraction,
        "samples": len(rows),
    }


def apply_balance_torque(data: mujoco.MjData, handles: ModelHandles, tau_balance_nm: float) -> None:
    data.ctrl[:] = 0.0
    data.ctrl[handles.left_actuator_id] = tau_balance_nm
    data.ctrl[handles.right_actuator_id] = tau_balance_nm


def step_control_tick(model: mujoco.MjModel, data: mujoco.MjData, config: SimConfig) -> None:
    inner_steps = round(config.control_dt_s / float(model.opt.timestep))
    for _ in range(inner_steps):
        mujoco.mj_step(model, data)


def signed_forward_wheel_rad(left_rad: float, right_rad: float) -> float:
    """Average wheel rotation after converting each encoder to forward-positive."""

    return 0.5 * (right_rad - left_rad)


def axis_angle_quat(*, axis: int, angle: float) -> np.ndarray:
    half = 0.5 * angle
    quat = np.array([math.cos(half), 0.0, 0.0, 0.0], dtype=float)
    quat[axis + 1] = math.sin(half)
    return quat


def quat_conj(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    xyz = quat[1:4]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(xyz_norm, float(quat[0]))
    return xyz / xyz_norm * angle


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def write_timeseries(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, float]]) -> bool:
    try:
        import matplotlib

        # mjpython runs the simulation script on a worker thread on macOS. A
        # file-only plot must therefore use a non-interactive backend.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    time_s = [row["time_s"] for row in rows]
    pitch = [row["pitch_rad"] for row in rows]
    forward_pos = [row["forward_pos_m"] for row in rows]
    tau = [row["tau_balance_nm"] for row in rows]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(time_s, pitch)
    axes[0].set_ylabel("pitch rad")
    axes[1].plot(time_s, forward_pos)
    axes[1].plot(
        time_s,
        [row["target_forward_pos_m"] for row in rows],
        linestyle="--",
        label="target",
    )
    axes[1].set_ylabel("forward m")
    axes[1].legend()
    axes[2].plot(time_s, tau)
    axes[2].set_ylabel("torque Nm")
    axes[2].set_xlabel("time s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    sys.exit(main())
