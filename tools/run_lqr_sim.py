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
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.linalg import solve_discrete_are

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "sim/generated/seeded/latest/finn.seeded.sim.xml"
DEFAULT_OUT_ROOT = REPO_ROOT / "logs/lqr_sim"
DEFAULT_CONVENTIONS = REPO_ROOT / "config/finn_conventions.yaml"


def portable_path(path: Path) -> str:
    """Prefer portable repository-relative paths in generated reports."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# Keep this order fixed.  The LQR gain columns are only understandable if the
# state vector has one canonical order everywhere in the script.
#
# This first controller deliberately leaves wheel position out of the LQR state.
# With only the common balance torque channel, position is an outer-loop problem;
# including it here creates an uncontrollable integrator in the generated model.
STATE_NAMES = ("pitch_rad", "pitch_rate_rad_s", "forward_vel_m_s")

# The authored model rest pose leaves the tires a few millimetres above the floor.
# Every experiment below must start in ground contact instead: a robot in free fall
# feels no gravity torque about its COM, so pitch produces no pitch acceleration and
# the identified plant loses the inverted-pendulum mode entirely.  Settling with the
# base orientation held fixed lets the contact reach its equilibrium penetration
# without the chassis tipping away from the operating point we are linearizing about.
SETTLE_STEPS = 50

# Long enough for the yaw rate to rise well clear of the wheels' dry-friction band,
# short enough that the chassis has not pitched away from the operating point.
YAW_SPINUP_TICKS = 20

# How long a drive command must hold still before its tracking error is scored.
# Shorter than this and the measurement is reading the slew ramp, not the loop.
COMMAND_SETTLE_S = 1.5

# Below this many scored samples a tracking number is noise, not a verdict.
MIN_TRACKING_SAMPLES = 50

# How close to upright a station-keeping rollout has to end.
SETTLED_PITCH_RAD = 0.08
GRAVITY_M_S2 = 9.81


class LqrSimError(Exception):
    """Expected failure with a concise user-facing message."""


# The command layer. Nothing below may write a torque or stall a control tick;
# docs/codebase-notes.md states the invariants and why they are shaped this way.


@dataclass(frozen=True)
class DriveCommand:
    """The entire vocabulary a command source may speak."""

    forward_vel_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0

    def is_finite(self) -> bool:
        return math.isfinite(self.forward_vel_m_s) and math.isfinite(self.yaw_rate_rad_s)


# A command source sees the clock and nothing else.  It gets no MjData, no model
# handles, and no gains, which is what makes it unable to bypass the arbiter.
CommandSource = Callable[[float], DriveCommand]

STOPPED = DriveCommand()


@dataclass(frozen=True)
class DriveLimits:
    """Envelope every command is clamped and rate-limited into.

    The acceleration limits are a safety element, not smoothing.  Holding a
    forward acceleration `a` costs a steady lean of atan(a/g), and the reviewed
    firmware pitch fault trips at 10 degrees, so acceleration is physically capped
    at g*tan(10 deg) = 1.73 m/s^2 before Finn faults out.  The 0.5 m/s^2 default
    leans 2.9 degrees and leaves the rest of that budget for disturbance rejection.
    """

    max_forward_vel_m_s: float = 0.6
    # Measured knee: spinning in place, |forward_vel| p95 drift grows from 0.22 to
    # 0.31 m/s between 1.0 and 1.5 rad/s, which the balance loop then has to reject.
    max_yaw_rate_rad_s: float = 1.0
    forward_accel_limit_m_s2: float = 0.5
    yaw_accel_limit_rad_s2: float = 3.0
    command_timeout_s: float = 0.5
    reference_position_band_m: float = 0.30


@dataclass
class CommandArbiter:
    """Turn intermittent, untrusted intent into a bounded, continuous reference.

    Key release, a dropped serial link, a policy that stops publishing, and a
    policy that raises all arrive here as the same event -- no fresh command --
    and all four ramp to zero through the same slew limiter.  Ramping matters:
    stepping the reference to zero is itself a disturbance the balance loop would
    have to reject.
    """

    limits: DriveLimits
    shaped: DriveCommand = STOPPED
    last_good_s: float | None = None
    stale: bool = False
    rejected_samples: int = 0
    _warned: bool = False

    def sample(self, source: CommandSource | None, time_s: float) -> DriveCommand:
        """Read the source without ever letting it break the caller."""

        if source is None:
            return STOPPED
        try:
            requested = source(time_s)
        # A tenant fault must never reach the balance loop, so nothing escapes here.
        except Exception:
            requested = None
        if not isinstance(requested, DriveCommand) or not requested.is_finite():
            self.rejected_samples += 1
            if not self._warned:
                self._warned = True
                print(
                    "warning: command source returned an unusable value; holding then "
                    "ramping to zero",
                    file=sys.stderr,
                )
            return self._held(time_s)
        self.last_good_s = time_s
        self.stale = False
        return requested

    def step(self, source: CommandSource | None, time_s: float, dt_s: float) -> DriveCommand:
        target = self.sample(source, time_s)
        limits = self.limits
        forward_target = clamp(
            target.forward_vel_m_s, -limits.max_forward_vel_m_s, limits.max_forward_vel_m_s
        )
        yaw_target = clamp(
            target.yaw_rate_rad_s, -limits.max_yaw_rate_rad_s, limits.max_yaw_rate_rad_s
        )
        self.shaped = DriveCommand(
            slew(
                self.shaped.forward_vel_m_s, forward_target, limits.forward_accel_limit_m_s2 * dt_s
            ),
            slew(self.shaped.yaw_rate_rad_s, yaw_target, limits.yaw_accel_limit_rad_s2 * dt_s),
        )
        return self.shaped

    def _held(self, time_s: float) -> DriveCommand:
        """Hold the last accepted command briefly, then command a stop.

        The hold is what decouples a 100 Hz control tick from a policy publishing
        at 10 Hz with jitter: a missing sample is normal, a missing second is not.
        """

        if self.last_good_s is None:
            return STOPPED
        if time_s - self.last_good_s <= self.limits.command_timeout_s:
            return self.shaped
        self.stale = True
        return STOPPED


def allocate_wheel_torques(
    tau_balance_nm: float, tau_yaw_nm: float, limit_nm: float
) -> tuple[float, float]:
    """Divide the per-wheel torque envelope between balance and steering.

    Clamping each wheel after summing the two channels would let a turn request
    eat balance authority asymmetrically, which is a fall.  Yaw instead gets only
    the headroom balance left behind, so every wheel command lands inside the
    envelope by construction and a saturated balance loop steers not at all.
    """

    tau_common = clamp(tau_balance_nm, -limit_nm, limit_nm)
    headroom_nm = limit_nm - abs(tau_common)
    return tau_common, clamp(tau_yaw_nm, -headroom_nm, headroom_nm)


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
    linearization_vel_eps_m_s: float
    fall_pitch_rad: float
    pitch_axis: int
    pitch_sign: float
    forward_sign: float
    yaw_axis: int
    yaw_sign: float
    yaw_left_actuator_sign: float
    drive: DriveLimits


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
    left_tire_geom_id: int
    right_tire_geom_id: int
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

    def yaw_rate_rad_s(self, data: mujoco.MjData) -> float:
        """Yaw rate straight off the gyro, with no wheel-difference odometry.

        Differencing the wheels would need a track width, and the three available
        numbers disagree by 40 percent; config/finn_conventions.yaml records why.
        """

        return self.config.yaw_sign * float(self.imu_gyro(data)[self.config.yaw_axis])

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


def add_drive_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the command-layer envelope shared by this script and drive_lqr_sim.py."""

    defaults = DriveLimits()
    parser.add_argument(
        "--q-yaw",
        type=float,
        default=100.0,
        help="Yaw-rate tracking penalty for the separate one-state steering loop.",
    )
    parser.add_argument(
        "--r-yaw",
        type=float,
        default=1.0,
        help="Differential torque penalty. Larger values make steering gentler.",
    )
    parser.add_argument(
        "--yaw-left-actuator-sign",
        type=float,
        default=None,
        choices=(-1.0, 1.0),
        help="Overrides config/finn_conventions.yaml yaw.sim_left_actuator_yaw_sign.",
    )
    parser.add_argument("--max-forward-vel-m-s", type=float, default=defaults.max_forward_vel_m_s)
    parser.add_argument("--max-yaw-rate-rad-s", type=float, default=defaults.max_yaw_rate_rad_s)
    parser.add_argument(
        "--drive-accel-limit-m-s2",
        type=float,
        default=defaults.forward_accel_limit_m_s2,
        help=(
            "Forward acceleration slew limit. This is a safety bound, not smoothing: "
            "holding acceleration a costs a steady lean of atan(a/g), and the reviewed "
            "firmware pitch fault trips at 10 degrees."
        ),
    )
    parser.add_argument(
        "--drive-yaw-accel-limit-rad-s2",
        type=float,
        default=defaults.yaw_accel_limit_rad_s2,
    )
    parser.add_argument(
        "--command-timeout-s",
        type=float,
        default=defaults.command_timeout_s,
        help="How long a command is held before the arbiter ramps the drive to zero.",
    )
    parser.add_argument(
        "--reference-position-band-m",
        type=float,
        default=defaults.reference_position_band_m,
        help=(
            "Anti-windup band around measured position. Without it, a command the "
            "robot cannot follow winds the reference away and returns as a long unwind."
        ),
    )


def conventions_yaw_left_actuator_sign(conventions_path: Path) -> float:
    """Read which wheel actuator receives +tau_yaw for a positive (left) yaw."""

    conventions = yaml.safe_load(Path(conventions_path).read_text(encoding="utf-8"))
    return float(conventions["yaw"]["sim_left_actuator_yaw_sign"])


def yaw_left_actuator_sign(args: argparse.Namespace) -> float:
    override = getattr(args, "yaw_left_actuator_sign", None)
    if override is not None:
        return float(override)
    return conventions_yaw_left_actuator_sign(args.conventions)


def drive_limits_from_args(args: argparse.Namespace) -> DriveLimits:
    return DriveLimits(
        max_forward_vel_m_s=args.max_forward_vel_m_s,
        max_yaw_rate_rad_s=args.max_yaw_rate_rad_s,
        forward_accel_limit_m_s2=args.drive_accel_limit_m_s2,
        yaw_accel_limit_rad_s2=args.drive_yaw_accel_limit_rad_s2,
        command_timeout_s=args.command_timeout_s,
        reference_position_band_m=args.reference_position_band_m,
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
    parser.add_argument(
        "--linearization-vel-eps-m-s",
        type=float,
        default=0.05,
        help=(
            "Finite forward-velocity perturbation used to identify the velocity "
            "columns of A.  Must sit above the wheels' dry-friction band or the "
            "identified plant describes stiction rather than rolling dynamics."
        ),
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
        "--yaw-axis",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="IMU local axis used as robot yaw. Finn's convention is Y=1.",
    )
    parser.add_argument("--yaw-sign", type=float, default=1.0, choices=(-1.0, 1.0))
    add_drive_arguments(parser)
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
    parser.add_argument(
        "--firmware-header",
        type=Path,
        help=(
            "Write the validated gain, trim, model hash, and convention constants "
            "to a C++ header for the real Finn controller."
        ),
    )
    parser.add_argument(
        "--conventions",
        type=Path,
        default=DEFAULT_CONVENTIONS,
        help="Machine-readable Finn frame/sign convention contract.",
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
    if args.firmware_header:
        print(f"wrote firmware controller header: {portable_path(args.firmware_header)}")
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
        linearization_vel_eps_m_s=args.linearization_vel_eps_m_s,
        fall_pitch_rad=args.fall_pitch_rad,
        pitch_axis=args.pitch_axis,
        pitch_sign=args.pitch_sign,
        forward_sign=args.forward_sign,
        yaw_axis=args.yaw_axis,
        yaw_sign=args.yaw_sign,
        yaw_left_actuator_sign=yaw_left_actuator_sign(args),
        drive=drive_limits_from_args(args),
    )

    validate_timing(model, config)
    validate_linearization_torque(config, handles)

    estimator = calibrated_estimator(model, handles, config)

    # Linearization produces the discrete-time model:
    #
    #   x[k+1] = A x[k] + B u[k]
    #
    # where u is one scalar: the common balance torque sent equally to both
    # wheel motors.  Steering rides on a second, decoupled channel identified by
    # linearize_yaw_dynamics below, so this stays a three-state design.
    a_matrix, b_matrix = linearize_balance_dynamics(model, handles, estimator, config)
    q_cost = np.diag(np.array(args.q_diag, dtype=float))
    r_cost = np.array([[float(args.r)]], dtype=float)
    gain = discrete_lqr(a_matrix, b_matrix, q_cost, r_cost)
    closed_loop_eigs = np.linalg.eigvals(a_matrix - b_matrix @ gain)

    a_yaw, b_yaw = linearize_yaw_dynamics(model, handles, estimator, config)
    gain_yaw = float(
        discrete_lqr(
            np.array([[a_yaw]]),
            np.array([[b_yaw]]),
            np.array([[float(args.q_yaw)]]),
            np.array([[float(args.r_yaw)]]),
        ).item()
    )

    rows, metrics = run_closed_loop(
        model,
        handles,
        estimator,
        config,
        gain,
        gain_yaw=gain_yaw,
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
        "model_sha256_12": sha256_12(args.model),
        "state_names": list(STATE_NAMES),
        "control": {
            "name": "common_balance_torque_nm_plus_differential_yaw_nm",
            "mapping": {
                "motor_left_wheel": "tau_balance + tau_yaw",
                "motor_right_wheel": "tau_balance - tau_yaw",
            },
            "torque_limit_nm": handles.torque_limit_nm,
            "control_dt_s": config.control_dt_s,
            "wheel_radius_m": handles.wheel_radius_m,
            "target_pitch_rad": config.target_pitch_rad,
            "target_forward_vel_m_s": config.target_forward_vel_m_s,
            "position_hold_kp_s": config.position_hold_kp_s,
            "max_position_correction_m_s": config.max_position_correction_m_s,
            "pitch_axis": config.pitch_axis,
            "pitch_sign": config.pitch_sign,
            "forward_sign": config.forward_sign,
            "yaw_axis": config.yaw_axis,
            "yaw_sign": config.yaw_sign,
            "yaw_left_actuator_sign": config.yaw_left_actuator_sign,
            "drive_limits": asdict(config.drive),
        },
        "linearization": {
            "torque_eps_nm": config.linearization_torque_eps_nm,
            "vel_eps_m_s": config.linearization_vel_eps_m_s,
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
        "yaw": {
            "a": a_yaw,
            "b": b_yaw,
            "q": float(args.q_yaw),
            "r": float(args.r_yaw),
            "gain": gain_yaw,
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
            (
                "Steering is a separate one-state loop on differential torque, and it "
                "only ever spends the torque headroom the balance loop leaves behind."
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

    if args.firmware_header:
        write_firmware_header(args.firmware_header, result, args.conventions)
        result["artifacts"]["firmware_header"] = portable_path(args.firmware_header)  # type: ignore[index]

    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def write_firmware_header(path: Path, result: dict[str, object], conventions_path: Path) -> None:
    """Export the exact validated sim controller into a small C++ header.

    The real firmware never carries a hand-copied gain. Regenerating this file
    records the model hash and pulls every frame/sign constant from the checked-in
    convention contract.
    """

    if result.get("status") != "pass":
        raise LqrSimError("refusing to export firmware constants from a failed sim rollout")
    if not conventions_path.exists():
        raise LqrSimError(f"missing convention contract: {conventions_path}")

    conventions = yaml.safe_load(conventions_path.read_text(encoding="utf-8"))
    try:
        imu = conventions["imu"]
        wheels = conventions["wheel_odometry"]
        actuation = conventions["actuation"]
        yaw = conventions["yaw"]
        control = result["control"]
        drive = control["drive_limits"]
        gain = result["lqr"]["gain"][0]
        gain_yaw = result["yaw"]["gain"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LqrSimError("controller result or convention contract is incomplete") from exc

    if result.get("state_names") != list(STATE_NAMES):
        raise LqrSimError("firmware export requires the canonical three-state LQR ordering")

    def cpp_bool(value: object) -> str:
        return "true" if bool(value) else "false"

    def cpp_float(value: object) -> str:
        rendered = f"{float(value):.9g}"
        if not any(marker in rendered for marker in (".", "e", "E")):
            rendered += ".0"
        return rendered + "f"

    control_period_us = round(float(control["control_dt_s"]) * 1_000_000.0)
    max_position_correction = cpp_float(control["max_position_correction_m_s"])
    left_encoder_sign = cpp_float(wheels["real_left_to_forward_sign"])
    right_encoder_sign = cpp_float(wheels["real_right_to_forward_sign"])
    left_torque_sign = cpp_float(actuation["real_left_common_torque_sign"])
    right_torque_sign = cpp_float(actuation["real_right_common_torque_sign"])
    differential_sign = cpp_float(yaw["real_left_actuator_yaw_sign"])
    command_timeout_ms = round(float(drive["command_timeout_s"]) * 1000.0)
    reference_position_band = cpp_float(drive["reference_position_band_m"])
    lines = [
        "#pragma once",
        "",
        "// Generated by tools/run_lqr_sim.py from the committed seeded model.",
        "// Do not tune this header by hand; change the sim/controller inputs and regenerate it.",
        "namespace FinnLqrSeeded {",
        f'constexpr char kModelSha256[] = "{result["model_sha256_12"]}";',
        f"constexpr unsigned long kControlPeriodUs = {control_period_us}UL;",
        f"constexpr float kWheelRadiusM = {cpp_float(control['wheel_radius_m'])};",
        f"constexpr float kTorqueLimitNm = {cpp_float(control['torque_limit_nm'])};",
        f"constexpr float kTargetPitchRad = {cpp_float(control['target_pitch_rad'])};",
        f"constexpr float kTargetForwardVelMS = {cpp_float(control['target_forward_vel_m_s'])};",
        f"constexpr float kPositionHoldKpS = {cpp_float(control['position_hold_kp_s'])};",
        f"constexpr float kMaxPositionCorrectionMS = {max_position_correction};",
        f"constexpr float kGainPitch = {cpp_float(gain[0])};",
        f"constexpr float kGainPitchRate = {cpp_float(gain[1])};",
        f"constexpr float kGainForwardVel = {cpp_float(gain[2])};",
        f"constexpr int kPitchAxis = {int(control['pitch_axis'])};",
        f"constexpr float kPitchSign = {cpp_float(imu['pitch_sign'])};",
        f"constexpr float kForwardAccelSign = {cpp_float(imu['forward_accel_sign'])};",
        f"constexpr float kRealLeftEncoderForwardSign = {left_encoder_sign};",
        f"constexpr float kRealRightEncoderForwardSign = {right_encoder_sign};",
        "constexpr bool kWheelEncoderDirectionsBenchVerified = "
        f"{cpp_bool(wheels['encoder_directions_bench_verified'])};",
        f"constexpr float kRealLeftCommonTorqueSign = {left_torque_sign};",
        f"constexpr float kRealRightCommonTorqueSign = {right_torque_sign};",
        "constexpr bool kPitchDirectionBenchVerified = "
        f"{cpp_bool(imu['pitch_direction_bench_verified'])};",
        "",
        "// Steering and the command layer above the always-on balance loop.",
        "// Firmware that ignores everything below still balances as validated.",
        f"constexpr int kYawAxis = {int(control['yaw_axis'])};",
        f"constexpr float kYawSign = {cpp_float(imu['yaw_sign'])};",
        f"constexpr float kGainYawRate = {cpp_float(gain_yaw)};",
        f"constexpr float kRealLeftActuatorYawSign = {differential_sign};",
        f"constexpr float kMaxForwardVelMS = {cpp_float(drive['max_forward_vel_m_s'])};",
        f"constexpr float kMaxYawRateRadS = {cpp_float(drive['max_yaw_rate_rad_s'])};",
        f"constexpr float kDriveAccelLimitMS2 = {cpp_float(drive['forward_accel_limit_m_s2'])};",
        f"constexpr float kDriveYawAccelLimitRadS2 = {cpp_float(drive['yaw_accel_limit_rad_s2'])};",
        f"constexpr unsigned long kCommandTimeoutMs = {command_timeout_ms}UL;",
        f"constexpr float kRefPositionBandM = {reference_position_band};",
        "constexpr bool kYawDirectionBenchVerified = "
        f"{cpp_bool(imu['yaw_direction_bench_verified'])};",
        "}  // namespace FinnLqrSeeded",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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

    left_tire_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_tire_collision")
    right_tire_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_tire_collision")
    require_floor_plane_at_origin(model)

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
        left_tire_geom_id=left_tire_geom_id,
        right_tire_geom_id=right_tire_geom_id,
        wheel_radius_m=wheel_radius(model),
        torque_limit_nm=abs(float(left_range[1])),
    )


def require_floor_plane_at_origin(model: mujoco.MjModel) -> None:
    """Ground settling measures tire height against z=0, so verify the floor is there."""

    floor_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if int(model.geom_type[floor_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise LqrSimError("expected the 'floor' geom to be a plane")
    if not math.isclose(float(model.geom_pos[floor_id][2]), 0.0, abs_tol=1e-9):
        raise LqrSimError(
            f"expected the floor plane at z=0, found z={float(model.geom_pos[floor_id][2])}"
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
    if config.linearization_vel_eps_m_s <= 0.0:
        raise LqrSimError("--linearization-vel-eps-m-s must be positive")


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
    you will simulate.

    Perturbation size is a physical choice here, not a numerical one.  The seeded
    wheel joints carry ~0.24 N*m of dry friction, and any probe small enough to
    stay inside that band measures stiction instead of dynamics.  The torque probe
    has always been finite for this reason; the forward-velocity probe needs the
    same treatment, because below roughly 0.01 m/s the wheels never break away and
    the model reports velocity collapsing by half every tick.  Pitch and pitch rate
    are flat across four decades of perturbation once the tires are on the ground,
    so those stay small.
    """

    state_eps = np.array(
        [1e-3, 1e-3, config.linearization_vel_eps_m_s],
        dtype=float,
    )
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

    require_controllable(a_matrix, b_matrix)
    return a_matrix, b_matrix


def linearize_yaw_dynamics(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
) -> tuple[float, float]:
    """Identify the scalar yaw plant w[k+1] = a*w[k] + b*d[k] from the model.

    Differential torque and common torque are decoupled in the linearization, so
    steering is its own one-state loop rather than a fourth LQR state; adding it
    to STATE_NAMES would break the firmware export contract for no benefit.

    Both coefficients are measured rather than derived, which is the point: a
    closed-form yaw plant needs a track width, and the CAD, compiled-model, and
    Batch 2 values disagree by 40 percent.  Measuring also means the sign of `b`
    carries the differential direction, so a mirrored joint cannot silently invert
    the steering gain the way it once inverted the velocity term.
    """

    torque_eps = config.linearization_torque_eps_nm
    zero_state = np.zeros(len(STATE_NAMES), dtype=float)

    def yaw_rate_after_tick(tau_yaw_nm: float) -> float:
        data = mujoco.MjData(model)
        set_reduced_state(model, data, handles, config, zero_state)
        apply_differential_torque(data, handles, config, tau_yaw_nm)
        step_control_tick(model, data, config)
        return estimator.yaw_rate_rad_s(data)

    b_yaw = (yaw_rate_after_tick(torque_eps) - yaw_rate_after_tick(-torque_eps)) / (
        2.0 * torque_eps
    )
    if abs(b_yaw) < 1e-9:
        raise LqrSimError(
            "identified yaw input gain is near zero; increase --linearization-torque-eps-nm "
            "or revisit wheel friction/contact parameters"
        )

    # Driving it up to speed keeps body and wheel velocities consistent without
    # anyone having to pick a track width to relate them.
    data = mujoco.MjData(model)
    set_reduced_state(model, data, handles, config, zero_state)
    for _ in range(YAW_SPINUP_TICKS):
        apply_differential_torque(data, handles, config, torque_eps)
        step_control_tick(model, data, config)
    spun_rate = estimator.yaw_rate_rad_s(data)
    apply_differential_torque(data, handles, config, 0.0)
    step_control_tick(model, data, config)
    decayed_rate = estimator.yaw_rate_rad_s(data)

    a_yaw = decayed_rate / spun_rate if abs(spun_rate) > 1e-9 else 1.0
    # A yawing robot only loses rate to friction, so anything outside (0, 1] is an
    # artifact; an undamped integrator asks more of the design than the real plant.
    if not 0.0 < a_yaw <= 1.0:
        a_yaw = 1.0
    return float(a_yaw), float(b_yaw)


def require_controllable(a_matrix: np.ndarray, b_matrix: np.ndarray) -> None:
    """Reject a plant the LQR cannot actually stabilize in every state direction.

    An uncontrollable direction leaves a closed-loop pole wherever the open-loop
    plant put it.  When that pole sits on the unit circle the rollout still looks
    stable while forward position random-walks, which is exactly what a
    free-falling linearization produces.  Catch it here rather than shipping the
    gain to hardware.
    """

    order = len(STATE_NAMES)
    controllability = np.hstack(
        [np.linalg.matrix_power(a_matrix, k) @ b_matrix for k in range(order)]
    )
    rank = int(np.linalg.matrix_rank(controllability))
    if rank < order:
        raise LqrSimError(
            f"linearized plant is uncontrollable (rank {rank} of {order}). The identified "
            "dynamics cannot be stabilized in every state direction; check that the robot "
            "is in ground contact and that the model's balance dynamics are present."
        )


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
    """Set a robot-readable initial condition with the tires on the ground.

    This function is allowed to touch MuJoCo qpos/qvel because it prepares a
    simulation experiment.  The controller itself never sees these internals.

    The base height is not taken from the model's authored rest pose: that pose
    floats the tires above the floor, and every experiment started from it would
    run in free fall.  Instead the chassis is settled onto the floor once at this
    attitude, and the commanded state is then re-applied at the settled height.
    """

    apply_reduced_state(model, data, handles, state, base_z_m=None)
    base_z_m = settle_base_height_m(model, data, handles, state)
    apply_reduced_state(model, data, handles, state, base_z_m=base_z_m)
    require_ground_contact(data)


def apply_reduced_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    state: np.ndarray,
    *,
    base_z_m: float | None,
) -> None:
    """Write the full reduced state into qpos/qvel, optionally overriding base height."""

    pitch_rad, pitch_rate_rad_s, forward_vel_m_s = state
    forward_pos_m = 0.0

    mujoco.mj_resetData(model, data)

    # The freejoint position keeps the chassis above the wheels.  We move x with
    # wheel odometry so the visual/root pose and encoder pose start consistent.
    data.qpos[0] = forward_pos_m
    if base_z_m is not None:
        data.qpos[2] = base_z_m

    # A world-Y rotation appears as local IMU X pitch after subtracting the
    # neutral IMU orientation.  This was verified against the generated model.
    data.qpos[3:7] = axis_angle_quat(axis=1, angle=float(pitch_rad))

    # Freejoint qvel order is [vx, vy, vz, wx, wy, wz].  World-Y angular velocity
    # maps to the current IMU pitch-rate axis for Finn's mounted IMU frame.
    data.qvel[0] = forward_vel_m_s
    data.qvel[4] = pitch_rate_rad_s

    wheel_rad = forward_pos_m / handles.wheel_radius_m
    wheel_rad_s = forward_vel_m_s / handles.wheel_radius_m

    # Sign-corrected forward convention: left wheel positive and right wheel
    # negative roll the generated model toward world +x.  Verified by rolling the
    # joints directly against world displacement, not assumed from the mirroring
    # (see tests/test_run_lqr_sim_smoke.py::test_odometry_forward_sign_matches_world_motion).
    data.qpos[handles.left_wheel_qposadr] = wheel_rad
    data.qpos[handles.right_wheel_qposadr] = -wheel_rad
    data.qvel[handles.left_wheel_dofadr] = wheel_rad_s
    data.qvel[handles.right_wheel_dofadr] = -wheel_rad_s

    mujoco.mj_forward(model, data)


def settle_base_height_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    state: np.ndarray,
) -> float:
    """Return the base height at which the tires rest on the floor at this attitude.

    Lowering the chassis to exact tangency is not enough: MuJoCo only generates a
    contact once the geometries actually overlap, and the resting penetration is a
    property of the contact solver, not something worth hard-coding.  So we drop the
    tires to tangency and let the model settle, holding the base orientation fixed so
    the inverted pendulum cannot tip away from the attitude being prepared.
    """

    data.qpos[2] -= lowest_tire_gap_m(model, data, handles)
    mujoco.mj_forward(model, data)

    quat = np.array(data.qpos[3:7], dtype=float)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        data.qpos[3:7] = quat
        data.qvel[3:6] = 0.0
    return float(data.qpos[2])


def lowest_tire_gap_m(model: mujoco.MjModel, data: mujoco.MjData, handles: ModelHandles) -> float:
    """Signed distance from the lowest tire surface down to the floor plane."""

    return min(
        float(data.geom_xpos[geom_id][2]) - handles.wheel_radius_m
        for geom_id in (handles.left_tire_geom_id, handles.right_tire_geom_id)
    )


def require_ground_contact(data: mujoco.MjData) -> None:
    """Fail loudly if an experiment is about to run with the robot in the air.

    A floating start silently removes the inverted-pendulum mode from every
    finite-difference measurement taken from this state.
    """

    if data.ncon == 0:
        raise LqrSimError(
            "robot is not touching the ground after settling; the identified plant "
            "would be a free-falling body with no balance dynamics"
        )


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
    realtime: bool | None = None,
    gain_yaw: float = 0.0,
    command_source: CommandSource | None = None,
    key_callback: Callable[[int], None] | None = None,
    on_viewer_sync: Callable[[object], None] | None = None,
    on_tick: Callable[[int, dict[str, float], mujoco.MjData], None] | None = None,
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
    arbiter = CommandArbiter(limits=config.drive)
    reference_forward_pos_m = 0.0
    pace_to_wall_clock = show_viewer if realtime is None else realtime

    viewer_context = (
        mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        if show_viewer
        else nullcontext(None)
    )
    with viewer_context as viewer:
        wall_start_s = time.perf_counter()
        if viewer is not None:
            if on_viewer_sync is not None:
                on_viewer_sync(viewer)
            viewer.sync()

        for tick in range(control_steps + 1):
            if tick > 0 and viewer is not None and not viewer.is_running():
                stopped_by_viewer = True
                break

            time_s = tick * config.control_dt_s
            state = estimator.state(data)
            yaw_rate_rad_s = estimator.yaw_rate_rad_s(data)
            forward_pos_m = estimator.relative_forward_pos_m(data)

            # Must run before the torque below, unlike on_tick, or the command
            # lands a tick late.
            command = arbiter.step(command_source, time_s, config.control_dt_s)
            reference_forward_pos_m += command.forward_vel_m_s * config.control_dt_s
            reference_forward_pos_m = clamp(
                reference_forward_pos_m,
                forward_pos_m - config.drive.reference_position_band_m,
                forward_pos_m + config.drive.reference_position_band_m,
            )
            target_forward_pos_m = (
                config.target_forward_vel_m_s * time_s
                if command_source is None
                else reference_forward_pos_m
            )
            position_error_m = forward_pos_m - target_forward_pos_m
            position_velocity_correction_m_s = clamp(
                -config.position_hold_kp_s * position_error_m,
                -config.max_position_correction_m_s,
                config.max_position_correction_m_s,
            )
            effective_target_forward_vel_m_s = (
                config.target_forward_vel_m_s
                + command.forward_vel_m_s
                + position_velocity_correction_m_s
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
            raw_tau_yaw = -gain_yaw * (yaw_rate_rad_s - command.yaw_rate_rad_s)
            tau, tau_yaw = allocate_wheel_torques(raw_tau, raw_tau_yaw, handles.torque_limit_nm)
            left_cmd_nm, right_cmd_nm = yaw_torque_to_wheels(tau_yaw, config.yaw_left_actuator_sign)
            left_cmd_nm += tau
            right_cmd_nm += tau
            saturated = not math.isclose(raw_tau, tau, rel_tol=0.0, abs_tol=1e-12)
            saturated_count += int(saturated)

            rows.append(
                {
                    "time_s": time_s,
                    "pitch_rad": float(state[0]),
                    "pitch_rate_rad_s": float(state[1]),
                    "yaw_rate_rad_s": yaw_rate_rad_s,
                    "forward_pos_m": forward_pos_m,
                    "target_forward_pos_m": target_forward_pos_m,
                    "forward_vel_m_s": float(state[2]),
                    "target_forward_vel_m_s": effective_target_forward_vel_m_s,
                    "cmd_forward_vel_m_s": command.forward_vel_m_s,
                    "cmd_yaw_rate_rad_s": command.yaw_rate_rad_s,
                    "command_stale": float(arbiter.stale),
                    "tau_balance_raw_nm": raw_tau,
                    "tau_balance_nm": tau,
                    "tau_yaw_raw_nm": raw_tau_yaw,
                    "tau_yaw_nm": tau_yaw,
                    "left_cmd_nm": left_cmd_nm,
                    "right_cmd_nm": right_cmd_nm,
                    "saturated": float(saturated),
                }
            )

            # Called with the state that produced this row, so anything it writes to
            # data (an external push, a camera) lands on the step taken just below.
            if on_tick is not None:
                on_tick(tick, rows[-1], data)

            finite = finite and np.all(np.isfinite(state)) and math.isfinite(tau)
            fell = fell or abs(float(state[0])) > config.fall_pitch_rad
            if tick == control_steps or not finite or fell:
                break

            apply_wheel_torques(data, handles, left_cmd_nm, right_cmd_nm)
            step_control_tick(model, data, config)

            if viewer is not None:
                if on_viewer_sync is not None:
                    on_viewer_sync(viewer)
                viewer.sync()
            if pace_to_wall_clock:
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
    settle_ticks = round(COMMAND_SETTLE_S / config.control_dt_s)
    velocity_tracking_p95, velocity_tracking_max, velocity_scored = settled_tracking_error(
        rows, "cmd_forward_vel_m_s", "forward_vel_m_s", settle_ticks
    )
    yaw_tracking_p95, yaw_tracking_max, yaw_scored = settled_tracking_error(
        rows, "cmd_yaw_rate_rad_s", "yaw_rate_rad_s", settle_ticks
    )
    velocity_tracking_assessed = velocity_scored >= MIN_TRACKING_SAMPLES
    yaw_tracking_assessed = yaw_scored >= MIN_TRACKING_SAMPLES

    if command_source is None:
        # Station keeping: the robot was asked to hold a spot and end upright, so
        # judge it on both.
        held_the_reference = config.position_hold_kp_s == 0.0 or (
            max_abs_position_error < 0.25 and final_abs_position_error < 0.10
        )
        ended_upright = final_abs_pitch < SETTLED_PITCH_RAD
    else:
        # An interactive session may never hold a command still long enough to
        # grade, so an unassessed axis is skipped rather than passed on no evidence.
        held_the_reference = (not velocity_tracking_assessed or velocity_tracking_p95 < 0.25) and (
            not yaw_tracking_assessed or yaw_tracking_p95 < 0.40
        )
        # Holding acceleration a costs a steady lean of atan(a/g), so a rollout
        # stopped mid ramp is upright exactly when it sits inside that.
        commanded_lean_rad = math.atan2(config.drive.forward_accel_limit_m_s2, GRAVITY_M_S2)
        settle_slack_rad = SETTLED_PITCH_RAD - abs(config.target_pitch_rad)
        ended_upright = (
            abs(rows[-1]["pitch_rad"] - config.target_pitch_rad)
            < commanded_lean_rad + settle_slack_rad
        )

    passed = (
        finite
        and not fell
        and ended_upright
        and max_abs_pitch < config.fall_pitch_rad
        and held_the_reference
        and saturation_fraction < 0.80
    )
    return rows, {
        "pass": passed,
        "finite": finite,
        "fell": fell,
        "stopped_by_viewer": stopped_by_viewer,
        "max_abs_pitch_rad": max_abs_pitch,
        "final_abs_pitch_rad": final_abs_pitch,
        "ended_upright": ended_upright,
        "final_forward_pos_m": rows[-1]["forward_pos_m"],
        "final_forward_vel_m_s": rows[-1]["forward_vel_m_s"],
        "max_abs_position_error_m": max_abs_position_error,
        "final_abs_position_error_m": final_abs_position_error,
        "velocity_tracking_assessed": velocity_tracking_assessed,
        "velocity_tracking_samples": velocity_scored,
        "yaw_tracking_assessed": yaw_tracking_assessed,
        "yaw_tracking_samples": yaw_scored,
        "velocity_tracking_p95_m_s": velocity_tracking_p95,
        "velocity_tracking_max_m_s": velocity_tracking_max,
        "yaw_tracking_p95_rad_s": yaw_tracking_p95,
        "yaw_tracking_max_rad_s": yaw_tracking_max,
        "max_abs_wheel_cmd_nm": max(
            max(abs(row["left_cmd_nm"]), abs(row["right_cmd_nm"])) for row in rows
        ),
        "rejected_command_samples": arbiter.rejected_samples,
        "saturation_fraction": saturation_fraction,
        "samples": len(rows),
    }


def settled_tracking_error(
    rows: list[dict[str, float]], command_key: str, actual_key: str, settle_ticks: int
) -> tuple[float, float, int]:
    """Tracking error over steady, nonzero commands, as (p95, max, samples_scored).

    Three filters, each for a different false failure.  Settled samples only, so
    the slew ramps are not read as tracking error.  Nonzero commands only, because
    grading a standing robot against a stop is a test it always wins.  A percentile
    rather than the worst sample, because a hard spin makes the tires stick and
    slip and odometry reads a slip as a momentary metre per second.  The max is
    returned beside it, and the count so the caller can tell an unmeasured run
    from a good one.
    """

    errors = []
    for index in range(settle_ticks, len(rows)):
        window = rows[index - settle_ticks : index + 1]
        target = window[-1][command_key]
        if any(abs(row[command_key] - target) > 1e-9 for row in window):
            continue
        if abs(target) <= 1e-9:
            # Counting stop commands is how this gate once reported a confident
            # pass on an interactive run it had never scored moving.
            continue
        errors.append(abs(window[-1][actual_key] - target))
    if not errors:
        return 0.0, 0.0, 0
    return float(np.percentile(errors, 95.0)), max(errors), len(errors)


def apply_wheel_torques(
    data: mujoco.MjData, handles: ModelHandles, left_nm: float, right_nm: float
) -> None:
    data.ctrl[:] = 0.0
    data.ctrl[handles.left_actuator_id] = left_nm
    data.ctrl[handles.right_actuator_id] = right_nm


def apply_balance_torque(data: mujoco.MjData, handles: ModelHandles, tau_balance_nm: float) -> None:
    apply_wheel_torques(data, handles, tau_balance_nm, tau_balance_nm)


def apply_differential_torque(
    data: mujoco.MjData, handles: ModelHandles, config: SimConfig, tau_yaw_nm: float
) -> None:
    left_nm, right_nm = yaw_torque_to_wheels(tau_yaw_nm, config.yaw_left_actuator_sign)
    apply_wheel_torques(data, handles, left_nm, right_nm)


def yaw_torque_to_wheels(tau_yaw_nm: float, left_actuator_sign: float) -> tuple[float, float]:
    """Split a positive-is-left yaw torque across the two wheel actuators.

    The sign is a contract value, not a derivation: the generated model names its
    wheel bodies opposite the robot frame, so guessing it from the actuator names
    turns left into right.  config/finn_conventions.yaml carries it and
    test_positive_yaw_torque_turns_the_model_left holds it to measured behaviour.
    """

    return left_actuator_sign * tau_yaw_nm, -left_actuator_sign * tau_yaw_nm


def step_control_tick(model: mujoco.MjModel, data: mujoco.MjData, config: SimConfig) -> None:
    inner_steps = round(config.control_dt_s / float(model.opt.timestep))
    for _ in range(inner_steps):
        mujoco.mj_step(model, data)


def signed_forward_wheel_rad(left_rad: float, right_rad: float) -> float:
    """Average wheel rotation after converting each encoder to forward-positive.

    The generated model mirrors the left wheel joint, so the two raw joint
    coordinates run opposite each other.  Rolling the joints directly shows that
    left-positive / right-negative carries the chassis toward world +x, so that is
    the combination that means "forward" here.  Getting this backwards inverts the
    velocity feedback term and turns the balance loop into positive feedback.
    """

    return 0.5 * (left_rad - right_rad)


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


def slew(current: float, target: float, max_step: float) -> float:
    return current + clamp(target - current, -max_step, max_step)


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
