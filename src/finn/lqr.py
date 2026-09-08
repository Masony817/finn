"""Validate the Finn controller and export its firmware configuration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from finn.control import DriveLimits
from finn.paths import (
    DEFAULT_CONVENTIONS,
    DEFAULT_MODEL,
    DEFAULT_OUT_ROOT,
    portable_path,
    sha256_12,
    timestamp,
)
from finn.reporting import write_plot, write_timeseries
from finn.simulation import (
    STATE_NAMES,
    LqrSimError,
    SimConfig,
    calibrated_estimator,
    discrete_lqr,
    estimate_balance_trim_pitch_rad,
    inspect_model,
    linearize_balance_dynamics,
    linearize_yaw_dynamics,
    run_closed_loop,
    validate_linearization_torque,
    validate_timing,
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
            "This does not establish hardware balance or world-trajectory accuracy."
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
