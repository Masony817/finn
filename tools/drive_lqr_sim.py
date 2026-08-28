#!/usr/bin/env python3
"""Drive the balancing Finn model around with WASD, or with a scripted profile.

This is layer 3 of the control stack: a command source and nothing else.  It hands
`tools/run_lqr_sim.py` a stream of DriveCommand intent and never touches torque,
gains, or the plant, so a wedged or wrong keyboard cannot cost the robot its
balance.  A future HRI or navigation policy plugs in at exactly this seam; the
scripted profiles below are the template for it.

macOS needs mjpython for the live viewer:

    uv run --isolated --python /opt/homebrew/bin/python3 \
      mjpython tools/drive_lqr_sim.py --duration-s 60
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import glfw
import mujoco
import numpy as np
from scopik.live import LiveSession

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "sim/generated/seeded/latest/finn.seeded.sim.xml"
DEFAULT_OUT_ROOT = REPO_ROOT / "logs/lqr_drive"
DEFAULT_CONVENTIONS = REPO_ROOT / "config/finn_conventions.yaml"

# What the live Scopik dashboard plots, grouped into one panel per key.
LIVE_SIGNALS = {
    "command": ("cmd_forward_vel_m_s", "cmd_yaw_rate_rad_s", "command_stale"),
    "attitude": ("pitch_rad", "pitch_rate_rad_s", "yaw_rate_rad_s"),
    "velocity": ("forward_vel_m_s", "target_forward_vel_m_s", "forward_pos_m"),
    "torque": ("tau_balance_nm", "tau_yaw_nm", "left_cmd_nm", "right_cmd_nm", "saturated"),
}

# GLFW key codes, which is what the MuJoCo viewer hands the callback.
KEY_W, KEY_A, KEY_S, KEY_D = 87, 65, 83, 68
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE = 32

# MuJoCo binds a shortcut to every letter A-Z, so arrows are the only clean scheme.
DRIVE_LETTERS = frozenset("WASD")
FORWARD_KEYS = (KEY_W, KEY_UP)
BACKWARD_KEYS = (KEY_S, KEY_DOWN)
LEFT_KEYS = (KEY_A, KEY_LEFT)
RIGHT_KEYS = (KEY_D, KEY_RIGHT)
DRIVE_KEYS = FORWARD_KEYS + BACKWARD_KEYS + LEFT_KEYS + RIGHT_KEYS


def viewer_flag_pins() -> list[int]:
    """Which MjvOption vis flags the drive letters would otherwise toggle.

    Only A and D are reachable. W and S are MjvScene render flags, and the passive
    viewer exposes no render scene, so those two cannot be suppressed at all;
    docs/codebase-notes.md has the detail. Indices come from MuJoCo's own table so
    a release that moves a shortcut moves this with it.
    """

    return [
        index
        for index in range(mujoco.mjtVisFlag.mjNVISFLAG)
        if mujoco.mjVISSTRING[index][2].upper() in DRIVE_LETTERS
    ]


def unpinnable_drive_letters() -> list[str]:
    """Drive letters whose viewer shortcut cannot be suppressed. Reported to the user."""

    return sorted(
        mujoco.mjRNDSTRING[index][2].upper()
        for index in range(mujoco.mjtRndFlag.mjNRNDFLAG)
        if mujoco.mjRNDSTRING[index][2].upper() in DRIVE_LETTERS
    )


class ViewerFlagKeeper:
    """Hold the reachable drive-key vis flags at whatever they were on launch."""

    def __init__(self) -> None:
        self._indices = viewer_flag_pins()
        self._wanted: list[int] | None = None

    def __call__(self, viewer) -> None:
        if self._wanted is None:
            self._wanted = [int(viewer.opt.flags[i]) for i in self._indices]
            return
        for index, value in zip(self._indices, self._wanted, strict=True):
            viewer.opt.flags[index] = value


class DriveError(Exception):
    """Expected failure with a concise user-facing message."""


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_lqr_sim():
    """Load run_lqr_sim.py, which is a script rather than an importable module."""

    path = REPO_ROOT / "tools/run_lqr_sim.py"
    spec = importlib.util.spec_from_file_location("finn_run_lqr_sim", path)
    if spec is None or spec.loader is None:
        raise DriveError(f"cannot load {portable_path(path)}")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class KeyboardDrive:
    """Turn viewer key input into a DriveCommand stream, holding while held.

    `key_callback` carries a keycode and nothing else, so a press-timestamp latch
    is all it can support and it makes a poor joystick. The latch is only the
    bootstrap: the first keypress captures the GLFW window off the thread holding
    the GL context, and `held()` reads true key state from then on. Every GLFW call
    falls back to the latch rather than ending the run.
    docs/codebase-notes.md explains why this is safe off-thread.
    """

    def __init__(
        self,
        lqr,
        *,
        hold_window_s: float,
        max_forward_vel_m_s: float,
        max_yaw_rate_rad_s: float,
        clock=time.perf_counter,
        poll_key_state: bool = True,
    ) -> None:
        self._lqr = lqr
        self._hold_window_s = hold_window_s
        self._max_forward = max_forward_vel_m_s
        self._max_yaw = max_yaw_rate_rad_s
        self._clock = clock
        self._poll_key_state = poll_key_state
        # The viewer runs its callback on the UI thread while the rollout steps on
        # a worker, so the latch is shared state.
        self._lock = threading.Lock()
        self._pressed_at: dict[int, float] = {}
        self._window = None

    @property
    def polling_key_state(self) -> bool:
        """True once real held/released state is available."""

        return self._window is not None

    def on_key(self, keycode: int) -> None:
        now = self._clock()
        if self._poll_key_state and self._window is None:
            self._window = self._capture_window()
        with self._lock:
            if keycode == KEY_SPACE:
                self._pressed_at.clear()
            elif keycode in DRIVE_KEYS:
                self._pressed_at[keycode] = now

    def _capture_window(self):
        """Grab the viewer's GLFW window from the thread that owns its context."""

        try:
            with warnings.catch_warnings():
                # No context here just means fall back to the latch, so GLFW's
                # "not initialized" warning is an expected outcome, not a problem.
                warnings.simplefilter("ignore")
                return glfw.get_current_context() or None
        except Exception:
            return None

    def _key_state_held(self, keycode: int) -> bool | None:
        if self._window is None:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return glfw.get_key(self._window, keycode) != glfw.RELEASE
        except Exception:
            # A closed window or an unusable handle must not end the run.
            self._window = None
            return None

    def held(self, keycode: int) -> bool:
        state = self._key_state_held(keycode)
        if state is not None:
            return state
        with self._lock:
            pressed_at = self._pressed_at.get(keycode)
        return pressed_at is not None and self._clock() - pressed_at <= self._hold_window_s

    def _any_held(self, keycodes) -> bool:
        return any(self.held(keycode) for keycode in keycodes)

    def command(self, _time_s: float):
        # Space overrides everything, so it still stops the robot mid-hold. In the
        # latch fallback it has already cleared the stamps and reads as not held.
        if self.held(KEY_SPACE):
            return self._lqr.DriveCommand(0.0, 0.0)
        forward = self._max_forward * (self._any_held(FORWARD_KEYS) - self._any_held(BACKWARD_KEYS))
        yaw = self._max_yaw * (self._any_held(LEFT_KEYS) - self._any_held(RIGHT_KEYS))
        return self._lqr.DriveCommand(forward, yaw)


def scripted_profile(lqr, name: str, *, max_forward_vel_m_s: float, max_yaw_rate_rad_s: float):
    """Deterministic command sources, so CI can drive without a keyboard."""

    stopped = lqr.DriveCommand()

    if name == "none":
        return None

    if name == "spin":

        def spin(time_s: float):
            if time_s < 2.0:
                return stopped
            return lqr.DriveCommand(0.0, max_yaw_rate_rad_s)

        return spin

    if name == "square":
        # Drive a leg, turn a corner, repeat: exercises forward tracking, steering,
        # and the transition between them, which is where they interact.
        leg_s, turn_s, settle_s = 3.0, 2.0, 2.0

        def square(time_s: float):
            if time_s < settle_s:
                return stopped
            phase_s = (time_s - settle_s) % (leg_s + turn_s)
            if phase_s < leg_s:
                return lqr.DriveCommand(max_forward_vel_m_s, 0.0)
            return lqr.DriveCommand(0.0, max_yaw_rate_rad_s)

        return square

    raise DriveError(f"unknown drive profile: {name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    lqr = load_lqr_sim()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--control-dt-s", type=float, default=0.01)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument(
        "--initial-pitch-rad",
        type=float,
        help="Defaults to the balance trim, so a driving session starts settled.",
    )
    parser.add_argument("--target-pitch-rad", type=float)
    parser.add_argument("--position-hold-kp-s", type=float, default=0.15)
    parser.add_argument("--max-position-correction-m-s", type=float, default=0.15)
    parser.add_argument("--linearization-vel-eps-m-s", type=float, default=0.05)
    parser.add_argument("--linearization-torque-eps-nm", type=float, default=0.18)
    parser.add_argument("--fall-pitch-rad", type=float, default=0.45)
    parser.add_argument("--q-diag", type=float, nargs=3, default=(160.0, 16.0, 2.0))
    parser.add_argument("--r", type=float, default=0.6)
    parser.add_argument(
        "--drive-profile",
        choices=("none", "square", "spin"),
        default="none",
        help="Scripted command source. 'none' means drive from the keyboard.",
    )
    parser.add_argument(
        "--key-hold-window-s",
        type=float,
        default=0.6,
        help=(
            "Fallback only: how long a keypress counts as held when true key state "
            "is unavailable. Must outlast the OS key-repeat delay."
        ),
    )
    parser.add_argument(
        "--no-key-state-polling",
        action="store_true",
        help="Drive from key events alone, without reading GLFW's real key state.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run headless. Required for a scripted profile in CI.",
    )
    parser.add_argument(
        "--no-scopik",
        action="store_true",
        help="Do not open a live Scopik dashboard alongside the run.",
    )
    parser.add_argument(
        "--scopik-rrd",
        type=Path,
        help="Stream to a recording instead of a viewer window. Implies --no-scopik window.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help=(
            "Run as fast as the machine allows. Pacing is on by default so the "
            "live dashboard and the keyboard both track wall clock."
        ),
    )
    parser.add_argument("--conventions", type=Path, default=DEFAULT_CONVENTIONS)
    lqr.add_drive_arguments(parser)
    args = parser.parse_args(argv)
    args.lqr = lqr
    if args.drive_profile == "none" and args.no_viewer:
        parser.error("--no-viewer needs a --drive-profile; there is no keyboard to read")
    return args


def live_session(args: argparse.Namespace) -> LiveSession | None:
    """Open the live dashboard, unless the run asked to go without one."""

    if args.no_scopik and not args.scopik_rrd:
        return None
    session = LiveSession.from_columns(LIVE_SIGNALS, app_id="finn-drive")
    if args.scopik_rrd:
        session.save(args.scopik_rrd)
    else:
        session.spawn()
    return session


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


def print_controls() -> None:
    print("  hold to drive; release to coast to a stop")
    print("  up / down  or  W / S      drive forward / back")
    print("  left / right  or  A / D   turn left / right")
    print("  space                     stop")
    print("  Finn keeps balancing whether or not you touch any of them.")
    stuck = unpinnable_drive_letters()
    if stuck:
        print(
            f"  prefer the arrows in the viewer: {' and '.join(stuck)} also toggle a "
            "MuJoCo render flag that no API lets us suppress (rendering only, "
            "driving is unaffected)."
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    lqr = args.lqr
    if not args.model.exists():
        raise DriveError(f"missing model XML: {args.model}")

    out_dir = args.out_dir or DEFAULT_OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = lqr.mujoco.MjModel.from_xml_path(str(args.model))
    handles = lqr.inspect_model(model)
    target_pitch_rad = (
        lqr.estimate_balance_trim_pitch_rad(model)
        if args.target_pitch_rad is None
        else float(args.target_pitch_rad)
    )
    config = lqr.SimConfig(
        control_dt_s=args.control_dt_s,
        duration_s=args.duration_s,
        initial_pitch_rad=(
            target_pitch_rad if args.initial_pitch_rad is None else args.initial_pitch_rad
        ),
        target_pitch_rad=target_pitch_rad,
        target_forward_vel_m_s=0.0,
        position_hold_kp_s=args.position_hold_kp_s,
        max_position_correction_m_s=args.max_position_correction_m_s,
        linearization_torque_eps_nm=args.linearization_torque_eps_nm,
        linearization_vel_eps_m_s=args.linearization_vel_eps_m_s,
        fall_pitch_rad=args.fall_pitch_rad,
        pitch_axis=0,
        pitch_sign=1.0,
        forward_sign=1.0,
        yaw_axis=1,
        yaw_sign=1.0,
        yaw_left_actuator_sign=lqr.yaw_left_actuator_sign(args),
        drive=lqr.drive_limits_from_args(args),
    )
    lqr.validate_timing(model, config)
    lqr.validate_linearization_torque(config, handles)

    estimator = lqr.calibrated_estimator(model, handles, config)
    a_matrix, b_matrix = lqr.linearize_balance_dynamics(model, handles, estimator, config)
    gain = lqr.discrete_lqr(
        a_matrix,
        b_matrix,
        np.diag(np.array(args.q_diag, dtype=float)),
        np.array([[float(args.r)]]),
    )
    a_yaw, b_yaw = lqr.linearize_yaw_dynamics(model, handles, estimator, config)
    gain_yaw = float(
        lqr.discrete_lqr(
            np.array([[a_yaw]]),
            np.array([[b_yaw]]),
            np.array([[float(args.q_yaw)]]),
            np.array([[float(args.r_yaw)]]),
        ).item()
    )

    keyboard = None
    key_callback = None
    if args.drive_profile == "none":
        keyboard = KeyboardDrive(
            lqr,
            poll_key_state=not args.no_key_state_polling,
            hold_window_s=args.key_hold_window_s,
            max_forward_vel_m_s=config.drive.max_forward_vel_m_s,
            max_yaw_rate_rad_s=config.drive.max_yaw_rate_rad_s,
        )
        command_source = keyboard.command
        key_callback = keyboard.on_key
        print_controls()
    else:
        command_source = scripted_profile(
            lqr,
            args.drive_profile,
            max_forward_vel_m_s=config.drive.max_forward_vel_m_s,
            max_yaw_rate_rad_s=config.drive.max_yaw_rate_rad_s,
        )

    live = live_session(args)
    if live is not None:
        print(f"streaming live to Scopik ({'recording' if args.scopik_rrd else 'viewer'})")

    def stream(_tick: int, row: dict[str, float], _data) -> None:
        live.log_row(row["time_s"], row)

    try:
        rows, metrics = lqr.run_closed_loop(
            model,
            handles,
            estimator,
            config,
            gain,
            gain_yaw=gain_yaw,
            command_source=command_source,
            key_callback=key_callback,
            on_viewer_sync=ViewerFlagKeeper() if keyboard is not None else None,
            on_tick=None if live is None else stream,
            show_viewer=not args.no_viewer,
            realtime=not args.no_realtime,
        )
    finally:
        if live is not None:
            live.close()

    key_state_polling = keyboard is not None and keyboard.polling_key_state
    if keyboard is not None and not key_state_polling:
        print(
            "note: could not read GLFW key state, so driving used the "
            f"{args.key_hold_window_s:.2f} s press latch instead of true hold"
        )

    csv_path = out_dir / "timeseries.csv"
    report_path = out_dir / "report.json"
    lqr.write_timeseries(csv_path, rows)

    result: dict[str, object] = {
        "status": "pass" if metrics["pass"] else "failed",
        "out_dir": portable_path(out_dir),
        "model": portable_path(args.model),
        "model_sha256_12": lqr.sha256_12(args.model),
        "drive_profile": args.drive_profile,
        "key_state_polling": key_state_polling,
        "drive_limits": lqr.asdict(config.drive),
        "lqr": {"gain": gain.tolist(), "yaw_gain": gain_yaw},
        "yaw": {"a": a_yaw, "b": b_yaw},
        "metrics": metrics,
        "artifacts": {
            "timeseries_csv": portable_path(csv_path),
            "report_json": portable_path(report_path),
            **({"scopik_rrd": portable_path(args.scopik_rrd)} if args.scopik_rrd else {}),
        },
        "scope": (
            "Sim-only teleop of the validated balance controller. The command layer "
            "is bounded and rate limited here exactly as the firmware port must be, "
            "but no part of this has run on hardware."
        ),
    }
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _tracking_text(label: str, metrics: dict, prefix: str, unit: str) -> str:
    """Never print a tracking number without saying whether it was measurable."""

    if not metrics[f"{prefix}_tracking_assessed"]:
        return (
            f"{label}_track=not assessed ({metrics[f'{prefix}_tracking_samples']} steady samples)"
        )
    return (
        f"{label}_track_p95={metrics[f'{prefix}_tracking_p95_{unit}']:.5f} "
        f"(max {metrics[f'{prefix}_tracking_max_{unit}']:.5f}, "
        f"n={metrics[f'{prefix}_tracking_samples']})"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except DriveError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    metrics = result["metrics"]
    print(f"wrote drive bundle: {result['out_dir']}")
    print(f"status: {result['status']}")
    print(
        "metrics: "
        f"max_abs_pitch_rad={metrics['max_abs_pitch_rad']:.5f}, "
        f"max_abs_wheel_cmd_nm={metrics['max_abs_wheel_cmd_nm']:.5f}, "
        f"{_tracking_text('vel', metrics, 'velocity', 'm_s')}, "
        f"{_tracking_text('yaw', metrics, 'yaw', 'rad_s')}"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
