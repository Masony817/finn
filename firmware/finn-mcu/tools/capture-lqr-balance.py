#!/usr/bin/env python3
"""Capture a Finn convention check, arming preflight, or real-robot LQR trial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from serial_log_capture import add_capture_arguments, print_host, run_capture

LOG_ROOT = Path("logs/finn-mcu/lqr")
UPLOAD_ENV = "lqr_balance"
PASS_MARKER = ",lqr_complete,"
CHECK_PASS_MARKER = ",convention_check_complete,"
PREFLIGHT_PASS_MARKER = ",preflight_passed,"
PREFLIGHT_FAIL_MARKER = ",preflight_failed,"
FAIL_MARKER = ",failsafe,"
TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[2]
PROFILE = REPO_ROOT / "config" / "viz" / "finn_lqr.yaml"
SYSID_POSTPROCESSOR = TOOLS_DIR / "postprocess_lqr_balance.py"

EXPECTED_SCHEMA = "lqr_v2"


class LiveIntegrityMonitor:
    """Verify, while the robot is still on the bench, that the host is recording.

    The firmware preflight can only check its own subsystems; it cannot know
    whether the rows it emitted arrived intact at the other end of the USB cable.
    A truncated row or a schema the postprocessor cannot read is only worth
    discovering before the operator releases an 8.44 kg robot, not after.
    """

    def __init__(self, expected_schema: str = EXPECTED_SCHEMA) -> None:
        self.expected_schema = expected_schema
        self.schema: str | None = None
        self.columns: list[str] = []
        self.rows = 0
        self.malformed_rows = 0
        self.nonfinite_rows = 0
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_skipped = 0
        self.failed_checks: list[str] = []

    def __call__(self, line: str) -> None:
        if line.startswith("schema,"):
            self.schema = line.split(",", 1)[1].strip()
        elif line.startswith("data,t_us,"):
            self.columns = line.split(",")[1:]
        elif line.startswith("data,"):
            self._observe_row(line)
        elif line.startswith("check,"):
            self._observe_check(line)
        elif ",preflight_passed," in line or ",preflight_failed," in line:
            self.report()

    def _observe_row(self, line: str) -> None:
        self.rows += 1
        fields = line.split(",")[1:]
        if self.columns and len(fields) != len(self.columns):
            self.malformed_rows += 1
        lowered = line.lower()
        if "nan" in lowered or "inf" in lowered:
            self.nonfinite_rows += 1

    def _observe_check(self, line: str) -> None:
        parts = line.split(",")
        if len(parts) < 7:
            return
        _, _, name, result, measured, limit, detail = parts[:7]
        if result == "pass":
            self.checks_passed += 1
        elif result == "skip":
            self.checks_skipped += 1
            print_host(f"  [SKIP] {name:<22} {measured:>12}  {detail}")
        else:
            self.checks_failed += 1
            self.failed_checks.append(name)
            print_host(f"  [FAIL] {name:<22} {measured:>12}  limit {limit}  {detail}")

    @property
    def recording_ok(self) -> bool:
        return (
            self.schema == self.expected_schema
            and bool(self.columns)
            and self.rows > 0
            and self.malformed_rows == 0
            and self.nonfinite_rows == 0
        )

    def report(self) -> None:
        print_host(
            f"firmware checks: {self.checks_passed} pass, {self.checks_failed} fail, "
            f"{self.checks_skipped} skip"
        )
        print_host(
            f"host recording: schema={self.schema} columns={len(self.columns)} "
            f"rows={self.rows} malformed={self.malformed_rows} "
            f"nonfinite={self.nonfinite_rows}"
        )
        if self.schema != self.expected_schema:
            print_host(
                f"  RECORDING FAULT: expected schema {self.expected_schema}; the profile and "
                "postprocessor will not read this run"
            )
        if self.malformed_rows:
            print_host("  RECORDING FAULT: truncated telemetry rows reached the host")
        if self.nonfinite_rows:
            print_host("  RECORDING FAULT: non-finite values in telemetry")
        if self.recording_ok and not self.checks_failed:
            print_host("preflight clear: firmware and host recording both verified")


def sysid_command() -> str:
    return f"{sys.executable} {SYSID_POSTPROCESSOR} --run-dir {{run_dir}}"


def scopik_command(*, replay: bool) -> str:
    command = (
        f"{sys.executable} -m scopik.cli gap --profile {PROFILE} "
        "--run {run_dir} --save {run_dir}/scopik_gap.rrd "
        "--out {run_dir}/scopik_gap.json --no-history"
    )
    return command if replay else command + " --no-replay"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_arguments(parser, default_log_root=LOG_ROOT, default_run_name="lqr")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-conventions",
        action="store_true",
        help="Motor-disabled 15 second sign check; no free-model replay is run.",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Motor-disabled arming preflight that reports every check and never arms.",
    )
    parser.add_argument(
        "--trial-ms",
        type=int,
        help="Trial length to request. The firmware clamps it to its reviewed limit.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Use firmware already on the Teensy instead of flashing lqr_balance.",
    )
    args = parser.parse_args(argv)
    args.upload_env = None if args.skip_upload else UPLOAD_ENV
    args.fail_marker = [FAIL_MARKER]
    args.periodic_command = ["HEARTBEAT"]
    args.periodic_command_interval_s = 0.1
    args.stop_command = ["STOP"]
    args.stop_command_repeats = 3
    args.stop_command_spacing_s = 0.03
    args.timeout_s = args.timeout_s or 120.0
    args.auto_command = [f"TRIAL {args.trial_ms}"] if args.trial_ms else []

    if args.check_conventions:
        args.run_name = "lqr_convention_check"
        args.pass_marker = [CHECK_PASS_MARKER]
        args.arm_command = []
        args.run_command = ["CHECK CONVENTIONS"]
        args.postprocess = [sysid_command(), scopik_command(replay=False)]
        args.show_telemetry = True
    elif args.preflight:
        args.run_name = "lqr_preflight"
        args.pass_marker = [PREFLIGHT_PASS_MARKER]
        args.fail_marker = [FAIL_MARKER, PREFLIGHT_FAIL_MARKER]
        args.arm_command = []
        args.run_command = ["PREFLIGHT"]
        args.postprocess = [sysid_command()]
    else:
        args.pass_marker = [PASS_MARKER]
        args.arm_command = ["ARM FINN"]
        args.run_command = ["RUN LQR"]
        args.postprocess = [sysid_command(), scopik_command(replay=True)]
    return args


def print_instructions(args: argparse.Namespace) -> None:
    if args.check_conventions:
        print_host("MOTORS STAY DISABLED in convention-check mode")
        print_host("hold Finn mechanically upright, then type: ZERO UPRIGHT")
        print_host("type: CHECK CONVENTIONS")
        print_host("tip the top forward and confirm pitch_rad and pitch_rate_rad_s go positive")
        print_host("roll both wheels robot-forward and confirm both wheel velocities go positive")
        print_host("rotate Finn left (CCW from above) and confirm yaw_rate_rad_s goes positive")
        return
    if args.preflight:
        print_host("MOTORS STAY DISABLED in preflight mode; this never arms")
        print_host("hold Finn upright and still, then type: ZERO UPRIGHT")
        print_host("type: PREFLIGHT and hold still for 2 seconds")
        print_host("every failing check prints here with its measured value and limit")
        return
    print_host("two people: one holds and catches Finn, one runs the host")
    print_host("keep the floor clear and a hardware motor-power cutoff within reach")
    print_host("type ZERO UPRIGHT, then ARM FINN, then RUN LQR only when ready to release")
    print_host("ARM FINN runs a 2 s motors-disabled preflight and arms only if every check passes")
    print_host("the host sends HEARTBEAT automatically and sends STOP before disconnecting")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print_instructions(args)
    monitor = LiveIntegrityMonitor()
    result = run_capture(args, monitor)
    if not monitor.recording_ok:
        print_host(
            "WARNING: host recording was not verified clean; treat derived values as suspect"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
