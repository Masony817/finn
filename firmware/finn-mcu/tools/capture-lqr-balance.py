#!/usr/bin/env python3
"""Capture one safe Finn convention check or short real-robot LQR trial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from serial_log_capture import add_capture_arguments, print_host, run_capture

LOG_ROOT = Path("logs/finn-mcu/lqr")
UPLOAD_ENV = "lqr_balance"
PASS_MARKER = ",lqr_complete,"
CHECK_PASS_MARKER = ",convention_check_complete,"
FAIL_MARKER = ",failsafe,"
PROFILE = Path(__file__).resolve().parents[3] / "config" / "viz" / "finn_lqr.yaml"


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
    parser.add_argument(
        "--check-conventions",
        action="store_true",
        help="Motor-disabled 15 second sign check; no free-model replay is run.",
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

    if args.check_conventions:
        args.run_name = "lqr_convention_check"
        args.pass_marker = [CHECK_PASS_MARKER]
        args.arm_command = []
        args.run_command = ["CHECK CONVENTIONS"]
        args.postprocess = scopik_command(replay=False)
        args.show_telemetry = True
    else:
        args.pass_marker = [PASS_MARKER]
        args.arm_command = ["ARM FINN"]
        args.run_command = ["RUN LQR"]
        args.postprocess = scopik_command(replay=True)
    return args


def print_instructions(check_conventions: bool) -> None:
    if check_conventions:
        print_host("MOTORS STAY DISABLED in convention-check mode")
        print_host("hold Finn mechanically upright, then type: ZERO UPRIGHT")
        print_host("type: CHECK CONVENTIONS")
        print_host("tip the top forward and confirm pitch_rad and pitch_rate_rad_s go positive")
        print_host("roll both wheels robot-forward and confirm both wheel velocities go positive")
        return
    print_host("one operator must hold, release, and catch Finn; keep the floor area clear")
    print_host("type ZERO UPRIGHT, then ARM FINN, then RUN LQR only when ready to release")
    print_host("the host sends HEARTBEAT automatically and sends STOP before disconnecting")
    print_host("the first trial stops after 3 seconds or immediately on any safety limit")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print_instructions(args.check_conventions)
    return run_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
