#!/usr/bin/env python3
"""Capture Finn Batch 1 sysid serial logs in realtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from serial_log_capture import add_capture_arguments, run_capture

SYSID_LOG_ROOT = Path("logs/finn-mcu/sysid")
SYSID_RUN_NAME = "batch_1"
BATCH1_PASS_MARKER = ",batch1_complete,"
BATCH1_FAIL_MARKER = ",failsafe,"
BATCH1_AUTO_COMMANDS = ("ARM FINN", "RUN BATCH1")
BATCH1_ARM_COMMANDS = ("ARM FINN",)
BATCH1_RUN_COMMANDS = ("RUN BATCH1",)
BATCH1_UPLOAD_ENV = "sysid_batch1_wheels_offground"
POSTPROCESS_SCRIPT = Path(__file__).resolve().parent / "postprocess_sysid_batch1.py"
# {run_dir} is substituted by serial_log_capture after the run dir is finalized.
BATCH1_POSTPROCESS = f"{sys.executable} {POSTPROCESS_SCRIPT} --run-dir {{run_dir}}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_arguments(parser, default_log_root=SYSID_LOG_ROOT, default_run_name=SYSID_RUN_NAME)
    parser.set_defaults(
        pass_marker=[BATCH1_PASS_MARKER],
        fail_marker=[BATCH1_FAIL_MARKER],
        arm_command=list(BATCH1_ARM_COMMANDS),
        run_command=list(BATCH1_RUN_COMMANDS),
        upload_env=BATCH1_UPLOAD_ENV,
        postprocess=BATCH1_POSTPROCESS,
    )
    parser.add_argument(
        "--auto-run", action="store_true", help="Send ARM FINN and RUN BATCH1 after startup delay."
    )
    args = parser.parse_args()
    if args.auto_run and not args.auto_command:
        args.auto_command = list(BATCH1_AUTO_COMMANDS)
    return args


def main() -> int:
    return run_capture(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
