#!/usr/bin/env python3
"""Capture Finn Batch 2 loaded ground-contact sysid serial logs in realtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from serial_log_capture import add_capture_arguments, run_capture

SYSID_LOG_ROOT = Path("logs/finn-mcu/sysid")
SYSID_RUN_NAME = "batch_2"
BATCH2_PASS_MARKER = ",batch2_complete,"
BATCH2_FAIL_MARKER = ",failsafe,"
BATCH2_AUTO_COMMANDS = ("ARM FINN", "RUN BATCH2")
BATCH2_ARM_COMMANDS = ("ARM FINN",)
BATCH2_RUN_COMMANDS = ("RUN BATCH2",)
BATCH2_UPLOAD_ENV = "sysid_batch2_loaded_ground"
POSTPROCESS_SCRIPT = Path(__file__).resolve().parent / "postprocess_sysid_batch2.py"
MEASUREMENTS_CONFIG = (
    Path(__file__).resolve().parents[3] / "sim" / "config" / "finn_measurements.yaml"
)
# {run_dir} is substituted by serial_log_capture after the run dir is finalized.
BATCH2_POSTPROCESS = (
    f"{sys.executable} {POSTPROCESS_SCRIPT} --run-dir {{run_dir}} "
    f"--measurements {MEASUREMENTS_CONFIG}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_arguments(parser, default_log_root=SYSID_LOG_ROOT, default_run_name=SYSID_RUN_NAME)
    parser.set_defaults(
        pass_marker=[BATCH2_PASS_MARKER],
        fail_marker=[BATCH2_FAIL_MARKER],
        arm_command=list(BATCH2_ARM_COMMANDS),
        run_command=list(BATCH2_RUN_COMMANDS),
        upload_env=BATCH2_UPLOAD_ENV,
        postprocess=BATCH2_POSTPROCESS,
    )
    parser.add_argument(
        "--auto-run", action="store_true", help="Send ARM FINN and RUN BATCH2 after startup delay."
    )
    args = parser.parse_args()
    if args.auto_run and not args.auto_command:
        args.auto_command = list(BATCH2_AUTO_COMMANDS)
    return args


def main() -> int:
    return run_capture(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
