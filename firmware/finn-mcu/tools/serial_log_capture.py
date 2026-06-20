#!/usr/bin/env python3
"""Reusable realtime serial logger for Finn MCU tools."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

TOOL_DIR = Path(__file__).resolve().parent
FIRMWARE_ROOT = TOOL_DIR.parent
REPO_ROOT = FIRMWARE_ROOT.parents[1]

DEFAULT_BAUD = 115200
DEFAULT_LOG_ROOT = Path("logs/finn-mcu/serial")
DEFAULT_RUN_NAME = "teensy_log"
DEFAULT_RAW_FILENAME = "raw.log"
DEFAULT_TELEMETRY_FILENAME = "telemetry.csv"
DEFAULT_EVENTS_FILENAME = "events.log"
DEFAULT_OPERATOR_INPUT_FILENAME = "operator_input.log"
DEFAULT_TELEMETRY_PREFIXES = ("schema,", "data,")
DEFAULT_EVENT_PREFIXES = ("event,", "status,")

BAUD_MAP = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: getattr(termios, "B230400", termios.B115200),
    460800: getattr(termios, "B460800", termios.B115200),
    921600: getattr(termios, "B921600", termios.B115200),
}


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def print_host(message: str) -> None:
    print(f"[HOST] {message}")


def find_serial_port() -> str:
    by_id = Path("/dev/serial/by-id")
    if by_id.exists():
        by_id_candidates = sorted(by_id.glob("*"))
        if len(by_id_candidates) == 1:
            return str(by_id_candidates[0])
        if len(by_id_candidates) > 1:
            options = "\n  ".join(str(candidate) for candidate in by_id_candidates)
            raise SystemExit(f"Multiple serial ports found. Pass --port explicitly:\n  {options}")

    candidates: list[Path] = []
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*", "/dev/cu.usbmodem*", "/dev/cu.usbserial*"):
        candidates.extend(sorted(Path("/").glob(pattern.lstrip("/"))))

    unique = [str(candidate) for candidate in candidates]
    if not unique:
        raise SystemExit("No serial port found. Pass --port explicitly, e.g. /dev/ttyACM0.")
    if len(unique) > 1:
        options = "\n  ".join(unique)
        raise SystemExit(f"Multiple serial ports found. Pass --port explicitly:\n  {options}")
    return unique[0]


def wait_for_serial_port(timeout_s: float = 10.0, interval_s: float = 0.25) -> str:
    """Poll for a serial port. Flashing reboots the Teensy, so the USB CDC device
    disappears and re-enumerates; opening immediately can fail or hit a stale node."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return find_serial_port()
        except SystemExit:
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval_s)


def configure_serial(fd: int, baud: int) -> None:
    if baud not in BAUD_MAP:
        supported = ", ".join(str(value) for value in sorted(BAUD_MAP))
        raise SystemExit(f"Unsupported baud {baud}. Supported: {supported}")

    tty.setraw(fd)
    attrs = termios.tcgetattr(fd)
    attrs[4] = BAUD_MAP[baud]
    attrs[5] = BAUD_MAP[baud]
    attrs[2] |= termios.CLOCAL | termios.CREAD
    if hasattr(termios, "CRTSCTS"):
        attrs[2] &= ~termios.CRTSCTS
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def prefixed_dir(log_root: Path, run_name: str, status: str, stamp: str) -> Path:
    return log_root / f"{run_name}_{status}" / stamp


def make_pending_dir(log_root: Path, run_name: str, stamp: str) -> Path:
    pending_dir = prefixed_dir(log_root, run_name, "pending", stamp)
    pending_dir.mkdir(parents=True, exist_ok=False)
    return pending_dir


def final_dir_for(log_root: Path, run_name: str, status: str, stamp: str) -> Path:
    base = prefixed_dir(log_root, run_name, status, stamp)
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = prefixed_dir(log_root, run_name, status, f"{stamp}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not choose a unique final log directory")


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def send_command(fd: int, command: str) -> None:
    os.write(fd, command.rstrip("\r\n").encode("utf-8") + b"\n")


def find_platformio_executable() -> str | None:
    for name in ("pio", "platformio"):
        resolved = shutil.which(name)
        if resolved:
            return resolved

    for path in (
        REPO_ROOT / ".venv" / "bin" / "pio",
        REPO_ROOT / ".venv" / "bin" / "platformio",
        Path.home() / ".platformio" / "penv" / "bin" / "pio",
        Path.home() / ".platformio" / "penv" / "bin" / "platformio",
    ):
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def upload_firmware(env: str, port: str | None = None) -> int:
    pio = find_platformio_executable()
    if not pio:
        print(
            "PlatformIO executable not found. Install PlatformIO or add `pio`/`platformio` to PATH.",
            file=sys.stderr,
        )
        return 127

    cmd = [pio, "run", "-e", env, "-t", "upload"]
    if port:
        cmd.extend(["--upload-port", port])
    print_host(f"uploading firmware: {' '.join(cmd)}")
    command_env = os.environ.copy()
    command_env.setdefault("PLATFORMIO_CORE_DIR", str(REPO_ROOT / ".platformio"))
    completed = subprocess.run(cmd, cwd=FIRMWARE_ROOT, env=command_env, check=False)
    if completed.returncode == 0:
        print_host(f"uploaded firmware env={env}")
    else:
        print_host(f"upload failed env={env} returncode={completed.returncode}")
    return completed.returncode


def run_postprocess(command: list[str], run_dir: Path) -> tuple[int, Path, Path]:
    resolved = [item.replace("{run_dir}", str(run_dir)) for item in command]
    post_dir = run_dir / "postprocess"
    post_dir.mkdir(exist_ok=True)
    print_host(f"postprocess started: {' '.join(resolved)}")
    completed = subprocess.run(resolved, text=True, capture_output=True, check=False)
    stdout_path = post_dir / "stdout.log"
    stderr_path = post_dir / "stderr.log"
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print_host(f"postprocess complete exit_code={completed.returncode}")
    return completed.returncode, stdout_path, stderr_path


def starts_with_any(line: str, prefixes: Iterable[str]) -> bool:
    return any(line.startswith(prefix) for prefix in prefixes)


def contains_any(line: str, markers: Iterable[str]) -> bool:
    return any(marker in line for marker in markers)


def print_teensy(line: str, args: argparse.Namespace) -> None:
    if line.startswith("data,") and not args.show_telemetry:
        return
    print(f"teensy> {line}")


def write_line(
    line: str,
    raw_file: TextIO,
    telemetry_file: TextIO,
    events_file: TextIO,
    args: argparse.Namespace,
) -> None:
    raw_file.write(line + "\n")
    raw_file.flush()
    if starts_with_any(line, args.telemetry_prefix):
        telemetry_file.write(line + "\n")
        telemetry_file.flush()
    elif starts_with_any(line, args.event_prefix):
        events_file.write(line + "\n")
        events_file.flush()


def write_event_line(line: str, events_file: TextIO) -> None:
    events_file.write(line + "\n")
    events_file.flush()


def write_operator_input(command: str, operator_file: TextIO) -> None:
    stripped = command.strip()
    if not stripped:
        return
    operator_file.write(f"{iso_now()},{stripped}\n")
    operator_file.flush()


def classify_line(line: str, current_status: str, args: argparse.Namespace) -> str:
    if contains_any(line, args.fail_marker):
        return "fail"
    if current_status == "pending" and contains_any(line, args.pass_marker):
        return "pass"
    return current_status


def line_has_run_marker(line: str) -> bool:
    return ",segment_start," in line or (line.startswith("data,") and ",running_batch1," in line)


def terminal_status_line(status: str) -> str:
    if status == "pass":
        return "PASS"
    if status == "fail":
        return "FAIL"
    if status == "aborted":
        return "ABORTED"
    return status.upper()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.pass_marker = args.pass_marker or []
    args.fail_marker = args.fail_marker or []
    args.auto_command = args.auto_command or []
    args.telemetry_prefix = args.telemetry_prefix or list(DEFAULT_TELEMETRY_PREFIXES)
    args.event_prefix = args.event_prefix or list(DEFAULT_EVENT_PREFIXES)
    args.arm_command = args.arm_command or []
    args.run_command = args.run_command or []
    return args


def run_capture(args: argparse.Namespace) -> int:
    args = normalize_args(args)
    if args.upload_env:
        upload_port = args.port or find_serial_port()
        print_host(f"upload port detected: {upload_port}")
        upload_rc = upload_firmware(args.upload_env, upload_port)
        if upload_rc != 0:
            print(f"Upload failed (returncode {upload_rc}); aborting.", file=sys.stderr)
            return upload_rc
    if args.port:
        port = args.port
    elif args.upload_env:
        # The board just rebooted from flashing; wait for it to re-enumerate.
        port = wait_for_serial_port()
    else:
        port = find_serial_port()
    stamp = args.timestamp or timestamp()
    pending_dir = make_pending_dir(args.log_root, args.run_name, stamp)

    manifest: dict[str, object] = {
        "run_name": args.run_name,
        "status": "pending",
        "started_at": iso_now(),
        "serial_port": port,
        "baud": args.baud,
        "pass_markers": args.pass_marker,
        "fail_markers": args.fail_marker,
        "auto_commands": args.auto_command,
        "arm_commands": args.arm_command,
        "run_commands": args.run_command,
        "pending_dir": str(pending_dir),
        "final_dir": None,
        "postprocess": None,
        "terminal_telemetry_shown": args.show_telemetry,
    }
    manifest_path = pending_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_serial(fd, args.baud)

    print_host(f"serial connected port={port} baud={args.baud}")
    print_host(f"active log dir: {pending_dir}")
    if args.show_telemetry:
        print_host("terminal telemetry display: enabled")
    else:
        print_host("terminal telemetry display: hidden; telemetry.csv still writes live")
    print_host("type commands normally, or Ctrl-C to stop capture")

    status = "pending"
    stop_requested = False
    line_buffer = bytearray()
    start_monotonic = time.monotonic()
    auto_commands_sent = False
    armed_seen = False
    run_seen = False

    def handle_signal(signum: int, frame: object) -> None:
        nonlocal stop_requested, status
        stop_requested = True
        if status == "pending":
            status = "aborted"

    old_sigint = signal.signal(signal.SIGINT, handle_signal)
    old_sigterm = signal.signal(signal.SIGTERM, handle_signal)

    try:
        with (
            (pending_dir / args.raw_filename).open("a", buffering=1) as raw_file,
            (pending_dir / args.telemetry_filename).open("a", buffering=1) as telemetry_file,
            (pending_dir / args.events_filename).open("a", buffering=1) as events_file,
            (pending_dir / args.operator_input_filename).open("a", buffering=1) as operator_file,
        ):
            while not stop_requested:
                now = time.monotonic()
                if args.timeout_s and now - start_monotonic > args.timeout_s:
                    status = args.timeout_status if status == "pending" else status
                    write_event_line("event,0,capture_timeout,host,timeout_s_elapsed", events_file)
                    print_host(f"timeout elapsed status={status}")
                    break

                if (
                    args.auto_command
                    and not auto_commands_sent
                    and now - start_monotonic >= args.auto_command_delay_s
                ):
                    for command in args.auto_command:
                        send_command(fd, command)
                        write_operator_input(command, operator_file)
                        print_host(f"sent auto command: {command}")
                        time.sleep(args.auto_command_spacing_s)
                    auto_commands_sent = True
                    write_event_line("event,0,capture_auto_command,host,commands_sent", events_file)

                read_fds = [fd]
                if sys.stdin.isatty():
                    read_fds.append(sys.stdin.fileno())
                ready, _, _ = select.select(read_fds, [], [], 0.1)

                if fd in ready:
                    try:
                        chunk = os.read(fd, 4096)
                    except OSError as exc:
                        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            chunk = b""
                        else:
                            raise
                    if not chunk:
                        continue
                    for byte in chunk:
                        if byte in (10, 13):
                            if not line_buffer:
                                continue
                            line = line_buffer.decode("utf-8", errors="replace")
                            line_buffer.clear()
                            print_teensy(line, args)
                            write_line(line, raw_file, telemetry_file, events_file, args)

                            if not armed_seen and contains_any(line, (",armed,",)):
                                armed_seen = True
                                print_host(f"armed observed; logging into {pending_dir}")
                            if not run_seen and line_has_run_marker(line):
                                run_seen = True
                                print_host("run capture active: raw.log, telemetry.csv, events.log")

                            previous_status = status
                            status = classify_line(line, status, args)
                            if status != previous_status and status in ("pass", "fail"):
                                print_host(f"{terminal_status_line(status)} marker observed")
                            if status in ("pass", "fail") and args.exit_on_terminal_event:
                                stop_requested = True
                                break
                        else:
                            line_buffer.append(byte)

                if sys.stdin.isatty() and sys.stdin.fileno() in ready:
                    command = sys.stdin.readline()
                    if command:
                        send_command(fd, command)
                        stripped = command.strip()
                        write_operator_input(stripped, operator_file)
                        if contains_any(stripped, args.arm_command):
                            armed_seen = True
                            print_host(f"ARM command forwarded; logging into {pending_dir}")
                        elif contains_any(stripped, args.run_command):
                            run_seen = True
                            print_host("RUN command forwarded; capture files are writing live")
                        elif stripped:
                            print_host(f"command forwarded: {stripped}")

            if line_buffer:
                line = line_buffer.decode("utf-8", errors="replace")
                write_line(line, raw_file, telemetry_file, events_file, args)

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        os.close(fd)

    ended_at = iso_now()
    if status == "pending":
        status = "aborted"
        print_host("capture aborted before terminal pass/fail marker")
    final_dir = final_dir_for(args.log_root, args.run_name, status, stamp)
    final_dir.parent.mkdir(parents=True, exist_ok=True)

    manifest.update(
        {
            "status": status,
            "ended_at": ended_at,
            "final_dir": str(final_dir),
        }
    )
    write_manifest(manifest_path, manifest)
    shutil.move(str(pending_dir), str(final_dir))

    print_host(f"{terminal_status_line(status)} finalized")
    print_host(f"final log dir: {final_dir}")

    if args.postprocess:
        post_rc, stdout_path, stderr_path = run_postprocess(
            shlex.split(args.postprocess), final_dir
        )
        manifest["postprocess"] = {
            "command": args.postprocess,
            "exit_code": post_rc,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        write_manifest(final_dir / "manifest.json", manifest)
        if status == "pass" and post_rc != 0:
            return 3

    return 0 if status == "pass" else 1


def add_capture_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_log_root: Path = DEFAULT_LOG_ROOT,
    default_run_name: str = DEFAULT_RUN_NAME,
) -> argparse.ArgumentParser:
    parser.add_argument("--port", help="Serial port, e.g. /dev/ttyACM0. Auto-detects if omitted.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--log-root", type=Path, default=default_log_root)
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--timestamp", help="Override timestamp folder name, useful for tests.")
    parser.add_argument(
        "--pass-marker",
        action="append",
        help="Line substring that marks the run as pass. Repeatable.",
    )
    parser.add_argument(
        "--fail-marker",
        action="append",
        help="Line substring that marks the run as fail. Repeatable.",
    )
    parser.add_argument(
        "--arm-command",
        action="append",
        help="Operator command that marks the arm lifecycle point. Repeatable.",
    )
    parser.add_argument(
        "--run-command",
        action="append",
        help="Operator command that marks the run lifecycle point. Repeatable.",
    )
    parser.add_argument(
        "--auto-command", action="append", help="Command to send after startup delay. Repeatable."
    )
    parser.add_argument("--auto-command-delay-s", type=float, default=2.0)
    parser.add_argument("--auto-command-spacing-s", type=float, default=0.1)
    parser.add_argument(
        "--timeout-s", type=float, default=0.0, help="Optional capture timeout. 0 disables."
    )
    parser.add_argument("--timeout-status", choices=("fail", "aborted"), default="fail")
    parser.add_argument("--raw-filename", default=DEFAULT_RAW_FILENAME)
    parser.add_argument("--telemetry-filename", default=DEFAULT_TELEMETRY_FILENAME)
    parser.add_argument("--events-filename", default=DEFAULT_EVENTS_FILENAME)
    parser.add_argument("--operator-input-filename", default=DEFAULT_OPERATOR_INPUT_FILENAME)
    parser.add_argument(
        "--telemetry-prefix", action="append", help="Prefix copied to telemetry file. Repeatable."
    )
    parser.add_argument(
        "--event-prefix", action="append", help="Prefix copied to events file. Repeatable."
    )
    parser.add_argument(
        "--show-telemetry",
        action="store_true",
        help="Also print high-rate data lines in the terminal. They are always logged.",
    )
    parser.add_argument(
        "--exit-on-terminal-event",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop capture once a pass/fail marker is seen.",
    )
    parser.add_argument(
        "--upload-env",
        help="PlatformIO env to flash (pio run -e ENV -t upload) before capturing.",
    )
    parser.add_argument(
        "--postprocess",
        help="Command to run after a final run dir exists. '{run_dir}' is substituted.",
    )
    return parser


def build_parser(
    *,
    description: str | None = None,
    default_log_root: Path = DEFAULT_LOG_ROOT,
    default_run_name: str = DEFAULT_RUN_NAME,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    return add_capture_arguments(
        parser, default_log_root=default_log_root, default_run_name=default_run_name
    )


def main() -> int:
    parser = build_parser()
    return run_capture(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
