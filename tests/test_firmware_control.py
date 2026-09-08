"""Host-compile deterministic Finn LQR control arithmetic."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from finn.control import CommandArbiter, DriveCommand, DriveLimits

ROOT = Path(__file__).resolve().parents[1]
CONTROL_DIR = ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance"
FIXTURE = ROOT / "tests" / "firmware" / "control_math_test.cpp"


def compile_fixture(source: Path, tmp_path: Path) -> Path:
    compiler = shutil.which("c++")
    assert compiler is not None, "a C++ compiler is required for firmware control regression tests"
    executable = tmp_path / source.stem
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(CONTROL_DIR),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return executable


def test_control_math_fixture_exercises_command_and_torque_safety(tmp_path: Path):
    executable = compile_fixture(FIXTURE, tmp_path)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_python_and_firmware_agree_on_intermittent_command_traces(tmp_path: Path):
    executable = compile_fixture(FIXTURE.with_name("command_trace.cpp"), tmp_path)
    arbiter = CommandArbiter(DriveLimits())
    # No command until 50 ms: both arbiters must agree they are stale before
    # the first accepted intent, not only after it.
    commands = {
        50: (0.3, 0.8),
        100: (2.0, -2.0),
        210: (float("nan"), 0.0),
        230: (float("inf"), 0.0),
        900: (-0.3, 0.0),
    }
    trace, expected = [], []
    for now_ms in range(0, 1600, 10):
        values = commands.get(now_ms)
        forward, yaw = values or (0.0, 0.0)
        trace.append(f"{now_ms} {int(values is not None)} {forward} {yaw}")
        command = None if values is None else DriveCommand(*values)
        shaped = arbiter.step(lambda _, command=command: command, now_ms / 1000, 0.01)
        expected.append((shaped.forward_vel_m_s, shaped.yaw_rate_rad_s, int(arbiter.stale)))
    result = subprocess.run(
        [str(executable)],
        input="\n".join(trace) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    actual = [tuple(map(float, line.split())) for line in result.stdout.splitlines()]
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected, strict=True):
        assert observed == pytest.approx(wanted, abs=1e-6)
