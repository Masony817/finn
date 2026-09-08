"""Pure command-layer behavior, without importing MuJoCo."""

import subprocess
import sys

import pytest

from finn.control import CommandArbiter, DriveCommand, DriveLimits


@pytest.mark.parametrize("value", ["bad", None, object(), float("nan"), float("inf")])
def test_invalid_command_fields_do_not_escape_the_arbiter(value):
    arbiter = CommandArbiter(DriveLimits())
    command = arbiter.step(lambda _: DriveCommand(value, 0.0), 0.0, 0.01)
    assert command == DriveCommand()
    assert arbiter.rejected_samples == 1


def test_intermittent_commands_keep_ramping_toward_the_requested_target():
    arbiter = CommandArbiter(DriveLimits())
    first = arbiter.step(lambda _: DriveCommand(0.3, 0.0), 0.0, 0.01)
    second = arbiter.step(lambda _: None, 0.01, 0.01)
    assert first.forward_vel_m_s == pytest.approx(0.005)
    assert second.forward_vel_m_s == pytest.approx(0.010)
    assert not arbiter.stale


def test_source_callbacks_are_synchronous_not_background_workers():
    calls = []
    arbiter = CommandArbiter(DriveLimits())
    arbiter.step(lambda t: calls.append(t) or DriveCommand(), 0.1, 0.01)
    assert calls == [0.1]


def test_control_import_does_not_load_simulator_or_visualization():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import finn.control, sys; assert 'mujoco' not in sys.modules; "
            "assert 'rerun' not in sys.modules",
        ],
        check=True,
    )
