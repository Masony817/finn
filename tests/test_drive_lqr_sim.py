"""Unit tests for the command layer above Finn's balance loop.

Exercises command shaping and the simulation viewer's keyboard adapter.
The hardware-independent control module is also tested against native C++.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import pytest

from finn import control, reporting
from finn import simulation as rls

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


drive = _load("drive_lqr_sim", "drive_lqr_sim.py")

LIMITS = control.DriveLimits(
    max_forward_vel_m_s=0.6,
    max_yaw_rate_rad_s=1.0,
    forward_accel_limit_m_s2=0.5,
    yaw_accel_limit_rad_s2=3.0,
    command_timeout_s=0.5,
    reference_position_band_m=0.3,
)
DT = 0.01


# --- allocation: balance keeps torque priority (invariant 5) ----------------


def test_yaw_only_gets_the_torque_balance_left_behind():
    balance, yaw = control.allocate_wheel_torques(0.4, 0.9, 1.0)

    assert balance == pytest.approx(0.4)
    assert yaw == pytest.approx(0.6)


def test_a_saturated_balance_loop_cannot_steer():
    balance, yaw = control.allocate_wheel_torques(2.0, 5.0, 1.0)

    assert balance == pytest.approx(1.0)
    assert yaw == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("tau_balance", "tau_yaw"),
    [(0.0, 10.0), (0.5, -10.0), (-1.5, 3.0), (0.999, 0.5), (-0.2, -0.7)],
)
def test_no_allocation_can_put_a_wheel_outside_the_envelope(tau_balance, tau_yaw):
    limit = 1.0
    balance, yaw = control.allocate_wheel_torques(tau_balance, tau_yaw, limit)
    left, right = rls.yaw_torque_to_wheels(yaw, -1.0)

    assert abs(left + balance) <= limit + 1e-12
    assert abs(right + balance) <= limit + 1e-12


def test_yaw_torque_split_is_equal_and_opposite():
    left, right = rls.yaw_torque_to_wheels(0.25, -1.0)

    assert left == pytest.approx(-0.25)
    assert right == pytest.approx(0.25)


# --- arbiter: bounded, rate limited, and fails closed (invariants 3 and 4) ---


def _arbiter():
    return control.CommandArbiter(limits=LIMITS)


def test_no_command_source_is_a_standing_stop():
    arbiter = _arbiter()

    for tick in range(50):
        command = arbiter.step(None, tick * DT, DT)

    assert command == control.DriveCommand(0.0, 0.0)


def test_a_wild_command_is_clamped_into_the_envelope():
    arbiter = _arbiter()
    source = lambda _t: control.DriveCommand(1e6, -1e6)  # noqa: E731

    for tick in range(2000):
        command = arbiter.step(source, tick * DT, DT)

    assert command.forward_vel_m_s == pytest.approx(LIMITS.max_forward_vel_m_s)
    assert command.yaw_rate_rad_s == pytest.approx(-LIMITS.max_yaw_rate_rad_s)


def test_the_command_ramps_rather_than_stepping():
    arbiter = _arbiter()
    source = lambda _t: control.DriveCommand(0.6, 0.0)  # noqa: E731

    first = arbiter.step(source, 0.0, DT)

    assert first.forward_vel_m_s == pytest.approx(LIMITS.forward_accel_limit_m_s2 * DT)


def test_a_source_that_goes_silent_is_held_then_ramped_down():
    """Invariant 4: a jittery 10 Hz policy is normal, a stopped one is not."""

    arbiter = _arbiter()
    live = lambda _t: control.DriveCommand(0.3, 0.0)  # noqa: E731

    time_s = 0.0
    while time_s < 5.0:
        arbiter.step(live, time_s, DT)
        time_s += DT
    cruising = arbiter.shaped.forward_vel_m_s
    assert cruising == pytest.approx(0.3, abs=1e-9)

    silent = lambda _t: None  # noqa: E731
    held = arbiter.step(silent, time_s + 0.2, DT)
    assert held.forward_vel_m_s == pytest.approx(cruising), "a dropped sample must be held"
    assert not arbiter.stale

    time_s += 1.0
    while arbiter.shaped.forward_vel_m_s > 1e-9:
        previous = arbiter.shaped.forward_vel_m_s
        step = arbiter.step(silent, time_s, DT)
        assert previous - step.forward_vel_m_s <= LIMITS.forward_accel_limit_m_s2 * DT + 1e-12
        time_s += DT

    assert arbiter.stale
    assert arbiter.shaped.forward_vel_m_s == pytest.approx(0.0)


@pytest.mark.parametrize(
    "source",
    [
        lambda _t: (_ for _ in ()).throw(RuntimeError("policy crashed")),
        lambda _t: control.DriveCommand(float("nan"), 0.0),
        lambda _t: control.DriveCommand(0.0, float("inf")),
        lambda _t: (1.0, 2.0),
        lambda _t: None,
    ],
)
def test_an_unusable_source_never_escapes_the_arbiter(source):
    arbiter = _arbiter()

    for tick in range(200):
        command = arbiter.step(source, tick * DT, DT)
        assert command.is_finite()

    assert command == control.DriveCommand(0.0, 0.0)
    assert arbiter.rejected_samples == 200


# --- keyboard latch: press events have to stand in for held keys -------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _keyboard(clock, hold_window_s=0.6):
    return drive.KeyboardDrive(
        hold_window_s=hold_window_s,
        max_forward_vel_m_s=LIMITS.max_forward_vel_m_s,
        max_yaw_rate_rad_s=LIMITS.max_yaw_rate_rad_s,
        clock=clock,
    )


def test_no_keys_means_stop():
    assert _keyboard(FakeClock()).command(0.0) == control.DriveCommand(0.0, 0.0)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (drive.KEY_UP, control.DriveCommand(0.6, 0.0)),
        (drive.KEY_DOWN, control.DriveCommand(-0.6, 0.0)),
        (drive.KEY_LEFT, control.DriveCommand(0.0, 1.0)),
        (drive.KEY_RIGHT, control.DriveCommand(0.0, -1.0)),
    ],
)
def test_each_key_drives_its_own_axis(key, expected):
    keyboard = _keyboard(FakeClock())
    keyboard.on_key(key)

    assert keyboard.command(0.0) == expected


def test_opposing_keys_cancel():
    keyboard = _keyboard(FakeClock())
    keyboard.on_key(drive.KEY_UP)
    keyboard.on_key(drive.KEY_DOWN)

    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)


def test_a_key_stays_held_across_the_repeat_gap_then_expires():
    clock = FakeClock()
    keyboard = _keyboard(clock, hold_window_s=0.6)
    keyboard.on_key(drive.KEY_UP)

    clock.now = 0.5  # inside a typical OS key-repeat delay
    assert keyboard.command(0.0).forward_vel_m_s == pytest.approx(0.6)

    clock.now = 0.7
    assert keyboard.command(0.0).forward_vel_m_s == pytest.approx(0.0)


def test_space_stops_immediately():
    clock = FakeClock()
    keyboard = _keyboard(clock)
    keyboard.on_key(drive.KEY_UP)
    keyboard.on_key(drive.KEY_LEFT)
    keyboard.on_key(drive.KEY_SPACE)

    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)


def test_unrelated_keys_are_ignored():
    keyboard = _keyboard(FakeClock())
    keyboard.on_key(ord("Q"))

    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)


# --- scripted profiles: the template a future policy tenant copies -----------


def test_scripted_profiles_only_ever_emit_bounded_commands():
    for name in ("square", "spin"):
        source = drive.scripted_profile(name, max_forward_vel_m_s=0.6, max_yaw_rate_rad_s=1.0)
        for tick in range(3000):
            command = source(tick * DT)
            assert isinstance(command, control.DriveCommand)
            assert abs(command.forward_vel_m_s) <= 0.6
            assert abs(command.yaw_rate_rad_s) <= 1.0


def test_the_none_profile_hands_back_no_source():
    assert drive.scripted_profile("none", max_forward_vel_m_s=0.6, max_yaw_rate_rad_s=1.0) is None


def test_an_unknown_profile_is_a_named_error():
    with pytest.raises(drive.DriveError):
        drive.scripted_profile("barrel-roll", max_forward_vel_m_s=0.6, max_yaw_rate_rad_s=1.0)


# --- viewer key collisions ---------------------------------------------------
#
# The MuJoCo viewer runs its own key handling alongside the callback it is given,
# and it has a shortcut on every letter A-Z. W toggled wireframe on every step
# forward until this was noticed. These are rendering flags, so control was never
# affected, but the scene strobed while driving.


def _viewer_shortcut_keys() -> dict[str, str]:
    keys = {}
    for index in range(mujoco.mjtVisFlag.mjNVISFLAG):
        name, _, key = mujoco.mjVISSTRING[index]
        if key:
            keys[key.upper()] = f"vis:{name}"
    for index in range(mujoco.mjtRndFlag.mjNRNDFLAG):
        name, _, key = mujoco.mjRNDSTRING[index]
        if key:
            keys[key.upper()] = f"rnd:{name}"
    return keys


def test_arrow_keys_collide_with_nothing():
    """Arrows are the clean scheme, which is why they are the recommended one."""

    shortcuts = _viewer_shortcut_keys()
    arrow_codes = {drive.KEY_UP, drive.KEY_DOWN, drive.KEY_LEFT, drive.KEY_RIGHT}

    assert arrow_codes <= set(drive.DRIVE_KEYS)
    # Every MuJoCo shortcut is a single printable character, so arrows cannot be one.
    assert all(len(key) == 1 and key.isprintable() for key in shortcuts)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (drive.KEY_UP, control.DriveCommand(0.6, 0.0)),
        (drive.KEY_DOWN, control.DriveCommand(-0.6, 0.0)),
        (drive.KEY_LEFT, control.DriveCommand(0.0, 1.0)),
        (drive.KEY_RIGHT, control.DriveCommand(0.0, -1.0)),
    ],
)
def test_arrows_drive_the_expected_axes(key, expected):
    keyboard = _keyboard(FakeClock())
    keyboard.on_key(key)

    assert keyboard.command(0.0) == expected


# --- tracking gates must not grade a robot that was standing still -----------


def _row(time_s, cmd_v, vel):
    return {
        "time_s": time_s,
        "cmd_forward_vel_m_s": cmd_v,
        "forward_vel_m_s": vel,
    }


def test_a_stop_command_is_never_scored_as_tracking():
    """The bug this pins: 1186 idle samples once produced a confident pass.

    A run where the robot was asked to stand still and did says nothing about how
    well it follows a command, so those samples must not reach the gate.
    """

    rows = [_row(i * 0.01, 0.0, 0.0) for i in range(500)]

    p95, worst, scored = reporting.settled_tracking_error(
        rows, "cmd_forward_vel_m_s", "forward_vel_m_s", 150
    )

    assert scored == 0
    assert (p95, worst) == (0.0, 0.0)


def test_a_steady_nonzero_command_is_scored():
    rows = [_row(i * 0.01, 0.4, 0.35) for i in range(500)]

    p95, worst, scored = reporting.settled_tracking_error(
        rows, "cmd_forward_vel_m_s", "forward_vel_m_s", 150
    )

    assert scored == 350
    assert p95 == pytest.approx(0.05, abs=1e-9)
    assert worst == pytest.approx(0.05, abs=1e-9)


def test_a_command_that_never_settles_is_not_scored():
    """Human driving ramps constantly, so a settle window can find nothing."""

    rows = [_row(i * 0.01, 0.005 * i, 0.0) for i in range(500)]

    _, _, scored = reporting.settled_tracking_error(
        rows, "cmd_forward_vel_m_s", "forward_vel_m_s", 150
    )

    assert scored == 0


def test_too_few_scored_samples_is_reported_rather_than_passed_over():
    assert reporting.MIN_TRACKING_SAMPLES > 0

    metrics = {
        "velocity_tracking_assessed": False,
        "velocity_tracking_samples": 0,
        "velocity_tracking_p95_m_s": 0.0,
        "velocity_tracking_max_m_s": 0.0,
    }

    text = drive._tracking_text("vel", metrics, "velocity", "m_s")

    assert "not assessed" in text
    assert "0.00000" not in text, "an unmeasured gate must not print a reassuring number"


# --- hold to drive -----------------------------------------------------------
#
# The viewer's callback carries a keycode and nothing else: no releases, no
# press/repeat distinction. The latch below is what a callback alone can manage,
# and it is a poor joystick. Real held/released state comes from GLFW once a
# keypress hands over the window, and these pin the handover both ways.


class FakeGlfw:
    """Stands in for the GLFW module, with a key state we control."""

    RELEASE = 0
    PRESS = 1

    def __init__(self, window="window", down=()):
        self.window = window
        self.down = set(down)
        self.get_key_calls = 0

    def get_current_context(self):
        return self.window

    def get_key(self, window, keycode):
        assert window is self.window
        self.get_key_calls += 1
        return self.PRESS if keycode in self.down else self.RELEASE


@pytest.fixture
def fake_glfw(monkeypatch):
    fake = FakeGlfw()
    monkeypatch.setattr(drive, "glfw", fake)
    return fake


def test_a_held_key_keeps_driving_long_past_the_latch_window(fake_glfw):
    """The whole point: no repeated presses, and no expiry while the key is down."""

    clock = FakeClock()
    keyboard = _keyboard(clock, hold_window_s=0.6)
    fake_glfw.down = {drive.KEY_UP}
    keyboard.on_key(drive.KEY_UP)

    assert keyboard.polling_key_state

    clock.now = 60.0  # a hundred latch windows later
    assert keyboard.command(0.0).forward_vel_m_s == pytest.approx(0.6)


def test_releasing_a_key_stops_without_waiting_out_the_latch(fake_glfw):
    clock = FakeClock()
    keyboard = _keyboard(clock)
    fake_glfw.down = {drive.KEY_UP}
    keyboard.on_key(drive.KEY_UP)

    fake_glfw.down = set()
    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)


def test_holding_two_keys_drives_and_turns_at_once(fake_glfw):
    keyboard = _keyboard(FakeClock())
    fake_glfw.down = {drive.KEY_UP, drive.KEY_LEFT}
    keyboard.on_key(drive.KEY_UP)

    assert keyboard.command(0.0) == control.DriveCommand(0.6, 1.0)


def test_space_stops_even_while_a_drive_key_is_held(fake_glfw):
    keyboard = _keyboard(FakeClock())
    fake_glfw.down = {drive.KEY_UP, drive.KEY_SPACE}
    keyboard.on_key(drive.KEY_UP)

    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)


def test_state_polling_only_starts_after_a_key_hands_over_the_window(fake_glfw):
    keyboard = _keyboard(FakeClock())

    assert not keyboard.polling_key_state
    fake_glfw.down = {drive.KEY_UP}
    # Until a key event arrives there is no window, so a held key reads as idle.
    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)

    keyboard.on_key(drive.KEY_UP)
    assert keyboard.polling_key_state


def test_a_glfw_failure_falls_back_to_the_latch_instead_of_ending_the_run(fake_glfw):
    clock = FakeClock()
    keyboard = _keyboard(clock, hold_window_s=0.6)
    fake_glfw.down = {drive.KEY_UP}
    keyboard.on_key(drive.KEY_UP)

    def explode(window, keycode):
        raise RuntimeError("window destroyed")

    fake_glfw.get_key = explode

    # The latch still remembers the press, so driving degrades rather than breaks.
    assert keyboard.command(0.0).forward_vel_m_s == pytest.approx(0.6)
    assert not keyboard.polling_key_state
    clock.now = 5.0
    assert keyboard.command(0.0) == control.DriveCommand(0.0, 0.0)


def test_polling_can_be_turned_off():
    keyboard = drive.KeyboardDrive(
        hold_window_s=0.6,
        max_forward_vel_m_s=0.6,
        max_yaw_rate_rad_s=1.0,
        clock=FakeClock(),
        poll_key_state=False,
    )
    keyboard.on_key(drive.KEY_UP)

    assert not keyboard.polling_key_state
