"""Static contracts that must agree before Finn can run LQR on hardware."""

from __future__ import annotations

import configparser
import hashlib
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONVENTIONS = ROOT / "config" / "finn_conventions.yaml"
MODEL = ROOT / "sim" / "generated" / "seeded" / "latest" / "finn.seeded.sim.xml"
HEADER = ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance" / "lqr_seeded_config.h"
FIRMWARE = ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance" / "main.cpp"
SAFETY_HEADER = (
    ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance" / "lqr_safety_config.h"
)
PLATFORMIO = ROOT / "firmware" / "finn-mcu" / "platformio.ini"


def header_value(name: str, text: str) -> str:
    match = re.search(rf"{re.escape(name)}\s*=\s*([^;]+);", text)
    assert match, f"missing {name} from generated firmware header"
    return match.group(1).strip()


def test_cppcheck_targets_the_same_sketch_as_each_firmware_build():
    config = configparser.ConfigParser()
    config.read(PLATFORMIO)
    root = PLATFORMIO.parent
    for section in config.sections():
        if not section.startswith("env:"):
            continue
        build = re.search(r"\+<([^>]+)>", config[section]["build_src_filter"]).group(1)
        check = re.search(r"\+<([^>]+)>", config[section]["check_src_filters"]).group(1)
        assert (root / "src" / build).resolve() == (root / check).resolve()
        assert (root / check / "main.cpp").is_file()


def test_model_actuator_gears_match_the_convention_contract():
    conventions = yaml.safe_load(CONVENTIONS.read_text(encoding="utf-8"))
    actuators = {
        motor.attrib["name"]: float(motor.attrib["gear"])
        for motor in ET.parse(MODEL).findall(".//actuator/motor")
    }

    assert actuators["motor_left_wheel"] == conventions["actuation"]["sim_left_actuator_gear"]
    assert actuators["motor_right_wheel"] == conventions["actuation"]["sim_right_actuator_gear"]


def test_generated_header_tracks_model_and_unverified_hardware_signs():
    conventions = yaml.safe_load(CONVENTIONS.read_text(encoding="utf-8"))
    header = HEADER.read_text(encoding="utf-8")
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:12]

    assert header_value("kModelSha256[]", header).strip('"') == model_hash
    assert (
        header_value("kPitchDirectionBenchVerified", header)
        == str(conventions["imu"]["pitch_direction_bench_verified"]).lower()
    )
    assert (
        header_value("kWheelEncoderDirectionsBenchVerified", header)
        == str(conventions["wheel_odometry"]["encoder_directions_bench_verified"]).lower()
    )


def test_real_lqr_environment_keeps_the_reviewed_safety_interlocks():
    platformio = PLATFORMIO.read_text(encoding="utf-8")
    firmware = FIRMWARE.read_text(encoding="utf-8")

    assert "[env:lqr_balance]" in platformio
    assert 'strcmp(line, "STOP")' in firmware
    assert 'latchFault("host_heartbeat_timeout")' in firmware
    assert 'latchFault("control_deadline_missed")' in firmware
    assert "kWheelEncoderDirectionsBenchVerified" in firmware
    assert 'printEvent("lqr_complete", reason)' in firmware


def test_generated_header_tracks_the_yaw_sign_contract():
    conventions = yaml.safe_load(CONVENTIONS.read_text(encoding="utf-8"))
    header = HEADER.read_text(encoding="utf-8")

    assert header_value("kYawSign", header) == f"{float(conventions['imu']['yaw_sign'])}f"
    assert (
        header_value("kRealLeftActuatorYawSign", header)
        == f"{float(conventions['yaw']['real_left_actuator_yaw_sign'])}f"
    )
    assert (
        header_value("kYawDirectionBenchVerified", header)
        == str(conventions["imu"]["yaw_direction_bench_verified"]).lower()
    )


def test_the_drive_envelope_reaches_the_firmware_header():
    """Steering constants have to travel the same generated path as the gain.

    Firmware consumes the generated values rather than tuning them separately.
    """

    header = HEADER.read_text(encoding="utf-8")

    for name in (
        "kGainYawRate",
        "kMaxForwardVelMS",
        "kMaxYawRateRadS",
        "kDriveAccelLimitMS2",
        "kDriveYawAccelLimitRadS2",
        "kCommandTimeoutMs",
        "kRefPositionBandM",
    ):
        header_value(name, header)


def test_the_command_timeout_absorbs_dropped_messages_at_the_host_command_rate():
    """The command timeout and the heartbeat fault cover different failures.

    Losing the serial link is the heartbeat's job, and at 300 ms it cuts the
    motors long before any ramp could matter. What the command timeout covers is
    the link staying healthy while whatever produces commands -- teleop, or later
    a policy -- stalls. That has room to degrade gracefully, so the timeout is
    sized to ride out a few missed messages at the 10 Hz host command rate and
    then ramp the drive to zero with the robot still balancing.
    """

    header = HEADER.read_text(encoding="utf-8")
    command_timeout_ms = float(header_value("kCommandTimeoutMs", header).rstrip("UL"))
    host_command_period_ms = 100.0

    assert command_timeout_ms >= 3 * host_command_period_ms
    assert command_timeout_ms <= 10 * host_command_period_ms


def safety_value(name: str, _depth: int = 0) -> float:
    """Evaluate a reviewed limit, following degree conversions and cross-references.

    The limits are written the way a reviewer reads them -- `10.0f * PI / 180.0`,
    or one constant defined as another -- so comparing them needs the expression,
    not the literal.
    """

    assert _depth < 8, f"cyclic constant reference resolving {name}"
    expression = header_value(name, SAFETY_HEADER.read_text(encoding="utf-8"))
    expression = re.sub(r"(?<=[\d.])[fFuUlL]+\b", "", expression)
    expression = expression.replace("PI", repr(math.pi))
    for reference in sorted(set(re.findall(r"\bk[A-Za-z]\w*", expression)), key=len, reverse=True):
        expression = expression.replace(reference, repr(safety_value(reference, _depth + 1)))
    return float(eval(expression, {"__builtins__": {}}, {}))


def test_arming_runs_the_preflight_rather_than_arming_directly():
    """ARM FINN must not be a shortcut past the checks it exists to run."""

    firmware = FIRMWARE.read_text(encoding="utf-8")

    assert 'strcmp(line, "ARM FINN") == 0' in firmware
    assert "startPreflight(true)" in firmware
    assert 'strcmp(line, "PREFLIGHT") == 0' in firmware
    assert "startPreflight(false)" in firmware
    # Arming is reached only by the preflight evaluator, never by a command branch.
    assert firmware.count("SystemState::kArmedIdle;") == 1


def test_every_bench_verified_flag_is_a_named_preflight_check():
    firmware = FIRMWARE.read_text(encoding="utf-8")

    assert '"conventions_pitch", FinnLqrSeeded::kPitchDirectionBenchVerified' in firmware
    assert '"conventions_wheels", FinnLqrSeeded::kWheelEncoderDirectionsBenchVerified' in firmware


def test_yaw_torque_is_zero_while_the_drive_layer_is_gated():
    """The command layer ships built but inert; balance must be bit-identical to the
    validated station keeper until driving is deliberately enabled."""

    firmware = FIRMWARE.read_text(encoding="utf-8")

    assert "FinnLqrSafety::kDriveEnabled" in firmware
    assert "? -FinnLqrSeeded::kGainYawRate * (yawRateRadS() - arbiter.yaw_rad_s)" in firmware
    assert ": 0.0f;" in firmware
    assert 'printEvent("drive_rejected", "drive_layer_gated_see_kDriveEnabled")' in firmware


def test_the_telemetry_applies_the_yaw_sign_contract_like_the_pitch_sign():
    """kYawSign is exported by the generator; a column that ignores it would make
    Scopik and the controller disagree the moment the bench check inverts yaw."""

    firmware = FIRMWARE.read_text(encoding="utf-8")

    assert "float yawRateRadS() {\n  return FinnLqrSeeded::kYawSign * imu.gyro_y_rad_s;" in firmware
    assert "static_cast<double>(yawRateRadS())" in firmware


def test_operator_selectable_trial_length_cannot_exceed_the_reviewed_one():
    assert safety_value("kMaxTrialDurationMs") <= safety_value("kFirstTrialDurationMs")
    assert safety_value("kMinTrialDurationMs") <= safety_value("kMaxTrialDurationMs")
    assert "kMaxTrialDurationMs" in FIRMWARE.read_text(encoding="utf-8")


def test_preflight_limits_sit_inside_the_limits_that_fault_a_running_trial():
    """A preflight that admits a state the run would immediately fault on is not a
    check, it is a way to arm into a fault."""

    assert safety_value("kMaxPreflightTempC") < safety_value("kMaxMoteusTempC")
    assert safety_value("kMaxStartPitchErrorRad") < safety_value("kMaxAbsPitchRad")
    assert safety_value("kMaxStartWheelSpeedRevS") < safety_value("kMaxWheelSpeedRevS")


def test_the_control_budget_fits_inside_one_control_period():
    header = HEADER.read_text(encoding="utf-8")
    period_us = float(header_value("kControlPeriodUs", header).rstrip("UL"))

    assert safety_value("kControlBudgetUs") < period_us
