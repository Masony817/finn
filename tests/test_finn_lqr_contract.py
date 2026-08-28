"""Static contracts that must agree before Finn can run LQR on hardware."""

from __future__ import annotations

import hashlib
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

    The firmware does not consume these yet. Exporting them now is what keeps the
    later port a firmware-only change instead of a second place to tune a robot.
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
