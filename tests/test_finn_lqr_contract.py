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
