from __future__ import annotations

import copy
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "postprocess_mujoco.py"
SPEC = importlib.util.spec_from_file_location("postprocess_mujoco", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ppm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ppm
SPEC.loader.exec_module(ppm)


BASE_CONFIG = {
    "schema_version": 1,
    "pipeline": {
        "output_model": "finn_sim_test",
        "compiler_meshdir": "auto",
    },
    "root_body": {
        "source_name": "base_2",
        "target_name": "base_link",
        "freejoint_name": "root_freejoint",
        "initial_clearance_m": 0.003,
    },
    "selectors": {
        "sites": {"imu": "imu", "base": "base"},
        "wheels": {
            "left": {
                "body": "left_wheel",
                "joint": "left_wheel",
                "center_site": "left_wheel_center",
            },
            "right": {
                "body": "right_wheel",
                "joint": "right_wheel",
                "center_site": "right_wheel_center",
            },
        },
    },
    "actuators": {"motor_prefix": "motor"},
    "collisions": {
        "remove_wheel_mesh_collisions": True,
        "remove_chassis_mesh_collisions": False,
        "preserve_collision_meshes": [],
        "wheels": {"class": "collision", "material": "motor_tire_material", "condim": 6},
        "primitive_rules": [],
    },
    "sensors": {
        "imu": {
            "site": "imu",
            "gyro": "imu_gyro",
            "accelerometer": "imu_accelerometer",
            "framequat": "imu_quat",
        },
        "base": {"site": "base", "framepos": "base_pos", "framequat": "base_quat"},
        "wheel_prefix": "wheel",
    },
    "scene": {"merge_visual": True, "merge_assets": True, "merge_worldbody": True},
    "validation": {"zero_control_steps": 3},
}


def pv(value, unit="unit", source="measured"):
    return {"value": value, "unit": unit, "source": source}


MEASUREMENTS = {
    "schema_version": 1,
    "wheels": {
        "left": {
            "radius_m": pv(0.0838, "m"),
            "width_m": pv(0.066, "m"),
            "mass_kg": pv(None, "kg", "todo"),
            "torque_limit_nm": pv(3.0, "N*m"),
            "gear_ratio": pv(1.0, "ratio"),
            "command_sign": pv(1.0, "sign"),
            "damping": pv(0.001, "N*m*s/rad"),
            "armature": pv(0.0002, "kg*m^2"),
            "frictionloss": pv(0.01, "N*m"),
        },
        "right": {
            "radius_m": pv(0.0838, "m"),
            "width_m": pv(0.066, "m"),
            "mass_kg": pv(None, "kg", "todo"),
            "torque_limit_nm": pv(3.0, "N*m"),
            "gear_ratio": pv(1.0, "ratio"),
            "command_sign": pv(-1.0, "sign"),
            "damping": pv(0.001, "N*m*s/rad"),
            "armature": pv(0.0002, "kg*m^2"),
            "frictionloss": pv(0.01, "N*m"),
        },
    },
    "contact": {
        "tire": {
            "friction": pv([1.2, 0.02, 0.002], "slide torsional rolling"),
            "solref": pv([0.008, 1.0], "timeconst dampratio"),
            "solimp": pv([0.9, 0.95, 0.001, 0.5, 2.0], "dmin dmax width midpoint power"),
        }
    },
}


def write_yaml(path: Path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_inspects_current_export():
    tree = ppm.load_xml(ROOT / "sim" / "model" / "finn" / "finn_robot.xml")
    selectors = ppm.parse_wheel_selectors(BASE_CONFIG)
    inspection = ppm.inspect_robot(tree.getroot(), BASE_CONFIG, selectors)

    assert inspection.root_body_name == "base_2"
    assert set(inspection.wheel_bodies) == {"left", "right"}
    assert inspection.visual_mesh_geoms == 23
    assert inspection.collision_mesh_geoms == 23
    assert inspection.generated_position_actuators == ["left_wheel", "right_wheel"]


def test_strict_mode_fails_closed_on_todo_measurements(tmp_path):
    config_path = tmp_path / "config.yaml"
    measurements_path = tmp_path / "measurements.yaml"
    out_path = tmp_path / "finn.sim.xml"
    report_path = tmp_path / "report.md"

    bad_measurements = copy.deepcopy(MEASUREMENTS)
    bad_measurements["wheels"]["left"]["radius_m"] = pv(None, "m", "todo")
    write_yaml(config_path, BASE_CONFIG)
    write_yaml(measurements_path, bad_measurements)

    ok, report = ppm.postprocess(
        robot_path=ROOT / "sim" / "model" / "finn" / "finn_robot.xml",
        scene_path=ROOT / "sim" / "model" / "finn" / "scene.xml",
        config_path=config_path,
        measurements_path=measurements_path,
        out_path=out_path,
        report_path=report_path,
        strict=True,
        skip_mujoco_validation=True,
    )

    assert not ok
    assert not out_path.exists()
    assert "wheels.left.radius_m.value is required" in report["missing_measurements"]
    assert report_path.exists()


def test_postprocesses_xml_without_guessing_measurements(tmp_path):
    config_path = tmp_path / "config.yaml"
    measurements_path = tmp_path / "measurements.yaml"
    out_path = tmp_path / "finn.sim.xml"
    report_path = tmp_path / "report.md"
    write_yaml(config_path, BASE_CONFIG)
    write_yaml(measurements_path, MEASUREMENTS)

    ok, report = ppm.postprocess(
        robot_path=ROOT / "sim" / "model" / "finn" / "finn_robot.xml",
        scene_path=ROOT / "sim" / "model" / "finn" / "scene.xml",
        config_path=config_path,
        measurements_path=measurements_path,
        out_path=out_path,
        report_path=report_path,
        strict=True,
        skip_mujoco_validation=True,
    )

    assert ok, report["errors"]
    root = ET.parse(out_path).getroot()
    body = root.find("./worldbody/body[@name='base_link']")
    assert body is not None
    assert body.find("freejoint[@name='root_freejoint']") is not None
    assert math.isclose(float(body.get("pos").split()[2]), 0.0468, abs_tol=1e-6)

    assert root.findall("./actuator/position") == []
    motors = {motor.get("name"): motor for motor in root.findall("./actuator/motor")}
    assert set(motors) == {"motor_left_wheel", "motor_right_wheel"}
    assert motors["motor_right_wheel"].get("gear") == "-1"

    visual_meshes = [
        geom
        for geom in root.iter("geom")
        if geom.get("class") == "visual" and geom.get("type") == "mesh"
    ]
    collision_meshes = [
        geom
        for geom in root.iter("geom")
        if geom.get("class") == "collision" and geom.get("type") == "mesh"
    ]
    assert len(visual_meshes) == 23
    # Wheel mesh collisions (motor_hub + motor_tire per wheel) are replaced by
    # cylinders; the 19 chassis collision meshes stay so the body collides too.
    assert len(collision_meshes) == 19
    wheel_bodies = {"left_wheel", "right_wheel"}
    assert not any(
        body.get("name") in wheel_bodies
        for body in root.iter("body")
        for geom in body.findall("geom")
        if geom.get("class") == "collision" and geom.get("type") == "mesh"
    )
    left_tire = root.find(".//geom[@name='left_tire_collision']")
    assert left_tire.get("type") == "cylinder"
    assert left_tire.get("condim") == "6"
    assert root.find(".//geom[@name='right_tire_collision']").get("size") == "0.0838 0.033"

    sensor_names = {sensor.get("name") for sensor in root.find("sensor")}
    assert {
        "imu_gyro",
        "imu_accelerometer",
        "imu_quat",
        "base_pos",
        "base_quat",
        "wheel_left_pos",
        "wheel_left_vel",
        "wheel_right_pos",
        "wheel_right_vel",
    } <= sensor_names


def test_auto_meshdir_follows_nested_export_layout(tmp_path):
    config_path = tmp_path / "config.yaml"
    measurements_path = tmp_path / "measurements.yaml"
    out_path = tmp_path / "generated" / "seeded" / "latest" / "finn.seeded.sim.xml"
    report_path = tmp_path / "report.md"
    write_yaml(config_path, BASE_CONFIG)
    write_yaml(measurements_path, MEASUREMENTS)

    ok, report = ppm.postprocess(
        robot_path=ROOT / "sim" / "model" / "finn" / "finn_robot.xml",
        scene_path=ROOT / "sim" / "model" / "finn" / "scene.xml",
        config_path=config_path,
        measurements_path=measurements_path,
        out_path=out_path,
        report_path=report_path,
        strict=True,
        skip_mujoco_validation=True,
    )

    assert ok, report["errors"]
    root = ET.parse(out_path).getroot()
    meshdir = root.find("compiler").get("meshdir")
    resolved = (out_path.parent / meshdir).resolve()
    assert resolved == (ROOT / "sim" / "model" / "finn" / "assets").resolve()
    assert report["changes"]["scene"]["compiler_meshdir"] == meshdir


_MESH_ASSETS_PRESENT = (
    ROOT / "sim" / "model" / "finn" / "assets" / "motues_bracket.stl"
).exists()


@pytest.mark.skipif(
    not _MESH_ASSETS_PRESENT,
    reason="STL mesh assets not present — run onshape-to-robot to generate them",
)
def test_generated_xml_compiles_with_mujoco(tmp_path):
    pytest.importorskip("mujoco")
    config_path = tmp_path / "config.yaml"
    measurements_path = tmp_path / "measurements.yaml"
    out_path = tmp_path / "finn.sim.xml"
    report_path = tmp_path / "report.md"
    write_yaml(config_path, BASE_CONFIG)
    write_yaml(measurements_path, MEASUREMENTS)

    ok, report = ppm.postprocess(
        robot_path=ROOT / "sim" / "model" / "finn" / "finn_robot.xml",
        scene_path=ROOT / "sim" / "model" / "finn" / "scene.xml",
        config_path=config_path,
        measurements_path=measurements_path,
        out_path=out_path,
        report_path=report_path,
        strict=True,
        skip_mujoco_validation=False,
    )

    assert ok, report["errors"]
    assert report["validation"]["mujoco"]["status"] == "ok"
    assert report["validation"]["mujoco"]["nu"] == 2
