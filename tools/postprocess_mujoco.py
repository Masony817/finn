"""Postprocess Onshape-to-Robot MuJoCo exports into a lab-grade sim model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_SOURCES = {"measured", "cad", "moteus", "estimated", "todo"}
STRICT_BLOCKED_SOURCES = {"estimated", "todo"}
TODO_STRINGS = {"", "todo", "tbd", "unknown", "null", "none"}


class PostprocessError(Exception):
    """Expected pipeline failure with reportable messages."""

    def __init__(self, messages: str | list[str]) -> None:
        self.messages = [messages] if isinstance(messages, str) else messages
        super().__init__("\n".join(self.messages))


@dataclass(frozen=True)
class WheelSelector:
    key: str
    body: str
    joint: str
    center_site: str


@dataclass(frozen=True)
class WheelValues:
    radius_m: float
    width_m: float
    torque_limit_nm: float
    gear_ratio: float
    command_sign: float
    damping: float
    armature: float
    frictionloss: float
    mass_kg: float | None = None


@dataclass(frozen=True)
class LabValues:
    wheels: dict[str, WheelValues]
    tire_friction: tuple[float, float, float]
    tire_solref: tuple[float, float]
    tire_solimp: tuple[float, float, float, float, float]


@dataclass
class Inspection:
    root_body: ET.Element
    root_body_name: str
    wheel_bodies: dict[str, ET.Element]
    wheel_joints: dict[str, ET.Element]
    wheel_center_sites: dict[str, ET.Element]
    imu_site: ET.Element
    base_site: ET.Element
    visual_mesh_geoms: int
    collision_mesh_geoms: int
    generated_position_actuators: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a standalone MuJoCo sim XML from an Onshape-to-Robot export."
    )
    parser.add_argument("--robot", type=Path, default=Path("sim/model/finn_robot.xml"))
    parser.add_argument("--scene", type=Path, default=Path("sim/model/scene.xml"))
    parser.add_argument("--config", type=Path, default=Path("sim/config/mujoco_postprocess.yaml"))
    parser.add_argument(
        "--measurements", type=Path, default=Path("sim/config/finn_measurements.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("sim/generated/finn.sim.xml"))
    parser.add_argument("--report", type=Path, default=Path("sim/generated/finn.sim.report.md"))
    parser.add_argument("--init-config", type=Path)
    parser.add_argument("--force", action="store_true", help="Overwrite --init-config output.")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument(
        "--skip-mujoco-validation",
        action="store_true",
        help="Write XML without compiling and stepping it with mujoco.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.init_config:
        try:
            init_measurements_template(args.robot, args.init_config, force=args.force)
        except PostprocessError as exc:
            for message in exc.messages:
                print(f"ERROR: {message}", file=sys.stderr)
            return 2
        print(f"wrote measurement template: {args.init_config}")
        return 0

    ok, report = postprocess(
        robot_path=args.robot,
        scene_path=args.scene,
        config_path=args.config,
        measurements_path=args.measurements,
        out_path=args.out,
        report_path=args.report,
        strict=args.strict,
        skip_mujoco_validation=args.skip_mujoco_validation,
    )
    print(f"wrote report: {args.report}")
    if ok:
        print(f"wrote sim: {args.out}")
        return 0

    for message in report.get("errors", []):
        print(f"ERROR: {message}", file=sys.stderr)
    for message in report.get("missing_measurements", []):
        print(f"ERROR: {message}", file=sys.stderr)
    return 1


def postprocess(
    *,
    robot_path: Path,
    scene_path: Path,
    config_path: Path,
    measurements_path: Path,
    out_path: Path,
    report_path: Path,
    strict: bool = True,
    skip_mujoco_validation: bool = False,
) -> tuple[bool, dict[str, Any]]:
    report = new_report(robot_path, scene_path, config_path, measurements_path, strict)

    try:
        robot_tree = load_xml(robot_path)
        scene_tree = load_xml(scene_path)
        config = load_yaml_file(config_path)
        measurements = load_yaml_file(measurements_path)

        fill_input_report(report, robot_path, scene_path, config_path, measurements_path)
        wheel_selectors = parse_wheel_selectors(config)
        inspection = inspect_robot(robot_tree.getroot(), config, wheel_selectors)
        fill_inspection_report(report, inspection, wheel_selectors)

        lab_values = validate_measurements(measurements, wheel_selectors, strict, report)

        normalize_root(robot_tree.getroot(), config, inspection, lab_values, report)
        clean_generated_physics(robot_tree.getroot(), config, inspection, wheel_selectors, report)
        apply_actuators(robot_tree.getroot(), config, inspection, lab_values, report)
        apply_collisions(robot_tree.getroot(), config, inspection, lab_values, report)
        apply_sensors(robot_tree.getroot(), config, inspection, wheel_selectors, report)
        assemble_scene(robot_tree.getroot(), scene_tree.getroot(), config, report)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = write_temp_xml(robot_tree, out_path)
        try:
            if skip_mujoco_validation:
                report["validation"]["mujoco"] = {"status": "skipped"}
            else:
                validate_mujoco(temp_path, config, wheel_selectors, report)
            temp_path.replace(out_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        report["status"] = "ok"
        write_report(report_path, report)
        return True, report
    except PostprocessError as exc:
        report["status"] = "failed"
        if report.get("missing_measurements") and exc.messages:
            report["errors"].append(exc.messages[0])
        else:
            report["errors"].extend(exc.messages)
        write_report(report_path, report)
        return False, report


def new_report(
    robot_path: Path,
    scene_path: Path,
    config_path: Path,
    measurements_path: Path,
    strict: bool,
) -> dict[str, Any]:
    return {
        "status": "started",
        "strict": strict,
        "inputs": {
            "robot": str(robot_path),
            "scene": str(scene_path),
            "config": str(config_path),
            "measurements": str(measurements_path),
        },
        "inspection": {},
        "measurements_used": [],
        "missing_measurements": [],
        "changes": {
            "root": {},
            "removed_collision_geoms": [],
            "removed_actuators": [],
            "updated_joints": [],
            "updated_inertials": [],
            "added_actuators": [],
            "added_collision_geoms": [],
            "added_sensors": [],
            "scene": {},
        },
        "validation": {},
        "errors": [],
    }


def fill_input_report(
    report: dict[str, Any],
    robot_path: Path,
    scene_path: Path,
    config_path: Path,
    measurements_path: Path,
) -> None:
    for key, path in (
        ("robot", robot_path),
        ("scene", scene_path),
        ("config", config_path),
        ("measurements", measurements_path),
    ):
        report["inputs"][f"{key}_sha256"] = sha256_file(path)

    robot_text = robot_path.read_text(encoding="utf-8")
    match = re.search(r"Onshape\s+(https?://\S+)", robot_text)
    if match:
        report["inputs"]["onshape_url"] = match.group(1)


def load_xml(path: Path) -> ET.ElementTree:
    if not path.exists():
        raise PostprocessError(f"missing XML file: {path}")
    try:
        return ET.parse(path)
    except ET.ParseError as exc:
        raise PostprocessError(f"could not parse XML {path}: {exc}") from exc


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PostprocessError(f"missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise PostprocessError(f"YAML root must be a mapping: {path}")
    return data


def parse_wheel_selectors(config: dict[str, Any]) -> dict[str, WheelSelector]:
    wheels = get_path(config, ["selectors", "wheels"])
    if not isinstance(wheels, dict) or not wheels:
        raise PostprocessError("config selectors.wheels must define at least one wheel")

    selectors: dict[str, WheelSelector] = {}
    errors: list[str] = []
    for key, item in wheels.items():
        if not isinstance(item, dict):
            errors.append(f"selectors.wheels.{key} must be a mapping")
            continue
        missing = [name for name in ("body", "joint", "center_site") if not item.get(name)]
        if missing:
            errors.append(f"selectors.wheels.{key} missing: {', '.join(missing)}")
            continue
        selectors[str(key)] = WheelSelector(
            key=str(key),
            body=str(item["body"]),
            joint=str(item["joint"]),
            center_site=str(item["center_site"]),
        )
    if errors:
        raise PostprocessError(errors)
    return selectors


def inspect_robot(
    robot_root: ET.Element,
    config: dict[str, Any],
    wheel_selectors: dict[str, WheelSelector],
) -> Inspection:
    worldbody = require_child(robot_root, "worldbody", "robot XML")
    root_cfg = config.get("root_body", {})
    source_name = str(root_cfg.get("source_name", "base_2"))
    target_name = str(root_cfg.get("target_name", "base_link"))

    root_body = find_direct_body(worldbody, source_name)
    if root_body is None:
        root_body = find_direct_body(worldbody, target_name)
    if root_body is None:
        raise PostprocessError(
            f"could not find root body '{source_name}' or '{target_name}' under worldbody"
        )

    imu_site_name = str(get_path(config, ["selectors", "sites", "imu"]) or "imu")
    base_site_name = str(get_path(config, ["selectors", "sites", "base"]) or "base")

    wheel_bodies: dict[str, ET.Element] = {}
    wheel_joints: dict[str, ET.Element] = {}
    wheel_center_sites: dict[str, ET.Element] = {}
    errors: list[str] = []

    for key, selector in wheel_selectors.items():
        body = find_unique_named(robot_root, "body", selector.body)
        joint = find_unique_named(robot_root, "joint", selector.joint)
        site = find_unique_named(robot_root, "site", selector.center_site)
        if body is None:
            errors.append(f"missing wheel body for {key}: {selector.body}")
        else:
            wheel_bodies[key] = body
        if joint is None:
            errors.append(f"missing wheel joint for {key}: {selector.joint}")
        else:
            wheel_joints[key] = joint
        if site is None:
            errors.append(f"missing wheel center site for {key}: {selector.center_site}")
        else:
            wheel_center_sites[key] = site

    imu_site = find_unique_named(robot_root, "site", imu_site_name)
    base_site = find_unique_named(robot_root, "site", base_site_name)
    if imu_site is None:
        errors.append(f"missing IMU site: {imu_site_name}")
    if base_site is None:
        errors.append(f"missing base site: {base_site_name}")
    if errors:
        raise PostprocessError(errors)

    actuator = robot_root.find("actuator")
    generated_positions: list[str] = []
    wheel_joint_names = {selector.joint for selector in wheel_selectors.values()}
    if actuator is not None:
        for child in actuator:
            if child.tag == "position" and child.get("joint") in wheel_joint_names:
                generated_positions.append(child.get("name") or child.get("joint") or "<unnamed>")

    return Inspection(
        root_body=root_body,
        root_body_name=root_body.get("name", ""),
        wheel_bodies=wheel_bodies,
        wheel_joints=wheel_joints,
        wheel_center_sites=wheel_center_sites,
        imu_site=imu_site,
        base_site=base_site,
        visual_mesh_geoms=count_geoms(robot_root, geom_class="visual", geom_type="mesh"),
        collision_mesh_geoms=count_geoms(robot_root, geom_class="collision", geom_type="mesh"),
        generated_position_actuators=generated_positions,
    )


def fill_inspection_report(
    report: dict[str, Any],
    inspection: Inspection,
    wheel_selectors: dict[str, WheelSelector],
) -> None:
    report["inspection"] = {
        "root_body": inspection.root_body_name,
        "wheels": {
            key: {
                "body": selector.body,
                "joint": selector.joint,
                "center_site": selector.center_site,
            }
            for key, selector in wheel_selectors.items()
        },
        "visual_mesh_geoms": inspection.visual_mesh_geoms,
        "collision_mesh_geoms": inspection.collision_mesh_geoms,
        "generated_position_actuators": inspection.generated_position_actuators,
    }


def validate_measurements(
    measurements: dict[str, Any],
    wheel_selectors: dict[str, WheelSelector],
    strict: bool,
    report: dict[str, Any],
) -> LabValues:
    missing: list[str] = []
    used: list[dict[str, Any]] = report["measurements_used"]
    wheels: dict[str, WheelValues] = {}

    for key in wheel_selectors:
        base = ["wheels", key]
        wheel = WheelValues(
            radius_m=required_float(measurements, [*base, "radius_m"], strict, missing, used),
            width_m=required_float(measurements, [*base, "width_m"], strict, missing, used),
            torque_limit_nm=required_float(
                measurements, [*base, "torque_limit_nm"], strict, missing, used
            ),
            gear_ratio=required_float(measurements, [*base, "gear_ratio"], strict, missing, used),
            command_sign=required_sign(
                measurements, [*base, "command_sign"], strict, missing, used
            ),
            damping=required_float(measurements, [*base, "damping"], strict, missing, used),
            armature=required_float(measurements, [*base, "armature"], strict, missing, used),
            frictionloss=required_float(
                measurements, [*base, "frictionloss"], strict, missing, used
            ),
            mass_kg=optional_float(measurements, [*base, "mass_kg"], strict, missing, used),
        )
        if not math.isnan(wheel.radius_m) and wheel.radius_m <= 0:
            missing.append(f"wheels.{key}.radius_m must be > 0")
        if not math.isnan(wheel.width_m) and wheel.width_m <= 0:
            missing.append(f"wheels.{key}.width_m must be > 0")
        if not math.isnan(wheel.torque_limit_nm) and wheel.torque_limit_nm <= 0:
            missing.append(f"wheels.{key}.torque_limit_nm must be > 0")
        if not math.isnan(wheel.gear_ratio) and wheel.gear_ratio <= 0:
            missing.append(f"wheels.{key}.gear_ratio must be > 0")
        wheels[key] = wheel

    tire_friction = required_float_tuple(
        measurements, ["contact", "tire", "friction"], 3, strict, missing, used
    )
    tire_solref = required_float_tuple(
        measurements, ["contact", "tire", "solref"], 2, strict, missing, used
    )
    tire_solimp = required_float_tuple(
        measurements, ["contact", "tire", "solimp"], 5, strict, missing, used
    )

    report["missing_measurements"] = missing
    if missing:
        raise PostprocessError(["measurement validation failed", *missing])

    return LabValues(
        wheels=wheels,
        tire_friction=tire_friction,
        tire_solref=tire_solref,
        tire_solimp=tire_solimp,
    )


def normalize_root(
    robot_root: ET.Element,
    config: dict[str, Any],
    inspection: Inspection,
    lab_values: LabValues,
    report: dict[str, Any],
) -> None:
    root_cfg = config.get("root_body", {})
    target_name = str(root_cfg.get("target_name", "base_link"))
    freejoint_name = str(root_cfg.get("freejoint_name", "root_freejoint"))
    clearance_m = float(root_cfg.get("initial_clearance_m", 0.003))

    original_name = inspection.root_body.get("name")
    inspection.root_body.set("name", target_name)

    if inspection.root_body.find("freejoint") is None:
        inspection.root_body.insert(0, ET.Element("freejoint", {"name": freejoint_name}))
        freejoint_added = True
    else:
        freejoint_added = False

    root_pos = parse_vec(inspection.root_body.get("pos"), default=(0.0, 0.0, 0.0))
    lowest_bottom = math.inf
    wheel_world: dict[str, dict[str, float]] = {}
    for key, wheel in lab_values.wheels.items():
        center_pos, _ = site_world_transform(
            inspection.root_body, inspection.wheel_center_sites[key]
        )
        bottom_z = center_pos[2] - wheel.radius_m
        lowest_bottom = min(lowest_bottom, bottom_z)
        wheel_world[key] = {
            "center_z_before_lift_m": center_pos[2],
            "bottom_z_before_lift_m": bottom_z,
        }

    lift_m = clearance_m - lowest_bottom
    new_pos = (root_pos[0], root_pos[1], root_pos[2] + lift_m)
    inspection.root_body.set("pos", format_vec(new_pos))

    report["changes"]["root"] = {
        "renamed_from": original_name,
        "renamed_to": target_name,
        "freejoint_added": freejoint_added,
        "freejoint_name": freejoint_name,
        "initial_clearance_m": clearance_m,
        "lift_applied_m": lift_m,
        "wheel_world": wheel_world,
    }


def clean_generated_physics(
    robot_root: ET.Element,
    config: dict[str, Any],
    inspection: Inspection,
    wheel_selectors: dict[str, WheelSelector],
    report: dict[str, Any],
) -> None:
    collision_cfg = config.get("collisions", {})
    preserve = set(collision_cfg.get("preserve_collision_meshes") or [])
    remove_wheel = collision_cfg.get("remove_wheel_mesh_collisions", True) is not False
    remove_chassis = bool(collision_cfg.get("remove_chassis_mesh_collisions", False))

    # Wheel bodies need analytic cylinders for clean rolling contact; the rest of
    # the chassis keeps its convex-hull mesh collisions so the whole robot still
    # collides with the world. All chassis parts share base_link, so MuJoCo
    # auto-disables collisions among them and only world contacts remain.
    wheel_body_geoms = {
        id(geom) for body in inspection.wheel_bodies.values() for geom in body.findall("geom")
    }

    for parent, geom in list(walk_with_parent(robot_root)):
        if geom.tag != "geom":
            continue
        if geom.get("type") != "mesh" or geom.get("class") != "collision":
            continue
        is_wheel_geom = id(geom) in wheel_body_geoms
        if is_wheel_geom and not remove_wheel:
            continue
        if not is_wheel_geom and not remove_chassis:
            continue
        name_or_mesh = {geom.get("name"), geom.get("mesh")} - {None}
        if name_or_mesh & preserve:
            continue
        report["changes"]["removed_collision_geoms"].append(
            {
                "name": geom.get("name"),
                "mesh": geom.get("mesh"),
                "material": geom.get("material"),
                "region": "wheel" if is_wheel_geom else "chassis",
            }
        )
        parent.remove(geom)

    actuator = robot_root.find("actuator")
    if actuator is None:
        return

    wheel_joint_names = {selector.joint for selector in wheel_selectors.values()}
    for child in list(actuator):
        if child.tag == "position" and child.get("joint") in wheel_joint_names:
            report["changes"]["removed_actuators"].append(
                {"tag": child.tag, "name": child.get("name"), "joint": child.get("joint")}
            )
            actuator.remove(child)


def apply_actuators(
    robot_root: ET.Element,
    config: dict[str, Any],
    inspection: Inspection,
    lab_values: LabValues,
    report: dict[str, Any],
) -> None:
    actuator = ensure_top_level(robot_root, "actuator")
    actuator_cfg = config.get("actuators", {})
    motor_prefix = str(actuator_cfg.get("motor_prefix", "motor"))

    target_names = {f"{motor_prefix}_{key}_wheel" for key in lab_values.wheels}
    for child in list(actuator):
        if child.get("name") in target_names:
            actuator.remove(child)

    for key, wheel in lab_values.wheels.items():
        joint = inspection.wheel_joints[key]
        joint.set("damping", format_scalar(wheel.damping))
        joint.set("armature", format_scalar(wheel.armature))
        joint.set("frictionloss", format_scalar(wheel.frictionloss))

        if wheel.mass_kg is not None:
            inertial = inspection.wheel_bodies[key].find("inertial")
            if inertial is None:
                raise PostprocessError(f"wheel {key} has no inertial for mass override")
            inertial.set("mass", format_scalar(wheel.mass_kg))
            report["changes"]["updated_inertials"].append(
                {
                    "wheel": key,
                    "body": inspection.wheel_bodies[key].get("name"),
                    "mass_kg": wheel.mass_kg,
                }
            )

        report["changes"]["updated_joints"].append(
            {
                "wheel": key,
                "joint": joint.get("name"),
                "damping": wheel.damping,
                "armature": wheel.armature,
                "frictionloss": wheel.frictionloss,
            }
        )

        motor_name = f"{motor_prefix}_{key}_wheel"
        gear = wheel.gear_ratio * wheel.command_sign
        motor = ET.Element(
            "motor",
            {
                "name": motor_name,
                "joint": joint.get("name", ""),
                "gear": format_scalar(gear),
                "ctrlrange": format_vec((-wheel.torque_limit_nm, wheel.torque_limit_nm)),
                "ctrllimited": "true",
            },
        )
        actuator.append(motor)
        report["changes"]["added_actuators"].append(
            {
                "name": motor_name,
                "joint": joint.get("name"),
                "gear": gear,
                "ctrlrange": [-wheel.torque_limit_nm, wheel.torque_limit_nm],
            }
        )


def apply_collisions(
    robot_root: ET.Element,
    config: dict[str, Any],
    inspection: Inspection,
    lab_values: LabValues,
    report: dict[str, Any],
) -> None:
    collision_cfg = config.get("collisions", {})
    wheel_cfg = collision_cfg.get("wheels", {})
    material = str(wheel_cfg.get("material", "motor_tire_material"))
    geom_class = str(wheel_cfg.get("class", "collision"))
    # condim 6 keeps tangential + torsional + rolling friction so the measured
    # 3-vector tire friction is actually used; default condim 3 silently ignores
    # the torsional/rolling terms, which matters for a balancing/turning robot.
    condim = int(wheel_cfg.get("condim", 6))

    for key, wheel in lab_values.wheels.items():
        body = inspection.wheel_bodies[key]
        site = inspection.wheel_center_sites[key]
        geom_name = f"{key}_tire_collision"
        remove_child_by_name(body, "geom", geom_name)
        geom = ET.Element(
            "geom",
            {
                "name": geom_name,
                "type": "cylinder",
                "class": geom_class,
                "condim": str(condim),
                "pos": site.get("pos", "0 0 0"),
                "quat": site.get("quat", "1 0 0 0"),
                "size": format_vec((wheel.radius_m, wheel.width_m / 2.0)),
                "material": material,
                "friction": format_vec(lab_values.tire_friction),
                "solref": format_vec(lab_values.tire_solref),
                "solimp": format_vec(lab_values.tire_solimp),
            },
        )
        body.append(geom)
        report["changes"]["added_collision_geoms"].append(
            {
                "name": geom_name,
                "body": body.get("name"),
                "type": "cylinder",
                "condim": condim,
                "radius_m": wheel.radius_m,
                "half_width_m": wheel.width_m / 2.0,
            }
        )

    for rule in collision_cfg.get("primitive_rules", []) or []:
        if not isinstance(rule, dict):
            raise PostprocessError("collisions.primitive_rules entries must be mappings")
        add_primitive_collision(rule, robot_root, report)


def add_primitive_collision(
    rule: dict[str, Any],
    robot_root: ET.Element,
    report: dict[str, Any],
) -> None:
    body_name = rule.get("body")
    geom_name = rule.get("name")
    geom_type = rule.get("type")
    size = rule.get("size")
    if not body_name or not geom_name or not geom_type or size is None:
        raise PostprocessError("primitive collision rule requires body, name, type, and size")
    body = find_unique_named(robot_root, "body", str(body_name))
    if body is None:
        raise PostprocessError(f"primitive collision body not found: {body_name}")
    remove_child_by_name(body, "geom", str(geom_name))
    attrs = {
        "name": str(geom_name),
        "type": str(geom_type),
        "class": str(rule.get("class", "collision")),
        "size": format_vec(as_float_sequence(size, f"primitive {geom_name}.size")),
    }
    for attr in ("pos", "quat", "rgba", "friction", "solref", "solimp", "material"):
        if attr in rule:
            value = rule[attr]
            attrs[attr] = format_vec(value) if isinstance(value, list | tuple) else str(value)
    geom = ET.Element("geom", attrs)
    body.append(geom)
    report["changes"]["added_collision_geoms"].append(
        {"name": geom_name, "body": body_name, "type": geom_type, "source": "primitive_rule"}
    )


def apply_sensors(
    robot_root: ET.Element,
    config: dict[str, Any],
    inspection: Inspection,
    wheel_selectors: dict[str, WheelSelector],
    report: dict[str, Any],
) -> None:
    sensor_cfg = config.get("sensors", {})
    sensor = ensure_top_level(robot_root, "sensor")
    names_to_remove: set[str] = set()

    imu_cfg = sensor_cfg.get("imu", {})
    imu_site_name = str(imu_cfg.get("site", inspection.imu_site.get("name", "imu")))
    imu_names = {
        "gyro": str(imu_cfg.get("gyro", "imu_gyro")),
        "accelerometer": str(imu_cfg.get("accelerometer", "imu_accelerometer")),
        "framequat": str(imu_cfg.get("framequat", "imu_quat")),
    }
    names_to_remove.update(imu_names.values())

    base_cfg = sensor_cfg.get("base", {})
    base_site_name = str(base_cfg.get("site", inspection.base_site.get("name", "base")))
    base_names = {
        "framepos": str(base_cfg.get("framepos", "base_pos")),
        "framequat": str(base_cfg.get("framequat", "base_quat")),
    }
    names_to_remove.update(base_names.values())

    wheel_sensor_prefix = str(sensor_cfg.get("wheel_prefix", "wheel"))
    wheel_names: dict[str, dict[str, str]] = {}
    for key in wheel_selectors:
        wheel_names[key] = {
            "jointpos": f"{wheel_sensor_prefix}_{key}_pos",
            "jointvel": f"{wheel_sensor_prefix}_{key}_vel",
        }
        names_to_remove.update(wheel_names[key].values())

    for child in list(sensor):
        if child.get("name") in names_to_remove:
            sensor.remove(child)

    sensors = [
        ET.Element("gyro", {"name": imu_names["gyro"], "site": imu_site_name}),
        ET.Element(
            "accelerometer",
            {"name": imu_names["accelerometer"], "site": imu_site_name},
        ),
        ET.Element(
            "framequat",
            {"name": imu_names["framequat"], "objtype": "site", "objname": imu_site_name},
        ),
        ET.Element(
            "framepos",
            {"name": base_names["framepos"], "objtype": "site", "objname": base_site_name},
        ),
        ET.Element(
            "framequat",
            {"name": base_names["framequat"], "objtype": "site", "objname": base_site_name},
        ),
    ]
    for key, selector in wheel_selectors.items():
        sensors.append(
            ET.Element(
                "jointpos",
                {"name": wheel_names[key]["jointpos"], "joint": selector.joint},
            )
        )
        sensors.append(
            ET.Element(
                "jointvel",
                {"name": wheel_names[key]["jointvel"], "joint": selector.joint},
            )
        )

    for item in sensors:
        sensor.append(item)
        report["changes"]["added_sensors"].append(
            {
                "tag": item.tag,
                "name": item.get("name"),
                "target": item.get("site") or item.get("joint"),
            }
        )

    report["validation"]["sensor_model"] = "ideal; no noise/filter settings were applied"


def assemble_scene(
    robot_root: ET.Element,
    scene_root: ET.Element,
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    pipeline_cfg = config.get("pipeline", {})
    robot_root.set("model", str(pipeline_cfg.get("output_model", "finn_sim")))

    compiler = robot_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        robot_root.insert(0, compiler)
    compiler.set("meshdir", str(pipeline_cfg.get("compiler_meshdir", "../model/assets")))

    scene_cfg = config.get("scene", {})
    if scene_cfg.get("merge_visual", True):
        scene_visual = scene_root.find("visual")
        if scene_visual is not None:
            remove_top_level(robot_root, "visual")
            insert_after(robot_root, copy.deepcopy(scene_visual), after_tags=("compiler", "option"))
            report["changes"]["scene"]["visual_merged"] = True

    if scene_cfg.get("merge_assets", True):
        robot_asset = ensure_top_level(robot_root, "asset")
        scene_asset = scene_root.find("asset")
        merged = 0
        if scene_asset is not None:
            for child in scene_asset:
                robot_asset.append(copy.deepcopy(child))
                merged += 1
        report["changes"]["scene"]["assets_merged"] = merged

    if scene_cfg.get("merge_worldbody", True):
        robot_worldbody = require_child(robot_root, "worldbody", "robot XML")
        scene_worldbody = scene_root.find("worldbody")
        merged = 0
        if scene_worldbody is not None:
            for child in reversed(list(scene_worldbody)):
                robot_worldbody.insert(0, copy.deepcopy(child))
                merged += 1
        report["changes"]["scene"]["worldbody_children_merged"] = merged


def validate_mujoco(
    xml_path: Path,
    config: dict[str, Any],
    wheel_selectors: dict[str, WheelSelector],
    report: dict[str, Any],
) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise PostprocessError("mujoco is not installed; cannot validate generated XML") from exc

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:
        raise PostprocessError(f"mujoco compile failed: {exc}") from exc

    checks: list[tuple[Any, str]] = []
    root_target = str(get_path(config, ["root_body", "target_name"]) or "base_link")
    checks.append((mujoco.mjtObj.mjOBJ_BODY, root_target))
    for selector in wheel_selectors.values():
        checks.append((mujoco.mjtObj.mjOBJ_BODY, selector.body))
        checks.append((mujoco.mjtObj.mjOBJ_JOINT, selector.joint))

    motor_prefix = str(get_path(config, ["actuators", "motor_prefix"]) or "motor")
    for key in wheel_selectors:
        checks.append((mujoco.mjtObj.mjOBJ_ACTUATOR, f"{motor_prefix}_{key}_wheel"))

    sensor_cfg = config.get("sensors", {})
    imu_cfg = sensor_cfg.get("imu", {})
    base_cfg = sensor_cfg.get("base", {})
    wheel_sensor_prefix = str(sensor_cfg.get("wheel_prefix", "wheel"))
    sensor_names = [
        str(imu_cfg.get("gyro", "imu_gyro")),
        str(imu_cfg.get("accelerometer", "imu_accelerometer")),
        str(imu_cfg.get("framequat", "imu_quat")),
        str(base_cfg.get("framepos", "base_pos")),
        str(base_cfg.get("framequat", "base_quat")),
    ]
    for key in wheel_selectors:
        sensor_names.extend(
            [f"{wheel_sensor_prefix}_{key}_pos", f"{wheel_sensor_prefix}_{key}_vel"]
        )
    for name in sensor_names:
        checks.append((mujoco.mjtObj.mjOBJ_SENSOR, name))

    missing: list[str] = []
    for obj_type, name in checks:
        if mujoco.mj_name2id(model, obj_type, name) < 0:
            missing.append(name)
    if missing:
        raise PostprocessError(f"mujoco model missing expected names: {', '.join(missing)}")

    data = mujoco.MjData(model)
    steps = int(get_path(config, ["validation", "zero_control_steps"]) or 25)
    for _ in range(steps):
        mujoco.mj_step(model, data)

    finite = all(math.isfinite(float(value)) for value in data.qpos) and all(
        math.isfinite(float(value)) for value in data.qvel
    )
    if not finite:
        raise PostprocessError("mujoco validation produced non-finite qpos/qvel")

    report["validation"]["mujoco"] = {
        "status": "ok",
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "nsensor": int(model.nsensor),
        "zero_control_steps": steps,
    }


def init_measurements_template(robot_path: Path, output_path: Path, *, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise PostprocessError(
            f"refusing to overwrite existing file without --force: {output_path}"
        )

    robot_tree = load_xml(robot_path)
    robot_root = robot_tree.getroot()
    wheel_keys = infer_wheel_keys(robot_root)
    if not wheel_keys:
        wheel_keys = ["left", "right"]

    template: dict[str, Any] = {
        "schema_version": 1,
        "wheels": {
            key: {
                "radius_m": provenance(None, "m", "todo"),
                "width_m": provenance(None, "m", "todo"),
                "mass_kg": provenance(
                    None, "kg", "todo", "Optional; omit if CAD inertial is accepted."
                ),
                "torque_limit_nm": provenance(None, "N*m", "todo"),
                "gear_ratio": provenance(
                    None,
                    "ratio",
                    "todo",
                    "1 for a direct-drive (hoverboard) motor; FOC torque is at the output.",
                ),
                "command_sign": provenance(None, "sign", "todo", "Use 1 or -1."),
                "damping": provenance(None, "N*m*s/rad", "todo"),
                "armature": provenance(None, "kg*m^2", "todo"),
                "frictionloss": provenance(None, "N*m", "todo"),
            }
            for key in wheel_keys
        },
        "contact": {
            "tire": {
                "friction": provenance(
                    None, "slide torsional rolling", "todo", "MuJoCo geom friction vector."
                ),
                "solref": provenance(None, "timeconst dampratio", "todo"),
                "solimp": provenance(None, "dmin dmax width midpoint power", "todo"),
            }
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(template, stream, sort_keys=False)


def infer_wheel_keys(robot_root: ET.Element) -> list[str]:
    keys: list[str] = []
    for body in robot_root.iter("body"):
        name = body.get("name", "")
        if "wheel" not in name:
            continue
        if name.startswith("left"):
            keys.append("left")
        elif name.startswith("right"):
            keys.append("right")
        else:
            keys.append(name)
    return sorted(set(keys))


def provenance(value: Any, unit: str, source: str, notes: str | None = None) -> dict[str, Any]:
    item = {"value": value, "unit": unit, "source": source}
    if notes:
        item["notes"] = notes
    return item


def required_float(
    data: dict[str, Any],
    path: list[str],
    strict: bool,
    missing: list[str],
    used: list[dict[str, Any]],
) -> float:
    value = required_value(data, path, strict, missing, used)
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        missing.append(f"{dotted(path)} must be numeric")
        return math.nan


def required_sign(
    data: dict[str, Any],
    path: list[str],
    strict: bool,
    missing: list[str],
    used: list[dict[str, Any]],
) -> float:
    value = required_float(data, path, strict, missing, used)
    if not math.isnan(value) and value not in (-1.0, 1.0):
        missing.append(f"{dotted(path)} must be exactly 1 or -1")
    return value


def optional_float(
    data: dict[str, Any],
    path: list[str],
    strict: bool,
    missing: list[str],
    used: list[dict[str, Any]],
) -> float | None:
    node = get_path(data, path)
    if node is None:
        return None
    value = validate_provenance_node(node, path, strict, missing, used, required=False)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        missing.append(f"{dotted(path)} must be numeric")
        return None


def required_float_tuple(
    data: dict[str, Any],
    path: list[str],
    length: int,
    strict: bool,
    missing: list[str],
    used: list[dict[str, Any]],
) -> tuple[float, ...]:
    value = required_value(data, path, strict, missing, used)
    if value is None:
        return tuple(math.nan for _ in range(length))
    try:
        parsed = tuple(as_float_sequence(value, dotted(path)))
    except PostprocessError as exc:
        missing.extend(exc.messages)
        return tuple(math.nan for _ in range(length))
    if len(parsed) != length:
        missing.append(f"{dotted(path)} must contain {length} numbers")
        return tuple(math.nan for _ in range(length))
    return parsed


def required_value(
    data: dict[str, Any],
    path: list[str],
    strict: bool,
    missing: list[str],
    used: list[dict[str, Any]],
) -> Any:
    node = get_path(data, path)
    return validate_provenance_node(node, path, strict, missing, used, required=True)


def validate_provenance_node(
    node: Any,
    path: list[str],
    strict: bool,
    missing: list[str],
    used: list[dict[str, Any]],
    *,
    required: bool,
) -> Any:
    label = dotted(path)
    if not isinstance(node, dict):
        if required:
            missing.append(f"{label} missing provenance mapping")
        return None

    value = node.get("value")
    value_is_todo = value is None or (
        isinstance(value, str) and value.strip().lower() in TODO_STRINGS
    )
    if not required and value_is_todo:
        return None

    source = node.get("source")
    unit = node.get("unit")
    if source not in ALLOWED_SOURCES:
        missing.append(f"{label}.source must be one of {sorted(ALLOWED_SOURCES)}")
    if unit is None:
        missing.append(f"{label}.unit is required")

    if required and value_is_todo:
        missing.append(f"{label}.value is required")
    if strict and source in STRICT_BLOCKED_SOURCES:
        missing.append(f"{label}.source '{source}' is not allowed in strict mode")

    if not value_is_todo:
        used.append(
            {
                "path": label,
                "value": value,
                "unit": unit,
                "source": source,
                "notes": node.get("notes"),
            }
        )
        return value
    return None


def get_path(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def dotted(path: list[str]) -> str:
    return ".".join(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_child(parent: ET.Element, tag: str, label: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        raise PostprocessError(f"{label} missing <{tag}>")
    return child


def find_direct_body(worldbody: ET.Element, name: str) -> ET.Element | None:
    matches = [body for body in worldbody.findall("body") if body.get("name") == name]
    if len(matches) > 1:
        raise PostprocessError(f"multiple direct worldbody bodies named {name}")
    return matches[0] if matches else None


def find_unique_named(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    matches = [item for item in root.iter(tag) if item.get("name") == name]
    if len(matches) > 1:
        raise PostprocessError(f"multiple <{tag}> elements named {name}")
    return matches[0] if matches else None


def count_geoms(root: ET.Element, *, geom_class: str, geom_type: str) -> int:
    return sum(
        1
        for geom in root.iter("geom")
        if geom.get("class") == geom_class and geom.get("type") == geom_type
    )


def walk_with_parent(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    items: list[tuple[ET.Element, ET.Element]] = []
    for child in list(root):
        items.append((root, child))
        items.extend(walk_with_parent(child))
    return items


def ensure_top_level(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is not None:
        return child
    child = ET.Element(tag)
    root.append(child)
    return child


def remove_child_by_name(parent: ET.Element, tag: str, name: str) -> None:
    for child in list(parent):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)


def remove_top_level(root: ET.Element, tag: str) -> None:
    for child in list(root):
        if child.tag == tag:
            root.remove(child)


def insert_after(root: ET.Element, item: ET.Element, *, after_tags: tuple[str, ...]) -> None:
    children = list(root)
    insert_at = 0
    for index, child in enumerate(children):
        if child.tag in after_tags:
            insert_at = index + 1
    root.insert(insert_at, item)


def parse_vec(value: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    return tuple(float(part) for part in value.split())


def as_float_sequence(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = value.split()
    elif isinstance(value, list | tuple):
        parts = value
    else:
        raise PostprocessError(f"{label} must be a string or list of numbers")
    try:
        return tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise PostprocessError(f"{label} must contain only numbers") from exc


def format_scalar(value: float) -> str:
    return f"{value:.9g}"


def format_vec(values: Any) -> str:
    if isinstance(values, str):
        return values
    return " ".join(format_scalar(float(value)) for value in values)


def site_world_transform(
    root_body: ET.Element, target_site: ET.Element
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    target_name = target_site.get("name")
    found = find_site_transform(root_body, target_name, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    if found is None:
        raise PostprocessError(f"could not compute world transform for site {target_name}")
    return found


def find_site_transform(
    body: ET.Element,
    target_site_name: str | None,
    parent_pos: tuple[float, ...],
    parent_quat: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    body_pos = parse_vec(body.get("pos"), default=(0.0, 0.0, 0.0))
    body_quat = parse_vec(body.get("quat"), default=(1.0, 0.0, 0.0, 0.0))
    world_pos, world_quat = compose_transform(parent_pos, parent_quat, body_pos, body_quat)

    for site in body.findall("site"):
        if site.get("name") != target_site_name:
            continue
        site_pos = parse_vec(site.get("pos"), default=(0.0, 0.0, 0.0))
        site_quat = parse_vec(site.get("quat"), default=(1.0, 0.0, 0.0, 0.0))
        return compose_transform(world_pos, world_quat, site_pos, site_quat)

    for child_body in body.findall("body"):
        found = find_site_transform(child_body, target_site_name, world_pos, world_quat)
        if found is not None:
            return found
    return None


def compose_transform(
    parent_pos: tuple[float, ...],
    parent_quat: tuple[float, ...],
    local_pos: tuple[float, ...],
    local_quat: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rotated = quat_rotate(parent_quat, local_pos)
    pos = tuple(parent_pos[i] + rotated[i] for i in range(3))
    quat = quat_normalize(quat_mul(parent_quat, local_quat))
    return pos, quat


def quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q: tuple[float, ...]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def quat_rotate(q: tuple[float, ...], vec: tuple[float, ...]) -> tuple[float, float, float]:
    qn = quat_normalize(q)
    rotated = quat_mul(quat_mul(qn, (0.0, vec[0], vec[1], vec[2])), quat_conj(qn))
    return (rotated[1], rotated[2], rotated[3])


def quat_normalize(q: tuple[float, ...]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(part * part for part in q))
    if norm == 0:
        raise PostprocessError("encountered zero-length quaternion")
    return (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm)


def write_temp_xml(tree: ET.ElementTree, out_path: Path) -> Path:
    ET.indent(tree, space="  ")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{out_path.name}.",
        suffix=".tmp.xml",
        dir=out_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        tree.write(temp_file, encoding="utf-8", xml_declaration=True)
    return temp_path


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Finn MuJoCo Postprocess Report",
        "",
        f"- status: `{report.get('status')}`",
        f"- strict: `{report.get('strict')}`",
        "",
        "## Inputs",
    ]
    for key, value in report.get("inputs", {}).items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Inspection"])
    for key, value in report.get("inspection", {}).items():
        lines.append(f"- {key}: `{value}`")

    missing = report.get("missing_measurements", [])
    lines.extend(["", "## Missing Measurements"])
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- none")

    lines.extend(["", "## Measurements Used"])
    used = report.get("measurements_used", [])
    if used:
        for item in used:
            lines.append(
                f"- {item['path']}: `{item['value']}` {item.get('unit')} "
                f"(source: `{item.get('source')}`)"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Changes"])
    for key, value in report.get("changes", {}).items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Validation"])
    for key, value in report.get("validation", {}).items():
        lines.append(f"- {key}: `{value}`")

    errors = report.get("errors", [])
    lines.extend(["", "## Errors"])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
