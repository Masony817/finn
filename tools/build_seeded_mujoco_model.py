#!/usr/bin/env python3
"""Build a generated MuJoCo seed model from Finn sysid batches."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from finn import model as ppm
from finn.paths import portable_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSID_ROOT = Path("logs/finn-mcu/sysid")
DEFAULT_MEASUREMENTS = Path("sim/config/finn_measurements.yaml")
DEFAULT_ROBOT = Path("sim/model/finn/finn_robot.xml")
DEFAULT_SCENE = Path("sim/model/finn/scene.xml")
DEFAULT_CONFIG = Path("sim/config/mujoco_postprocess.yaml")
DEFAULT_CONTACT_FRICTION = [1.0, 0.02, 0.002]

# Per-wheel torque envelope the sim validates the balance controller against.
#
# Sized from the corrected seeded model rather than inherited from Batch 1: with
# both wheels at this cap the controller recovers roughly 11 degrees of lean from
# the balance trim, which covers the firmware's 8 degree arm window with margin,
# while capping chassis acceleration near 3 m/s^2 so a fault cannot launch the
# robot across the room. The hub motors can deliver considerably more; that extra
# authority stays unused until a milestone needs it.
DEFAULT_TORQUE_LIMIT_NM = 1.0
DEFAULT_CONTACT_SOLREF = [0.02, 1.0]
DEFAULT_CONTACT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]


class BuildSeededModelError(Exception):
    """Expected CLI failure with a concise user-facing message."""


@dataclass(frozen=True)
class SysidRun:
    batch: str
    run_dir: Path
    derived_path: Path
    derived: dict[str, Any]
    rows: int
    duration_s: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a generated seeded Finn MuJoCo model from sysid batches."
    )
    parser.add_argument("--sysid-root", type=Path, default=DEFAULT_SYSID_ROOT)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument("--robot", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--auto-select-latest", type=int, default=3)
    parser.add_argument("--batch1-run", type=Path, action="append", default=[])
    parser.add_argument("--batch2-run", type=Path, action="append", default=[])
    parser.add_argument("--wheel-width-m", type=float)
    parser.add_argument(
        "--torque-limit-nm",
        type=float,
        default=DEFAULT_TORQUE_LIMIT_NM,
        help=(
            "Per-wheel actuator torque envelope written into the model's ctrlrange. "
            "This is a reviewed operating limit, not an identified motor capability: "
            "Batch 1 only records whatever hard cap the bench firmware happened to "
            "run with, which is far below what the hub motors can deliver."
        ),
    )
    parser.add_argument("--validate-mujoco", action="store_true")
    parser.add_argument("--replay-batch2", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write run selection, aggregation, and seeded measurements without XML generation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_seeded_model(args)
    except BuildSeededModelError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote seeded bundle: {result['out_dir']}")
    if result.get("status") != "ok":
        for message in result.get("errors", []):
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


def build_seeded_model(args: argparse.Namespace) -> dict[str, Any]:
    sysid_root = args.sysid_root
    measurements_path = args.measurements
    robot_path = args.robot
    scene_path = args.scene
    config_path = args.config
    out_dir = args.out_dir or Path("sim/generated/seeded") / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    measurements = load_yaml(measurements_path)
    selected = select_runs(
        sysid_root=sysid_root,
        latest_count=args.auto_select_latest,
        batch1_runs=args.batch1_run,
        batch2_runs=args.batch2_run,
    )
    aggregate = aggregate_runs(selected["batch1"], selected["batch2"])
    seeded_measurements = build_seeded_measurements(
        measurements=measurements,
        aggregate=aggregate,
        wheel_width_m=args.wheel_width_m,
        torque_limit_nm=args.torque_limit_nm,
    )

    selected_json = out_dir / "selected_runs.json"
    aggregate_json = out_dir / "aggregation.json"
    aggregate_md = out_dir / "aggregation_report.md"
    seeded_yaml = out_dir / "seeded_measurements.yaml"
    validation_json = out_dir / "validation.json"
    validation_md = out_dir / "validation.md"
    model_xml = out_dir / "finn.seeded.sim.xml"
    mujoco_report = out_dir / "finn.seeded.sim.report.md"

    write_json(selected_json, selected_runs_payload(selected))
    write_json(aggregate_json, aggregate)
    write_markdown(aggregate_md, aggregation_markdown(selected, aggregate))
    write_yaml(seeded_yaml, seeded_measurements)

    validation: dict[str, Any] = {
        "status": "not_requested",
        "model_xml": str(model_xml),
        "mujoco_report": str(mujoco_report),
        "onboard_replay_scope": (
            "onboard-signal validation only; this does not validate world pose, "
            "absolute slip, endpoint error, or closed-loop LQR transfer"
        ),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    status = "ok"
    errors: list[str] = []
    postprocess_report: dict[str, Any] | None = None
    missing_assets = preflight_mesh_assets(robot_path)
    validation["mesh_assets"] = {
        "status": "missing" if missing_assets else "ok",
        "missing": [str(path) for path in missing_assets],
    }

    needs_compile = bool(args.validate_mujoco or args.replay_batch2)
    if args.report_only:
        validation["status"] = "report_only"
    else:
        skip_mujoco_validation = not args.validate_mujoco or bool(missing_assets)
        ok, postprocess_report = ppm.postprocess(
            robot_path=robot_path,
            scene_path=scene_path,
            config_path=config_path,
            measurements_path=seeded_yaml,
            out_path=model_xml,
            report_path=mujoco_report,
            strict=False,
            skip_mujoco_validation=skip_mujoco_validation,
        )
        validation["postprocess"] = {
            "status": "ok" if ok else "failed",
            "strict": False,
            "skipped_internal_mujoco_validation": skip_mujoco_validation,
            "report": str(mujoco_report),
        }
        if not ok:
            status = "failed"
            errors.extend(postprocess_report.get("errors", []) if postprocess_report else [])
        elif needs_compile and missing_assets:
            status = "failed"
            validation["status"] = "failed_missing_mesh_assets"
            errors.append(missing_assets_message(missing_assets))
        elif needs_compile:
            try:
                validation["checks"] = run_mujoco_cli_checks(model_xml)
                validation["status"] = "ok"
            except Exception as exc:  # pragma: no cover - exact MuJoCo failures vary
                status = "failed"
                validation["status"] = "failed_mujoco_checks"
                errors.append(f"MuJoCo CLI checks failed: {type(exc).__name__}: {exc}")

        if args.replay_batch2 and status == "ok":
            try:
                replay = replay_batch2_open_loop(model_xml, selected["batch2"][-1].run_dir)
                write_json(out_dir / "replay_batch2.json", replay)
                write_markdown(out_dir / "replay_batch2.md", replay_markdown(replay))
                validation["replay_batch2"] = {
                    "status": replay.get("status"),
                    "json": str(out_dir / "replay_batch2.json"),
                    "markdown": str(out_dir / "replay_batch2.md"),
                }
            except Exception as exc:  # pragma: no cover - exact MuJoCo failures vary
                status = "failed"
                validation["status"] = "failed_replay_batch2"
                errors.append(f"Batch 2 replay failed: {type(exc).__name__}: {exc}")

    validation["errors"] = errors
    write_json(validation_json, validation)
    write_markdown(validation_md, validation_markdown(validation))

    return {
        "status": status,
        "out_dir": str(out_dir),
        "errors": errors,
        "selected_runs": str(selected_json),
        "aggregation": str(aggregate_json),
        "seeded_measurements": str(seeded_yaml),
        "model_xml": str(model_xml) if model_xml.exists() else None,
        "validation": str(validation_json),
    }


def select_runs(
    *,
    sysid_root: Path,
    latest_count: int,
    batch1_runs: list[Path],
    batch2_runs: list[Path],
) -> dict[str, list[SysidRun]]:
    if latest_count <= 0:
        raise BuildSeededModelError("--auto-select-latest must be > 0")

    selected_b1 = (
        validate_explicit_runs("batch1", batch1_runs)
        if batch1_runs
        else discover_clean_runs("batch1", sysid_root / "batch_1_pass", latest_count)
    )
    selected_b2 = (
        validate_explicit_runs("batch2", batch2_runs)
        if batch2_runs
        else discover_clean_runs("batch2", sysid_root / "batch_2_pass", latest_count)
    )

    if len(selected_b1) < latest_count and not batch1_runs:
        raise BuildSeededModelError(
            f"found only {len(selected_b1)} clean Batch 1 runs; need {latest_count}"
        )
    if len(selected_b2) < latest_count and not batch2_runs:
        raise BuildSeededModelError(
            f"found only {len(selected_b2)} clean Batch 2 runs; need {latest_count}"
        )
    return {"batch1": selected_b1, "batch2": selected_b2}


def discover_clean_runs(batch: str, root: Path, latest_count: int) -> list[SysidRun]:
    runs: list[SysidRun] = []
    for run_dir in sorted(root.glob("*")):
        if not run_dir.is_dir():
            continue
        try:
            run = load_sysid_run(batch, run_dir)
            validate_clean_run(run)
        except BuildSeededModelError:
            continue
        runs.append(run)
    return runs[-latest_count:]


def validate_explicit_runs(batch: str, run_dirs: list[Path]) -> list[SysidRun]:
    runs = []
    for run_dir in run_dirs:
        run = load_sysid_run(batch, run_dir)
        validate_clean_run(run)
        runs.append(run)
    return runs


def load_sysid_run(batch: str, run_dir: Path) -> SysidRun:
    derived_path = run_dir / "postprocess" / "derived.yaml"
    if run_dir.name == "derived.yaml":
        derived_path = run_dir
        run_dir = run_dir.parents[1]
    if not derived_path.exists():
        raise BuildSeededModelError(f"missing derived.yaml for {run_dir}")
    derived = load_yaml(derived_path)
    if batch == "batch1":
        rows = int(get_path(derived, ["run", "rows"]) or 0)
        duration = float(get_path(derived, ["run", "duration_s"]) or 0.0)
    elif batch == "batch2":
        rows = int(get_path(derived, ["metadata", "rows"]) or 0)
        duration = float(get_path(derived, ["metadata", "duration_s"]) or 0.0)
    else:
        raise BuildSeededModelError(f"unknown batch: {batch}")
    return SysidRun(
        batch=batch,
        run_dir=run_dir,
        derived_path=derived_path,
        derived=derived,
        rows=rows,
        duration_s=duration,
    )


def validate_clean_run(run: SysidRun) -> None:
    if run.rows < 10_000:
        raise BuildSeededModelError(f"{run.run_dir} has only {run.rows} rows")
    if run.batch == "batch1":
        if int(get_path(run.derived, ["health", "fault_rows"]) or 0) != 0:
            raise BuildSeededModelError(f"{run.run_dir} has Batch 1 faults")
        for side in ("left", "right"):
            for key in ("frictionloss", "damping", "armature", "torque_limit_nm", "command_sign"):
                value = get_path(run.derived, ["wheels", side, "suggested_sim", key, "value"])
                if value is None:
                    raise BuildSeededModelError(f"{run.run_dir} missing {side} {key}")
    else:
        fault_counts = get_path(run.derived, ["metadata", "fault_counts"]) or {}
        if int(fault_counts.get("fault_rows") or 0) != 0:
            raise BuildSeededModelError(f"{run.run_dir} has Batch 2 fault rows")
        if int(fault_counts.get("failsafe_events") or 0) != 0:
            raise BuildSeededModelError(f"{run.run_dir} has Batch 2 failsafe events")
        if get_path(run.derived, ["lqr_readiness", "pass"]) is not True:
            raise BuildSeededModelError(f"{run.run_dir} did not pass Batch 2 readiness")


def aggregate_runs(batch1: list[SysidRun], batch2: list[SysidRun]) -> dict[str, Any]:
    if not batch1:
        raise BuildSeededModelError("no Batch 1 runs selected")
    if not batch2:
        raise BuildSeededModelError("no Batch 2 runs selected")

    actuators: dict[str, Any] = {}
    for side in ("left", "right"):
        signs = [
            int_numeric(
                get_path(run.derived, ["wheels", side, "suggested_sim", "command_sign", "value"])
            )
            for run in batch1
        ]
        if any(sign not in (-1, 1) for sign in signs):
            raise BuildSeededModelError(f"missing command sign for {side} in Batch 1 runs")
        if len(set(signs)) != 1:
            raise BuildSeededModelError(f"Batch 1 command sign disagreement for {side}: {signs}")
        actuators[side] = {
            "command_sign": signs[0],
            "torque_limit_nm": median_path(
                batch1, ["wheels", side, "suggested_sim", "torque_limit_nm", "value"]
            ),
            "frictionloss_nm": median_path(
                batch1, ["wheels", side, "suggested_sim", "frictionloss", "value"]
            ),
            "damping_nm_s_per_rad": median_path(
                batch1, ["wheels", side, "suggested_sim", "damping", "value"]
            ),
            "armature_kg_m2": median_path(
                batch1, ["wheels", side, "suggested_sim", "armature", "value"]
            ),
            "batch1_actuator_gain": median_path(
                batch1, ["wheels", side, "diagnostics", "actuator_tracking", "gain"]
            ),
            "batch2_actuator_tracking": {
                "gain": median_path(
                    batch2, ["suggested_sim", "wheels", "actuator_tracking", side, "gain"]
                ),
                "bias_nm": median_path(
                    batch2, ["suggested_sim", "wheels", "actuator_tracking", side, "bias_nm"]
                ),
                "rmse_nm": median_path(
                    batch2, ["suggested_sim", "wheels", "actuator_tracking", side, "rmse_nm"]
                ),
            },
        }

    aggregate = {
        "schema_version": 1,
        "source": "finn_seeded_mujoco_model_builder",
        "selected_runs": selected_runs_payload({"batch1": batch1, "batch2": batch2}),
        "actuators": actuators,
        "batch2": {
            "command_to_measured_torque_delay_s": median_path(
                batch2,
                ["suggested_sim", "delays", "command_to_measured_torque", "average", "delay_s"],
            ),
            "loaded_loss_correction": {
                "friction_accel_m_s2": median_path(
                    batch2,
                    ["suggested_sim", "wheels", "loaded_loss_correction", "friction_accel_m_s2"],
                ),
                "linear_damping_s": median_path(
                    batch2,
                    ["suggested_sim", "wheels", "loaded_loss_correction", "linear_damping_s"],
                ),
                "per_wheel_torque_nm": median_path(
                    batch2,
                    ["suggested_sim", "wheels", "loaded_loss_correction", "per_wheel_torque_nm"],
                ),
                "confidence": "provisional",
            },
            "contact_lower_bounds": {
                "straight_mu_lower_bound": median_path(
                    batch2,
                    [
                        "suggested_sim",
                        "contact",
                        "tire_friction_lower_bounds",
                        "straight_mu_lower_bound",
                    ],
                )
            },
            "loaded_radius_estimate_m": median_path(
                batch2, ["suggested_sim", "wheels", "loaded_radius_estimate", "radius_m"]
            ),
            "effective_track_width_estimate_m": median_path(
                batch2, ["suggested_sim", "wheels", "yaw_response", "effective_track_width_m"]
            ),
        },
        "contact_seed_defaults": {
            "friction": DEFAULT_CONTACT_FRICTION,
            "solref": DEFAULT_CONTACT_SOLREF,
            "solimp": DEFAULT_CONTACT_SOLIMP,
            "notes": [
                "provisional_seed_defaults",
                "batch2_does_not_identify_solref_solimp_or_world_pose_friction",
            ],
        },
        "do_not_infer": [
            "world_pose_accuracy",
            "absolute_slip",
            "contact_solref",
            "contact_solimp",
            "closed_loop_lqr_transfer",
        ],
    }
    return aggregate


def build_seeded_measurements(
    *,
    measurements: dict[str, Any],
    aggregate: dict[str, Any],
    wheel_width_m: float | None,
    torque_limit_nm: float = DEFAULT_TORQUE_LIMIT_NM,
) -> dict[str, Any]:
    seeded = copy.deepcopy(measurements)
    seeded.setdefault("schema_version", 1)
    seeded.setdefault("wheels", {})
    seeded.setdefault("contact", {}).setdefault("tire", {})

    for side in ("left", "right"):
        wheel = seeded["wheels"].setdefault(side, {})
        width = measurement_value(wheel.get("width_m"))
        if width is None:
            if wheel_width_m is None:
                raise BuildSeededModelError(
                    "wheel width is missing from measurements; pass --wheel-width-m"
                )
            if wheel_width_m <= 0:
                raise BuildSeededModelError("--wheel-width-m must be > 0")
            wheel["width_m"] = provenance(
                wheel_width_m,
                "m",
                "measured",
                "CLI-supplied wheel width; Batch 1/2 cannot identify wheel width.",
            )

        if measurement_value(wheel.get("radius_m")) is None:
            raise BuildSeededModelError(f"wheels.{side}.radius_m is required from measurements")
        wheel.setdefault("gear_ratio", provenance(1.0, "ratio", "measured"))
        actuator = aggregate["actuators"][side]
        if torque_limit_nm <= 0.0:
            raise BuildSeededModelError("--torque-limit-nm must be > 0")
        wheel["torque_limit_nm"] = provenance(
            torque_limit_nm,
            "N*m",
            "reviewed",
            (
                "Reviewed operating envelope, not an identified capability. Batch 1 "
                f"only observed the bench firmware's own hard cap "
                f"({actuator['torque_limit_nm']} N*m), which is well below both the "
                "hub motors' capability and the torque needed to balance."
            ),
        )
        wheel["command_sign"] = provenance(
            actuator["command_sign"],
            "sign",
            "moteus",
            "Consensus Batch 1 command sign.",
        )
        wheel["damping"] = provenance(
            actuator["damping_nm_s_per_rad"],
            "N*m*s/rad",
            "moteus",
            "Median Batch 1 off-ground damping seed.",
        )
        wheel["armature"] = provenance(
            actuator["armature_kg_m2"],
            "kg*m^2",
            "moteus",
            "Median Batch 1 armature seed after CAD axial inertia subtraction.",
        )
        wheel["frictionloss"] = provenance(
            actuator["frictionloss_nm"],
            "N*m",
            "moteus",
            "Median Batch 1 off-ground frictionloss seed.",
        )

    tire = seeded["contact"]["tire"]
    tire["friction"] = provenance(
        DEFAULT_CONTACT_FRICTION,
        "slide torsional rolling",
        "estimated",
        (
            "Provisional seed default. Batch 2 only gives lower bounds; tune with "
            "controlled LQR validation."
        ),
    )
    tire["solref"] = provenance(
        DEFAULT_CONTACT_SOLREF,
        "timeconst dampratio",
        "estimated",
        "Provisional seed default. Do not treat as identified contact compliance.",
    )
    tire["solimp"] = provenance(
        DEFAULT_CONTACT_SOLIMP,
        "dmin dmax width midpoint power",
        "estimated",
        "Provisional seed default. Do not treat as identified contact compliance.",
    )
    seeded["seed_metadata"] = {
        "source": "tools/build_seeded_mujoco_model.py",
        "generated_at": timestamp(),
        "selected_runs": aggregate["selected_runs"],
        "notes": [
            "generated_bundle_only",
            "canonical_measurements_not_modified",
            "contact_values_are_provisional_defaults",
        ],
    }
    return seeded


def preflight_mesh_assets(robot_path: Path) -> list[Path]:
    root = ET.parse(robot_path).getroot()
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", "assets") if compiler is not None else "assets")
    if not meshdir.is_absolute():
        meshdir = robot_path.parent / meshdir
    missing: list[Path] = []
    for mesh in root.findall("./asset/mesh"):
        file_name = mesh.get("file")
        if not file_name or not file_name.endswith(".stl"):
            continue
        path = meshdir / file_name
        if not path.exists():
            missing.append(path)
    return missing


def run_mujoco_cli_checks(xml_path: Path) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    checks: dict[str, Any] = {
        "compile": "ok",
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
        "nsensor": int(model.nsensor),
        "scenarios": {},
    }
    if int(model.nu) != 2:
        raise BuildSeededModelError(f"expected 2 actuators, got {model.nu}")

    scenarios = {
        "zero_control_settle": ([0.0, 0.0], 50),
        "left_impulse": ([0.05, 0.0], 40),
        "right_impulse": ([0.0, 0.05], 40),
        "equal_wheel_forward": ([0.05, 0.05], 80),
        "opposite_wheel_yaw": ([0.05, -0.05], 80),
    }
    for name, (ctrl, steps) in scenarios.items():
        data = mujoco.MjData(model)
        for _ in range(steps):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        assert_finite(data.qpos, f"{name} qpos")
        assert_finite(data.qvel, f"{name} qvel")
        checks["scenarios"][name] = {
            "steps": steps,
            "ctrl": ctrl,
            "qpos_norm": float_norm(data.qpos),
            "qvel_norm": float_norm(data.qvel),
        }
    return checks


def replay_batch2_open_loop(xml_path: Path, run_dir: Path) -> dict[str, Any]:
    from scopik.gap import run_gap
    from scopik.profile import load_profile

    profile = load_profile(REPO_ROOT / "config/viz/finn.yaml")
    report = run_gap(profile, run_dir, model_path=xml_path)
    return {
        "status": "ok",
        "run_dir": portable_path(run_dir),
        "scope": "open-loop onboard-signal replay using the Finn Scopik profile",
        "metrics": report.summary,
        "replay": {} if report.sim_run is None else report.sim_run.meta,
    }


def selected_runs_payload(selected: dict[str, list[SysidRun]]) -> dict[str, Any]:
    return {
        batch: [
            {
                "run_dir": str(run.run_dir),
                "derived": str(run.derived_path),
                "rows": run.rows,
                "duration_s": run.duration_s,
            }
            for run in runs
        ]
        for batch, runs in selected.items()
    }


def aggregation_markdown(selected: dict[str, list[SysidRun]], aggregate: dict[str, Any]) -> str:
    lines = [
        "# Seeded MuJoCo Aggregation",
        "",
        "This is a generated seed-model bundle. It does not mutate canonical measurements.",
        "",
        "## Selected Runs",
    ]
    for batch, runs in selected.items():
        lines.append(f"### {batch}")
        for run in runs:
            lines.append(f"- `{run.run_dir}` rows={run.rows} duration_s={run.duration_s}")
    lines.extend(["", "## Actuator Seeds"])
    for side, values in aggregate["actuators"].items():
        lines.append(
            f"- {side}: sign={values['command_sign']}, limit={values['torque_limit_nm']}, "
            f"frictionloss={values['frictionloss_nm']}, damping={values['damping_nm_s_per_rad']}, "
            f"armature={values['armature_kg_m2']}"
        )
    lines.extend(
        [
            "",
            "## Contact Seeds",
            "- friction, solref, and solimp are provisional defaults.",
            "- Batch 2 lower bounds and loaded-loss diagnostics are preserved in JSON "
            "but not treated as identified contact parameters.",
            "",
            "## Do Not Infer",
        ]
    )
    lines.extend(f"- {item}" for item in aggregate["do_not_infer"])
    return "\n".join(lines) + "\n"


def validation_markdown(validation: dict[str, Any]) -> str:
    lines = [
        "# Seeded MuJoCo Validation",
        "",
        f"- status: `{validation.get('status')}`",
        f"- model_xml: `{validation.get('model_xml')}`",
        "",
        "## Mesh Assets",
    ]
    mesh = validation.get("mesh_assets", {})
    lines.append(f"- status: `{mesh.get('status')}`")
    for missing in mesh.get("missing", []):
        lines.append(f"- missing: `{missing}`")
    lines.extend(["", "## Checks"])
    checks = validation.get("checks", {})
    if checks:
        lines.append(f"- compile: `{checks.get('compile')}`")
        lines.append(f"- actuators: `{checks.get('nu')}`")
        for name, item in checks.get("scenarios", {}).items():
            lines.append(f"- {name}: qvel_norm={item.get('qvel_norm')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Scope", f"- {validation.get('onboard_replay_scope')}"])
    errors = validation.get("errors", [])
    lines.extend(["", "## Errors"])
    lines.extend(f"- {item}" for item in errors) if errors else lines.append("- none")
    return "\n".join(lines) + "\n"


def replay_markdown(replay: dict[str, Any]) -> str:
    lines = [
        "# Batch 2 Open-Loop Replay Metrics",
        "",
        f"- status: `{replay.get('status')}`",
        f"- run_dir: `{replay.get('run_dir')}`",
        f"- scope: {replay.get('scope')}",
        "",
        "## Metrics",
    ]
    for key, metric in replay.get("metrics", {}).items():
        lines.append(
            f"- {key}: rmse={metric.get('rmse')}, mae={metric.get('mae')}, "
            f"samples={metric.get('sample_count')}"
        )
    return "\n".join(lines) + "\n"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BuildSeededModelError(f"missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise BuildSeededModelError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_path(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def numeric(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def int_numeric(value: Any) -> int | None:
    number = numeric(value)
    if not math.isfinite(number):
        return None
    return int(number)


def median_path(runs: list[SysidRun], path: list[str]) -> float:
    values = [numeric(get_path(run.derived, path)) for run in runs]
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(values):
        raise BuildSeededModelError(f"cannot aggregate non-finite values at {'.'.join(path)}")
    return float(median(finite))


def measurement_value(node: Any) -> float | None:
    value = node.get("value") if isinstance(node, dict) else node
    number = numeric(value)
    return float(number) if math.isfinite(number) else None


def provenance(value: Any, unit: str, source: str, notes: str | None = None) -> dict[str, Any]:
    item = {"value": value, "unit": unit, "source": source}
    if notes:
        item["notes"] = notes
    return item


def missing_assets_message(missing_assets: list[Path]) -> str:
    first = ", ".join(path.name for path in missing_assets[:5])
    suffix = "" if len(missing_assets) <= 5 else f", ... ({len(missing_assets)} total)"
    parents = {path.parent for path in missing_assets}
    location = str(next(iter(parents))) if len(parents) == 1 else "mesh asset directories"
    return (
        "MuJoCo compile validation requested, but STL mesh assets are missing under "
        f"{location}: {first}{suffix}. Regenerate/export the Onshape-to-Robot "
        "STL assets before running --validate-mujoco or --replay-batch2."
    )


def assert_finite(values: Any, label: str) -> None:
    for value in values:
        if not math.isfinite(float(value)):
            raise BuildSeededModelError(f"{label} contains non-finite values")


def float_norm(values: Any) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def timestamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
