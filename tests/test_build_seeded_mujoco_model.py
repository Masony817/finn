from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_seeded_mujoco_model.py"
SPEC = importlib.util.spec_from_file_location("build_seeded_mujoco_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bsm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bsm
SPEC.loader.exec_module(bsm)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def pv(value, unit="unit", source="measured"):
    return {"value": value, "unit": unit, "source": source}


def make_batch1_run(
    sysid_root: Path,
    name: str,
    *,
    rows: int = 12_000,
    faults: int = 0,
    left_sign: int = -1,
    right_sign: int = 1,
    left_friction: float = 0.1,
    right_friction: float = 0.2,
) -> Path:
    run_dir = sysid_root / "batch_1_pass" / name
    derived = {
        "schema_version": 1,
        "source": "finn_mcu_batch1_sysid",
        "run": {"rows": rows, "duration_s": 120.0},
        "health": {"fault_rows": faults},
        "wheels": {
            "left": batch1_wheel(left_sign, left_friction, 0.004, 0.0, 0.25, 0.94),
            "right": batch1_wheel(right_sign, right_friction, 0.005, 0.0, 0.25, 0.93),
        },
    }
    write_yaml(run_dir / "postprocess" / "derived.yaml", derived)
    return run_dir


def batch1_wheel(
    sign: int,
    friction: float,
    damping: float,
    armature: float,
    torque_limit: float,
    gain: float,
) -> dict:
    return {
        "suggested_sim": {
            "command_sign": {"value": sign},
            "frictionloss": {"value": friction},
            "damping": {"value": damping},
            "armature": {"value": armature},
            "torque_limit_nm": {"value": torque_limit},
        },
        "diagnostics": {"actuator_tracking": {"gain": gain}},
    }


def make_batch2_run(
    sysid_root: Path,
    name: str,
    *,
    rows: int = 12_000,
    fault_rows: int = 0,
    failsafe_events: int = 0,
    readiness: bool = True,
    delay_s: float = 0.011,
    left_gain: float = 1.0,
    right_gain: float = 1.0,
) -> Path:
    run_dir = sysid_root / "batch_2_pass" / name
    derived = {
        "schema_version": 1,
        "source": "finn_mcu_batch2_loaded_ground_sysid",
        "metadata": {
            "rows": rows,
            "duration_s": 200.0,
            "fault_counts": {"fault_rows": fault_rows, "failsafe_events": failsafe_events},
        },
        "lqr_readiness": {"pass": readiness},
        "suggested_sim": {
            "wheels": {
                "actuator_tracking": {
                    "left": {"gain": left_gain, "bias_nm": 0.001, "rmse_nm": 0.02},
                    "right": {"gain": right_gain, "bias_nm": -0.001, "rmse_nm": 0.021},
                },
                "loaded_loss_correction": {
                    "friction_accel_m_s2": 0.05,
                    "linear_damping_s": 2.0,
                    "per_wheel_torque_nm": 0.04,
                },
                "loaded_radius_estimate": {"radius_m": 0.055},
                "yaw_response": {"effective_track_width_m": 0.34},
            },
            "contact": {
                "tire_friction_lower_bounds": {"straight_mu_lower_bound": 0.025}
            },
            "delays": {
                "command_to_measured_torque": {"average": {"delay_s": delay_s}}
            },
        },
    }
    write_yaml(run_dir / "postprocess" / "derived.yaml", derived)
    return run_dir


def base_measurements(*, with_width: bool = False) -> dict:
    left_width = pv(0.066, "m") if with_width else pv(None, "m", "todo")
    right_width = pv(0.066, "m") if with_width else pv(None, "m", "todo")
    return {
        "schema_version": 1,
        "robot": {
            "mass_kg": pv(8.44, "kg"),
            "wheel_track_width_m": pv(0.517, "m", "cad"),
            "com_height_m": pv(0.19, "m", "cad"),
        },
        "wheels": {
            "left": {
                "radius_m": pv(0.081, "m"),
                "width_m": left_width,
                "mass_kg": pv(2.1, "kg"),
                "gear_ratio": pv(1.0, "ratio"),
            },
            "right": {
                "radius_m": pv(0.081, "m"),
                "width_m": right_width,
                "mass_kg": pv(2.1, "kg"),
                "gear_ratio": pv(1.0, "ratio"),
            },
        },
        "contact": {"tire": {}},
    }


def test_discover_clean_runs_filters_and_selects_latest(tmp_path: Path):
    sysid = tmp_path / "sysid"
    make_batch1_run(sysid, "20260101_000000", rows=0)
    make_batch1_run(sysid, "20260102_000000", left_friction=0.1)
    make_batch1_run(sysid, "20260103_000000", left_friction=0.2)
    make_batch1_run(sysid, "20260104_000000", left_friction=0.3)
    make_batch1_run(sysid, "20260105_000000", left_friction=0.4)

    runs = bsm.discover_clean_runs("batch1", sysid / "batch_1_pass", 3)

    assert [run.run_dir.name for run in runs] == [
        "20260103_000000",
        "20260104_000000",
        "20260105_000000",
    ]


def test_select_runs_respects_explicit_pins_and_batch2_filters(tmp_path: Path):
    sysid = tmp_path / "sysid"
    b1 = make_batch1_run(sysid, "20260102_000000")
    make_batch2_run(sysid, "20260101_000000", failsafe_events=1)
    b2 = make_batch2_run(sysid, "20260102_000000")

    selected = bsm.select_runs(
        sysid_root=sysid,
        latest_count=1,
        batch1_runs=[b1],
        batch2_runs=[b2],
    )

    assert selected["batch1"][0].run_dir == b1
    assert selected["batch2"][0].run_dir == b2


def test_aggregate_uses_medians_and_rejects_sign_disagreement(tmp_path: Path):
    sysid = tmp_path / "sysid"
    b1_dirs = [
        make_batch1_run(sysid, "20260101_000000", left_friction=0.1),
        make_batch1_run(sysid, "20260102_000000", left_friction=0.3),
        make_batch1_run(sysid, "20260103_000000", left_friction=0.2),
    ]
    b2_dirs = [
        make_batch2_run(sysid, "20260101_000000", delay_s=0.010),
        make_batch2_run(sysid, "20260102_000000", delay_s=0.012),
        make_batch2_run(sysid, "20260103_000000", delay_s=0.011),
    ]
    batch1 = [bsm.load_sysid_run("batch1", path) for path in b1_dirs]
    batch2 = [bsm.load_sysid_run("batch2", path) for path in b2_dirs]

    aggregate = bsm.aggregate_runs(batch1, batch2)

    assert aggregate["actuators"]["left"]["frictionloss_nm"] == 0.2
    assert aggregate["batch2"]["command_to_measured_torque_delay_s"] == 0.011

    bad = make_batch1_run(sysid, "20260104_000000", left_sign=1)
    with pytest.raises(bsm.BuildSeededModelError, match="command sign disagreement"):
        bsm.aggregate_runs([*batch1[:2], bsm.load_sysid_run("batch1", bad)], batch2)


def test_seeded_measurements_fill_required_fields_without_mutating_canonical(tmp_path: Path):
    sysid = tmp_path / "sysid"
    batch1 = [bsm.load_sysid_run("batch1", make_batch1_run(sysid, "20260101_000000"))]
    batch2 = [bsm.load_sysid_run("batch2", make_batch2_run(sysid, "20260101_000000"))]
    aggregate = bsm.aggregate_runs(batch1, batch2)
    canonical = base_measurements(with_width=False)
    original = yaml.safe_dump(canonical)

    with pytest.raises(bsm.BuildSeededModelError, match="wheel width"):
        bsm.build_seeded_measurements(
            measurements=canonical, aggregate=aggregate, wheel_width_m=None
        )

    seeded = bsm.build_seeded_measurements(
        measurements=canonical, aggregate=aggregate, wheel_width_m=0.066
    )

    assert yaml.safe_dump(canonical) == original
    assert seeded["wheels"]["left"]["width_m"]["value"] == 0.066
    assert seeded["wheels"]["left"]["frictionloss"]["value"] == 0.1
    assert seeded["wheels"]["left"]["frictionloss"]["source"] == "moteus"
    assert seeded["contact"]["tire"]["friction"]["source"] == "estimated"
    assert "canonical_measurements_not_modified" in seeded["seed_metadata"]["notes"]


def test_preflight_mesh_assets_reports_missing_stls(tmp_path: Path):
    robot = tmp_path / "robot.xml"
    robot.write_text(
        """
        <mujoco>
          <compiler meshdir="assets"/>
          <asset><mesh file="missing.stl"/></asset>
          <worldbody/>
        </mujoco>
        """,
        encoding="utf-8",
    )

    missing = bsm.preflight_mesh_assets(robot)

    assert missing == [tmp_path / "assets" / "missing.stl"]
    assert "missing.stl" in bsm.missing_assets_message(missing)


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        bsm.parse_args(["--help"])
    assert exc.value.code == 0
    assert "Build a generated seeded Finn MuJoCo model" in capsys.readouterr().out


def test_report_only_writes_bundle_and_does_not_write_model(tmp_path: Path):
    sysid = tmp_path / "sysid"
    measurements = tmp_path / "measurements.yaml"
    out_dir = tmp_path / "bundle"
    make_batch1_run(sysid, "20260101_000000")
    make_batch2_run(sysid, "20260101_000000")
    write_yaml(measurements, base_measurements(with_width=True))

    args = Namespace(
        sysid_root=sysid,
        measurements=measurements,
        robot=ROOT / "sim/model/finn_robot.xml",
        scene=ROOT / "sim/model/scene.xml",
        config=ROOT / "sim/config/mujoco_postprocess.yaml",
        out_dir=out_dir,
        auto_select_latest=1,
        batch1_run=[],
        batch2_run=[],
        wheel_width_m=None,
        validate_mujoco=False,
        replay_batch2=False,
        report_only=True,
    )

    result = bsm.build_seeded_model(args)

    assert result["status"] == "ok"
    assert (out_dir / "selected_runs.json").exists()
    assert (out_dir / "aggregation_report.md").exists()
    assert (out_dir / "seeded_measurements.yaml").exists()
    assert not (out_dir / "finn.seeded.sim.xml").exists()
    validation = json.loads((out_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "report_only"
