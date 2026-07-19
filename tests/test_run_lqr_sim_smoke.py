"""Smoke test: the committed seed model runs end-to-end through the LQR sim.

This is deliberately minimal. Its job is to catch the "works on my machine"
regression -- the seed-model bundle going missing, or the model/script drifting
apart -- on a fresh clone in CI, not to assert controller quality.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_lqr_sim.py"
MODEL = ROOT / "sim" / "generated" / "seeded" / "latest" / "finn.seeded.sim.xml"

SPEC = importlib.util.spec_from_file_location("run_lqr_sim", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rls = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rls
SPEC.loader.exec_module(rls)


def test_committed_seed_model_is_present():
    # If this fails, a fresh clone cannot run the sim -- the bundle must ship.
    assert MODEL.exists(), f"missing committed seed model: {MODEL}"


def test_viewer_is_opt_in():
    assert rls.parse_args([]).viewer is False
    assert rls.parse_args(["--viewer"]).viewer is True


def test_seeded_model_has_expected_forward_balance_trim():
    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    trim_rad = rls.estimate_balance_trim_pitch_rad(model)

    assert 0.03 < trim_rad < 0.05


def test_lqr_sim_runs_on_committed_model(tmp_path: Path):
    args = rls.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path),
            "--duration-s",
            "0.2",
            "--no-plot",
        ]
    )
    result = rls.run(args)

    assert result["status"] in {"pass", "failed"}  # ran to completion, not errored
    gain = result["lqr"]["gain"]  # type: ignore[index]
    assert len(gain) == 1 and len(gain[0]) == len(rls.STATE_NAMES)
    assert 0.03 < result["control"]["target_pitch_rad"] < 0.05  # type: ignore[index]
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "timeseries.csv").exists()


def test_default_position_hold_limits_30_second_drift(tmp_path: Path):
    args = rls.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path),
            "--duration-s",
            "30",
            "--no-plot",
        ]
    )
    result = rls.run(args)
    metrics = result["metrics"]

    assert result["status"] == "pass"
    assert metrics["max_abs_position_error_m"] < 0.15  # type: ignore[index]
    assert metrics["final_abs_position_error_m"] < 0.05  # type: ignore[index]


def test_firmware_header_is_exported_from_a_passing_sim(tmp_path: Path):
    header = tmp_path / "lqr_seeded_config.h"
    args = rls.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path / "run"),
            "--duration-s",
            "1",
            "--no-plot",
            "--firmware-header",
            str(header),
        ]
    )

    result = rls.run(args)
    text = header.read_text(encoding="utf-8")

    assert result["status"] == "pass"
    assert result["model_sha256_12"] in text
    assert "kGainPitch" in text
    assert "kRealLeftEncoderForwardSign = 1.0f" in text
    assert "kPitchDirectionBenchVerified = false" in text
    assert "kWheelEncoderDirectionsBenchVerified = false" in text
