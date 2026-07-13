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
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "timeseries.csv").exists()
