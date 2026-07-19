"""End-to-end: synthetic run -> gap pipeline -> .rrd + gap.json + history."""

import json
import subprocess
import sys

import pytest
from scopik.gap import run_gap
from scopik.profile import load_profile

pytest.importorskip("mujoco")


def test_run_gap_pipeline(spinner):
    profile = load_profile(spinner["profile"])
    report = run_gap(profile, spinner["run_dir"])

    assert report.sim_run is not None
    assert [pair.name for pair in report.results] == ["vel"]
    assert report.summary["vel"]["sample_count"] > 100
    # Events include the log file entries plus phase-change annotations.
    labels = [message for _, message in report.events]
    assert any("run_started" in label for label in labels)
    assert any(label == "phase: coast" for label in labels)
    assert "| vel |" in report.markdown()


def test_run_gap_no_replay(spinner):
    profile = load_profile(spinner["profile"])
    report = run_gap(profile, spinner["run_dir"], replay=False)
    assert report.sim_run is None
    assert report.results == []


def test_cli_gap_writes_rrd_json_and_history(spinner):
    pytest.importorskip("rerun")
    rrd_path = spinner["root"] / "out.rrd"
    history_path = spinner["root"] / "history.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scopik.cli",
            "gap",
            "--profile",
            str(spinner["profile"]),
            "--run",
            str(spinner["run_dir"]),
            "--save",
            str(rrd_path),
            "--history",
            str(history_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert rrd_path.exists() and rrd_path.stat().st_size > 1000
    assert (spinner["run_dir"] / "scopik_gap.json").exists()
    payload = json.loads((spinner["run_dir"] / "scopik_gap.json").read_text())
    assert "vel" in payload["metrics"]
    assert payload["phase_stats"]
    assert {"bias", "rms", "gain", "correlation", "lag_s"} <= set(payload["phase_stats"][0])
    assert payload["findings"]

    lines = history_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["profile"] == "spinner"
    assert "vel" in record["metrics"]
    assert len(record["model_sha256_12"]) == 12
