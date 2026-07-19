import numpy as np
import pytest
from scopik.datamodel import ScopikError
from scopik.profile import load_profile
from scopik.sources import load_events, load_run, phase_changes


def test_load_run_parses_signals_and_text(spinner):
    profile = load_profile(spinner["profile"])
    run = load_run(spinner["run_dir"], profile)

    assert run.label == "real"
    assert set(run.signals) == {"vel_rad_s", "cmd_nm"}
    times = run.require_times()
    assert times[0] == 0.0
    assert times[1] == pytest.approx(0.01)  # 100 Hz, us -> s applied
    # text columns captured for mostly-non-numeric columns
    assert "state" in run.text_columns
    assert "phase" in run.text_columns
    assert run.text_columns["phase"][0] == "pulse"
    # every column also has a numeric view (NaN where unparseable)
    assert np.isnan(run.columns["state"]).all()


def test_torn_lines_are_skipped(tmp_path, spinner):
    profile = load_profile(spinner["profile"])
    telemetry = spinner["run_dir"] / "telemetry.csv"
    content = telemetry.read_text() + "data,999999,run\n"  # truncated row
    telemetry.write_text(content)
    run = load_run(spinner["run_dir"], profile)
    assert len(run.require_times()) == 200  # torn line dropped


def test_missing_header_marker_raises(tmp_path, spinner):
    profile = load_profile(spinner["profile"])
    (spinner["run_dir"] / "telemetry.csv").write_text("data,foo,bar\ndata,1,2\n", encoding="utf-8")
    with pytest.raises(ScopikError, match="no header line"):
        load_run(spinner["run_dir"], profile)


def test_missing_time_column_raises(spinner):
    from scopik.sources import build_run_data

    profile = load_profile(spinner["profile"])
    with pytest.raises(ScopikError, match="time column"):
        build_run_data("real", {"foo": np.array([1.0])}, profile)


def test_load_events(spinner):
    profile = load_profile(spinner["profile"])
    events = load_events(spinner["run_dir"], profile)
    assert len(events) == 2
    assert events[0][0] == 0.0
    assert "run_started" in events[0][1]
    assert events[1][0] == pytest.approx(1.0)


def test_phase_changes(spinner):
    profile = load_profile(spinner["profile"])
    run = load_run(spinner["run_dir"], profile)
    changes = phase_changes(run, "phase")
    assert [label for _, label in changes] == ["phase: pulse", "phase: coast"]
    assert changes[1][0] == pytest.approx(1.0)


def test_events_absent_is_empty(tmp_path, spinner):
    profile = load_profile(spinner["profile"])
    (spinner["run_dir"] / "events.log").unlink()
    assert load_events(spinner["run_dir"], profile) == []
