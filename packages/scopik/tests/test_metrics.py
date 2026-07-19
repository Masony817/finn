import numpy as np
import pytest
from scopik.datamodel import RunData, ScopikError, Signal, TimeBase
from scopik.metrics import compare_runs, rolling_rmse, summarize, summary_markdown
from scopik.profile import ComparePair


def make_run(label, name, times, values):
    run = RunData(label=label, times=np.asarray(times, dtype=float))
    run.signals[name] = Signal(name=name, unit="u", times=times, values=values)
    return run


def test_summarize_hand_computed():
    real = np.array([0.0, 1.0, 2.0])
    sim = np.array([0.0, 2.0, 4.0])  # err = [0, 1, 2]
    s = summarize(real, sim)
    assert s["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert s["mae"] == pytest.approx(1.0)
    assert s["max_abs"] == pytest.approx(2.0)
    assert s["sample_count"] == 3


def test_summarize_all_nan_raises():
    with pytest.raises(ScopikError, match="no finite"):
        summarize(np.array([np.nan]), np.array([1.0]))


def test_rolling_rmse_flat_signal():
    times = np.arange(100) * 0.01
    residual = np.full(100, 2.0)
    out = rolling_rmse(residual, times, window_s=0.1)
    assert np.allclose(out, 2.0)


def test_rolling_rmse_windows_forget():
    times = np.arange(200) * 0.01
    residual = np.concatenate([np.full(100, 3.0), np.zeros(100)])
    out = rolling_rmse(residual, times, window_s=0.05)
    assert out[99] == pytest.approx(3.0)
    assert out[199] == pytest.approx(0.0)  # old error left the window


def test_compare_identical_timestamps_no_resampling():
    times = np.arange(50) * 0.01
    real = make_run("real", "v", times, np.sin(times))
    sim = make_run("sim", "v", times, np.sin(times) + 0.1)
    results = compare_runs(real, sim, (ComparePair(name="v", real="v", sim="v"),))
    assert len(results) == 1
    assert np.array_equal(results[0].times, times)
    assert results[0].summary["rmse"] == pytest.approx(0.1)


def test_compare_resamples_different_timebases():
    real_times = np.arange(100) * 0.01
    sim_times = np.arange(1, 100) * 0.01 + 0.001
    real = make_run("real", "v", real_times, 2.0 * real_times)
    sim = make_run("sim", "v", sim_times, 2.0 * sim_times)
    results = compare_runs(real, sim, (ComparePair(name="v", real="v", sim="v"),))
    assert results[0].summary["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_compare_skips_missing_signals():
    times = np.arange(10) * 0.1
    real = make_run("real", "v", times, times)
    sim = make_run("sim", "other", times, times)
    assert compare_runs(real, sim, (ComparePair(name="v", real="v", sim="v"),)) == []


def test_timebase_no_overlap_raises():
    with pytest.raises(ScopikError, match="do not overlap"):
        TimeBase.overlap(np.array([0.0, 1.0]), np.array([2.0, 3.0]))


def test_summary_markdown_contains_rows():
    times = np.arange(50) * 0.01
    real = make_run("real", "v", times, np.zeros(50))
    sim = make_run("sim", "v", times, np.ones(50))
    results = compare_runs(real, sim, (ComparePair(name="v", real="v", sim="v"),))
    md = summary_markdown(results, scope_note="scope")
    assert "| v |" in md
    assert "*scope*" in md
