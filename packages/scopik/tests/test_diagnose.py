import numpy as np
from scopik.datamodel import RunData
from scopik.diagnose import (
    PhaseSegment,
    analyze,
    estimate_gain,
    estimate_lag_s,
    phase_segments,
    phase_stats,
    stats_markdown,
)
from scopik.metrics import PairResult, rolling_rmse, summarize


def make_pair(times, real, sim, name="sig", unit="u"):
    residual = sim - real
    return PairResult(
        name=name,
        unit=unit,
        times=times,
        real=real,
        sim=sim,
        residual=residual,
        rolling_rmse=rolling_rmse(residual, times, 1.0),
        summary=summarize(real, sim),
    )


def test_phase_segments_merges_consecutive():
    run = RunData(label="real", times=np.arange(6) * 0.1)
    run.text_columns["phase"] = ["a", "a", "b", "b", "b", "c"]
    segments = phase_segments(run, "phase")
    assert [s.label for s in segments] == ["a", "b", "c"]
    assert segments[1].start_s == 0.2
    assert segments[0].end_s == 0.1


def test_phase_stats_do_not_duplicate_boundary_samples():
    times = np.arange(60) * 0.01
    pair = make_pair(times, np.ones(60), np.ones(60))
    segments = [PhaseSegment("first", 0.0, 0.29), PhaseSegment("second", 0.3, 0.59)]
    stats, _ = analyze([pair], segments)
    assert [s.sample_count for s in stats] == [30, 30]
    assert sum(s.sample_count for s in stats) == len(times)


def test_concentration_finding():
    times = np.arange(300) * 0.01
    real = np.sin(2 * np.pi * times)
    sim = real.copy()
    sim[100:200] += 0.8  # big error only in the middle phase
    segments = [
        PhaseSegment("quiet1", 0.0, 0.99),
        PhaseSegment("bad", 1.0, 1.99),
        PhaseSegment("quiet2", 2.0, 2.99),
    ]
    _, findings = analyze([make_pair(times, real, sim)], segments)
    concentration = [f for f in findings if f.kind == "concentration"]
    assert len(concentration) == 1
    assert concentration[0].phase == "bad"
    assert concentration[0].severity == "issue"


def test_static_offset_finding():
    times = np.arange(300) * 0.01
    real = np.concatenate([np.zeros(150), np.sin(8 * times[150:])])  # quiet then active
    sim = real.copy()
    sim[:150] += 0.4  # sim drifts while real is still
    segments = [PhaseSegment("still", 0.0, 1.49), PhaseSegment("active", 1.5, 2.99)]
    _, findings = analyze([make_pair(times, real, sim)], segments)
    assert any(f.kind == "static_offset" and f.phase == "still" for f in findings)


def test_steady_nonzero_motion_is_not_called_stationary_drift():
    times = np.arange(300) * 0.01
    real = np.concatenate([np.sin(8 * times[:150]), np.full(150, 2.5)])
    sim = real.copy()
    sim[150:] += 0.5
    segments = [PhaseSegment("active", 0.0, 1.49), PhaseSegment("coast", 1.5, 2.99)]
    _, findings = analyze([make_pair(times, real, sim)], segments)
    assert not any(f.kind == "static_offset" and f.phase == "coast" for f in findings)


def test_gain_estimation_and_finding():
    times = np.arange(500) * 0.01
    real = np.sin(2 * np.pi * 0.7 * times)
    sim = 1.5 * real
    gain, correlation = estimate_gain(make_pair(times, real, sim))
    assert gain is not None and abs(gain - 1.5) < 1e-9
    assert correlation is not None and correlation > 0.99
    _, findings = analyze([make_pair(times, real, sim)], [])
    assert any(f.kind == "gain" for f in findings)


def test_lag_estimation_and_finding():
    times = np.arange(1000) * 0.01
    real = np.sin(2 * np.pi * 0.5 * times)
    sim = np.roll(real, 5)  # sim lags real by 50 ms
    pair = make_pair(times, real, sim)
    lag = estimate_lag_s(pair)
    assert abs(lag - 0.05) < 0.011
    _, findings = analyze([pair], [])
    assert any(f.kind == "lag" for f in findings)


def test_phase_stats_include_gain_and_lag():
    times = np.arange(400) * 0.01
    real = np.sin(2 * np.pi * 0.8 * times)
    sim = np.roll(1.4 * real, 4)
    segments = [PhaseSegment("excitation", 0.0, 3.99)]
    stats, findings = analyze([make_pair(times, real, sim)], segments)
    assert len(stats) == 1
    assert stats[0].gain is not None and stats[0].gain > 1.2
    assert stats[0].correlation is not None and stats[0].correlation > 0.9
    assert stats[0].lag_s is not None and abs(stats[0].lag_s - 0.04) < 0.011
    assert {finding.kind for finding in findings} >= {"gain", "lag"}


def test_phase_gain_and_lag_are_null_when_real_signal_has_no_activity():
    times = np.arange(200) * 0.01
    real = 1e-6 * np.sin(2 * np.pi * times)
    sim = 4_000 * real
    segments = [PhaseSegment("sensor_noise", 0.0, 1.99)]
    stats = phase_stats(make_pair(times, real, sim), segments, min_real_std=1e-3)
    assert stats[0].gain is None
    assert stats[0].correlation is None
    assert stats[0].lag_s is None


def test_lag_at_search_boundary_is_not_reported_for_phase():
    times = np.arange(400) * 0.01
    real = times.copy()
    sim = np.roll(real, 25)
    segments = [PhaseSegment("ramp", 0.0, 3.99)]
    stats = phase_stats(make_pair(times, real, sim), segments)
    assert stats[0].lag_s is None


def test_clean_pair_gets_ok_finding():
    times = np.arange(400) * 0.01
    real = np.sin(2 * np.pi * times)
    rng = np.random.default_rng(0)
    sim = real + 0.005 * rng.standard_normal(len(real))
    _, findings = analyze([make_pair(times, real, sim)], [])
    assert [f.kind for f in findings] == ["clean"]
    assert findings[0].severity == "ok"


def test_large_unstructured_gap_is_not_called_clean():
    times = np.arange(400) * 0.01
    real = np.sin(2 * np.pi * times)
    sim = np.cos(2 * np.pi * 3.7 * times)
    _, findings = analyze([make_pair(times, real, sim)], [])
    assert [f.kind for f in findings] == ["unclassified"]
    assert findings[0].severity == "note"


def test_markdown_renders_findings_and_table():
    times = np.arange(300) * 0.01
    real = np.sin(2 * np.pi * times)
    sim = real + 0.3
    segments = [PhaseSegment("all", 0.0, 2.99)]
    stats, findings = analyze([make_pair(times, real, sim)], segments)
    md = stats_markdown(stats, findings)
    assert "## Findings" in md
    assert "| sig | all |" in md
    assert "lag (ms)" in md
