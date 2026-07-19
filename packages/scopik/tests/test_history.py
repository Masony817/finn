import numpy as np
from scopik.datamodel import RunData, Signal
from scopik.history import append_history, build_record, read_history
from scopik.metrics import compare_runs
from scopik.profile import ComparePair, load_profile


def test_history_round_trip(tmp_path, spinner):
    profile = load_profile(spinner["profile"])
    times = np.arange(20) * 0.01
    real = RunData(label="real", times=times)
    real.signals["v"] = Signal(name="v", unit="u", times=times, values=np.zeros(20))
    sim = RunData(label="sim", times=times)
    sim.signals["v"] = Signal(name="v", unit="u", times=times, values=np.ones(20))
    results = compare_runs(real, sim, (ComparePair(name="v", real="v", sim="v"),))

    history_path = tmp_path / "h.jsonl"
    record = build_record(profile, spinner["run_dir"], spinner["model"], results)
    append_history(history_path, record)
    append_history(history_path, record)

    records = read_history(history_path)
    assert len(records) == 2
    assert records[0]["metrics"]["v"]["rmse"] == 1.0
    assert records[0]["profile"] == "spinner"


def test_read_missing_history_is_empty(tmp_path):
    assert read_history(tmp_path / "none.jsonl") == []
