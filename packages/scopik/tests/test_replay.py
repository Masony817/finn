import numpy as np
import pytest
from scopik.datamodel import ScopikError
from scopik.profile import load_profile
from scopik.reconstruct.replay import replay_commands
from scopik.sources import load_run

pytest.importorskip("mujoco")


def test_replay_produces_sim_run(spinner):
    profile = load_profile(spinner["profile"])
    real_run = load_run(spinner["run_dir"], profile)
    sim_run = replay_commands(profile, real_run)

    assert sim_run.label == "sim"
    assert set(sim_run.signals) == {"vel_rad_s"}
    signal = sim_run.signals["vel_rad_s"]
    assert len(signal.times) == len(real_run.require_times()) - 1
    # Positive constant torque during the pulse phase must spin the hinge forward.
    assert signal.values[50] > 0.1
    # During coast (zero torque + damping) speed must not grow.
    assert signal.values[-1] <= signal.values[99] + 1e-9


def test_replay_is_deterministic(spinner):
    profile = load_profile(spinner["profile"])
    real_run = load_run(spinner["run_dir"], profile)
    first = replay_commands(profile, real_run)
    second = replay_commands(profile, real_run)
    assert np.array_equal(first.signals["vel_rad_s"].values, second.signals["vel_rad_s"].values)


def test_replay_unknown_actuator_raises(spinner, tmp_path):
    profile = load_profile(spinner["profile"])
    bad = (spinner["profile"]).read_text().replace("motor_hinge:", "motor_nope:")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(bad)
    bad_profile = load_profile(bad_path)
    real_run = load_run(spinner["run_dir"], profile)
    with pytest.raises(ScopikError, match="no actuator"):
        replay_commands(bad_profile, real_run, model_path=spinner["model"])


def test_replay_without_replay_section_raises(spinner, tmp_path):
    minimal = tmp_path / "min.yaml"
    minimal.write_text(
        "name: m\nmodel: spinner.xml\n"
        "source: {type: prefixed_csv, file: telemetry.csv}\n"
        "time: {column: t_us, transform: us_to_s}\n"
    )
    profile = load_profile(minimal)
    full_profile = load_profile(spinner["profile"])
    real_run = load_run(spinner["run_dir"], full_profile)
    with pytest.raises(ScopikError, match="no replay"):
        replay_commands(profile, real_run, model_path=spinner["model"])
