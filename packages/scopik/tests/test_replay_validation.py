"""Reject invented inputs and expose replay timing adjustments."""

from dataclasses import replace

import numpy as np
import pytest
from scopik.datamodel import ScopikError
from scopik.profile import ProfileError, load_profile, parse_replay
from scopik.reconstruct.replay import replay_commands
from scopik.sources import load_run


@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_sensor_index_requires_a_nonnegative_integer(value, tmp_path):
    with pytest.raises(ProfileError, match="non-negative integer"):
        parse_replay(
            {"actuators": {"motor": "torque"}, "sample": {"x": {"sensor": "gyro", "index": value}}},
            tmp_path / "profile.yaml",
        )


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_gravity_compensation_requires_a_yaml_boolean(value, tmp_path):
    with pytest.raises(ProfileError, match="must be a boolean"):
        parse_replay(
            {
                "actuators": {"motor": "torque"},
                "sample": {"x": {"sensor": "accel", "gravity_compensated": value}},
            },
            tmp_path / "profile.yaml",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_commands_are_not_replaced_with_zero(spinner, value):
    profile = load_profile(spinner["profile"])
    run = load_run(spinner["run_dir"], profile)
    column = next(iter(profile.replay.actuators.values()))
    run.columns[column][1] = value
    with pytest.raises(ScopikError, match="one finite value per row"):
        replay_commands(profile, run)


@pytest.mark.parametrize("value", [float("nan"), 0.0, -1.0])
def test_replay_requires_increasing_finite_timestamps(spinner, value):
    profile = load_profile(spinner["profile"])
    run = load_run(spinner["run_dir"], profile)
    run.times[1] = value
    with pytest.raises(ScopikError, match="strictly increasing"):
        replay_commands(profile, run)


def test_programmatic_profiles_cannot_read_a_previous_sensor(spinner):
    profile = load_profile(spinner["profile"])
    run = load_run(spinner["run_dir"], profile)
    sample = replace(profile.replay.samples[0], index=-1)
    profile = replace(profile, replay=replace(profile.replay, samples=(sample,)))
    with pytest.raises(ScopikError, match="out of range"):
        replay_commands(profile, run)


def test_clipped_gaps_are_reported_not_disguised_as_matching_time(spinner):
    profile = load_profile(spinner["profile"])
    run = load_run(spinner["run_dir"], profile)
    run.times = np.arange(len(run.times)) * 0.2
    with pytest.warns(UserWarning, match="integration time differs"):
        sim = replay_commands(profile, run)
    timing = sim.meta["timing"]
    assert timing["clipped_intervals"] == len(run.times) - 1
    assert timing["simulated_duration_s"] < timing["recorded_duration_s"]
    assert timing["max_abs_time_error_s"] > 0
