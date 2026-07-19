import pytest
from scopik.profile import ProfileError, load_profile

MINIMAL = """
name: bot
model: bot.xml
source: {type: prefixed_csv, file: telemetry.csv}
time: {column: t_us, transform: us_to_s}
"""


def test_minimal_profile_loads(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    profile = load_profile(path)
    assert profile.name == "bot"
    assert profile.model_path == tmp_path / "bot.xml"
    assert profile.replay is None
    assert profile.compare == ()
    assert profile.events is None


def test_spinner_profile_loads(spinner):
    profile = load_profile(spinner["profile"])
    assert profile.signal_groups == ["motion", "commands"]
    assert profile.replay is not None
    assert profile.replay.actuators == {"motor_hinge": "cmd_nm"}
    assert profile.events is not None
    assert profile.events.phase_column == "phase"


def test_missing_profile_raises(tmp_path):
    with pytest.raises(ProfileError, match="not found"):
        load_profile(tmp_path / "nope.yaml")


def test_missing_name_raises(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("model: x.xml\nsource: {type: csv, file: f}\ntime: {column: t}\n")
    with pytest.raises(ProfileError, match="name"):
        load_profile(path)


def test_compare_referencing_unknown_signal_raises(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        MINIMAL + "signals:\n  a: {unit: x}\ncompare:\n  - {name: c, real: nope, sim: a}\n"
    )
    with pytest.raises(ProfileError, match="unknown real signal"):
        load_profile(path)


def test_duplicate_signal_rename_raises(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL + "signals:\n  a: {rename: same}\n  b: {rename: same}\n")
    with pytest.raises(ProfileError, match="duplicate signal names"):
        load_profile(path)


def test_bad_transform_in_signal_raises(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(MINIMAL + "signals:\n  a: {transform: bogus}\n")
    with pytest.raises(ProfileError, match="unknown transform"):
        load_profile(path)
