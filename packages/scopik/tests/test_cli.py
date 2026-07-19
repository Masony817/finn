import pytest
from scopik.cli import main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "gap" in capsys.readouterr().out


def test_gap_help_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        main(["gap", "--help"])
    assert excinfo.value.code == 0


def test_reserved_commands_exit_2(capsys):
    assert main(["record"]) == 2
    assert main(["live"]) == 2
    assert "not built yet" in capsys.readouterr().err


def test_missing_profile_is_concise_error(tmp_path, capsys):
    code = main(["gap", "--profile", str(tmp_path / "nope.yaml"), "--run", str(tmp_path)])
    assert code == 2
    assert "ERROR:" in capsys.readouterr().err
