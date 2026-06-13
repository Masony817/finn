from __future__ import annotations

import argparse
import json
import os
import select
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "firmware" / "finn-mcu" / "tools"
sys.path.insert(0, str(TOOLS))

import postprocess_sysid_batch1 as batch1_post  # noqa: E402
import serial_log_capture as slc  # noqa: E402


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(pass_marker=[",batch1_complete,"], fail_marker=[",failsafe,"])
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_classify_line_pass_only_from_pending():
    args = _args()
    assert slc.classify_line("event,2,batch1_complete,complete,x", "pending", args) == "pass"
    # a pass marker after we've already left pending must not override
    assert slc.classify_line("event,2,batch1_complete,complete,x", "fail", args) == "fail"


def test_classify_line_fail_marker_always_wins():
    args = _args()
    status = slc.classify_line(
        "event,2,failsafe,fault,left_moteus_no_reply",
        "pending",
        args,
    )
    assert status == "fail"
    assert slc.classify_line("data,1,running,0,settle_stop", "pending", args) == "pending"


def test_final_dir_for_avoids_collisions(tmp_path: Path):
    first = slc.final_dir_for(tmp_path, "batch_1", "pass", "stamp")
    first.mkdir(parents=True)
    second = slc.final_dir_for(tmp_path, "batch_1", "pass", "stamp")
    assert first != second
    assert second.name == "stamp_001"


def test_write_manifest_is_atomic_json(tmp_path: Path):
    path = tmp_path / "manifest.json"
    slc.write_manifest(path, {"status": "pass", "rows": 3})
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text()) == {"status": "pass", "rows": 3}


def test_send_command_appends_single_newline():
    read_fd, write_fd = os.pipe()
    try:
        slc.send_command(write_fd, "STATUS\r\n")
        ready, _, _ = select.select([read_fd], [], [], 1.0)
        assert read_fd in ready
        assert os.read(read_fd, 1024) == b"STATUS\n"
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_print_teensy_hides_data_by_default(capsys):
    args = argparse.Namespace(show_telemetry=False)
    slc.print_teensy("data,123,running_batch1,0,settle_stop", args)
    slc.print_teensy("event,123,armed,armed_idle,awaiting_run_batch1", args)
    assert capsys.readouterr().out == "teensy> event,123,armed,armed_idle,awaiting_run_batch1\n"


def test_print_teensy_can_show_data(capsys):
    args = argparse.Namespace(show_telemetry=True)
    slc.print_teensy("data,123,running_batch1,0,settle_stop", args)
    assert capsys.readouterr().out == "teensy> data,123,running_batch1,0,settle_stop\n"


def test_batch1_postprocess_skips_schema_line(tmp_path: Path):
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "schema,batch1_v1\n"
        "data,t_us,state,phase_index,phase,armed\n"
        "data,1,safe_idle,-1,idle,0\n"
        "data,2,running_batch1,0,settle_stop,1\n"
    )
    rows = batch1_post.read_batch1_rows(telemetry)
    assert [row["phase"] for row in rows] == ["idle", "settle_stop"]


def test_arm_command_override_replaces_default():
    # action="append" must not be paired with a non-empty default, or a user
    # value gets appended to the default instead of replacing it.
    parser = slc.build_parser()
    assert slc.normalize_args(parser.parse_args([])).arm_command == []
    assert slc.normalize_args(parser.parse_args(["--arm-command", "GO"])).arm_command == ["GO"]


def test_run_postprocess_tolerates_stray_braces(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rc, _, _ = slc.run_postprocess(["true", "{run_dir}", "literal{brace}"], run_dir)
    assert rc == 0
