from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import select
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "firmware" / "finn-mcu" / "tools"
sys.path.insert(0, str(TOOLS))

import postprocess_sysid_batch1 as batch1_post  # noqa: E402
import postprocess_sysid_batch2 as batch2_post  # noqa: E402
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


def test_upload_firmware_passes_upload_port(monkeypatch):
    calls = []

    class Completed:
        returncode = 0

    def fake_run(cmd, cwd, env, check):
        calls.append((cmd, cwd, env.get("PLATFORMIO_CORE_DIR"), check))
        return Completed()

    monkeypatch.setattr(slc, "find_platformio_executable", lambda: "pio")
    monkeypatch.setattr(slc.subprocess, "run", fake_run)

    rc = slc.upload_firmware("sysid_batch1_wheels_offground", "/dev/cu.usbmodem123")

    assert rc == 0
    assert calls == [
        (
            [
                "pio",
                "run",
                "-e",
                "sysid_batch1_wheels_offground",
                "-t",
                "upload",
                "--upload-port",
                "/dev/cu.usbmodem123",
            ],
            slc.FIRMWARE_ROOT,
            str(slc.REPO_ROOT / ".platformio"),
            False,
        )
    ]


def test_upload_firmware_can_omit_upload_port(monkeypatch):
    calls = []

    class Completed:
        returncode = 0

    def fake_run(cmd, cwd, env, check):
        calls.append((cmd, env.get("PLATFORMIO_CORE_DIR")))
        return Completed()

    monkeypatch.setattr(slc, "find_platformio_executable", lambda: "pio")
    monkeypatch.setattr(slc.subprocess, "run", fake_run)

    assert slc.upload_firmware("main") == 0
    assert calls == [
        (
            ["pio", "run", "-e", "main", "-t", "upload"],
            str(slc.REPO_ROOT / ".platformio"),
        )
    ]


def test_upload_firmware_reports_missing_platformio(monkeypatch, capsys):
    monkeypatch.setattr(slc, "find_platformio_executable", lambda: None)

    assert slc.upload_firmware("main") == 127
    assert "PlatformIO executable not found" in capsys.readouterr().err


def test_run_capture_detects_port_before_upload(monkeypatch):
    calls = []
    args = slc.build_parser().parse_args(["--upload-env", "sysid_batch1_wheels_offground"])

    monkeypatch.setattr(slc, "find_serial_port", lambda: "/dev/cu.usbmodem123")

    def fake_upload(env, port=None):
        calls.append((env, port))
        return 2

    monkeypatch.setattr(slc, "upload_firmware", fake_upload)

    assert slc.run_capture(args) == 2
    assert calls == [("sysid_batch1_wheels_offground", "/dev/cu.usbmodem123")]


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


def test_batch1_postprocess_reads_v2_schema_and_segment_end(tmp_path: Path):
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "schema,batch1_v2\n"
        "data,t_us,state,phase_index,phase,armed,left_cmd_nm,right_cmd_nm,left_mode,"
        "left_pos_rev,left_vel_rev_s,left_torque_nm,left_voltage_v,left_temp_c,left_fault,"
        "right_mode,right_pos_rev,right_vel_rev_s,right_torque_nm,right_voltage_v,right_temp_c,"
        "right_fault,imu_ok,imu_age_ms,imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,fault_reason\n"
        "data,1,running_batch1,0,breakaway_r1_left_pos_0p04,1,0.04,0,10,"
        "0,0,0.04,24,25,0,0,0,0,0,24,25,0,1,0,1,0,0,0,0,none\n"
    )
    events = tmp_path / "events.log"
    events.write_text(
        "event,10,segment_start,running_batch1,breakaway_r1_left_pos_0p04\n"
        "event,20,segment_end,running_batch1,breakaway_r1_left_pos_0p04,motion_reached,110\n"
    )

    assert batch1_post.read_schema(telemetry) == "batch1_v2"
    parsed = batch1_post.read_events(events)
    assert parsed[1]["phase"] == "breakaway_r1_left_pos_0p04"
    assert parsed[1]["reason"] == "motion_reached"
    assert parsed[1]["elapsed_ms"] == 110


def test_batch1_segment_end_motion_marks_phase_as_moving():
    rows = [
        {
            "t_us": 0,
            "phase": "breakaway_r1_left_pos_0p04",
            "left_cmd_nm": 0.04,
            "right_cmd_nm": 0.0,
            "left_pos_rev": 0.0,
            "left_vel_rev_s": 0.0,
            "left_torque_nm": 0.04,
        },
        {
            "t_us": 10_000,
            "phase": "breakaway_r1_left_pos_0p04",
            "left_cmd_nm": 0.04,
            "right_cmd_nm": 0.0,
            "left_pos_rev": 0.0,
            "left_vel_rev_s": 0.0,
            "left_torque_nm": 0.04,
        },
    ]

    stats = batch1_post.phase_stats(
        rows,
        {"breakaway_r1_left_pos_0p04": "motion_reached"},
    )

    assert stats[0].moving


def test_batch1_breakaway_detection_excludes_anomalous_direction():
    def ps(
        phase: str,
        direction: str,
        command: float,
        max_velocity: float,
        position_delta: float,
        moving: bool,
    ) -> batch1_post.PhaseStats:
        return batch1_post.PhaseStats(
            phase,
            "left",
            direction,
            command,
            command,
            0.7,
            max_velocity,
            max_velocity * batch1_post.RAD_PER_REV,
            position_delta,
            moving,
            10,
        )

    stats = [
        ps("breakaway_left_pos_0p04", "pos", 0.04, 0.0, 0.0, False),
        ps("breakaway_left_pos_0p08", "pos", 0.08, 0.0, 0.0, False),
        ps("breakaway_left_pos_0p10", "pos", 0.10, 0.04, 0.02, True),
        ps("breakaway_left_neg_0p22", "neg", -0.22, 0.0, 0.0, False),
    ]

    intervals = batch1_post.detect_breakaway_intervals(stats)

    assert math.isclose(intervals["left"]["pos"].estimate_nm, 0.09)
    assert intervals["left"]["neg"].anomalous
    assert intervals["left"]["neg"].estimate_nm is None


def test_batch1_extracts_cad_axial_inertia_from_robot_xml():
    inertias = batch1_post.cad_axial_inertias_from_robot_xml(
        ROOT / "sim" / "model" / "finn_robot.xml"
    )

    assert math.isclose(inertias["left"], 0.00587115, rel_tol=1e-6)
    assert math.isclose(inertias["right"], 0.00587115, rel_tol=1e-6)


def test_batch1_single_cog_hop_is_not_sustained_motion():
    rows = [
        {
            "t_us": 0,
            "phase": "breakaway_r1_left_pos_0p04",
            "left_cmd_nm": 0.04,
            "right_cmd_nm": 0.0,
            "left_pos_rev": 0.0,
            "left_vel_rev_s": 0.0,
            "left_torque_nm": 0.04,
        },
        {
            "t_us": 10_000,
            "phase": "breakaway_r1_left_pos_0p04",
            "left_cmd_nm": 0.04,
            "right_cmd_nm": 0.0,
            "left_pos_rev": 0.011,
            "left_vel_rev_s": 0.0,
            "left_torque_nm": 0.04,
        },
    ]

    stats = batch1_post.phase_stats(rows)

    assert stats[0].first_motion
    assert not stats[0].sustained_motion


def test_batch1_dynamic_fit_recovers_synthetic_wheel_values():
    rows = []
    t_us = 0
    expected_i = 0.004
    expected_b = 0.002
    expected_c = 0.08
    for phase, sign in (("dynamic_left_pos_0p20", 1), ("dynamic_left_neg_0p20", -1)):
        for index in range(80):
            t_s = index * 0.01
            omega = sign * (0.2 + 0.6 * t_s + 0.7 * t_s * t_s)
            alpha = sign * (0.6 + 1.4 * t_s)
            tau = expected_i * alpha + expected_b * omega + expected_c * sign
            rows.append(
                {
                    "t_us": t_us,
                    "phase": phase,
                    "left_cmd_nm": sign * 0.2,
                    "right_cmd_nm": 0.0,
                    "left_pos_rev": sign * t_s,
                    "left_vel_rev_s": omega / batch1_post.RAD_PER_REV,
                    "left_torque_nm": tau,
                    "right_pos_rev": 0.0,
                    "right_vel_rev_s": 0.0,
                    "right_torque_nm": 0.0,
                }
            )
            t_us += 10_000

    fit = batch1_post.fit_wheel_dynamics(rows, "left")

    assert fit.sample_count > 100
    assert math.isclose(fit.armature, expected_i, rel_tol=0.12)
    assert math.isclose(fit.damping, expected_b, rel_tol=0.12)
    assert math.isclose(fit.frictionloss, expected_c, rel_tol=0.05)


def test_batch1_coastdown_output_error_recovers_loss_ratios():
    rows = []
    t_us = 0
    damping_per_j = 0.35
    friction_per_j = 1.8
    omega = 10.0
    for index in range(180):
        rows.append(
            {
                "t_us": t_us,
                "phase": "coastdown_left_pos_2p50rps",
                "left_cmd_nm": 0.0,
                "right_cmd_nm": 0.0,
                "left_pos_rev": index * 0.01,
                "left_vel_rev_s": omega / batch1_post.RAD_PER_REV,
                "left_torque_nm": 0.0,
                "right_pos_rev": 0.0,
                "right_vel_rev_s": 0.0,
                "right_torque_nm": 0.0,
            }
        )
        dt = 0.01
        omega += (-damping_per_j * omega - friction_per_j) * dt
        t_us += 10_000

    fit = batch1_post.fit_coastdown_output_error(rows, "left")

    assert fit.sample_count == 180
    assert math.isclose(fit.damping_per_inertia, damping_per_j, rel_tol=0.08)
    assert math.isclose(fit.friction_per_inertia, friction_per_j, rel_tol=0.08)


def test_batch1_powered_output_error_recovers_total_inertia():
    rows = []
    t_us = 0
    inertia = 0.0045
    damping_per_j = 0.3
    friction_per_j = 1.2
    omega = 0.2
    for index in range(160):
        tau = 0.09
        rows.append(
            {
                "t_us": t_us,
                "phase": "dynamic_left_pos_0p12",
                "left_cmd_nm": 0.12,
                "right_cmd_nm": 0.0,
                "left_pos_rev": index * 0.01,
                "left_vel_rev_s": omega / batch1_post.RAD_PER_REV,
                "left_torque_nm": tau,
                "right_pos_rev": 0.0,
                "right_vel_rev_s": 0.0,
                "right_torque_nm": 0.0,
            }
        )
        dt = 0.01
        omega += ((tau / inertia) - damping_per_j * omega - friction_per_j) * dt
        t_us += 10_000

    fit = batch1_post.fit_powered_inertia_output_error(
        rows,
        "left",
        damping_per_j,
        friction_per_j,
        initial_inertia_kg_m2=0.006,
    )

    assert fit.sample_count == 160
    assert math.isclose(fit.inertia_kg_m2, inertia, rel_tol=0.08)


def test_batch1_cad_subtraction_emits_near_zero_armature(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    telemetry = run_dir / "telemetry.csv"
    telemetry.write_text(
        "schema,batch1_v2\n"
        "data,t_us,state,phase_index,phase,armed,left_cmd_nm,right_cmd_nm,left_mode,"
        "left_pos_rev,left_vel_rev_s,left_torque_nm,left_voltage_v,left_temp_c,left_fault,"
        "right_mode,right_pos_rev,right_vel_rev_s,right_torque_nm,right_voltage_v,right_temp_c,"
        "right_fault,imu_ok,imu_age_ms,imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,fault_reason\n"
        + "\n".join(
            [
                (
                    f"data,{i * 10000},running_batch1,1,dynamic_left_pos_0p12,1,0.12,0,10,"
                    f"{i * 0.01},{(0.2 + i * 0.08) / batch1_post.RAD_PER_REV},0.09,24,25,0,"
                    "0,0,0,0,24,25,0,1,0,1,0,0,0,0,none"
                )
                for i in range(80)
            ]
        )
        + "\n"
    )

    derived = batch1_post.analyze_run(
        run_dir,
        hard_torque_limit_nm=0.25,
        command_signs={"left": -1, "right": 1},
        cad_axial_inertia_overrides={"left": 0.0045, "right": None},
    )

    sim = derived["wheels"]["left"]["suggested_sim"]
    assert sim["armature"]["value"] is not None
    assert sim["armature"]["value"] >= 0.0


def test_batch1_dynamic_fit_ignores_other_wheel_coast_phases():
    rows = []
    t_us = 0
    for phase, left_cmd, right_velocity in (
        ("dynamic_left_pos_0p20", 0.2, 0.0),
        ("coastdown_right_pos_1p50rps", 0.0, 1.5),
    ):
        for index in range(30):
            rows.append(
                {
                    "t_us": t_us,
                    "phase": phase,
                    "left_cmd_nm": left_cmd,
                    "right_cmd_nm": 0.0,
                    "left_pos_rev": index * 0.01,
                    "left_vel_rev_s": 0.25,
                    "left_torque_nm": 0.12,
                    "right_pos_rev": index * 0.01,
                    "right_vel_rev_s": right_velocity,
                    "right_torque_nm": 0.0,
                }
            )
            t_us += 10_000

    fit = batch1_post.fit_wheel_dynamics(rows, "left")

    assert fit.sample_count == 26


def test_batch1_current_trial_outputs_sim_overlay():
    run_dir = ROOT / "logs" / "finn-mcu" / "sysid" / "batch_1_pass" / "20260613_124610"

    derived = batch1_post.analyze_run(
        run_dir,
        hard_torque_limit_nm=0.25,
        command_signs={"left": -1, "right": 1},
    )

    assert derived["health"]["fault_rows"] == 0
    assert derived["wheels"]["left"]["suggested_sim"]["command_sign"]["value"] == -1
    assert derived["wheels"]["right"]["suggested_sim"]["command_sign"]["value"] == 1
    assert derived["wheels"]["left"]["suggested_sim"]["torque_limit_nm"]["value"] == 0.25
    assert "left_neg_no_motion_at_highest_tested_breakaway_torque" in derived["warnings"]
    assert derived["wheels"]["left"]["suggested_sim"]["friction_source"]["value"] in {
        "coastdown",
        "staircase_fallback",
    }
    assert (
        derived["wheels"]["left"]["suggested_sim"]["frictionloss"]["source"]
        != "breakaway_interval_midpoint"
    )
    assert set(derived["wheels"]["left"]["suggested_sim"]) >= {
        "command_sign",
        "torque_limit_nm",
        "validated_torque_nm",
        "frictionloss",
        "damping",
        "armature",
        "friction_source",
        "armature_vs_cad_pct",
        "bound_active",
    }
    assert set(derived["wheels"]["left"]["diagnostics"]) >= {
        "cad_axial_wheel_inertia_kg_m2",
        "fitted_total_wheel_inertia_kg_m2",
        "coastdown_output_error_fit",
        "powered_output_error_fit",
        "gradient_crosscheck",
    }


def test_batch1_empty_capture_writes_diagnostics(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "telemetry.csv").write_text("schema,batch1_v2\n")

    derived = batch1_post.analyze_run(
        run_dir,
        hard_torque_limit_nm=0.25,
        command_signs={"left": -1, "right": 1},
    )

    assert derived["run"]["rows"] == 0
    assert "no_telemetry_rows" in derived["warnings"]
    assert derived["wheels"]["left"]["diagnostics"]["ranges"]["voltage_v"] == [None, None]


def test_batch1_cli_writes_standalone_overlay(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    telemetry = run_dir / "telemetry.csv"
    telemetry.write_text(
        "schema,batch1_v2\n"
        "data,t_us,state,phase_index,phase,armed,left_cmd_nm,right_cmd_nm,left_mode,"
        "left_pos_rev,left_vel_rev_s,left_torque_nm,left_voltage_v,left_temp_c,left_fault,"
        "right_mode,right_pos_rev,right_vel_rev_s,right_torque_nm,right_voltage_v,right_temp_c,"
        "right_fault,imu_ok,imu_age_ms,imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,fault_reason\n"
        "data,0,running_batch1,0,settle_stop,1,0,0,0,0,0,0,24,25,0,0,0,0,0,24,25,0,1,0,1,0,0,0,0,none\n"
        "data,10000,running_batch1,1,breakaway_r1_left_pos_0p04,1,0.04,0,10,0,0,0.04,24,25,0,0,0,0,0,24,25,0,1,0,1,0,0,0,0,none\n"
        "data,20000,running_batch1,2,breakaway_r1_left_pos_0p06,1,0.06,0,10,0.02,0.03,0.06,24,25,0,0,0,0,0,24,25,0,1,0,1,0,0,0,0,none\n"
    )

    out_dir = run_dir / "postprocess"
    out_dir.mkdir()
    data = batch1_post.analyze_run(
        run_dir,
        hard_torque_limit_nm=0.25,
        command_signs={"left": -1, "right": 1},
    )
    batch1_post.write_derived_yaml(out_dir / "derived.yaml", data)
    batch1_post.write_report(out_dir / "report.md", data)

    data = yaml.safe_load((run_dir / "postprocess" / "derived.yaml").read_text())
    assert data["source"] == "finn_mcu_batch1_sysid"
    assert data["wheels"]["left"]["suggested_sim"]["command_sign"]["value"] == -1


def test_batch2_postprocess_reads_v1_schema_and_segment_end(tmp_path: Path):
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "schema,batch2_v1\n"
        "data,t_us,state,phase_index,phase,armed,control_tick_us,segment_elapsed_ms,"
        "left_cmd_nm,right_cmd_nm,left_mode,left_pos_rev,left_vel_rev_s,left_torque_nm,"
        "left_voltage_v,left_temp_c,left_fault,right_mode,right_pos_rev,right_vel_rev_s,"
        "right_torque_nm,right_voltage_v,right_temp_c,right_fault,imu_ok,imu_age_ms,"
        "imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,imu_gyro_x_rad_s,imu_gyro_y_rad_s,"
        "imu_gyro_z_rad_s,imu_linear_accel_x_m_s2,imu_linear_accel_y_m_s2,"
        "imu_linear_accel_z_m_s2,pitch_rad,pitch_rate_rad_s,yaw_rad,yaw_rate_rad_s,"
        "left_pos_delta_rev,right_pos_delta_rev,avg_wheel_pos_rev,diff_wheel_pos_rev,"
        "pitch_limit,speed_limit,travel_limit,fault_reason\n"
        "data,1,running_batch2,0,straight_r1_average_pos_0p12,1,1,0,0.12,0.12,10,"
        "0,0,0.10,24,25,0,10,0,0,0.10,24,25,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        "0,0,0,0,0,0,0,0,none\n"
    )
    events = tmp_path / "events.log"
    events.write_text(
        "event,10,segment_start,running_batch2,straight_r1_average_pos_0p12\n"
        "event,20,segment_end,running_batch2,straight_r1_average_pos_0p12,timeout,600\n"
    )

    assert batch2_post.read_schema(telemetry) == "batch2_v1"
    rows = batch2_post.rows_as_numeric(batch2_post.read_batch2_rows(telemetry))
    assert rows[0]["phase"] == "straight_r1_average_pos_0p12"
    parsed = batch2_post.read_events(events)
    assert parsed[1]["reason"] == "timeout"
    assert parsed[1]["elapsed_ms"] == 600


def test_batch2_physical_inputs_read_repo_measurements():
    measurements = batch2_post.load_measurements(ROOT / "sim" / "config" / "finn_measurements.yaml")
    phys = batch2_post.physical_inputs(measurements)

    # Assert presence and plausible range rather than exact values so this test
    # survives measurement refinements without needing a manual update.
    assert phys["robot_mass_kg"] is not None and 1.0 < phys["robot_mass_kg"] < 20.0
    assert phys["loaded_wheel_radius_m"] is not None and 0.03 < phys["loaded_wheel_radius_m"] < 0.2
    assert phys["wheel_track_width_m"] is not None and 0.1 < phys["wheel_track_width_m"] < 1.0
    assert phys["com_height_m"] is not None and 0.0 < phys["com_height_m"] < 2.0
    assert phys["com_fore_aft_m"] is not None


def test_batch1_analyze_run_records_measurements(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "telemetry.csv").write_text("schema,batch1_v2\n")

    derived = batch1_post.analyze_run(
        run_dir,
        hard_torque_limit_nm=0.25,
        command_signs={"left": -1, "right": 1},
        measurements_path=ROOT / "sim" / "config" / "finn_measurements.yaml",
    )

    physical = derived["input_models"]["physical"]
    assert physical["robot_mass_kg"] is not None and physical["robot_mass_kg"] > 0
    assert physical["loaded_wheel_radius_m"] is not None and physical["loaded_wheel_radius_m"] > 0
    assert physical["left_gear_ratio"] is not None and physical["left_gear_ratio"] > 0


def test_batch2_synthetic_straight_pulse_recovers_delay_and_torque_gain():
    rows = []
    commands = [0.0, 0.06, 0.12, -0.06, -0.12, 0.18]
    for index in range(120):
        t_us = index * 10_000
        cmd = commands[index // 20]
        delayed_cmd = commands[max(index - 3, 0) // 20]
        measured = 0.8 * delayed_cmd + 0.01
        rows.append(
            {
                "t_us": t_us,
                "phase": "straight_r1_average_pos_0p12",
                "left_cmd_nm": cmd,
                "right_cmd_nm": cmd,
                "left_torque_nm": measured,
                "right_torque_nm": measured,
                "left_vel_rev_s": 0.0,
                "right_vel_rev_s": 0.0,
                "imu_linear_accel_z_m_s2": measured,
                "yaw_rate_rad_s": 0.0,
            }
        )

    tracking = batch2_post.actuator_tracking(rows, "left")
    delay = batch2_post.cross_correlation_delay_s(rows, "left_cmd_nm", "left_torque_nm")

    assert math.isclose(tracking["gain"], 0.8, rel_tol=0.05)
    assert math.isclose(delay["delay_s"], 0.03, abs_tol=0.011)


def test_batch2_synthetic_coastdown_recovers_loaded_loss():
    rows = []
    radius_m = 0.05
    mass_kg = 12.0
    damping_s = 0.35
    friction_accel = 0.45
    v = 2.0
    for index in range(150):
        rows.append(
            {
                "t_us": index * 10_000,
                "phase": "coastdown_average_pos_1p00rps",
                "left_vel_rev_s": v / (batch2_post.RAD_PER_REV * radius_m),
                "right_vel_rev_s": v / (batch2_post.RAD_PER_REV * radius_m),
            }
        )
        v += (-damping_s * v - friction_accel) * 0.01

    fit = batch2_post.fit_loaded_coastdown_loss(rows, radius_m, mass_kg)

    assert fit.sample_count > 100
    assert math.isclose(fit.linear_damping_s, damping_s, rel_tol=0.2)
    assert math.isclose(fit.friction_accel_m_s2, friction_accel, rel_tol=0.2)


def test_batch2_synthetic_differential_yaw_recovers_effective_track_width():
    rows = []
    radius_m = 0.05
    track_m = 0.32
    for index in range(80):
        left = -1.0
        right = 1.0
        yaw = (right - left) * batch2_post.RAD_PER_REV * radius_m / track_m
        rows.append(
            {
                "t_us": index * 10_000,
                "phase": "yaw_r1_differential_pos_0p08",
                "left_vel_rev_s": left,
                "right_vel_rev_s": right,
                "yaw_rate_rad_s": yaw,
            }
        )

    response = batch2_post.yaw_response(rows, radius_m, track_m)

    assert response["confidence"] == "measured"
    assert math.isclose(response["effective_track_width_m"], track_m, rel_tol=0.02)
    assert math.isclose(response["track_width_correction"], 1.0, rel_tol=0.02)


def test_batch2_stationary_noise_excludes_moving_settle_segments():
    rows = []
    for _index in range(20):
        rows.append(
            {
                "phase": "initial_stationary_noise",
                "left_cmd_nm": 0.0,
                "right_cmd_nm": 0.0,
                "left_vel_rev_s": 0.0,
                "right_vel_rev_s": 0.0,
                "imu_gyro_x_rad_s": 0.01,
            }
        )
    for _index in range(10):
        rows.append(
            {
                "phase": "straight_r1_average_pos_0p12_settle",
                "left_cmd_nm": 0.0,
                "right_cmd_nm": 0.0,
                "left_vel_rev_s": 0.0,
                "right_vel_rev_s": 0.0,
                "imu_gyro_x_rad_s": 10.0,
            }
        )

    stats = batch2_post.noise_stats(rows)

    assert stats["stationary_row_count"] == 20
    assert math.isclose(stats["gyro_rad_s"]["x"]["mean"], 0.01)


def test_batch2_partial_capture_writes_diagnostics(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "telemetry.csv").write_text("schema,batch2_v1\n")
    out_dir = run_dir / "postprocess"
    out_dir.mkdir()

    derived = batch2_post.analyze_run(
        run_dir,
        physical_overrides={
            "mass_kg": None,
            "loaded_wheel_radius_m": None,
            "wheel_track_width_m": None,
            "com_height_m": None,
            "com_fore_aft_m": None,
            "pitch_inertia_kg_m2": None,
        },
    )
    batch2_post.write_derived_yaml(out_dir / "derived.yaml", derived)
    batch2_post.write_report(out_dir / "report.md", derived)

    loaded = yaml.safe_load((out_dir / "derived.yaml").read_text())
    assert loaded["metadata"]["rows"] == 0
    assert "no_telemetry_rows" in loaded["warnings"]
    assert (out_dir / "report.md").exists()


def test_capture_sysid_batch2_defaults(monkeypatch):
    module_path = TOOLS / "capture-sysid-batch2.py"
    spec = importlib.util.spec_from_file_location("capture_sysid_batch2", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", ["capture-sysid-batch2.py", "--auto-run"])
    args = module.parse_args()

    assert args.run_name == "batch_2"
    assert args.upload_env == "sysid_batch2_loaded_ground"
    assert args.pass_marker == [",batch2_complete,"]
    assert args.fail_marker == [",failsafe,"]
    assert args.auto_command == ["ARM FINN", "RUN BATCH2"]
    assert "postprocess_sysid_batch2.py" in args.postprocess
    assert "--run-dir {run_dir}" in args.postprocess
    assert "--measurements" in args.postprocess
    assert "finn_measurements.yaml" in args.postprocess


def test_capture_sysid_batch1_defaults_include_measurements(monkeypatch):
    module_path = TOOLS / "capture-sysid-batch1.py"
    spec = importlib.util.spec_from_file_location("capture_sysid_batch1", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(sys, "argv", ["capture-sysid-batch1.py", "--auto-run"])
    args = module.parse_args()

    assert args.run_name == "batch_1"
    assert args.upload_env == "sysid_batch1_wheels_offground"
    assert args.auto_command == ["ARM FINN", "RUN BATCH1"]
    assert "--measurements" in args.postprocess
    assert "finn_measurements.yaml" in args.postprocess


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


# ---------------------------------------------------------------------------
# Batch 2 postprocessor — new functions added in the batch-1-priors wiring pass
# ---------------------------------------------------------------------------


def _make_batch1_priors_dict(
    left_friction: float = 0.115,
    right_friction: float = 0.127,
    left_damping: float = 0.0041,
    right_damping: float = 0.0044,
    left_armature: float = 0.0,
    right_armature: float = 0.0,
    left_inertia: float = 0.00404,
    right_inertia: float = 0.00400,
    left_sign: int = -1,
    right_sign: int = 1,
) -> dict:
    def _wheel(friction, damping, armature, inertia, sign):
        return {
            "suggested_sim": {
                "frictionloss": {"value": friction, "unit": "N*m", "source": "coastdown"},
                "damping": {
                    "value": damping,
                    "unit": "N*m*s/rad",
                    "source": "coastdown_output_error",
                },
                "armature": {
                    "value": armature,
                    "unit": "kg*m^2",
                    "source": "powered_output_error_minus_cad_axial_inertia",
                },
                "torque_limit_nm": {"value": 0.25, "unit": "N*m", "source": "firmware_hard_cap"},
                "command_sign": {"value": sign, "unit": "sign", "source": "operator_video_default"},
            },
            "diagnostics": {
                "fitted_total_wheel_inertia_kg_m2": inertia,
                "actuator_tracking": {
                    "gain": 0.942,
                    "bias_nm": 0.001,
                    "rmse_nm": 0.04,
                    "sample_count": 100,
                },
                "response_delay": {"median_s": 0.01, "min_s": 0.0, "max_s": 0.05},
            },
        }

    return {
        "wheels": {
            "left": _wheel(left_friction, left_damping, left_armature, left_inertia, left_sign),
            "right": _wheel(
                right_friction, right_damping, right_armature, right_inertia, right_sign
            ),
        }
    }


def test_extract_batch1_priors_parses_provenance_nodes():
    priors_dict = _make_batch1_priors_dict()
    b1 = batch2_post.extract_batch1_priors(priors_dict)

    assert set(b1.keys()) == {"left", "right"}
    left = b1["left"]
    assert math.isclose(left["frictionloss_nm"], 0.115)
    assert math.isclose(left["damping_nm_s_per_rad"], 0.0041)
    assert left["command_sign"] == -1
    assert math.isclose(left["torque_limit_nm"], 0.25)
    assert math.isclose(left["total_wheel_inertia_kg_m2"], 0.00404)
    assert math.isclose(left["actuator_gain"], 0.942)

    right = b1["right"]
    assert right["command_sign"] == 1
    assert math.isclose(right["frictionloss_nm"], 0.127)


def test_extract_batch1_priors_returns_empty_for_none():
    assert batch2_post.extract_batch1_priors(None) == {}
    assert batch2_post.extract_batch1_priors({}) == {}


def test_extract_batch1_priors_none_for_missing_wheel():
    b1 = batch2_post.extract_batch1_priors({"wheels": {"left": {}}})
    assert b1["left"]["frictionloss_nm"] is None
    assert b1["left"]["command_sign"] is None


def test_estimate_loaded_radius_recovers_known_radius():
    radius_m = 0.065
    rows = []
    t_us = 0
    omega_rev_s = 0.0
    for _ in range(80):
        # Simulate constant angular acceleration via a constant torque
        alpha_rad_s2 = 3.0  # rad/s²
        a_imu = radius_m * alpha_rad_s2
        omega_rev_s += alpha_rad_s2 / batch2_post.RAD_PER_REV * 0.01
        rows.append(
            {
                "t_us": t_us,
                "phase": "straight_r1_average_pos_0p12",
                "imu_linear_accel_z_m_s2": a_imu,
                "left_vel_rev_s": omega_rev_s,
                "right_vel_rev_s": omega_rev_s,
            }
        )
        t_us += 10_000  # 100 Hz

    result = batch2_post.estimate_loaded_radius(rows)

    assert result["sample_count"] > 0
    assert result["radius_m"] is not None
    assert math.isclose(result["radius_m"], radius_m, rel_tol=0.15)


def test_estimate_loaded_radius_tolerates_forward_accel_sign():
    radius_m = 0.065
    rows = []
    t_us = 0
    omega_rev_s = 0.0
    for _ in range(80):
        alpha_rad_s2 = 3.0
        a_imu = -radius_m * alpha_rad_s2
        omega_rev_s += alpha_rad_s2 / batch2_post.RAD_PER_REV * 0.01
        rows.append(
            {
                "t_us": t_us,
                "phase": "straight_r1_average_pos_0p12",
                "imu_linear_accel_z_m_s2": a_imu,
                "left_vel_rev_s": omega_rev_s,
                "right_vel_rev_s": omega_rev_s,
            }
        )
        t_us += 10_000

    result = batch2_post.estimate_loaded_radius(rows)

    assert result["radius_m"] is not None
    assert math.isclose(result["radius_m"], radius_m, rel_tol=0.15)


def test_estimate_loaded_radius_insufficient_for_empty_rows():
    result = batch2_post.estimate_loaded_radius([])
    assert result["radius_m"] is None
    assert result["confidence"] == "insufficient"


def test_decompose_losses_separates_motor_and_tire():
    # Construct a LossFit where total friction = motor + tire contribution
    radius_m = 0.065
    mass_kg = 8.0
    j_wheel = 0.004
    m_eff = mass_kg + 2 * j_wheel / radius_m**2
    # Motor params (per wheel)
    motor_friction_nm = 0.12
    motor_damping = 0.004
    # Convert to linear
    motor_friction_accel = 2 * motor_friction_nm / (radius_m * m_eff)
    motor_damping_s = 2 * motor_damping / (radius_m**2 * m_eff)
    # Add a known tire contribution
    tire_friction_accel = 0.15
    tire_damping_s = 0.05
    total_friction = motor_friction_accel + tire_friction_accel
    total_damping = motor_damping_s + tire_damping_s

    loss_fit = batch2_post.LossFit(
        sample_count=200,
        linear_damping_s=total_damping,
        friction_accel_m_s2=total_friction,
        per_wheel_torque_nm=total_friction * mass_kg * radius_m / 2,
        rmse_m_s=0.01,
        confidence="measured",
        bound_active={"linear_damping": False, "friction_accel": False},
        notes=[],
    )
    b1_left = {"frictionloss_nm": motor_friction_nm, "damping_nm_s_per_rad": motor_damping}
    b1_right = {"frictionloss_nm": motor_friction_nm, "damping_nm_s_per_rad": motor_damping}

    result = batch2_post.decompose_losses(
        loss_fit, b1_left, b1_right, radius_m, mass_kg, j_wheel, j_wheel
    )

    assert result["tire_friction_accel_m_s2"] is not None
    assert math.isclose(result["tire_friction_accel_m_s2"], tire_friction_accel, rel_tol=0.02)
    assert math.isclose(result["tire_damping_s"], tire_damping_s, rel_tol=0.02)
    assert result["rolling_resistance_coeff"] is not None
    assert result["rolling_resistance_coeff"] > 0


def test_decompose_losses_returns_insufficient_without_radius():
    loss_fit = batch2_post.LossFit(
        sample_count=10,
        linear_damping_s=0.1,
        friction_accel_m_s2=0.2,
        per_wheel_torque_nm=None,
        rmse_m_s=None,
        confidence="provisional",
        bound_active={},
        notes=[],
    )
    result = batch2_post.decompose_losses(loss_fit, {}, {}, None, 8.0, None, None)
    assert result["confidence"] == "insufficient"
    assert result["tire_friction_accel_m_s2"] is None


def test_decompose_losses_returns_insufficient_without_batch1():
    loss_fit = batch2_post.LossFit(
        sample_count=100,
        linear_damping_s=0.1,
        friction_accel_m_s2=0.2,
        per_wheel_torque_nm=0.05,
        rmse_m_s=0.01,
        confidence="measured",
        bound_active={},
        notes=[],
    )
    result = batch2_post.decompose_losses(loss_fit, {}, {}, 0.065, 8.0, None, None)
    assert result["confidence"] == "insufficient"
    assert "batch1_motor_parameters_unavailable" in result["notes"]


def test_actuator_cross_check_consistent():
    b2 = {"gain": 0.95, "bias_nm": 0.002, "sample_count": 100, "rmse_nm": 0.04}
    b1 = {"actuator_gain": 0.94, "actuator_bias_nm": 0.001}
    result = batch2_post.actuator_cross_check(b2, b1)
    assert result["status"] == "consistent"
    assert math.isclose(result["gain_delta"], 0.01, abs_tol=1e-9)


def test_actuator_cross_check_diverged():
    b2 = {"gain": 0.70, "bias_nm": 0.05, "sample_count": 50, "rmse_nm": 0.08}
    b1 = {"actuator_gain": 0.94, "actuator_bias_nm": 0.001}
    result = batch2_post.actuator_cross_check(b2, b1)
    assert result["status"] == "diverged"


def test_actuator_cross_check_insufficient_without_batch1():
    b2 = {"gain": 0.94, "bias_nm": 0.001, "sample_count": 100, "rmse_nm": 0.04}
    result = batch2_post.actuator_cross_check(b2, {})
    assert result["status"] == "insufficient_data"
    assert result["gain_delta"] is None


def test_mujoco_params_assembles_from_priors_and_batch2(tmp_path: Path):
    b1_priors = batch2_post.extract_batch1_priors(_make_batch1_priors_dict())
    radius_est = {"radius_m": 0.065, "confidence": "measured", "notes": []}
    loss_decomp = {
        "tire_friction_accel_m_s2": 0.15,
        "rolling_resistance_coeff": 0.015,
        "confidence": "measured",
        "notes": [],
    }
    yaw = {"effective_track_width_m": 0.32, "confidence": "measured", "notes": []}
    traction = {"straight_mu_lower_bound": 0.45, "straight_accel_m_s2": 4.4}
    phys = {"loaded_wheel_radius_m": None, "wheel_track_width_m": None, "robot_mass_kg": 8.0}

    result = batch2_post.mujoco_params(b1_priors, radius_est, loss_decomp, yaw, traction, phys)

    assert result["actuators"]["left"]["command_sign"] == -1
    assert result["actuators"]["right"]["command_sign"] == 1
    assert math.isclose(result["actuators"]["left"]["frictionloss_nm"], 0.115)
    assert result["geometry"]["loaded_wheel_radius_m"]["value"] == 0.065
    assert (
        result["geometry"]["loaded_wheel_radius_m"]["source"] == "batch2_imu_wheel_kinematic_ratio"
    )
    assert result["geometry"]["effective_track_width_m"]["value"] == 0.32
    assert result["contact"]["rolling_resistance_coeff"] == 0.015
    assert result["contact"]["solref"] is None
    assert result["contact"]["solimp"] is None
    assert result["readiness"]["batch3_can_proceed"] is True
    assert "contact_solref" in result["readiness"]["open_params_for_batch3"]


def test_mujoco_params_batch3_cannot_proceed_without_radius():
    b1_priors = batch2_post.extract_batch1_priors(_make_batch1_priors_dict())
    radius_est = {"radius_m": None, "confidence": "insufficient", "notes": []}
    loss_decomp = {
        "rolling_resistance_coeff": None,
        "tire_friction_accel_m_s2": None,
        "confidence": "insufficient",
        "notes": [],
    }
    yaw = {"effective_track_width_m": None, "confidence": "insufficient", "notes": []}
    traction = {"straight_mu_lower_bound": None}
    phys = {"loaded_wheel_radius_m": None, "wheel_track_width_m": None, "robot_mass_kg": None}

    result = batch2_post.mujoco_params(b1_priors, radius_est, loss_decomp, yaw, traction, phys)

    assert result["readiness"]["batch3_can_proceed"] is False
    assert "loaded_wheel_radius_m" in result["readiness"]["open_params_for_batch3"]


def test_lqr_readiness_has_batch2_schema_false_for_wrong_schema(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Write a file with the wrong schema tag
    (run_dir / "telemetry.csv").write_text("schema,batch2_v2\n")
    (run_dir / "events.log").write_text("")

    derived = batch2_post.analyze_run(
        run_dir,
        physical_overrides={
            "mass_kg": None,
            "loaded_wheel_radius_m": None,
            "wheel_track_width_m": None,
            "com_height_m": None,
            "com_fore_aft_m": None,
            "pitch_inertia_kg_m2": None,
        },
    )

    assert derived["lqr_readiness"]["checklist"]["has_batch2_schema"] is False
    assert derived["lqr_readiness"]["pass"] is False


def test_lqr_readiness_has_batch2_schema_true_for_correct_schema(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    header = (
        "schema,batch2_v1\n"
        "data,t_us,state,phase_index,phase,armed,control_tick_us,segment_elapsed_ms,"
        "left_cmd_nm,right_cmd_nm,left_mode,left_pos_rev,left_vel_rev_s,left_torque_nm,"
        "left_voltage_v,left_temp_c,left_fault,right_mode,right_pos_rev,right_vel_rev_s,"
        "right_torque_nm,right_voltage_v,right_temp_c,right_fault,imu_ok,imu_age_ms,"
        "imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,imu_gyro_x_rad_s,imu_gyro_y_rad_s,"
        "imu_gyro_z_rad_s,imu_linear_accel_x_m_s2,imu_linear_accel_y_m_s2,"
        "imu_linear_accel_z_m_s2,pitch_rad,pitch_rate_rad_s,yaw_rad,yaw_rate_rad_s,"
        "left_pos_delta_rev,right_pos_delta_rev,avg_wheel_pos_rev,diff_wheel_pos_rev,"
        "pitch_limit,speed_limit,travel_limit,fault_reason\n"
    )
    (run_dir / "telemetry.csv").write_text(header)
    (run_dir / "events.log").write_text("")

    derived = batch2_post.analyze_run(
        run_dir,
        physical_overrides={
            "mass_kg": None,
            "loaded_wheel_radius_m": None,
            "wheel_track_width_m": None,
            "com_height_m": None,
            "com_fore_aft_m": None,
            "pitch_inertia_kg_m2": None,
        },
    )

    assert derived["lqr_readiness"]["checklist"]["has_batch2_schema"] is True


def test_analyze_run_uses_batch1_command_signs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "telemetry.csv").write_text("schema,batch2_v1\n")
    (run_dir / "events.log").write_text("")

    priors_dict = _make_batch1_priors_dict(left_sign=-1, right_sign=1)
    priors_path = tmp_path / "batch1_derived.yaml"
    priors_path.write_text(yaml.safe_dump(priors_dict))

    derived = batch2_post.analyze_run(
        run_dir,
        batch1_derived=priors_path,
        physical_overrides={
            "mass_kg": None,
            "loaded_wheel_radius_m": None,
            "wheel_track_width_m": None,
            "com_height_m": None,
            "com_fore_aft_m": None,
            "pitch_inertia_kg_m2": None,
        },
    )

    assert derived["suggested_sim"]["wheels"]["command_signs"]["left"] == -1
    assert derived["suggested_sim"]["wheels"]["command_signs"]["right"] == 1
    assert derived["inputs"]["batch1_priors_available"] is True
    assert derived["inputs"]["batch1_priors"]["left"]["command_sign"] == -1
