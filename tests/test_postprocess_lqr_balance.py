"""Tests for the LQR balance postprocessor and the telemetry contract it reads."""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance" / "main.cpp"
POSTPROCESSOR = ROOT / "firmware" / "finn-mcu" / "tools" / "postprocess_lqr_balance.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register before executing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


post = load_module(POSTPROCESSOR, "finn_postprocess_lqr_balance")


def firmware_columns() -> list[str]:
    """The telemetry column list straight from the firmware that emits it."""

    block = FIRMWARE.read_text(encoding="utf-8").split("void printTelemetryHeader()")[1]
    block = block.split("\n}")[0]
    literals = re.findall(r'"((?:[^"\\]|\\.)*)"', block)
    header = "".join(lit for lit in literals if not lit.startswith("schema"))
    columns = [c for c in header.split(",") if c]
    assert columns[0] == "data"
    return columns[1:]


COLUMNS = firmware_columns()

TEXT_DEFAULTS = {
    "state": "running_lqr",
    "phase": "balance",
    "model_sha256": "4a85ddfeea07",
    "fault_reason": "none",
}


def make_row(index: int, phase: str, **overrides) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(COLUMNS, 0.0)
    row.update(TEXT_DEFAULTS)
    row["phase"] = phase
    row["state"] = "running_lqr" if phase == "balance" else "preflight"
    row["t_us"] = index * 10000
    row["control_dt_us"] = 10000
    row["tick_duration_us"] = 4200
    row["imu_ok"] = 1
    row["target_pitch_rad"] = 0.0411449906
    row.update(overrides)
    return row


def write_run(
    tmp_path: Path, rows: list[dict[str, object]], *, schema: str = "lqr_v2", checks: str = ""
) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    lines = [f"schema,{schema}", "data," + ",".join(COLUMNS)]
    for row in rows:
        lines.append("data," + ",".join(str(row[c]) for c in COLUMNS))
    (run_dir / "telemetry.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "events.log").write_text(checks, encoding="utf-8")
    return run_dir


def balance_rows(count: int, *, torque: float, seed: int = 0) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        rows.append(
            make_row(
                i,
                "balance",
                # Consistent with the closed loop, where tau ~= -Kp * pitch_error:
                # a positive steady torque can only coexist with a negative error.
                pitch_rad=0.0411449906 - torque / 62.0 + float(rng.normal(0, 1e-4)),
                pitch_rate_rad_s=float(rng.normal(0, 0.01)),
                forward_vel_m_s=float(rng.normal(0, 0.005)),
                balance_tau_nm=torque + float(rng.normal(0, 0.01)),
                left_cmd_nm=torque,
                right_cmd_nm=torque,
                left_torque_nm=0.9 * torque + 0.01,
                right_torque_nm=0.9 * torque + 0.01,
            )
        )
    return rows


def analyze(run_dir: Path):
    return post.analyze_run(
        run_dir,
        ROOT / "sim" / "config" / "finn_measurements.yaml",
        ROOT / "firmware" / "finn-mcu" / "control" / "04_lqr_balance" / "lqr_seeded_config.h",
    )


def test_firmware_header_and_row_have_the_same_column_count():
    """A header that disagrees with the row silently corrupts every artifact."""

    body = FIRMWARE.read_text(encoding="utf-8").split("void printTelemetry()")[1]
    body = body.split("\nvoid printStatus()")[0]
    conversions = 0
    for call in body.split("Serial.printf(")[1:]:
        fmt, index = "", 0
        while index < len(call):
            char = call[index]
            if char in " \t\r\n":
                index += 1
            elif char == '"':
                index += 1
                while call[index] != '"':
                    if call[index] == "\\":
                        fmt += call[index : index + 2]
                        index += 2
                    else:
                        fmt += call[index]
                        index += 1
                index += 1
            else:
                break
        conversions += len(re.findall(r"%[-+ #0-9.]*l*[dfsu]", fmt))
    assert conversions == len(COLUMNS)


def test_scopik_lqr_profile_only_reads_columns_the_firmware_emits():
    profile = yaml.safe_load(
        (ROOT / "config" / "viz" / "finn_lqr.yaml").read_text(encoding="utf-8")
    )
    missing = set(profile["signals"]) - set(COLUMNS)
    assert not missing, f"profile reads columns the firmware does not emit: {sorted(missing)}"


def test_a_steady_torque_bias_identifies_a_trim_correction(tmp_path: Path):
    run_dir = write_run(tmp_path, balance_rows(300, torque=0.05))
    trim = analyze(run_dir)["identification"]["balance_trim"]

    assert trim["status"] == "identified"
    assert trim["steady_samples"] >= post.MIN_STEADY_SAMPLES
    # The robot holds station below the commanded trim while pushing forward, so
    # the true balance pitch -- and the suggested target -- sit below the target.
    assert trim["trim_offset_rad"] < 0.0
    assert trim["suggested_target_pitch_rad"] < 0.0411449906
    assert trim["suggested_com_fore_aft_shift_m"] < 0.0
    # Magnitude is 2*tau / (m*g*l) with the committed measurements.
    assert abs(trim["trim_offset_rad"]) == pytest.approx(math.asin(2 * 0.05 / 15.936), rel=0.05)


def test_torque_inside_its_own_noise_is_not_reported_as_a_correction(tmp_path: Path):
    """A still robot must not be handed a COM change built out of noise."""

    run_dir = write_run(tmp_path, balance_rows(300, torque=0.0))
    trim = analyze(run_dir)["identification"]["balance_trim"]

    assert trim["status"] == "no_correction_needed"


def test_a_quiet_closed_loop_run_refuses_to_fit_the_pitch_plant(tmp_path: Path):
    """Lean and torque are collinear in closed loop; the fit must decline."""

    run_dir = write_run(tmp_path, balance_rows(300, torque=0.05))
    plant = analyze(run_dir)["identification"]["pitch_plant"]

    assert plant["status"] == "not_excited"


def test_an_excited_run_recovers_the_pitch_plant_coefficients(tmp_path: Path):
    rng = np.random.default_rng(7)
    count = 400
    mgl_over_i = 26.0
    inverse_inertia = 1.7

    pitch_rate = np.cumsum(rng.normal(0, 0.02, count)) * 0.1
    times_s = np.arange(count) * 0.01
    pitch_accel = np.gradient(pitch_rate, times_s)
    pitch = np.cumsum(rng.normal(0, 0.01, count)) * 0.05
    tau_total = (pitch_accel - mgl_over_i * np.sin(pitch)) / inverse_inertia

    rows = [
        make_row(
            i,
            "balance",
            pitch_rad=float(pitch[i]),
            pitch_rate_rad_s=float(pitch_rate[i]),
            balance_tau_nm=float(tau_total[i] / 2.0),
            forward_vel_m_s=0.0,
        )
        for i in range(count)
    ]
    plant = analyze(write_run(tmp_path, rows))["identification"]["pitch_plant"]

    assert plant["status"] == "identified"
    assert plant["mgl_over_inertia_s2"] == pytest.approx(mgl_over_i, rel=0.02)
    assert plant["inverse_pitch_inertia_per_kg_m2"] == pytest.approx(inverse_inertia, rel=0.02)


def test_actuator_tracking_recovers_the_commanded_to_measured_slope(tmp_path: Path):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(300):
        commanded = float(rng.normal(0, 0.2))
        rows.append(
            make_row(
                i,
                "balance",
                left_cmd_nm=commanded,
                right_cmd_nm=commanded,
                left_torque_nm=0.9 * commanded + 0.01,
                right_torque_nm=0.9 * commanded + 0.01,
            )
        )
    tracking = analyze(write_run(tmp_path, rows))["identification"]["actuator_tracking"]

    for side in ("left", "right"):
        assert tracking[side]["status"] == "identified"
        assert tracking[side]["gain"] == pytest.approx(0.9, rel=1e-3)
        assert tracking[side]["offset_nm"] == pytest.approx(0.01, abs=1e-3)


def test_sensor_noise_comes_from_the_stationary_preflight_window(tmp_path: Path):
    rng = np.random.default_rng(11)
    rows = [
        make_row(
            i,
            "preflight",
            imu_gyro_x_rad_s=float(rng.normal(0, 0.004)),
            imu_gyro_y_rad_s=float(rng.normal(0, 0.004)),
            imu_gyro_z_rad_s=float(rng.normal(0, 0.004)),
        )
        for i in range(200)
    ]
    noise = analyze(write_run(tmp_path, rows))["identification"]["sensor_noise"]

    assert noise["status"] == "identified"
    assert noise["imu_gyro_x_rad_s"]["std"] == pytest.approx(0.004, rel=0.2)


def test_preflight_checks_are_parsed_out_of_the_event_log(tmp_path: Path):
    checks = (
        "check,1000,imu_quat_rate_hz,pass,101.0000,80.0000,rotation_vector_reports\n"
        "check,1001,moteus_left_link_hz,fail,0.0000,80.0000,query_replies\n"
        "check,1002,bus_voltage_floor,skip,24.1000,0.0000,set_kMinBusVoltageV_from_pack_spec\n"
    )
    run_dir = write_run(tmp_path, balance_rows(120, torque=0.02), checks=checks)
    summary = analyze(run_dir)["integrity"]["preflight_checks"]

    assert summary["pass"] == 1
    assert summary["failed"] == ["moteus_left_link_hz"]
    assert summary["skipped"] == ["bus_voltage_floor"]


def test_a_run_from_different_firmware_is_refused(tmp_path: Path):
    run_dir = write_run(tmp_path, balance_rows(120, torque=0.0), schema="lqr_v1")

    with pytest.raises(post.LqrBalancePostprocessError, match="lqr_v1"):
        analyze(run_dir)


def test_truncated_rows_are_counted_rather_than_parsed(tmp_path: Path):
    run_dir = write_run(tmp_path, balance_rows(120, torque=0.02))
    telemetry = run_dir / "telemetry.csv"
    telemetry.write_text(telemetry.read_text() + "data,1,2,3\n", encoding="utf-8")

    assert analyze(run_dir)["integrity"]["malformed_rows"] == 1


def test_the_recommendation_names_one_family_and_the_report_writes(tmp_path: Path):
    run_dir = write_run(tmp_path, balance_rows(300, torque=0.05))
    derived = analyze(run_dir)

    assert derived["recommended_next_change"]["parameter_family"] == "balance trim / COM fore-aft"
    out_dir = run_dir / "postprocess"
    out_dir.mkdir(exist_ok=True)
    post.write_report(out_dir / "report.md", derived)
    post.write_derived_yaml(out_dir / "derived.yaml", derived)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Recommended next model change" in report
    assert "Not identified by a closed-loop balance run" in report
    assert yaml.safe_load((out_dir / "derived.yaml").read_text(encoding="utf-8"))


def test_nothing_identifiable_recommends_a_disturbance_rather_than_a_guess(tmp_path: Path):
    run_dir = write_run(tmp_path, balance_rows(60, torque=0.0))
    recommended = analyze(run_dir)["recommended_next_change"]

    assert recommended["parameter_family"] is None
    assert "disturbance" in recommended["action"]


def test_a_preflight_only_run_still_writes_a_report(tmp_path: Path):
    """The first thing anyone runs is a preflight with no balance phase. The
    report writer used to crash on it because the short-run paths returned a bare
    verdict where it iterated per-key dicts."""

    rows = [make_row(i, "preflight") for i in range(200)]
    derived = analyze(write_run(tmp_path, rows))

    tracking = derived["identification"]["actuator_tracking"]
    latency = derived["identification"]["loop_latency"]
    assert set(tracking) == {"left", "right"}
    assert set(latency) == {"command_to_measured_torque", "command_to_pitch_acceleration"}
    assert all(item["status"] == "insufficient_data" for item in tracking.values())

    out = tmp_path / "out"
    out.mkdir()
    post.write_report(out / "report.md", derived)
    assert "Recording integrity" in (out / "report.md").read_text(encoding="utf-8")


def test_a_healthy_quaternion_stream_reports_its_true_rate(tmp_path: Path):
    """The dedup tolerance once sat above the 10 ms report interval, collapsing a
    perfect 100 Hz stream to one arrival -- the same signature as a dead sensor,
    on the line an operator reads before releasing the robot."""

    rows = [
        make_row(i, "preflight", imu_ok=1, imu_age_us=(i % 3) * 100, imu_age_ms=0)
        for i in range(300)
    ]
    audit = analyze(write_run(tmp_path, rows))["integrity"]

    assert audit["rotation_vector"]["arrivals"] >= 295
    assert audit["rotation_vector"]["rate_hz"] == pytest.approx(100.0, rel=0.05)


def test_telemetry_rows_sharing_one_quaternion_arrival_are_deduplicated(tmp_path: Path):
    rows = []
    for i in range(300):
        # Pairs of rows carry the same arrival stamp: age grows 10 ms within a pair.
        rows.append(make_row(i, "preflight", imu_ok=1, imu_age_us=(i % 2) * 10_000))
    audit = analyze(write_run(tmp_path, rows))["integrity"]

    assert audit["rotation_vector"]["rate_hz"] == pytest.approx(50.0, rel=0.05)


def test_a_dead_rotation_vector_is_reported_even_though_the_imu_reads_alive(tmp_path: Path):
    """imu_ok is isImuAlive(): any report inside 250 ms. A gyro streaming beside a
    dead fused quaternion reads healthy there while arming stays blocked, so the
    audit measures the quaternion itself."""

    rows = []
    for i in range(120):
        # One quaternion at t=0, none after: age grows by one telemetry period.
        rows.append(
            make_row(i, "preflight", imu_ok=1, imu_age_us=i * 1_000_000, imu_age_ms=i * 1000)
        )
        rows[-1]["t_us"] = i * 1_000_000
    audit = analyze(write_run(tmp_path, rows))["integrity"]

    assert audit["imu_fresh_fraction"] == 1.0
    assert audit["rotation_vector"]["arrivals"] == 1
    assert audit["rotation_vector"]["rate_hz"] < 0.1
    assert audit["rotation_vector"]["max_age_ms"] == 119000.0
