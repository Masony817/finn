"""Integration: the committed finn profile works with scopik and the seeded model.

scopik is a real workspace package, so unlike the tools/ scripts it is imported
normally (no importlib gymnastics).
"""

from pathlib import Path

import pytest
from scopik import load_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "config/viz/finn.yaml"

BATCH2_HEADER = (
    "data,t_us,state,phase_index,phase,armed,control_tick_us,segment_elapsed_ms,"
    "left_cmd_nm,right_cmd_nm,left_mode,left_pos_rev,left_vel_rev_s,left_torque_nm,"
    "left_voltage_v,left_temp_c,left_fault,right_mode,right_pos_rev,right_vel_rev_s,"
    "right_torque_nm,right_voltage_v,right_temp_c,right_fault,imu_ok,imu_age_ms,"
    "imu_qr,imu_qi,imu_qj,imu_qk,imu_accuracy_rad,imu_gyro_x_rad_s,imu_gyro_y_rad_s,"
    "imu_gyro_z_rad_s,imu_linear_accel_x_m_s2,imu_linear_accel_y_m_s2,"
    "imu_linear_accel_z_m_s2,robot_forward_accel_m_s2,pitch_rad,pitch_rate_rad_s,"
    "yaw_rad,yaw_rate_rad_s,left_pos_delta_rev,right_pos_delta_rev,avg_wheel_pos_rev,"
    "diff_wheel_pos_rev,pitch_limit,speed_limit,travel_limit,fault_reason"
)


def write_synthetic_batch2_run(run_dir: Path, rows: int = 120) -> None:
    run_dir.mkdir(parents=True)
    n_columns = len(BATCH2_HEADER.split(",")) - 1
    lines = ["schema,batch2_v2", BATCH2_HEADER]
    for i in range(rows):
        cells = ["0"] * n_columns
        cells[0] = str(int(i * 1e4))  # t_us at 100 Hz
        cells[1] = "running"
        cells[3] = "straight_pulse" if i < rows // 2 else "coastdown"
        cells[7] = "-0.1" if i < rows // 2 else "0"  # left_cmd_nm
        cells[8] = "0.1" if i < rows // 2 else "0"  # right_cmd_nm
        cells[11] = "-0.5"  # left_vel_rev_s
        cells[18] = "0.5"  # right_vel_rev_s
        cells[-1] = "none"
        lines.append("data," + ",".join(cells))
    (run_dir / "telemetry.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "events.log").write_text("event,0,run_started,running,ok\n", encoding="utf-8")


def test_finn_profile_loads_and_matches_seeded_model():
    profile = load_profile(PROFILE_PATH)
    assert profile.name == "finn"
    assert profile.model_path.exists(), "committed seeded model should resolve from profile"
    assert {pair.name for pair in profile.compare} == {
        "left_vel",
        "right_vel",
        "pitch_rate",
        "yaw_rate",
        "forward_accel",
    }


def test_finn_gap_pipeline_on_synthetic_batch2(tmp_path):
    pytest.importorskip("mujoco")
    from scopik.gap import run_gap

    profile = load_profile(PROFILE_PATH)
    run_dir = tmp_path / "run"
    write_synthetic_batch2_run(run_dir)

    report = run_gap(profile, run_dir)
    assert report.sim_run is not None
    assert set(report.summary) == {pair.name for pair in profile.compare}
    for summary in report.summary.values():
        assert summary["sample_count"] == 119
    labels = [message for _, message in report.events]
    assert any("phase: coastdown" in label for label in labels)
