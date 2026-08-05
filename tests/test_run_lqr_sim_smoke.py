"""Smoke test: the committed seed model runs end-to-end through the LQR sim.

This is deliberately minimal. Its job is to catch the "works on my machine"
regression -- the seed-model bundle going missing, or the model/script drifting
apart -- on a fresh clone in CI, not to assert controller quality.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_lqr_sim.py"
MODEL = ROOT / "sim" / "generated" / "seeded" / "latest" / "finn.seeded.sim.xml"

SPEC = importlib.util.spec_from_file_location("run_lqr_sim", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rls = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rls
SPEC.loader.exec_module(rls)


def test_committed_seed_model_is_present():
    # If this fails, a fresh clone cannot run the sim -- the bundle must ship.
    assert MODEL.exists(), f"missing committed seed model: {MODEL}"


def test_viewer_is_opt_in():
    assert rls.parse_args([]).viewer is False
    assert rls.parse_args(["--viewer"]).viewer is True


def test_seeded_model_has_expected_forward_balance_trim():
    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    trim_rad = rls.estimate_balance_trim_pitch_rad(model)

    assert 0.03 < trim_rad < 0.05


def test_experiments_start_in_ground_contact():
    """The whole controller design rests on this.

    The model's authored rest pose floats the tires a few millimetres above the
    floor. An experiment started from that pose free-falls through the entire
    finite-difference tick, and a free-falling body feels no gravity torque about
    its COM -- so the identified plant silently loses its inverted-pendulum mode.
    """

    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    config = rls.parse_args(["--model", str(MODEL)])
    sim_config = _sim_config(config, handles, model)
    data = rls.mujoco.MjData(model)

    rls.set_reduced_state(model, data, handles, sim_config, rls.np.zeros(3))

    assert data.ncon > 0, "robot must be touching the floor before any measurement"


def test_odometry_forward_sign_matches_world_motion():
    """Pin the wheel sign against physics, not against the model's mirroring.

    signed_forward_wheel_rad() decides which combination of the two mirrored joint
    coordinates counts as "forward". Getting it backwards inverts the LQR velocity
    term and turns the balance loop into positive feedback, so assert it against
    which way the chassis actually travels.
    """

    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    config = rls.parse_args(["--model", str(MODEL)])
    sim_config = _sim_config(config, handles, model)

    data = rls.mujoco.MjData(model)
    rls.set_reduced_state(model, data, handles, sim_config, rls.np.zeros(3))
    start_x = float(data.qpos[0])

    # Roll the joints in the combination the odometry helper calls "forward".
    left_rate, right_rate = 2.0, -2.0
    assert rls.signed_forward_wheel_rad(left_rate, right_rate) > 0
    for _ in range(100):
        data.qvel[handles.left_wheel_dofadr] = left_rate
        data.qvel[handles.right_wheel_dofadr] = right_rate
        rls.mujoco.mj_step(model, data)

    assert float(data.qpos[0]) - start_x > 0.005, (
        "wheel rotation the odometry reports as forward must carry the robot toward world +x"
    )


def test_identified_plant_keeps_the_inverted_pendulum_mode():
    """A balancing robot's linearization must contain an unstable pole.

    Both a floating start and a stiction-bound perturbation produce a plant whose
    poles all sit inside or on the unit circle -- a robot that cannot fall over.
    The LQR happily designs a gain for that fiction.
    """

    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    config = rls.parse_args(["--model", str(MODEL)])
    sim_config = _sim_config(config, handles, model)
    estimator = rls.calibrated_estimator(model, handles, sim_config)

    a_matrix, b_matrix = rls.linearize_balance_dynamics(model, handles, estimator, sim_config)
    eigenvalues = rls.np.linalg.eigvals(a_matrix)

    assert max(abs(eigenvalues)) > 1.0, "open-loop plant must be unstable; the robot falls over"
    # Gravity must couple pitch into pitch acceleration.
    assert a_matrix[1, 0] > 0.01
    rls.require_controllable(a_matrix, b_matrix)


def _sim_config(args, handles, model):
    """Build a SimConfig the way run() does, without running a rollout."""

    return rls.SimConfig(
        control_dt_s=args.control_dt_s,
        duration_s=args.duration_s,
        initial_pitch_rad=args.initial_pitch_rad,
        target_pitch_rad=rls.estimate_balance_trim_pitch_rad(model),
        target_forward_vel_m_s=args.target_forward_vel_m_s,
        position_hold_kp_s=args.position_hold_kp_s,
        max_position_correction_m_s=args.max_position_correction_m_s,
        linearization_torque_eps_nm=args.linearization_torque_eps_nm,
        linearization_vel_eps_m_s=args.linearization_vel_eps_m_s,
        fall_pitch_rad=args.fall_pitch_rad,
        pitch_axis=args.pitch_axis,
        pitch_sign=args.pitch_sign,
        forward_sign=args.forward_sign,
    )


def test_lqr_sim_runs_on_committed_model(tmp_path: Path):
    args = rls.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path),
            "--duration-s",
            "0.2",
            "--no-plot",
        ]
    )
    result = rls.run(args)

    assert result["status"] in {"pass", "failed"}  # ran to completion, not errored
    gain = result["lqr"]["gain"]  # type: ignore[index]
    assert len(gain) == 1 and len(gain[0]) == len(rls.STATE_NAMES)
    assert 0.03 < result["control"]["target_pitch_rad"] < 0.05  # type: ignore[index]
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "timeseries.csv").exists()


def test_default_position_hold_limits_30_second_drift(tmp_path: Path):
    args = rls.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path),
            "--duration-s",
            "30",
            "--no-plot",
        ]
    )
    result = rls.run(args)
    metrics = result["metrics"]

    assert result["status"] == "pass"
    assert metrics["max_abs_position_error_m"] < 0.15  # type: ignore[index]
    assert metrics["final_abs_position_error_m"] < 0.05  # type: ignore[index]


def test_firmware_header_is_exported_from_a_passing_sim(tmp_path: Path):
    header = tmp_path / "lqr_seeded_config.h"
    args = rls.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path / "run"),
            "--duration-s",
            "1",
            "--no-plot",
            "--firmware-header",
            str(header),
        ]
    )

    result = rls.run(args)
    text = header.read_text(encoding="utf-8")

    assert result["status"] == "pass"
    assert result["model_sha256_12"] in text
    assert "kGainPitch" in text
    assert "kRealLeftEncoderForwardSign = 1.0f" in text
    assert "kPitchDirectionBenchVerified = false" in text
    assert "kWheelEncoderDirectionsBenchVerified = false" in text
