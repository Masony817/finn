"""Smoke test: the committed seed model runs end-to-end through the LQR sim.

This is deliberately minimal. Its job is to catch the "works on my machine"
regression -- the seed-model bundle going missing, or the model/script drifting
apart -- on a fresh clone in CI, not to assert controller quality.
"""

from __future__ import annotations

import math
from pathlib import Path

from finn import control
from finn import lqr as cli
from finn import simulation as rls

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_lqr_sim.py"
MODEL = ROOT / "sim" / "generated" / "seeded" / "latest" / "finn.seeded.sim.xml"


def test_committed_seed_model_is_present():
    # If this fails, a fresh clone cannot run the sim -- the bundle must ship.
    assert MODEL.exists(), f"missing committed seed model: {MODEL}"


def test_viewer_is_opt_in():
    assert cli.parse_args([]).viewer is False
    assert cli.parse_args(["--viewer"]).viewer is True


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
    config = cli.parse_args(["--model", str(MODEL)])
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
    config = cli.parse_args(["--model", str(MODEL)])
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
    config = cli.parse_args(["--model", str(MODEL)])
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
        yaw_axis=args.yaw_axis,
        yaw_sign=args.yaw_sign,
        yaw_left_actuator_sign=cli.yaw_left_actuator_sign(args),
        drive=cli.drive_limits_from_args(args),
    )


def test_lqr_sim_runs_on_committed_model(tmp_path: Path):
    args = cli.parse_args(
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
    result = cli.run(args)

    assert result["status"] in {"pass", "failed"}  # ran to completion, not errored
    gain = result["lqr"]["gain"]  # type: ignore[index]
    assert len(gain) == 1 and len(gain[0]) == len(rls.STATE_NAMES)
    assert 0.03 < result["control"]["target_pitch_rad"] < 0.05  # type: ignore[index]
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "timeseries.csv").exists()


def test_default_position_hold_limits_30_second_drift(tmp_path: Path):
    args = cli.parse_args(
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
    result = cli.run(args)
    metrics = result["metrics"]

    assert result["status"] == "pass"
    assert metrics["max_abs_position_error_m"] < 0.15  # type: ignore[index]
    assert metrics["final_abs_position_error_m"] < 0.05  # type: ignore[index]


def test_firmware_header_is_exported_from_a_passing_sim(tmp_path: Path):
    header = tmp_path / "lqr_seeded_config.h"
    args = cli.parse_args(
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

    result = cli.run(args)
    text = header.read_text(encoding="utf-8")

    assert result["status"] == "pass"
    assert result["model_sha256_12"] in text
    assert "kGainPitch" in text
    assert "kRealLeftEncoderForwardSign = 1.0f" in text
    assert "kPitchDirectionBenchVerified = false" in text
    assert "kWheelEncoderDirectionsBenchVerified = false" in text


# ---------------------------------------------------------------------------
# The command layer sits above the balance loop. These tests hold that boundary:
# balance must not depend on a command source, and no command source may cost it.
# docs/codebase-notes.md states the invariants these are named for.
# ---------------------------------------------------------------------------


def _rollout(command_source=None, duration_s: float = 2.0):
    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    args = cli.parse_args(["--model", str(MODEL), "--duration-s", str(duration_s)])
    config = _sim_config(args, handles, model)
    estimator = rls.calibrated_estimator(model, handles, config)
    a_matrix, b_matrix = rls.linearize_balance_dynamics(model, handles, estimator, config)
    gain = rls.discrete_lqr(
        a_matrix,
        b_matrix,
        rls.np.diag(rls.np.array(args.q_diag, dtype=float)),
        rls.np.array([[float(args.r)]]),
    )
    a_yaw, b_yaw = rls.linearize_yaw_dynamics(model, handles, estimator, config)
    gain_yaw = float(
        rls.discrete_lqr(
            rls.np.array([[a_yaw]]),
            rls.np.array([[b_yaw]]),
            rls.np.array([[float(args.q_yaw)]]),
            rls.np.array([[float(args.r_yaw)]]),
        ).item()
    )
    return cli.run_closed_loop(
        model,
        handles,
        estimator,
        config,
        gain,
        gain_yaw=gain_yaw,
        command_source=command_source,
    )


def test_balance_runs_identically_with_no_command_source():
    """Invariant 1: the balance loop is complete without anything above it.

    An explicit stop command and no command source at all must produce the same
    rollout, which is what lets the generated firmware header keep being exported
    from the plain station-keeping path.
    """

    without, _ = _rollout(command_source=None)
    with_stop, _ = _rollout(command_source=lambda _t: control.DriveCommand(0.0, 0.0))

    assert len(without) == len(with_stop)
    for left, right in zip(without, with_stop, strict=True):
        assert left == right


def test_positive_yaw_torque_turns_the_model_left():
    """Pin the steering sign against physics, not against the actuator names.

    The generated model names its wheel bodies opposite the robot frame, so a sign
    read off `motor_left_wheel` turns left into right. Balance never noticed
    because both wheels get the same torque. Same failure class as the odometry
    sign inversion above.
    """

    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    args = cli.parse_args(["--model", str(MODEL)])
    config = _sim_config(args, handles, model)
    estimator = rls.calibrated_estimator(model, handles, config)

    data = rls.mujoco.MjData(model)
    rls.set_reduced_state(model, data, handles, config, rls.np.zeros(3))
    start_quat = rls.np.array(data.qpos[3:7])
    for _ in range(40):
        rls.apply_differential_torque(data, handles, config, 0.3)
        rls.step_control_tick(model, data, config)

    relative = rls.quat_mul(rls.quat_conj(start_quat), rls.np.array(data.qpos[3:7]))
    world_yaw_rad = float(rls.quat_to_rotvec(relative)[2])

    assert world_yaw_rad > 0.01, "positive yaw torque must rotate the model counter-clockwise"
    assert estimator.yaw_rate_rad_s(data) > 0.0, "the estimator must agree with world yaw"


def test_identified_yaw_plant_is_a_damped_integrator():
    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    args = cli.parse_args(["--model", str(MODEL)])
    config = _sim_config(args, handles, model)
    estimator = rls.calibrated_estimator(model, handles, config)

    a_yaw, b_yaw = rls.linearize_yaw_dynamics(model, handles, estimator, config)

    assert 0.0 < a_yaw <= 1.0, "a yawing robot only ever loses rate to friction"
    assert b_yaw > 0.0, "positive yaw torque must raise the yaw rate"


def test_balance_survives_a_hostile_command_source():
    """Invariants 2 and 3: a command source cannot take the robot down.

    Whatever a tenant does -- raise, emit NaN, or demand far outside the envelope
    -- the arbiter turns it into no-command or a clamped command, and the balance
    loop underneath keeps its footing.
    """

    def raises(_time_s):
        raise RuntimeError("policy crashed")

    def not_finite(_time_s):
        return control.DriveCommand(float("nan"), float("inf"))

    def absurd(_time_s):
        return control.DriveCommand(1e6, -1e6)

    def wrong_type(_time_s):
        return (1.0, 2.0)

    for source in (raises, not_finite, absurd, wrong_type):
        rows, metrics = _rollout(command_source=source)

        assert not metrics["fell"], f"{source.__name__} toppled the robot"
        assert metrics["finite"], f"{source.__name__} produced non-finite state"
        assert metrics["max_abs_wheel_cmd_nm"] <= 1.0 + 1e-9, (
            f"{source.__name__} escaped the torque envelope"
        )
        assert all(math.isfinite(row["left_cmd_nm"]) for row in rows)
