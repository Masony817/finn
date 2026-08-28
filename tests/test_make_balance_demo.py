"""Tests for the shareable balance demo and the rollout hook it depends on.

The demo drives the same controller `run_lqr_sim.py` validates, so the tests here
cover what the demo adds: the disturbance schedule, the recovery measurement, and
the on_tick hook contract that lets an external caller push the robot mid-rollout.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim" / "generated" / "seeded" / "latest" / "finn.seeded.sim.xml"


def _load(name: str, script: Path):
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


demo = _load("make_balance_demo", ROOT / "tools" / "make_balance_demo.py")
rls = _load("run_lqr_sim", ROOT / "tools" / "run_lqr_sim.py")


def _renderer_or_skip(width: int, height: int):
    """Skip rather than fail where no GL context exists, such as a bare CI runner."""

    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    try:
        renderer = rls.mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    renderer.close()


def test_pushes_alternate_direction_and_must_fit_the_rollout():
    args = demo.parse_args(["--push-at-s", "1.0", "2.0", "3.0", "--duration-s", "5"])
    pushes = demo.build_pushes(args)

    assert [push.force_n > 0 for push in pushes] == [True, False, True]
    assert all(abs(push.force_n) == args.push_force_n for push in pushes)

    too_late = demo.parse_args(["--push-at-s", "4.99", "--duration-s", "5"])
    with pytest.raises(demo.DemoError):
        demo.build_pushes(too_late)


def test_push_force_is_zero_outside_its_window():
    push = demo.Push(start_s=1.0, duration_s=0.1, force_n=20.0)

    assert push.force_at(0.99) == 0.0
    assert push.force_at(1.0) == 20.0
    assert push.force_at(1.1) == 0.0


def test_recovery_requires_the_lean_to_stay_settled():
    """A single zero crossing mid-oscillation is not a recovery."""

    push = demo.Push(start_s=0.0, duration_s=0.1, force_n=20.0)
    swinging = [
        {"time_s": 0.0, "pitch_rad": 0.20, "forward_pos_m": 0.0, "tau_balance_nm": 1.0},
        {"time_s": 0.5, "pitch_rad": 0.00, "forward_pos_m": 0.1, "tau_balance_nm": 0.5},
        {"time_s": 1.0, "pitch_rad": -0.18, "forward_pos_m": 0.2, "tau_balance_nm": -1.0},
    ]

    recovery = demo.measure_recovery(swinging, push, trim_pitch_rad=0.0)

    assert recovery is not None
    assert recovery["settle_s"] is None
    assert recovery["peak_lean_deg"] == pytest.approx(11.459, abs=1e-2)


def test_recovery_is_none_when_the_rollout_never_reached_the_push():
    push = demo.Push(start_s=8.0, duration_s=0.1, force_n=20.0)
    stopped_early = [{"time_s": 0.0, "pitch_rad": 0.0, "forward_pos_m": 0.0, "tau_balance_nm": 0.0}]

    assert demo.measure_recovery(stopped_early, push, trim_pitch_rad=0.0) is None


def test_frames_decimation_matches_the_requested_frame_rate():
    assert demo.frames_decimation(fps=20, control_dt_s=0.01) == 5
    assert demo.frames_decimation(fps=100, control_dt_s=0.01) == 1
    # Never zero: a control loop slower than the frame rate still captures every tick.
    assert demo.frames_decimation(fps=200, control_dt_s=0.01) == 1


def test_on_tick_hook_reaches_the_simulation_and_defaults_to_off():
    """The hook is how the demo pushes the robot, so it must see rows and write data."""

    model = rls.mujoco.MjModel.from_xml_path(str(MODEL))
    handles = rls.inspect_model(model)
    args = rls.parse_args(["--model", str(MODEL), "--duration-s", "0.2"])
    config = rls.SimConfig(
        control_dt_s=args.control_dt_s,
        duration_s=0.2,
        initial_pitch_rad=args.initial_pitch_rad,
        target_pitch_rad=rls.estimate_balance_trim_pitch_rad(model),
        target_forward_vel_m_s=0.0,
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
        yaw_left_actuator_sign=rls.yaw_left_actuator_sign(args),
        drive=rls.drive_limits_from_args(args),
    )
    estimator = rls.calibrated_estimator(model, handles, config)
    gain = rls.np.array([[60.0, 9.0, 13.0]])
    body_id = rls.require_id(model, rls.mujoco.mjtObj.mjOBJ_BODY, "base_link")

    seen: list[dict[str, float]] = []

    def hook(tick, row, data):
        seen.append(row)
        data.xfrc_applied[body_id, 0] = 5.0

    pushed_rows, _ = rls.run_closed_loop(model, handles, estimator, config, gain, on_tick=hook)
    quiet_rows, _ = rls.run_closed_loop(model, handles, estimator, config, gain)

    assert len(seen) == len(pushed_rows)
    assert seen[0] is pushed_rows[0], "the hook receives the row it can annotate"
    assert pushed_rows[-1]["pitch_rad"] != quiet_rows[-1]["pitch_rad"], (
        "a force written by the hook must change the trajectory"
    )


def test_demo_runs_end_to_end_and_writes_a_chart(tmp_path: Path):
    args = demo.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path),
            "--duration-s",
            "2.5",
            "--push-at-s",
            "0.5",
            "--no-gif",
        ]
    )
    result = demo.run(args)

    assert result["status"] in {"pass", "failed"}  # ran to completion, not errored
    assert len(result["recoveries"]) == 1
    assert result["recoveries"][0]["peak_lean_deg"] > 1.0, "a 22 N shove must move the robot"
    assert (tmp_path / "balance_chart.png").exists()
    assert (tmp_path / "demo.json").exists()

    with (tmp_path / "timeseries.csv").open(encoding="utf-8") as file:
        header = file.readline()
    assert "push_force_n" in header


def test_demo_writes_an_animated_gif(tmp_path: Path):
    _renderer_or_skip(240, 180)
    args = demo.parse_args(
        [
            "--model",
            str(MODEL),
            "--out-dir",
            str(tmp_path),
            "--duration-s",
            "1.0",
            "--push-at-s",
            "0.3",
            "--width",
            "240",
            "--height",
            "180",
            "--fps",
            "10",
            "--no-plot",
        ]
    )
    demo.run(args)

    from PIL import Image

    with Image.open(tmp_path / "balance.gif") as gif:
        # 100 control ticks plus the final sample, captured every 10th tick.
        assert gif.n_frames == 11
        assert gif.size == (240, 180 + max(80, round(0.22 * 180)))
