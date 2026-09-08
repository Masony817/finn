"""Assess rollout traces and write plots and telemetry."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from finn.simulation import SimConfig

COMMAND_SETTLE_S = 1.5
MIN_TRACKING_SAMPLES = 50
SETTLED_PITCH_RAD = 0.08
GRAVITY_M_S2 = 9.81


def assess_rollout(
    rows: list[dict[str, float]],
    config: SimConfig,
    *,
    finite: bool,
    fell: bool,
    stopped_by_viewer: bool,
    saturated_count: int,
    driven: bool,
    rejected_samples: int,
) -> dict[str, float | bool]:
    max_abs_pitch = max(abs(row["pitch_rad"]) for row in rows)
    final_abs_pitch = abs(rows[-1]["pitch_rad"])
    max_abs_position_error = max(
        abs(row["forward_pos_m"] - row["target_forward_pos_m"]) for row in rows
    )
    final_abs_position_error = abs(rows[-1]["forward_pos_m"] - rows[-1]["target_forward_pos_m"])
    saturation_fraction = saturated_count / max(1, len(rows))
    settle_ticks = round(COMMAND_SETTLE_S / config.control_dt_s)
    velocity_tracking_p95, velocity_tracking_max, velocity_scored = settled_tracking_error(
        rows, "cmd_forward_vel_m_s", "forward_vel_m_s", settle_ticks
    )
    yaw_tracking_p95, yaw_tracking_max, yaw_scored = settled_tracking_error(
        rows, "cmd_yaw_rate_rad_s", "yaw_rate_rad_s", settle_ticks
    )
    velocity_tracking_assessed = velocity_scored >= MIN_TRACKING_SAMPLES
    yaw_tracking_assessed = yaw_scored >= MIN_TRACKING_SAMPLES

    if not driven:
        # Station keeping: the robot was asked to hold a spot and end upright, so
        # judge it on both.
        held_the_reference = config.position_hold_kp_s == 0.0 or (
            max_abs_position_error < 0.25 and final_abs_position_error < 0.10
        )
        ended_upright = final_abs_pitch < SETTLED_PITCH_RAD
    else:
        # An interactive session may never hold a command still long enough to
        # grade, so an unassessed axis is skipped rather than passed on no evidence.
        held_the_reference = (not velocity_tracking_assessed or velocity_tracking_p95 < 0.25) and (
            not yaw_tracking_assessed or yaw_tracking_p95 < 0.40
        )
        # Holding acceleration a costs a steady lean of atan(a/g), so a rollout
        # stopped mid ramp is upright exactly when it sits inside that.
        commanded_lean_rad = math.atan2(config.drive.forward_accel_limit_m_s2, GRAVITY_M_S2)
        settle_slack_rad = SETTLED_PITCH_RAD - abs(config.target_pitch_rad)
        ended_upright = (
            abs(rows[-1]["pitch_rad"] - config.target_pitch_rad)
            < commanded_lean_rad + settle_slack_rad
        )

    passed = (
        finite
        and not fell
        and ended_upright
        and max_abs_pitch < config.fall_pitch_rad
        and held_the_reference
        and saturation_fraction < 0.80
    )
    return {
        "pass": passed,
        "finite": finite,
        "fell": fell,
        "stopped_by_viewer": stopped_by_viewer,
        "max_abs_pitch_rad": max_abs_pitch,
        "final_abs_pitch_rad": final_abs_pitch,
        "ended_upright": ended_upright,
        "final_forward_pos_m": rows[-1]["forward_pos_m"],
        "final_forward_vel_m_s": rows[-1]["forward_vel_m_s"],
        "max_abs_position_error_m": max_abs_position_error,
        "final_abs_position_error_m": final_abs_position_error,
        "velocity_tracking_assessed": velocity_tracking_assessed,
        "velocity_tracking_samples": velocity_scored,
        "yaw_tracking_assessed": yaw_tracking_assessed,
        "yaw_tracking_samples": yaw_scored,
        "velocity_tracking_p95_m_s": velocity_tracking_p95,
        "velocity_tracking_max_m_s": velocity_tracking_max,
        "yaw_tracking_p95_rad_s": yaw_tracking_p95,
        "yaw_tracking_max_rad_s": yaw_tracking_max,
        "max_abs_wheel_cmd_nm": max(
            max(abs(row["left_cmd_nm"]), abs(row["right_cmd_nm"])) for row in rows
        ),
        "rejected_command_samples": rejected_samples,
        "saturation_fraction": saturation_fraction,
        "samples": len(rows),
    }


def write_plot(path: Path, rows: list[dict[str, float]]) -> bool:
    try:
        import matplotlib

        # mjpython runs the simulation script on a worker thread on macOS. A
        # file-only plot must therefore use a non-interactive backend.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    time_s = [row["time_s"] for row in rows]
    pitch = [row["pitch_rad"] for row in rows]
    forward_pos = [row["forward_pos_m"] for row in rows]
    tau = [row["tau_balance_nm"] for row in rows]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(time_s, pitch)
    axes[0].set_ylabel("pitch rad")
    axes[1].plot(time_s, forward_pos)
    axes[1].plot(
        time_s,
        [row["target_forward_pos_m"] for row in rows],
        linestyle="--",
        label="target",
    )
    axes[1].set_ylabel("forward m")
    axes[1].legend()
    axes[2].plot(time_s, tau)
    axes[2].set_ylabel("torque Nm")
    axes[2].set_xlabel("time s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def settled_tracking_error(
    rows: list[dict[str, float]], command_key: str, actual_key: str, settle_ticks: int
) -> tuple[float, float, int]:
    """Tracking error over steady, nonzero commands, as (p95, max, samples_scored).

    Three filters, each for a different false failure.  Settled samples only, so
    the slew ramps are not read as tracking error.  Nonzero commands only, because
    grading a standing robot against a stop is a test it always wins.  A percentile
    rather than the worst sample, because a hard spin makes the tires stick and
    slip and odometry reads a slip as a momentary metre per second.  The max is
    returned beside it, and the count so the caller can tell an unmeasured run
    from a good one.
    """

    errors = []
    for index in range(settle_ticks, len(rows)):
        window = rows[index - settle_ticks : index + 1]
        target = window[-1][command_key]
        if any(abs(row[command_key] - target) > 1e-9 for row in window):
            continue
        if abs(target) <= 1e-9:
            # Counting stop commands is how this gate once reported a confident
            # pass on an interactive run it had never scored moving.
            continue
        errors.append(abs(window[-1][actual_key] - target))
    if not errors:
        return 0.0, 0.0, 0
    return float(np.percentile(errors, 95.0)), max(errors), len(errors)


def write_timeseries(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
