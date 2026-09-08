"""Open-loop command replay: drive the MuJoCo model with recorded commands.

This is the measurement at the heart of the gap profiler. The recorded
actuator commands are applied to the model tick by tick (holding each command
for the telemetry inter-sample interval), and the profile's declared sensors
are sampled after every tick. The resulting sim signals land on the real run's
timestamps, so residuals need no resampling.

Two physical-honesty features, both profile-opt-in:

- gravity_compensated samples: a MuJoCo accelerometer reports specific force
  (gravity included) in the site frame; an IMU's linear-acceleration output is
  gravity-removed. Subtracting R_site^T · (-g) per frame makes them
  comparable regardless of the robot's current orientation.
- hold_upright: if the real run was externally supported (gantry, stand) and
  the model is not, an unbalanced model simply falls over during replay and
  every comparison is polluted by fall dynamics. Holding the named free joint
  upright (roll/pitch projected out each step, yaw and translation free)
  models an ideally stiff support.

Scope caveat carried over from the original finn implementation: this is an
onboard-signal comparison. Rate/velocity signals are the honest comparison
set; absolute pose is not validated.

Intervals are rounded to physics steps and gaps are capped at MAX_DT_S.
Timing adjustments are reported explicitly; assigned recording timestamps do
not imply the simulator integrated exactly that elapsed time.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scopik.datamodel import RunData, ScopikError, Signal
from scopik.profile import Profile
from scopik.transforms import apply_transform

MAX_DT_S = 0.1  # clamp gaps (dropped serial lines) so one gap cannot free-run the sim


@dataclass(frozen=True)
class _SampleAddress:
    address: int  # sensordata address of the indexed component
    vector_address: int  # sensordata address of the sensor's first component
    site_id: int  # -1 when not site-attached
    index: int
    gravity_compensated: bool


def replay_commands(
    profile: Profile,
    real_run: RunData,
    model_path: Path | None = None,
) -> RunData:
    """Replay the real run's commands through the model; return a sim RunData."""

    try:
        import mujoco
    except ImportError as exc:
        raise ScopikError(
            "command replay requires mujoco (install scopik with the [mujoco] extra)"
        ) from exc

    replay = profile.replay
    if replay is None:
        raise ScopikError(f"profile {profile.name!r} has no replay: section")

    xml_path = model_path or profile.model_path
    if not xml_path.exists():
        raise ScopikError(f"missing model XML: {xml_path}")

    times = real_run.require_times()
    if len(times) < 2:
        raise ScopikError("need at least 2 telemetry rows to replay")

    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ScopikError("replay timestamps must be finite and strictly increasing")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Resolve all names up front so model/profile drift fails loudly, not row 30000.
    actuator_ids: list[tuple[int, np.ndarray]] = []
    for actuator_name, command_column in replay.actuators.items():
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ScopikError(f"model has no actuator {actuator_name!r}")
        commands = real_run.require_column(command_column)
        if len(commands) != len(times) or not np.all(np.isfinite(commands)):
            raise ScopikError(
                f"replay command {command_column!r} must have one finite value per row"
            )
        actuator_ids.append((int(actuator_id), commands))

    samples: list[_SampleAddress] = []
    for sample in replay.samples:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sample.sensor)
        if sensor_id < 0:
            raise ScopikError(f"model has no sensor {sample.sensor!r}")
        adr = int(model.sensor_adr[sensor_id])
        dim = int(model.sensor_dim[sensor_id])
        index = sample.index or 0
        if index < 0 or index >= dim:
            raise ScopikError(
                f"replay.sample {sample.name!r}: index {index} out of range for "
                f"sensor {sample.sensor!r} (dim {dim})"
            )
        site_id = -1
        if sample.gravity_compensated:
            if int(model.sensor_type[sensor_id]) != int(mujoco.mjtSensor.mjSENS_ACCELEROMETER):
                raise ScopikError(
                    f"replay.sample {sample.name!r}: gravity_compensated requires an "
                    f"accelerometer sensor, {sample.sensor!r} is not one"
                )
            site_id = int(model.sensor_objid[sensor_id])
        samples.append(
            _SampleAddress(
                address=adr + index,
                vector_address=adr,
                site_id=site_id,
                index=index,
                gravity_compensated=sample.gravity_compensated,
            )
        )

    hold = _resolve_hold(model, replay.hold_upright) if replay.hold_upright else None
    gravity_world = -np.array(model.opt.gravity, dtype=float)  # reaction, e.g. (0, 0, +9.81)

    timestep = float(model.opt.timestep)
    n_out = len(times) - 1
    outputs = np.empty((n_out, len(samples)), dtype=float)

    intervals = np.diff(times)
    step_counts = np.maximum(1, np.rint(np.minimum(intervals, MAX_DT_S) / timestep)).astype(int)
    timing_error = np.cumsum(step_counts * timestep - intervals)
    timing = {
        "clipped_intervals": int(np.count_nonzero(intervals > MAX_DT_S)),
        "recorded_duration_s": float(times[-1] - times[0]),
        "simulated_duration_s": float(np.sum(step_counts) * timestep),
        "max_abs_time_error_s": float(np.max(np.abs(timing_error))),
    }
    if timing["clipped_intervals"] or timing["max_abs_time_error_s"] > timestep:
        warnings.warn(f"replay integration time differs from recording: {timing}", stacklevel=2)

    for i in range(1, len(times)):
        for actuator_id, commands in actuator_ids:
            data.ctrl[actuator_id] = commands[i]
        for _ in range(step_counts[i - 1]):
            mujoco.mj_step(model, data)
            if hold is not None:
                _apply_hold(data, hold)
        if hold is not None:
            mujoco.mj_forward(model, data)  # refresh sensors after the hold projection
        for k, spec in enumerate(samples):
            if spec.gravity_compensated:
                site_rotation = data.site_xmat[spec.site_id].reshape(3, 3)
                reading = np.array(data.sensordata[spec.vector_address : spec.vector_address + 3])
                compensated = reading - site_rotation.T @ gravity_world
                outputs[i - 1, k] = float(compensated[spec.index])
            else:
                outputs[i - 1, k] = float(data.sensordata[spec.address])

    sim_times = np.asarray(times[1:], dtype=float)
    run = RunData(label="sim", times=sim_times)
    run.meta["model_path"] = str(xml_path)
    run.meta["replayed_from"] = real_run.meta.get("run_dir", "")
    run.meta["hold_upright"] = replay.hold_upright or ""
    run.meta["timing"] = timing
    for k, sample in enumerate(replay.samples):
        run.signals[sample.name] = Signal(
            name=sample.name,
            unit=sample.unit,
            times=sim_times,
            values=apply_transform(outputs[:, k], sample.transform),
            group=sample.group,
        )
    return run


@dataclass(frozen=True)
class _HoldAddresses:
    qpos_adr: int  # free joint qpos start (x y z qw qx qy qz)
    dof_adr: int  # free joint dof start (vx vy vz wx wy wz)


def _resolve_hold(model, joint_name: str) -> _HoldAddresses:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ScopikError(f"replay.hold_upright: model has no joint {joint_name!r}")
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ScopikError(f"replay.hold_upright: joint {joint_name!r} is not a free joint")
    return _HoldAddresses(
        qpos_adr=int(model.jnt_qposadr[joint_id]),
        dof_adr=int(model.jnt_dofadr[joint_id]),
    )


def _apply_hold(data, hold: _HoldAddresses) -> None:
    """Project the free joint to yaw-only rotation; zero roll/pitch rates.

    Models an ideally stiff external support (gantry): the base may translate
    and yaw but cannot roll or pitch.
    """

    quat_adr = hold.qpos_adr + 3
    qw, qx, qy, qz = data.qpos[quat_adr : quat_adr + 4]
    # Yaw (world-z) component of the quaternion, renormalized.
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    half = 0.5 * yaw
    data.qpos[quat_adr : quat_adr + 4] = (math.cos(half), 0.0, 0.0, math.sin(half))
    # Zero world-frame roll/pitch angular velocity, keep yaw rate (wz).
    data.qvel[hold.dof_adr + 3] = 0.0
    data.qvel[hold.dof_adr + 4] = 0.0
