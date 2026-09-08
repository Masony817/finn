"""MuJoCo estimation, plant identification, and closed-loop rollout."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_discrete_are

from finn.control import CommandArbiter, CommandSource, DriveLimits, allocate_wheel_torques, clamp
from finn.reporting import assess_rollout

STATE_NAMES = ("pitch_rad", "pitch_rate_rad_s", "forward_vel_m_s")
SETTLE_STEPS = 50  # The authored pose floats; identify dynamics only after tire contact settles.
YAW_SPINUP_TICKS = 20  # Clear dry friction before measuring yaw response.


class LqrSimError(Exception):
    """Expected failure with a concise user-facing message."""


@dataclass(frozen=True)
class SimConfig:
    """Simulation and controller timing.

    Each controller tick holds torque across several MuJoCo integration steps,
    matching the firmware's fixed-rate command updates.
    """

    control_dt_s: float
    duration_s: float
    initial_pitch_rad: float
    target_pitch_rad: float
    target_forward_vel_m_s: float
    position_hold_kp_s: float
    max_position_correction_m_s: float
    linearization_torque_eps_nm: float
    linearization_vel_eps_m_s: float
    fall_pitch_rad: float
    pitch_axis: int
    pitch_sign: float
    forward_sign: float
    yaw_axis: int
    yaw_sign: float
    yaw_left_actuator_sign: float
    drive: DriveLimits


@dataclass(frozen=True)
class ModelHandles:
    """Compiled MuJoCo addresses for names this controller depends on."""

    left_actuator_id: int
    right_actuator_id: int
    left_wheel_qposadr: int
    right_wheel_qposadr: int
    left_wheel_dofadr: int
    right_wheel_dofadr: int
    imu_quat_adr: int
    imu_quat_dim: int
    imu_gyro_adr: int
    imu_gyro_dim: int
    wheel_left_pos_adr: int
    wheel_left_vel_adr: int
    wheel_right_pos_adr: int
    wheel_right_vel_adr: int
    left_tire_geom_id: int
    right_tire_geom_id: int
    wheel_radius_m: float
    torque_limit_nm: float


@dataclass
class StateEstimator:
    """Robot-shaped estimator used by the LQR loop.

    In sim we could read perfect root position and body angular velocity from
    qpos/qvel.  Do not do that in the controller.  The real robot will not have
    MuJoCo qpos; it will have an IMU and motor encoders.  This estimator mirrors
    that boundary:

    - pitch comes from the calibrated IMU quaternion;
    - pitch rate comes from the IMU gyro;
    - forward velocity comes from sign-corrected wheel odometry.
    """

    model: mujoco.MjModel
    handles: ModelHandles
    config: SimConfig
    neutral_imu_quat: np.ndarray | None = None
    neutral_forward_pos_m: float = 0.0

    def calibrate(self, data: mujoco.MjData) -> None:
        """Capture the upright reference, like zeroing the robot on boot."""

        self.neutral_imu_quat = self.imu_quat(data)
        self.neutral_forward_pos_m = self.forward_pos_m(data)

    def state(self, data: mujoco.MjData) -> np.ndarray:
        """Return x = [pitch, pitch_rate, forward_vel]."""

        if self.neutral_imu_quat is None:
            raise LqrSimError("StateEstimator.calibrate() must be called before state().")

        current_quat = self.imu_quat(data)

        # MuJoCo framequat is a world orientation quaternion in wxyz order.  We
        # subtract the calibrated upright orientation in the IMU/site frame:
        # relative = inverse(neutral) * current.
        #
        # For Finn's current IMU site, this makes robot pitch land on local IMU
        # axis X, matching the firmware convention from Batch 2 telemetry.
        relative_quat = quat_mul(quat_conj(self.neutral_imu_quat), current_quat)
        pitch_rotvec = quat_to_rotvec(relative_quat)
        pitch_rad = self.config.pitch_sign * pitch_rotvec[self.config.pitch_axis]

        gyro = self.imu_gyro(data)
        pitch_rate_rad_s = self.config.pitch_sign * gyro[self.config.pitch_axis]

        forward_vel_m_s = self.forward_vel_m_s(data)

        return np.array(
            [pitch_rad, pitch_rate_rad_s, forward_vel_m_s],
            dtype=float,
        )

    def yaw_rate_rad_s(self, data: mujoco.MjData) -> float:
        """Yaw rate straight off the gyro, with no wheel-difference odometry.

        Differencing the wheels would need a track width, and the three available
        numbers disagree by 40 percent; config/finn_conventions.yaml records why.
        """

        return self.config.yaw_sign * float(self.imu_gyro(data)[self.config.yaw_axis])

    def relative_forward_pos_m(self, data: mujoco.MjData) -> float:
        return self.forward_pos_m(data) - self.neutral_forward_pos_m

    def imu_quat(self, data: mujoco.MjData) -> np.ndarray:
        adr = self.handles.imu_quat_adr
        dim = self.handles.imu_quat_dim
        return np.array(data.sensordata[adr : adr + dim], dtype=float)

    def imu_gyro(self, data: mujoco.MjData) -> np.ndarray:
        adr = self.handles.imu_gyro_adr
        dim = self.handles.imu_gyro_dim
        return np.array(data.sensordata[adr : adr + dim], dtype=float)

    def forward_pos_m(self, data: mujoco.MjData) -> float:
        left = float(data.sensordata[self.handles.wheel_left_pos_adr])
        right = float(data.sensordata[self.handles.wheel_right_pos_adr])
        return (
            self.config.forward_sign
            * signed_forward_wheel_rad(left, right)
            * (self.handles.wheel_radius_m)
        )

    def forward_vel_m_s(self, data: mujoco.MjData) -> float:
        left = float(data.sensordata[self.handles.wheel_left_vel_adr])
        right = float(data.sensordata[self.handles.wheel_right_vel_adr])
        return (
            self.config.forward_sign
            * signed_forward_wheel_rad(left, right)
            * (self.handles.wheel_radius_m)
        )


def inspect_model(model: mujoco.MjModel) -> ModelHandles:
    """Resolve names once so later code fails loudly on model/config drift."""

    left_actuator_id = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_left_wheel")
    right_actuator_id = require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_right_wheel")

    left_joint_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel")
    right_joint_id = require_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel")

    imu_quat_adr, imu_quat_dim = sensor_adr_dim(model, "imu_quat")
    imu_gyro_adr, imu_gyro_dim = sensor_adr_dim(model, "imu_gyro")
    wheel_left_pos_adr, _ = sensor_adr_dim(model, "wheel_left_pos")
    wheel_left_vel_adr, _ = sensor_adr_dim(model, "wheel_left_vel")
    wheel_right_pos_adr, _ = sensor_adr_dim(model, "wheel_right_pos")
    wheel_right_vel_adr, _ = sensor_adr_dim(model, "wheel_right_vel")

    left_tire_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_tire_collision")
    right_tire_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_tire_collision")
    require_floor_plane_at_origin(model)

    left_range = model.actuator_ctrlrange[left_actuator_id]
    right_range = model.actuator_ctrlrange[right_actuator_id]
    if not np.allclose(left_range, right_range):
        raise LqrSimError(
            f"left/right actuator ranges differ: left={left_range}, right={right_range}"
        )
    if not math.isclose(abs(float(left_range[0])), abs(float(left_range[1])), rel_tol=1e-6):
        raise LqrSimError(f"expected symmetric actuator range, got {left_range}")

    return ModelHandles(
        left_actuator_id=left_actuator_id,
        right_actuator_id=right_actuator_id,
        left_wheel_qposadr=int(model.jnt_qposadr[left_joint_id]),
        right_wheel_qposadr=int(model.jnt_qposadr[right_joint_id]),
        left_wheel_dofadr=int(model.jnt_dofadr[left_joint_id]),
        right_wheel_dofadr=int(model.jnt_dofadr[right_joint_id]),
        imu_quat_adr=imu_quat_adr,
        imu_quat_dim=imu_quat_dim,
        imu_gyro_adr=imu_gyro_adr,
        imu_gyro_dim=imu_gyro_dim,
        wheel_left_pos_adr=wheel_left_pos_adr,
        wheel_left_vel_adr=wheel_left_vel_adr,
        wheel_right_pos_adr=wheel_right_pos_adr,
        wheel_right_vel_adr=wheel_right_vel_adr,
        left_tire_geom_id=left_tire_geom_id,
        right_tire_geom_id=right_tire_geom_id,
        wheel_radius_m=wheel_radius(model),
        torque_limit_nm=abs(float(left_range[1])),
    )


def require_floor_plane_at_origin(model: mujoco.MjModel) -> None:
    """Ground settling measures tire height against z=0, so verify the floor is there."""

    floor_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if int(model.geom_type[floor_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
        raise LqrSimError("expected the 'floor' geom to be a plane")
    if not math.isclose(float(model.geom_pos[floor_id][2]), 0.0, abs_tol=1e-9):
        raise LqrSimError(
            f"expected the floor plane at z=0, found z={float(model.geom_pos[floor_id][2])}"
        )


def estimate_balance_trim_pitch_rad(model: mujoco.MjModel) -> float:
    """Return the pitch that places the whole-robot COM above the wheel axle.

    Finn's CAD-derived COM is slightly behind the axle at zero pitch. Asking the
    velocity LQR to hold zero pitch therefore forces the wheels to keep moving
    underneath that offset COM. Rotating the axle-to-COM vector until its
    forward component is zero gives the stationary balance trim.
    """

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    base_body_id = require_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    left_site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "left_wheel_center")
    right_site_id = require_id(model, mujoco.mjtObj.mjOBJ_SITE, "right_wheel_center")

    axle_pos = 0.5 * (data.site_xpos[left_site_id] + data.site_xpos[right_site_id])
    axle_to_com = data.subtree_com[base_body_id] - axle_pos
    vertical_offset_m = float(axle_to_com[2])
    if vertical_offset_m <= 0.0:
        raise LqrSimError("cannot derive balance trim: whole-robot COM is not above the wheel axle")

    return math.atan2(-float(axle_to_com[0]), vertical_offset_m)


def require_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise LqrSimError(f"model is missing expected MuJoCo object: {name}")
    return int(obj_id)


def sensor_adr_dim(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    sensor_id = require_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sensor_id]), int(model.sensor_dim[sensor_id])


def wheel_radius(model: mujoco.MjModel) -> float:
    geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_tire_collision")
    return float(model.geom_size[geom_id][0])


def validate_timing(model: mujoco.MjModel, config: SimConfig) -> None:
    if config.control_dt_s <= 0.0:
        raise LqrSimError("--control-dt-s must be positive")
    if config.duration_s <= 0.0:
        raise LqrSimError("--duration-s must be positive")
    if config.position_hold_kp_s < 0.0:
        raise LqrSimError("--position-hold-kp-s must be nonnegative")
    if config.max_position_correction_m_s < 0.0:
        raise LqrSimError("--max-position-correction-m-s must be nonnegative")
    steps = config.control_dt_s / float(model.opt.timestep)
    if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1e-9):
        raise LqrSimError(
            f"control_dt_s={config.control_dt_s} is not an integer multiple of "
            f"MuJoCo timestep={model.opt.timestep}"
        )


def validate_linearization_torque(config: SimConfig, handles: ModelHandles) -> None:
    eps = abs(config.linearization_torque_eps_nm)
    if eps <= 0.0:
        raise LqrSimError("--linearization-torque-eps-nm must be positive")
    if eps > handles.torque_limit_nm:
        raise LqrSimError(
            f"linearization torque {eps} exceeds actuator limit {handles.torque_limit_nm}"
        )
    if config.linearization_vel_eps_m_s <= 0.0:
        raise LqrSimError("--linearization-vel-eps-m-s must be positive")


def calibrated_estimator(
    model: mujoco.MjModel, handles: ModelHandles, config: SimConfig
) -> StateEstimator:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    estimator = StateEstimator(model=model, handles=handles, config=config)
    estimator.calibrate(data)
    return estimator


def linearize_balance_dynamics(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference the one-tick dynamics around upright.

    A textbook LQR starts from analytical equations.  For Finn's generated model,
    the safer first pass is to ask MuJoCo for the local dynamics of the exact XML
    you will simulate.

    Perturbation size is a physical choice here, not a numerical one.  The seeded
    wheel joints carry ~0.24 N*m of dry friction, and any probe small enough to
    stay inside that band measures stiction instead of dynamics.  The torque probe
    has always been finite for this reason; the forward-velocity probe needs the
    same treatment, because below roughly 0.01 m/s the wheels never break away and
    the model reports velocity collapsing by half every tick.  Pitch and pitch rate
    are flat across four decades of perturbation once the tires are on the ground,
    so those stay small.
    """

    state_eps = np.array(
        [1e-3, 1e-3, config.linearization_vel_eps_m_s],
        dtype=float,
    )
    zero_state = np.zeros(len(STATE_NAMES), dtype=float)

    a_matrix = np.zeros((len(STATE_NAMES), len(STATE_NAMES)), dtype=float)
    for column, eps in enumerate(state_eps):
        plus = zero_state.copy()
        minus = zero_state.copy()
        plus[column] = eps
        minus[column] = -eps
        f_plus = simulate_one_control_tick(model, handles, estimator, config, plus, 0.0)
        f_minus = simulate_one_control_tick(model, handles, estimator, config, minus, 0.0)
        a_matrix[:, column] = (f_plus - f_minus) / (2.0 * eps)

    torque_eps = config.linearization_torque_eps_nm
    f_plus = simulate_one_control_tick(model, handles, estimator, config, zero_state, torque_eps)
    f_minus = simulate_one_control_tick(model, handles, estimator, config, zero_state, -torque_eps)
    b_matrix = ((f_plus - f_minus) / (2.0 * torque_eps)).reshape(-1, 1)

    if np.linalg.norm(b_matrix) < 1e-9:
        raise LqrSimError(
            "linearized input matrix is near zero; increase --linearization-torque-eps-nm "
            "or revisit wheel friction/contact parameters"
        )

    require_controllable(a_matrix, b_matrix)
    return a_matrix, b_matrix


def linearize_yaw_dynamics(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
) -> tuple[float, float]:
    """Identify the scalar yaw plant w[k+1] = a*w[k] + b*d[k] from the model.

    Differential torque and common torque are decoupled in the linearization, so
    steering is its own one-state loop rather than a fourth LQR state; adding it
    to STATE_NAMES would break the firmware export contract for no benefit.

    Both coefficients are measured rather than derived, which is the point: a
    closed-form yaw plant needs a track width, and the CAD, compiled-model, and
    Batch 2 values disagree by 40 percent.  Measuring also means the sign of `b`
    carries the differential direction, so a mirrored joint cannot silently invert
    the steering gain the way it once inverted the velocity term.
    """

    torque_eps = config.linearization_torque_eps_nm
    zero_state = np.zeros(len(STATE_NAMES), dtype=float)

    def yaw_rate_after_tick(tau_yaw_nm: float) -> float:
        data = mujoco.MjData(model)
        set_reduced_state(model, data, handles, config, zero_state)
        apply_differential_torque(data, handles, config, tau_yaw_nm)
        step_control_tick(model, data, config)
        return estimator.yaw_rate_rad_s(data)

    b_yaw = (yaw_rate_after_tick(torque_eps) - yaw_rate_after_tick(-torque_eps)) / (
        2.0 * torque_eps
    )
    if abs(b_yaw) < 1e-9:
        raise LqrSimError(
            "identified yaw input gain is near zero; increase --linearization-torque-eps-nm "
            "or revisit wheel friction/contact parameters"
        )

    # Driving it up to speed keeps body and wheel velocities consistent without
    # anyone having to pick a track width to relate them.
    data = mujoco.MjData(model)
    set_reduced_state(model, data, handles, config, zero_state)
    for _ in range(YAW_SPINUP_TICKS):
        apply_differential_torque(data, handles, config, torque_eps)
        step_control_tick(model, data, config)
    spun_rate = estimator.yaw_rate_rad_s(data)
    apply_differential_torque(data, handles, config, 0.0)
    step_control_tick(model, data, config)
    decayed_rate = estimator.yaw_rate_rad_s(data)

    a_yaw = decayed_rate / spun_rate if abs(spun_rate) > 1e-9 else 1.0
    # A yawing robot only loses rate to friction, so anything outside (0, 1] is an
    # artifact; an undamped integrator asks more of the design than the real plant.
    if not 0.0 < a_yaw <= 1.0:
        a_yaw = 1.0
    return float(a_yaw), float(b_yaw)


def require_controllable(a_matrix: np.ndarray, b_matrix: np.ndarray) -> None:
    """Reject a plant the LQR cannot actually stabilize in every state direction.

    An uncontrollable direction leaves a closed-loop pole wherever the open-loop
    plant put it.  When that pole sits on the unit circle the rollout still looks
    stable while forward position random-walks, which is exactly what a
    free-falling linearization produces.  Catch it here rather than shipping the
    gain to hardware.
    """

    order = len(STATE_NAMES)
    controllability = np.hstack(
        [np.linalg.matrix_power(a_matrix, k) @ b_matrix for k in range(order)]
    )
    rank = int(np.linalg.matrix_rank(controllability))
    if rank < order:
        raise LqrSimError(
            f"linearized plant is uncontrollable (rank {rank} of {order}). The identified "
            "dynamics cannot be stabilized in every state direction; check that the robot "
            "is in ground contact and that the model's balance dynamics are present."
        )


def simulate_one_control_tick(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
    initial_state: np.ndarray,
    tau_balance_nm: float,
) -> np.ndarray:
    data = mujoco.MjData(model)
    set_reduced_state(model, data, handles, config, initial_state)
    apply_balance_torque(data, handles, tau_balance_nm)
    step_control_tick(model, data, config)
    return estimator.state(data)


def set_reduced_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    config: SimConfig,
    state: np.ndarray,
) -> None:
    """Set a robot-readable initial condition with the tires on the ground.

    This function is allowed to touch MuJoCo qpos/qvel because it prepares a
    simulation experiment.  The controller itself never sees these internals.

    The base height is not taken from the model's authored rest pose: that pose
    floats the tires above the floor, and every experiment started from it would
    run in free fall.  Instead the chassis is settled onto the floor once at this
    attitude, and the commanded state is then re-applied at the settled height.
    """

    apply_reduced_state(model, data, handles, state, base_z_m=None)
    base_z_m = settle_base_height_m(model, data, handles, state)
    apply_reduced_state(model, data, handles, state, base_z_m=base_z_m)
    require_ground_contact(data)


def apply_reduced_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    state: np.ndarray,
    *,
    base_z_m: float | None,
) -> None:
    """Write the full reduced state into qpos/qvel, optionally overriding base height."""

    pitch_rad, pitch_rate_rad_s, forward_vel_m_s = state
    forward_pos_m = 0.0

    mujoco.mj_resetData(model, data)

    # The freejoint position keeps the chassis above the wheels.  We move x with
    # wheel odometry so the visual/root pose and encoder pose start consistent.
    data.qpos[0] = forward_pos_m
    if base_z_m is not None:
        data.qpos[2] = base_z_m

    # A world-Y rotation appears as local IMU X pitch after subtracting the
    # neutral IMU orientation.  This was verified against the generated model.
    data.qpos[3:7] = axis_angle_quat(axis=1, angle=float(pitch_rad))

    # Freejoint qvel order is [vx, vy, vz, wx, wy, wz].  World-Y angular velocity
    # maps to the current IMU pitch-rate axis for Finn's mounted IMU frame.
    data.qvel[0] = forward_vel_m_s
    data.qvel[4] = pitch_rate_rad_s

    wheel_rad = forward_pos_m / handles.wheel_radius_m
    wheel_rad_s = forward_vel_m_s / handles.wheel_radius_m

    # Sign-corrected forward convention: left wheel positive and right wheel
    # negative roll the generated model toward world +x.  Verified by rolling the
    # joints directly against world displacement, not assumed from the mirroring
    # (see tests/test_run_lqr_sim_smoke.py::test_odometry_forward_sign_matches_world_motion).
    data.qpos[handles.left_wheel_qposadr] = wheel_rad
    data.qpos[handles.right_wheel_qposadr] = -wheel_rad
    data.qvel[handles.left_wheel_dofadr] = wheel_rad_s
    data.qvel[handles.right_wheel_dofadr] = -wheel_rad_s

    mujoco.mj_forward(model, data)


def settle_base_height_m(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    handles: ModelHandles,
    state: np.ndarray,
) -> float:
    """Return the base height at which the tires rest on the floor at this attitude.

    Lowering the chassis to exact tangency is not enough: MuJoCo only generates a
    contact once the geometries actually overlap, and the resting penetration is a
    property of the contact solver, not something worth hard-coding.  So we drop the
    tires to tangency and let the model settle, holding the base orientation fixed so
    the inverted pendulum cannot tip away from the attitude being prepared.
    """

    data.qpos[2] -= lowest_tire_gap_m(model, data, handles)
    mujoco.mj_forward(model, data)

    quat = np.array(data.qpos[3:7], dtype=float)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
        data.qpos[3:7] = quat
        data.qvel[3:6] = 0.0
    return float(data.qpos[2])


def lowest_tire_gap_m(model: mujoco.MjModel, data: mujoco.MjData, handles: ModelHandles) -> float:
    """Signed distance from the lowest tire surface down to the floor plane."""

    return min(
        float(data.geom_xpos[geom_id][2]) - handles.wheel_radius_m
        for geom_id in (handles.left_tire_geom_id, handles.right_tire_geom_id)
    )


def require_ground_contact(data: mujoco.MjData) -> None:
    """Fail loudly if an experiment is about to run with the robot in the air.

    A floating start silently removes the inverted-pendulum mode from every
    finite-difference measurement taken from this state.
    """

    if data.ncon == 0:
        raise LqrSimError(
            "robot is not touching the ground after settling; the identified plant "
            "would be a free-falling body with no balance dynamics"
        )


def discrete_lqr(
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    q_cost: np.ndarray,
    r_cost: np.ndarray,
) -> np.ndarray:
    """Solve u = -Kx for the discrete system x[k+1] = A x[k] + B u[k]."""

    p_matrix = solve_discrete_are(a_matrix, b_matrix, q_cost, r_cost)
    lhs = b_matrix.T @ p_matrix @ b_matrix + r_cost
    rhs = b_matrix.T @ p_matrix @ a_matrix
    return np.linalg.solve(lhs, rhs)


def run_closed_loop(
    model: mujoco.MjModel,
    handles: ModelHandles,
    estimator: StateEstimator,
    config: SimConfig,
    gain: np.ndarray,
    *,
    show_viewer: bool = False,
    realtime: bool | None = None,
    gain_yaw: float = 0.0,
    command_source: CommandSource | None = None,
    key_callback: Callable[[int], None] | None = None,
    on_tick: Callable[[int, dict[str, float], mujoco.MjData], None] | None = None,
) -> tuple[list[dict[str, float]], dict[str, float | bool]]:
    data = mujoco.MjData(model)
    initial_state = np.array([config.initial_pitch_rad, 0.0, 0.0], dtype=float)
    set_reduced_state(model, data, handles, config, initial_state)

    rows: list[dict[str, float]] = []
    control_steps = round(config.duration_s / config.control_dt_s)
    saturated_count = 0
    finite = True
    fell = False
    stopped_by_viewer = False
    arbiter = CommandArbiter(limits=config.drive)
    reference_forward_pos_m = 0.0
    pace_to_wall_clock = show_viewer if realtime is None else realtime

    viewer_context = (
        mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        if show_viewer
        else nullcontext(None)
    )
    with viewer_context as viewer:
        wall_start_s = time.perf_counter()
        if viewer is not None:
            viewer.sync()

        for tick in range(control_steps + 1):
            if tick > 0 and viewer is not None and not viewer.is_running():
                stopped_by_viewer = True
                break

            time_s = tick * config.control_dt_s
            state = estimator.state(data)
            yaw_rate_rad_s = estimator.yaw_rate_rad_s(data)
            forward_pos_m = estimator.relative_forward_pos_m(data)

            # Must run before the torque below, unlike on_tick, or the command
            # lands a tick late.
            command = arbiter.step(command_source, time_s, config.control_dt_s)
            reference_forward_pos_m += command.forward_vel_m_s * config.control_dt_s
            reference_forward_pos_m = clamp(
                reference_forward_pos_m,
                forward_pos_m - config.drive.reference_position_band_m,
                forward_pos_m + config.drive.reference_position_band_m,
            )
            target_forward_pos_m = (
                config.target_forward_vel_m_s * time_s
                if command_source is None
                else reference_forward_pos_m
            )
            position_error_m = forward_pos_m - target_forward_pos_m
            position_velocity_correction_m_s = clamp(
                -config.position_hold_kp_s * position_error_m,
                -config.max_position_correction_m_s,
                config.max_position_correction_m_s,
            )
            effective_target_forward_vel_m_s = (
                config.target_forward_vel_m_s
                + command.forward_vel_m_s
                + position_velocity_correction_m_s
            )
            target_state = np.array(
                [config.target_pitch_rad, 0.0, effective_target_forward_vel_m_s],
                dtype=float,
            )
            error = state - target_state

            raw_tau = float((-gain @ error.reshape(-1, 1)).item())
            raw_tau_yaw = -gain_yaw * (yaw_rate_rad_s - command.yaw_rate_rad_s)
            tau, tau_yaw = allocate_wheel_torques(raw_tau, raw_tau_yaw, handles.torque_limit_nm)
            left_cmd_nm, right_cmd_nm = yaw_torque_to_wheels(tau_yaw, config.yaw_left_actuator_sign)
            left_cmd_nm += tau
            right_cmd_nm += tau
            saturated = not math.isclose(raw_tau, tau, rel_tol=0.0, abs_tol=1e-12)
            saturated_count += int(saturated)

            rows.append(
                {
                    "time_s": time_s,
                    "pitch_rad": float(state[0]),
                    "pitch_rate_rad_s": float(state[1]),
                    "yaw_rate_rad_s": yaw_rate_rad_s,
                    "forward_pos_m": forward_pos_m,
                    "target_forward_pos_m": target_forward_pos_m,
                    "forward_vel_m_s": float(state[2]),
                    "target_forward_vel_m_s": effective_target_forward_vel_m_s,
                    "cmd_forward_vel_m_s": command.forward_vel_m_s,
                    "cmd_yaw_rate_rad_s": command.yaw_rate_rad_s,
                    "command_stale": float(arbiter.stale),
                    "tau_balance_raw_nm": raw_tau,
                    "tau_balance_nm": tau,
                    "tau_yaw_raw_nm": raw_tau_yaw,
                    "tau_yaw_nm": tau_yaw,
                    "left_cmd_nm": left_cmd_nm,
                    "right_cmd_nm": right_cmd_nm,
                    "saturated": float(saturated),
                }
            )

            # Called with the state that produced this row, so anything it writes to
            # data (an external push, a camera) lands on the step taken just below.
            if on_tick is not None:
                on_tick(tick, rows[-1], data)

            finite = (
                finite
                and np.all(np.isfinite(state))
                and all(
                    math.isfinite(value)
                    for value in (yaw_rate_rad_s, raw_tau, raw_tau_yaw, left_cmd_nm, right_cmd_nm)
                )
            )
            fell = fell or abs(float(state[0])) > config.fall_pitch_rad
            if tick == control_steps or not finite or fell:
                break

            apply_wheel_torques(data, handles, left_cmd_nm, right_cmd_nm)
            step_control_tick(model, data, config)

            if viewer is not None:
                viewer.sync()
            if pace_to_wall_clock:
                target_wall_s = wall_start_s + (tick + 1) * config.control_dt_s
                remaining_s = target_wall_s - time.perf_counter()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)

    return rows, assess_rollout(
        rows,
        config,
        finite=bool(finite),
        fell=fell,
        stopped_by_viewer=stopped_by_viewer,
        saturated_count=saturated_count,
        driven=command_source is not None,
        rejected_samples=arbiter.rejected_samples,
    )


def apply_wheel_torques(
    data: mujoco.MjData, handles: ModelHandles, left_nm: float, right_nm: float
) -> None:
    data.ctrl[:] = 0.0
    data.ctrl[handles.left_actuator_id] = left_nm
    data.ctrl[handles.right_actuator_id] = right_nm


def apply_balance_torque(data: mujoco.MjData, handles: ModelHandles, tau_balance_nm: float) -> None:
    apply_wheel_torques(data, handles, tau_balance_nm, tau_balance_nm)


def apply_differential_torque(
    data: mujoco.MjData, handles: ModelHandles, config: SimConfig, tau_yaw_nm: float
) -> None:
    left_nm, right_nm = yaw_torque_to_wheels(tau_yaw_nm, config.yaw_left_actuator_sign)
    apply_wheel_torques(data, handles, left_nm, right_nm)


def yaw_torque_to_wheels(tau_yaw_nm: float, left_actuator_sign: float) -> tuple[float, float]:
    """Split a positive-is-left yaw torque across the two wheel actuators.

    The sign is a contract value, not a derivation: the generated model names its
    wheel bodies opposite the robot frame, so guessing it from the actuator names
    turns left into right.  config/finn_conventions.yaml carries it and
    test_positive_yaw_torque_turns_the_model_left holds it to measured behaviour.
    """

    return left_actuator_sign * tau_yaw_nm, -left_actuator_sign * tau_yaw_nm


def step_control_tick(model: mujoco.MjModel, data: mujoco.MjData, config: SimConfig) -> None:
    inner_steps = round(config.control_dt_s / float(model.opt.timestep))
    for _ in range(inner_steps):
        mujoco.mj_step(model, data)


def signed_forward_wheel_rad(left_rad: float, right_rad: float) -> float:
    """Average wheel rotation after converting each encoder to forward-positive.

    The generated model mirrors the left wheel joint, so the two raw joint
    coordinates run opposite each other.  Rolling the joints directly shows that
    left-positive / right-negative carries the chassis toward world +x, so that is
    the combination that means "forward" here.  Getting this backwards inverts the
    velocity feedback term and turns the balance loop into positive feedback.
    """

    return 0.5 * (left_rad - right_rad)


def axis_angle_quat(*, axis: int, angle: float) -> np.ndarray:
    half = 0.5 * angle
    quat = np.array([math.cos(half), 0.0, 0.0, 0.0], dtype=float)
    quat[axis + 1] = math.sin(half)
    return quat


def quat_conj(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_mul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    xyz = quat[1:4]
    xyz_norm = float(np.linalg.norm(xyz))
    if xyz_norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(xyz_norm, float(quat[0]))
    return xyz / xyz_norm * angle
