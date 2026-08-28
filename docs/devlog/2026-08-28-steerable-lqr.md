# 2026-08-28 - steerable LQR: WASD teleop as the first tenant of a command layer

Raw notes for a later write-up. Everything below was checked against the working
tree on `fix/lqr-grounded-linearization`.

## What shipped

- `tools/drive_lqr_sim.py` (new, 351 lines): WASD teleop plus two scripted
  profiles (`square`, `spin`) over the existing balance controller.
- `tests/test_drive_lqr_sim.py` (new, 264 lines): pure-arithmetic tests for the
  arbiter, the torque allocator, and the keyboard latch. No MuJoCo needed.
- `tools/run_lqr_sim.py`: +529 lines. Command layer, yaw loop, yaw plant
  identification, torque allocation, new pass gate, `on_tick` hook.
- `config/finn_conventions.yaml`: +32 lines. New `yaw:` block.
- `docs/codebase-notes.md`: +64 lines. Control layering and steering signs.
- `docs/real_lqr_bringup.md`: +51 lines. Stage 1 yaw check, new Stage 3.
- `firmware/finn-mcu/control/04_lqr_balance/lqr_seeded_config.h`: +15 lines.
- `CLAUDE.md`: new hard rule 4, old rules 4/5/6 renumbered to 5/6/7.
- Zero firmware source changes. `main.cpp` references none of the new constants
  (`grep -c "kYaw\|kMaxForwardVel\|kDriveAccel" main.cpp` returns 0).

## The actual point: the layer boundary

WASD was the excuse. The goal was a seam an HRI or navigation policy can plug
into later without any of them being able to cost the robot its balance.

```
L3  command sources   WASD teleop | scripted profile | (later) a policy
L2  command arbiter   holds, clamps, slew-limits, ramps stale commands to zero
L1  balance + yaw     runs every tick whether or not a command ever arrives
L0  safety / arming   stops everything above it; nothing above relaxes it
```

- A command source's whole signature is `Callable[[float], DriveCommand]`. It
  sees the clock. It gets no `MjData`, no model handles, no gains. That is the
  entire enforcement mechanism, and it is enforced by the type, not by review.
- `DriveCommand` is two floats: `forward_vel_m_s`, `yaw_rate_rad_s`. Frozen
  dataclass. That is the complete vocabulary L3 may speak.
- Five invariants, each with a test named for it. Written down in
  `docs/codebase-notes.md`, promoted to hard rule 4 in `CLAUDE.md`.
- Nice accident of the design: releasing a key is byte-identical to a dropped
  serial link and to a policy that stopped publishing. All three land in
  `CommandArbiter._held`. So every time someone drives with the keyboard they are
  exercising the policy-failure path.
- The `on_tick` hook that `make_balance_demo.py` uses for shoves is deliberately
  *not* the teleop seam: it hands out `MjData` and fires after the torque is
  already computed. Two hooks, two trust levels.

## Design decision: yaw is not a fourth LQR state

- Common torque and differential torque are decoupled in the linearization, so
  steering is a separate one-state loop reusing the same `discrete_lqr()`.
- Adding a fourth state to `STATE_NAMES` would have broken the firmware export
  contract and changed the committed balance gain. Not worth it for a channel
  that does not couple.
- Consequence: the balance gain is still exactly
  `[61.8993272, 9.56442891, 13.0267381]`, byte-identical to before.

## Design decision: no track width anywhere in the controller

Three numbers for the same physical quantity, disagreeing by 40 percent:

| value | source |
|---|---|
| 0.517 m | CAD body origins (`left_wheel` y = -0.009, `right_wheel` y = +0.508) |
| 0.583 m | compiled tire contact patch separation (measured off the compiled model: 0.584 m) |
| 0.351 m | Batch 2 differential-yaw response, `aggregation.json: effective_track_width_estimate_m = 0.351004`, gantry-biased |

- Rather than pick one, `linearize_yaw_dynamics()` identifies the scalar yaw
  plant `w[k+1] = a*w[k] + b*d[k]` by finite difference on the model, the same
  technique already used for the balance A/B.
- Measured values: `a_yaw = 0.8752`, `b_yaw = +0.0337`.
- Spin-up uses torque, not a written-in yaw-rate initial condition, precisely
  because writing one would require a track width to keep body and wheel
  velocities consistent.
- `a_yaw` is clamped: anything outside `(0, 1]` falls back to `1.0`, an undamped
  integrator, which asks the design to work harder than the real plant will.
- Yaw rate comes straight off `imu_gyro[1]`, never from differencing wheels, for
  the same no-track-width reason.

## The real bug: the model's wheel names are backwards

- Symptom: the identified `b_yaw` came out **negative**, contradicting a hand
  derivation.
- Cause, confirmed in `sim/generated/seeded/latest/finn.seeded.sim.xml` line 67
  and 75: `motor_left_wheel` drives the body at world y = **-0.009**, and
  `motor_right_wheel` drives the one at y = **+0.508**. Robot +y is left. The
  names come from the CAD export.
- Why nothing caught it earlier: balance sends both wheels the *same* torque.
  Steering is the first feature in the repo that needs them to differ. The bug
  had been sitting in the model harmlessly since the model existed.
- Fix per hard rule 2 (signs live in the contract, never as a compensating
  negation at a call site): `config/finn_conventions.yaml` gained
  `yaw.sim_left_actuator_yaw_sign: -1`.
- Pinned by `test_positive_yaw_torque_turns_the_model_left`, which applies +0.3
  N.m of differential torque for 40 ticks and asserts the *world* quaternion
  rotates counter-clockwise. It pins against measured model behaviour, not
  against the actuator names, which is the whole point.
- Open question worth a look: `real_left_actuator_yaw_sign` is also `-1` in the
  conventions file, even though the surrounding comment says the physical robot
  is not mirrored and its controllers are labelled correctly. It carries its own
  bench check (`yaw_direction_bench_verified: false`), so nothing is unsafe, but
  the value and the comment do not obviously agree.

## Design decision: balance has torque priority

- `allocate_wheel_torques(tau_balance, tau_yaw, limit)` clamps balance first,
  then gives yaw only `limit - abs(tau_common)`.
- The naive alternative, summing then clamping per wheel, lets a turn request eat
  balance authority *asymmetrically*. That is a fall.
- Property under this rule: a saturated balance loop steers not at all. Tested
  directly: `allocate_wheel_torques(2.0, 5.0, 1.0)` returns `(1.0, 0.0)`.
- `test_no_allocation_can_put_a_wheel_outside_the_envelope` parametrizes five
  hostile pairs and asserts both wheel commands land inside the envelope by
  construction.

## Design decision: the accel slew limit is safety, not smoothing

- Holding forward acceleration `a` costs a steady lean of `atan(a/g)`.
- The reviewed firmware pitch fault is `kMaxAbsPitchRad = 10 degrees`, so
  acceleration is physically capped at `g*tan(10 deg) = 1.73 m/s^2` before Finn
  faults out. No command layer can change that.
- Default `forward_accel_limit_m_s2 = 0.5` leans `atan(0.5/9.81) = 2.9 degrees`,
  leaving the rest of the 10 degree budget for disturbance rejection.
- This is the nicest fact from the whole session: a rate limiter whose value is
  derived from a fault threshold, not from feel.

## Tuning, all measured

Yaw gain sweep, `--drive-profile spin`, 1.5 rad/s command, reproduced today:

| `--q-yaw` | gain | settled yaw rate | error | settled `tau_yaw` |
|---|---|---|---|---|
| 4 | 0.4675 | 0.703 rad/s | 53% | 0.372 N.m |
| 100 | 5.7142 | 1.431 rad/s | 5% | 0.384 N.m |

- The steady differential torque is ~0.38 N.m *regardless of gain*. That is the
  friction load, not a controller property. Recognising that is what made the low
  gain obviously wrong rather than arguably conservative: the loop was not
  fighting inertia, it was failing to overcome a constant.
- Default set to `--q-yaw 100`, exported as `kGainYawRate = 5.71421979f`.

Max yaw rate sweep:

- At 1.5 rad/s, spinning couples into fore-aft motion. Reproduced: the spin
  profile at 1.5 rad/s with q=100 gives `vel_track_p95 = 0.296`, which **fails**
  the 0.25 gate, while yaw tracking is fine at 0.074.
- At the 1.0 rad/s default the same profile passes: `vel_track_p95 = 0.142`,
  `yaw_track_p95 = 0.071`.
- The in-code note records the settled forward-velocity spread growing from 0.22
  to 0.30 m/s between the two.
- Default set to `max_yaw_rate_rad_s = 1.0`, which is still 360 degrees in 6.3 s.
- This is the balance-priority principle applied to picking an envelope rather
  than to allocating a torque.

## Two mistakes worth recording

**Duplicated default silently shadowed the new one.**

- `drive_lqr_sim.py` had its own `--q-yaw` default of 4.0 while `run_lqr_sim.py`
  had the new 100.0. Changing the gain appeared to do nothing.
- Caught by noticing the reported differential torque exactly matched the OLD
  gain times the error. Arithmetic on a telemetry column found it, not reading.
- Fixed by single-sourcing every drive argument into
  `run_lqr_sim.add_drive_arguments()`, which `drive_lqr_sim.parse_args` now calls.
  `drive_lqr_sim.py` defines zero drive defaults of its own.

**A diagnostic that "proved" odometry was broken was itself broken.**

- The diagnostic measured the velocity of the model's *origin* body. The wheel
  midline sits at y = (-0.009 + 0.508)/2 = 0.2495 m; the root body origin sits at
  y = 0. (`finn_measurements.yaml: com_lateral_m = 0.24903644` is the same
  offset.)
- So while the robot pivoted in place, the point being measured traced a circle
  of radius 0.25 m and reported a healthy forward velocity.
- Odometry had been accurate the entire time.
- Lesson: before concluding an estimator is broken, check *what point* the
  reference measurement is attached to.

## Pass-gate design

The original station-keeping gate (`max_abs_position_error < 0.25` and
`final_abs_position_error < 0.10`) fails every successful drive, because a
driven rollout is not trying to hold a spot. Driven rollouts are judged on
command tracking instead: `velocity_tracking_p95 < 0.25` and
`yaw_tracking_p95 < 0.40`. Two refinements were needed:

- **Only score settled samples.** `settled_tracking_error()` requires the command
  to have been unchanged for `COMMAND_SETTLE_S = 1.5` s. Scoring the slew ramps
  flags the rate limiter for doing exactly its job.
- **95th percentile, not max.** A hard spin makes the tires stick and slip, and
  odometry reads a slip as a momentary spurious forward velocity that no
  controller could command away. The max is reported alongside so genuine
  transients stay visible: the `square` profile reports
  `vel_track_p95=0.069 (max 0.071)`, the default `spin` reports
  `vel_track_p95=0.142 (max 0.353)`, and that gap is the slip.

## Firmware export: additive by construction

- `lqr_seeded_config.h` gained 11 new constants across 15 added lines (3 comment
  lines and a blank). Note: not "15 constants".
- All **20** pre-existing constants are byte-identical. Verified by diffing lines
  1-25 of the old and new header.
- New: `kYawAxis`, `kYawSign`, `kGainYawRate`, `kRealLeftActuatorYawSign`,
  `kMaxForwardVelMS`, `kMaxYawRateRadS`, `kDriveAccelLimitMS2`,
  `kDriveYawAccelLimitRadS2`, `kCommandTimeoutMs`, `kRefPositionBandM`,
  `kYawDirectionBenchVerified`.
- Exported now, consumed by nothing. The reason is in
  `test_the_drive_envelope_reaches_the_firmware_header`: exporting now keeps the
  later firmware port a firmware-only change instead of creating a second place
  to tune a robot.
- `kCommandTimeoutMs = 500` is bounded by a test against the 10 Hz host command
  rate: at least 3 and at most 10 missed messages. It covers a *different*
  failure from `kHeartbeatTimeoutMs = 300`: the heartbeat covers the link dying,
  the command timeout covers the link staying healthy while whatever produces
  commands stalls.

## Verification

- `uv run pytest -q`: **190 passed in 4.38s**.
- `uv run ruff check .`: clean. `ruff format --check`: 49 files already formatted.
- PlatformIO is not installed on this machine, so `pio run` could not be run. The
  headers were instead compile-checked directly with
  `clang++ -std=c++17 -fsyntax-only`, including the two `static_assert`s from
  `main.cpp` (`kHardTorqueLimitNm <= kTorqueLimitNm`, `kControlPeriodUs == 10000`).
  Both hold. `lqr_safety_config.h` needs a `PI` shim outside Arduino.
- End-to-end: `drive_lqr_sim.py --drive-profile square --no-viewer` passes with
  `max_abs_pitch_rad=0.118`, `max_abs_wheel_cmd_nm=1.000`.

## What blocks real-robot driving

Three gates, documented in `docs/real_lqr_bringup.md` Stage 3, in order:

1. **Finn has not balanced unsupported yet.** Both arming
   `*_bench_verified` flags are still `false`. Steering a robot that has not
   stood up is not a meaningful experiment.
2. **The reviewed safety envelope forbids driving, correctly.**
   `kMaxWheelTravelRev = 2.0` faults at about 1.02 m of travel
   (2 rev x 2*pi x 0.081 m), and `kFirstTrialDurationMs = 3000` ends the trial at
   3 s. Any real drive attempt trips a limit within a second or two. Driving
   needs its own reviewed limit block, set from short-balance-trial evidence.
3. **`kMaxAbsPitchRad = 10 degrees` caps sustained acceleration** at 1.73 m/s^2
   no matter what the command layer permits.

When those clear, the port is mechanical because the layering already matches:
`runControllerTick()` is already always-on and already independent of serial
input. What it needs is a command arbiter struct beside
`last_target_forward_vel_m_s` (main.cpp:160), `sendBalanceTorque` (main.cpp:302)
widened to `sendWheelTorques(tau_common, tau_yaw)` with the same headroom
allocation, a `DRIVE <v> <w>` branch in `handleCommand` (main.cpp:687, and the
first numeric parsing in any sketch in this repo), and a `schema,lqr_v2` bump
with `config/viz/finn_lqr.yaml` updated in the same commit. The two
`SetPosition` calls already cost up to 6 ms of the 10 ms control budget;
steering changes their values, not their count.

## Bench check added

`docs/real_lqr_bringup.md` Stage 1 gained step 5: with Finn held and motors
stopped, rotate left (counter-clockwise from above) and confirm `yaw_rate_rad_s`
goes positive, then set `imu.yaw_direction_bench_verified: true`. It fits in the
existing 15 second window. Unlike the other two flags it gates only the drive
layer, not arming, so it can be deferred.
