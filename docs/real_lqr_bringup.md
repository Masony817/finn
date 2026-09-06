# Real-robot LQR bring-up

This is the supported path from the committed seeded MuJoCo model to a short
free-standing Finn balance trial. The controller has not balanced the physical
robot yet. The interlocks here reduce the cost of a mistake; they do not make a
falling 8.44 kg robot safe.

## What is canonical

- `sim/generated/seeded/latest/finn.seeded.sim.xml` is the model used to design
  and validate the gain.
- `config/finn_conventions.yaml` is the sign and frame contract shared by the
  model, Scopik, and firmware.
- `tools/run_lqr_sim.py` derives the gain and exports
  `firmware/finn-mcu/control/04_lqr_balance/lqr_seeded_config.h` after a passing
  rollout. Do not tune the generated header by hand.
- `firmware/finn-mcu/control/04_lqr_balance/lqr_safety_config.h` contains the
  separately reviewed physical safety limits, including the preflight thresholds
  and the drive gate.
- `config/viz/finn_lqr.yaml` tells Scopik how to replay a recorded LQR command
  trace through the free model. It reads telemetry schema `lqr_v2`.
- `firmware/finn-mcu/tools/postprocess_lqr_balance.py` turns a captured run into
  derived model corrections, the way the Batch 1 and Batch 2 postprocessors do.
- `tools/drive_lqr_sim.py` drives the same validated controller around from the
  keyboard or a scripted profile. Sim only; see Stage 5.

The controller state is `[pitch, pitch_rate, forward_velocity]`. Positive robot
X is forward, positive Y is left, and positive Z is up. Both physical wheel
encoders are expected to be positive for robot-forward motion. The mirrored
MuJoCo left joint is negative for forward motion, which is why Scopik negates
only the simulated left-wheel sensor.

## The trial setup

Finn stands free on the floor with a Dyneema sling attached to the top of the
mast. An operator holds the sling slack and takes up no load. The robot balances
entirely on its own wheels; the sling is a catch, not a support.

That distinction is what keeps the run usable as data. A slack sling applies no
force, so the recorded response is the real free-balance response and the model
comparison means something. **A sling that goes taut becomes an external force
the telemetry cannot see**, and nothing downstream can distinguish it from robot
dynamics. If you catch Finn mid-trial, treat everything after that moment as a
catch recording rather than a balance recording, and say so in the run notes. The
postprocessor's steady-state filter drops most of it, but it cannot detect a
gentle load.

For a first free trial, use two people: one on the sling, one operating the host.
Clear the floor and keep hands, clothing, and the USB cable away from the wheels.
Have an accessible hardware motor-power cutoff. The serial `STOP` command and the
moteus watchdog are useful interlocks, but neither replaces a physical cutoff or
a person on the sling. Do not perform the first release alone.

## Before touching the floor

Run the host checks and a full 30 second sim validation:

```bash
uv sync --extra dev
uv run pytest -q
uv run python tools/run_lqr_sim.py \
  --duration-s 30 \
  --no-plot \
  --firmware-header firmware/finn-mcu/control/04_lqr_balance/lqr_seeded_config.h
```

The last command must report `status=pass`. It records the model hash, gain,
trim pitch, controller timing, and convention flags in the firmware header.

The torque envelope is `1.0 N m` per wheel. It was `0.25 N m` until 2026-08-05,
inherited from whatever hard cap the Batch 1 bench firmware happened to run with
rather than chosen. That value is below what the robot physically needs: with
both wheels capped there, gravity wins past 2.8 degrees of lean, so the arm gate
was allowing release poses the controller could not recover from. At `1.0 N m`
the sim recovers about 11.3 degrees, which covers the 8 degree arm window with
margin, while capping chassis acceleration near 3 m/s^2. The hub motors can
deliver considerably more; raising the envelope again is a later, deliberate
validation step.

## Stage 1: motor-disabled convention check

The committed firmware deliberately refuses to arm because its two hands-on
convention flags start as `false`. Flash and capture the check:

```bash
uv run python firmware/finn-mcu/tools/capture-lqr-balance.py --check-conventions
```

With Finn held and the motors stopped:

1. Type `ZERO UPRIGHT` while the chassis is mechanically upright.
2. Type `CHECK CONVENTIONS`.
3. Tip the top of Finn forward. `pitch_rad` and `pitch_rate_rad_s` must become
   positive.
4. Roll each wheel by hand in the direction that would drive the robot forward.
   Both reported wheel velocities must become positive.
5. Rotate Finn to the left, counter-clockwise seen from above. `yaw_rate_rad_s`
   must become positive. This one gates steering rather than balance, so it can
   be deferred, but it costs nothing to do in the same 15 second window.
6. Let the 15 second check finish. The run is saved under
   `logs/finn-mcu/lqr/lqr_convention_check_pass/`.

If a sign is wrong, update the corresponding sign in
`config/finn_conventions.yaml`, rerun the sim/export command so the motor-disabled
firmware receives the correction, repeat the check, and do not arm. When the
observations are correct, set the matching fields to `true`:

```yaml
imu:
  pitch_direction_bench_verified: true
  yaw_direction_bench_verified: true      # steering only
wheel_odometry:
  encoder_directions_bench_verified: true
```

The first and third gate arming at all. The yaw flag gates only the drive layer,
which is built but held off; see Stage 5.

Then rerun the 30 second sim/export command. That is the only supported way to
open the firmware arm gate.

## Stage 2: arming preflight

`ARM FINN` no longer arms directly. It spends two motors-disabled seconds
sampling every subsystem at the control rate, prints one `check,` line per test,
and arms only if all of them pass. `PREFLIGHT` runs the identical battery but
never arms, so you can iterate on a failure without touching the arm path:

```bash
uv run python firmware/finn-mcu/tools/capture-lqr-balance.py --preflight
```

Hold Finn upright and still for the whole window. What it checks, and why each
one is worth a failed arm rather than a failed trial:

| Group | Checks | What a failure means |
|---|---|---|
| Conventions | `conventions_pitch`, `conventions_wheels` | Stage 1 was not completed, or the header was not regenerated after it. |
| IMU | `imu_present`, `imu_fresh`, `imu_quat_rate_hz`, `imu_gyro_rate_hz`, `imu_accel_rate_hz`, `imu_no_reset`, `imu_quat_norm` | The estimator would run on stale or malformed attitude. Marginal I2C pull-ups show up here and nowhere else. |
| Pose | `imu_stationary`, `pitch_zeroed`, `pitch_near_trim` | The release pose is outside what the torque envelope can recover. |
| CAN and motors | `moteus_left_link_hz`, `moteus_right_link_hz`, `moteus_left_misses`, `moteus_right_misses`, `moteus_left_fault`, `moteus_right_fault`, `moteus_temp_c` | The control loop would lose a wheel mid-trial. |
| Power | `bus_voltage_stable`, `bus_voltage_floor` | The pack sags with the motors already stopped, so it will collapse under balance current. |
| Recording | `telemetry_rate_hz`, `encoders_finite` | The run would produce data too thin to learn anything from. |
| Timing | `control_budget_us` | The CAN round trip does not fit in the 10 ms control period. |
| Host | `host_heartbeat` | The host is not actually holding the deadman. |

`bus_voltage_floor` reports `skip` until you set `kMinBusVoltageV` in
`lqr_safety_config.h` from the pack spec. No bus voltage is recorded anywhere in
this repo, so the check reports the measured value rather than guessing a floor.
Setting it turns the reading into a gate.

The host runs its own check in parallel, because the firmware cannot see the far
end of the USB cable. It verifies the schema tag is one the postprocessor reads,
that every row has the expected column count, and that no non-finite value
arrived. A truncated capture is worth discovering before the release, not after.

## Stage 3: the balance trial

```bash
uv run python firmware/finn-mcu/tools/capture-lqr-balance.py
```

The sling operator holds Finn mechanically upright for `ZERO UPRIGHT`, then holds
it near the generated target trim (`0.0411 rad`, about 2.36 degrees) and still,
for the two second preflight, before `ARM FINN`. Only when both people are ready,
type `RUN LQR` and let the sling go slack.

The trial ends after `kFirstTrialDurationMs`, which is 3 seconds. `TRIAL <ms>`
selects a shorter one; it clamps to `kMaxTrialDurationMs`, which ships equal to
the reviewed 3 seconds so the selectable envelope is exactly the reviewed
envelope. Lengthening a trial means raising `kMaxTrialDurationMs` deliberately,
after short trials show the expected sign, strong real/sim correlation,
acceptable lag, and no unexplained saturation or drift. Every other limit in that
header was sized for a 3 second catch.

Catch Finn on every stop, timeout, or fault; stopped wheels do not keep the
chassis upright. The firmware also stops or refuses to run on stale IMU data,
missing host heartbeat, control-loop deadline miss, moteus fault or missing
reply, excessive pitch, wheel speed, travel, or temperature. A fault stays
latched through `STOP`; fix its cause and type `CLEAR` before rearming.

## Stage 4: read the run back

The capture wrapper runs two postprocess steps automatically.

`postprocess_lqr_balance.py` treats the trial as a sysid run and writes
`postprocess/derived.yaml` and `postprocess/report.md`. It audits recording
integrity first (schema, malformed rows, control-loop jitter, tick duration, IMU
freshness, saturation, preflight check results), then attempts five estimates and
reports an observability verdict for each:

| Estimate | What it needs | Realistic verdict on a quiet trial |
|---|---|---|
| Balance trim / COM fore-aft | A steady mean wheel torque distinguishable from its own noise | Usually identified. This is the most valuable output. |
| Pitch plant (`mgl/I`, `1/I`) | Lean and torque not collinear | Usually `not_excited`. The controller makes torque a function of lean. |
| Actuator torque tracking | Commanded torque range above 0.05 N m | Identified once the controller is working. |
| Loop latency | Correlated command and response | Identified if the trial had any real motion. |
| IMU noise floor | The stationary preflight window | Always identified. Cleanest estimate in the run. |

A closed-loop balance run is a poor identification experiment and the script is
built to say so rather than fit anyway. A parameter whose regressors arrived
collinear is reported `not_excited` with its condition number, not fitted. The
report ends by naming **one** parameter family to change next, because coupled
families cannot be separated in a single iteration.

Change that one thing, regenerate the seeded model and the gain header, and
capture another short trial.

`scopik gap` then replays the recorded command trace through the free seeded
model, producing `scopik_gap.json` and `scopik_gap.rrd`. Treat a failed physical
balance attempt as useful evidence. Scopik's LQR replay compares onboard sensor
and actuator response; it does not prove world trajectory, tire slip, or contact
fidelity, which need an independent pose measurement. Do not tune model
parameters from gain or lag when the absolute correlation for that phase is below
`0.6`.

## Where the earlier batch runs fit

- Batch 1 is the off-ground actuator characterization behind wheel damping,
  friction loss, command signs, and torque limits. Rerun it only after a motor,
  controller configuration, or wheel changes.
- Batch 2 is the gantry-supported ground-contact dataset behind the current
  loaded model checks. It validates onboard sensor/actuator response, not free
  balance or world-path fidelity. Since the gantry is no longer available, keep
  the existing clean runs as provenance rather than pretending an unsupported
  run is the same experiment.
- Short LQR captures are now the next controlled dataset. They should refine
  the transfer story without overwriting the Batch 1/2 evidence.

## Stage 5: driving, and what is still gated

The command layer is now **in** the firmware, not just in the sim. A
`CommandArbiter` holds, clamps, slew limits, and ramps stale intent to zero;
`allocateWheelTorques` gives yaw only the headroom balance leaves behind; and
`DRIVE <forward_m_s> <yaw_rad_s>` parses and validates bounded intent. The
balance loop runs every tick regardless, exactly as before.

None of it moves the robot yet, by three separate gates:

1. **`kDriveEnabled` in `lqr_safety_config.h` is `false`.** Yaw torque is forced
   to exactly zero and the position reference is pinned, so balance is identical
   to the validated station keeper. `DRIVE` is refused with a reason.
2. **`yaw_direction_bench_verified` is `false`.** Even with the gate open, a
   drive command is refused until Stage 1's yaw observation is recorded.
3. **The reviewed safety envelope forbids driving, correctly.**
   `kMaxWheelTravelRev = 2.0` faults at about 1.02 m of travel and
   `kFirstTrialDurationMs = 3000` ends the trial at 3 s, so any real drive
   attempt trips a limit within a second or two. Driving needs its own reviewed
   limit block, set from evidence produced by short balance trials, not from the
   sim.

One physical ceiling sits above all of it: `kMaxAbsPitchRad = 10 degrees` caps
sustained acceleration at `g*tan(10 deg) = 1.73 m/s^2` no matter what the command
layer allows. The sim's 0.5 m/s^2 slew limit is chosen to sit well inside that,
leaning about 2.9 degrees and leaving the rest for disturbance rejection.

Note that the two `SetPosition` calls already cost up to 6 ms of the 10 ms
control budget. Steering changes their values, not their count; the
`control_budget_us` preflight check is what watches that.

`docs/codebase-notes.md` states the layer invariants the firmware preserves.
