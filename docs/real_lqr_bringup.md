# Real-robot LQR bring-up

This is the supported path from the committed seeded MuJoCo model to a short
unsupported-floor Finn trial. The controller has not balanced the physical robot
yet. The interlocks here reduce the cost of a mistake; they do not make a falling
8.44 kg robot safe.

## What is canonical

- `sim/generated/seeded/latest/finn.seeded.sim.xml` is the model used to design
  and validate the gain.
- `config/finn_conventions.yaml` is the sign and frame contract shared by the
  model, Scopik, and firmware.
- `tools/run_lqr_sim.py` derives the gain and exports
  `firmware/finn-mcu/control/04_lqr_balance/lqr_seeded_config.h` after a passing
  rollout. Do not tune the generated header by hand.
- `firmware/finn-mcu/control/04_lqr_balance/lqr_safety_config.h` contains the
  separately reviewed physical safety limits.
- `config/viz/finn_lqr.yaml` tells Scopik how to replay a recorded LQR command
  trace through the free model.

The controller state is `[pitch, pitch_rate, forward_velocity]`. Positive robot
X is forward, positive Y is left, and positive Z is up. Both physical wheel
encoders are expected to be positive for robot-forward motion. The mirrored
MuJoCo left joint is negative for forward motion, which is why Scopik negates
only the simulated left-wheel sensor.

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

For a first unsupported trial, use two people: one holding and catching Finn,
and one operating the host. Clear the floor and keep hands, clothing, and the
USB cable away from the wheels. Have an accessible hardware motor-power cutoff.
The serial `STOP` command and moteus watchdog are useful interlocks, but neither
replaces a physical cutoff or a catch operator. Do not perform the first release
alone.

## Stage 1: motor-disabled convention check

The committed firmware deliberately refuses to arm because the two hands-on
convention flags start as `false`. Flash and capture the check:

```bash
uv run python firmware/finn-mcu/tools/capture-lqr-balance.py \
  --check-conventions
```

With Finn held and the motors stopped:

1. Type `ZERO UPRIGHT` while the chassis is mechanically upright.
2. Type `CHECK CONVENTIONS`.
3. Tip the top of Finn forward. `pitch_rad` and `pitch_rate_rad_s` must become
   positive.
4. Roll each wheel by hand in the direction that would drive the robot forward.
   Both reported wheel velocities must become positive.
5. Let the 15 second check finish. The run is saved under
   `logs/finn-mcu/lqr/lqr_convention_check_pass/`.

If either sign is wrong, update the corresponding sign in
`config/finn_conventions.yaml`, rerun the sim/export command so the motor-disabled
firmware receives the correction, repeat the check, and do not arm. When both
observations are correct, set these two fields to `true`:

```yaml
imu:
  pitch_direction_bench_verified: true
wheel_odometry:
  encoder_directions_bench_verified: true
```

Then rerun the 30 second sim/export command. That is the only supported way to
open the firmware arm gate.

## Stage 2: first three-second balance attempt

Start a capture, which flashes `lqr_balance`, sends a 10 Hz heartbeat, and sends
`STOP` three times before releasing the serial port:

```bash
uv run python firmware/finn-mcu/tools/capture-lqr-balance.py
```

The catch operator should hold Finn mechanically upright for `ZERO UPRIGHT`,
then hold it near the generated target trim (`0.0411 rad`, about 2.36 degrees)
before `ARM FINN`. Only when both people are ready, type `RUN LQR` and release.
The first trial ends after three seconds. Catch Finn on every stop, timeout, or
fault; stopped wheels do not keep the chassis upright.

The firmware also stops or refuses to run on stale IMU data, missing host
heartbeat, control-loop deadline miss, moteus fault or missing reply, excessive
pitch, wheel speed, travel, or temperature. A fault stays latched through
`STOP`; fix its cause and type `CLEAR` before rearming.

## Use Scopik after each attempt

After the motors stop, the capture wrapper automatically runs the recorded
command trace through the free seeded model. Each finalized run contains:

- `telemetry.csv` and `events.log`: the source evidence;
- `scopik_gap.json`: deterministic per-signal and per-phase metrics;
- `scopik_gap.rrd`: the Rerun dashboard;
- `manifest.json`: capture status, commands, and postprocess result.

Treat a failed physical balance attempt as useful evidence. Compare pitch rate,
forward wheel velocity, command saturation, correlation, and lag over the short
balance phase. Change one controller or model assumption at a time, rerun the
30 second sim, regenerate the header, and repeat a short real trial.

Scopik's LQR replay compares onboard sensor and actuator response. It does not
prove world trajectory, tire slip, or contact fidelity; those require an
independent pose measurement. Do not tune model parameters from gain or lag when
the absolute correlation for that phase is below `0.6`.

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

Only lengthen `kFirstTrialDurationMs`, relax a safety limit, or raise the torque
cap after repeated short trials show the expected sign, strong real/sim
correlation, acceptable lag, and no unexplained saturation or drift.
