# Codebase notes

Cross-cutting facts that are easy to get wrong and belong to no single file.
Conventions and rules live in `CLAUDE.md`; this is the reference behind them.

## Signal frames

`config/finn_conventions.yaml` is the contract. Robot +x is forward, +y left,
+z up. The BNO085 is mounted so its own axes do not line up with the robot's:

| Robot quantity | IMU sensor axis | Sign |
|---|---|---|
| pitch, pitch rate | X | +1 |
| yaw, yaw rate | Y | +1 |
| forward acceleration | Z | -1, sensor +Z points aft |

Both physical wheel encoders read positive for robot-forward motion. The
generated MuJoCo model mirrors its left joint, so the simulated left wheel reads
negative for the same motion and Scopik negates only that sensor. These are
separate from the model's own joint-to-world mapping, which is not the same
pairing; `config/finn_conventions.yaml` explains why and holds both.

## MuJoCo

**Accelerometers report specific force**, gravity included, in the sensor site
frame. The firmware publishes BNO085 linear acceleration with gravity already
removed. Comparing them raw shows a ~9.7 m/s^2 gap that is just g, which is why
replay samples take `gravity_compensated: true` and subtract the gravity-reaction
term from the site's live orientation each frame.

**`framequat` objtype matters.** `objtype="body"` reports the body's inertial
(principal-axis) frame, not its body frame; `objtype="xbody"` reports the body
frame. They differ whenever the inertia tensor is not axis-aligned, and the
difference is silent. The seeded model reads attitude off `objtype="site"`
sensors, which sidesteps this.

**A bare model falls over.** Finn is an inverted pendulum, so open-loop replay of
a gantry-supported recording tips within about two seconds and pollutes every
signal with fall dynamics. `hold_upright: <freejoint>` projects roll and pitch
out each step to model an ideally stiff support, leaving yaw and translation
free. Use it only for runs that really were externally supported.

## Control layering

The balance loop is always on. Everything that steers Finn sits above it and can
only move a bounded reference, never a torque. WASD teleop is the first tenant of
that layer; an HRI or navigation policy is meant to be the next one, and the
boundary is shaped so a slow or crashed tenant degrades to "no command" rather
than to "no balance".

```
L3  command sources   WASD teleop | scripted profile | (later) a policy
      |               returns DriveCommand(forward_vel_m_s, yaw_rate_rad_s)
L2  command arbiter   holds, clamps, slew-limits, ramps stale commands to zero
      |               output is a reference, never a torque
L1  balance + yaw     runs every tick whether or not a command ever arrives
      |               balance torque allocated first, yaw gets the headroom
L0  safety / arming   stops everything above it; nothing above relaxes it
```

Five invariants, each with a test named for it in
`tests/test_run_lqr_sim_smoke.py` and `tests/test_drive_lqr_sim.py`:

1. **L1 never depends on L3 existing.** With no command source, the rollout is
   identical to the station-keeper, which is what lets the generated firmware
   header keep being exported from the plain balance path.
2. **L3 cannot reach torque.** A command source takes `time_s` and returns a
   `DriveCommand`. It gets no `MjData`, no handles, no gains. This is why the
   `on_tick` hook `make_balance_demo.py` uses for shoves is *not* the teleop seam:
   `on_tick` hands out `MjData`, and it fires after the torque is already computed.
3. **L2 fails closed and fails soft.** A source that raises, returns non-finite or
   wrong-typed values, or goes silent is held briefly and then *ramped* to zero.
   Ramped, not stepped: a step is itself a disturbance the balance loop must reject.
4. **L2 decouples rates.** L1 runs at 100 Hz; a policy will run at 5-30 Hz with
   jitter. Sampling is non-blocking and the last command is held, so a slow tenant
   cannot stall a control tick.
5. **Balance has torque priority.** `allocate_wheel_torques` clamps the balance
   torque first and gives yaw only the leftover headroom, so a saturated balance
   loop steers not at all rather than losing authority to a turn request.

Key release, a dropped link, and a stopped policy are the same event to L2, which
is why the teleop release path is also the policy-failure path.

## Viewer keys

**The MuJoCo viewer binds a shortcut to every letter A-Z**, so no teleop scheme
built on letters can avoid a collision. `key_callback` is called *in addition to*
the viewer's own handling, not instead of it, and there is no way to consume the
event. W is wireframe, S is shadow, A is auto connect, D is static body.

None of these touch control - they are rendering flags, and `data.ctrl` is
rewritten every tick regardless - but the scene strobes while you drive.

Only half of it can be suppressed, and the split is worth knowing before trying
again. A and D are `MjvOption` vis flags, reachable as `Handle.opt`, so
`ViewerFlagKeeper` pins them back before each `sync()`. W and S are `MjvScene`
*render* flags, and the passive viewer exposes no render scene at all: `Handle`
has `opt`, `cam`, `perturb`, `user_scn`, `m`, `d`, `viewport`, and `user_scn` is a
separate overlay for custom geometry, not the scene the viewer draws with. The
internal `_Simulate` does not expose one either.

**So arrow keys are the scheme to prefer in the viewer.** WASD still drives, and
`drive_lqr_sim.py` says at startup which letters it could not tame.

Flag indices are looked up from `mjVISSTRING`/`mjRNDSTRING` rather than hardcoded,
so a release that moves a shortcut moves the pin with it, and a test fails if a
render scene ever appears on `Handle` and makes W and S pinnable.

## Hold to drive

`key_callback` is `Callable[[int], None]`: a keycode and nothing else. No release
events, and no way to separate a press from an auto-repeat. Everything a callback
can manage on its own is therefore a latch - stamp each press, treat the key as
held while the stamp is fresh - and that is a poor joystick. It needs auto-repeat
to arrive, it stutters when the repeat delay outlasts the window, and it keeps
driving for a whole window after release. A 30 s hand-driven session produced no
command plateau longer than a single tick.

GLFW knows the true state of every key and MuJoCo already depends on it. The one
missing piece is the window handle, which the viewer does not expose. It can be
taken instead: `key_callback` runs on the thread holding the GL context, so the
first keypress captures it with `glfw.get_current_context()`. After that
`held()` reads real state, and holding a key holds the command.

Two things make this safe rather than clever. Reading key state off-thread races
the render loop by at most a frame, which a 100 Hz loop cannot notice. And every
GLFW call is guarded, falling back to the latch, so a closed window or an
unusable handle degrades driving instead of ending the run. `report.json` records
`key_state_polling` so a run says which path it actually used.

## Judging a driven rollout

`run_lqr_sim.py`'s pass gate assumes station keeping: hold a spot, end upright at
trim. Every one of those criteria is wrong for a rollout that was told to drive,
and each was found by a driven run failing for a reason that was actually correct
behaviour.

- **Position hold** becomes command tracking, since a driving robot is not trying
  to hold a spot.
- **Ending upright** allows the lean a still-accelerating command requires.
  Holding forward acceleration `a` costs a steady lean of `atan(a/g)`, so a run
  that stops mid-ramp is upright exactly when it sits inside that plus the
  settling slack.
- **Tracking error** is scored only on samples whose command has been steady for
  `COMMAND_SETTLE_S`, so the slew ramps are not counted as failures, and only on
  *nonzero* commands, and as a p95 rather than a max because a hard spin makes the
  tires stick and slip.

That last filter matters more than it looks. Grading zero commands means grading a
standing robot against a stop, which it always wins. A 30 s hand-driven session
scored 1186 such samples, reported `yaw_tracking_p95 = 0.0009` while yaw had been
commanded to plus and minus 1.0 rad/s, and passed - because a human never holds a
key still for 1.5 s, so not one moving sample was ever graded. The metrics now
carry `*_tracking_assessed` and `*_tracking_samples`, and an unassessed gate says
so instead of printing a reassuring number.

Interactive sessions routinely come back unassessed. That is the honest answer:
the checks that need no steady command - did not fall, ended upright, stayed
inside the torque envelope, did not saturate - still apply.

## Steering signs

Two traps, both recorded in `config/finn_conventions.yaml`.

**The generated model's wheel names are backwards.** `motor_left_wheel` drives the
wheel at world y = -0.009 and `motor_right_wheel` the one at y = +0.508, while
robot +y is left. The names come from the CAD export. Balance never noticed,
because both wheels get the same torque; steering is the first thing that needs
them to differ, and a sign read off the names turns left into right.
`yaw.sim_left_actuator_yaw_sign` carries it, and
`test_positive_yaw_torque_turns_the_model_left` holds it to measured behaviour.

**There is no usable track width.** Three values disagree: 0.517 m from CAD body
origins, 0.583 m between compiled tire contact patches, and 0.351 m from the
gantry-biased Batch 2 yaw response. So no track width appears in the controller at
all. `linearize_yaw_dynamics` identifies the scalar yaw plant by finite difference
on the model, the same way the balance A/B are identified, which also means the
sign of the identified input gain carries the steering direction rather than a
constant someone derived.

Yaw rate itself comes straight off `imu_gyro[1]`, never from differencing the
wheels, for the same reason.

## Telemetry format

The MCU writes line-prefixed CSV over serial: `schema,` and `data,` rows are
telemetry, and `event,`, `status,`, and `check,` lines are the event log.
`serial_log_capture.py` splits them into `telemetry.csv` and `events.log` in the
run directory.

Scopik profiles address telemetry columns by name, and the batch postprocessors
address them positionally by schema tag. Renaming or reordering a column is a
contract change across firmware, profile, and postprocessor; change all three in
one commit, and bump the schema tag when the layout moves.

The LQR controller is on `lqr_v2`, which added command-layer, timing, and
microsecond IMU-age columns to `lqr_v1`. `postprocess_lqr_balance.py` reads it by
name off the header row and refuses a run whose schema tag it does not know,
rather than silently misreading columns. Two tests hold the format together:
`test_firmware_header_and_row_have_the_same_column_count` counts printf
conversions against the header string in `main.cpp`, and
`test_scopik_lqr_profile_only_reads_columns_the_firmware_emits` stops the profile
drifting away from the firmware.

## Arming preflight

`ARM FINN` does not arm. It enters a two second motors-stopped state that samples
the IMU and the CAN bus at the control rate, then evaluates a battery of named
checks and arms only if every one passes. `PREFLIGHT` runs the same battery and
returns to idle, so a failure can be chased without touching the arm path.

Sampling over a window rather than probing once is the whole point. The failures
that end a balance trial are intermittent: a marginal I2C pull-up that drops one
report in twenty, a CAN termination fault that costs replies under load, a pack
that only sags when queried at 100 Hz. A single-shot check sees none of them.

The checks are emitted as `check,<t_us>,<name>,<pass|fail|skip>,<measured>,<limit>,<detail>`
so the operator, the terminal, and the postprocessor all read the same record.
`skip` exists for a real case rather than as a placeholder: `bus_voltage_floor`
has no value to compare against, because no bus voltage is recorded anywhere in
this repo, so it reports the measurement and declines to gate until
`kMinBusVoltageV` is set.

The host runs a parallel check the firmware cannot: `LiveIntegrityMonitor` in
`capture-lqr-balance.py` watches the schema tag, per-row column count, and
non-finite values as they land, because the MCU has no way to know what actually
arrived at the other end of the USB cable.

## Host tooling

`tools/*.py` and `firmware/finn-mcu/tools/*.py` are standalone scripts, not an
importable package, so tests load them through
`importlib.util.spec_from_file_location`. Scopik is a real workspace package and
is imported normally. Adding a script means following the first pattern.

Rerun is pinned to 0.34.x and every call into its API lives in
`packages/scopik/src/scopik/rrlog/`. Keep it there: the SDK is pre-1.0 and moves,
and containment is what makes a version bump a one-directory change.

## Model provenance

`sim/model/finn/finn_robot.xml` is an onshape-to-robot export, kept as a source
and re-exported when the CAD changes. Its `<!-- Part ... -->` labels are
generator output. Do not hand-tune inertias or geometry here; correct the CAD or
the measurement, then rebuild through the seeded-model pipeline.
