# scopik

**A sim-to-real gap profiler built on [Rerun](https://rerun.io).**

Every robotics project with a simulator eventually asks the same three questions:
*how wrong is my model, where in the run does it diverge, and is it getting better as I
tune it?* Answering them usually means a pile of private matplotlib scripts. scopik is
that pile, done once, properly: bring a MuJoCo model, a robot log, and a ~40-line YAML
profile — get a scrubbable dashboard and a tracked gap history.

```sh
scopik gap --profile config/viz/finn.yaml --run logs/finn-mcu/sysid/batch_2_pass/<ts>
```

```
sim-to-real gap  (5 signals, model: .../finn.seeded.sim.xml)
  signal           unit         rmse        mae  max |err|       n
  ----------------------------------------------------------------
  left_vel         rev/s      0.4387     0.1365      2.979   34075
  right_vel        rev/s      0.4114     0.1241      2.831   34075
  pitch_rate       rad/s     0.01553   0.004634     0.2011   34075
  yaw_rate         rad/s     0.09119    0.03268     0.8195   34075
  forward_accel    m/s²       0.1637    0.07488      1.846   34075
```

## What it does

`scopik gap` loads a recorded run, replays its recorded actuator commands **open-loop**
through the MuJoCo model (each command held for the real inter-sample interval), samples
the model's sensors on the real run's own timestamps, and logs everything to Rerun:

- **Overlay charts** — real and sim in the same plot per compared signal.
- **Residual + rolling RMSE** — model error *localized in time*. A residual spike at
  breakaway points at stiction; error during a PRBS segment points at inertia or
  damping; drift during coastdown points at friction.
- **Events and phases** — lifecycle events and phase-column changes as timeline
  annotations, so a 10-minute sysid run is navigable by segment.
- **Summary** — per-signal RMSE / MAE / max-error table, in the viewer and as
  `scopik_gap.json` next to the run.
- **Deterministic diagnosis** — per phase and signal, compute bias, RMS, gain,
  correlation, and lag. Unsupported gain/lag estimates are written as `null`
  instead of being inferred from quiet sensor noise; findings aggregate the
  strongest patterns without requiring an LLM.
- **Gap history** — each run appends one JSON line (timestamp, model hash, per-signal
  metrics) to `<profile>_gap_history.jsonl`. Tune the model, rerun, watch the numbers
  move. This is the regression-tracking habit that makes the gap actually close.

## Reading the dashboard

The viewer opens with three kinds of tabs:

| Tab | Question it answers |
|---|---|
| **Overview** | How wrong is everything at a glance? All real-vs-sim overlays, plus the summary table and event log. |
| **One tab per signal** | Where does this signal diverge? Big overlay on top, residual + rolling RMSE below. Scrub to the spikes. |
| **Telemetry** | What was the robot actually doing? Raw signal groups (attitude, wheels, commands, health, ...). |

The timeline panel at the bottom scrubs every panel in sync. Double-click a plot to
expand it; drag on the timeline to zoom a segment.

## Physical honesty

Two problems bite every naive sim-vs-real comparison, and scopik has explicit,
opt-in answers for both:

**Gravity in accelerometers.** A MuJoCo accelerometer reports *specific force*
(gravity included) in the sensor's site frame; real IMUs typically report
gravity-removed linear acceleration. `gravity_compensated: true` on a replay sample
subtracts the gravity-reaction term using the site's live orientation each frame —
correct at any attitude, not just upright. On finn this took the forward-acceleration
"gap" from a meaningless 9.7 m/s² (≈ g) to a real 0.16 m/s².

**Externally supported runs.** If the real robot was on a gantry or stand during the
recording, the bare model has no such support — an unbalanced model simply falls over
during open-loop replay, and every comparison is polluted by fall dynamics.
`hold_upright: <free_joint>` models an ideally stiff support: roll and pitch are
projected out every physics step; yaw and translation stay free. The summary panel
states when a hold is active.

And the standing caveat: this is an **onboard-signal, open-loop** comparison.
Rate and velocity signals are the honest comparison set; absolute pose is not
validated by replay.

## The profile

One YAML file adapts scopik to a robot. Paths are relative to the profile file.

```yaml
name: myrobot
model: ../sim/myrobot.xml            # MJCF the replay runs against

source:                              # how to parse the log
  type: prefixed_csv                 # or csv, or "mypkg.sources:MySource"
  file: telemetry.csv
  line_prefix: "data,"               # rows start with this tag
  header_marker: "data,t_us,"        # the first matching line is the CSV header

time: {column: t_us, transform: us_to_s}

signals:                             # column -> chart, with units and transforms
  pitch_rad:      {unit: rad,   group: attitude}
  left_vel_rev_s: {unit: rev/s, group: wheels}
  motor_temp_c:   {unit: "°C",  group: health}

events:                              # optional: timeline annotations
  file: events.log
  prefix: "event,"
  time_index: 1                      # csv field holding the timestamp
  time_transform: us_to_s
  phase_column: phase                # text column whose changes get annotated

replay:                              # optional: enables the sim side + metrics
  actuators: {motor_left: left_cmd_nm}   # model actuator <- command column
  hold_upright: root_freejoint           # optional: ideal external support
  sample:                                # model sensors -> sim signals
    left_vel_rev_s: {sensor: wheel_left_vel, transform: rad_to_rev, unit: rev/s}
    forward_accel:  {sensor: imu_accel, index: 2, gravity_compensated: true}

compare:                             # real signal vs sim signal, by name
  - {name: left_vel, real: left_vel_rev_s, sim: left_vel_rev_s, unit: rev/s}
```

Transforms: `rev_to_rad`, `rad_to_rev`, `deg_to_rad`, `us_to_s`, `ms_to_s`, `negate`,
`{scale: x, offset: y}`, or a list to chain them. The full worked example is finn's
[`config/viz/finn.yaml`](../../config/viz/finn.yaml).

## CLI

```
scopik gap --profile P.yaml --run RUN_DIR      # open the native viewer (default)
    --save out.rrd        write a shareable recording instead
    --serve               host for a connecting viewer
    --no-replay           charts + events only, no sim / metrics
    --model M.xml         override the profile's model
    --window-s 2.0        rolling RMSE window (default 1.0)
    --out gap.json        metrics JSON path (default: RUN_DIR/scopik_gap.json)
    --history H.jsonl     gap history path; --no-history to skip
```

Recordings (`.rrd`) open with `rerun file.rrd` or the web viewer — that's the
share-a-run story: send the file, the recipient scrubs the same dashboard.

For a Finn-specific physical interpretation, invoke `finn-gap-review` in Codex
or `/finn-gap-review` in Claude Code. Both entrypoints use the same checked-in
workflow and treat Scopik's numeric output as evidence rather than asking an LLM
to re-estimate time-series features.

## Extending

- **New log format**: implement `read_table(path, profile) -> (numeric, text)` in your
  own package and set `source.type: "mypkg.sources:MySource"`. No scopik changes.
- **Different robot**: write a profile. The core has no robot-specific code.

## Status & roadmap

Built and used inside the [finn](../../README.md) self-balancing robot project; the
core is robot-agnostic and will be extracted to its own repo once interfaces settle.

- `scopik gap` — works (this document).
- `scopik live` — next: stream serial telemetry onto the timeline during bench runs,
  with a pre-logged sim reference run for realtime sim-vs-real overlay.
- 3D scene views — later, pulled in when a milestone needs them.
