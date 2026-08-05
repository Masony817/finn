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

## Telemetry format

The MCU writes line-prefixed CSV over serial: `schema,` and `data,` rows are
telemetry, `event,` and `status,` lines are the event log. `serial_log_capture.py`
splits them into `telemetry.csv` and `events.log` in the run directory.

Scopik profiles address telemetry columns by name, and the batch postprocessors
address them positionally by schema tag. Renaming or reordering a column is a
contract change across firmware, profile, and postprocessor; change all three in
one commit, and bump the schema tag when the layout moves.

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
