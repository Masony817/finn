# FINN

**FINN** is a self-balancing, two-wheeled robot — an open hardware/software
platform for balance control, system identification, and sim-to-real work. This
repo holds everything for the robot: Teensy firmware, a system-ID pipeline that
turns bench measurements into a physically-grounded MuJoCo model, and a MuJoCo
sim with a first LQR balance controller.

Status: early and under active solo development. Interfaces move fast and the
controllers are sim-validated, not yet hardware-tuned (see
[Status & limitations](#status--limitations)). If you're building a similar
two-wheeler or reusing the control/sysid tooling, it should be a useful
starting point — issues and questions are welcome.

Licensed under [Apache 2.0](LICENSE).

## Repository layout

```
firmware/finn-mcu/   Teensy 4.1 firmware (PlatformIO) + moteus motor control,
                     including the on-robot system-ID sketches and capture tools
sim/
  model/finn/        Hand-authored MuJoCo model, meshes (STL), and scene
  config/            Canonical measurements + postprocess config (source of truth)
  generated/         Generated seed models (only the reference bundle is committed)
tools/               Host-side Python: build the seed model, run the LQR sim,
                     postprocess sysid, environment check
tests/               pytest suite for the host tools
config/              Motor (moteus) calibration logs
```

## Quick start (host / sim)

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Nothing else is
needed for the sim — MuJoCo and SciPy install as self-contained wheels.

```bash
uv sync --extra dev                     # create .venv from the locked deps
uv run python tools/check_env.py        # verify the toolchain imports
uv run python tools/run_lqr_sim.py      # run the LQR balance sim
```

`run_lqr_sim.py` loads the committed seed model at
`sim/generated/seeded/latest/finn.seeded.sim.xml`, linearizes the balance
dynamics, designs a discrete LQR gain, and runs a closed-loop MuJoCo rollout.
It writes a timeseries CSV, a JSON report, and a plot to `logs/lqr_sim/<ts>/`.
See `--help` for tuning knobs (Q/R weights, initial pitch, duration).

## From robot to model: the system-ID pipeline

The MuJoCo model is not guessed — its wheel friction, damping, torque limits,
and inertias come from measurements. The flow is:

1. **Bench + on-robot sysid.** Firmware sketches under
   `firmware/finn-mcu/sysid/` excite the wheels/IMU; capture tools log the
   telemetry.
2. **Postprocess** each batch into identified parameters
   (`tools/postprocess_mujoco.py`, `firmware/finn-mcu/tools/postprocess_*`).
3. **Build the seed model** from the latest clean runs:
   ```bash
   uv run python tools/build_seeded_mujoco_model.py --auto-select-latest 3
   ```
   This aggregates the runs, writes `seeded_measurements.yaml`, and emits the
   MuJoCo XML the sim consumes.

**Raw sysid telemetry is intentionally not committed** — it's large and
robot-specific (see `.gitignore`). The committed
`sim/generated/seeded/latest/` bundle is a working reference model plus its
provenance (`selected_runs.json`, `aggregation_report.md`). If you're building
your own robot, capture your own sysid batches and regenerate the model against
them; if you just want to run the sim, the committed bundle is enough.

## Firmware

The Teensy 4.1 firmware lives in `firmware/finn-mcu/` and builds with
PlatformIO (`pio run`). It talks to moteus motor controllers over CAN-FD. CI
compiles every `[env:*]` in `platformio.ini` and runs cppcheck, so the MCU code
is build-verified on every push.

## Status & limitations

- The first LQR is a **sim validation**, not a tuned hardware controller. Two
  things must be addressed before hardware balance tests: the model's center of
  mass sits behind the wheel axle (the controller should regulate to that trim
  pitch, not zero), and the seeded `±0.25 N·m` torque cap is a firmware sysid
  safety limit that gives a very small recovery envelope — hub-motor capability
  is much higher.
- Contact parameters (friction, solref, solimp) are provisional defaults, not
  identified. World-pose / slip accuracy and closed-loop sim-to-real transfer
  are explicitly **not** validated yet.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest -q             # host tests
```

CI runs lint, format-check, tests, and the firmware build on every push to
`main`/`dev` and on PRs.
