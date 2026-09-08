# FINN

**FINN** is a self-balancing, two-wheeled humanoid style robot. The current state is an open hardware/software
platform for balance control, system identification, and sim-to-real work. This
repo holds everything for the robot: firmware, a system-ID pipeline that
turns bench measurements into a physically-grounded MuJoCo model, and a MuJoCo
sim with a first LQR balance controller.

Status: early and under active solo development. Interfaces move fast and the
controllers are sim-validated, not yet hardware-tuned (see
[Status & limitations](#status--limitations)). If you're building a similar
two-wheeler or reusing the control/sysid tooling, it should be a useful
starting point - hopefully. Issues and questions are more than welcome.

Licensed under [Apache 2.0](LICENSE).

## Repository layout - under development and active changes

```
firmware/finn-mcu/   Teensy 4.1 firmware (PlatformIO) + MJBots Moteus motor control,
                     including system-ID, safety-gated LQR, and capture tools
sim/
  model/finn/        Hand-authored MuJoCo model, meshes (STL), and scene
  config/            Canonical measurements + postprocess config (source of truth)
  generated/         Generated seed models (only the reference bundle is committed)
src/finn/            Shared control, simulation, model, reporting, telemetry
tools/               Host-side CLIs: build the seed model, run the LQR sim,
                     postprocess sysid, environment check
tests/               pytest suite for the host tools
config/              Frame/sign contracts, Scopik profiles, and local motor config
```

## Quick start (host / sim)

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Nothing else is
needed for the sim — MuJoCo and SciPy install as self-contained wheels.

```bash
uv sync --locked --extra dev            # create .venv from the locked deps
uv run python tools/check_env.py        # verify the toolchain imports
uv run python tools/run_lqr_sim.py      # run the LQR balance sim
```

To watch the closed-loop controller balance Finn in real time on macOS, run
the same simulation through MuJoCo's `mjpython` launcher. The isolated
Homebrew-Python environment is intentional: `mjpython` requires a framework
Python, while uv's standalone Python does not expose the shared library it
needs.

```bash
uv run --isolated --python /opt/homebrew/bin/python3 \
  mjpython tools/run_lqr_sim.py --viewer --duration-s 30
```

On Linux, use `uv run python tools/run_lqr_sim.py --viewer --duration-s 30`.
Closing the viewer stops the rollout and still writes the partial validation
bundle to `logs/lqr_sim/<ts>/`.

Motor calibration utilities are optional because their Qt GUI dependencies are
large. Install them only on machines that talk to the hardware:

```bash
uv sync --extra dev --extra hardware
```

`run_lqr_sim.py` loads the committed seed model at
`sim/generated/seeded/latest/finn.seeded.sim.xml`, linearizes the balance
dynamics, designs a discrete LQR gain, and runs a closed-loop MuJoCo rollout.
It writes a timeseries CSV, a JSON report, and a plot to `logs/lqr_sim/<ts>/`.
By default it derives the stationary balance trim from the model's axle-to-COM
offset and uses a slower position outer loop to prevent accumulated wheel creep.
Override the trim with `--target-pitch-rad`, or disable position hold with
`--position-hold-kp-s 0`, when testing the inner balance loop in isolation.
See `--help` for tuning knobs (Q/R weights, initial pitch, duration).

## Sim-to-real gap profiling

[`packages/scopik`](packages/scopik) replays a recorded Finn run through the
seeded MuJoCo model, compares signals phase by phase, and writes deterministic
bias/RMS/gain/lag diagnostics alongside a Rerun dashboard:

```bash
uv run scopik gap --profile config/viz/finn.yaml \
  --run logs/finn-mcu/sysid/batch_2_pass/<timestamp>
```

Use the repo's `finn-gap-review` agent skill after a run when you want a
physics-grounded brief that connects those measurements to the next model
parameter or validation check.

## Real-robot LQR bring-up

The repo now has a deliberately gated Teensy controller and capture path for
the first unsupported-floor balance trials. Start with the motor-disabled frame
and sign check; the firmware will not arm until those physical observations are
recorded in `config/finn_conventions.yaml` and a passing sim run regenerates the
controller header.

See [Real-robot LQR bring-up](docs/real_lqr_bringup.md) for the exact commands,
two-person no-stand procedure, safety limits, and Scopik feedback loop.

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

- The LQR gain and trim are **sim validated, not hardware tuned**. The real
  controller starts behind physical sign checks, a host heartbeat, a three-second
  trial limit, and conservative fault limits. The seeded `±1.0 N·m` torque
  envelope recovers about 11 degrees of lean in sim and caps chassis acceleration
  near 3 m/s²; hub-motor capability is much higher, but the first trials
  intentionally do not use it.
- Contact parameters (friction, solref, solimp) are provisional defaults, not
  identified. World-pose / slip accuracy and closed-loop sim-to-real transfer
  are explicitly **not** validated yet.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest -q             # host tests; requires a C++ compiler
```

CI runs lint, format-check, tests, and the firmware build on every push to
`main`/`dev` and on PRs.
