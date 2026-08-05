# CLAUDE.md

Working guide for AI agents in the **Finn** repo. `AGENTS.md` is a symlink to this
file, so Claude Code, Codex, and any other harness read the same instructions.
Keep it that way: edit `CLAUDE.md`, never replace the symlink with a second copy.

Update this file with the `finn-docs-sync` skill whenever the repo changes shape
(see [Keeping this file current](#keeping-this-file-current)).

## What Finn is

Finn is a self-balancing, two-wheeled humanoid-style robot: 8.44 kg, Teensy 4.1
brain, two moteus-driven hub motors over CAN-FD, BNO085 IMU. This repo holds the
whole stack - firmware, a system-ID pipeline that turns bench measurements into a
physically grounded MuJoCo model, a MuJoCo sim with an LQR balance controller, and
a sim-to-real gap profiler.

Solo project, early, moving fast. The LQR controller is **sim validated, not
hardware tuned**; the robot has not balanced unsupported yet. That single fact
drives most of the rules below.

Read `README.md` for the user-facing story and `docs/real_lqr_bringup.md` before
touching anything on the hardware path.

## Repo map

```
firmware/finn-mcu/          Teensy 4.1 firmware, PlatformIO
  sysid/NN_<name>/          one bring-up or sysid sketch per directory
  control/04_lqr_balance/   the real-robot LQR controller + its generated/safety headers
  tools/                    host-side capture wrappers and per-batch postprocessors
sim/
  model/finn/               hand-authored MJCF (finn_robot.xml, scene.xml) + STL assets
  config/                   finn_measurements.yaml (source of truth), mujoco_postprocess.yaml
  generated/seeded/latest/  committed seed-model bundle the sim and firmware export load
tools/                      host Python: seed-model build, LQR sim, MuJoCo postprocess, env check
packages/scopik/            standalone sim-to-real gap profiler on Rerun (uv workspace member)
config/
  finn_conventions.yaml     sign/frame contract shared by model, Scopik, and firmware
  viz/                      Scopik profiles: finn.yaml (Batch 2), finn_lqr.yaml (LQR replay)
  moteus-cali/              motor calibration logs
tests/                      pytest for host tools and cross-artifact contracts
docs/                       operator procedures
.agents/skills/             canonical, harness-neutral agent skills
.claude/skills/             thin pointers into .agents/skills
logs/                       run artifacts, gitignored
```

## Commands

Python 3.11 + [uv](https://docs.astral.sh/uv/). Everything runs through `uv run`.

```bash
uv sync --extra dev                       # locked install; --extra hardware adds moteus tooling
uv run python tools/check_env.py          # toolchain smoke check
uv run ruff check . && uv run ruff format .
uv run pytest -q                          # host tests (root tests/ + packages/scopik/tests/)

uv run python tools/run_lqr_sim.py                       # closed-loop LQR rollout -> logs/lqr_sim/<ts>/
uv run python tools/build_seeded_mujoco_model.py --auto-select-latest 3
uv run scopik gap --profile config/viz/finn.yaml --run <run_dir>

cd firmware/finn-mcu && pio run -e <env>   # build one firmware environment
```

Live viewer on macOS needs a framework Python because `mjpython` links against
the shared library uv's standalone build does not expose:

```bash
uv run --isolated --python /opt/homebrew/bin/python3 \
  mjpython tools/run_lqr_sim.py --viewer --duration-s 30
```

On Linux, `uv run python tools/run_lqr_sim.py --viewer` is enough.

## The pipeline

Everything downstream is derived. Change a source, regenerate, never patch the
output:

```
bench measurements  ->  sim/config/finn_measurements.yaml
sysid firmware runs ->  logs/finn-mcu/sysid/batch_N_pass/<ts>/  (raw, gitignored)
  postprocess_sysid_batch{1,2}.py  ->  derived params per run
  build_seeded_mujoco_model.py     ->  sim/generated/seeded/latest/finn.seeded.sim.xml
  run_lqr_sim.py                   ->  LQR gain + logs/lqr_sim/<ts>/, and with
                                       --firmware-header, lqr_seeded_config.h
  firmware lqr_balance             ->  logs/finn-mcu/lqr/<run>/
  scopik gap                       ->  scopik_gap.json + .rrd + appended gap history
```

Batch 1 is off-ground actuator characterization (damping, friction, torque
limits, command signs). Batch 2 is gantry-supported ground contact; the gantry no
longer exists, so those runs are frozen provenance, not a repeatable experiment.
Short LQR captures are the next controlled dataset.

## Hard rules

1. **Never hand-edit generated artifacts.** That means
   `sim/generated/seeded/latest/*` and
   `firmware/finn-mcu/control/04_lqr_balance/lqr_seeded_config.h`. Change a source
   measurement, derived sysid output, or config, then regenerate. The generated
   header is written only by a passing `run_lqr_sim.py --firmware-header` run.
2. **`config/finn_conventions.yaml` is the sign contract**, shared by MuJoCo,
   Scopik, and firmware. If a sign looks wrong, fix it there and regenerate
   downstream; do not add a compensating negation at a call site.
3. **Safety gates are load-bearing, not friction.** The firmware refuses to arm
   until the two `*_bench_verified` flags in the conventions file are true and the
   header has been rebuilt. `lqr_safety_config.h` holds separately reviewed
   physical limits. Do not flip a flag, widen a limit, raise the 0.25 N.m torque
   cap, or lengthen `kFirstTrialDurationMs` on an agent's own initiative - those
   are the user's calls, made from evidence.
4. **Raw telemetry stays out of git.** `logs/`, `*.csv`, `*.rrd`, `*.png`, and
   most binary/CAD formats are gitignored. The committed seed bundle and small
   test fixtures are the deliberate exceptions; check `.gitignore` before adding
   a file type.
5. **Don't claim a parameter is identified from a signal that does not excite it
   independently.** Replay is an open-loop, onboard-signal comparison: it says
   nothing about world trajectory, slip, or free-balance transfer.
6. **One coupled parameter family per iteration**, unless the effects are
   independently observable.

## Area conventions

### Host Python (`tools/`, `firmware/finn-mcu/tools/`)

- Ruff, line length 100, `select = ["E","F","I","N","UP","B","SIM","RUF"]`.
  Per-file ignores go in `pyproject.toml` with a comment saying why.
- Scripts, not a package: `from __future__ import annotations`, module docstring
  first line explains the job, module-level `REPO_ROOT = Path(__file__).resolve().parents[N]`,
  constants in caps near the top, `argparse` in `parse_args`, `main()` returning an
  exit code.
- Expected failures raise a named error (`PostprocessError`,
  `BuildSeededModelError`) carrying a user-facing message, rather than a traceback.
- Generated reports use repo-relative paths via the local `portable_path` helper
  so artifacts are portable across machines.
- Comments explain *why* (physics, hardware quirks, a MuJoCo gotcha). The existing
  density is high and intentional; match it.

### scopik (`packages/scopik/`)

A real installed package (`src/` layout, console script `scopik`), deliberately
robot-agnostic and deliberately narrow: recorded, open-loop sim-to-real gap
analysis. Adapting it to a robot means writing a YAML profile, not editing the
package. All Rerun API usage is confined to `src/scopik/rrlog/`; keep it there so
SDK churn stays local. Rerun is pinned `>=0.34,<0.35`.

Before adding a feature, check that a robot milestone actually needs it - plugin
registries, live streaming, and 3D reconstruction were scoped out on purpose.
Read `packages/scopik/README.md` first; it is the profile reference.

### Firmware (`firmware/finn-mcu/`)

- One directory per sketch, numbered by bring-up order, with a matching
  `[env:*]` in `platformio.ini` using `build_src_filter` to select it. Adding a
  sketch needs no CI change: the workflow discovers every environment.
- Google-ish C++ style as written: anonymous namespace for internals, `k`-prefixed
  `constexpr` constants with explicit units in the name, `static_assert` for
  invariants that must hold between the generated header and the code.
- Telemetry is line-prefixed CSV on serial (`schema,` / `data,` rows, `event,` /
  `status,` lines). Scopik profiles parse it directly, so a column rename is a
  contract change: update the profile and the postprocessor in the same commit.
- Agents build and static-check firmware; they do not flash or run it. Flashing,
  arming, and any hands-on check belong to the user at the robot.

### Configs

YAML with `schema_version` at the top. Measurement entries carry
`value` / `unit` / `source` / `notes`, where `source` is one of `measured`, `cad`,
`moteus`, `estimated`, `todo`. Strict mode rejects `estimated` and `todo`, which
is how placeholders are kept out of the generated model. Never upgrade a `source`
label to make a build pass.

## Testing and CI

CI runs two independent gates on push to `main`/`dev` and on PRs:
`python` (ruff check, ruff format --check, pytest) and `firmware` (`pio run` plus
`pio check --severity=high` for every environment). Both auto-discover new work.

Test conventions:

- `tools/` scripts are loaded with `importlib.util.spec_from_file_location`, since
  they are not importable modules. scopik is imported normally.
- `tests/test_finn_lqr_contract.py` asserts that the conventions file, seeded
  model, generated header, and firmware still agree. If it fails, something in the
  chain was regenerated without its dependents - fix the chain, not the test.
- `tests/test_run_lqr_sim_smoke.py` guards the fresh-clone path. Keep it fast.
- Tests must pass without hardware, without network, and without raw telemetry.
  Build fixtures in `tmp_path`.

## Hardware safety

An agent's reach ends at the USB cable. Do not flash firmware, energize motors,
run a capture script, or tell the user a physical trial is safe. The bring-up doc
is explicit: an unsupported first release needs two people, a clear floor, and an
accessible hardware power cutoff. When work reaches that boundary, hand back a
checklist and the exact commands, and stop.

## Agent skills

Canonical skills live in `.agents/skills/<name>/SKILL.md`, harness-neutral.
`.claude/skills/<name>/SKILL.md` is a thin pointer that redirects to the canonical
file, and `agents/openai.yaml` inside a skill carries Codex-specific presentation.
Add new skills the same way so every harness gets the same behavior.

| Skill | Use it for |
|---|---|
| `finn-gap-review` | Interpreting `scopik_gap.json`, gap history, or an LQR/Batch 2 replay into a physics-grounded diagnosis and one next model change. |
| `finn-docs-sync` | Updating this file after the repo changes. |

## Keeping this file current

This guide describes structure, contracts, and rules - things that stay true for
months. It is not a changelog, a task list, or a place for current findings, and
it should not restate `README.md` or `docs/`. When the repo grows a new package,
pipeline stage, generated artifact, or safety gate, invoke `finn-docs-sync` rather
than appending prose by hand, so the file stays the same size and shape as it
grows.
