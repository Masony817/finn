---
name: finn-gap-review
description: Review Finn Scopik sim-to-real gap artifacts and turn deterministic per-phase bias, RMS, gain, correlation, and lag evidence into a concise physics-grounded diagnosis and next model-validation step. Use when asked to interpret scopik_gap.json, compare Finn gap runs or history, explain which MuJoCo parameter is suspect, or prepare a sim-real brief after Batch 2 or LQR telemetry.
---

# Finn Gap Review

Review the computed evidence; do not ask an LLM to re-estimate time-series features.

## Gather evidence

1. Resolve the requested run. If none is named, list candidate run directories and use the newest run containing `scopik_gap.json`; state the choice.
2. Read, in order:
   - `<run>/scopik_gap.json`
   - `<run>/events.log`
   - `config/viz/finn.yaml`
   - `config/viz/finn_gap_history.jsonl`, when present
   - `sim/generated/seeded/latest/selected_runs.json`
   - `sim/generated/seeded/latest/seeded_measurements.yaml`
   - `sim/generated/seeded/latest/validation.json`
3. Read `references/phase-physics.md` completely before mapping a pattern to a parameter.
4. Inspect telemetry rows around a cited phase only when the JSON evidence is ambiguous. Do not load an entire large CSV into the prompt.

If the gap JSON is missing or older than the telemetry or model, rerun without changing tracked history:

```bash
uv run scopik gap --profile config/viz/finn.yaml --run <run> \
  --out /tmp/finn-scopik-gap.json --save /tmp/finn-scopik-gap.rrd --no-history
```

Append normal history only when the user intends the run to become a recorded baseline.

## Validate evidence

- Confirm the run, model path, model hash when available, profile, and replay scope.
- Treat `bias` as `sim - real`.
- Treat `gain` and `lag_s` as unavailable when null.
- Use gain only with adequate phase activity and `abs(correlation) >= 0.6`.
- Positive lag means sim later than real. Trust lag primarily in PRBS or pulse excitation. Discount settle, stationary, coastdown, monotonic, and near-window-boundary estimates.
- Compare repeated phases, both directions, and both wheels. A one-off phase is weak evidence.
- Separate a deterministic Scopik finding from its physical interpretation. A finding is a screening result, not proof of one parameter.

## Preserve validation boundaries

- Batch 2 replay is open-loop and gantry-supported. It validates onboard sensor and actuator response, not world trajectory, absolute slip, endpoint error, or free-balancing transfer.
- `hold_upright` changes the physics. Do not interpret supported pitch response as free LQR balance behavior.
- Never claim a parameter is identified from a signal that does not excite it independently.
- Do not hand-edit `sim/generated/seeded/latest/finn.seeded.sim.xml`. Change a source measurement, derived sysid output, or an explicit candidate overlay, then rebuild and compare.
- Change one coupled parameter family per iteration unless the effects are independently observable.

## Produce the brief

Lead with a one-sentence verdict, then use exactly these sections:

1. `Good` — evidence that agrees, is repeatable, or clears a concern.
2. `Issues` — each item includes signal, phase family, numeric evidence, interpretation, and confidence (`high`, `medium`, or `low`).
3. `Next change` — one parameter family, its source file or generation path, expected directional effect, rebuild/rerun command, and a falsifiable success criterion.
4. `Boundary` — the strongest conclusion this run cannot support.

Prefer “inspect X next” over “tune X” when evidence is not identifiable. Never invent a numeric parameter adjustment from residual metrics alone.
