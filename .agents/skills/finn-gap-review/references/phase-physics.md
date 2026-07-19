# Finn phase-to-physics map

Use this map after validating Scopik's numeric evidence. Parameter suspects are ordered, not exclusive.

## Evidence semantics

- Bias: mean `sim - real`. Check sign consistency across positive/negative commands before calling it a constant offset.
- RMS: magnitude only. Concentration by phase is more useful than whole-run rank.
- Gain: scaling mismatch after de-meaning. Negative gain usually means sign/frame error or poor excitation, not a negative physical constant.
- Lag: timing clue, not an actuator-delay measurement unless an excitation phase has enough bandwidth and repeatability.
- Left/right symmetry: common error points toward shared motor/contact/model terms; opposite error points toward sign, asymmetry, or yaw geometry.

## Phase families

| Phase family | What it excites | First suspects | Do not infer |
|---|---|---|---|
| `initial_stationary_noise`, `final_stationary_noise`, `idle` | zero-command hold and sensor floor | wheel `frictionloss`, contact stiction, command bias, sensor zero | inertia or delay |
| `creep_*` | loaded breakaway at low torque | loaded friction/stiction, torque calibration, left/right asymmetry | viscous damping from a single plateau |
| `straight_*` | common-mode loaded drive | torque scale, wheel radius, reflected inertia/armature, rolling resistance | track width from common motion |
| `yaw_*` | differential drive and yaw response | effective track width, yaw inertia, command signs, asymmetric friction | forward trajectory fidelity |
| `coast_spinup_*` | driven approach to coast initial state | torque scale, inertia, damping | pure coast loss until command is zero |
| `coastdown_*` | unpowered decay | Coulomb/rolling loss, viscous damping, wheel/system inertia | actuator delay |
| `*_settle` | return to rest | stiction floor, damping, residual command or support interaction | trustworthy gain/lag when real activity is small |
| `prbs_straight` | broadband common-mode wheel dynamics | actuator delay, torque scale, armature/inertia, damping | yaw geometry |
| `prbs_differential` | broadband differential/yaw dynamics | actuator delay, track width, yaw inertia, asymmetric friction | absolute world pose |

## Signal-specific checks

### Wheel velocity

- Real near zero while sim moves during stationary or settle phases strongly supports insufficient simulated loss under load.
- Similar left/right gain error supports a shared torque-scale, radius, inertia, or damping mismatch.
- Opposite signs or strong left/right disagreement requires checking command and sensor conventions before tuning physics.
- Coastdown error that changes with speed supports damping; a roughly speed-independent stopping threshold supports Coulomb/rolling loss.

### Pitch and yaw rate

- With `hold_upright`, pitch-rate agreement is only a supported onboard-rate check.
- Yaw-rate mismatch concentrated in differential phases supports track-width/yaw-inertia review.
- Straight-phase yaw rate points first to left/right asymmetry or support interaction.

### Forward acceleration

- Check IMU axis/sign and gravity compensation before physical tuning.
- Compare positive and negative straight phases. A sign-dependent discrepancy suggests frames, command convention, or asymmetric contact; a symmetric scale error suggests mass/torque/radius.
- Large coastdown acceleration error may reflect contact loss modeling, but does not by itself identify a unique friction coefficient.

## Parameter source map

- Physical measurements and CAD-reviewed values: `sim/config/finn_measurements.yaml`.
- Batch-derived actuator and loaded-loss seeds: each selected run's `derived.yaml`, aggregated by `tools/build_seeded_mujoco_model.py`.
- MuJoCo translation rules and contact defaults: `sim/config/mujoco_postprocess.yaml` and `tools/postprocess_mujoco.py`.
- Generated evidence: `sim/generated/seeded/latest/seeded_measurements.yaml`, `aggregation.json`, and `validation.json`.
- Generated MJCF: inspect it, but do not edit it as the source of truth.

When testing a hypothesis, prefer a standalone candidate measurement YAML or explicit build input over mutating a generated bundle in place. Reprocess the same old runs after a pipeline change before comparing new captures.
