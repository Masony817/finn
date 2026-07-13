# Seeded MuJoCo Aggregation

This is a generated seed-model bundle. It does not mutate canonical measurements.

## Selected Runs
### batch1
- `logs/finn-mcu/sysid/batch_1_pass/20260615_150912` rows=13136 duration_s=136.076248
- `logs/finn-mcu/sysid/batch_1_pass/20260620_144630` rows=12941 duration_s=134.145014
- `logs/finn-mcu/sysid/batch_1_pass/20260621_131252` rows=14363 duration_s=145.519528
### batch2
- `logs/finn-mcu/sysid/batch_2_pass/20260626_140757` rows=14647 duration_s=378.44495
- `logs/finn-mcu/sysid/batch_2_pass/20260626_141541` rows=13709 duration_s=202.137841
- `logs/finn-mcu/sysid/batch_2_pass/20260626_142027` rows=14018 duration_s=208.080939

## Actuator Seeds
- left: sign=-1, limit=0.25, frictionloss=0.115429, damping=0.00410667, armature=0.0
- right: sign=1, limit=0.25, frictionloss=0.126772, damping=0.00441399, armature=0.0

## Contact Seeds
- friction, solref, and solimp are provisional defaults.
- Batch 2 lower bounds and loaded-loss diagnostics are preserved in JSON but not treated as identified contact parameters.

## Do Not Infer
- world_pose_accuracy
- absolute_slip
- contact_solref
- contact_solimp
- closed_loop_lqr_transfer
