# Seeded MuJoCo Validation

- status: `ok`
- model_xml: `sim/generated/seeded/latest/finn.seeded.sim.xml`

## Mesh Assets
- status: `ok`

## Checks
- compile: `ok`
- actuators: `2`
- zero_control_settle: qvel_norm=0.05351447453519983
- left_impulse: qvel_norm=0.04576040087843951
- right_impulse: qvel_norm=0.045746626234539824
- equal_wheel_forward: qvel_norm=0.08167473477524184
- opposite_wheel_yaw: qvel_norm=0.08527562447436896

## Scope
- onboard-signal validation only; this does not validate world pose, absolute slip, endpoint error, or closed-loop LQR transfer

## Errors
- none
