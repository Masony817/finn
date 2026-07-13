# Finn MuJoCo Postprocess Report

- status: `ok`
- strict: `False`

## Inputs
- robot: `sim/model/finn/finn_robot.xml`
- scene: `sim/model/finn/scene.xml`
- config: `sim/config/mujoco_postprocess.yaml`
- measurements: `sim/generated/seeded/latest/seeded_measurements.yaml`
- robot_sha256: `5f6efd330880e4f90dd55f0cedcf454640507549607517dc8a8347013dfbdcfa`
- scene_sha256: `89402558d8c14cec25d80d707238bdd4679cb2d03deae2740c4df397a81d4c57`
- config_sha256: `9bee25e0d508273d1e2ba68235594842f72ff06dbf8bd16e912f3bdfbdfc83ce`
- measurements_sha256: `2fb41ccf879d78afed42ff43c41d85c61c60abbf4f993c41b893cf2694efec78`
- onshape_url: `https://cad.onshape.com/documents/ed1c90944582329b3a0c4b53/w/ef6c6e50f2f631eadb017a5a/e/35effa0e1acd53deffeb4bb9`

## Inspection
- root_body: `base_2`
- wheels: `{'left': {'body': 'left_wheel', 'joint': 'left_wheel', 'center_site': 'left_wheel_center'}, 'right': {'body': 'right_wheel', 'joint': 'right_wheel', 'center_site': 'right_wheel_center'}}`
- visual_mesh_geoms: `23`
- collision_mesh_geoms: `23`
- generated_position_actuators: `['left_wheel', 'right_wheel']`

## Missing Measurements
- none

## Measurements Used
- wheels.left.radius_m: `0.081` m (source: `measured`)
- wheels.left.width_m: `0.053` m (source: `measured`)
- wheels.left.torque_limit_nm: `0.25` N*m (source: `moteus`)
- wheels.left.gear_ratio: `1.0` ratio (source: `measured`)
- wheels.left.command_sign: `-1` sign (source: `moteus`)
- wheels.left.damping: `0.00410667` N*m*s/rad (source: `moteus`)
- wheels.left.armature: `0.0` kg*m^2 (source: `moteus`)
- wheels.left.frictionloss: `0.115429` N*m (source: `moteus`)
- wheels.left.mass_kg: `2.1` kg (source: `measured`)
- wheels.right.radius_m: `0.081` m (source: `measured`)
- wheels.right.width_m: `0.053` m (source: `measured`)
- wheels.right.torque_limit_nm: `0.25` N*m (source: `moteus`)
- wheels.right.gear_ratio: `1.0` ratio (source: `measured`)
- wheels.right.command_sign: `1` sign (source: `moteus`)
- wheels.right.damping: `0.00441399` N*m*s/rad (source: `moteus`)
- wheels.right.armature: `0.0` kg*m^2 (source: `moteus`)
- wheels.right.frictionloss: `0.126772` N*m (source: `moteus`)
- wheels.right.mass_kg: `2.1` kg (source: `measured`)
- contact.tire.friction: `[1.0, 0.02, 0.002]` slide torsional rolling (source: `estimated`)
- contact.tire.solref: `[0.02, 1.0]` timeconst dampratio (source: `estimated`)
- contact.tire.solimp: `[0.9, 0.95, 0.001, 0.5, 2.0]` dmin dmax width midpoint power (source: `estimated`)

## Changes
- root: `{'renamed_from': 'base_2', 'renamed_to': 'base_link', 'freejoint_added': True, 'freejoint_name': 'root_freejoint', 'initial_clearance_m': 0.003, 'lift_applied_m': 0.044000000000000004, 'wheel_world': {'left': {'center_z_before_lift_m': 0.04, 'bottom_z_before_lift_m': -0.041}, 'right': {'center_z_before_lift_m': 0.04, 'bottom_z_before_lift_m': -0.041}}}`
- removed_collision_geoms: `[{'name': None, 'mesh': 'motor_hub', 'material': 'motor_hub_material', 'region': 'wheel'}, {'name': None, 'mesh': 'motor_tire', 'material': 'motor_tire_material', 'region': 'wheel'}, {'name': None, 'mesh': 'motor_hub', 'material': 'motor_hub_material', 'region': 'wheel'}, {'name': None, 'mesh': 'motor_tire', 'material': 'motor_tire_material', 'region': 'wheel'}]`
- removed_actuators: `[{'tag': 'position', 'name': 'left_wheel', 'joint': 'left_wheel'}, {'tag': 'position', 'name': 'right_wheel', 'joint': 'right_wheel'}]`
- updated_joints: `[{'wheel': 'left', 'joint': 'left_wheel', 'damping': 0.00410667, 'armature': 0.0, 'frictionloss': 0.115429}, {'wheel': 'right', 'joint': 'right_wheel', 'damping': 0.00441399, 'armature': 0.0, 'frictionloss': 0.126772}]`
- updated_inertials: `[{'wheel': 'left', 'body': 'left_wheel', 'mass_kg': 2.1}, {'wheel': 'right', 'body': 'right_wheel', 'mass_kg': 2.1}]`
- added_actuators: `[{'name': 'motor_left_wheel', 'joint': 'left_wheel', 'gear': -1.0, 'ctrlrange': [-0.25, 0.25]}, {'name': 'motor_right_wheel', 'joint': 'right_wheel', 'gear': 1.0, 'ctrlrange': [-0.25, 0.25]}]`
- added_collision_geoms: `[{'name': 'left_tire_collision', 'body': 'left_wheel', 'type': 'cylinder', 'condim': 6, 'radius_m': 0.081, 'half_width_m': 0.0265}, {'name': 'right_tire_collision', 'body': 'right_wheel', 'type': 'cylinder', 'condim': 6, 'radius_m': 0.081, 'half_width_m': 0.0265}]`
- added_sensors: `[{'tag': 'gyro', 'name': 'imu_gyro', 'target': 'imu'}, {'tag': 'accelerometer', 'name': 'imu_accelerometer', 'target': 'imu'}, {'tag': 'framequat', 'name': 'imu_quat', 'target': None}, {'tag': 'framepos', 'name': 'base_pos', 'target': None}, {'tag': 'framequat', 'name': 'base_quat', 'target': None}, {'tag': 'jointpos', 'name': 'wheel_left_pos', 'target': 'left_wheel'}, {'tag': 'jointvel', 'name': 'wheel_left_vel', 'target': 'left_wheel'}, {'tag': 'jointpos', 'name': 'wheel_right_pos', 'target': 'right_wheel'}, {'tag': 'jointvel', 'name': 'wheel_right_vel', 'target': 'right_wheel'}]`
- scene: `{'compiler_meshdir': '../../../model/finn/assets', 'visual_merged': True, 'assets_merged': 3, 'worldbody_children_merged': 2}`

## Validation
- sensor_model: `ideal; no noise/filter settings were applied`
- mujoco: `{'status': 'ok', 'nbody': 4, 'njnt': 3, 'nu': 2, 'nsensor': 9, 'zero_control_steps': 25}`

## Errors
- none
