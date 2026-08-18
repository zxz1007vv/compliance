# ZGWSARM model contract

The training asset is `urdf/zgwsarm.urdf`. It differs from the supplied export
only by removing the invalid `world -> BASE_LINK` revolute joint named `fixed`;
`BASE_LINK` is therefore the floating root expected by Isaac Gym. Mesh aliases
keep both the URDF and the supplied MuJoCo files repository-local.

## Link mapping

| Role | URDF name | Environment config |
|---|---|---|
| Floating base | `BASE_LINK` | `asset.base_name` |
| Front-right foot/wheel | `FAR_FOOT_LINK` | `asset.foot_names[0]` |
| Front-left foot/wheel | `FBL_FOOT_LINK` | `asset.foot_names[1]` |
| Rear-right foot/wheel | `RAR_FOOT_LINK` | `asset.foot_names[2]` |
| Rear-left foot/wheel | `RBL_FOOT_LINK` | `asset.foot_names[3]` |
| End effector | `ROBOT_ARM_LINK7` | `asset.end_effector_name` |

`FAR/FBL/RAR/RBL` map to the policy's `FR/FL/RR/RL` gait order.

## Isaac Gym DOF and action order

| Action | URDF joint | Group | Lower / upper (rad) | Velocity | Effort |
|---:|---|---|---:|---:|---:|
| 0 | `FAR_ABAD_JOINT` | FR leg | -0.697 / 0.523 | 16.75 | 180 |
| 1 | `FAR_HIP_JOINT` | FR leg | -2.443 / 2.443 | 16.75 | 180 |
| 2 | `FAR_KNEE_JOINT` | FR leg | -2.801 / 2.801 | 16.75 | 180 |
| 3 | `FAR_FOOT_JOINT` | FR wheel | continuous export range | 110 | 28 |
| 4 | `FBL_ABAD_JOINT` | FL leg | -0.523 / 0.697 | 16.75 | 180 |
| 5 | `FBL_HIP_JOINT` | FL leg | -2.443 / 2.443 | 16.75 | 180 |
| 6 | `FBL_KNEE_JOINT` | FL leg | -2.801 / 2.801 | 16.75 | 180 |
| 7 | `FBL_FOOT_JOINT` | FL wheel | continuous export range | 110 | 28 |
| 8 | `RAR_ABAD_JOINT` | RR leg | -0.697 / 0.523 | 16.75 | 180 |
| 9 | `RAR_HIP_JOINT` | RR leg | -2.443 / 2.443 | 16.75 | 180 |
| 10 | `RAR_KNEE_JOINT` | RR leg | -2.801 / 2.801 | 16.75 | 180 |
| 11 | `RAR_FOOT_JOINT` | RR wheel | continuous export range | 110 | 28 |
| 12 | `RBL_ABAD_JOINT` | RL leg | -0.523 / 0.697 | 16.75 | 180 |
| 13 | `RBL_HIP_JOINT` | RL leg | -2.443 / 2.443 | 16.75 | 180 |
| 14 | `RBL_KNEE_JOINT` | RL leg | -2.801 / 2.801 | 16.75 | 180 |
| 15 | `RBL_FOOT_JOINT` | RL wheel | continuous export range | 110 | 28 |
| 16 | `ROBOT_ARM_JOINT1` | arm | -2.618 / 2.618 | 10 | 100 |
| 17 | `ROBOT_ARM_JOINT2` | arm | 0 / 3.14 | 10 | 100 |
| 18 | `ROBOT_ARM_JOINT3` | arm | -2.9671 / 0 | 10 | 100 |
| 19 | `ROBOT_ARM_JOINT4` | arm | -1.57 / 1.57 | 10 | 100 |
| 20 | `ROBOT_ARM_JOINT5` | arm | -1.57 / 1.57 | 10 | 100 |
| 21 | `ROBOT_ARM_JOINT6` | arm | -6.28 / 6.28 | 5 | 100 |

The default leg pose follows the supplied MuJoCo `home` keyframe. Arm joints 2
and 3 are moved away from their exported zero-limit boundaries to 0.8 and -1.5
rad for reset stability. Initial base height is 0.55 m and the nominal working
height is 0.54 m.

## Simulation and actuator contract

- Physics step: 0.002 s (500 Hz).
- Control decimation: 5 (100 Hz policy/control rate).
- ABAD/HIP/KNEE position-PD gains: 90/1, 120/1, 120/1.
- Wheel command: `torque = action * 0.25 * 60 - 0.2 * wheel_velocity`,
  clipped to the URDF effort limit of 28 N.m.
- Wheel angles are zeroed only in the policy position observation; physical
  wheel positions remain untouched and wheel velocities remain observable.
- Legs/wheels use effort drives while the arm retains the compliance task's
  position drive. Cylinder-to-capsule replacement and self-collision are off.
- Fixed joints are not collapsed because `ROBOT_ARM_LINK7` must remain a
  distinct force/position-control body.

The URDF and MuJoCo models both total 47.5524395 kg (within export precision),
including the 20.25 kg base and the explicitly modelled arm. No extra arm
payload mass is added by the task configuration.
