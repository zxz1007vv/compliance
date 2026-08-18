"""Robot-specific configuration for ZGWSARM."""

import numpy as np


LEG_DOF_NAMES = [
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
]
WHEEL_DOF_NAMES = [
    "FAR_FOOT_JOINT",
    "FBL_FOOT_JOINT",
    "RAR_FOOT_JOINT",
    "RBL_FOOT_JOINT",
]
ARM_DOF_NAMES = [f"ROBOT_ARM_JOINT{i}" for i in range(1, 7)]
HIP_DOF_NAMES = [
    "FAR_ABAD_JOINT",
    "FBL_ABAD_JOINT",
    "RAR_ABAD_JOINT",
    "RBL_ABAD_JOINT",
]
FOOT_LINK_NAMES = [
    "FAR_FOOT_LINK",
    "FBL_FOOT_LINK",
    "RAR_FOOT_LINK",
    "RBL_FOOT_LINK",
]


def config_zgwsarm(cfg):
    """Apply ZGWSARM overrides on the shared compliance-task baseline."""
    cfg.robot.name = "zgwsarm"
    cfg.asset.file = (
        "{WBC_COMPLIANCE_ROOT_DIR}/resources/robots/zgwsarm/urdf/zgwsarm.urdf"
    )
    cfg.asset.collapse_fixed_joints = False
    cfg.asset.flip_visual_attachments = False
    cfg.asset.fix_base_link = False
    # Keep the wheel cylinders and collision filtering consistent with the
    # ZGWT robot.  Fixed links stay expanded because LINK7 is the compliance
    # task's force/position end-effector body.
    cfg.asset.replace_cylinder_with_capsule = False
    cfg.asset.self_collisions = 1
    # Legs and wheels use explicit effort control; the arm keeps the existing
    # compliance task's Isaac position drive.
    cfg.asset.default_dof_drive_mode = 0
    cfg.asset.foot_name = "_FOOT_LINK"
    cfg.asset.foot_names = list(FOOT_LINK_NAMES)
    cfg.asset.base_name = "BASE_LINK"
    cfg.asset.end_effector_name = "ROBOT_ARM_LINK7"
    cfg.asset.leg_dof_names = list(LEG_DOF_NAMES)
    cfg.asset.wheel_dof_names = list(WHEEL_DOF_NAMES)
    cfg.asset.arm_dof_names = list(ARM_DOF_NAMES)
    cfg.asset.hip_dof_names = list(HIP_DOF_NAMES)
    cfg.asset.zero_position_observation_dof_names = list(WHEEL_DOF_NAMES)
    cfg.asset.fixed_action_targets = {}
    cfg.asset.penalize_contacts_on = [
        "BASE_LINK",
        "ABAD_LINK",
        "HIP_LINK",
        "KNEE_LINK",
        "ROBOT_ARM_LINK2",
        "ROBOT_ARM_LINK3",
        "ROBOT_ARM_LINK4",
        "ROBOT_ARM_LINK5",
    ]
    cfg.asset.terminate_after_contacts_on = ["BASE_LINK"]

    cfg.init_state.pos = [0.0, 0.0, 0.55]
    cfg.init_state.default_joint_angles = {
        "FAR_ABAD_JOINT": 0.0,
        "FAR_HIP_JOINT": 0.6,
        "FAR_KNEE_JOINT": -1.2,
        "FAR_FOOT_JOINT": 0.0,
        "FBL_ABAD_JOINT": 0.0,
        "FBL_HIP_JOINT": 0.6,
        "FBL_KNEE_JOINT": -1.2,
        "FBL_FOOT_JOINT": 0.0,
        "RAR_ABAD_JOINT": 0.0,
        "RAR_HIP_JOINT": -0.6,
        "RAR_KNEE_JOINT": 1.2,
        "RAR_FOOT_JOINT": 0.0,
        "RBL_ABAD_JOINT": 0.0,
        "RBL_HIP_JOINT": -0.6,
        "RBL_KNEE_JOINT": 1.2,
        "RBL_FOOT_JOINT": 0.0,
        "ROBOT_ARM_JOINT1": 0.0,
        "ROBOT_ARM_JOINT2": 0.8,
        "ROBOT_ARM_JOINT3": -1.5,
        "ROBOT_ARM_JOINT4": 0.0,
        "ROBOT_ARM_JOINT5": 0.0,
        "ROBOT_ARM_JOINT6": 0.0,
    }

    cfg.env.num_actions = 22
    cfg.env.num_scalar_observations = 96
    cfg.env.num_observations = 96
    cfg.env.default_leg_dof_pos_RL = [
        0.0,
        0.6,
        -1.2,
        0.0,
        0.6,
        -1.2,
        0.0,
        -0.6,
        1.2,
        0.0,
        -0.6,
        1.2,
    ]

    cfg.control.control_type = "P"
    cfg.control.action_scale = 0.25
    cfg.control.hip_scale_reduction = 1.0
    cfg.control.decimation = 5
    cfg.sim.dt = 0.002
    # The shared delay buffer advances once per physics step.  Four buffered
    # steps at 2 ms reproduce the real controller's maximum 8 ms delay.
    cfg.domain_rand.lag_timesteps = 4

    cfg.control.stiffness = {"JOINT": 40.0}
    cfg.control.damping = {"JOINT": 2.0}

    cfg.commands.p_gains_legs = [90.0, 120.0, 120.0] * 4
    cfg.commands.d_gains_legs = [1.0, 1.0, 1.0] * 4
    # For wheels Kp is the action-to-torque gain, not a position stiffness.
    cfg.commands.p_gains_wheels = [60.0] * 4
    cfg.commands.d_gains_wheels = [0.2] * 4
    cfg.commands.p_gains_arm = [64.0, 128.0, 64.0, 64.0, 64.0, 64.0]
    cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5]
    cfg.commands.arm_mount_translation = [-0.195, 0.0, 0.1703]
    cfg.commands.arm_mount_yaw = np.pi
    cfg.commands.command_base_height = 0.54

    cfg.rewards.base_height_target = 0.54
    return cfg


__all__ = [
    "ARM_DOF_NAMES",
    "FOOT_LINK_NAMES",
    "HIP_DOF_NAMES",
    "LEG_DOF_NAMES",
    "WHEEL_DOF_NAMES",
    "config_zgwsarm",
]
