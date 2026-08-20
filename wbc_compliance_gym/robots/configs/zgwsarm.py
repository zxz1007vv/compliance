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
ABAD_DOF_NAMES = [
    "FAR_ABAD_JOINT",
    "FBL_ABAD_JOINT",
    "RAR_ABAD_JOINT",
    "RBL_ABAD_JOINT",
]
# Backward-compatible export for code that still calls the lateral hip joint
# group "hip".  ZGWSARM control uses the explicit ABAD name below.
HIP_DOF_NAMES = ABAD_DOF_NAMES
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
    cfg.asset.abad_dof_names = list(ABAD_DOF_NAMES)
    cfg.asset.hip_dof_names = list(HIP_DOF_NAMES)
    cfg.asset.wheel_radius = 0.095
    cfg.asset.zero_position_observation_dof_names = list(WHEEL_DOF_NAMES)
    cfg.asset.fixed_action_targets = {}
    cfg.asset.penalize_contacts_on = [
        "BASE_LINK",
        "ABAD_LINK",
        "HIP_LINK",
        "KNEE_LINK",
        "base_connector_link",
        "ROBOT_ARM_LINK0",
        "ROBOT_ARM_LINK1",
        "ROBOT_ARM_LINK2",
        "ROBOT_ARM_LINK3",
        "ROBOT_ARM_LINK4",
        "ROBOT_ARM_LINK5",
        "ROBOT_ARM_LINK6",
        "ROBOT_ARM_LINK7",
    ]
    # Every collision shape except the four wheel/foot links is a failure
    # contact for this task.  End-effector forces are applied explicitly and
    # therefore do not require the arm to touch the terrain.
    cfg.asset.terminate_after_contacts_on = [
        "BASE_LINK",
        "ABAD_LINK",
        "HIP_LINK",
        "KNEE_LINK",
        "base_connector_link",
        "ROBOT_ARM_LINK",
    ]

    cfg.init_state.pos = [0.0, 0.0, 0.55]
    # Initial arm-pose perturbation used by the shared reset path. The former
    # hard-coded +/-0.5 rad range produced a large recovery motion at every
    # reset; keep a small perturbation for training robustness instead.
    cfg.init_state.arm_reset_position_range = [-0.10, 0.10]
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
    # ABAD has the narrowest asymmetric leg limits.  With clip_actions=4 this
    # gives at most +/-0.4 rad around the default pose, instead of +/-1 rad.
    cfg.control.abad_scale_reduction = 0.4
    # Kept only for compatibility with the shared controller configuration.
    cfg.control.hip_scale_reduction = 1.0
    # ZGWSARM's arm already receives the global 0.25 rad action scale.  The
    # inherited B1+Z1 value of 2.0 enlarged arm requests and drove the policy
    # into hard-limit/slew-rate saturation.
    cfg.control.arm_scale_reduction = 1.0
    # Position-drive targets are updated every 2 ms physics step and may not
    # advance faster than the corresponding URDF velocity limit.
    cfg.control.arm_target_velocity_limit_scale = 1.0
    # Reset isolated pathological states before sending them through another
    # PhysX articulation solve. The margin avoids reacting to solver tolerance.
    cfg.control.safety_dof_velocity_ratio = 2.0
    cfg.control.safety_dof_position_margin = 0.05
    # Foot impacts are excluded; large contacts on the base, legs, or arm are
    # treated as trapped-articulation states and reset before the next solve.
    cfg.control.safety_nonfoot_contact_force = 5000.0
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
    # Forward kinematics of the configured [0, 0.8, -1.5, 0, 0, 0] arm pose,
    # expressed in the command arm frame.  Diagnostic locomotion scenarios use
    # this instead of asking a fixed arm to track the unrelated 0.4 m target.
    cfg.commands.default_ee_position_spherical = [
        0.7714136743,
        -1.3098718840,
        0.0,
    ]
    # LINK7 has a fixed orientation offset at the same nominal pose.  For
    # ZGWSARM, a zero RPY command means this calibrated tool orientation.
    cfg.commands.ee_nominal_orientation_rpy = [
        0.0,
        -0.8707963268,
        0.0,
    ]
    return cfg


__all__ = [
    "ABAD_DOF_NAMES",
    "ARM_DOF_NAMES",
    "FOOT_LINK_NAMES",
    "HIP_DOF_NAMES",
    "LEG_DOF_NAMES",
    "WHEEL_DOF_NAMES",
    "config_zgwsarm",
]
