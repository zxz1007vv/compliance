"""Independent training and play configuration for ZGWSARM compliance."""

import numpy as np

from wbc_compliance_gym.envs.base.compliance_task_config import (
    apply_reward_scales,
    build_compliance_ppo_config,
    new_compliance_env_config,
    validate_active_reward_scales,
)
from wbc_compliance_gym.robots.configs.zgwsarm import config_zgwsarm
from wbc_compliance_gym.utils.config_utils import ConfigNode


ZGWSARM_REWARD_SCALES = {
    "action_magnitude": -0.05,
    "action_smoothness_1_arm": -0.01,
    "action_smoothness_1_leg": -0.03,
    "action_smoothness_2_arm": -0.01,
    "ang_vel_xy": -0.05,
    "base_height": -30.0,
    "collision": -5.0,
    "dof_acc_arm": -3e-8,
    "dof_acc_leg": -1.5e-6,
    "dof_pos_limits_arm": -10.0,
    "dof_pos_limits_leg": -1.0,
    "dof_vel_arm": -0.003,
    "dof_vel_leg": -0.0008,
    "ee_force_x": 3.0,
    "ee_force_y": 3.0,
    "ee_force_z": 3.0,
    "feet_contact_forces": -0.1,
    "lin_vel_z": -4.0,
    "manip_ori_tracking": 2.5,
    "manip_pos_tracking": 3.0,
    "orientation": -5.0,
    "survival": 1.0,
    "termination": -10.0,
    "torque_limits_arm": -0.005,
    "torque_limits_leg": -0.005,
    "torques": -8e-5,
    "torques_arm": -1e-5,
    "tracking_ang_vel_yaw": 2.0,
    "tracking_lin_vel": 1.0,
}


def _configure_zgwsarm_environment(cfg):
    cfg.env.num_envs = 4000
    cfg.env.num_observation_history = 10
    cfg.env.num_privileged_obs = 16
    cfg.env.observe_vel = False
    cfg.env.priv_passthrough = False
    cfg.env.recording_width_px = 180
    cfg.env.recording_height_px = 120


def _configure_zgwsarm_observations(cfg):
    cfg.sensors.sensor_names = [
        "OrientationSensor",
        "RCSensor",
        "JointPositionSensor",
        "JointVelocitySensor",
        "ActionSensor",
        "ClockSensor",
    ]
    cfg.sensors.sensor_args = {name: {} for name in cfg.sensors.sensor_names}
    cfg.sensors.privileged_sensor_names = [
        "BodyVelocitySensor",
        "JointDynamicsSensor",
        "EeGripperForceSensor",
        "FrictionSensor",
        "EeGripperPositionSensor",
        "EeGripperTargetPositionSensor",
    ]
    cfg.sensors.privileged_sensor_args = {
        name: {} for name in cfg.sensors.privileged_sensor_names
    }

    cfg.obs_scales.ee_sphe_radius_cmd = 0.5
    cfg.obs_scales.ee_sphe_pitch_cmd = 1.0
    cfg.obs_scales.ee_sphe_yaw_cmd = 1.3
    cfg.obs_scales.ee_timing_cmd = 0.1
    cfg.obs_scales.ee_force_magnitude = 0.01
    cfg.obs_scales.ee_force_direction_angle = 0.3
    cfg.obs_scales.ee_force_z = 0.01


def _configure_zgwsarm_commands(cfg):
    cfg.commands.num_commands = 23
    cfg.commands.resampling_time = 10
    cfg.commands.command_curriculum = True
    cfg.commands.distributional_commands = True
    cfg.commands.num_lin_vel_bins = 30
    cfg.commands.num_ang_vel_bins = 30

    cfg.commands.lin_vel_x = [-1.0, 1.0]
    cfg.commands.limit_vel_x = [-1.0, 1.0]
    cfg.commands.lin_vel_y = [0.0, 0.0]
    cfg.commands.limit_vel_y = [0.0, 0.0]
    cfg.commands.ang_vel_yaw = [-1.5, 1.5]
    cfg.commands.limit_vel_yaw = [-1.5, 1.5]
    cfg.commands.heading_command = False

    cfg.commands.body_height_cmd = [0.0, 0.0]
    cfg.commands.limit_body_height = [0.0, 0.0]
    cfg.commands.command_base_height = 0.54
    cfg.commands.body_pitch_range = [0.0, 0.0]
    cfg.commands.limit_body_pitch = [0.0, 0.0]
    cfg.commands.body_roll_range = [0.0, 0.0]
    cfg.commands.limit_body_roll = [0.0, 0.0]

    # Retain the legacy 23-dimensional command contract in phase one.  These
    # fixed B1 gait slots are interface compatibility fields only; no active
    # ZGWSARM reward consumes their contact schedule.
    cfg.commands.gait_frequency_cmd_range = [2.2, 2.2]
    cfg.commands.limit_gait_frequency = [2.2, 2.2]
    cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    cfg.commands.limit_gait_phase = [0.5, 0.5]
    cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    cfg.commands.limit_gait_offset = [0.0, 0.0]
    cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    cfg.commands.limit_gait_bound = [0.0, 0.0]
    cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    cfg.commands.limit_gait_duration = [0.5, 0.5]
    cfg.commands.footswing_height_range = [0.2, 0.2]
    cfg.commands.limit_footswing_height = [0.2, 0.2]
    cfg.commands.stance_width_range = [0.6, 0.6]
    cfg.commands.limit_stance_width = [0.6, 0.6]
    cfg.commands.stance_length_range = [0.65, 0.65]
    cfg.commands.limit_stance_length = [0.65, 0.65]
    cfg.commands.aux_reward_coef_range = [0.0, 0.0]
    cfg.commands.limit_aux_reward_coef = [0.0, 0.0]

    for name in (
        "vel_x",
        "vel_y",
        "vel_yaw",
        "body_height",
        "gait_frequency",
        "gait_phase",
        "gait_offset",
        "gait_bound",
        "gait_duration",
        "footswing_height",
        "body_roll",
        "body_pitch",
        "stance_width",
        "stance_length",
        "aux_reward_coef",
    ):
        setattr(cfg.commands, f"num_bins_{name}", 1)
    cfg.commands.exclusive_phase_offset = False
    cfg.commands.pacing_offset = False
    cfg.commands.binary_phases = False
    cfg.commands.gaitwise_curricula = False
    cfg.commands.balance_gait_distribution = False

    cfg.commands.hybrid_mode = "binary"
    cfg.commands.force_control = False
    cfg.commands.control_only_z1 = False
    cfg.commands.interpolate_ee_cmds = True
    cfg.commands.sample_feasible_commands = False
    cfg.commands.teleop_occulus = False
    # This legacy switch selects the shared 23-D compliance command generator;
    # it does not make ZGWSARM inherit the B1 robot or reward configuration.
    cfg.commands.inverse_IK_door_opening = True

    cfg.commands.ee_force_z = [-70.0, 70.0]
    cfg.commands.limit_ee_force_z = [-70.0, 70.0]
    cfg.commands.ee_force_magnitude = [-70.0, 70.0]
    cfg.commands.limit_ee_force_magnitude = [-70.0, 70.0]
    cfg.commands.ee_sphe_radius = [0.25, 0.65]
    cfg.commands.limit_ee_sphe_radius = [0.25, 0.65]
    cfg.commands.ee_sphe_pitch = [-2 * np.pi / 5, 2 * np.pi / 5]
    cfg.commands.limit_ee_sphe_pitch = [-2 * np.pi / 5, 2 * np.pi / 5]
    cfg.commands.ee_sphe_yaw = [-3 * np.pi / 5, 3 * np.pi / 5]
    cfg.commands.limit_ee_sphe_yaw = [-3 * np.pi / 5, 3 * np.pi / 5]
    cfg.commands.ee_timing = [1.0, 4.0]
    cfg.commands.limit_ee_timing = [1.0, 4.0]
    cfg.commands.settle_time = 2.0
    cfg.commands.end_effector_roll = [0.0, 0.0]
    cfg.commands.limit_end_effector_roll = [0.0, 0.0]
    cfg.commands.end_effector_pitch = [0.0, 0.0]
    cfg.commands.limit_end_effector_pitch = [0.0, 0.0]
    cfg.commands.end_effector_yaw = [0.0, 0.0]
    cfg.commands.limit_end_effector_yaw = [0.0, 0.0]


def _configure_zgwsarm_rewards(cfg):
    cfg.rewards.reward_container_name = "ZGWSARMRewards"
    cfg.rewards.base_height_target = 0.54
    cfg.rewards.only_positive_rewards = True
    cfg.rewards.only_positive_rewards_ji22_style = False
    cfg.rewards.sigma_rew_neg = 0.02
    cfg.rewards.total_rew_scale = 0.2

    cfg.rewards.use_terminal_foot_height = False
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.terminal_body_height = 0.38
    cfg.rewards.use_terminal_roll_pitch = True
    cfg.rewards.terminal_body_ori = 0.5
    cfg.rewards.use_terminal_torque_legs_limits = False
    cfg.rewards.use_terminal_torque_arm_limits = False
    cfg.rewards.termination_torque_min_time = 25
    # Semantic failure contacts use a low, debounced threshold; the 5000 N
    # control threshold remains an independent solver protection.
    cfg.rewards.terminal_contact_force = 10.0
    cfg.rewards.terminal_contact_debounce_steps = 2

    cfg.rewards.soft_dof_pos_limit = 0.9
    cfg.rewards.soft_action_limit = 3.0
    cfg.rewards.stand_still_command_threshold = 0.10
    cfg.rewards.soft_torque_limit_leg = 1.0
    cfg.rewards.soft_torque_limit_arm = 1.0
    cfg.rewards.max_contact_force = 550.0
    cfg.rewards.gait_force_sigma = 30000
    cfg.rewards.gait_vel_sigma = 10.0
    cfg.rewards.footswing_height = 0.1
    cfg.rewards.swing_ratio = 0.3
    cfg.rewards.stance_ratio = 0.3
    cfg.rewards.stance_width = 0.45
    cfg.rewards.stance_length = 0.65
    cfg.rewards.sigma_force_magnitude = 1 / 50
    cfg.rewards.sigma_force_z = 1 / 50
    cfg.rewards.maintain_ori_force_envs = True

    apply_reward_scales(cfg, ZGWSARM_REWARD_SCALES)


def _configure_zgwsarm_domain_randomization(cfg):
    cfg.domain_rand.rand_interval_s = 4
    cfg.domain_rand.lag_timesteps = 4
    cfg.domain_rand.randomize_lag_timesteps = True
    cfg.domain_rand.randomize_rigids_after_start = False

    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.friction_range = [0.6, 1.5]
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.restitution = 0.5
    cfg.domain_rand.restitution_range = [0.0, 0.4]
    cfg.domain_rand.randomize_ground_friction = True
    cfg.domain_rand.ground_friction_range = [0.6, 1.2]
    cfg.domain_rand.randomize_ground_restitution = False

    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    cfg.domain_rand.randomize_com_displacement = True
    cfg.domain_rand.com_displacement_range = [-0.05, 0.05]
    cfg.domain_rand.randomize_gravity = True
    cfg.domain_rand.gravity_range = [-0.5, 0.5]
    cfg.domain_rand.gravity_rand_interval_s = 8.0
    cfg.domain_rand.gravity_impulse_duration = 0.99

    cfg.domain_rand.randomize_motor_strength = True
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_Kd_factor = True

    cfg.domain_rand.randomize_tile_roughness = True
    cfg.domain_rand.tile_roughness_range = [0.0, 0.08]
    cfg.domain_rand.push_robots = True
    cfg.domain_rand.max_push_vel_xy = 0.8

    cfg.domain_rand.randomize_gripper_force_gains = True
    cfg.domain_rand.gripper_forced_prob = 0.8
    cfg.domain_rand.gripper_force_kp_range = [25.0, 400.0]
    cfg.domain_rand.gripper_force_kd_range = [3.0, 10.0]
    cfg.domain_rand.prop_kd = 0.1
    cfg.domain_rand.max_push_force_xyz_gripper = [-70.0, 70.0]
    cfg.domain_rand.max_push_force_xyz_gripper_freed = [-70.0, 70.0]
    cfg.domain_rand.push_gripper_stators = False
    cfg.domain_rand.push_gripper_interval_s = [3.5, 9.0]
    cfg.domain_rand.push_gripper_duration_s = [1.0, 3.0]
    cfg.domain_rand.max_push_vel_xyz_gripper = [-40.0, 40.0]

    cfg.domain_rand.push_robot_base = False
    cfg.domain_rand.push_robot_interval_s = 5.0
    cfg.domain_rand.push_robot_duration_s = [1.0, 2.0]
    cfg.domain_rand.max_push_vel_xyz_robot = [-40.0, 40.0]


def _configure_zgwsarm_terrain(cfg):
    cfg.terrain.border_size = 0.0
    cfg.terrain.mesh_type = "boxes_tm"
    cfg.terrain.num_cols = 20
    cfg.terrain.num_rows = 20
    cfg.terrain.terrain_width = 5.0
    cfg.terrain.terrain_length = 5.0
    cfg.terrain.x_init_range = 1.0
    cfg.terrain.y_init_range = 1.0
    cfg.terrain.yaw_init_range = 3.14
    cfg.terrain.teleport_thresh = 0.3
    cfg.terrain.teleport_robots = False
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 4
    cfg.terrain.horizontal_scale = 0.10
    cfg.terrain.curriculum = False
    cfg.terrain.terrain_noise_magnitude = 0.0
    cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]


def _configure_zgwsarm_simulation(cfg):
    cfg.sim.physx.max_gpu_contact_pairs = 2 ** 23
    cfg.sim.physx.default_buffer_size_multiplier = 3
    cfg.normalization.clip_actions = 4.0
    cfg.normalization.friction_range = [0, 1]
    cfg.normalization.ground_friction_range = [0, 1]


def _validate_zgwsarm_config(cfg):
    validate_active_reward_scales(cfg, ZGWSARM_REWARD_SCALES)

    dof_groups = (
        list(cfg.asset.leg_dof_names)
        + list(cfg.asset.wheel_dof_names)
        + list(cfg.asset.arm_dof_names)
    )
    if len(dof_groups) != len(set(dof_groups)):
        raise ValueError("ZGWSARM leg, wheel, and arm DOF groups must be disjoint")
    if cfg.env.num_actions != len(dof_groups):
        raise ValueError(
            f"ZGWSARM num_actions={cfg.env.num_actions} but has "
            f"{len(dof_groups)} controlled DOFs"
        )
    expected_observations = (
        3 + cfg.commands.num_commands + 3 * cfg.env.num_actions + 4
    )
    if cfg.env.num_observations != expected_observations:
        raise ValueError(
            f"ZGWSARM num_observations={cfg.env.num_observations}, "
            f"expected {expected_observations} from the sensor contract"
        )
    if cfg.commands.hybrid_mode not in {"position", "force", "binary", "mixed"}:
        raise ValueError(f"invalid ZGWSARM hybrid mode {cfg.commands.hybrid_mode!r}")
    for name in (
        "ee_sphe_radius",
        "ee_sphe_pitch",
        "ee_sphe_yaw",
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_yaw",
    ):
        value_range = getattr(cfg.commands, name)
        if len(value_range) != 2 or value_range[0] > value_range[1]:
            raise ValueError(f"invalid ZGWSARM command range {name}={value_range!r}")
    return cfg


def configure_zgwsarm_compliance(cfg=None):
    """Build ZGWSARM directly from the neutral compliance environment config."""
    cfg = new_compliance_env_config(cfg)
    config_zgwsarm(cfg)

    _configure_zgwsarm_environment(cfg)
    _configure_zgwsarm_observations(cfg)
    _configure_zgwsarm_commands(cfg)
    _configure_zgwsarm_rewards(cfg)
    _configure_zgwsarm_domain_randomization(cfg)
    _configure_zgwsarm_terrain(cfg)
    _configure_zgwsarm_simulation(cfg)
    _validate_zgwsarm_config(cfg)

    return cfg


def configure_zgwsarm_compliance_play(
    cfg,
    *,
    num_envs=1,
    control_mode= "position",
    seed=1,
    force_amplitude=None,
    fix_base=False,
    teleop=False,
    interpolate_ee_cmds=True,
    sample_feasible_commands=False,
    control_only_z1=False,
):
    """Build a play configuration for ZGWSARM compliance."""
    cfg.env.num_recording_envs = 1
    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = 10000
    cfg.terrain.num_rows = 10
    cfg.terrain.num_cols = 10
    cfg.terrain.border_size = 0
    cfg.terrain.num_border_boxes = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.terrain.teleport_robots = False
    cfg.terrain.mesh_type = "plane"    #plane 需要 teleport_robots = False

    if control_mode is not None:
        cfg.commands.hybrid_mode = control_mode
    if seed is not None:
        cfg.commands.curriculum_seed = seed
    if force_amplitude is not None:
        cfg.domain_rand.max_push_force_xyz_gripper = [
            -float(force_amplitude),
            float(force_amplitude),
        ]

    cfg.commands.lin_vel_x = [0.0, 0.0]
    cfg.commands.limit_vel_x = [0.0, 0.0]
    cfg.commands.lin_vel_y = [0.0, 0.0]
    cfg.commands.limit_vel_y = [0.0, 0.0]
    cfg.commands.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.limit_vel_yaw = [0.0, 0.0]

    cfg.commands.ee_sphe_radius = [0.40, 0.40]
    cfg.commands.limit_ee_sphe_radius = [0.40, 0.40]
    cfg.commands.ee_sphe_pitch = [0.0, 0.0]
    cfg.commands.limit_ee_sphe_pitch = [0.0, 0.0]
    cfg.commands.ee_sphe_yaw = [0.0, 0.0]
    cfg.commands.limit_ee_sphe_yaw = [0.0, 0.0]

    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_tile_roughness = False
    cfg.asset.fix_base_link = fix_base
    cfg.commands.teleop_occulus = teleop
    cfg.commands.interpolate_ee_cmds = interpolate_ee_cmds
    cfg.commands.sample_feasible_commands = sample_feasible_commands
    cfg.commands.control_only_z1 = control_only_z1

    cfg.env.recording_height_px = 720
    cfg.env.recording_width_px = 1280
    cfg.env.record_video = True
    cfg.env.send_eval_data = True
    return cfg


class ZGWSARMComplianceCfg(ConfigNode):
    def __init__(self):
        configured = configure_zgwsarm_compliance()
        super().__init__(**vars(configured))


class ZGWSARMComplianceCfgPPO(ConfigNode):
    def __init__(self):
        configured = build_compliance_ppo_config("zgwsarm_compliance")
        super().__init__(**vars(configured))


__all__ = [
    "ZGWSARM_REWARD_SCALES",
    "ZGWSARMComplianceCfg",
    "ZGWSARMComplianceCfgPPO",
    "configure_zgwsarm_compliance",
    "configure_zgwsarm_compliance_play",
]
