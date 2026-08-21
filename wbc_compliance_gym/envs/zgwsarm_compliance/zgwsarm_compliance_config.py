"""Independent training and play configuration for ZGWSARM compliance."""

from wbc_compliance_gym.envs.base.compliance_task_config import (
    apply_reward_scales,
    build_compliance_ppo_config,
    new_compliance_env_config,
    validate_active_reward_scales,
)
from wbc_compliance_gym.robots.configs.zgwsarm import config_zgwsarm
from wbc_compliance_gym.utils.config_utils import ConfigNode


ZGWSARM_REWARD_SCALES = {
    # 任务跟踪
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel_yaw": 2.0,
    "manip_pos_tracking": 3.0,
    "manip_ori_tracking": 2.5,
    "ee_force_x": 3.0,
    "ee_force_y": 3.0,
    "ee_force_z": 3.0,

    # 底盘稳定
    "survival": 1.0,
    "orientation": -5.0,
    "base_height": -30.0,
    "ang_vel_xy": -0.05,
    "lin_vel_z": -4.0,
    # Only active for near-zero base commands, so locomotion remains free to
    # articulate the legs while stationary manipulation gets a firmer stance.
    "stance_posture": -2.0,

    # 轮足支撑
    "wheel_contact_consistency": -2.0,
    "wheel_support_load": -1.0,
    "wheel_support_load_balance": -1.0,
    # Softly regularize the support footprint; this does not bind paired
    # joints or prevent independent suspension motion.
    "wheel_support_geometry": -2.0,
    "wheel_lateral_slip": -0.5,
    "wheel_rolling_consistency": -0.5,

    # 动作平滑与能耗
    "action_magnitude": -0.05,
    "action_smoothness_1_arm": -0.01,
    "action_smoothness_1_leg": -0.03,
    "action_smoothness_2_arm": -0.01,
    "dof_acc_arm": -3e-8,
    "dof_acc_leg": -1.5e-6,
    "dof_vel_arm": -0.003,
    "dof_vel_leg": -0.0008,
    "torques": -8e-5,
    "torques_arm": -1e-5,

    # 关节、力矩与接触安全
    "dof_pos_limits_arm": -10.0,
    "dof_pos_limits_leg": -2.0,
    # ABAD joints receive an earlier, robot-specific soft barrier because the
    # observed distorted stance repeatedly dwelled near their hard limits.
    "abad_pos_limits": -2.0,
    "torque_limits_arm": -0.005,
    "torque_limits_leg": -0.005,
    "feet_contact_forces": -0.1,
    "collision": -5.0,

    # 回合终止：仅真实失败生效，正常超时不扣分
    "termination": -10.0,
}

ZGWSARM_DIAGNOSTIC_SCENARIOS = (
    "zero_action",
    "velocity_arm_fixed",
    "position_arm",
    "force",
)

#机械臂运动范围指令
ZGWSARM_EE_RADIUS_RANGE = (0.35, 0.80)
ZGWSARM_EE_PITCH_RANGE = (-1.35, 0.60)
ZGWSARM_EE_YAW_RANGE = (-1.20, 1.20)

#机械臂力指令范围。
# The largest possible resultant is therefore sqrt(3) * 30 ~= 52 N.
ZGWSARM_FORCE_COMPONENT_RANGE = (-30.0, 30.0)


def _configure_zgwsarm_environment(cfg):
    cfg.env.num_envs = 4000
    cfg.env.num_observation_history = 10
    cfg.env.num_privileged_obs = 16
    cfg.env.observe_vel = False
    cfg.env.priv_passthrough = False
    cfg.env.recording_width_px = 180
    cfg.env.recording_height_px = 120

    # Read-only rollout diagnostics. These values do not participate in the
    # observation, reward, control, or termination paths.
    cfg.diagnostics = ConfigNode(
        zgwsarm_scenario=None,
        hard_limit_margin=0.05,
    )


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

    # Slots 12-14 retain historical force-range names in the shared command
    # schema.  Keep their compatibility bounds synchronized with the actual
    # XYZ force sampler configured in domain_rand below.
    cfg.commands.ee_force_z = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.limit_ee_force_z = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.ee_force_magnitude = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.limit_ee_force_magnitude = list(
        ZGWSARM_FORCE_COMPONENT_RANGE
    )
    cfg.commands.ee_sphe_radius = list(ZGWSARM_EE_RADIUS_RANGE)
    cfg.commands.limit_ee_sphe_radius = list(ZGWSARM_EE_RADIUS_RANGE)
    cfg.commands.ee_sphe_pitch = list(ZGWSARM_EE_PITCH_RANGE)
    cfg.commands.limit_ee_sphe_pitch = list(ZGWSARM_EE_PITCH_RANGE)
    cfg.commands.ee_sphe_yaw = list(ZGWSARM_EE_YAW_RANGE)
    cfg.commands.limit_ee_sphe_yaw = list(ZGWSARM_EE_YAW_RANGE)
    # All corners of the configured spherical box remain above this height at
    # the nominal 0.54 m base height; the shared B1+Z1 default remains 0.05 m.
    cfg.commands.ee_min_world_height = 0.15
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
    cfg.rewards.soft_dof_pos_limit_arm = 0.9
    cfg.rewards.soft_abad_pos_limit = 0.70
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
    # Match B1+Z1: Fx/Fy commands are expressed in the heading-only base
    # frame (base yaw, independent of roll/pitch). PhysX still receives the
    # resulting force tensor in GLOBAL_SPACE; measured world forces are
    # rotated back into this yaw frame before reward evaluation.
    cfg.rewards.force_command_frame = "yaw"
    # Geodesic quaternion error width in radians for the calibrated tool frame.
    cfg.rewards.manip_ori_tracking_sigma = 0.5

    # Wheel-specific normalized reward parameters.  A 20 N minimum is far
    # below the nominal ~116 N/wheel static load, so brief dynamics are allowed
    # while a persistently unloaded/lifted wheel is penalized.
    cfg.rewards.wheel_contact_force_threshold = 5.0
    cfg.rewards.wheel_min_support_force = 20.0
    # The loaded zero-action stance is about +/-0.21 m in base Y. Keep each
    # front/rear pair centered with a 4 cm deadband and allow a broad 0.32--0.50
    # m track width. These are task-space costs, not joint equality constraints.
    cfg.rewards.wheel_support_front_x_range = [0.25, 0.45]
    cfg.rewards.wheel_support_rear_x_range = [-0.45, -0.25]
    cfg.rewards.wheel_support_track_width_range = [0.32, 0.50]
    cfg.rewards.wheel_support_center_y_deadband = 0.04
    cfg.rewards.wheel_support_geometry_scale = 0.10
    # Relative wheel-load deviations inside 25% of the four-wheel mean are
    # free. Larger imbalance is normalized by the same tolerance.
    cfg.rewards.wheel_support_load_balance_tolerance = 0.25
    cfg.rewards.wheel_lateral_slip_scale = 0.25
    cfg.rewards.wheel_rolling_error_scale = 0.25

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
    cfg.domain_rand.max_push_force_xyz_gripper = list(
        ZGWSARM_FORCE_COMPONENT_RANGE
    )
    cfg.domain_rand.max_push_force_xyz_gripper_freed = list(
        ZGWSARM_FORCE_COMPONENT_RANGE
    )
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
    if cfg.rewards.force_command_frame not in {"world", "yaw"}:
        raise ValueError(
            "invalid ZGWSARM force command frame "
            f"{cfg.rewards.force_command_frame!r}"
        )
    arm_reset_range = cfg.init_state.arm_reset_position_range
    if len(arm_reset_range) != 2 or arm_reset_range[0] > arm_reset_range[1]:
        raise ValueError(
            "invalid ZGWSARM arm reset range "
            f"{arm_reset_range!r}"
        )
    for name in (
        "wheel_support_front_x_range",
        "wheel_support_rear_x_range",
        "wheel_support_track_width_range",
    ):
        value_range = getattr(cfg.rewards, name)
        if len(value_range) != 2 or value_range[0] > value_range[1]:
            raise ValueError(
                f"invalid ZGWSARM rewards.{name}={value_range!r}"
            )
    for name in (
        "manip_ori_tracking_sigma",
        "wheel_contact_force_threshold",
        "wheel_min_support_force",
        "wheel_support_geometry_scale",
        "wheel_support_load_balance_tolerance",
        "wheel_lateral_slip_scale",
        "wheel_rolling_error_scale",
    ):
        if getattr(cfg.rewards, name) <= 0.0:
            raise ValueError(f"ZGWSARM rewards.{name} must be positive")
    if cfg.rewards.wheel_support_center_y_deadband < 0.0:
        raise ValueError(
            "ZGWSARM rewards.wheel_support_center_y_deadband must be "
            "non-negative"
        )
    if not 0.0 < cfg.rewards.soft_abad_pos_limit <= 1.0:
        raise ValueError(
            "ZGWSARM rewards.soft_abad_pos_limit must lie in (0, 1]"
        )
    if cfg.rewards.soft_abad_pos_limit > cfg.rewards.soft_dof_pos_limit:
        raise ValueError(
            "ZGWSARM ABAD soft limit must not be looser than the general "
            "leg soft limit"
        )
    # Reward aggregation is a task tuning choice. Positive clipping, Ji22,
    # and fully signed rewards are all supported by the shared reward loop;
    # only enabling both positive-only modes at once is ambiguous because the
    # first branch would silently shadow Ji22.
    if (
        cfg.rewards.only_positive_rewards
        and cfg.rewards.only_positive_rewards_ji22_style
    ):
        raise ValueError(
            "ZGWSARM reward aggregation modes are mutually exclusive: "
            "enable only one of only_positive_rewards and "
            "only_positive_rewards_ji22_style"
        )
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
    for index, name in enumerate(
        ("ee_sphe_radius", "ee_sphe_pitch", "ee_sphe_yaw")
    ):
        default_value = cfg.commands.default_ee_position_spherical[index]
        value_range = getattr(cfg.commands, name)
        if not value_range[0] <= default_value <= value_range[1]:
            raise ValueError(
                f"ZGWSARM default EE {name}={default_value} lies outside "
                f"the command range {value_range}"
            )
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
    diagnostic_scenario=None,
    diagnostic_lin_vel_x=None,
    diagnostic_ang_vel_yaw=None,
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
        force_range = [
            -float(force_amplitude),
            float(force_amplitude),
        ]
        cfg.domain_rand.max_push_force_xyz_gripper = list(force_range)
        cfg.domain_rand.max_push_force_xyz_gripper_freed = list(force_range)

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
    if diagnostic_scenario is not None:
        configure_zgwsarm_diagnostic_play(
            cfg,
            diagnostic_scenario,
            lin_vel_x=diagnostic_lin_vel_x,
            ang_vel_yaw=diagnostic_ang_vel_yaw,
            force_amplitude=force_amplitude,
        )
    return cfg


def configure_zgwsarm_diagnostic_play(
    cfg,
    scenario,
    *,
    lin_vel_x=None,
    ang_vel_yaw=None,
    force_amplitude=None,
):
    """Apply deterministic, evaluation-only ZGWSARM diagnostic conditions."""
    if scenario not in ZGWSARM_DIAGNOSTIC_SCENARIOS:
        raise ValueError(
            f"unknown ZGWSARM diagnostic scenario {scenario!r}; expected one "
            f"of {ZGWSARM_DIAGNOSTIC_SCENARIOS}"
        )

    cfg.diagnostics.zgwsarm_scenario = scenario
    cfg.terrain.mesh_type = "plane"
    cfg.terrain.teleport_robots = False
    cfg.terrain.yaw_init_range = 0.0

    # Remove disturbances from the A/B test. Gripper force gains are the one
    # exception: the legacy environment initializes them only through its
    # randomization path, so a degenerate range is used below instead.
    for name, value in vars(cfg.domain_rand).items():
        if name.startswith("randomize_") and isinstance(value, bool):
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.push_gripper_stators = False
    cfg.domain_rand.push_robot_base = False
    cfg.domain_rand.randomize_lag_timesteps = False
    cfg.noise.add_noise = False

    kp_min, kp_max = cfg.domain_rand.gripper_force_kp_range
    nominal_force_kp = 0.5 * (float(kp_min) + float(kp_max))
    cfg.domain_rand.gripper_force_kp_range = [
        nominal_force_kp,
        nominal_force_kp,
    ]
    cfg.domain_rand.randomize_gripper_force_gains = True
    cfg.domain_rand.gripper_forced_prob = 1.0

    fixed_lin_vel_x = 0.0 if lin_vel_x is None else float(lin_vel_x)
    fixed_ang_vel_yaw = 0.0 if ang_vel_yaw is None else float(ang_vel_yaw)
    if scenario == "velocity_arm_fixed" and lin_vel_x is None:
        fixed_lin_vel_x = 0.5
    cfg.commands.lin_vel_x = [fixed_lin_vel_x, fixed_lin_vel_x]
    cfg.commands.limit_vel_x = [fixed_lin_vel_x, fixed_lin_vel_x]
    cfg.commands.lin_vel_y = [0.0, 0.0]
    cfg.commands.limit_vel_y = [0.0, 0.0]
    cfg.commands.ang_vel_yaw = [fixed_ang_vel_yaw, fixed_ang_vel_yaw]
    cfg.commands.limit_vel_yaw = [fixed_ang_vel_yaw, fixed_ang_vel_yaw]

    if scenario in {"zero_action", "velocity_arm_fixed"}:
        # These scenarios isolate the base controller. Starting the arm at its
        # exact default pose prevents a reset perturbation from masquerading
        # as a locomotion-induced arm motion.
        cfg.init_state.arm_reset_position_range = [0.0, 0.0]
        radius, pitch, yaw = cfg.commands.default_ee_position_spherical
        cfg.commands.ee_sphe_radius = [radius, radius]
        cfg.commands.limit_ee_sphe_radius = [radius, radius]
        cfg.commands.ee_sphe_pitch = [pitch, pitch]
        cfg.commands.limit_ee_sphe_pitch = [pitch, pitch]
        cfg.commands.ee_sphe_yaw = [yaw, yaw]
        cfg.commands.limit_ee_sphe_yaw = [yaw, yaw]

    cfg.asset.fixed_action_targets = {}
    if scenario == "zero_action":
        cfg.asset.fixed_action_targets = {
            name: 0.0
            for name in (
                list(cfg.asset.leg_dof_names)
                + list(cfg.asset.wheel_dof_names)
                + list(cfg.asset.arm_dof_names)
            )
        }
    elif scenario == "velocity_arm_fixed":
        cfg.asset.fixed_action_targets = {
            name: 0.0 for name in cfg.asset.arm_dof_names
        }

    if scenario == "position_arm":
        cfg.commands.hybrid_mode = "position"
        cfg.commands.ee_sphe_radius = list(ZGWSARM_EE_RADIUS_RANGE)
        cfg.commands.limit_ee_sphe_radius = list(
            ZGWSARM_EE_RADIUS_RANGE
        )
        cfg.commands.ee_sphe_pitch = list(ZGWSARM_EE_PITCH_RANGE)
        cfg.commands.limit_ee_sphe_pitch = list(ZGWSARM_EE_PITCH_RANGE)
        cfg.commands.ee_sphe_yaw = list(ZGWSARM_EE_YAW_RANGE)
        cfg.commands.limit_ee_sphe_yaw = list(ZGWSARM_EE_YAW_RANGE)
    elif scenario == "force":
        amplitude = 10.0 if force_amplitude is None else float(force_amplitude)
        if amplitude < 0.0:
            raise ValueError("force_amplitude must be non-negative")
        cfg.commands.hybrid_mode = "force"
        cfg.domain_rand.max_push_force_xyz_gripper = [-amplitude, amplitude]
        cfg.domain_rand.max_push_force_xyz_gripper_freed = [-amplitude, amplitude]
    else:
        cfg.commands.hybrid_mode = "position"

    if scenario != "force":
        cfg.domain_rand.max_push_force_xyz_gripper = [0.0, 0.0]
        cfg.domain_rand.max_push_force_xyz_gripper_freed = [0.0, 0.0]
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
    "ZGWSARM_DIAGNOSTIC_SCENARIOS",
    "ZGWSARM_EE_PITCH_RANGE",
    "ZGWSARM_EE_RADIUS_RANGE",
    "ZGWSARM_EE_YAW_RANGE",
    "ZGWSARM_FORCE_COMPONENT_RANGE",
    "ZGWSARM_REWARD_SCALES",
    "ZGWSARMComplianceCfg",
    "ZGWSARMComplianceCfgPPO",
    "configure_zgwsarm_compliance",
    "configure_zgwsarm_compliance_play",
    "configure_zgwsarm_diagnostic_play",
]
