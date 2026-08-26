"""Independent training and play configuration for ZGWSARM compliance."""

from wbc_compliance_gym.commands import (
    force_command_ranges,
    set_force_command_ranges,
)
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
    "base_height": -5.0,
    "ang_vel_xy": -0.05,
    "lin_vel_z": -4.0,
    # Only active for near-zero base commands, so locomotion remains free.
    "stance_posture": -0.5,
    # Reject one-sided arm-load collapse while allowing symmetric crouching.
    "stance_symmetry": -2.0,

    # 轮足支撑
    "wheel_contact_consistency": -2.0,
    "wheel_support_load": -1.0,
    # Soft footprint bounds and front/rear X alignment; no joint equality binding.
    "wheel_support_geometry": -2.0,
    # wheel_lateral_slip is currently disabled. To enable its mode-dependent
    # penalty, add it here with a negative non-zero weight (for example -0.5).
    "wheel_rolling_consistency": -0.5,
    "wheel_v_tracking": 0.5,
    "wheel_yaw_tracking": 0.5,

    # 动作平滑与能耗
    "action_magnitude": -0.05,
    "action_smoothness_1_arm": -0.01,
    "action_smoothness_1_leg": -0.03,
    "action_smoothness_2_arm": -0.01,
    "dof_acc_arm": -3e-8,
    "dof_acc_leg": -1.5e-8,   #-1.5e-6
    "dof_vel_arm": -0.003,
    "dof_vel_leg": -0.0008,
    "torques": -8e-5,
    "torques_arm": -1e-5,

    # 关节、力矩与接触安全
    "dof_pos_limits_arm": -10.0,
    "dof_pos_limits_leg": -1.0,
    "torque_limits_arm": -0.005,
    "torque_limits_leg": -0.005,
    "feet_contact_forces": -0.01,   #-0.1
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


# =============================================================================
# Environment
# =============================================================================


def _configure_zgwsarm_environment(cfg):
    cfg.env.num_envs = 4096
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


# =============================================================================
# Terrain
# =============================================================================


def _configure_zgwsarm_terrain(cfg):
    cfg.terrain.border_size = 0.0 # 整张地形网格外围的额外边界宽度，单位 m；
    # 地形实现类型：使用由方块地形拼接并转换得到的三角网格。
    cfg.terrain.mesh_type = "boxes_tm"
    cfg.terrain.num_cols = 20
    cfg.terrain.num_rows = 20
    cfg.terrain.terrain_width = 5.0
    cfg.terrain.terrain_length = 5.0
    cfg.terrain.x_init_range = 1.0  #复位时相对地形单位中心位置
    cfg.terrain.y_init_range = 1.0
    cfg.terrain.yaw_init_range = 3.14
    cfg.terrain.teleport_thresh = 0.3   # 启用越界传送时，机器人距地形单元边缘小于该距离便触发传送，单位 m。
    # 是否在机器人越过地形单元边界时将其传送至另一侧；
    cfg.terrain.teleport_robots = False
    # 是否只从整张地形网格的中央区域为机器人选择出生单元。
    cfg.terrain.center_robots = True
    # 中央出生区域相对网格中心的半跨度；4 对应中心附近 8 行 x 8 列。
    cfg.terrain.center_span = 4
    # 高度场相邻采样点之间的水平距离，单位 m；越小则网格越精细。
    cfg.terrain.horizontal_scale = 0.10
    # 是否按照训练表现逐步提升地形难度；False 表示不使用地形课程。
    cfg.terrain.curriculum = False
    # 基础地形生成阶段的高度噪声幅值，单位 m；0 表示该项不添加噪声。
    # 注意：domain_rand.randomize_tile_roughness 仍可另外叠加单块地形粗糙度。
    cfg.terrain.terrain_noise_magnitude = 0.0
    # 第一层：基础高度场的采样概率，必须为 10 项且总和为 1。
    cfg.terrain.heightfield_terrain_proportions = [
        0.0,  # 0：光滑金字塔坡面（内部各半采样正、负坡度）
        0.0,  # 1：叠加随机起伏的金字塔坡面
        0.0,  # 2：负高度阶梯
        0.0,  # 3：正高度阶梯
        0.0,  # 4：离散方块障碍
        0.0,  # 5：踏脚石
        0.0,  # 6：预留槽位，当前实现为平地
        0.0,  # 7：预留槽位，当前实现为平地
        1.0,  # 8：均匀随机粗糙地形，幅值由 terrain_noise_magnitude 控制
        0.0,  # 9：半块平地、半块固定幅值粗糙地形
    ]
    # 第二层：boxes/boxes_tm 结构地形的采样概率，必须为 5 项且总和为 1。
    cfg.terrain.box_terrain_proportions = [
        1.0,  # 0：平地结构；之后仍可叠加 tile roughness
        0.0,  # 1：从机器人出生区域向外走时为下楼梯
        0.0,  # 2：从机器人出生区域向外走时为上楼梯
        0.0,  # 3：带中央平台的坡面
        0.0,  # 4：深坑结构
    ]


# =============================================================================
# Observations
# =============================================================================


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
    cfg.obs_scales.ee_force_x = 0.01
    cfg.obs_scales.ee_force_y = 0.01
    cfg.obs_scales.ee_force_z = 0.01


# =============================================================================
# Commands
# =============================================================================


def _configure_zgwsarm_commands(cfg):
    cfg.commands.num_commands = 23
    cfg.commands.resampling_time = 10
    cfg.commands.command_curriculum = False   #开启课程
    cfg.commands.distributional_commands = True
    cfg.commands.velocity_sampling = "continuous_modes"
    cfg.commands.lin_vel_x = [-0.5, 0.5]
    cfg.commands.limit_vel_x = [-1.0, 1.0]
    cfg.commands.lin_vel_y = [0.0, 0.0]
    cfg.commands.limit_vel_y = [0.0, 0.0]
    cfg.commands.ang_vel_yaw = [-0.5, 0.5]
    cfg.commands.limit_vel_yaw = [-1.5, 1.5]
    cfg.commands.vx_curriculum_step = 0.1
    cfg.commands.vy_curriculum_step = 0.1
    cfg.commands.yaw_curriculum_step = 0.1
    cfg.commands.vx_success_threshold = 0.15
    cfg.commands.vy_success_threshold = 0.15
    cfg.commands.yaw_success_threshold = 0.15
    cfg.commands.velocity_curriculum_min_samples = 50000
    cfg.commands.velocity_curriculum_required_successes = 2
    cfg.commands.velocity_curriculum_ema_alpha = 0.5
    cfg.commands.velocity_curriculum_active_threshold = 0.05
    cfg.commands.planar_command_mixture = {
        "stand": 0.2,
        "pure_x": 0.3,
        "pure_y": 0.0,
        "pure_yaw": 0.4,
        "xy": 0.0,
        "x_yaw": 0.1,
        "y_yaw": 0.0,
        "full": 0.0,
    }
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
    # Velocity dimensions no longer enter the legacy Cartesian grid.
    for name in ("num_bins_vel_x", "num_bins_vel_y", "num_bins_vel_yaw"):
        if hasattr(cfg.commands, name):
            delattr(cfg.commands, name)
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

    # Cartesian force targets are sampled independently in the base-yaw frame.
    # Equal training ranges remain concise while play can isolate any axis.
    cfg.commands.ee_force_x = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.limit_ee_force_x = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.ee_force_y = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.limit_ee_force_y = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.ee_force_z = list(ZGWSARM_FORCE_COMPONENT_RANGE)
    cfg.commands.limit_ee_force_z = list(ZGWSARM_FORCE_COMPONENT_RANGE)
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


# =============================================================================
# Domain randomization
# =============================================================================


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
    cfg.domain_rand.gripper_force_kp_range = [25.0, 400.0]
    cfg.domain_rand.gripper_force_kd_range = [3.0, 10.0]
    # 每次进入新的末端受力片段时，启用虚拟力场的概率；0.8 即约 80% 的
    # 片段由锚点弹簧施力，其余约 20% 为 freed（无锚点约束）片段。
    cfg.domain_rand.gripper_forced_prob = 0.8
    # 可采样的力场锚点模式：两种模式都使用机器人相对坐标系；static 模式下
    # 锚点局部位置不动，moving 模式下锚点局部位置会随时间分段移动。
    cfg.domain_rand.force_anchor_modes = [
        "robot_relative_static",
        "robot_relative_moving",
    ]
    # 上述锚点模式的采样概率，顺序一一对应：80% 静态、20% 移动。
    cfg.domain_rand.force_anchor_mode_probs = [0.8, 0.2]
    # moving 模式下锚点速度模长的随机范围，单位 m/s；三维运动方向另行随机。
    cfg.domain_rand.force_anchor_velocity_range = [0.0, 0.02]
    # moving 模式下每段恒定运动持续时间的随机范围，单位 s；到时重新采样运动段。
    cfg.domain_rand.force_anchor_motion_duration_s = [1.0, 3.0]
    # moving 锚点相对最初锁定局部点的 XYZ 最大绝对偏移，单位 m。
    cfg.domain_rand.force_anchor_offset_limit = [0.05, 0.05, 0.03]

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


# =============================================================================
# Rewards and termination
# =============================================================================


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
    # Soft footprint limits. The lateral upper bounds prevent an outward-splay
    # solution while retaining enough width variation for load compensation.
    cfg.rewards.wheel_support_front_x_range = [0.25, 0.45]
    cfg.rewards.wheel_support_rear_x_range = [-0.45, -0.25]
    cfg.rewards.wheel_support_right_y_range = [-0.32, -0.12]
    cfg.rewards.wheel_support_left_y_range = [0.12, 0.32]
    cfg.rewards.wheel_support_geometry_scale = 0.10
    cfg.rewards.stance_lateral_symmetry_tolerance = 0.04
    cfg.rewards.stance_height_symmetry_tolerance = 0.04  #左右支撑高度相差容忍高度
    cfg.rewards.stance_symmetry_scale = 0.10
    cfg.rewards.wheel_lateral_slip_scale = 0.25
    cfg.rewards.wheel_rolling_error_scale = 0.25
    cfg.rewards.wheel_v_tracking_scale = 0.40
    cfg.rewards.wheel_yaw_tracking_scale = 0.50
    cfg.rewards.wheel_command_active_threshold = 0.05
    cfg.rewards.wheel_lateral_slip_mode_weights = {
        "stand": 1.0,
        "pure_x": 1.0,
        "pure_y": 0.0,
        "pure_yaw": 0.2,
        "xy": 0.0,
        "x_yaw": 0.2,
        "y_yaw": 0.0,
        "full": 0.0,
    }

    apply_reward_scales(cfg, ZGWSARM_REWARD_SCALES)


# =============================================================================
# Normalization
# =============================================================================


def _configure_zgwsarm_normalization(cfg):
    cfg.normalization.clip_actions = 4.0
    cfg.normalization.friction_range = [0, 1]
    cfg.normalization.ground_friction_range = [0, 1]


# =============================================================================
# Simulation
# =============================================================================


def _configure_zgwsarm_simulation(cfg):
    cfg.sim.physx.max_gpu_contact_pairs = 2 ** 23
    cfg.sim.physx.default_buffer_size_multiplier = 3


# =============================================================================
# PPO training and run identity
# =============================================================================


def _configure_zgwsarm_training(cfg):
    """Task-owned training defaults; explicit CLI arguments override these."""
    # Rollout length and training schedule.
    cfg.runner.num_steps_per_env = 48
    cfg.runner.max_iterations = 10000
    cfg.runner.save_interval = 500
    cfg.runner.save_video_interval = 0
    cfg.runner.log_freq = 1

    # Common PPO tuning parameters.
    cfg.algorithm.learning_rate = 1.0e-3
    cfg.algorithm.adaptation_module_learning_rate = 1.0e-3
    cfg.algorithm.entropy_coef = 0.005
    cfg.algorithm.num_learning_epochs = 5
    cfg.algorithm.num_mini_batches = 4

    # Run identity. ``--run-name`` overrides training_name for one launch.
    cfg.run.training_name = "826v1_lateral_slip_off"
    cfg.run.experiment_group = "wbc"
    cfg.run.experiment_job_type = "release"

    # Local checkpoint resume. This deliberately does not use
    # cfg.runner.resume, which is the legacy remote W&B resume path.
    cfg.run.resume = False
    cfg.run.resume_run_dir = None  # None: use the latest local run for this task.
    cfg.run.resume_checkpoint = "latest"
    return cfg


# =============================================================================
# Validation and environment factory
# =============================================================================


def _validate_zgwsarm_config(cfg):
    validate_active_reward_scales(cfg, ZGWSARM_REWARD_SCALES)
    force_command_ranges(cfg.commands)

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
        "wheel_support_right_y_range",
        "wheel_support_left_y_range",
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
        "stance_lateral_symmetry_tolerance",
        "stance_height_symmetry_tolerance",
        "stance_symmetry_scale",
        "wheel_lateral_slip_scale",
        "wheel_rolling_error_scale",
        "wheel_v_tracking_scale",
        "wheel_yaw_tracking_scale",
        "wheel_command_active_threshold",
    ):
        if getattr(cfg.rewards, name) <= 0.0:
            raise ValueError(f"ZGWSARM rewards.{name} must be positive")
    lateral_mode_weights = cfg.rewards.wheel_lateral_slip_mode_weights
    expected_modes = {
        "stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "full"
    }
    if set(lateral_mode_weights) != expected_modes:
        raise ValueError("wheel_lateral_slip_mode_weights must define all modes")
    if any(float(value) < 0.0 for value in lateral_mode_weights.values()):
        raise ValueError("wheel lateral-slip mode weights must be non-negative")
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
    _configure_zgwsarm_terrain(cfg)
    _configure_zgwsarm_observations(cfg)
    _configure_zgwsarm_commands(cfg)
    _configure_zgwsarm_domain_randomization(cfg)
    _configure_zgwsarm_rewards(cfg)
    _configure_zgwsarm_normalization(cfg)
    _configure_zgwsarm_simulation(cfg)
    _validate_zgwsarm_config(cfg)

    return cfg

# =============================================================================
# Play and diagnostic overrides
# =============================================================================


def configure_zgwsarm_compliance_play(
    cfg,
    *,
    num_envs=1,
    control_mode=None,
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
    """Apply ZGWSARM play defaults, then explicit caller overrides."""
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
    cfg.terrain.mesh_type = "plane"  # plane requires teleport_robots = False

    cfg.commands.lin_vel_x = [0.0, 0.0]
    cfg.commands.limit_vel_x = [0.0, 0.0]
    cfg.commands.lin_vel_y = [0.0, 0.0]
    cfg.commands.limit_vel_y = [0.0, 0.0]
    cfg.commands.ang_vel_yaw = [0.3, 0.3]
    cfg.commands.limit_vel_yaw = [0.3, 0.3]
    cfg.commands.planar_command_mixture = {"pure_yaw": 1.0}
    # Task-owned play defaults. Edit this block to change normal play behavior.
    cfg.commands.hybrid_mode = "position"
    cfg.commands.ee_sphe_radius = [0.40, 0.40]
    cfg.commands.limit_ee_sphe_radius = [0.40, 0.40]
    cfg.commands.ee_sphe_pitch = [0.0, 0.0]
    cfg.commands.limit_ee_sphe_pitch = [0.0, 0.0]
    cfg.commands.ee_sphe_yaw = [0.0, 0.0]
    cfg.commands.limit_ee_sphe_yaw = [0.0, 0.0]
    cfg.commands.ee_force_x = [10.0, 10.0]
    cfg.commands.limit_ee_force_x = [10.0, 10.0]
    cfg.commands.ee_force_y = [0.0, 0.0]
    cfg.commands.limit_ee_force_y = [0.0, 0.0]
    cfg.commands.ee_force_z = [0.0, 0.0]
    cfg.commands.limit_ee_force_z = [0.0, 0.0]

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

    # Optional caller overrides. ``scripts/play.py`` supplies ``control_mode``
    # only when --control-mode is explicitly provided.
    if control_mode is not None:
        cfg.commands.hybrid_mode = control_mode
    if seed is not None:
        cfg.commands.curriculum_seed = seed
    if force_amplitude is not None:
        force_range = [
            -float(force_amplitude),
            float(force_amplitude),
        ]
        set_force_command_ranges(cfg.commands, force_range)
        # Retain the legacy field for old deployment metadata and keep the
        # virtual-spring output limit consistent with the requested test.
        cfg.domain_rand.max_push_force_xyz_gripper = list(force_range)
        cfg.domain_rand.max_push_force_xyz_gripper_freed = list(force_range)

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
    if fixed_lin_vel_x != 0.0 and fixed_ang_vel_yaw != 0.0:
        cfg.commands.planar_command_mixture = {"x_yaw": 1.0}
    elif fixed_lin_vel_x != 0.0:
        cfg.commands.planar_command_mixture = {"pure_x": 1.0}
    elif fixed_ang_vel_yaw != 0.0:
        cfg.commands.planar_command_mixture = {"pure_yaw": 1.0}
    else:
        cfg.commands.planar_command_mixture = {"stand": 1.0}

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
        set_force_command_ranges(cfg.commands, [-amplitude, amplitude])
        cfg.domain_rand.max_push_force_xyz_gripper = [-amplitude, amplitude]
        cfg.domain_rand.max_push_force_xyz_gripper_freed = [-amplitude, amplitude]
    else:
        cfg.commands.hybrid_mode = "position"

    if scenario != "force":
        set_force_command_ranges(cfg.commands, [0.0, 0.0])
        cfg.domain_rand.max_push_force_xyz_gripper = [0.0, 0.0]
        cfg.domain_rand.max_push_force_xyz_gripper_freed = [0.0, 0.0]
    return cfg


# =============================================================================
# Public config wrappers
# =============================================================================


class ZGWSARMComplianceCfg(ConfigNode):
    def __init__(self):
        configured = configure_zgwsarm_compliance()
        super().__init__(**vars(configured))


class ZGWSARMComplianceCfgPPO(ConfigNode):
    def __init__(self):
        configured = build_compliance_ppo_config("zgwsarm_compliance")
        _configure_zgwsarm_training(configured)
        super().__init__(**vars(configured))


__all__ = [
    "ZGWSARM_DIAGNOSTIC_SCENARIOS",
    "ZGWSARM_EE_PITCH_RANGE",
    "ZGWSARM_EE_RADIUS_RANGE",
    "ZGWSARM_EE_YAW_RANGE",
    "ZGWSARM_FORCE_COMPONENT_RANGE",
    "ZGWSARM_REWARD_SCALES",
    "ZGWSARM_TRAINING_NAME",
    "ZGWSARMComplianceCfg",
    "ZGWSARMComplianceCfgPPO",
    "configure_zgwsarm_compliance",
    "configure_zgwsarm_compliance_play",
    "configure_zgwsarm_diagnostic_play",
]
