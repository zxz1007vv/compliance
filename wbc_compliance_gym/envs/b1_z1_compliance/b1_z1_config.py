"""B1+Z1 whole-body compliance task configuration.

The environment assignments are intentionally kept byte-for-byte equivalent to
the former scripts/train.py configuration until numerical equivalence is proven.
"""

import os

import numpy as np

from wbc_compliance_gym.envs.base.legged_robot_config import Cfg
from wbc_compliance_gym.robots.configs.b1_z1 import config_b1_plus_z1
from wbc_compliance_gym.utils.config_utils import ConfigNode, clone_config
from wbc_compliance_rl.algorithms.ppo_cse import PPO_Args
from wbc_compliance_rl.modules.actor_critic import AC_Args
from wbc_compliance_rl.runners.on_policy_runner import RunnerArgs


def configure_b1_z1_ik(cfg=None):

    cfg = clone_config(Cfg) if cfg is None else cfg
    config_b1_plus_z1(cfg)

    # cfg.commands.control_only_z1 = True
    # cfg.commands.interpolate_ee_cmds = False
    cfg.commands.control_ee_ori = False
    cfg.commands.control_ee_ori_only_yaw = False

    # cfg.env.num_envs = 30
    # cfg.sim.physx.max_gpu_contact_pairs = 2 ** 25

    # RunnerArgs.resume = False
    # RunnerArgs.resume_path = "improbableai/b1-z1-IK/gtkntvpq"
    # RunnerArgs.resume_checkpoint = 'tmp/legged_data/ac_weights_5600.pt'

    cfg.robot.name = "b1_plus_z1"

    # Observations
    cfg.sensors.sensor_names = [
                        "OrientationSensor",    #          : size 3   projected gravity
                        "RCSensor",             # Commands : size 19
                        "JointPositionSensor",  #          : size 19
                        "JointVelocitySensor",  #          : size 19
                        "ActionSensor",         #          : size 19
                        "ClockSensor",          #          : size 4
                        ]
    cfg.sensors.sensor_args = {
                        "OrientationSensor": {},
                        "RCSensor": {},
                        "JointPositionSensor": {},
                        "JointVelocitySensor": {},
                        "ActionSensor": {},
                        "ClockSensor": {},
                        }
    cfg.env.num_scalar_observations = 83 # 95 for arm
    cfg.env.num_observations = 83 # 99
    cfg.env.episode_length_s = 20
    cfg.commands.resampling_time = 10

    # Privileged observations
    cfg.sensors.privileged_sensor_names = [
                        # "EeGripperForceSensor": {},  # size 1
                        # "EeBaseForceSensor": {},  # size 3
                        # "FrictionSensor": {},       # size 1
                        # "RestitutionSensor": {},    # size 1
                        "BodyVelocitySensor",   # size 3
                        "JointDynamicsSensor",  # size 3
                        "EeGripperForceSensor", # size 3
                        "FrictionSensor",
                        "EeGripperPositionSensor",
                        "EeGripperTargetPositionSensor",
                        # "ComDisplacementSensor",
                        # "EeGripperPosSensor",
    ]
    cfg.sensors.privileged_sensor_args = {
                        # "EeGripperForceSensor": {},  # size 1
                        # # "EeBaseForceSensor": {},  # size 3
                        # "FrictionSensor": {},
                        # "RestitutionSensor": {},
                        "BodyVelocitySensor": {},
                        "JointDynamicsSensor": {},
                        "EeGripperForceSensor": {},
                        "FrictionSensor": {},
                        "EeGripperPositionSensor": {},
                        "EeGripperTargetPositionSensor": {},
                        # "ComDisplacementSensor": {},
                        # "EeGripperPosSensor": {},
    }
    cfg.env.num_privileged_obs = 16

    cfg.commands.num_lin_vel_bins = 30
    cfg.commands.num_ang_vel_bins = 30
    cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    cfg.commands.distributional_commands = True

    # Domain randomization
    cfg.domain_rand.rand_interval_s = 4
    cfg.domain_rand.lag_timesteps = 6
    cfg.domain_rand.randomize_lag_timesteps = True
    cfg.domain_rand.randomize_rigids_after_start = False
    cfg.domain_rand.randomize_friction_indep = False
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [0.6, 5.0]
    cfg.domain_rand.randomize_restitution = False
    cfg.domain_rand.restitution_range = [0.0, 0.4]
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    cfg.domain_rand.randomize_gravity = True
    cfg.domain_rand.gravity_range = [-0.01, 0.01]
    cfg.domain_rand.gravity_rand_interval_s = 8.0
    cfg.domain_rand.gravity_impulse_duration = 0.99
    cfg.domain_rand.randomize_com_displacement = True
    cfg.domain_rand.com_displacement_range = [-0.05, 0.05]
    cfg.domain_rand.randomize_ground_friction = True
    cfg.domain_rand.ground_friction_range = [0.0, 0.01]
    cfg.domain_rand.randomize_motor_strength = False
    cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    cfg.domain_rand.randomize_motor_offset = False
    cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_Kd_factor = False
    cfg.domain_rand.randomize_tile_roughness = True
    cfg.domain_rand.tile_roughness_range = [0.0, 0.1]

    # Privileged info
    # cfg.env.priv_observe_Kd_factor = False
    # cfg.env.priv_observe_body_velocity = False
    # cfg.env.priv_observe_body_height = False
    # cfg.env.priv_observe_desired_contact_states = False
    # cfg.env.priv_observe_contact_forces = False
    # cfg.env.priv_observe_foot_displacement = False
    # cfg.env.priv_observe_gravity_transformed_foot_displacement = False
    # cfg.env.priv_observe_Kp_factor = False
    # cfg.env.priv_observe_motor_offset = False
    # cfg.env.priv_observe_motor_strength = False
    # cfg.env.priv_observe_ground_friction = False
    # cfg.env.priv_observe_ground_friction_per_foot = False
    # cfg.env.priv_observe_com_displacement = False
    # cfg.env.priv_observe_base_mass = False
    # cfg.env.priv_observe_restitution = True
    # cfg.env.priv_observe_friction = True
    # cfg.env.priv_observe_friction_indep = False
    # cfg.env.priv_observe_motion = False
    # cfg.env.priv_observe_gravity_transformed_motion = False
    # cfg.env.priv_observe_gravity = False

    cfg.env.num_observation_history = 10

    cfg.commands.num_commands = 19

    cfg.terrain.border_size = 0.0
    cfg.terrain.mesh_type = "boxes_tm"
    cfg.terrain.num_cols = 20
    cfg.terrain.num_rows = 20
    cfg.terrain.terrain_width = 5.0
    cfg.terrain.terrain_length = 5.0
    cfg.terrain.x_init_range = 1.0
    cfg.terrain.y_init_range = 1.0
    cfg.terrain.teleport_thresh = 0.3
    cfg.terrain.teleport_robots = False
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 4
    cfg.terrain.horizontal_scale = 0.10

    cfg.rewards.use_terminal_foot_height = False
    cfg.rewards.use_terminal_body_height = False
    cfg.rewards.terminal_body_height = 0.3  #0.2
    cfg.rewards.use_terminal_roll_pitch = True
    cfg.rewards.terminal_body_ori = 0.9
    # cfg.rewards.base_height_target = 0.5 #0.30
    cfg.rewards.kappa_gait_probs = 0.07
    cfg.rewards.gait_force_sigma = 100.
    cfg.rewards.gait_vel_sigma = 10.
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.only_positive_rewards_ji22_style = True
    cfg.rewards.sigma_rew_neg = 0.02

    # Rewards in use
    cfg.reward_scales.manip_pos_tracking = 3.0
    cfg.reward_scales.manip_energy = -0.000
    cfg.reward_scales.tracking_lin_vel_x = -0.0
    cfg.reward_scales.tracking_ang_vel_yaw = 0.0
    cfg.reward_scales.alive = 0.0
    cfg.reward_scales.loco_energy = -0.00000

    cfg.reward_scales.tracking_lin_vel = 0.0  #1.0
    cfg.reward_scales.tracking_ang_vel = 0.0
    cfg.reward_scales.orientation = 0.0 #-5.0
    cfg.reward_scales.torques = 0.0
    # cfg.reward_scales.dof_vel = -0.0001 #-1e-4
    # cfg.reward_scales.dof_acc = -2.5e-7
    # cfg.reward_scales.collision = -5.0 # -5.0
    # cfg.reward_scales.action_rate = -0.01 # -0.01
    # cfg.reward_scales.action_smoothness_1 = -0.1
    # cfg.reward_scales.action_smoothness_2 = -0.1 # -0.1
    cfg.reward_scales.tracking_contacts_shaped_force = 0.0 # 4.0
    cfg.reward_scales.tracking_contacts_shaped_vel = 0.0 # 4.0
    cfg.reward_scales.dof_pos_limits = -10.0

    # Rewards not in use
    cfg.reward_scales.orientation_control = 0.0
    cfg.reward_scales.feet_contact_forces = 0.0
    cfg.reward_scales.feet_slip = 0.0
    cfg.reward_scales.dof_pos = 0.0
    cfg.reward_scales.jump = 0.0
    cfg.reward_scales.base_height = 0.0
    cfg.reward_scales.estimation_bonus = 0.0
    cfg.reward_scales.feet_impact_vel = -0.0
    cfg.reward_scales.feet_clearance = -0.0
    cfg.reward_scales.feet_clearance_cmd = -0.0
    cfg.reward_scales.feet_clearance_cmd_linear = 0.0
    cfg.reward_scales.tracking_stance_width = -0.0
    cfg.reward_scales.tracking_stance_length = -0.0
    cfg.reward_scales.lin_vel_z = -0.0
    cfg.reward_scales.ang_vel_xy = -0.00
    cfg.reward_scales.feet_air_time = 0.0
    cfg.reward_scales.hop_symmetry = 0.0
    cfg.reward_scales.tracking_contacts_shaped_force = 0.0
    cfg.reward_scales.tracking_contacts_shaped_vel = 0.0

    # Commands
    cfg.commands.teleop_occulus = False
    cfg.commands.ee_sphe_radius = [0.4, 0.7] #
    cfg.commands.ee_sphe_pitch = [-2*np.pi/5, 0] #
    cfg.commands.ee_sphe_yaw = [-3*np.pi/5, 3*np.pi/5] #
    if cfg.commands.interpolate_ee_cmds:
        cfg.commands.ee_timing = [4.0, 7.0] #
    else:
        cfg.commands.ee_timing = [0.0, 0.0] #

    cfg.commands.lin_vel_x = [0.0, 0.0 ] #[-1.0, 1.0]
    cfg.commands.lin_vel_y = [0.0, 0.0] # [-0.6, 0.6]
    cfg.commands.ang_vel_yaw = [0.0, 0.0] #[-1.0, 1.0]
    cfg.commands.body_height_cmd = [0.0, 0.0]
    cfg.commands.gait_frequency_cmd_range = [1.5, 1.5]
    cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    cfg.commands.footswing_height_range = [0.2, 0.2]
    cfg.commands.body_pitch_range = [0.0, 0.0]
    cfg.commands.body_roll_range = [0.0, 0.0]
    cfg.commands.stance_width_range = [0.6, 0.6]
    cfg.commands.stance_length_range = [0.65, 0.65]
    cfg.commands.aux_reward_coef_range = [0.0, 0.0]

    # Limits
    cfg.commands.limit_ee_sphe_radius = [0.4, 0.7] #
    cfg.commands.limit_ee_sphe_pitch = [-2*np.pi/5, 0] #
    cfg.commands.limit_ee_sphe_yaw = [-3*np.pi/5, 3*np.pi/5] #
    if cfg.commands.interpolate_ee_cmds:
        cfg.commands.limit_ee_timing = [4.0, 7.0] #
    else:
        cfg.commands.limit_ee_timing = [0.0, 0.0] #

    cfg.commands.limit_vel_x = [0.0, 0.0] #[-1.0, 1.0]
    cfg.commands.limit_vel_y = [0.0, 0.0] # [-0.6, 0.6]
    cfg.commands.limit_vel_yaw = [0.0, 0.0] # = [-1.0, 1.0]
    cfg.commands.limit_body_height = [0.0, 0.0]
    cfg.commands.limit_gait_frequency = [1.5, 1.5]
    cfg.commands.limit_gait_phase = [0.5, 0.5]
    cfg.commands.limit_gait_offset = [0.0, 0.0]
    cfg.commands.limit_gait_bound = [0.0, 0.0]
    cfg.commands.limit_gait_duration = [0.5, 0.5]
    cfg.commands.limit_footswing_height = [0.2, 0.2]
    cfg.commands.limit_body_pitch = [0.0, 0.0]
    cfg.commands.limit_body_roll = [0.0, 0.0]
    cfg.commands.limit_stance_width = [0.6, 0.6]
    cfg.commands.limit_stance_length = [0.65, 0.65]
    cfg.commands.limit_aux_reward_coef = [0.0, 0.0]

    # Num bins
    cfg.commands.num_bins_ee_sphe_radius = 1 #
    cfg.commands.num_bins_ee_sphe_pitch = 1 #
    cfg.commands.num_bins_ee_sphe_yaw = 1 #
    cfg.commands.num_bins_ee_timing = 1 #

    cfg.commands.num_bins_vel_x = 1
    cfg.commands.num_bins_vel_y = 1
    cfg.commands.num_bins_vel_yaw = 1
    cfg.commands.num_bins_body_height = 1
    cfg.commands.num_bins_gait_frequency = 1
    cfg.commands.num_bins_gait_phase = 1
    cfg.commands.num_bins_gait_offset = 1
    cfg.commands.num_bins_gait_bound = 1
    cfg.commands.num_bins_gait_duration = 1
    cfg.commands.num_bins_footswing_height = 1
    cfg.commands.num_bins_body_roll = 1
    cfg.commands.num_bins_body_pitch = 1
    cfg.commands.num_bins_stance_width = 1

    cfg.viewer.follow_robot = False

    cfg.normalization.friction_range = [0, 1]
    cfg.normalization.ground_friction_range = [0, 1]
    cfg.terrain.yaw_init_range = 3.14
    cfg.normalization.clip_actions = 10.0

    cfg.commands.exclusive_phase_offset = False
    cfg.commands.pacing_offset = False
    cfg.commands.binary_phases = False
    cfg.commands.gaitwise_curricula = False
    cfg.commands.balance_gait_distribution = False

    ############################
    # Inverse IK: door opening #
    ############################
    # cfg.rewards.reward_container_name = "InverseKinematicsRewards"
    cfg.commands.inverse_IK_door_opening = True      # Specify which commands to define in leggedrobot.py/_init_command_distribution


    cfg.obs_scales.ee_sphe_radius_cmd = 0.5   # 0.2 - 0.7
    cfg.obs_scales.ee_sphe_pitch_cmd = 1.0    # -1.3 , 1.3
    cfg.obs_scales.ee_sphe_yaw_cmd = 1.3      # -1.9 , 1.9
    cfg.obs_scales.ee_timing_cmd = 0.1       # 1.0, 3.0 #

    cfg.obs_scales.ee_force_magnitude = 0.01
    cfg.obs_scales.ee_force_direction_angle = 0.3
    cfg.obs_scales.ee_force_z = 0.01

    # Collisons
    #cfg.asset.penalize_contacts_on = ["gripperStator", "gripperMover"]
    cfg.asset.terminate_after_contacts_on = [] #["thigh", "calf"] # ["base"]


    cfg.init_state.default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.10,    # 0.3110,  # [rad]
        'RL_hip_joint': 0.10,    # 0.5512,  # [rad]
        'FR_hip_joint':  -0.10,    # -0.2273, # [rad]
        'RR_hip_joint':  -0.10,    # -0.4806, # [rad]

        'FL_thigh_joint': 0.6, # 0.8530,  # [rad]
        'RL_thigh_joint': 1.0, #  0.9293,  # [rad]
        'FR_thigh_joint': 0.6, #  0.7936,  # [rad]
        'RR_thigh_joint': 1.0, #  1.0087,  # [rad]

        'FL_calf_joint': -1.3, # -1.3280,  # [rad]
        'RL_calf_joint': -1.3, #-0.8820,  # [rad]
        'FR_calf_joint': -1.3, #-1.4317,  # [rad]
        'RR_calf_joint': -1.3, #-0.7590,  # [rad]

        'joint1': 0.0,
        'joint2': 1.0, # 1.5
        'joint3': -1.8, # -1.5
        'joint4': -0.1, # -0.54
        'joint5': 0.0,
        'joint6': 0.0,
        'jointGripper': 0.0,

        'hinge1': 0.0,
        'handle_joint': 0.0,
    }

    cfg.init_state.pos = [0.0, 0.0, 0.65]

    cfg.rewards.reward_container_name = "B1LocoZ1GaitfreeRewards"
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.terminal_body_height = 0.3

    cfg.env.default_leg_dof_pos_RL = [ 0.0951,  1.0503, -1.2972, -0.0347,  0.9972, -1.3584,  0.4487,  0.8049, -0.8918, -0.3377,  0.8800, -0.8595]

    ######################
    ######## ARM #########
    ######################
    cfg.reward_scales.manip_pos_tracking_radius = 0.0 #0.5
    cfg.reward_scales.torque_limits_arm = -0.005
    cfg.rewards.soft_torque_limit_arm = 1.0
    cfg.reward_scales.dof_vel_arm = -0.003
    cfg.reward_scales.dof_acc_arm = -3e-8
    cfg.reward_scales.action_rate_arm = 0.0 #-0.003
    cfg.reward_scales.action_smoothness_1_arm = -0.01 #-0.05
    cfg.reward_scales.action_smoothness_2_arm = -0.01 #-0.02
    cfg.reward_scales.dof_pos_limits_arm = -10.0

    cfg.rewards.only_positive_rewards_ji22_style = False
    cfg.rewards.only_positive_rewards = True
    cfg.rewards.total_rew_scale = 0.2

    ######################
    ######## LEG #########
    ######################

    cfg.reward_scales.dof_vel_leg = -0.0008
    cfg.reward_scales.dof_acc_leg = -1.5e-6 #0.0 #-1.5e-7 #-6e-6
    cfg.reward_scales.torques_arm = -1e-5
    cfg.reward_scales.lin_vel_z = -4.0
    cfg.reward_scales.ang_vel_xy = -0.05
    cfg.reward_scales.action_rate_leg = 0.0 #-0.003
    cfg.reward_scales.action_smoothness_1_leg = -0.03 #0.0 #-0.03
    cfg.reward_scales.action_smoothness_2_leg = 0.0 #-0.015
    cfg.reward_scales.dof_pos_limits_leg = -1.0 # -3
    cfg.rewards.soft_dof_pos_limit = 0.9
    cfg.reward_scales.dof_pos = -0.5

    cfg.rewards.soft_torque_limit_leg = 1.0
    cfg.reward_scales.torque_limits_leg = -0.005 # -0.1

    cfg.rewards.swing_ratio = 0.3
    cfg.rewards.stance_ratio = 0.3

    ### LEG locomotion ###
    cfg.reward_scales.tracking_lin_vel_x = 0.0
    cfg.rewards.tracking_sigma_v_x = 0.25
    cfg.reward_scales.tracking_lin_vel_y = 0.0
    cfg.rewards.tracking_sigma_v_y = 0.25
    cfg.reward_scales.tracking_lin_vel = 1.0
    cfg.reward_scales.tracking_ang_vel_yaw = 2.0 #[0.5, 0.75, 1.0, 1.5, 2.0]
    cfg.rewards.tracking_sigma_v_yaw = 0.25
    cfg.reward_scales.tracking_contacts_shaped_force = 3.0
    cfg.reward_scales.tracking_contacts_shaped_vel = 3.0
    cfg.commands.lin_vel_x = [-1.0, 1.0]
    cfg.commands.limit_vel_x = [-1.0, 1.0]
    cfg.commands.lin_vel_y = [-0.0, 0.0]
    cfg.commands.limit_vel_y = [-0.0, 0.0]
    cfg.commands.ang_vel_yaw = [-1.5, 1.5]
    cfg.commands.limit_vel_yaw = [-1.5, 1.5]
    cfg.commands.gait_frequency_cmd_range = [2.2, 2.2]
    cfg.commands.limit_gait_frequency = [2.2, 2.2]



    cfg.commands.end_effector_pitch = [0., 0.]
    cfg.commands.end_effector_roll = [0., 0.]
    cfg.commands.end_effector_yaw = [0., 0.]
    cfg.commands.limit_end_effector_pitch = [0., 0.]
    cfg.commands.limit_end_effector_roll = [0., 0.]
    cfg.commands.limit_end_effector_yaw = [0., 0.]

    cfg.commands.num_commands = 23
    cfg.env.num_scalar_observations = 87
    cfg.env.num_observations = 87

    cfg.reward_scales.feet_contact_forces = -0.1

    cfg.reward_scales.raibert_heuristic = -30.0

    cfg.rewards.stance_length = 0.65
    cfg.rewards.stance_width = 0.45
    cfg.reward_scales.survival = 5.0

    cfg.rewards.max_contact_force = 550.0
    # cfg.reward_scales.feet_contact_forces = -0.01
    cfg.reward_scales.torques = -8e-5

    cfg.rewards.sigma_force_magnitude = 1/50
    cfg.rewards.sigma_force_z = 1/50
    cfg.reward_scales.manip_ori_tracking = 2.5
    cfg.reward_scales.manip_ori_tracking_yaw_only = 0.0
    cfg.asset.default_dof_drive_mode = 1

    cfg.reward_scales.feet_clearance_cmd = -10.0 #-15.0
    cfg.rewards.footswing_height = 0.10

    cfg.rewards.maintain_ori_force_envs = True

    cfg.env.priv_passthrough = False #True

    # reward_scales.manip_ori_tracking = 0.0 #1.5
    cfg.reward_scales.manip_pos_tracking = 3.0
    cfg.reward_scales.ee_force_z = 3.0
    cfg.reward_scales.ee_force_x = 3.0
    cfg.reward_scales.ee_force_y = 3.0
    cfg.reward_scales.ee_force_magnitude = 0.0
    cfg.reward_scales.ee_force_direction_angle = 0.0 #3.0

    cfg.domain_rand.gripper_forced_prob = 0.8
    cfg.domain_rand.randomize_gripper_force_gains = True
    cfg.domain_rand.gripper_force_kp_range = [25., 400.]
    cfg.domain_rand.gripper_force_kd_range = [3.0, 10.0]
    cfg.domain_rand.prop_kd = 0.1
    cfg.commands.ee_force_z = [-70, 70]
    cfg.commands.limit_ee_force_z = [-70, 70]
    cfg.commands.ee_force_magnitude = [-70, 70]
    cfg.commands.limit_ee_force_magnitude = [-70, 70]

    cfg.domain_rand.max_push_force_xyz_gripper = [-70, 70]
    cfg.domain_rand.max_push_force_xyz_gripper_freed = [-70, 70]

    cfg.reward_scales.base_height = 0.0
    cfg.rewards.base_height_target = 0.55


    cfg.reward_scales.body_height_tracking = 0.0 #5.0
    cfg.reward_scales.dof_stand_up_pos_tracking = 0.0 #3.0

    # reward_scales.feet_contact_forces = 0.0 # 0.001
    # rewards.min_contact_force = 30.0
    # rewards.max_contact_force = 250.0 # 100

    cfg.rewards.gait_force_sigma = 30000

    ############ BOTH ###########
    cfg.reward_scales.collision = -5.0



    ####### TERMINATION #######

    cfg.rewards.use_terminal_torque_legs_limits = False #True
    cfg.rewards.soft_torque_limit_leg = 1.0
    cfg.rewards.termination_torque_min_time = 25

    cfg.rewards.use_terminal_torque_arm_limits = False #True
    cfg.rewards.soft_torque_limit_arm = 1.0

    cfg.commands.control_only_z1 = False
    cfg.commands.interpolate_ee_cmds = True
    cfg.commands.sample_feasible_commands = False
    cfg.commands.teleop_occulus = False

    cfg.asset.fix_base_link = False
    cfg.asset.penalize_contacts_on = ["thigh", "calf", "link02", "link03", "link06", "hip"]
    cfg.asset.terminate_after_contacts_on = ["gripperMover"]

    # noise_scales.dof_vel = 0.0

    cfg.commands.ee_sphe_radius = [0.3, 0.9]
    cfg.commands.limit_ee_sphe_radius = [0.3, 0.9]
    cfg.commands.ee_sphe_pitch = [-2*np.pi/5, 2*np.pi/5]
    cfg.commands.limit_ee_sphe_pitch = [-2*np.pi/5, 2*np.pi/5]
    cfg.commands.ee_timing = [1.0, 4.0]
    cfg.commands.limit_ee_timing = [1.0, 4.0]
    cfg.commands.settle_time = 2.0

    # Push gripper
    cfg.domain_rand.push_gripper_stators = False
    cfg.domain_rand.push_gripper_interval_s = [3.5, 9.0]
    cfg.domain_rand.max_push_vel_xyz_gripper = [-40.0, 40.0] # N
    cfg.domain_rand.push_gripper_duration_s = [1.0, 3.0]
    # domain_rand.push_interval_s = 4. #15.

    # Push robot with v_max
    cfg.domain_rand.push_robots = True
    cfg.domain_rand.max_push_vel_xy = 0.8

    # Push base
    cfg.domain_rand.push_robot_base = False
    cfg.domain_rand.push_robot_interval_s = 5.0
    cfg.domain_rand.max_push_vel_xyz_robot = [-40.0, 40.0] # N
    cfg.domain_rand.push_robot_duration_s = [1.0, 2.0]

    cfg.domain_rand.randomize_motor_strength = True
    cfg.domain_rand.randomize_Kp_factor = False
    cfg.domain_rand.randomize_Kd_factor = True

    # domain_rand.tile_roughness_range = [0.0, 0.1]
    cfg.domain_rand.tile_roughness_range = [0.0, 0.25]

    cfg.domain_rand.gravity_range = [-0.5, 0.5] #[-1.5, 1.5]

    # Arm PD gains
    default_p_gains = [20.0, 30.0, 30.0, 20.0, 15.0, 10.0, 20.0]
    default_d_gains = [2000.0]*7

    unitree_p_gains = [kp*25.6 for kp in default_p_gains]
    unitree_d_gains = [kd*0.0128 for kd in default_d_gains]

    unitree_p_gains_div6 = [kp/6 for kp in unitree_p_gains]
    unitree_d_gains_div6 = [kd/6 for kd in unitree_d_gains]
    unitree_p_gains_div6[5] = unitree_p_gains_div6[5]*6/4
    unitree_d_gains_div6[5] = unitree_d_gains_div6[5]*6/4

    cfg.commands.p_gains_arm = unitree_p_gains_div6
    cfg.commands.d_gains_arm = unitree_d_gains_div6

    # # Legs PD gains

    cfg.commands.p_gains_legs = [180.0, 180.0, 300.0]*4
    cfg.commands.d_gains_legs = [8.0, 8.0, 15.0]*4

    # Position control
    cfg.control.decimation = 4
    # override position gains
    cfg.commands.p_gains_arm = [64., 128., 64., 64., 64., 64., 64.]
    cfg.commands.d_gains_arm = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.5]

    # force position binary
    cfg.commands.hybrid_mode = "binary"  #50% position, 50% force mode

    cfg.control.arm_scale_reduction = 2.0

    cfg.env.recording_width_px = 180
    cfg.env.recording_height_px = 120

    return cfg


def configure_b1_z1_play(
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
):
    """Apply evaluation-only overrides to a B1+Z1 task config."""
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

    cfg.commands.ee_sphe_radius = [0.55, 0.55]
    cfg.commands.limit_ee_sphe_radius = [0.55, 0.55]
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


class B1Z1Cfg(ConfigNode):
    """Fresh, isolated environment config for the b1_z1_ik task."""

    def __init__(self):
        configured = configure_b1_z1_ik()
        super().__init__(**vars(configured))


class B1Z1CfgPPO(ConfigNode):
    """Training config preserving the current PPO-CSE hyperparameters."""

    def __init__(self):
        policy = clone_config(AC_Args)
        policy.init_noise_std = 1.0
        policy.adaptation_labels = [
            "motion_loss",
            "dynamics_loss",
            "force_loss",
            "friction_loss",
            "gripper_pos_loss",
            "gripper_target_pos_loss",
        ]
        policy.adaptation_dims = [3, 3, 3, 1, 3, 3]
        policy.adaptation_weights = [1, 1, 0.05, 1, 10, 1]

        algorithm = clone_config(PPO_Args)
        algorithm.entropy_coef = 0.005

        runner = clone_config(RunnerArgs)
        runner.num_steps_per_env = 48
        runner.save_video_interval = 0

        run = ConfigNode(
            task_name=os.environ.get("COMPLIANCE_TASK_NAME", "b1_z1_ik"),
            training_name=os.environ.get("COMPLIANCE_TRAINING_NAME", "wbc_release"),
            experiment_group="wbc",
            experiment_job_type="release",
        )
        super().__init__(policy=policy, algorithm=algorithm, runner=runner, run=run)

