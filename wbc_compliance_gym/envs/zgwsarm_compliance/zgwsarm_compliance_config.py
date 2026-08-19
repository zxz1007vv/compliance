"""Independent training and play configuration for ZGWSARM compliance."""

from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import (
    B1Z1CfgPPO,
    configure_b1_z1_ik,
    configure_b1_z1_play,
)
from wbc_compliance_gym.robots.configs.zgwsarm import config_zgwsarm
from wbc_compliance_gym.utils.config_utils import ConfigNode


def configure_zgwsarm_compliance(cfg=None):
    """Build the mobile-manipulation task from the shared compliance baseline."""
    cfg = configure_b1_z1_ik(cfg)
    config_zgwsarm(cfg)

    
    cfg.env.num_envs = 4000
    cfg.sim.physx.max_gpu_contact_pairs = 2 ** 23
    cfg.sim.physx.default_buffer_size_multiplier = 3

    cfg.rewards.reward_container_name = "ZGWSARMRewards"
    cfg.reward_scales.raibert_heuristic = 0.0
    cfg.reward_scales.dof_pos_limits = 0.0

    # The inherited 2.2 Hz contact schedule was tuned for B1 feet, not driven
    # ZGWSARM wheels. Velocity tracking remains the locomotion objective; do
    # not force wheel contacts to imitate an unvalidated B1 gait.
    cfg.reward_scales.tracking_contacts_shaped_force = 0.0
    cfg.reward_scales.tracking_contacts_shaped_vel = 0.0
    cfg.reward_scales.feet_clearance_cmd = 0.0

    # Keep locomotion tracking rewards inherited from B1+Z1, while making
    # base stability a requirement rather than replacing locomotion with a
    # fixed standing objective.  stance_posture is active only near zero
    # velocity command.
    cfg.reward_scales.survival = 1.0
    cfg.reward_scales.termination = -10.0
    cfg.reward_scales.orientation = -5.0
    cfg.reward_scales.base_height = -30.0
    # A default-pose penalty on every locomotion step fights the leg motion
    # needed for velocity tracking.  stance_posture below replaces it and is
    # gated to the near-zero command slice.
    cfg.reward_scales.dof_pos = 0.0
    # cfg.reward_scales.stance_posture = -0.5
    cfg.reward_scales.action_magnitude = -0.05
    cfg.rewards.only_positive_rewards = False
    cfg.rewards.only_positive_rewards_ji22_style = False   #ji22风格，负奖励让正奖励变小而非截断
    cfg.rewards.use_terminal_body_height = True
    cfg.rewards.terminal_body_height = 0.38
    cfg.rewards.use_terminal_roll_pitch = True
    # projected_gravity_xy squared: 0.5 corresponds to 45 degrees of tilt.
    cfg.rewards.terminal_body_ori = 0.5

    # Keep locomotion commands active.  These explicit assignments also guard
    # against accidentally turning the training task into the zero-command
    # play configuration.
    cfg.commands.lin_vel_x = [-1.0, 1.0]
    cfg.commands.limit_vel_x = [-1.0, 1.0]
    cfg.commands.ang_vel_yaw = [-1.5, 1.5]
    cfg.commands.limit_vel_yaw = [-1.5, 1.5]

    # Bound the policy-facing action space before applying robot-specific
    # scales.  This still permits +/-1 rad leg targets and full wheel torque.
    cfg.normalization.clip_actions = 4.0

    # ZGWSARM-specific progressive randomization.  Locomotion remains active
    # throughout, while the arm starts position-only. Force-mode environments
    # are introduced after the base/arm have first learned together without
    # external EE loading, then force and other disturbances ramp gradually.
    cfg.domain_rand.zgwsarm_curriculum_enabled = True
    cfg.domain_rand.zgwsarm_force_mode_start_step = 25000
    cfg.domain_rand.zgwsarm_curriculum_start_step = 25000
    cfg.domain_rand.zgwsarm_curriculum_end_step = 125000
    cfg.domain_rand.zgwsarm_force_initial = 10.0
    cfg.domain_rand.zgwsarm_force_final = 70.0
    cfg.domain_rand.zgwsarm_push_velocity_initial = 0.0
    cfg.domain_rand.zgwsarm_push_velocity_final = 0.8
    cfg.domain_rand.zgwsarm_gravity_initial = 0.0
    cfg.domain_rand.zgwsarm_gravity_final = 0.5
    cfg.domain_rand.zgwsarm_motor_strength_initial = [0.98, 1.02]
    cfg.domain_rand.zgwsarm_motor_strength_final = [0.9, 1.1]
    cfg.domain_rand.zgwsarm_Kd_factor_initial = [0.9, 1.1]
    cfg.domain_rand.zgwsarm_Kd_factor_final = [0.5, 1.5]
    cfg.domain_rand.max_push_force_xyz_gripper = [-10.0, 10.0]
    cfg.domain_rand.max_push_force_xyz_gripper_freed = [-10.0, 10.0]
    cfg.domain_rand.max_push_vel_xy = 0.0
    cfg.domain_rand.gravity_range = [0.0, 0.0]
    cfg.domain_rand.motor_strength_range = [0.98, 1.02]
    cfg.domain_rand.Kd_factor_range = [0.9, 1.1]
    cfg.commands.hybrid_mode = "position"   #训练初期配置

    # Retain domain randomization, but remove the unvalidated B1+Z1 extremes.
    cfg.domain_rand.friction_range = [0.6, 1.5]
    cfg.domain_rand.ground_friction_range = [0.6, 1.2]
    cfg.domain_rand.tile_roughness_range = [0.0, 0.08]

    cfg.commands.ee_sphe_radius = [0.25, 0.65]
    cfg.commands.limit_ee_sphe_radius = [0.25, 0.65]
    return cfg


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
):
    """Apply evaluation-only overrides without robot details in play.py."""
    configure_b1_z1_play(
        cfg,
        num_envs=num_envs,
        control_mode=control_mode,
        seed=seed,
        force_amplitude=force_amplitude,
        fix_base=fix_base,
        teleop=teleop,
        interpolate_ee_cmds=interpolate_ee_cmds,
        sample_feasible_commands=sample_feasible_commands,
        control_only_z1=control_only_z1,
    )
    # Evaluation must use the explicitly requested force/push settings rather
    # than restarting the training curriculum at its 10 N initial value.
    cfg.domain_rand.zgwsarm_curriculum_enabled = False
    # With no explicit override, evaluate the final 70 N task rather than the
    # curriculum's 10 N starting point. This is irrelevant in position mode.
    amplitude = (
        cfg.domain_rand.zgwsarm_force_final
        if force_amplitude is None
        else float(force_amplitude)
    )
    cfg.domain_rand.max_push_force_xyz_gripper = [-amplitude, amplitude]
    cfg.domain_rand.max_push_force_xyz_gripper_freed = [-amplitude, amplitude]
    cfg.commands.ee_sphe_radius = [0.45, 0.45]
    cfg.commands.limit_ee_sphe_radius = [0.45, 0.45]
    return cfg


class ZGWSARMComplianceCfg(ConfigNode):
    def __init__(self):
        configured = configure_zgwsarm_compliance()
        super().__init__(**vars(configured))


class ZGWSARMComplianceCfgPPO(ConfigNode):
    def __init__(self):
        configured = B1Z1CfgPPO()
        configured.run.task_name = "zgwsarm_compliance"
        super().__init__(**vars(configured))


__all__ = [
    "ZGWSARMComplianceCfg",
    "ZGWSARMComplianceCfgPPO",
    "configure_zgwsarm_compliance",
    "configure_zgwsarm_compliance_play",
]
