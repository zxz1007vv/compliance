"""Independent training and play configuration for ZGWSARM compliance."""

from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import (
    B1Z1CfgPPO,
    configure_b1_z1_ik,
    configure_b1_z1_play,
)
from wbc_compliance_gym.robots.configs.zgwsarm import config_zgwsarm
from wbc_compliance_gym.utils.config_utils import ConfigNode


def configure_zgwsarm_compliance(cfg=None):
    """Build a fresh robot config on the unchanged compliance baseline."""
    cfg = configure_b1_z1_ik(cfg)
    config_zgwsarm(cfg)
    cfg.rewards.reward_container_name = "ZGWSARMRewards"
    cfg.reward_scales.raibert_heuristic = 0.0
    cfg.reward_scales.dof_pos_limits = 0.0
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
