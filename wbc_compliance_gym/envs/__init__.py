DEFAULT_TASK = "b1_z1_ik"


def register_tasks():
    """Register built-in tasks once, after Isaac Gym has been imported."""
    from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_env import B1Z1Env
    from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import (
        B1Z1Cfg,
        B1Z1CfgPPO,
        configure_b1_z1_play,
    )
    from wbc_compliance_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from wbc_compliance_gym.utils.task_registry import task_registry
    from wbc_compliance_rl.runners.on_policy_runner import OnPolicyRunner

    task_registry.register(
        DEFAULT_TASK,
        B1Z1Env,
        B1Z1Cfg,
        B1Z1CfgPPO,
        OnPolicyRunner,
        wrappers=(HistoryWrapper,),
        play_cfg_hook=configure_b1_z1_play,
    )

    from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_config import (
        ZGWSARMComplianceCfg,
        ZGWSARMComplianceCfgPPO,
        configure_zgwsarm_compliance_play,
    )
    from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_env import (
        ZGWSARMComplianceEnv,
    )

    task_registry.register(
        "zgwsarm_compliance",
        ZGWSARMComplianceEnv,
        ZGWSARMComplianceCfg,
        ZGWSARMComplianceCfgPPO,
        OnPolicyRunner,
        wrappers=(HistoryWrapper,),
        play_cfg_hook=configure_zgwsarm_compliance_play,
    )
    return task_registry
