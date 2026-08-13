def register_tasks():
    """Register built-in tasks once, after Isaac Gym has been imported."""
    from b1_gym.envs.b1_z1.b1_z1 import B1Z1Env
    from b1_gym.envs.b1_z1.b1_z1_config import B1Z1Cfg, B1Z1CfgPPO
    from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from b1_gym.utils.task_registry import task_registry
    from b1_gym_learn.runners.on_policy_runner import OnPolicyRunner

    task_registry.register(
        "b1_z1_ik",
        B1Z1Env,
        B1Z1Cfg,
        B1Z1CfgPPO,
        OnPolicyRunner,
        wrappers=(HistoryWrapper,),
    )
    return task_registry
