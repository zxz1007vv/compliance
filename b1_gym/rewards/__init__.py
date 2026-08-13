"""Task reward containers and their stable config-name registry."""

from .b1_z1_rewards import B1LocoZ1GaitfreeRewards, B1Z1Rewards


REWARD_CONTAINERS = {
    "B1Z1Rewards": B1Z1Rewards,
    "B1LocoZ1GaitfreeRewards": B1Z1Rewards,
}


def make_reward_container(name, env):
    try:
        reward_type = REWARD_CONTAINERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(REWARD_CONTAINERS))
        raise ValueError(
            f"Unknown reward container {name!r}; available: {available}"
        ) from exc
    return reward_type(env)


__all__ = [
    "B1LocoZ1GaitfreeRewards",
    "B1Z1Rewards",
    "REWARD_CONTAINERS",
    "make_reward_container",
]
