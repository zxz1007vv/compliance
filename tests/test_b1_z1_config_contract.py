import unittest

import isaacgym

from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import B1Z1Cfg, B1Z1CfgPPO
from wbc_compliance_gym.utils.config_utils import config_fingerprint


EXPECTED_ENV_CONFIG_SHA256 = (
    "e05f1682ed01fd52b847ebd14e4cf185de228ede91e9f42bbdf1ecdba53b5f4f"
)
EXPECTED_TRAIN_CONFIG_SHA256 = (
    "3971bb0ad9795963582e9f46a46121722055b5390850bee43e38798b6596df9b"
)


class B1Z1ConfigContract(unittest.TestCase):
    def test_environment_config_matches_pre_refactor_baseline(self):
        cfg = B1Z1Cfg()
        self.assertEqual(config_fingerprint(cfg), EXPECTED_ENV_CONFIG_SHA256)
        self.assertEqual(cfg.env.num_observations, 87)
        self.assertEqual(cfg.env.num_privileged_obs, 16)
        self.assertEqual(cfg.env.num_observation_history, 10)
        self.assertEqual(cfg.env.num_actions, 19)
        self.assertEqual(cfg.commands.num_commands, 23)

    def test_training_config_matches_pre_refactor_baseline(self):
        cfg = B1Z1CfgPPO()
        fingerprint_target = {
            "policy": cfg.policy,
            "algorithm": cfg.algorithm,
            "runner": cfg.runner,
        }
        self.assertEqual(
            config_fingerprint(fingerprint_target), EXPECTED_TRAIN_CONFIG_SHA256
        )

    def test_config_factories_return_isolated_objects(self):
        first = B1Z1Cfg()
        second = B1Z1Cfg()
        first.env.num_envs = 1
        first.commands.lin_vel_x[0] = -99.0
        self.assertEqual(second.env.num_envs, 4000)
        self.assertEqual(second.commands.lin_vel_x, [-1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
