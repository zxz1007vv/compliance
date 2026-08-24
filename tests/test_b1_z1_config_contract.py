import unittest

import isaacgym

from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import (
    B1_Z1_REWARD_SCALES,
    B1Z1Cfg,
    B1Z1CfgPPO,
    configure_b1_z1_play,
)
from wbc_compliance_gym.envs.base.compliance_task_config import (
    active_reward_scales,
)
from wbc_compliance_gym.utils.config_utils import config_fingerprint


EXPECTED_ENV_CONFIG_SHA256 = (
    "cf9478b1efa3a3d6eb591b721b2b2e0b175419f5cb101f4b367591e12bf2f427"
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

    def test_active_rewards_match_the_b1_z1_task_manifest(self):
        self.assertEqual(B1_Z1_REWARD_SCALES, active_reward_scales(B1Z1Cfg()))

    def test_config_factories_return_isolated_objects(self):
        first = B1Z1Cfg()
        second = B1Z1Cfg()
        first.env.num_envs = 1
        first.commands.lin_vel_x[0] = -99.0
        self.assertEqual(second.env.num_envs, 4000)
        self.assertEqual(second.commands.lin_vel_x, [-1.0, 1.0])

    def test_play_uses_task_owned_default_and_accepts_cli_style_override(self):
        default_cfg = B1Z1Cfg()
        self.assertEqual("binary", default_cfg.commands.hybrid_mode)

        configure_b1_z1_play(default_cfg)

        self.assertEqual("position", default_cfg.commands.hybrid_mode)

        override_cfg = B1Z1Cfg()
        configure_b1_z1_play(override_cfg, control_mode="force")
        self.assertEqual("force", override_cfg.commands.hybrid_mode)


if __name__ == "__main__":
    unittest.main()
