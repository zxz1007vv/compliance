import unittest

import isaacgym  # noqa: F401 - must precede torch imports in this project.

from wbc_compliance_gym.envs.base.compliance_task_config import (
    active_reward_scales,
    apply_reward_scales,
    build_compliance_ppo_config,
    new_compliance_env_config,
    validate_active_reward_scales,
)


class ComplianceTaskConfigTests(unittest.TestCase):
    def test_reward_manifest_disables_generic_defaults(self):
        cfg = new_compliance_env_config()
        self.assertNotEqual({}, active_reward_scales(cfg))

        manifest = {"tracking_lin_vel": 2.0, "termination": -5.0}
        apply_reward_scales(cfg, manifest)

        self.assertEqual(manifest, active_reward_scales(cfg))
        validate_active_reward_scales(cfg, manifest)

    def test_reward_manifest_rejects_zero_entries(self):
        cfg = new_compliance_env_config()
        with self.assertRaises(ValueError):
            apply_reward_scales(cfg, {"tracking_lin_vel": 0.0})

    def test_shared_ppo_builder_returns_isolated_task_configs(self):
        first = build_compliance_ppo_config("first_task")
        second = build_compliance_ppo_config("second_task")

        first.policy.adaptation_dims[0] = 99
        self.assertEqual("first_task", first.run.task_name)
        self.assertEqual("second_task", second.run.task_name)
        self.assertEqual(3, second.policy.adaptation_dims[0])


if __name__ == "__main__":
    unittest.main()
