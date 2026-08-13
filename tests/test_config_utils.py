import unittest

import isaacgym

from b1_gym.envs.base.legged_robot_config import Cfg
from b1_gym.envs.b1_z1.b1_z1_config import B1Z1Cfg
from b1_gym.utils.config_utils import (
    apply_config,
    clone_config,
    config_fingerprint,
    config_to_dict,
    normalize_saved_env_config,
)


class ConfigUtilsTests(unittest.TestCase):
    def test_clone_does_not_share_nested_sections(self):
        original_num_envs = Cfg.env.num_envs
        cloned = clone_config(Cfg)
        cloned.env.num_envs = original_num_envs + 1
        self.assertEqual(Cfg.env.num_envs, original_num_envs)

    def test_serialization_keeps_all_top_level_sections(self):
        serialized = config_to_dict(Cfg)
        self.assertIn("env", serialized)
        self.assertIn("commands", serialized)
        self.assertIn("terrain", serialized)
        self.assertIn("rewards", serialized)

    def test_config_nodes_support_isaac_gym_mapping_access(self):
        cloned = clone_config(Cfg)
        self.assertIn("solver_type", cloned.sim.physx)
        self.assertEqual(
            cloned.sim.physx["solver_type"], cloned.sim.physx.solver_type
        )
        cloned.sim.physx["solver_type"] = cloned.sim.physx.solver_type + 1
        self.assertEqual(
            cloned.sim.physx["solver_type"], cloned.sim.physx.solver_type
        )

    def test_saved_b1_z1_config_restores_nested_dictionary_values(self):
        source = B1Z1Cfg()
        saved = config_to_dict(source)
        restored = B1Z1Cfg()
        restored.control.stiffness = {"joint": -1.0}
        restored.control.damping = {"joint": -1.0}

        apply_config(restored, saved)

        self.assertEqual({"joint": 40.0}, restored.control.stiffness)
        self.assertEqual({"joint": 2.0}, restored.control.damping)
        self.assertEqual(config_fingerprint(source), config_fingerprint(restored))

    def test_runtime_scaled_reward_config_is_repaired(self):
        source = B1Z1Cfg()
        saved = config_to_dict(source)
        dt = source.control.decimation * source.sim.dt
        saved["reward_scales"] = {
            name: value * dt
            for name, value in saved["reward_scales"].items()
            if value != 0
        }

        normalized, report = normalize_saved_env_config(source, saved)
        restored = B1Z1Cfg()
        apply_config(restored, normalized)

        self.assertTrue(report["legacy_reward_scale_repaired"])
        self.assertGreaterEqual(report["matching_scaled_coefficients"], 3)
        self.assertEqual(config_fingerprint(source), config_fingerprint(restored))

    def test_raw_reward_config_is_not_modified(self):
        source = B1Z1Cfg()
        saved = config_to_dict(source)

        normalized, report = normalize_saved_env_config(source, saved)

        self.assertFalse(report["legacy_reward_scale_repaired"])
        self.assertEqual(saved, normalized)


if __name__ == "__main__":
    unittest.main()
