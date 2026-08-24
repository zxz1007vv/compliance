import unittest
from types import SimpleNamespace
from unittest.mock import patch

import isaacgym

from scripts import utils as script_utils
from wbc_compliance_gym.envs.base.legged_robot_config import Cfg
from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import B1Z1Cfg
from wbc_compliance_gym.utils.config_utils import (
    ConfigNode,
    apply_config,
    clone_config,
    config_fingerprint,
    config_to_dict,
    normalize_saved_env_config,
)


class ConfigUtilsTests(unittest.TestCase):
    def test_load_env_omits_absent_control_mode_and_forwards_explicit_override(self):
        received_overrides = []

        def env_cfg_factory():
            return ConfigNode(
                commands=ConfigNode(hybrid_mode="binary"),
                domain_rand=ConfigNode(max_push_force_xyz_gripper=[-70.0, 70.0]),
            )

        def play_cfg_hook(cfg, **kwargs):
            received_overrides.append(dict(kwargs))
            cfg.commands.hybrid_mode = "position"
            if "control_mode" in kwargs:
                cfg.commands.hybrid_mode = kwargs["control_mode"]

        class DummyEnv:
            num_actions = 1

            def __init__(self, *, sim_device, headless, cfg):
                self.sim_device = sim_device
                self.headless = headless
                self.cfg = cfg

        task_spec = SimpleNamespace(
            env_cfg_factory=env_cfg_factory,
            env_class=DummyEnv,
            play_cfg_hook=play_cfg_hook,
            wrappers=(),
        )

        with patch.object(script_utils, "register_tasks"), patch.object(
            script_utils.task_registry, "get_spec", return_value=task_spec
        ):
            default_env, _ = script_utils.load_env(
                sim_device="cpu", task_name="dummy", control_mode=None
            )
            override_env, _ = script_utils.load_env(
                sim_device="cpu", task_name="dummy", control_mode="force"
            )

        self.assertNotIn("control_mode", received_overrides[0])
        self.assertEqual("force", received_overrides[1]["control_mode"])
        self.assertEqual("position", default_env.cfg.commands.hybrid_mode)
        self.assertEqual("force", override_env.cfg.commands.hybrid_mode)

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

    def test_legacy_b1_asset_path_is_migrated(self):
        source = B1Z1Cfg()
        saved = config_to_dict(source)
        saved["asset"]["file"] = saved["asset"]["file"].replace(
            "/resources/robots/b1_z1/", "/resources/robots/b1/"
        )

        normalized, report = normalize_saved_env_config(source, saved)
        restored = B1Z1Cfg()
        apply_config(restored, normalized)

        self.assertTrue(report["legacy_asset_path_migrated"])
        self.assertIn("/resources/robots/b1_z1/", restored.asset.file)
        self.assertEqual(config_fingerprint(source), config_fingerprint(restored))


if __name__ == "__main__":
    unittest.main()
