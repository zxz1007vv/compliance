import hashlib
import json
import unittest

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import numpy as np
import torch

from wbc_compliance_gym.commands import (
    COMMAND_SCHEMA,
    COMMAND_DIMENSION,
    CommandLifecycleMixin,
    INDEX_ANG_VEL_YAW,
    INDEX_BODY_HEIGHT,
    INDEX_BODY_PITCH,
    INDEX_BODY_ROLL,
    INDEX_EE_FORCE_X,
    INDEX_EE_FORCE_Y,
    INDEX_EE_FORCE_Z,
    INDEX_EE_PITCH_CMD,
    INDEX_EE_POS_PITCH_CMD,
    INDEX_EE_POS_RADIUS_CMD,
    INDEX_EE_POS_TIMING_CMD,
    INDEX_EE_POS_YAW_CMD,
    INDEX_EE_ROLL_CMD,
    INDEX_EE_YAW_CMD,
    INDEX_FOOTSWING_HEIGHT,
    INDEX_FORCE_OR_POSITION_INDICATOR,
    INDEX_GAIT_BOUND,
    INDEX_GAIT_DURATION,
    INDEX_GAIT_FREQUENCY,
    INDEX_GAIT_OFFSET,
    INDEX_GAIT_PHASE,
    INDEX_LIN_VEL_X,
    INDEX_LIN_VEL_Y,
    build_command_curricula,
    force_command_ranges,
    sample_control_modes,
    set_force_command_ranges,
)
from wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config import B1Z1Cfg
from wbc_compliance_gym.envs.base.legged_robot import LeggedRobot
from wbc_compliance_gym.curriculum import RewardThresholdCurriculum
from wbc_compliance_gym.rewards import B1LocoZ1GaitfreeRewards, B1Z1Rewards, REWARD_CONTAINERS
from wbc_compliance_gym.sensors import OrientationSensor, SENSOR_TYPES, make_sensor


class EnvironmentStructureContracts(unittest.TestCase):
    def test_command_vector_layout_is_unchanged(self):
        indices = (
            INDEX_LIN_VEL_X,
            INDEX_LIN_VEL_Y,
            INDEX_ANG_VEL_YAW,
            INDEX_BODY_HEIGHT,
            INDEX_GAIT_FREQUENCY,
            INDEX_GAIT_PHASE,
            INDEX_GAIT_OFFSET,
            INDEX_GAIT_BOUND,
            INDEX_GAIT_DURATION,
            INDEX_FOOTSWING_HEIGHT,
            INDEX_BODY_PITCH,
            INDEX_BODY_ROLL,
            INDEX_EE_FORCE_X,
            INDEX_EE_FORCE_Y,
            INDEX_EE_FORCE_Z,
            INDEX_EE_POS_RADIUS_CMD,
            INDEX_EE_POS_PITCH_CMD,
            INDEX_EE_POS_YAW_CMD,
            INDEX_EE_POS_TIMING_CMD,
            INDEX_EE_ROLL_CMD,
            INDEX_EE_PITCH_CMD,
            INDEX_EE_YAW_CMD,
            INDEX_FORCE_OR_POSITION_INDICATOR,
        )
        self.assertEqual(tuple(range(23)), indices)
        self.assertEqual(23, COMMAND_DIMENSION)
        self.assertEqual(indices, tuple(item.index for item in COMMAND_SCHEMA))

    def test_command_curriculum_grid_and_rng_match_frozen_baseline(self):
        cfg = B1Z1Cfg()
        env = object.__new__(LeggedRobot)
        env.cfg = cfg
        env.device = "cpu"
        env.num_envs = 8

        env._init_command_distribution(torch.arange(8))
        curriculum = env.curricula[0]
        samples, bins = curriculum.sample(64)

        digest = hashlib.sha256()
        for value in (
            curriculum.grid,
            curriculum.idx_grid,
            curriculum.weights,
            curriculum.lows,
            curriculum.highs,
            samples,
            bins,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(
            json.dumps(
                {
                    "categories": env.category_names,
                    "keys": curriculum.keys,
                    "ls": curriculum.ls,
                },
                sort_keys=True,
            ).encode()
        )

        self.assertEqual(
            "3cadd6fc1512d094a92239698fa41e5824ac1227115b74754fa6e540f58d1a82",
            digest.hexdigest(),
        )
        self.assertEqual((8,), env.env_command_bins.shape)
        self.assertEqual((8,), env.env_command_categories.shape)
        self.assertTrue(torch.equal(torch.zeros(8), env.force_or_position_control))

    def test_command_curriculum_factory_rejects_unknown_type(self):
        cfg = B1Z1Cfg()
        cfg.commands.curriculum_type = "UnknownCurriculum"
        with self.assertRaises(ValueError):
            build_command_curricula(cfg.commands)

    def test_command_curriculum_rejects_one_bin_active_limit_mismatch(self):
        cfg = B1Z1Cfg()
        cfg.commands.lin_vel_x = [-0.5, 0.5]
        cfg.commands.limit_vel_x = [-1.0, 1.0]
        cfg.commands.num_bins_vel_x = 1

        with self.assertRaisesRegex(ValueError, "uses one bin"):
            build_command_curricula(cfg.commands)

    def test_curriculum_rejects_empty_or_zero_weight_sampling_domains(self):
        curriculum = RewardThresholdCurriculum(seed=1, x=(-1.0, 1.0, 2))
        with self.assertRaisesRegex(ValueError, "select no bin centres"):
            curriculum.set_to(np.array([0.0]), np.array([0.0]))
        with self.assertRaisesRegex(ValueError, "no finite positive sampling weight"):
            curriculum.sample(batch_size=1)

    def test_curriculum_samples_inside_selected_cell_edges(self):
        curriculum = RewardThresholdCurriculum(seed=1, x=(-1.0, 1.0, 4))
        curriculum.set_to(np.array([-0.5]), np.array([0.5]))

        samples, _ = curriculum.sample(batch_size=4096)
        self.assertTrue(np.all(samples[:, 0] >= -0.5))
        self.assertTrue(np.all(samples[:, 0] <= 0.5))

    def test_cartesian_force_ranges_do_not_filter_the_legacy_curriculum(self):
        cfg = B1Z1Cfg()
        set_force_command_ranges(
            cfg.commands,
            ([20.0, 20.0], [0.0, 0.0], [-15.0, -15.0]),
        )

        self.assertEqual(
            ([20.0, 20.0], [0.0, 0.0], [-15.0, -15.0]),
            force_command_ranges(cfg.commands),
        )
        _, curricula = build_command_curricula(cfg.commands)
        curriculum = curricula[0]
        self.assertFalse(np.isnan(curriculum.weights).any())
        self.assertGreater(curriculum.weights.sum(), 0.0)
        curriculum.sample(batch_size=8)

    def test_cartesian_force_sampling_uses_active_not_limit_ranges(self):
        cfg = B1Z1Cfg()
        set_force_command_ranges(
            cfg.commands,
            ([-20.0, 20.0], [-10.0, 10.0], [5.0, 15.0]),
            update_limits=False,
        )

        self.assertEqual(
            ([-20.0, 20.0], [-10.0, 10.0], [5.0, 15.0]),
            force_command_ranges(cfg.commands),
        )

    def test_control_mode_sampling_matches_legacy_operations(self):
        for mode in ("mixed", "binary"):
            torch.manual_seed(123)
            actual = sample_control_modes(mode, 128, "cpu")
            torch.manual_seed(123)
            if mode == "mixed":
                expected = torch.rand(128, device="cpu")
            else:
                expected = torch.randint(0, 2, (128,), device="cpu").float()
            self.assertTrue(torch.equal(expected, actual))

        self.assertTrue(
            torch.equal(torch.zeros(4), sample_control_modes("position", 4, "cpu"))
        )
        self.assertTrue(
            torch.equal(torch.ones(4), sample_control_modes("force", 4, "cpu"))
        )
        with self.assertRaises(ValueError):
            sample_control_modes("postion", 1, "cpu")

    def test_reward_registry_preserves_saved_config_names(self):
        self.assertIs(B1Z1Rewards, B1LocoZ1GaitfreeRewards)
        self.assertIs(B1Z1Rewards, REWARD_CONTAINERS["B1Z1Rewards"])
        self.assertIs(B1Z1Rewards, REWARD_CONTAINERS["B1LocoZ1GaitfreeRewards"])

    def test_sensor_catalog_keeps_existing_types(self):
        self.assertIs(OrientationSensor, SENSOR_TYPES["OrientationSensor"])
        self.assertEqual(28, len(SENSOR_TYPES))

    def test_command_lifecycle_is_owned_by_command_boundary(self):
        for method_name in (
            "_init_command_distribution",
            "_resample_force_or_position_control",
            "_update_command_ranges",
            "_resample_commands",
            "_step_contact_targets",
            "compute_intermediate_ee_pos_command",
            "get_measured_ee_pos_spherical",
            "get_measured_ee_rpy_yrf",
        ):
            self.assertIs(
                getattr(LeggedRobot, method_name),
                getattr(CommandLifecycleMixin, method_name),
            )
            self.assertNotIn(method_name, LeggedRobot.__dict__)

        for physics_method in (
            "step",
            "_compute_torques",
            "_push_gripper",
            "_push_robot_base",
        ):
            self.assertIn(physics_method, LeggedRobot.__dict__)
            self.assertNotIn(physics_method, CommandLifecycleMixin.__dict__)

    def test_sensor_factory_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            make_sensor("UnknownSensor", object())


if __name__ == "__main__":
    unittest.main()
