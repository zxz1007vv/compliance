import hashlib
import importlib
import json
import unittest

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import numpy as np
import torch

from b1_gym.commands import (
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
    sample_control_modes,
)
from b1_gym.curriculum import RewardThresholdCurriculum
from b1_gym.envs.b1_z1.b1_z1_config import B1Z1Cfg
from b1_gym.envs.base.legged_robot import LeggedRobot
from b1_gym.envs.base.curriculum import (
    RewardThresholdCurriculum as LegacyRewardThresholdCurriculum,
)
from b1_gym.rewards import B1LocoZ1GaitfreeRewards, B1Z1Rewards, REWARD_CONTAINERS
from b1_gym.rewards.b1_loco_z1_gaitfree_rewards import (
    B1LocoZ1GaitfreeRewards as LegacyB1Z1Rewards,
)
from b1_gym.sensors import SENSOR_TYPES, make_sensor
from b1_gym.sensors.orientation_sensor import OrientationSensor


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
            "44e68ff3ab777f43aaace74a9901bb962e0f2f4614a7e59b9620231552be09a2",
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

    def test_reward_and_curriculum_legacy_imports_are_exact_aliases(self):
        self.assertIs(B1Z1Rewards, B1LocoZ1GaitfreeRewards)
        self.assertIs(B1Z1Rewards, LegacyB1Z1Rewards)
        self.assertIs(B1Z1Rewards, REWARD_CONTAINERS["B1Z1Rewards"])
        self.assertIs(B1Z1Rewards, REWARD_CONTAINERS["B1LocoZ1GaitfreeRewards"])
        self.assertIs(RewardThresholdCurriculum, LegacyRewardThresholdCurriculum)

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

    def test_legacy_sensor_modules_are_exact_aliases(self):
        legacy_modules = {
            "ActionSensor": "action_sensor",
            "AttachedCameraSensor": "attached_camera_sensor",
            "BodyVelocitySensor": "body_velocity_sensor",
            "ClockSensor": "clock_sensor",
            "EeBaseForceSensor": "ee_base_force_sensor",
            "EeGripperForceDirSensor": "ee_gripper_force_dir_sensor",
            "EeGripperForceMagnSensor": "ee_gripper_force_magn_sensor",
            "EeGripperForceSensor": "ee_gripper_force_sensor",
            "EeGripperPositionSensor": "ee_gripper_position_sensor",
            "EeGripperTargetPositionSensor": "ee_gripper_target_position_sensor",
            "EgomotionSensor": "egomotion_sensor",
            "FloatingCameraSensor": "floating_camera_sensor",
            "FrictionSensor": "friction_sensor",
            "GroundFrictionSensor": "ground_friction_sensor",
            "GroundRoughnessSensor": "ground_roughness_sensor",
            "HeightmapSensor": "heightmap_sensor",
            "JointDynamicsSensor": "joint_dynamics_sensor",
            "JointPositionSensor": "joint_position_sensor",
            "JointPositionTargetSensor": "joint_position_target_sensor",
            "JointVelocitySensor": "joint_velocity_sensor",
            "LastActionSensor": "last_action_sensor",
            "ObjectSensor": "object_sensor",
            "ObjectVelocitySensor": "object_velocity_sensor",
            "OrientationSensor": "orientation_sensor",
            "RCSensor": "rc_sensor",
            "RestitutionSensor": "restitution_sensor",
            "TimingSensor": "timing_sensor",
            "YawSensor": "yaw_sensor",
        }
        for class_name, module_name in legacy_modules.items():
            module = importlib.import_module(f"b1_gym.sensors.{module_name}")
            self.assertIs(getattr(module, class_name), SENSOR_TYPES[class_name])

        with self.assertRaises(ValueError):
            make_sensor("UnknownSensor", object())


if __name__ == "__main__":
    unittest.main()
