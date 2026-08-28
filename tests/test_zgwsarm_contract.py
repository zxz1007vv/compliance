import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import numpy as np
from isaacgym import gymapi
import torch

from wbc_compliance_gym.envs.base.compliance_task_config import (
    active_reward_scales,
)
from wbc_compliance_gym.commands.commands import CommandLifecycleMixin
from wbc_compliance_gym.commands import (
    LOCOMOTION_MODE_NAMES,
    LOCOMOTION_MODE_TO_ID,
    ContinuousVelocityCurriculum,
    build_command_curricula,
    resolve_yaw_gait_phase_slots,
)
from wbc_compliance_gym.envs import register_tasks
from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_config import (
    ZGWSARM_DIAGNOSTIC_SCENARIOS,
    ZGWSARM_EE_PITCH_RANGE,
    ZGWSARM_EE_RADIUS_RANGE,
    ZGWSARM_EE_YAW_RANGE,
    ZGWSARM_FORCE_COMPONENT_RANGE,
    ZGWSARM_REWARD_SCALES,
    ZGWSARMComplianceCfg,
    ZGWSARMComplianceCfgPPO,
    configure_zgwsarm_compliance_play,
    configure_zgwsarm_diagnostic_play,
)
from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_env import (
    ZGWSARMComplianceEnv,
)
from wbc_compliance_gym.rewards import (
    B1Z1Rewards,
    REWARD_CONTAINERS,
    WholeBodyComplianceRewards,
    ZGWSARMRewards,
)
from wbc_compliance_gym.robots.configs.zgwsarm import (
    ABAD_DOF_NAMES,
    ARM_DOF_NAMES,
    FOOT_LINK_NAMES,
    LEG_DOF_NAMES,
    WHEEL_DOF_NAMES,
)
from wbc_compliance_gym.robots.zgwsarm import EXPECTED_DOF_NAMES
from wbc_compliance_gym.sensors.sensors import JointPositionSensor


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "resources" / "robots" / "zgwsarm"
URDF_PATH = MODEL_ROOT / "urdf" / "zgwsarm.urdf"


class ZGWSARMContractTests(unittest.TestCase):
    def test_terrain_layers_have_independent_probability_vectors(self):
        cfg = ZGWSARMComplianceCfg()
        self.assertEqual(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            cfg.terrain.heightfield_terrain_proportions,
        )
        self.assertEqual(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            cfg.terrain.box_terrain_proportions,
        )
        self.assertFalse(hasattr(cfg.terrain, "terrain_proportions"))

    def test_force_anchor_latches_in_yaw_only_robot_frame(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.num_envs = 1
        env.device = "cpu"
        env.cfg = SimpleNamespace(
            commands=SimpleNamespace(command_base_height=0.54),
            domain_rand=SimpleNamespace(
                force_anchor_mode="robot_relative_static"
            ),
        )
        env.robot_actor_idxs = torch.tensor([0], dtype=torch.long)
        env.gripper_stator_index = 0
        env.root_states = torch.tensor(
            [[0.0, 0.0, 0.54, 0.0, 0.0, 0.0, 1.0] + [0.0] * 6]
        )
        env.rigid_body_state = torch.zeros(1, 1, 13)
        env.rigid_body_state[0, 0, 0:3] = torch.tensor([1.0, 0.0, 0.54])
        env.force_anchor_local = torch.zeros(1, 3)
        env.force_anchor_latched_local = torch.zeros(1, 3)
        env.force_anchor_world = torch.zeros(1, 3)
        env.force_anchor_velocity_local = torch.zeros(1, 3)
        env.force_anchor_motion_end_step = torch.zeros(1)
        env.force_anchor_mode = torch.tensor([0], dtype=torch.long)
        env.force_anchor_active = torch.zeros(1, dtype=torch.bool)
        env.force_anchor_latch_pending = torch.ones(1, dtype=torch.bool)

        env._latch_force_anchor(torch.tensor([0], dtype=torch.long))
        torch.testing.assert_close(
            env.force_anchor_local, torch.tensor([[1.0, 0.0, 0.0]])
        )

        half_yaw = np.pi / 4.0
        env.root_states[0, 0:2] = torch.tensor([2.0, 3.0])
        env.root_states[0, 3:7] = torch.tensor(
            [0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]
        )
        env._update_force_anchor_world()
        torch.testing.assert_close(
            env.force_anchor_world,
            torch.tensor([[2.0, 4.0, 0.54]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_force_anchor_lifecycle_restarts_without_stale_state(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.device = "cpu"
        env.cfg = SimpleNamespace(
            domain_rand=SimpleNamespace(
                force_anchor_mode="robot_relative_static"
            )
        )
        env.force_or_position_control = torch.tensor([1.0, 0.0])
        env.force_anchor_active = torch.ones(2, dtype=torch.bool)
        env.force_anchor_latch_pending = torch.zeros(2, dtype=torch.bool)
        env.force_anchor_local = torch.ones(2, 3)
        env.force_anchor_latched_local = torch.ones(2, 3)
        env.force_anchor_world = torch.ones(2, 3)
        env.force_anchor_velocity_local = torch.ones(2, 3)
        env.force_anchor_motion_end_step = torch.ones(2)
        env.force_anchor_mode = torch.tensor([0, 2], dtype=torch.long)
        env.force_anchor_displacement_world = torch.ones(2, 3)
        env.force_anchor_spring_force_world = torch.ones(2, 3)
        env.freed_envs = torch.ones(2, dtype=torch.bool)
        ids = torch.tensor([0, 1], dtype=torch.long)

        env._on_force_or_position_control_resampled(
            ids, torch.tensor([0.0, 1.0])
        )

        torch.testing.assert_close(
            env.force_anchor_latch_pending,
            torch.tensor([True, False]),
        )
        self.assertFalse(torch.any(env.force_anchor_active))
        self.assertFalse(torch.any(env.freed_envs))
        torch.testing.assert_close(
            env.force_anchor_local, torch.zeros(2, 3)
        )

    def test_force_anchor_distribution_and_motion_are_explicit(self):
        cfg = ZGWSARMComplianceCfg()
        self.assertEqual(
            [
                "robot_relative_static",
                "robot_relative_moving",
            ],
            list(cfg.domain_rand.force_anchor_modes),
        )
        self.assertEqual(
            [0.8, 0.2],
            list(cfg.domain_rand.force_anchor_mode_probs),
        )

        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.num_envs = 2
        env.device = "cpu"
        env.dt = 0.02
        env.sim_params = SimpleNamespace(dt=0.005)
        env.cfg = SimpleNamespace(
            commands=SimpleNamespace(command_base_height=0.54),
            domain_rand=SimpleNamespace(
                force_anchor_modes=["robot_relative_moving"],
                force_anchor_mode_probs=[1.0],
                force_anchor_velocity_range=[0.0, 0.02],
                force_anchor_motion_duration_s=[1.0, 3.0],
                force_anchor_offset_limit=[0.05, 0.05, 0.03],
            ),
        )
        env.robot_actor_idxs = torch.tensor([0, 1], dtype=torch.long)
        env.gripper_stator_index = 0
        env.root_states = torch.tensor(
            [
                [0.0, 0.0, 0.54, 0.0, 0.0, 0.0, 1.0] + [0.0] * 6,
                [0.0, 0.0, 0.54, 0.0, 0.0, 0.0, 1.0] + [0.0] * 6,
            ]
        )
        env.rigid_body_state = torch.zeros(2, 1, 13)
        env.rigid_body_state[:, 0, 0:3] = torch.tensor(
            [[1.0, 0.0, 0.54], [1.0, 0.0, 0.54]]
        )
        env.episode_length_buf = torch.zeros(2)
        env.force_anchor_local = torch.zeros(2, 3)
        env.force_anchor_latched_local = torch.zeros(2, 3)
        env.force_anchor_world = torch.zeros(2, 3)
        env.force_anchor_velocity_local = torch.zeros(2, 3)
        env.force_anchor_motion_end_step = torch.zeros(2)
        env.force_anchor_offset_limit = torch.tensor([0.05, 0.05, 0.03])
        env.force_anchor_displacement_world = torch.zeros(2, 3)
        env.force_anchor_spring_force_world = torch.zeros(2, 3)
        env.force_anchor_active = torch.zeros(2, dtype=torch.bool)
        env.force_anchor_latch_pending = torch.ones(2, dtype=torch.bool)
        env.force_anchor_mode = torch.full((2,), -1, dtype=torch.long)

        ids = torch.tensor([0, 1], dtype=torch.long)
        env._sample_force_anchor_modes(ids)
        torch.testing.assert_close(
            env.force_anchor_mode, torch.tensor([1, 1], dtype=torch.long)
        )
        env._latch_force_anchor(ids)
        local_before = env.force_anchor_local.clone()
        env._update_force_anchor_world()
        displacement = env.force_anchor_local - local_before
        self.assertTrue(torch.all(torch.linalg.norm(displacement, dim=1) <= 0.000101))
        self.assertTrue(torch.all(torch.linalg.norm(displacement, dim=1) > 0.0))


    def test_urdf_dependency_graph_and_root_are_repository_local(self):
        robot = ET.parse(URDF_PATH).getroot()
        joints = robot.findall("joint")
        child_links = {joint.find("child").get("link") for joint in joints}
        root_links = [
            link.get("name")
            for link in robot.findall("link")
            if link.get("name") not in child_links
        ]
        self.assertEqual(["BASE_LINK"], root_links)
        self.assertNotIn("world", root_links)

        movable = [
            joint.get("name") for joint in joints if joint.get("type") != "fixed"
        ]
        self.assertEqual(list(EXPECTED_DOF_NAMES), movable)
        for mesh in robot.findall(".//mesh"):
            filename = mesh.get("filename")
            self.assertFalse(Path(filename).is_absolute())
            self.assertTrue((URDF_PATH.parent / filename).resolve().is_file())

    def test_urdf_links_have_valid_inertial_and_collision_contract(self):
        robot = ET.parse(URDF_PATH).getroot()
        links = {link.get("name"): link for link in robot.findall("link")}
        for name, link in links.items():
            inertial = link.find("inertial")
            self.assertIsNotNone(inertial, name)
            self.assertGreater(float(inertial.find("mass").get("value")), 0.0)
            values = inertial.find("inertia").attrib
            inertia = np.array(
                [
                    [values["ixx"], values["ixy"], values["ixz"]],
                    [values["ixy"], values["iyy"], values["iyz"]],
                    [values["ixz"], values["iyz"], values["izz"]],
                ],
                dtype=float,
            )
            eigenvalues = np.linalg.eigvalsh(inertia)
            self.assertTrue(np.all(eigenvalues > 0.0), name)
            self.assertGreaterEqual(
                eigenvalues[0] + eigenvalues[1] + 1e-12,
                eigenvalues[2],
                name,
            )
        for name in FOOT_LINK_NAMES:
            self.assertGreater(len(links[name].findall("collision")), 0)
        self.assertGreater(len(links["BASE_LINK"].findall("collision")), 0)
        self.assertGreater(len(links["ROBOT_ARM_LINK7"].findall("collision")), 0)

        for joint in robot.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            limit = joint.find("limit")
            self.assertIsNotNone(limit, joint.get("name"))
            self.assertLess(float(limit.get("lower")), float(limit.get("upper")))
            self.assertGreater(float(limit.get("velocity")), 0.0)
            self.assertGreater(float(limit.get("effort")), 0.0)

    def test_mujoco_mesh_and_heightfield_dependencies_are_local(self):
        model = ET.parse(MODEL_ROOT / "zgwsarm.xml").getroot()
        mesh_dir = MODEL_ROOT / model.find("compiler").get("meshdir")
        for mesh in model.findall("./asset/mesh"):
            self.assertTrue((mesh_dir / mesh.get("file")).resolve().is_file())

        scene = ET.parse(MODEL_ROOT / "scene_terrain.xml").getroot()
        self.assertTrue((MODEL_ROOT / scene.find("include").get("file")).is_file())
        for heightfield in scene.findall("./asset/hfield"):
            self.assertTrue((MODEL_ROOT / heightfield.get("file")).is_file())

    def test_urdf_and_mujoco_mass_models_match(self):
        urdf = ET.parse(URDF_PATH).getroot()
        urdf_mass = sum(
            float(mass.get("value"))
            for mass in urdf.findall("./link/inertial/mass")
        )
        mujoco = ET.parse(MODEL_ROOT / "zgwsarm.xml").getroot()
        mujoco_mass = sum(
            float(inertial.get("mass"))
            for inertial in mujoco.findall(".//body/inertial")
            if inertial.get("mass") is not None
        )
        self.assertAlmostEqual(47.5524395, urdf_mass, places=6)
        self.assertAlmostEqual(urdf_mass, mujoco_mass, places=6)

    def test_config_dimensions_and_named_dof_groups(self):
        cfg = ZGWSARMComplianceCfg()
        self.assertEqual(22, cfg.env.num_actions)
        self.assertEqual(96, cfg.env.num_observations)
        self.assertEqual(16, cfg.env.num_privileged_obs)
        self.assertEqual(10, cfg.env.num_observation_history)
        self.assertEqual(23, cfg.commands.num_commands)
        expected_observations = 3 + 23 + 22 + 22 + 22 + 4
        self.assertEqual(expected_observations, cfg.env.num_observations)

        configured_names = (
            cfg.asset.leg_dof_names
            + cfg.asset.wheel_dof_names
            + cfg.asset.arm_dof_names
        )
        self.assertEqual(set(EXPECTED_DOF_NAMES), set(configured_names))
        self.assertEqual(12, len(LEG_DOF_NAMES))
        self.assertEqual(4, len(WHEEL_DOF_NAMES))
        self.assertEqual(6, len(ARM_DOF_NAMES))
        self.assertEqual(set(EXPECTED_DOF_NAMES), set(cfg.init_state.default_joint_angles))
        self.assertEqual("ROBOT_ARM_LINK7", cfg.asset.end_effector_name)
        self.assertEqual("BASE_LINK", cfg.asset.base_name)
        self.assertEqual(FOOT_LINK_NAMES, cfg.asset.foot_names)
        self.assertEqual(WHEEL_DOF_NAMES, cfg.asset.zero_position_observation_dof_names)
        self.assertEqual(ABAD_DOF_NAMES, cfg.asset.abad_dof_names)

        self.assertEqual([0.0, 0.0, 0.55], cfg.init_state.pos)
        self.assertEqual(
            [-0.10, 0.10], cfg.init_state.arm_reset_position_range
        )
        self.assertEqual(0.54, cfg.rewards.base_height_target)
        self.assertEqual(0.54, cfg.commands.command_base_height)
        self.assertEqual(0.002, cfg.sim.dt)
        self.assertEqual(5, cfg.control.decimation)
        self.assertEqual(0.01, cfg.sim.dt * cfg.control.decimation)
        self.assertEqual(4096, cfg.env.num_envs)
        self.assertEqual(4.0, cfg.normalization.clip_actions)
        self.assertEqual(0.4, cfg.control.abad_scale_reduction)
        self.assertEqual(1.0, cfg.control.arm_scale_reduction)
        self.assertEqual(1.0, cfg.control.arm_target_velocity_limit_scale)
        self.assertEqual(2.0, cfg.control.safety_dof_velocity_ratio)
        self.assertEqual(0.05, cfg.control.safety_dof_position_margin)
        self.assertEqual(5000.0, cfg.control.safety_nonfoot_contact_force)
        self.assertEqual(10.0, cfg.rewards.terminal_contact_force)
        self.assertEqual(2, cfg.rewards.terminal_contact_debounce_steps)
        self.assertEqual(4, cfg.domain_rand.lag_timesteps)
        self.assertEqual(2 ** 23, cfg.sim.physx.max_gpu_contact_pairs)
        self.assertEqual(3, cfg.sim.physx.default_buffer_size_multiplier)
        self.assertEqual(0.095, cfg.asset.wheel_radius)
        self.assertEqual(5.0, cfg.rewards.wheel_contact_force_threshold)
        self.assertEqual(20.0, cfg.rewards.wheel_min_support_force)
        self.assertEqual(
            [0.25, 0.45], cfg.rewards.wheel_support_front_x_range
        )
        self.assertEqual(
            [-0.45, -0.25], cfg.rewards.wheel_support_rear_x_range
        )
        self.assertEqual(
            [-0.32, -0.12], cfg.rewards.wheel_support_right_y_range
        )
        self.assertEqual(
            [0.12, 0.32], cfg.rewards.wheel_support_left_y_range
        )
        self.assertEqual(0.04, cfg.rewards.stance_lateral_symmetry_tolerance)
        self.assertEqual(0.04, cfg.rewards.stance_height_symmetry_tolerance)
        self.assertEqual(0.05, cfg.diagnostics.hard_limit_margin)
        self.assertIsNone(cfg.diagnostics.zgwsarm_scenario)
        self.assertEqual(0, cfg.asset.default_dof_drive_mode)
        self.assertFalse(cfg.asset.replace_cylinder_with_capsule)
        self.assertEqual(1, cfg.asset.self_collisions)
        self.assertEqual([90.0, 120.0, 120.0] * 4, cfg.commands.p_gains_legs)
        self.assertEqual([1.0, 1.0, 1.0] * 4, cfg.commands.d_gains_legs)
        self.assertEqual([60.0] * 4, cfg.commands.p_gains_wheels)
        self.assertEqual([0.2] * 4, cfg.commands.d_gains_wheels)
        self.assertTrue(cfg.rewards.only_positive_rewards)
        self.assertFalse(cfg.rewards.only_positive_rewards_ji22_style)
        self.assertEqual(0.02, cfg.rewards.sigma_rew_neg)
        self.assertEqual("yaw", cfg.rewards.force_command_frame)
        self.assertEqual(0.5, cfg.rewards.manip_ori_tracking_sigma)
        self.assertEqual(
            list(ZGWSARM_EE_RADIUS_RANGE), cfg.commands.ee_sphe_radius
        )
        self.assertEqual(
            list(ZGWSARM_EE_PITCH_RANGE), cfg.commands.ee_sphe_pitch
        )
        self.assertEqual(
            list(ZGWSARM_EE_YAW_RANGE), cfg.commands.ee_sphe_yaw
        )
        self.assertEqual(0.15, cfg.commands.ee_min_world_height)
        for axis in "xyz":
            self.assertEqual(
                list(ZGWSARM_FORCE_COMPONENT_RANGE),
                getattr(cfg.commands, f"ee_force_{axis}"),
            )
        self.assertEqual(-10.0, cfg.reward_scales.termination)
        self.assertEqual(-5.0, cfg.reward_scales.orientation)
        self.assertEqual(-5.0, cfg.reward_scales.base_height)
        self.assertEqual(0.0, cfg.reward_scales.dof_pos)
        self.assertEqual([0.0, 0.0], cfg.commands.lin_vel_x)
        self.assertEqual([0.0, 0.0], cfg.commands.limit_vel_x)
        self.assertEqual([0.0, 0.0], cfg.commands.lin_vel_y)
        self.assertEqual([0.0, 0.0], cfg.commands.limit_vel_y)
        self.assertEqual([-0.3, 0.3], cfg.commands.ang_vel_yaw)
        self.assertEqual([-0.3, 0.3], cfg.commands.limit_vel_yaw)
        self.assertFalse(cfg.commands.command_curriculum)
        self.assertEqual({"pure_yaw": 1.0}, cfg.commands.planar_command_mixture)
        self.assertEqual(0.05, cfg.commands.yaw_success_threshold)
        self.assertEqual("continuous_modes", cfg.commands.velocity_sampling)
        self.assertFalse(hasattr(cfg.commands, "num_bins_vel_x"))
        self.assertFalse(hasattr(cfg.commands, "num_bins_vel_y"))
        self.assertFalse(hasattr(cfg.commands, "num_bins_vel_yaw"))
        self.assertGreater(cfg.reward_scales.tracking_lin_vel, 0.0)
        self.assertGreater(cfg.reward_scales.tracking_ang_vel_yaw, 0.0)
        self.assertEqual(0.0, cfg.reward_scales.tracking_contacts_shaped_force)
        self.assertEqual(0.0, cfg.reward_scales.tracking_contacts_shaped_vel)
        self.assertEqual(0.0, cfg.reward_scales.feet_clearance_cmd)
        self.assertLess(cfg.reward_scales.wheel_contact_consistency, 0.0)
        self.assertLess(cfg.reward_scales.wheel_support_load, 0.0)
        self.assertLess(cfg.reward_scales.wheel_support_geometry, 0.0)
        self.assertFalse(hasattr(cfg.reward_scales, "wheel_lateral_slip"))
        self.assertLess(cfg.reward_scales.wheel_rolling_consistency, 0.0)
        self.assertGreater(cfg.reward_scales.wheel_v_tracking, 0.0)
        self.assertGreater(cfg.reward_scales.wheel_yaw_tracking, 0.0)
        self.assertLess(cfg.reward_scales.stance_posture, 0.0)
        self.assertEqual(-1.0, cfg.reward_scales.stance_posture)
        self.assertEqual(-2.0, cfg.reward_scales.stance_symmetry)
        self.assertEqual(-1.0, cfg.reward_scales.dof_pos_limits_leg)
        self.assertIn("BASE_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("ABAD_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("HIP_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("KNEE_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("ROBOT_ARM_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertEqual(
            list(ZGWSARM_FORCE_COMPONENT_RANGE),
            cfg.domain_rand.max_push_force_xyz_gripper,
        )
        self.assertEqual(
            list(ZGWSARM_FORCE_COMPONENT_RANGE),
            cfg.domain_rand.max_push_force_xyz_gripper_freed,
        )
        self.assertEqual("binary", cfg.commands.hybrid_mode)
        self.assertEqual(0.8, cfg.domain_rand.gripper_forced_prob)
        self.assertEqual(0.6, cfg.domain_rand.force_zero_command_prob)

    def test_yaw_mechanism_learning_config(self):
        cfg = ZGWSARMComplianceCfg()

        self.assertEqual(2.0, cfg.reward_scales.wheel_yaw_tracking)
        self.assertEqual(-1.5, cfg.reward_scales.yaw_leg_posture)
        self.assertEqual(-1.5, cfg.reward_scales.yaw_height_floor)
        self.assertEqual(1.0, cfg.reward_scales.yaw_gait_support)
        self.assertEqual(1.5, cfg.reward_scales.yaw_foothold_tracking)
        self.assertEqual(
            0.3, cfg.reward_scales.yaw_swing_tangential_velocity
        )
        self.assertEqual(0.10, cfg.rewards.tracking_sigma_v_yaw)
        self.assertEqual(0.15, cfg.rewards.wheel_yaw_tracking_scale)
        self.assertEqual(
            [0.22, 0.06, 0.08] * 4,
            cfg.rewards.yaw_leg_posture_allowed_deviation,
        )
        self.assertEqual(0.10, cfg.rewards.yaw_leg_posture_scale)
        self.assertEqual(0.03, cfg.rewards.yaw_height_allowed_drop)
        self.assertEqual(0.04, cfg.rewards.yaw_height_scale)
        self.assertEqual(
            [0.0, 0.0, 0.15],
            cfg.commands.min_active_command_magnitude,
        )
        self.assertEqual(
            {
                "stand": 0.1,
                "pure_x": 0.0,
                "pure_y": 0.0,
                "pure_yaw": 0.9,
                "xy": 0.0,
                "x_yaw": 0.0,
                "y_yaw": 0.0,
                "full": 0.0,
            },
            cfg.commands.planar_command_mixture,
        )
        self.assertEqual([-0.3, 0.3], cfg.commands.ang_vel_yaw)
        self.assertEqual(1.2, cfg.rewards.yaw_gait_frequency)
        self.assertEqual(0.05, cfg.rewards.yaw_gait_step_length)
        self.assertEqual(0.04, cfg.rewards.yaw_gait_foothold_sigma)
        self.assertEqual(30.0, cfg.rewards.yaw_gait_stance_min_force)
        self.assertEqual(8.0, cfg.rewards.yaw_gait_stance_force_scale)
        self.assertEqual(10.0, cfg.rewards.yaw_gait_swing_force_scale)
        self.assertEqual(
            ["FAR", "RBL", "FBL", "RAR"],
            cfg.rewards.yaw_gait_phase_order,
        )
        self.assertEqual("plane", cfg.terrain.mesh_type)
        self.assertEqual("position", cfg.commands.hybrid_mode)
        self.assertEqual([0.0, 0.0], cfg.init_state.arm_reset_position_range)
        for axis in "xyz":
            self.assertEqual(
                [0.0, 0.0], getattr(cfg.commands, f"ee_force_{axis}")
            )
        self.assertFalse(cfg.domain_rand.randomize_com_displacement)
        self.assertFalse(cfg.domain_rand.randomize_gravity)
        self.assertFalse(cfg.domain_rand.randomize_motor_strength)
        self.assertFalse(cfg.domain_rand.randomize_Kd_factor)
        self.assertFalse(cfg.domain_rand.push_robots)
        self.assertFalse(cfg.domain_rand.randomize_tile_roughness)
        self.assertEqual([0.0, 0.0], cfg.domain_rand.tile_roughness_range)

    def test_initial_velocity_curriculum_samples_stay_inside_active_ranges(self):
        cfg = ZGWSARMComplianceCfg()
        _, curricula = build_command_curricula(cfg.commands)

        samples, _ = curricula[0].sample(batch_size=4096)
        expected_ranges = (
            cfg.commands.lin_vel_x,
            cfg.commands.lin_vel_y,
            cfg.commands.ang_vel_yaw,
        )
        for command_index, value_range in enumerate(expected_ranges):
            self.assertTrue(np.all(samples[:, command_index] >= value_range[0]))
            self.assertTrue(np.all(samples[:, command_index] <= value_range[1]))

    def test_config_does_not_build_from_the_b1_z1_task(self):
        with patch(
            "wbc_compliance_gym.envs.b1_z1_compliance.b1_z1_config."
            "configure_b1_z1_ik",
            side_effect=AssertionError("ZGWSARM must not call the B1+Z1 builder"),
        ):
            cfg = ZGWSARMComplianceCfg()

        self.assertEqual("zgwsarm", cfg.robot.name)
        self.assertEqual(ZGWSARM_REWARD_SCALES, active_reward_scales(cfg))

    def test_active_rewards_are_an_explicit_zgwsarm_manifest(self):
        cfg = ZGWSARMComplianceCfg()
        active = active_reward_scales(cfg)
        self.assertEqual(ZGWSARM_REWARD_SCALES, active)
        self.assertNotIn("raibert_heuristic", active)
        self.assertNotIn(
            "tracking_contacts_shaped_force", active
        )
        self.assertNotIn(
            "tracking_contacts_shaped_vel", active
        )
        self.assertNotIn("feet_clearance_cmd", active)
        for reward_name in active:
            self.assertTrue(
                hasattr(ZGWSARMRewards, f"_reward_{reward_name}"),
                reward_name,
            )

    def test_play_keeps_explicit_force_override(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_compliance_play(
            cfg, control_mode="force", force_amplitude=45.0
        )

        self.assertEqual([-45.0, 45.0], cfg.domain_rand.max_push_force_xyz_gripper)
        self.assertEqual(
            [-45.0, 45.0],
            cfg.domain_rand.max_push_force_xyz_gripper_freed,
        )
        for axis in "xyz":
            self.assertEqual(
                [-45.0, 45.0], getattr(cfg.commands, f"ee_force_{axis}")
            )
        self.assertEqual([0.0, 0.0], cfg.commands.lin_vel_x)

    def test_play_can_fix_the_base_for_force_tracking_diagnostics(self):
        cfg = ZGWSARMComplianceCfg()
        self.assertFalse(cfg.asset.fix_base_link)

        configure_zgwsarm_compliance_play(cfg, fix_base=True)

        self.assertTrue(cfg.asset.fix_base_link)

    def test_play_uses_task_owned_default_position_mode(self):
        cfg = ZGWSARMComplianceCfg()
        self.assertEqual("position", cfg.commands.hybrid_mode)

        configure_zgwsarm_compliance_play(cfg)

        self.assertEqual("position", cfg.commands.hybrid_mode)

    def test_play_without_force_override_uses_task_owned_xyz_target(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_compliance_play(cfg, control_mode="force")

        self.assertEqual([10.0, 10.0], cfg.commands.ee_force_x)
        self.assertEqual([0.0, 0.0], cfg.commands.ee_force_y)
        self.assertEqual([0.0, 0.0], cfg.commands.ee_force_z)
        self.assertEqual(
            list(ZGWSARM_FORCE_COMPONENT_RANGE),
            cfg.domain_rand.max_push_force_xyz_gripper_freed,
        )

    def test_diagnostic_zero_action_is_deterministic_and_force_free(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_compliance_play(
            cfg, diagnostic_scenario="zero_action"
        )

        self.assertEqual("zero_action", cfg.diagnostics.zgwsarm_scenario)
        self.assertEqual(0.0, cfg.terrain.yaw_init_range)
        self.assertEqual("position", cfg.commands.hybrid_mode)
        self.assertEqual([0.0, 0.0], cfg.commands.lin_vel_x)
        self.assertEqual([0.0, 0.0], cfg.commands.ang_vel_yaw)
        self.assertEqual([0.0, 0.0], cfg.domain_rand.max_push_force_xyz_gripper)
        self.assertEqual(
            [0.0, 0.0], cfg.domain_rand.max_push_force_xyz_gripper_freed
        )
        for axis in "xyz":
            self.assertEqual(
                [0.0, 0.0], getattr(cfg.commands, f"ee_force_{axis}")
            )
        expected_dofs = set(LEG_DOF_NAMES + WHEEL_DOF_NAMES + ARM_DOF_NAMES)
        self.assertEqual(expected_dofs, set(cfg.asset.fixed_action_targets))
        self.assertTrue(all(
            value == 0.0 for value in cfg.asset.fixed_action_targets.values()
        ))
        self.assertFalse(cfg.domain_rand.randomize_lag_timesteps)
        self.assertFalse(cfg.domain_rand.push_robots)
        self.assertFalse(cfg.noise.add_noise)
        # The legacy force controller initializes gains through this flag. Its
        # degenerate range is deterministic rather than randomized.
        self.assertTrue(cfg.domain_rand.randomize_gripper_force_gains)
        self.assertEqual(
            cfg.domain_rand.gripper_force_kp_range[0],
            cfg.domain_rand.gripper_force_kp_range[1],
        )
        radius, pitch, yaw = cfg.commands.default_ee_position_spherical
        self.assertEqual([radius, radius], cfg.commands.ee_sphe_radius)
        self.assertEqual([pitch, pitch], cfg.commands.ee_sphe_pitch)
        self.assertEqual([yaw, yaw], cfg.commands.ee_sphe_yaw)
        self.assertEqual(
            [0.0, 0.0], cfg.init_state.arm_reset_position_range
        )

    def test_diagnostic_velocity_scenario_fixes_only_the_arm(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_diagnostic_play(
            cfg,
            "velocity_arm_fixed",
            lin_vel_x=0.7,
            ang_vel_yaw=-0.2,
        )

        self.assertEqual([0.7, 0.7], cfg.commands.lin_vel_x)
        self.assertEqual([-0.2, -0.2], cfg.commands.ang_vel_yaw)
        self.assertEqual(
            set(ARM_DOF_NAMES), set(cfg.asset.fixed_action_targets)
        )
        self.assertTrue(all(
            name not in cfg.asset.fixed_action_targets
            for name in LEG_DOF_NAMES + WHEEL_DOF_NAMES
        ))
        self.assertEqual(
            [0.0, 0.0], cfg.init_state.arm_reset_position_range
        )

    def test_diagnostic_position_scenario_moves_arm_without_force(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_diagnostic_play(cfg, "position_arm")

        self.assertEqual("position", cfg.commands.hybrid_mode)
        self.assertEqual({}, cfg.asset.fixed_action_targets)
        self.assertEqual(
            list(ZGWSARM_EE_RADIUS_RANGE), cfg.commands.ee_sphe_radius
        )
        self.assertEqual(
            list(ZGWSARM_EE_PITCH_RANGE), cfg.commands.ee_sphe_pitch
        )
        self.assertEqual(
            list(ZGWSARM_EE_YAW_RANGE), cfg.commands.ee_sphe_yaw
        )
        self.assertEqual([0.0, 0.0], cfg.domain_rand.max_push_force_xyz_gripper)
        for axis in "xyz":
            self.assertEqual(
                [0.0, 0.0], getattr(cfg.commands, f"ee_force_{axis}")
            )

    def test_diagnostic_force_scenario_supports_low_and_full_load(self):
        self.assertEqual(
            ("zero_action", "velocity_arm_fixed", "position_arm", "force"),
            ZGWSARM_DIAGNOSTIC_SCENARIOS,
        )
        for amplitude in (10.0, 70.0):
            cfg = ZGWSARMComplianceCfg()
            configure_zgwsarm_compliance_play(
                cfg,
                diagnostic_scenario="force",
                force_amplitude=amplitude,
            )
            expected = [-amplitude, amplitude]
            self.assertEqual("force", cfg.commands.hybrid_mode)
            for axis in "xyz":
                self.assertEqual(
                    expected, getattr(cfg.commands, f"ee_force_{axis}")
                )
            self.assertEqual(
                expected, cfg.domain_rand.max_push_force_xyz_gripper
            )
            self.assertEqual(
                expected, cfg.domain_rand.max_push_force_xyz_gripper_freed
            )

    def test_locomotion_diagnostic_tensors_use_semantic_dof_groups(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.device = "cpu"
        env.num_envs = 1
        env.dof_names = list(EXPECTED_DOF_NAMES)
        env.dof_name_to_index = {
            name: index for index, name in enumerate(EXPECTED_DOF_NAMES)
        }
        env.leg_dof_indices = torch.tensor(
            [env.dof_name_to_index[name] for name in LEG_DOF_NAMES]
        )
        env.wheel_dof_indices = torch.tensor(
            [env.dof_name_to_index[name] for name in WHEEL_DOF_NAMES]
        )
        env.feet_indices = torch.tensor([0, 1, 2, 3])
        env.cfg = SimpleNamespace(
            asset=SimpleNamespace(
                leg_dof_names=list(LEG_DOF_NAMES),
                wheel_dof_names=list(WHEEL_DOF_NAMES),
                wheel_radius=0.095,
            ),
            rewards=SimpleNamespace(wheel_contact_force_threshold=1.0),
            normalization=SimpleNamespace(clip_actions=4.0),
            diagnostics=SimpleNamespace(hard_limit_margin=0.05),
        )
        env.dof_pos = torch.zeros(1, 22)
        env.dof_vel = torch.zeros(1, 22)
        env.joint_pos_target = torch.zeros(1, 22)
        env.default_dof_pos = torch.zeros(1, 22)
        env.actions = torch.zeros(1, 22)
        env.dof_pos_hard_limits = torch.tensor([[-2.0, 2.0]] * 22)

        far_abad_index = env.dof_name_to_index["FAR_ABAD_JOINT"]
        far_wheel_index = env.dof_name_to_index["FAR_FOOT_JOINT"]
        env.dof_pos[0, far_abad_index] = 0.96
        env.joint_pos_target[0, far_abad_index] = 0.8
        env.actions[0, far_abad_index] = 4.0
        env.dof_pos_hard_limits[far_abad_index] = torch.tensor([-1.0, 1.0])
        env.dof_vel[0, far_wheel_index] = 10.0

        env.base_pos = torch.tensor([[1.0, 2.0, 0.5]])
        env.base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        env.base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
        env.rigid_body_state = torch.zeros(1, 4, 13)
        env.rigid_body_state[0, 0, 0:3] = torch.tensor([1.3, 1.8, 0.1])
        env.rigid_body_state[0, 0, 7:10] = torch.tensor([1.0, 0.2, 0.0])
        env.contact_forces = torch.zeros(1, 4, 3)
        env.contact_forces[0, 0, 2] = 10.0

        metrics = env.get_zgwsarm_diagnostic_tensors()

        torch.testing.assert_close(
            metrics["leg/FAR/ABAD/position_rad"], torch.tensor([0.96])
        )
        torch.testing.assert_close(
            metrics["leg/FAR/ABAD/hard_limit_dwell"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            metrics["leg/FAR/ABAD/action_saturation"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            metrics["wheel/FAR/contact"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            metrics["wheel/FAR/rolling_speed_mps"], torch.tensor([0.95])
        )
        torch.testing.assert_close(
            metrics["wheel/FAR/rolling_residual_contact_mps"],
            torch.tensor([0.05]),
        )
        torch.testing.assert_close(
            metrics["wheel/FAR/base_speed_residual_mps"], torch.tensor([0.45])
        )
        torch.testing.assert_close(
            metrics["wheel/FAR/base_position_x_m"], torch.tensor([0.3])
        )
        torch.testing.assert_close(
            metrics["wheel/FAR/base_position_y_m"], torch.tensor([-0.2])
        )

    def test_task_and_reward_registration(self):
        registry = register_tasks()
        self.assertIn("zgwsarm_compliance", registry.names())
        self.assertNotIn("zgwsarm", registry.names())
        spec = registry.get_spec("zgwsarm_compliance")
        self.assertIsNotNone(spec.play_cfg_hook)
        self.assertEqual(
            "zgwsarm_compliance", ZGWSARMComplianceCfgPPO().run.task_name
        )
        self.assertIs(ZGWSARMRewards, REWARD_CONTAINERS["ZGWSARMRewards"])

    def test_training_launch_defaults_are_owned_by_zgwsarm_config(self):
        cfg = ZGWSARMComplianceCfgPPO()

        self.assertEqual(48, cfg.runner.num_steps_per_env)
        self.assertEqual(10000, cfg.runner.max_iterations)
        self.assertEqual(500, cfg.runner.save_interval)
        self.assertEqual(0, cfg.runner.save_video_interval)
        self.assertEqual(1, cfg.runner.log_freq)
        self.assertEqual(1.0e-3, cfg.algorithm.learning_rate)
        self.assertEqual(0.005, cfg.algorithm.entropy_coef)
        self.assertEqual("yaw_sideswing_ablation", cfg.run.training_name)
        self.assertFalse(cfg.run.resume)
        self.assertIsNone(cfg.run.resume_run_dir)
        self.assertEqual("latest", cfg.run.resume_checkpoint)

    def test_reward_container_does_not_inherit_b1_gait_semantics(self):
        self.assertTrue(issubclass(ZGWSARMRewards, WholeBodyComplianceRewards))
        self.assertTrue(issubclass(B1Z1Rewards, WholeBodyComplianceRewards))
        self.assertFalse(issubclass(ZGWSARMRewards, B1Z1Rewards))
        self.assertFalse(hasattr(ZGWSARMRewards, "_reward_raibert_heuristic"))
        self.assertFalse(
            hasattr(ZGWSARMRewards, "_reward_tracking_contacts_shaped_force")
        )

    def test_specialized_wheel_torque_formula_and_effort_clipping(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.device = "cpu"
        env.num_envs = 1
        env.num_dof = 22
        env.num_actions = 22
        name_to_index = {
            name: index for index, name in enumerate(EXPECTED_DOF_NAMES)
        }
        env.leg_dof_indices = torch.tensor(
            [name_to_index[name] for name in LEG_DOF_NAMES]
        )
        env.wheel_dof_indices = torch.tensor(
            [name_to_index[name] for name in WHEEL_DOF_NAMES]
        )
        env.arm_dof_indices = torch.tensor(
            [name_to_index[name] for name in ARM_DOF_NAMES]
        )
        env.hip_dof_indices = torch.tensor([0, 4, 8, 12])
        env.abad_dof_indices = torch.tensor([0, 4, 8, 12])
        env.cfg = SimpleNamespace(
            control=SimpleNamespace(
                control_type="P",
                action_scale=0.25,
                abad_scale_reduction=0.4,
                hip_scale_reduction=1.0,
                arm_scale_reduction=1.0,
                arm_target_velocity_limit_scale=1.0,
            ),
            sim=SimpleNamespace(dt=0.002),
            domain_rand=SimpleNamespace(randomize_lag_timesteps=False),
        )
        env.default_dof_pos = torch.zeros(1, 22)
        env.dof_pos = torch.zeros(1, 22)
        env.dof_vel = torch.zeros(1, 22)
        env.joint_pos_target = torch.zeros(1, 22)
        env.p_gains = torch.zeros(22)
        env.d_gains = torch.zeros(22)
        env.p_gains[env.leg_dof_indices] = torch.tensor(
            [90.0, 120.0, 120.0] * 4
        )
        env.d_gains[env.leg_dof_indices] = 1.0
        env.p_gains[env.wheel_dof_indices] = 60.0
        env.d_gains[env.wheel_dof_indices] = 0.2
        env.p_gains[env.arm_dof_indices] = torch.tensor(
            [64.0, 128.0, 64.0, 64.0, 64.0, 64.0]
        )
        env.d_gains[env.arm_dof_indices] = torch.tensor(
            [1.5, 3.0, 1.5, 1.5, 1.5, 1.5]
        )
        env.Kp_factors = torch.ones(1, 1)
        env.Kd_factors = torch.ones(1, 1)
        env.torque_limits = torch.tensor(
            [180.0, 180.0, 180.0, 28.0] * 4 + [100.0] * 6
        )
        env.dof_vel_limits = torch.tensor(
            [16.75, 16.75, 16.75, 110.0] * 4
            + [10.0, 10.0, 10.0, 10.0, 10.0, 5.0]
        )
        env.dof_pos_hard_limits = torch.tensor(
            [[-100.0, 100.0]] * env.num_dof
        )
        env.dof_pos_hard_limits[17] = torch.tensor([-1.0, 1.0])
        env.dof_limited_indices = torch.tensor(
            [i for i in range(env.num_dof) if i not in env.wheel_dof_indices]
        )
        env._cuda_debugger = None

        actions = torch.zeros(1, 22)
        actions[0, 0] = 1.0
        actions[0, 3] = 2.0
        actions[0, 16] = 1.0
        actions[0, 17] = 10.0
        env.dof_vel[0, 3] = 10.0
        env.dof_pos[0, 7] = 123.0
        env._compute_torques(actions)

        self.assertAlmostEqual(9.0, env.torques[0, 0].item())
        self.assertAlmostEqual(28.0, env.torques[0, 3].item())
        self.assertAlmostEqual(0.0, env.torques[0, 7].item())
        self.assertAlmostEqual(0.02, env.joint_pos_target[0, 16].item())
        self.assertAlmostEqual(0.02, env.joint_pos_target[0, 17].item())

    def test_pathological_states_are_selected_for_reset(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.device = "cpu"
        env.num_bodies = 3
        env.feet_indices = torch.tensor([2])
        env.cfg = SimpleNamespace(
            control=SimpleNamespace(
                safety_dof_velocity_ratio=2.0,
                safety_dof_position_margin=0.05,
                safety_nonfoot_contact_force=5000.0,
            )
        )
        env.dof_vel = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 20.1, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        env.dof_vel_limits = torch.tensor([5.0, 10.0, 100.0])
        env.dof_pos = torch.tensor(
            [
                [0.0, 0.0, 500.0],
                [0.0, 0.0, -500.0],
                [1.06, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        env.dof_pos_hard_limits = torch.tensor(
            [[-1.0, 1.0], [-2.0, 2.0], [-99999.0, 99999.0]]
        )
        env.dof_limited_indices = torch.tensor([0, 1])
        env.contact_forces = torch.zeros(4, 3, 3)
        env.contact_forces[0, 2, 0] = 10000.0
        env.contact_forces[3, 1, 0] = 5000.1

        velocity_reset, position_reset, contact_reset = env._safety_reset_masks()

        self.assertTrue(torch.equal(
            velocity_reset, torch.tensor([False, True, False, False])
        ))
        self.assertTrue(torch.equal(
            position_reset, torch.tensor([False, False, True, False])
        ))
        self.assertTrue(torch.equal(
            contact_reset, torch.tensor([False, False, False, True])
        ))

    def test_semantic_contacts_are_low_force_and_debounced(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.device = "cpu"
        env.num_envs = 3
        env.termination_contact_indices = torch.tensor([0, 1])
        env.cfg = SimpleNamespace(
            rewards=SimpleNamespace(
                terminal_contact_force=10.0,
                terminal_contact_debounce_steps=2,
            )
        )
        env.contact_forces = torch.zeros(3, 3, 3)
        env.contact_forces[0, 0, 2] = 11.0
        env.contact_forces[1, 2, 2] = 1000.0

        first = env._semantic_contact_reset_mask()
        second = env._semantic_contact_reset_mask()
        env.contact_forces.zero_()
        cleared = env._semantic_contact_reset_mask()

        self.assertTrue(torch.equal(first, torch.tensor([False, False, False])))
        self.assertTrue(torch.equal(second, torch.tensor([True, False, False])))
        self.assertTrue(torch.equal(cleared, torch.tensor([False, False, False])))

    def test_base_height_is_measured_relative_to_local_terrain(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.cfg = SimpleNamespace(
            terrain=SimpleNamespace(mesh_type="boxes_tm")
        )
        env.num_envs = 2
        env.base_pos = torch.tensor(
            [[1.0, 2.0, 0.70], [3.0, 4.0, 1.10]], dtype=torch.float
        )
        env.get_heights_points = lambda positions: torch.tensor([0.20, 0.55])

        torch.testing.assert_close(
            env.base_height_above_terrain(), torch.tensor([0.50, 0.55])
        )

    def test_stance_posture_is_active_only_for_stand_mode(self):
        cfg = ZGWSARMComplianceCfg()
        self.assertIn("stance_posture", active_reward_scales(cfg))
        env = SimpleNamespace(
            commands=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.5, 0.0, 0.0]]
            ),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["stand"],
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["pure_x"],
                ]
            ),
            dof_pos=torch.ones(3, 12),
            default_dof_pos=torch.zeros(3, 12),
            leg_dof_indices=torch.arange(12),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(stand_still_command_threshold=0.1)
            ),
        )
        reward = ZGWSARMRewards(env)._reward_stance_posture()
        torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0]))

    def test_stance_posture_fallback_checks_yaw_command(self):
        env = SimpleNamespace(
            commands=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]
            ),
            dof_pos=torch.ones(2, 12),
            default_dof_pos=torch.zeros(2, 12),
            leg_dof_indices=torch.arange(12),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(stand_still_command_threshold=0.1)
            ),
        )
        reward = ZGWSARMRewards(env)._reward_stance_posture()
        torch.testing.assert_close(reward, torch.tensor([1.0, 0.0]))

    def test_yaw_leg_posture_has_joint_deadzone_and_pure_yaw_mask(self):
        allowed = torch.tensor([0.22, 0.06, 0.08] * 4)
        positions = torch.stack((allowed, allowed, allowed))
        positions[1:, 1] += 0.10
        env = SimpleNamespace(
            commands=torch.tensor(
                [[0.0, 0.0, 0.2], [0.0, 0.0, 0.2], [0.2, 0.0, 0.2]]
            ),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["x_yaw"],
                ]
            ),
            dof_pos=positions,
            default_dof_pos=torch.zeros(3, 12),
            leg_dof_indices=torch.arange(12),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    yaw_leg_posture_allowed_deviation=allowed.tolist(),
                    yaw_leg_posture_scale=0.10,
                )
            ),
        )

        reward = ZGWSARMRewards(env)._reward_yaw_leg_posture()
        torch.testing.assert_close(
            reward, torch.tensor([0.0, 1.0 / 12.0, 0.0])
        )

    def test_yaw_height_floor_penalizes_only_low_pure_yaw_base(self):
        env = SimpleNamespace(
            commands=torch.tensor(
                [[0.0, 0.0, 0.2], [0.0, 0.0, 0.2], [0.2, 0.0, 0.2]]
            ),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["x_yaw"],
                ]
            ),
            base_height_above_terrain=lambda: torch.tensor([0.51, 0.47, 0.47]),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    base_height_target=0.54,
                    yaw_height_allowed_drop=0.03,
                    yaw_height_scale=0.04,
                )
            ),
        )

        reward = ZGWSARMRewards(env)._reward_yaw_height_floor()
        torch.testing.assert_close(reward, torch.tensor([0.0, 1.0, 0.0]))

    def test_yaw_crawl_clock_and_reward_share_gait_phase(self):
        cfg = ZGWSARMComplianceCfg()
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            dt=0.1,
            cfg=cfg,
            commands=torch.zeros(2, 23),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["stand"],
                ]
            ),
            gait_indices=torch.zeros(2),
            foot_indices=torch.zeros(2, 4),
            clock_inputs=torch.zeros(2, 4),
            doubletime_clock_inputs=torch.zeros(2, 4),
            halftime_clock_inputs=torch.zeros(2, 4),
            desired_contact_states=torch.zeros(2, 4),
        )
        env.commands[:, 4] = 2.2
        env.commands[:, 8] = 0.5
        env.commands[0, 2] = 0.15

        CommandLifecycleMixin._step_contact_targets(env)

        torch.testing.assert_close(env.gait_indices, torch.tensor([0.12, 0.22]))
        self.assertAlmostEqual(1.2, env.commands[0, 4].item())
        self.assertFalse(torch.all(env.clock_inputs[0] == 1.0))
        torch.testing.assert_close(env.clock_inputs[1], torch.ones(4))
        torch.testing.assert_close(
            env.desired_contact_states[0], torch.tensor([0.0, 1.0, 1.0, 1.0])
        )
        self.assertEqual(
            (0, 2, 3, 1),
            resolve_yaw_gait_phase_slots(
                cfg.asset.wheel_dof_names,
                cfg.rewards.yaw_gait_phase_order,
            ),
        )

    def test_yaw_gait_rewards_support_and_mirror_footholds(self):
        cfg = ZGWSARMComplianceCfg()
        nominal_xy = torch.tensor(
            [
                [0.35, -0.22],
                [0.35, 0.22],
                [-0.35, -0.22],
                [-0.35, 0.22],
            ]
        )
        positions = torch.zeros(3, 4, 3)
        positions[:, :, :2] = nominal_xy
        velocities = torch.zeros(3, 4, 3)
        normal_forces = torch.tensor(
            [[0.0, 40.0, 40.0, 40.0], [0.0, 40.0, 40.0, 40.0], [0.0] * 4]
        )
        state = {
            "positions_base": positions,
            "velocities_base": velocities,
            "normal_forces": normal_forces,
            "contact": normal_forces > cfg.rewards.wheel_contact_force_threshold,
        }
        env = SimpleNamespace(
            num_envs=3,
            commands=torch.tensor(
                [[0.0, 0.0, 0.2], [0.0, 0.0, -0.2], [0.2, 0.0, 0.0]]
            ),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["pure_x"],
                ]
            ),
            gait_indices=torch.full((3,), 0.125),
            cfg=cfg,
            get_wheel_kinematics=lambda: state,
            get_wheel_command_kinematics=lambda: {
                "yaw_hat": torch.zeros(3),
            },
            leg_dof_indices=torch.arange(12),
            abad_dof_indices=torch.tensor([0, 3, 6, 9]),
            dof_pos=torch.zeros(3, 12),
            default_dof_pos=torch.zeros(3, 12),
            base_ang_vel=torch.zeros(3, 3),
            base_height_above_terrain=lambda: torch.full((3,), 0.54),
        )
        rewards = ZGWSARMRewards(env)
        targets = rewards._get_yaw_target_footholds(positions[:, :, :2])
        torch.testing.assert_close(
            targets[0] - nominal_xy, -(targets[1] - nominal_xy)
        )
        positions[:2] = torch.cat(
            (targets[:2], torch.zeros(2, 4, 1)), dim=2
        )
        tangent = rewards._get_yaw_tangent_vectors(nominal_xy)
        velocities[0, 0, :2] = 0.2 * tangent[0]
        velocities[1, 0, :2] = -0.2 * tangent[0]

        support = rewards._reward_yaw_gait_support()
        expected_support = (
            1.0 + 3.0 * torch.sigmoid(torch.tensor((40.0 - 30.0) / 8.0))
        ) / 4.0
        torch.testing.assert_close(
            support, torch.tensor([expected_support, expected_support, 0.0])
        )
        torch.testing.assert_close(
            rewards._reward_yaw_foothold_tracking(),
            torch.tensor([1.0, 1.0, 0.0]),
        )
        expected_velocity = torch.tanh(torch.tensor(1.0))
        torch.testing.assert_close(
            rewards._reward_yaw_swing_tangential_velocity(),
            torch.tensor([expected_velocity, expected_velocity, 0.0]),
        )
        torch.testing.assert_close(
            rewards._reward_wheel_contact_consistency(),
            torch.tensor([0.0, 0.0, 1.0]),
        )
        torch.testing.assert_close(
            rewards._reward_wheel_support_load(),
            torch.tensor([0.0, 0.0, 1.0]),
        )
        diagnostics = rewards.get_yaw_gait_diagnostics()
        for name in (
            "body_yaw_error",
            "wheel_yaw_error",
            "base_height",
            "mean_leg_posture_error",
            "contact_count",
            "swing_foot_normal_force",
            "stance_feet_normal_force",
            "foothold_tracking_error",
            "abad_max_deviation",
            "FAR_contact",
            "FBL_contact",
            "RAR_contact",
            "RBL_contact",
        ):
            self.assertIn(name, diagnostics)
        torch.testing.assert_close(
            diagnostics["contact_count"], torch.tensor([3.0, 3.0, 0.0])
        )

    def test_leg_velocity_regularizers_exclude_wheel_dofs(self):
        env = SimpleNamespace(
            leg_dof_indices=torch.tensor([0, 2]),
            motion_dof_indices=torch.arange(4),
            dof_vel=torch.tensor([[1.0, 100.0, 2.0, 100.0]]),
            last_dof_vel=torch.zeros(1, 4),
            dt=1.0,
        )
        rewards = ZGWSARMRewards(env)

        torch.testing.assert_close(
            rewards._reward_dof_vel_leg(), torch.tensor([5.0])
        )
        torch.testing.assert_close(
            rewards._reward_dof_acc_leg(), torch.tensor([5.0])
        )

    def test_continuous_velocity_sampler_produces_structured_modes(self):
        cfg = ZGWSARMComplianceCfg()
        cfg.commands.command_curriculum = True
        curriculum = ContinuousVelocityCurriculum(cfg.commands)
        generator = torch.Generator().manual_seed(17)
        commands, mode_ids = curriculum.sample(
            100000, "cpu", generator=generator
        )

        expected = {"stand": 0.1, "pure_yaw": 0.9}
        for name in LOCOMOTION_MODE_NAMES:
            selected = mode_ids == LOCOMOTION_MODE_TO_ID[name]
            frequency = selected.float().mean().item()
            self.assertAlmostEqual(expected.get(name, 0.0), frequency, delta=0.01)
            if name == "stand":
                self.assertTrue(torch.all(commands[selected] == 0.0))
            elif name == "pure_x":
                self.assertTrue(torch.all(commands[selected, 1:] == 0.0))
            elif name == "pure_yaw":
                self.assertTrue(torch.all(commands[selected, :2] == 0.0))
            elif name == "x_yaw":
                self.assertTrue(torch.all(commands[selected, 1] == 0.0))
        self.assertEqual(0.0, commands[:, 0].abs().max().item())
        self.assertLessEqual(commands[:, 2].abs().max().item(), 0.3)
        active_yaw = mode_ids == LOCOMOTION_MODE_TO_ID["pure_yaw"]
        self.assertGreaterEqual(
            commands[active_yaw, 2].abs().min().item(), 0.15
        )
        self.assertTrue(torch.any(commands[active_yaw, 2] < 0.0))
        self.assertTrue(torch.any(commands[active_yaw, 2] > 0.0))

    def test_minimum_active_magnitude_ignores_inactive_axis(self):
        cfg = ZGWSARMComplianceCfg()
        cfg.commands.ang_vel_yaw = [0.0, 0.0]
        cfg.commands.planar_command_mixture = {"stand": 1.0}
        curriculum = ContinuousVelocityCurriculum(cfg.commands)

        commands, mode_ids = curriculum.sample(64, "cpu")
        self.assertTrue(torch.all(commands == 0.0))
        self.assertTrue(
            torch.all(mode_ids == LOCOMOTION_MODE_TO_ID["stand"])
        )

    def test_zgwsarm_legacy_grid_has_no_velocity_cartesian_product(self):
        cfg = ZGWSARMComplianceCfg()
        _, curricula = build_command_curricula(cfg.commands)
        curriculum = curricula[0]
        for key in ("x_vel", "y_vel", "yaw_vel"):
            axis = curriculum.keys.index(key)
            self.assertEqual(1, curriculum.ls[key])
            self.assertTrue(np.all(curriculum.grid[axis, :] == 0.0))

    def test_continuous_velocity_ranges_expand_independently_and_cap(self):
        cfg = ZGWSARMComplianceCfg()
        cfg.commands.command_curriculum = True
        cfg.commands.lin_vel_x = [-0.5, 0.5]
        cfg.commands.limit_vel_x = [-1.0, 1.0]
        cfg.commands.ang_vel_yaw = [-0.2, 0.2]
        cfg.commands.limit_vel_yaw = [-1.5, 1.5]
        cfg.commands.planar_command_mixture = {
            "pure_x": 0.5,
            "pure_yaw": 0.5,
        }
        cfg.commands.velocity_curriculum_min_samples = 2
        cfg.commands.velocity_curriculum_required_successes = 1
        cfg.commands.velocity_curriculum_ema_alpha = 1.0
        curriculum = ContinuousVelocityCurriculum(cfg.commands)
        curriculum.record([0.1, 0.0, 1.0], [2, 0, 2], [0.1, 0.0, 1.0], [2, 0, 2])
        self.assertEqual([-0.6, 0.6], curriculum.current_ranges[0])
        self.assertEqual([-0.2, 0.2], curriculum.current_ranges[2])
        for _ in range(20):
            curriculum.expand(0)
        self.assertEqual([-1.0, 1.0], curriculum.current_ranges[0])

    def test_disabled_continuous_curriculum_samples_final_ranges(self):
        cfg = ZGWSARMComplianceCfg()
        cfg.commands.command_curriculum = False
        curriculum = ContinuousVelocityCurriculum(cfg.commands)
        self.assertEqual([[0.0, 0.0], [0.0, 0.0], [-0.3, 0.3]], curriculum.current_ranges)

    def test_continuous_curriculum_runtime_state_round_trips(self):
        cfg = ZGWSARMComplianceCfg()
        cfg.commands.command_curriculum = True
        source = ContinuousVelocityCurriculum(cfg.commands)
        source.expand(0)
        source.statistics[0].pure_mae = 0.12
        restored = ContinuousVelocityCurriculum(cfg.commands)
        restored.load_state_dict(source.state_dict())
        self.assertEqual(source.current_ranges, restored.current_ranges)
        self.assertEqual(0.12, restored.statistics[0].pure_mae)

    def test_wheel_command_estimator_recovers_vx_and_yaw(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        y = torch.tensor([[-0.2, 0.2, -0.2, 0.2]])
        rolling = 0.4 - 0.5 * y
        env.get_wheel_kinematics = lambda: {
            "positions_base": torch.stack(
                (torch.zeros_like(y), y, torch.zeros_like(y)), dim=2
            ),
            "rolling_speeds": rolling,
        }
        estimate = env.get_wheel_command_kinematics()
        torch.testing.assert_close(estimate["vx_hat"], torch.tensor([0.4]))
        torch.testing.assert_close(estimate["yaw_hat"], torch.tensor([0.5]))

    def test_wheel_shaping_is_masked_by_locomotion_mode(self):
        env = SimpleNamespace(
            commands=torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.2, 0.0]]),
            locomotion_mode_ids=torch.tensor(
                [LOCOMOTION_MODE_TO_ID["pure_x"], LOCOMOTION_MODE_TO_ID["pure_y"]]
            ),
            get_wheel_command_kinematics=lambda: {
                "vx_hat": torch.tensor([0.4, 0.0]),
                "yaw_hat": torch.tensor([0.0, 0.0]),
            },
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    wheel_v_tracking_scale=0.4,
                    wheel_yaw_tracking_scale=0.5,
                    wheel_command_active_threshold=0.05,
                )
            ),
        )
        rewards = ZGWSARMRewards(env)
        torch.testing.assert_close(
            rewards._reward_wheel_v_tracking(), torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(
            rewards._reward_wheel_yaw_tracking(), torch.tensor([0.0, 0.0])
        )

    def test_yaw_tracking_is_active_masked_and_has_cauchy_tail(self):
        env = SimpleNamespace(
            commands=torch.tensor(
                [[0.0, 0.0, 0.1], [0.4, 0.0, 0.0], [0.4, 0.0, 0.2]]
            ),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                    LOCOMOTION_MODE_TO_ID["pure_x"],
                    LOCOMOTION_MODE_TO_ID["x_yaw"],
                ]
            ),
            base_ang_vel=torch.zeros(3, 3),
            ang_vel_tracking_error_buf=torch.zeros(3),
            get_wheel_command_kinematics=lambda: {
                "vx_hat": torch.zeros(3),
                "yaw_hat": torch.zeros(3),
            },
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    tracking_sigma_v_yaw=0.10,
                    wheel_yaw_tracking_scale=0.15,
                    wheel_command_active_threshold=0.05,
                )
            ),
        )
        rewards = ZGWSARMRewards(env)

        torch.testing.assert_close(
            rewards._reward_tracking_ang_vel_yaw(),
            torch.tensor([0.5, 0.0, 0.2]),
        )
        torch.testing.assert_close(
            rewards._reward_wheel_yaw_tracking(),
            torch.tensor([9.0 / 13.0, 0.0, 0.36]),
        )

    def test_wheel_lateral_slip_penalty_is_mode_dependent(self):
        state = {
            "contact": torch.ones(2, 4, dtype=torch.bool),
            "velocities_base": torch.tensor(
                [[[0.0, 0.25, 0.0]] * 4, [[0.0, 0.25, 0.0]] * 4]
            ),
        }
        env = SimpleNamespace(
            locomotion_mode_ids=torch.tensor(
                [LOCOMOTION_MODE_TO_ID["pure_x"], LOCOMOTION_MODE_TO_ID["pure_yaw"]]
            ),
            get_wheel_kinematics=lambda: state,
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    wheel_lateral_slip_scale=0.25,
                    wheel_lateral_slip_mode_weights={
                        "stand": 1.0, "pure_x": 1.0, "pure_y": 0.0,
                        "pure_yaw": 0.2, "xy": 0.0, "x_yaw": 0.2,
                        "y_yaw": 0.0, "full": 0.0,
                    },
                )
            ),
        )
        torch.testing.assert_close(
            ZGWSARMRewards(env)._reward_wheel_lateral_slip(),
            torch.tensor([1.0, 0.2]),
        )

    def test_stance_symmetry_rejects_one_sided_collapse_but_allows_crouching(self):
        symmetric = [
            [0.35, -0.20, -0.40], [0.35, 0.20, -0.40],
            [-0.35, -0.20, -0.40], [-0.35, 0.20, -0.40],
        ]
        asymmetric = [
            [0.35, -0.10, -0.50], [0.35, 0.30, -0.35],
            [-0.35, -0.20, -0.40], [-0.35, 0.20, -0.40],
        ]
        state = {
            "positions_base": torch.tensor(
                [symmetric, asymmetric, asymmetric, asymmetric], dtype=torch.float
            )
        }
        env = SimpleNamespace(
            commands=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.5],
                ]
            ),
            locomotion_mode_ids=torch.tensor(
                [
                    LOCOMOTION_MODE_TO_ID["stand"],
                    LOCOMOTION_MODE_TO_ID["stand"],
                    LOCOMOTION_MODE_TO_ID["pure_x"],
                    LOCOMOTION_MODE_TO_ID["pure_yaw"],
                ]
            ),
            get_wheel_kinematics=lambda: state,
            cfg=SimpleNamespace(
                asset=SimpleNamespace(wheel_dof_names=list(WHEEL_DOF_NAMES)),
                rewards=SimpleNamespace(
                    stand_still_command_threshold=0.1,
                    stance_lateral_symmetry_tolerance=0.04,
                    stance_height_symmetry_tolerance=0.04,
                    stance_symmetry_scale=0.10,
                ),
            ),
        )

        reward = ZGWSARMRewards(env)._reward_stance_symmetry()
        torch.testing.assert_close(
            reward, torch.tensor([0.0, 0.9425, 0.0, 0.0])
        )

    def test_leg_soft_limit_penalizes_policy_target_before_hard_limit(self):
        env = SimpleNamespace(
            leg_dof_indices=torch.tensor([0]),
            dof_pos=torch.tensor([[0.0], [0.95]]),
            joint_pos_target=torch.tensor([[0.95], [0.0]]),
            dof_pos_limits=torch.tensor([[-0.9, 0.9]]),
        )
        reward = ZGWSARMRewards(env)._reward_dof_pos_limits_leg()
        torch.testing.assert_close(reward, torch.tensor([0.05, 0.05]))

    def test_calibrated_orientation_uses_geodesic_error(self):
        nominal_pitch = -0.8707963268
        nominal_quat = torch.tensor(
            [[0.0, np.sin(nominal_pitch / 2.0), 0.0,
              np.cos(nominal_pitch / 2.0)]],
            dtype=torch.float32,
        )
        env = SimpleNamespace(
            num_envs=1,
            commands=torch.zeros(1, 23),
            force_or_position_control=torch.zeros(1),
            gripper_ori_tracking_error_buf=torch.ones(1),
            get_measured_ee_quat_yrf=lambda: nominal_quat,
            cfg=SimpleNamespace(
                commands=SimpleNamespace(
                    ee_nominal_orientation_rpy=[0.0, nominal_pitch, 0.0]
                ),
                rewards=SimpleNamespace(
                    manip_ori_tracking_sigma=0.5,
                    maintain_ori_force_envs=True,
                ),
            ),
        )

        reward = ZGWSARMRewards(env)._reward_manip_ori_tracking()

        torch.testing.assert_close(reward, torch.ones(1))
        torch.testing.assert_close(
            env.gripper_ori_tracking_error_buf, torch.zeros(1), atol=1e-6,
            rtol=0.0,
        )

    def test_nominal_orientation_is_zero_in_zgwsarm_command_frame(self):
        nominal_pitch = -0.8707963268
        nominal_quat = torch.tensor(
            [[0.0, np.sin(nominal_pitch / 2.0), 0.0,
              np.cos(nominal_pitch / 2.0)]],
            dtype=torch.float32,
        )
        commands = CommandLifecycleMixin()
        commands.num_envs = 1
        commands.cfg = SimpleNamespace(
            commands=SimpleNamespace(
                ee_nominal_orientation_rpy=[0.0, nominal_pitch, 0.0]
            )
        )
        commands.commands = torch.zeros(1, 23)
        commands.gripper_ori_tracking_error_buf = torch.ones(1)
        commands.get_measured_ee_quat_yrf = lambda: nominal_quat

        measured_command_rpy = commands.get_measured_ee_rpy_yrf()

        torch.testing.assert_close(
            measured_command_rpy, torch.zeros(1, 3), atol=1e-6, rtol=0.0
        )
        torch.testing.assert_close(
            commands.gripper_ori_tracking_error_buf,
            torch.zeros(1),
            atol=1e-6,
            rtol=0.0,
        )

    def test_zgwsarm_ee_height_filter_uses_task_clearance(self):
        commands = CommandLifecycleMixin()
        commands.device = "cpu"
        commands.cfg = SimpleNamespace(
            commands=SimpleNamespace(
                arm_mount_translation=[-0.195, 0.0, 0.1703],
                arm_mount_yaw=np.pi,
                command_base_height=0.54,
                ee_min_world_height=0.15,
            )
        )
        radius = torch.tensor([0.8, 0.8])
        pitch = torch.tensor([0.6, 1.0])

        feasible, rejected = commands.is_ee_cmd_feasible(radius, pitch)

        self.assertFalse(feasible)
        torch.testing.assert_close(rejected, torch.tensor([1]))

    def test_wheel_rewards_detect_lift_tuck_slip_and_rolling_error(self):
        state = {
            "contact": torch.tensor([[True, False, True, True]]),
            "normal_forces": torch.tensor([[100.0, 0.0, 100.0, 100.0]]),
            "velocities_base": torch.tensor(
                [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [1.0, 0.25, 0.0], [1.0, 0.0, 0.0]]]
            ),
            "positions_base": torch.tensor(
                [[[0.30, -0.20, -0.5], [0.30, 0.00, -0.5],
                  [-0.30, -0.20, -0.5], [-0.30, 0.20, -0.5]]]
            ),
            "rolling_residual": torch.tensor([[0.0, 0.0, 0.25, 0.0]]),
        }
        env = SimpleNamespace(
            get_wheel_kinematics=lambda: state,
            cfg=SimpleNamespace(
                asset=SimpleNamespace(
                    wheel_dof_names=list(WHEEL_DOF_NAMES)
                ),
                rewards=SimpleNamespace(
                    wheel_min_support_force=20.0,
                    wheel_support_front_x_range=[0.25, 0.45],
                    wheel_support_rear_x_range=[-0.45, -0.25],
                    wheel_support_right_y_range=[-0.32, -0.12],
                    wheel_support_left_y_range=[0.12, 0.32],
                    wheel_support_geometry_scale=0.10,
                    wheel_lateral_slip_scale=0.25,
                    wheel_rolling_error_scale=0.25,
                ),
            ),
        )
        rewards = ZGWSARMRewards(env)
        torch.testing.assert_close(
            rewards._reward_wheel_contact_consistency(), torch.tensor([0.25])
        )
        torch.testing.assert_close(
            rewards._reward_wheel_support_load(), torch.tensor([0.25])
        )
        torch.testing.assert_close(
            rewards._reward_wheel_support_geometry(), torch.tensor([0.144])
        )
        torch.testing.assert_close(
            rewards._reward_wheel_lateral_slip(), torch.tensor([1.0 / 3.0])
        )
        torch.testing.assert_close(
            rewards._reward_wheel_rolling_consistency(),
            torch.tensor([1.0 / 3.0]),
        )

    def test_wheel_support_geometry_softly_constrains_pair_x_positions(self):
        state = {
            "positions_base": torch.tensor(
                [[[0.34, -0.20, -0.5], [0.50, 0.20, -0.5],
                  [-0.34, -0.20, -0.5], [-0.50, 0.20, -0.5]]]
            )
        }
        env = SimpleNamespace(
            get_wheel_kinematics=lambda: state,
            cfg=SimpleNamespace(
                asset=SimpleNamespace(
                    wheel_dof_names=list(WHEEL_DOF_NAMES)
                ),
                rewards=SimpleNamespace(
                    wheel_support_front_x_range=[0.25, 0.45],
                    wheel_support_rear_x_range=[-0.45, -0.25],
                    wheel_support_right_y_range=[-0.32, -0.12],
                    wheel_support_left_y_range=[0.12, 0.32],
                    wheel_support_geometry_scale=0.10,
                ),
            ),
        )
        rewards = ZGWSARMRewards(env)

        # Each outboard wheel exceeds its range by 0.05 m and each pair
        # differs by 0.16 m. Ten normalized geometry terms are averaged.
        expected = (2 * 0.5 ** 2 + 2 * 1.6 ** 2) / 10.0
        torch.testing.assert_close(
            rewards._reward_wheel_support_geometry(),
            torch.tensor([expected]),
        )

    def test_wheel_support_geometry_penalizes_outward_splay(self):
        state = {
            "positions_base": torch.tensor(
                [[[0.35, -0.40, -0.4], [0.35, 0.40, -0.4],
                  [-0.35, -0.40, -0.4], [-0.35, 0.40, -0.4]]]
            )
        }
        env = SimpleNamespace(
            get_wheel_kinematics=lambda: state,
            cfg=SimpleNamespace(
                asset=SimpleNamespace(wheel_dof_names=list(WHEEL_DOF_NAMES)),
                rewards=SimpleNamespace(
                    wheel_support_front_x_range=[0.25, 0.45],
                    wheel_support_rear_x_range=[-0.45, -0.25],
                    wheel_support_right_y_range=[-0.32, -0.12],
                    wheel_support_left_y_range=[0.12, 0.32],
                    wheel_support_geometry_scale=0.10,
                ),
            ),
        )

        # Four wheels exceed their lateral bounds by 0.08 m. The geometry
        # vector contains four X, four Y, and two paired-X terms.
        expected = 4 * 0.8 ** 2 / 10.0
        torch.testing.assert_close(
            ZGWSARMRewards(env)._reward_wheel_support_geometry(),
            torch.tensor([expected]),
        )

    def test_reset_reason_tensors_report_individual_and_overlapping_causes(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.reset_buf = torch.tensor([True, True, True])
        env.time_out_buf = torch.tensor([True, False, False])
        env.body_height_buf = torch.tensor([False, True, False])
        env.body_ori_buf = torch.tensor([False, False, False])
        env.semantic_contact_reset_buf = torch.tensor([False, False, False])
        env.contact_buf = torch.tensor([False, False, False])
        env.dof_velocity_safety_reset_buf = torch.tensor([False, True, False])
        env.dof_position_safety_reset_buf = torch.tensor([False, False, False])
        env.nonfoot_contact_safety_reset_buf = torch.tensor([False, False, True])

        reasons = env.get_reset_reason_tensors()

        torch.testing.assert_close(
            reasons["timeout"], torch.tensor([True, False, False])
        )
        torch.testing.assert_close(
            reasons["body_height"], torch.tensor([False, True, False])
        )
        torch.testing.assert_close(
            reasons["dof_velocity_safety"],
            torch.tensor([False, True, False]),
        )
        torch.testing.assert_close(
            reasons["nonfoot_contact_safety"],
            torch.tensor([False, False, True]),
        )

    def test_termination_reward_excludes_timeouts(self):
        env = SimpleNamespace(
            reset_buf=torch.tensor([True, True, False]),
            time_out_buf=torch.tensor([False, True, False]),
        )
        rewards = ZGWSARMRewards(env)

        torch.testing.assert_close(
            rewards._reward_termination(), torch.tensor([1.0, 0.0, 0.0])
        )

    def test_wheel_position_observation_is_zero_without_mutating_state(self):
        wheel_indices = torch.tensor([3, 7, 11, 15])
        dof_pos = torch.arange(22, dtype=torch.float).unsqueeze(0)
        original = dof_pos.clone()
        cfg = SimpleNamespace(
            obs_scales=SimpleNamespace(dof_pos=1.0),
            noise_scales=SimpleNamespace(dof_pos=1.0),
            noise=SimpleNamespace(noise_level=1.0),
            commands=SimpleNamespace(control_only_z1=False),
        )
        env = SimpleNamespace(
            dof_pos=dof_pos,
            default_dof_pos=torch.zeros(1, 22),
            num_actuated_dof=22,
            zero_position_observation_dof_indices=wheel_indices,
            device="cpu",
            cfg=cfg,
        )
        sensor = JointPositionSensor(env)
        observation = sensor.get_observation()
        self.assertTrue(torch.all(observation[:, wheel_indices] == 0.0))
        self.assertTrue(torch.all(sensor.get_noise_vec()[wheel_indices] == 0.0))
        torch.testing.assert_close(env.dof_pos, original)

    def test_isaac_gym_loads_cleaned_asset_with_expected_order(self):
        gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.use_gpu_pipeline = False
        params.physx.use_gpu = False
        sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
        self.assertIsNotNone(sim)
        try:
            options = gymapi.AssetOptions()
            options.collapse_fixed_joints = False
            options.flip_visual_attachments = False
            asset = gym.load_asset(
                sim, str(URDF_PATH.parent), URDF_PATH.name, options
            )
            self.assertIsNotNone(asset)
            self.assertEqual(22, gym.get_asset_dof_count(asset))
            self.assertEqual(
                EXPECTED_DOF_NAMES, tuple(gym.get_asset_dof_names(asset))
            )
            body_names = tuple(gym.get_asset_rigid_body_names(asset))
            self.assertEqual("BASE_LINK", body_names[0])
            self.assertIn("ROBOT_ARM_LINK7", body_names)
            for foot_name in FOOT_LINK_NAMES:
                self.assertIn(foot_name, body_names)
            cfg = ZGWSARMComplianceCfg()
            termination_names = {
                body_name
                for pattern in cfg.asset.terminate_after_contacts_on
                for body_name in body_names
                if pattern in body_name
            }
            self.assertEqual(
                set(body_names) - set(FOOT_LINK_NAMES), termination_names
            )
        finally:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    unittest.main()
