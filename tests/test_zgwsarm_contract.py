import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import numpy as np
from isaacgym import gymapi
import torch

from wbc_compliance_gym.envs import register_tasks
from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_config import (
    ZGWSARMComplianceCfg,
    ZGWSARMComplianceCfgPPO,
    configure_zgwsarm_compliance_play,
)
from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_env import (
    ZGWSARMComplianceEnv,
)
from wbc_compliance_gym.rewards import REWARD_CONTAINERS, ZGWSARMRewards
from wbc_compliance_gym.robots.configs.zgwsarm import (
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

        self.assertEqual([0.0, 0.0, 0.55], cfg.init_state.pos)
        self.assertEqual(0.54, cfg.rewards.base_height_target)
        self.assertEqual(0.54, cfg.commands.command_base_height)
        self.assertEqual(0.002, cfg.sim.dt)
        self.assertEqual(5, cfg.control.decimation)
        self.assertEqual(0.01, cfg.sim.dt * cfg.control.decimation)
        self.assertEqual(2048, cfg.env.num_envs)
        self.assertEqual(4.0, cfg.normalization.clip_actions)
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
        self.assertEqual(0, cfg.asset.default_dof_drive_mode)
        self.assertFalse(cfg.asset.replace_cylinder_with_capsule)
        self.assertEqual(1, cfg.asset.self_collisions)
        self.assertEqual([90.0, 120.0, 120.0] * 4, cfg.commands.p_gains_legs)
        self.assertEqual([1.0, 1.0, 1.0] * 4, cfg.commands.d_gains_legs)
        self.assertEqual([60.0] * 4, cfg.commands.p_gains_wheels)
        self.assertEqual([0.2] * 4, cfg.commands.d_gains_wheels)
        self.assertFalse(cfg.rewards.only_positive_rewards)
        self.assertEqual(-10.0, cfg.reward_scales.termination)
        self.assertEqual(-5.0, cfg.reward_scales.orientation)
        self.assertEqual(-30.0, cfg.reward_scales.base_height)
        self.assertEqual(0.0, cfg.reward_scales.dof_pos)
        self.assertEqual([-1.0, 1.0], cfg.commands.lin_vel_x)
        self.assertEqual([-1.5, 1.5], cfg.commands.ang_vel_yaw)
        self.assertGreater(cfg.reward_scales.tracking_lin_vel, 0.0)
        self.assertGreater(cfg.reward_scales.tracking_ang_vel_yaw, 0.0)
        self.assertEqual(0.0, cfg.reward_scales.tracking_contacts_shaped_force)
        self.assertEqual(0.0, cfg.reward_scales.tracking_contacts_shaped_vel)
        self.assertEqual(0.0, cfg.reward_scales.feet_clearance_cmd)
        self.assertIn("BASE_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("ABAD_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("HIP_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("KNEE_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertIn("ROBOT_ARM_LINK", cfg.asset.terminate_after_contacts_on)
        self.assertEqual([-10.0, 10.0], cfg.domain_rand.max_push_force_xyz_gripper)
        self.assertEqual("position", cfg.commands.hybrid_mode)

    def test_play_disables_training_curriculum_but_keeps_explicit_force(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_compliance_play(
            cfg, control_mode="force", force_amplitude=30.0
        )

        self.assertFalse(cfg.domain_rand.zgwsarm_curriculum_enabled)
        self.assertEqual([-30.0, 30.0], cfg.domain_rand.max_push_force_xyz_gripper)
        self.assertEqual(
            [-30.0, 30.0], cfg.domain_rand.max_push_force_xyz_gripper_freed
        )
        self.assertEqual([0.0, 0.0], cfg.commands.lin_vel_x)

    def test_play_without_force_override_uses_final_curriculum_load(self):
        cfg = ZGWSARMComplianceCfg()
        configure_zgwsarm_compliance_play(cfg, control_mode="force")

        self.assertEqual([-70.0, 70.0], cfg.domain_rand.max_push_force_xyz_gripper)
        self.assertEqual(
            [-70.0, 70.0], cfg.domain_rand.max_push_force_xyz_gripper_freed
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
        env.cfg = SimpleNamespace(
            control=SimpleNamespace(
                control_type="P",
                action_scale=0.25,
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

        self.assertAlmostEqual(22.5, env.torques[0, 0].item())
        self.assertAlmostEqual(28.0, env.torques[0, 3].item())
        self.assertAlmostEqual(0.0, env.torques[0, 7].item())
        self.assertAlmostEqual(0.02, env.joint_pos_target[0, 16].item())
        self.assertAlmostEqual(0.02, env.joint_pos_target[0, 17].item())

    def test_progressive_randomization_reaches_midpoint(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.common_step_counter = 75000
        env.cfg = SimpleNamespace(
            domain_rand=SimpleNamespace(
                zgwsarm_curriculum_enabled=True,
                zgwsarm_force_mode_start_step=25000,
                zgwsarm_curriculum_start_step=25000,
                zgwsarm_curriculum_end_step=125000,
                zgwsarm_force_initial=10.0,
                zgwsarm_force_final=70.0,
                zgwsarm_push_velocity_initial=0.0,
                zgwsarm_push_velocity_final=0.8,
                zgwsarm_gravity_initial=0.0,
                zgwsarm_gravity_final=0.5,
                zgwsarm_motor_strength_initial=[0.98, 1.02],
                zgwsarm_motor_strength_final=[0.9, 1.1],
                zgwsarm_Kd_factor_initial=[0.9, 1.1],
                zgwsarm_Kd_factor_final=[0.5, 1.5],
            ),
            commands=SimpleNamespace(hybrid_mode="position"),
        )

        env._update_training_curriculum()

        domain_rand = env.cfg.domain_rand
        self.assertEqual([-40.0, 40.0], domain_rand.max_push_force_xyz_gripper)
        self.assertAlmostEqual(0.4, domain_rand.max_push_vel_xy)
        self.assertEqual([-0.25, 0.25], domain_rand.gravity_range)
        self.assertAlmostEqual(0.94, domain_rand.motor_strength_range[0])
        self.assertAlmostEqual(1.06, domain_rand.motor_strength_range[1])
        self.assertAlmostEqual(0.7, domain_rand.Kd_factor_range[0])
        self.assertAlmostEqual(1.3, domain_rand.Kd_factor_range[1])
        self.assertEqual("binary", env.cfg.commands.hybrid_mode)

    def test_force_mode_is_not_enabled_during_initial_locomotion_stage(self):
        env = ZGWSARMComplianceEnv.__new__(ZGWSARMComplianceEnv)
        env.common_step_counter = 24999
        env.cfg = SimpleNamespace(
            domain_rand=SimpleNamespace(
                zgwsarm_curriculum_enabled=True,
                zgwsarm_force_mode_start_step=25000,
                zgwsarm_curriculum_start_step=25000,
                zgwsarm_curriculum_end_step=125000,
                zgwsarm_force_initial=10.0,
                zgwsarm_force_final=70.0,
                zgwsarm_push_velocity_initial=0.0,
                zgwsarm_push_velocity_final=0.8,
                zgwsarm_gravity_initial=0.0,
                zgwsarm_gravity_final=0.5,
                zgwsarm_motor_strength_initial=[0.98, 1.02],
                zgwsarm_motor_strength_final=[0.9, 1.1],
                zgwsarm_Kd_factor_initial=[0.9, 1.1],
                zgwsarm_Kd_factor_final=[0.5, 1.5],
            ),
            commands=SimpleNamespace(hybrid_mode="binary"),
        )

        env._update_training_curriculum()

        self.assertEqual("position", env.cfg.commands.hybrid_mode)
        self.assertEqual(
            [-10.0, 10.0],
            env.cfg.domain_rand.max_push_force_xyz_gripper,
        )

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

    def test_stance_posture_reward_only_applies_to_zero_velocity_commands(self):
        env = SimpleNamespace(
            leg_dof_indices=torch.arange(12),
            commands=torch.tensor(
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float
            ),
            dof_pos=torch.ones(2, 12),
            default_dof_pos=torch.zeros(1, 12),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(stand_still_command_threshold=0.1)
            ),
        )
        rewards = ZGWSARMRewards(env)

        value = rewards._reward_stance_posture()

        torch.testing.assert_close(value, torch.tensor([12.0, 0.0]))

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
