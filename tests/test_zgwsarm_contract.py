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
        self.assertEqual(4, cfg.domain_rand.lag_timesteps)
        self.assertEqual(0, cfg.asset.default_dof_drive_mode)
        self.assertFalse(cfg.asset.replace_cylinder_with_capsule)
        self.assertEqual(1, cfg.asset.self_collisions)
        self.assertEqual([90.0, 120.0, 120.0] * 4, cfg.commands.p_gains_legs)
        self.assertEqual([1.0, 1.0, 1.0] * 4, cfg.commands.d_gains_legs)
        self.assertEqual([60.0] * 4, cfg.commands.p_gains_wheels)
        self.assertEqual([0.2] * 4, cfg.commands.d_gains_wheels)

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
                arm_scale_reduction=2.0,
            ),
            domain_rand=SimpleNamespace(randomize_lag_timesteps=False),
        )
        env.default_dof_pos = torch.zeros(1, 22)
        env.dof_pos = torch.zeros(1, 22)
        env.dof_vel = torch.zeros(1, 22)
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

        actions = torch.zeros(1, 22)
        actions[0, 0] = 1.0
        actions[0, 3] = 2.0
        actions[0, 16] = 1.0
        env.dof_vel[0, 3] = 10.0
        env.dof_pos[0, 7] = 123.0
        env._compute_torques(actions)

        self.assertAlmostEqual(22.5, env.torques[0, 0].item())
        self.assertAlmostEqual(28.0, env.torques[0, 3].item())
        self.assertAlmostEqual(0.0, env.torques[0, 7].item())
        self.assertAlmostEqual(0.5, env.joint_pos_target[0, 16].item())

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
        finally:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    unittest.main()
