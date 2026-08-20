import unittest
from types import SimpleNamespace
from unittest.mock import patch

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import numpy as np
import torch

from wbc_compliance_gym.envs.base.legged_robot import LeggedRobot


class FakeDofProperties(dict):
    def __len__(self):
        return len(self["lower"])


class LeggedRobotSafetyTests(unittest.TestCase):
    @staticmethod
    def _make_b1_env_and_props():
        dof_count = 19
        lower = np.full(dof_count, -1.0, dtype=np.float32)
        upper = np.full(dof_count, 1.0, dtype=np.float32)
        lower[[2, 5]] = -2.6
        upper[[2, 5]] = -0.6
        props = FakeDofProperties(
            lower=lower,
            upper=upper,
            velocity=np.full(dof_count, 10.0, dtype=np.float32),
            effort=np.full(dof_count, 100.0, dtype=np.float32),
            damping=np.zeros(dof_count, dtype=np.float32),
            friction=np.zeros(dof_count, dtype=np.float32),
            stiffness=np.zeros(dof_count, dtype=np.float32),
            driveMode=np.zeros(dof_count, dtype=np.int32),
        )
        env = object.__new__(LeggedRobot)
        env.device = "cpu"
        env.num_dof = dof_count
        env.has_custom_dof_layout = False
        env.cfg = SimpleNamespace(
            asset=SimpleNamespace(default_dof_drive_mode=1),
            commands=SimpleNamespace(
                p_gains_arm=[1.0] * 7,
                d_gains_arm=[1.0] * 7,
                p_gains_legs=[1.0] * 12,
                d_gains_legs=[1.0] * 12,
            ),
            rewards=SimpleNamespace(soft_dof_pos_limit=0.9),
        )
        return env, props

    def test_hard_limits_are_preserved_for_runtime_safety_checks(self):
        env, props = self._make_b1_env_and_props()

        env._process_dof_props(props, env_id=0)

        self.assertAlmostEqual(-2.6, env.dof_pos_hard_limits[2, 0].item(), places=5)
        self.assertAlmostEqual(-2.6, env.dof_pos_hard_limits[5, 0].item(), places=5)
        self.assertEqual(19, env.dof_limited_indices.numel())

    def test_position_targets_are_clamped_to_finite_hard_limits(self):
        env = object.__new__(LeggedRobot)
        env._cuda_debugger = None
        env.dof_pos_hard_limits = torch.tensor(
            [[-1.0, 1.0], [-2.0, 2.0], [0.0, 0.0]]
        )
        env.dof_limited_indices = torch.tensor([0, 1], dtype=torch.long)
        env.joint_pos_target = torch.tensor(
            [[-3.0, 3.0, 5.0], [0.5, -1.5, -4.0]]
        )

        env._clamp_joint_pos_target_to_hard_limits()

        expected = torch.tensor(
            [[-1.0, 2.0, 5.0], [0.5, -1.5, -4.0]]
        )
        self.assertTrue(torch.equal(expected, env.joint_pos_target))

    def test_custom_arm_reset_uses_task_configured_position_range(self):
        env = object.__new__(LeggedRobot)
        env.device = "cpu"
        env.has_custom_dof_layout = True
        env.arm_dof_indices = torch.tensor([2, 3], dtype=torch.long)
        env.default_dof_pos = torch.tensor([[0.1, 0.2, 0.8, -1.5]])
        env.dof_state = torch.zeros(1, 4, 2)
        env.dof_pos = env.dof_state[..., 0]
        env.dof_vel = env.dof_state[..., 1]
        env.joint_pos_target = torch.zeros(1, 4)
        env.dof_pos_limits = torch.tensor(
            [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]
        )
        env.robot_actor_idxs = torch.tensor([0], dtype=torch.long)
        env.gym = SimpleNamespace(
            set_dof_state_tensor_indexed=lambda *args, **kwargs: None
        )
        env.sim = None
        env.cfg = SimpleNamespace(
            env=SimpleNamespace(add_balls=False),
            ball=SimpleNamespace(asset="ball"),
        )
        cfg = SimpleNamespace(
            init_state=SimpleNamespace(
                arm_reset_position_range=[0.0, 0.0]
            )
        )

        with patch(
            "wbc_compliance_gym.envs.base.legged_robot.gymtorch.unwrap_tensor",
            side_effect=lambda value: value,
        ):
            env._reset_dofs(torch.tensor([0]), cfg)

        torch.testing.assert_close(env.dof_pos, env.default_dof_pos)
        torch.testing.assert_close(env.joint_pos_target, env.default_dof_pos)
        torch.testing.assert_close(env.dof_vel, torch.zeros_like(env.dof_vel))


if __name__ == "__main__":
    unittest.main()
