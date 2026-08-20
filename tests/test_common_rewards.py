import math
import unittest
from types import SimpleNamespace

import isaacgym  # noqa: F401 - must precede torch imports in this project.
import torch

from wbc_compliance_gym.commands import (
    INDEX_EE_FORCE_X,
    INDEX_EE_FORCE_Y,
    INDEX_EE_FORCE_Z,
)
from wbc_compliance_gym.rewards.common import WholeBodyComplianceRewards


class CommonRewardTests(unittest.TestCase):
    def test_velocity_tracking_updates_diagnostic_buffers(self):
        env = SimpleNamespace(
            commands=torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.0, 0.0]]),
            base_lin_vel=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            base_ang_vel=torch.tensor([[0.0, 0.0, 0.25], [0.0, 0.0, 0.0]]),
            lin_vel_tracking_error_buf=torch.zeros(2),
            ang_vel_tracking_error_buf=torch.zeros(2),
            cfg=SimpleNamespace(
                rewards=SimpleNamespace(
                    tracking_sigma_v_x=0.25,
                    tracking_sigma_v_yaw=0.25,
                )
            ),
        )
        rewards = WholeBodyComplianceRewards(env)

        linear = rewards._reward_tracking_lin_vel()
        angular = rewards._reward_tracking_ang_vel_yaw()

        torch.testing.assert_close(
            env.lin_vel_tracking_error_buf, torch.tensor([0.25, 0.0])
        )
        torch.testing.assert_close(
            env.ang_vel_tracking_error_buf, torch.tensor([0.25, 0.0])
        )
        torch.testing.assert_close(linear, torch.tensor([math.exp(-1.0), 1.0]))
        torch.testing.assert_close(angular, torch.tensor([math.exp(-1.0), 1.0]))

    def test_force_tracking_uses_yaw_frame_and_control_mode_mask(self):
        commands = torch.zeros(2, 23)
        commands[:, INDEX_EE_FORCE_X] = 5.0
        commands[:, INDEX_EE_FORCE_Y] = -2.0
        commands[:, INDEX_EE_FORCE_Z] = 8.0
        forces = torch.zeros(2, 1, 3)
        forces[:, 0, :] = torch.tensor([10.0, -2.0, 3.0])
        env = SimpleNamespace(
            num_envs=2,
            commands=commands,
            forces=forces,
            gripper_stator_index=0,
            base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
            force_or_position_control=torch.tensor([1.0, 0.0]),
            cfg=SimpleNamespace(rewards=SimpleNamespace(sigma_force_z=0.02)),
        )
        rewards = WholeBodyComplianceRewards(env)

        expected = torch.tensor([math.exp(-0.1), 0.0])
        torch.testing.assert_close(rewards._reward_ee_force_x(), expected)
        torch.testing.assert_close(
            rewards._reward_ee_force_y(), torch.tensor([1.0, 0.0])
        )
        torch.testing.assert_close(rewards._reward_ee_force_z(), expected)


if __name__ == "__main__":
    unittest.main()
