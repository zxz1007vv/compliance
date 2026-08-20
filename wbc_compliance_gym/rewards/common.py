"""Robot-neutral reward formulas shared by whole-body compliance tasks."""

from __future__ import annotations

import numpy as np
import torch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
)

from wbc_compliance_gym.commands import (
    INDEX_EE_FORCE_X,
    INDEX_EE_FORCE_Y,
    INDEX_EE_FORCE_Z,
    INDEX_EE_ROLL_CMD,
    INDEX_EE_YAW_CMD,
)


class RewardContainer:
    """Minimal lifecycle shared by dynamically selected reward containers."""

    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env


class LocomotionRewardMixin:
    """Reward formulas whose semantics do not assume a specific robot layout."""

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_lin_vel_z(self):
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_survival(self):
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]),
            dim=1,
        )
        self.env.lin_vel_tracking_error_buf[:] = lin_vel_error
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma_v_x)

    def _reward_tracking_ang_vel_yaw(self):
        ang_vel_error = torch.abs(
            self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]
        )
        self.env.ang_vel_tracking_error_buf[:] = ang_vel_error
        return torch.exp(
            -ang_vel_error / self.env.cfg.rewards.tracking_sigma_v_yaw
        )

    def _reward_feet_contact_forces(self):
        foot_forces = torch.norm(
            self.env.contact_forces[:, self.env.feet_indices, :], dim=-1
        )
        return torch.sum(
            (foot_forces - self.env.cfg.rewards.max_contact_force).clip(min=0.0),
            dim=1,
        )

    def _reward_collision(self):
        contact = torch.norm(
            self.env.contact_forces[
                :, self.env.penalised_contact_indices, :
            ],
            dim=-1,
        )
        return torch.sum((contact > 0.1).float(), dim=1)


class ManipulationRewardMixin:
    """End-effector orientation and force rewards shared by both robots."""

    def _reward_manip_ori_tracking(self):
        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()
        ee_ori_cmd = self.env.commands[
            :, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
        ].clone()

        ee_current_quat = quat_from_euler_xyz(
            ee_rpy_yrf[:, 0], ee_rpy_yrf[:, 1], ee_rpy_yrf[:, 2]
        )
        ee_cmd_quat = quat_from_euler_xyz(
            ee_ori_cmd[:, 0], ee_ori_cmd[:, 1], ee_ori_cmd[:, 2]
        )
        ee_quat_error = quat_mul(quat_conjugate(ee_current_quat), ee_cmd_quat)
        ee_quat_error /= torch.norm(ee_quat_error, dim=1).unsqueeze(1)
        error_value = torch.norm(ee_quat_error[:, :3], dim=1)

        roll_error = torch.minimum(
            torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:, 0]),
            2 * np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:, 0]),
        )
        assert not torch.any(
            torch.logical_or(roll_error < 0, roll_error > np.pi)
        )

        reward = torch.exp(-5.0 * error_value)
        if self.env.cfg.rewards.maintain_ori_force_envs:
            return reward
        return reward * (1 - self.env.force_or_position_control)

    def _reward_manip_ori_tracking_yaw_only(self):
        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()
        ee_ori_cmd = self.env.commands[
            :, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
        ].clone()
        zeros = torch.zeros_like(ee_rpy_yrf[:, 0])
        ee_current_quat = quat_from_euler_xyz(zeros, zeros, ee_rpy_yrf[:, 2])
        ee_cmd_quat = quat_from_euler_xyz(zeros, zeros, ee_ori_cmd[:, 2])
        ee_quat_error = quat_mul(quat_conjugate(ee_current_quat), ee_cmd_quat)
        ee_quat_error /= torch.norm(ee_quat_error, dim=1).unsqueeze(1)
        error_value = torch.norm(ee_quat_error[:, :3], dim=1)

        reward = torch.exp(-5.0 * error_value)
        if self.env.cfg.rewards.maintain_ori_force_envs:
            return reward
        return reward * (1 - self.env.force_or_position_control)

    def _forces_in_yaw_frame(self):
        forces_world = self.env.forces[
            :, self.env.gripper_stator_index, 0:3
        ]
        base_rpy_world = torch.stack(get_euler_xyz(self.env.base_quat), dim=1)
        base_yaw = quat_from_euler_xyz(
            torch.zeros_like(base_rpy_world[:, 0]),
            torch.zeros_like(base_rpy_world[:, 1]),
            base_rpy_world[:, 2],
        )
        return quat_rotate_inverse(base_yaw, forces_world)

    def _forces_in_command_frame(self):
        """Match measured forces to the frame used by the task command."""
        command_frame = getattr(
            self.env.cfg.rewards, "force_command_frame", "yaw"
        )
        if command_frame == "yaw":
            return self._forces_in_yaw_frame()
        if command_frame == "world":
            return self.env.forces[:, self.env.gripper_stator_index, 0:3]
        raise ValueError(f"unsupported force command frame {command_frame!r}")

    def _reward_ee_force_x(self):
        force_measured = self._forces_in_command_frame()[:, 0]
        force_command = self.env.commands[:, INDEX_EE_FORCE_X]
        error = torch.abs(force_measured - force_command)
        return (
            torch.exp(-self.env.cfg.rewards.sigma_force_z * error)
            * self.env.force_or_position_control
        )

    def _reward_ee_force_y(self):
        force_measured = self._forces_in_command_frame()[:, 1]
        force_command = self.env.commands[:, INDEX_EE_FORCE_Y]
        error = torch.abs(force_measured - force_command)
        return (
            torch.exp(-self.env.cfg.rewards.sigma_force_z * error)
            * self.env.force_or_position_control
        )

    def _reward_ee_force_z(self):
        force_measured = self._forces_in_command_frame()[:, 2]
        force_command = self.env.commands[:, INDEX_EE_FORCE_Z]
        error = torch.abs(force_measured - force_command)
        return (
            torch.exp(-self.env.cfg.rewards.sigma_force_z * error)
            * self.env.force_or_position_control
        )

    def _reward_ee_force_magnitude_x_pen(self):
        force = torch.abs(
            self.env.forces[:, self.env.gripper_stator_index, 0]
        )
        return force * self.env.force_or_position_control

    def _reward_ee_force_magnitude_y_pen(self):
        force = torch.abs(
            self.env.forces[:, self.env.gripper_stator_index, 1]
        )
        return force * self.env.force_or_position_control


class WholeBodyComplianceRewards(
    LocomotionRewardMixin,
    ManipulationRewardMixin,
    RewardContainer,
):
    """Shared reward surface; concrete tasks add robot-specific terms."""


__all__ = [
    "LocomotionRewardMixin",
    "ManipulationRewardMixin",
    "RewardContainer",
    "WholeBodyComplianceRewards",
]
