"""ZGWSARM reward adapter for its interleaved leg/wheel/arm DOF layout."""

import torch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_rotate_inverse,
)

from wbc_compliance_gym.commands import (
    INDEX_EE_POS_PITCH_CMD,
    INDEX_EE_POS_RADIUS_CMD,
    INDEX_EE_POS_YAW_CMD,
)

from .common import WholeBodyComplianceRewards


class ZGWSARMRewards(WholeBodyComplianceRewards):
    """ZGWSARM-specific rewards with semantic joint-group selection."""

    @property
    def _arm(self):
        return self.env.arm_dof_indices

    @property
    def _motion(self):
        return self.env.motion_dof_indices

    @property
    def _legs(self):
        return self.env.leg_dof_indices

    def _ee_position_error_squared(self):
        radius = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD : INDEX_EE_POS_RADIUS_CMD + 1]
        pitch = self.env.commands[:, INDEX_EE_POS_PITCH_CMD : INDEX_EE_POS_PITCH_CMD + 1]
        yaw = self.env.commands[:, INDEX_EE_POS_YAW_CMD : INDEX_EE_POS_YAW_CMD + 1]
        position_arm = torch.cat(
            (
                radius * torch.cos(pitch) * torch.cos(yaw),
                radius * torch.cos(pitch) * torch.sin(yaw),
                -radius * torch.sin(pitch),
            ),
            dim=1,
        )
        position_base = self.env.arm_position_to_base(position_arm)
        base_rpy = torch.stack(get_euler_xyz(self.env.base_quat), dim=1)
        yaw_quat = quat_from_euler_xyz(
            torch.zeros_like(base_rpy[:, 0]),
            torch.zeros_like(base_rpy[:, 1]),
            base_rpy[:, 2],
        )
        target_world = quat_rotate_inverse(
            quat_conjugate(yaw_quat), position_base
        ) + self.env.command_base_position_world()
        current_world = self.env.rigid_body_state[
            :, self.env.gripper_stator_index, 0:3
        ]
        error_squared = torch.sum(torch.square(target_world - current_world), dim=1)
        self.env.gripper_pos_tracking_error_buf[:] = torch.sqrt(error_squared)
        return error_squared

    def _reward_manip_pos_tracking(self):
        reward = torch.exp(-15.0 * self._ee_position_error_squared())
        return reward * (1 - self.env.force_or_position_control)

    def _reward_termination(self):
        """Penalize true failures while excluding normal episode timeouts."""
        return (
            self.env.reset_buf.bool() & ~self.env.time_out_buf.bool()
        ).float()

    def _reward_orientation(self):
        """Penalize roll/pitch without constraining commanded yaw motion."""
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Track height relative to terrain under the moving robot."""
        error = (
            self.env.base_height_above_terrain()
            - self.env.cfg.rewards.base_height_target
        )
        return torch.square(error)

    # def _reward_stance_posture(self):
    #     """Regularize posture only for the zero-velocity command slice."""
    #     command_motion = torch.linalg.norm(self.env.commands[:, :2], dim=1)
    #     command_motion += torch.abs(self.env.commands[:, 2])
    #     standing_command = (
    #         command_motion
    #         < self.env.cfg.rewards.stand_still_command_threshold
    #     )
    #     posture_error = torch.sum(
    #         torch.square(
    #             self.env.dof_pos[:, self._legs]
    #             - self.env.default_dof_pos[:, self._legs]
    #         ),
    #         dim=1,
    #     )
    #     return posture_error * standing_command.float()

    def _reward_action_magnitude(self):
        """Penalize only the action tail near the policy clip boundary."""
        excess = (
            torch.abs(self.env.actions)
            - self.env.cfg.rewards.soft_action_limit
        ).clip(min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_torque_limits_arm(self):
        excess = (
            torch.abs(self.env.torques[:, self._arm])
            - self.env.torque_limits[self._arm]
            * self.env.cfg.rewards.soft_torque_limit_arm
        ).clip(min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_dof_vel_arm(self):
        return torch.sum(torch.square(self.env.dof_vel[:, self._arm]), dim=1)

    def _reward_dof_acc_arm(self):
        return torch.sum(
            torch.square(
                (self.env.last_dof_vel[:, self._arm] - self.env.dof_vel[:, self._arm])
                / self.env.dt
            ),
            dim=1,
        )

    def _reward_action_rate_arm(self):
        return torch.sum(
            torch.square(
                self.env.last_actions[:, self._arm] - self.env.actions[:, self._arm]
            ),
            dim=1,
        )

    def _reward_action_smoothness_1_arm(self):
        diff = torch.square(
            self.env.joint_pos_target[:, self._arm]
            - self.env.last_joint_pos_target[:, self._arm]
        )
        diff *= self.env.last_actions[:, self._arm] != 0
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_arm(self):
        diff = torch.square(
            self.env.joint_pos_target[:, self._arm]
            - 2 * self.env.last_joint_pos_target[:, self._arm]
            + self.env.last_last_joint_pos_target[:, self._arm]
        )
        diff *= self.env.last_actions[:, self._arm] != 0
        diff *= self.env.last_last_actions[:, self._arm] != 0
        return torch.sum(diff, dim=1)

    def _reward_dof_pos_limits_arm(self):
        positions = self.env.dof_pos[:, self._arm]
        targets = self.env.joint_pos_target[:, self._arm]
        lower = self.env.dof_pos_limits[self._arm, 0]
        upper = self.env.dof_pos_limits[self._arm, 1]
        position_excess = -(positions - lower).clip(max=0.0)
        position_excess += (positions - upper).clip(min=0.0)
        target_excess = -(targets - lower).clip(max=0.0)
        target_excess += (targets - upper).clip(min=0.0)
        return torch.sum(position_excess + target_excess, dim=1)

    def _reward_torque_limits_leg(self):
        excess = (
            torch.abs(self.env.torques[:, self._motion])
            - self.env.torque_limits[self._motion]
            * self.env.cfg.rewards.soft_torque_limit_leg
        ).clip(min=0.0)
        return torch.sum(torch.square(excess), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.env.torques[:, self._motion]), dim=1)

    def _reward_torques_arm(self):
        return torch.sum(torch.square(self.env.torques[:, self._arm]), dim=1)

    def _reward_dof_pos_limits_leg(self):
        positions = self.env.dof_pos[:, self._legs]
        lower = self.env.dof_pos_limits[self._legs, 0]
        upper = self.env.dof_pos_limits[self._legs, 1]
        excess = -(positions - lower).clip(max=0.0)
        excess += (positions - upper).clip(min=0.0)
        return torch.sum(excess, dim=1)

    def _reward_dof_vel_leg(self):
        return torch.sum(torch.square(self.env.dof_vel[:, self._motion]), dim=1)

    def _reward_dof_acc_leg(self):
        return torch.sum(
            torch.square(
                (self.env.last_dof_vel[:, self._motion] - self.env.dof_vel[:, self._motion])
                / self.env.dt
            ),
            dim=1,
        )

    def _reward_action_rate_leg(self):
        return torch.sum(
            torch.square(
                self.env.last_actions[:, self._motion]
                - self.env.actions[:, self._motion]
            ),
            dim=1,
        )

    def _reward_action_smoothness_1_leg(self):
        diff = torch.square(
            self.env.joint_pos_target[:, self._motion]
            - self.env.last_joint_pos_target[:, self._motion]
        )
        diff *= self.env.last_actions[:, self._motion] != 0
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_leg(self):
        diff = torch.square(
            self.env.joint_pos_target[:, self._motion]
            - 2 * self.env.last_joint_pos_target[:, self._motion]
            + self.env.last_last_joint_pos_target[:, self._motion]
        )
        diff *= self.env.last_actions[:, self._motion] != 0
        diff *= self.env.last_last_actions[:, self._motion] != 0
        return torch.sum(diff, dim=1)

    def _reward_dof_pos(self):
        return torch.sum(
            torch.square(
                self.env.dof_pos[:, self._legs]
                - self.env.default_dof_pos[:, self._legs]
            ),
            dim=1,
        )


__all__ = ["ZGWSARMRewards"]
