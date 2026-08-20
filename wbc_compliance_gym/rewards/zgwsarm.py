"""ZGWSARM reward adapter for its interleaved leg/wheel/arm DOF layout."""

import torch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
)

from wbc_compliance_gym.commands import (
    INDEX_EE_ROLL_CMD,
    INDEX_EE_YAW_CMD,
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

    def _reward_manip_ori_tracking(self):
        """Track a calibrated LINK7 tool frame with geodesic angle error."""
        current_quat = self.env.get_measured_ee_quat_yrf()
        command_rpy = self.env.commands[
            :, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
        ]
        command_delta = quat_from_euler_xyz(
            command_rpy[:, 0], command_rpy[:, 1], command_rpy[:, 2]
        )
        nominal_rpy = torch.as_tensor(
            self.env.cfg.commands.ee_nominal_orientation_rpy,
            dtype=current_quat.dtype,
            device=current_quat.device,
        )
        nominal_quat = quat_from_euler_xyz(
            nominal_rpy[0].expand(self.env.num_envs),
            nominal_rpy[1].expand(self.env.num_envs),
            nominal_rpy[2].expand(self.env.num_envs),
        )
        target_quat = quat_mul(nominal_quat, command_delta)
        error_quat = quat_mul(quat_conjugate(current_quat), target_quat)
        error_quat = error_quat / torch.linalg.norm(
            error_quat, dim=1, keepdim=True
        ).clamp(min=1e-8)
        error_angle = 2.0 * torch.atan2(
            torch.linalg.norm(error_quat[:, :3], dim=1),
            torch.abs(error_quat[:, 3]),
        )
        self.env.gripper_ori_tracking_error_buf[:] = error_angle
        sigma = float(self.env.cfg.rewards.manip_ori_tracking_sigma)
        reward = torch.exp(-torch.square(error_angle / sigma))
        if self.env.cfg.rewards.maintain_ori_force_envs:
            return reward
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

    def _reward_stance_posture(self):
        """Regularize leg posture only for the zero-velocity command slice."""
        command_motion = torch.linalg.norm(self.env.commands[:, :2], dim=1)
        command_motion += torch.abs(self.env.commands[:, 2])
        standing_command = (
            command_motion
            < self.env.cfg.rewards.stand_still_command_threshold
        )
        posture_error = torch.mean(
            torch.square(
                self.env.dof_pos[:, self._legs]
                - self.env.default_dof_pos[:, self._legs]
            ),
            dim=1,
        )
        return posture_error * standing_command.float()

    @staticmethod
    def _contact_weighted_mean(values, contact):
        weights = contact.float()
        return torch.sum(values * weights, dim=1) / torch.sum(
            weights, dim=1
        ).clamp(min=1.0)

    def _reward_wheel_contact_consistency(self):
        wheel_state = self.env.get_wheel_kinematics()
        return 1.0 - wheel_state["contact"].float().mean(dim=1)

    def _reward_wheel_support_load(self):
        wheel_state = self.env.get_wheel_kinematics()
        minimum = float(self.env.cfg.rewards.wheel_min_support_force)
        shortfall = (
            (minimum - wheel_state["normal_forces"]) / minimum
        ).clip(min=0.0)
        return torch.mean(torch.square(shortfall), dim=1)

    def _reward_wheel_support_geometry(self):
        positions = self.env.get_wheel_kinematics()["positions_base"]
        scale = float(self.env.cfg.rewards.wheel_support_geometry_scale)
        longitudinal = []
        lateral = []
        for wheel_index, dof_name in enumerate(
            self.env.cfg.asset.wheel_dof_names
        ):
            wheel_name = dof_name[: -len("_FOOT_JOINT")]
            x = positions[:, wheel_index, 0]
            y = positions[:, wheel_index, 1]
            if wheel_name.startswith("F"):
                x_error = (
                    self.env.cfg.rewards.wheel_support_front_x_min - x
                ).clip(min=0.0)
            else:
                x_error = (
                    x - self.env.cfg.rewards.wheel_support_rear_x_max
                ).clip(min=0.0)
            if wheel_name.endswith("R"):
                y_error = (
                    y - self.env.cfg.rewards.wheel_support_right_y_max
                ).clip(min=0.0)
            else:
                y_error = (
                    self.env.cfg.rewards.wheel_support_left_y_min - y
                ).clip(min=0.0)
            longitudinal.append(x_error)
            lateral.append(y_error)
        errors = torch.stack(longitudinal + lateral, dim=1) / scale
        return torch.mean(torch.square(errors), dim=1)

    def _reward_wheel_lateral_slip(self):
        wheel_state = self.env.get_wheel_kinematics()
        scale = float(self.env.cfg.rewards.wheel_lateral_slip_scale)
        error = torch.square(
            wheel_state["velocities_base"][:, :, 1] / scale
        )
        return self._contact_weighted_mean(error, wheel_state["contact"])

    def _reward_wheel_rolling_consistency(self):
        wheel_state = self.env.get_wheel_kinematics()
        scale = float(self.env.cfg.rewards.wheel_rolling_error_scale)
        error = torch.square(wheel_state["rolling_residual"] / scale)
        return self._contact_weighted_mean(error, wheel_state["contact"])

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
        hard_lower = self.env.dof_pos_hard_limits[self._arm, 0]
        hard_upper = self.env.dof_pos_hard_limits[self._arm, 1]
        midpoint = 0.5 * (hard_lower + hard_upper)
        half_span = 0.5 * (hard_upper - hard_lower)
        soft_fraction = float(self.env.cfg.rewards.soft_dof_pos_limit_arm)
        lower = midpoint - soft_fraction * half_span
        upper = midpoint + soft_fraction * half_span
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
        targets = self.env.joint_pos_target[:, self._legs]
        # Custom DOF setup has already contracted these URDF limits by
        # cfg.rewards.soft_dof_pos_limit.
        lower = self.env.dof_pos_limits[self._legs, 0]
        upper = self.env.dof_pos_limits[self._legs, 1]
        position_excess = -(positions - lower).clip(max=0.0)
        position_excess += (positions - upper).clip(min=0.0)
        target_excess = -(targets - lower).clip(max=0.0)
        target_excess += (targets - upper).clip(min=0.0)
        return torch.sum(position_excess + target_excess, dim=1)

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
