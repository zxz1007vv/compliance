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
    LOCOMOTION_MODE_ACTIVE_AXES,
    LOCOMOTION_MODE_NAMES,
    LOCOMOTION_MODE_TO_ID,
    resolve_yaw_gait_phase_slots,
    yaw_swing_envelope,
    yaw_swing_trajectory,
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

    def _stand_command_mask(self):
        """Select true stand commands without suppressing pure-yaw leg motion."""
        if hasattr(self.env, "locomotion_mode_ids"):
            return (
                self.env.locomotion_mode_ids
                == LOCOMOTION_MODE_TO_ID["stand"]
            )

        planar_command = torch.linalg.norm(self.env.commands[:, :3], dim=1)
        return (
            planar_command
            < self.env.cfg.rewards.stand_still_command_threshold
        )

    def _pure_yaw_command_mask(self):
        """Select yaw-only commands without affecting combined locomotion."""
        if hasattr(self.env, "locomotion_mode_ids"):
            return (
                self.env.locomotion_mode_ids
                == LOCOMOTION_MODE_TO_ID["pure_yaw"]
            )

        if not hasattr(self.env, "commands"):
            reference = self.env.get_wheel_kinematics()["contact"]
            return torch.zeros(
                reference.shape[0], dtype=torch.bool, device=reference.device
            )

        threshold = float(
            getattr(
                self.env.cfg.rewards,
                "wheel_command_active_threshold",
                0.05,
            )
        )
        return (
            (torch.abs(self.env.commands[:, 0]) <= threshold)
            & (torch.abs(self.env.commands[:, 1]) <= threshold)
            & (torch.abs(self.env.commands[:, 2]) > threshold)
        )

    def _get_yaw_gait_phase(self):
        """Return the phase buffer that also drives ClockSensor."""
        return torch.remainder(self.env.gait_indices, 1.0)

    def _get_yaw_gait_phase_slots(self):
        return torch.tensor(
            resolve_yaw_gait_phase_slots(
                self.env.cfg.asset.wheel_dof_names,
                self.env.cfg.rewards.yaw_gait_phase_order,
            ),
            dtype=torch.long,
            device=self.env.commands.device,
        )

    def _get_yaw_swing_mask(self):
        """Select exactly one repositioning wheel in each crawl phase."""
        active_slot = torch.floor(self._get_yaw_gait_phase() * 4.0).long()
        return active_slot[:, None] == self._get_yaw_gait_phase_slots()[None, :]

    def _get_yaw_swing_progress(self):
        return torch.remainder(self._get_yaw_gait_phase() * 4.0, 1.0)

    def _get_yaw_swing_weight(self):
        envelope = yaw_swing_envelope(
            self._get_yaw_swing_progress(),
            self.env.cfg.rewards.yaw_gait_transition_fraction,
        )
        return self._get_yaw_swing_mask().float() * envelope[:, None]

    def _get_yaw_motion_weight(self):
        """Weight velocity shaping by the derivative of the swing trajectory."""
        transition = float(
            self.env.cfg.rewards.yaw_gait_transition_fraction
        )
        progress = self._get_yaw_swing_progress()
        motion_progress = (
            (progress - transition) / (1.0 - 2.0 * transition)
        ).clip(0.0, 1.0)
        motion_envelope = 4.0 * motion_progress * (1.0 - motion_progress)
        return self._get_yaw_swing_mask().float() * motion_envelope[:, None]

    def _get_yaw_step_scale(self):
        reference_yaw = float(
            self.env.cfg.rewards.yaw_gait_step_reference_yaw
        )
        return (torch.abs(self.env.commands[:, 2]) / reference_yaw).clip(
            0.0, 1.0
        )

    def _get_nominal_wheel_xy(self, reference):
        """Build nominal footholds from the configured support-box centers."""
        nominal = torch.empty(
            (len(self.env.cfg.asset.wheel_dof_names), 2),
            dtype=reference.dtype,
            device=reference.device,
        )
        for wheel_index, dof_name in enumerate(
            self.env.cfg.asset.wheel_dof_names
        ):
            wheel_name = dof_name[: -len("_FOOT_JOINT")]
            x_range = (
                self.env.cfg.rewards.wheel_support_front_x_range
                if wheel_name.startswith("F")
                else self.env.cfg.rewards.wheel_support_rear_x_range
            )
            y_range = (
                self.env.cfg.rewards.wheel_support_right_y_range
                if wheel_name.endswith("R")
                else self.env.cfg.rewards.wheel_support_left_y_range
            )
            nominal[wheel_index, 0] = 0.5 * (
                float(x_range[0]) + float(x_range[1])
            )
            nominal[wheel_index, 1] = 0.5 * (
                float(y_range[0]) + float(y_range[1])
            )
        return nominal

    def _get_yaw_tangent_vectors(self, nominal_xy):
        tangent = torch.stack((-nominal_xy[:, 1], nominal_xy[:, 0]), dim=1)
        return tangent / torch.linalg.norm(tangent, dim=1, keepdim=True).clamp(
            min=1e-6
        )

    def _get_yaw_target_footholds(self, current_positions):
        nominal_xy = self._get_nominal_wheel_xy(current_positions)
        tangent = self._get_yaw_tangent_vectors(nominal_xy)
        trajectory = yaw_swing_trajectory(
            self._get_yaw_swing_progress(),
            self.env.cfg.rewards.yaw_gait_transition_fraction,
        )
        direction = torch.sign(self.env.commands[:, 2])
        displacement = (
            direction[:, None, None]
            * self._get_yaw_step_scale()[:, None, None]
            * trajectory[:, None, None]
            * float(self.env.cfg.rewards.yaw_gait_step_length)
            * tangent[None, :, :]
        )
        return nominal_xy[None, :, :] + displacement

    def _yaw_foothold_error_squared(self, wheel_state):
        positions_xy = wheel_state["positions_base"][:, :, :2]
        target_xy = self._get_yaw_target_footholds(positions_xy)
        return torch.sum(torch.square(positions_xy - target_xy), dim=2)

    def _reward_stance_posture(self):
        """Regularize leg posture only for the zero-velocity command slice."""
        posture_error = torch.mean(
            torch.square(
                self.env.dof_pos[:, self._legs]
                - self.env.default_dof_pos[:, self._legs]
            ),
            dim=1,
        )
        return posture_error * self._stand_command_mask().float()

    def _reward_yaw_leg_posture(self):
        """Allow small yaw steering motions but reject large leg distortion."""
        allowed = torch.as_tensor(
            self.env.cfg.rewards.yaw_leg_posture_allowed_deviation,
            dtype=self.env.dof_pos.dtype,
            device=self.env.dof_pos.device,
        )
        deviation = torch.abs(
            self.env.dof_pos[:, self._legs]
            - self.env.default_dof_pos[:, self._legs]
        )
        excess = (deviation - allowed).clip(min=0.0)
        scale = float(self.env.cfg.rewards.yaw_leg_posture_scale)
        posture_error = torch.mean(torch.square(excess / scale), dim=1)
        return posture_error * self._pure_yaw_command_mask().float()

    def _reward_yaw_height_floor(self):
        """Prevent pure-yaw tracking from exploiting a deep body crouch."""
        minimum_height = (
            float(self.env.cfg.rewards.base_height_target)
            - float(self.env.cfg.rewards.yaw_height_allowed_drop)
        )
        height_deficit = (
            minimum_height - self.env.base_height_above_terrain()
        ).clip(min=0.0)
        scale = float(self.env.cfg.rewards.yaw_height_scale)
        height_error = torch.square(height_deficit / scale)
        return height_error * self._pure_yaw_command_mask().float()

    def _reward_yaw_gait_support(self):
        """Reward one unloaded swing wheel and three load-bearing wheels."""
        wheel_state = self.env.get_wheel_kinematics()
        normal_forces = wheel_state["normal_forces"]
        swing_weight = self._get_yaw_swing_weight()
        stance_score = torch.sigmoid(
            (
                normal_forces
                - float(self.env.cfg.rewards.yaw_gait_stance_min_force)
            )
            / float(self.env.cfg.rewards.yaw_gait_stance_force_scale)
        )
        swing_score = torch.exp(
            -torch.square(
                normal_forces
                / float(self.env.cfg.rewards.yaw_gait_swing_force_scale)
            )
        )
        score = (
            stance_score + swing_weight * (swing_score - stance_score)
        ).mean(dim=1)
        return score * self._pure_yaw_command_mask().float()

    def _reward_yaw_foothold_tracking(self):
        """Move only the scheduled swing wheel along the commanded tangent."""
        wheel_state = self.env.get_wheel_kinematics()
        error_squared = self._yaw_foothold_error_squared(wheel_state)
        sigma = float(self.env.cfg.rewards.yaw_gait_foothold_sigma)
        scores = torch.exp(-error_squared / (sigma * sigma))
        swing_mask = self._get_yaw_swing_mask().float()
        swing_score = torch.sum(scores * swing_mask, dim=1)
        return swing_score * self._pure_yaw_command_mask().float()

    def _reward_yaw_swing_tangential_velocity(self):
        """Give a bounded progress signal to the scheduled swing wheel."""
        wheel_state = self.env.get_wheel_kinematics()
        positions_xy = wheel_state["positions_base"][:, :, :2]
        nominal_xy = self._get_nominal_wheel_xy(positions_xy)
        tangent = self._get_yaw_tangent_vectors(nominal_xy)
        direction = torch.sign(self.env.commands[:, 2])
        commanded_tangent = direction[:, None, None] * tangent[None, :, :]
        tangential_velocity = torch.sum(
            wheel_state["velocities_base"][:, :, :2] * commanded_tangent,
            dim=2,
        )
        scale = float(
            self.env.cfg.rewards.yaw_gait_tangential_velocity_scale
        )
        scores = torch.tanh(tangential_velocity / scale).clip(min=0.0)
        motion_weight = self._get_yaw_motion_weight()
        swing_score = torch.sum(scores * motion_weight, dim=1)
        return swing_score * self._pure_yaw_command_mask().float()

    def get_yaw_gait_diagnostics(self):
        """Return per-environment pure-yaw mechanism metrics for logging."""
        wheel_state = self.env.get_wheel_kinematics()
        wheel_command = self.env.get_wheel_command_kinematics()
        pure_yaw = self._pure_yaw_command_mask()
        swing_mask = self._get_yaw_swing_mask()
        stance_mask = ~swing_mask
        normal_forces = wheel_state["normal_forces"]
        foothold_error = torch.sqrt(
            self._yaw_foothold_error_squared(wheel_state).clamp(min=0.0)
        )
        swing_foothold_error = torch.sum(
            foothold_error * swing_mask.float(), dim=1
        )
        swing_force = torch.sum(
            normal_forces * swing_mask.float(), dim=1
        )
        stance_force = torch.sum(
            normal_forces * stance_mask.float(), dim=1
        ) / stance_mask.sum(dim=1).clamp(min=1)
        leg_deviation = torch.abs(
            self.env.dof_pos[:, self._legs]
            - self.env.default_dof_pos[:, self._legs]
        )
        abad_deviation = torch.abs(
            self.env.dof_pos[:, self.env.abad_dof_indices]
            - self.env.default_dof_pos[:, self.env.abad_dof_indices]
        )
        diagnostics = {
            "mask": pure_yaw,
            "body_yaw_error": torch.abs(
                self.env.base_ang_vel[:, 2] - self.env.commands[:, 2]
            ),
            "wheel_yaw_error": torch.abs(
                wheel_command["yaw_hat"] - self.env.commands[:, 2]
            ),
            "base_height": self.env.base_height_above_terrain(),
            "mean_leg_posture_error": leg_deviation.mean(dim=1),
            "contact_count": wheel_state["contact"].float().sum(dim=1),
            "swing_foot_normal_force": swing_force,
            "stance_feet_normal_force": stance_force,
            "foothold_tracking_error": swing_foothold_error,
            "commanded_step_length": self._get_yaw_step_scale()
            * float(self.env.cfg.rewards.yaw_gait_step_length),
            "swing_activation": self._get_yaw_swing_weight().sum(dim=1),
            "abad_max_deviation": abad_deviation.max(dim=1).values,
        }
        for wheel_index, dof_name in enumerate(
            self.env.cfg.asset.wheel_dof_names
        ):
            wheel_name = dof_name[: -len("_FOOT_JOINT")]
            diagnostics[f"{wheel_name}_contact"] = wheel_state["contact"][
                :, wheel_index
            ].float()
        return diagnostics

    def _reward_stance_symmetry(self):
        """Prevent one-sided collapse while allowing symmetric crouching."""
        positions = self.env.get_wheel_kinematics()["positions_base"]
        positions_by_wheel = {}
        for wheel_index, dof_name in enumerate(
            self.env.cfg.asset.wheel_dof_names
        ):
            wheel_name = dof_name[: -len("_FOOT_JOINT")]
            positions_by_wheel[wheel_name] = positions[:, wheel_index]

        lateral_tolerance = float(
            self.env.cfg.rewards.stance_lateral_symmetry_tolerance
        )
        height_tolerance = float(
            self.env.cfg.rewards.stance_height_symmetry_tolerance
        )
        errors = []
        for right_name, left_name in (("FAR", "FBL"), ("RAR", "RBL")):
            right = positions_by_wheel[right_name]
            left = positions_by_wheel[left_name]
            lateral_error = (
                torch.abs(right[:, 1] + left[:, 1]) - lateral_tolerance
            ).clip(min=0.0)
            height_error = (
                torch.abs(right[:, 2] - left[:, 2]) - height_tolerance
            ).clip(min=0.0)
            errors.extend((lateral_error, height_error))

        scale = float(self.env.cfg.rewards.stance_symmetry_scale)
        symmetry_error = torch.mean(
            torch.square(torch.stack(errors, dim=1) / scale), dim=1
        )
        return symmetry_error * self._stand_command_mask().float()

    def _locomotion_axis_mask(self, axis):
        if hasattr(self.env, "locomotion_mode_ids"):
            return LOCOMOTION_MODE_ACTIVE_AXES.to(self.env.commands.device)[
                self.env.locomotion_mode_ids, axis
            ]
        return torch.abs(self.env.commands[:, axis]) > float(
            self.env.cfg.rewards.wheel_command_active_threshold
        )

    def _reward_tracking_ang_vel_yaw(self):
        """Track yaw only when commanded, with a non-vanishing long tail."""
        error = torch.abs(
            self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]
        )
        self.env.ang_vel_tracking_error_buf[:] = error
        scale = float(self.env.cfg.rewards.tracking_sigma_v_yaw)
        reward = 1.0 / (1.0 + torch.square(error / scale))
        return reward * self._locomotion_axis_mask(2).float()

    def _reward_wheel_v_tracking(self):
        wheel = self.env.get_wheel_command_kinematics()
        error = wheel["vx_hat"] - self.env.commands[:, 0]
        self.env.wheel_v_tracking_error_buf = torch.abs(error)
        scale = float(self.env.cfg.rewards.wheel_v_tracking_scale)
        reward = torch.exp(-torch.square(error / scale))
        return reward * self._locomotion_axis_mask(0).float()

    def _reward_wheel_yaw_tracking(self):
        wheel = self.env.get_wheel_command_kinematics()
        error = wheel["yaw_hat"] - self.env.commands[:, 2]
        self.env.wheel_yaw_tracking_error_buf = torch.abs(error)
        scale = float(self.env.cfg.rewards.wheel_yaw_tracking_scale)
        reward = 1.0 / (1.0 + torch.square(error / scale))
        return reward * self._locomotion_axis_mask(2).float()

    @staticmethod
    def _contact_weighted_mean(values, contact):
        weights = contact.float()
        return torch.sum(values * weights, dim=1) / torch.sum(
            weights, dim=1
        ).clamp(min=1.0)

    def _reward_wheel_contact_consistency(self):
        wheel_state = self.env.get_wheel_kinematics()
        penalty = 1.0 - wheel_state["contact"].float().mean(dim=1)
        return penalty * (~self._pure_yaw_command_mask()).float()

    def _reward_wheel_support_load(self):
        wheel_state = self.env.get_wheel_kinematics()
        minimum = float(self.env.cfg.rewards.wheel_min_support_force)
        shortfall = (
            (minimum - wheel_state["normal_forces"]) / minimum
        ).clip(min=0.0)
        penalty = torch.mean(torch.square(shortfall), dim=1)
        return penalty * (~self._pure_yaw_command_mask()).float()

    def _reward_wheel_support_geometry(self):
        positions = self.env.get_wheel_kinematics()["positions_base"]
        scale = float(self.env.cfg.rewards.wheel_support_geometry_scale)
        longitudinal = []
        lateral = []
        x_by_wheel = {}
        for wheel_index, dof_name in enumerate(
            self.env.cfg.asset.wheel_dof_names
        ):
            wheel_name = dof_name[: -len("_FOOT_JOINT")]
            x = positions[:, wheel_index, 0]
            y = positions[:, wheel_index, 1]
            x_by_wheel[wheel_name] = x
            if wheel_name.startswith("F"):
                x_range = self.env.cfg.rewards.wheel_support_front_x_range
            else:
                x_range = self.env.cfg.rewards.wheel_support_rear_x_range
            x_error = torch.maximum(
                (float(x_range[0]) - x).clip(min=0.0),
                (x - float(x_range[1])).clip(min=0.0),
            )
            if wheel_name.endswith("R"):
                y_range = self.env.cfg.rewards.wheel_support_right_y_range
            else:
                y_range = self.env.cfg.rewards.wheel_support_left_y_range
            y_error = torch.maximum(
                (float(y_range[0]) - y).clip(min=0.0),
                (y - float(y_range[1])).clip(min=0.0),
            )
            longitudinal.append(x_error)
            lateral.append(y_error)

        # Longitudinal pair alignment remains a soft cost. Lateral and height
        # pair symmetry are handled separately only for stationary commands.
        paired_longitudinal = [
            x_by_wheel["FAR"] - x_by_wheel["FBL"],
            x_by_wheel["RAR"] - x_by_wheel["RBL"],
        ]
        errors = torch.stack(
            longitudinal + lateral + paired_longitudinal, dim=1
        ) / scale
        return torch.mean(torch.square(errors), dim=1)

    def _reward_wheel_lateral_slip(self):
        wheel_state = self.env.get_wheel_kinematics()
        scale = float(self.env.cfg.rewards.wheel_lateral_slip_scale)
        error = torch.square(
            wheel_state["velocities_base"][:, :, 1] / scale
        )
        penalty = self._contact_weighted_mean(error, wheel_state["contact"])
        if not hasattr(self.env, "locomotion_mode_ids"):
            return penalty
        weights = self.env.cfg.rewards.wheel_lateral_slip_mode_weights
        if hasattr(weights, "items"):
            weights = dict(weights.items())
        mode_weights = torch.tensor(
            [float(weights[name]) for name in LOCOMOTION_MODE_NAMES],
            dtype=penalty.dtype,
            device=penalty.device,
        )
        return penalty * mode_weights[self.env.locomotion_mode_ids]

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
        return torch.sum(torch.square(self.env.dof_vel[:, self._legs]), dim=1)

    def _reward_dof_acc_leg(self):
        return torch.sum(
            torch.square(
                (self.env.last_dof_vel[:, self._legs] - self.env.dof_vel[:, self._legs])
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
