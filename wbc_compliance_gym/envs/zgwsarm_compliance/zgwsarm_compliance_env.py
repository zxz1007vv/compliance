"""ZGWSARM compliance environment with real wheel-control semantics."""

import torch
from isaacgym.torch_utils import quat_rotate_inverse

from wbc_compliance_gym.envs.base.velocity_tracking_env import VelocityTrackingEnv


class ZGWSARMComplianceEnv(VelocityTrackingEnv):
    """Keep the shared compliance task while adapting ZGWT actuators."""

    @staticmethod
    def _select_group_factors(factors, indices, num_dof):
        """Support both per-environment and per-DOF randomization tensors."""
        if factors.ndim == 2 and factors.shape[1] == num_dof:
            return factors[:, indices]
        return factors

    def _compute_torques(self, actions):
        """Apply leg position PD and wheel torque control.

        The arm keeps the existing compliance task's position targets.  In the
        mixed Isaac drive configuration only leg/wheel torques are applied;
        arm entries remain useful for the existing torque diagnostics/rewards.
        """
        if self.cfg.control.control_type != "P":
            raise ValueError(
                "ZGWSARM compliance requires P control for legs/arm plus "
                "the dedicated wheel torque path"
            )

        actions_scaled = torch.zeros(
            self.num_envs, self.num_dof, device=self.device
        )
        actions_scaled[:, : self.num_actions] = (
            actions[:, : self.num_actions] * self.cfg.control.action_scale
        )
        actions_scaled[:, self.abad_dof_indices] *= (
            self.cfg.control.abad_scale_reduction
        )
        actions_scaled[:, self.arm_dof_indices] *= (
            self.cfg.control.arm_scale_reduction
        )

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            actions_post = self.lag_buffer[0]
        else:
            actions_post = actions_scaled

        previous_arm_targets = self.joint_pos_target[
            :, self.arm_dof_indices
        ].clone()
        self.joint_pos_target = actions_post + self.default_dof_pos
        # The policy regularly produces targets outside ZGWSARM's asymmetric
        # hip/arm limits. Sending those directly to the arm position drives can
        # make PhysX solve hard joint constraints under sustained load.
        self._clamp_joint_pos_target_to_hard_limits()
        if self._cuda_debugger is not None:
            self._cuda_debug_hard_clamped_joint_pos_target = (
                self.joint_pos_target.clone()
            )
        arm_velocity_scale = self.cfg.control.arm_target_velocity_limit_scale
        max_arm_target_step = (
            self.dof_vel_limits[self.arm_dof_indices]
            * self.cfg.sim.dt
            * arm_velocity_scale
        )
        arm_target_delta = torch.clamp(
            self.joint_pos_target[:, self.arm_dof_indices]
            - previous_arm_targets,
            -max_arm_target_step,
            max_arm_target_step,
        )
        self.joint_pos_target[:, self.arm_dof_indices] = (
            previous_arm_targets + arm_target_delta
        )

        ideal_torques = (
            self.p_gains
            * self.Kp_factors
            * (self.joint_pos_target - self.dof_pos)
            - self.d_gains * self.Kd_factors * self.dof_vel
        )

        wheel_kd_factors = self._select_group_factors(
            self.Kd_factors, self.wheel_dof_indices, self.num_dof
        )
        ideal_torques[:, self.wheel_dof_indices] = (
            actions_post[:, self.wheel_dof_indices]
            * self.p_gains[self.wheel_dof_indices]
            - self.d_gains[self.wheel_dof_indices]
            * wheel_kd_factors
            * self.dof_vel[:, self.wheel_dof_indices]
        )

        self.torques = torch.clamp(
            ideal_torques, -self.torque_limits, self.torque_limits
        )

    def _terrain_height_at_base(self):
        """Sample terrain under the moving base, including trimesh courses."""
        if self.cfg.terrain.mesh_type in ("plane", "none"):
            return torch.zeros(
                self.num_envs, dtype=self.base_pos.dtype, device=self.device
            )
        return self.get_heights_points(self.base_pos).reshape(self.num_envs)

    def base_height_above_terrain(self):
        """Return base height relative to its current terrain location."""
        return self.base_pos[:, 2] - self._terrain_height_at_base()

    def get_wheel_kinematics(self):
        """Return wheel support and rolling state in the robot base frame."""
        cache_step = getattr(self, "common_step_counter", None)
        if (
            cache_step is not None
            and getattr(self, "_wheel_kinematics_cache_step", None)
            == cache_step
        ):
            return self._wheel_kinematics_cache

        wheel_count = len(self.cfg.asset.wheel_dof_names)
        if wheel_count != len(self.feet_indices):
            raise ValueError(
                "ZGWSARM requires one foot body per wheel DOF: "
                f"wheels={wheel_count}, feet={len(self.feet_indices)}"
            )

        wheel_states = self.rigid_body_state[:, self.feet_indices, :]
        wheel_vel_world = wheel_states[:, :, 7:10]
        wheel_pos_relative_world = (
            wheel_states[:, :, 0:3] - self.base_pos[:, None, :]
        )
        base_quat = self.base_quat[:, None, :].expand(-1, wheel_count, -1)
        wheel_vel_base = quat_rotate_inverse(
            base_quat.reshape(-1, 4), wheel_vel_world.reshape(-1, 3)
        ).reshape(self.num_envs, wheel_count, 3)
        wheel_pos_base = quat_rotate_inverse(
            base_quat.reshape(-1, 4), wheel_pos_relative_world.reshape(-1, 3)
        ).reshape(self.num_envs, wheel_count, 3)

        normal_forces = self.contact_forces[:, self.feet_indices, 2].abs()
        contact = normal_forces > float(
            self.cfg.rewards.wheel_contact_force_threshold
        )
        angular_speeds = self.dof_vel[:, self.wheel_dof_indices]
        rolling_speeds = angular_speeds * float(self.cfg.asset.wheel_radius)
        wheel_kinematics = {
            "contact": contact,
            "normal_forces": normal_forces,
            "velocities_base": wheel_vel_base,
            "positions_base": wheel_pos_base,
            "angular_speeds": angular_speeds,
            "rolling_speeds": rolling_speeds,
            "rolling_residual": wheel_vel_base[:, :, 0] - rolling_speeds,
        }
        if cache_step is not None:
            self._wheel_kinematics_cache_step = cache_step
            self._wheel_kinematics_cache = wheel_kinematics
        return wheel_kinematics

    def get_zgwsarm_diagnostic_tensors(self):
        """Return per-environment locomotion metrics without changing state."""
        metrics = {}
        clip = float(self.cfg.normalization.clip_actions)
        limit_margin = float(self.cfg.diagnostics.hard_limit_margin)

        for dof_name in self.cfg.asset.leg_dof_names:
            dof_index = self.dof_name_to_index[dof_name]
            leg_name, joint_name, _ = dof_name.split("_", 2)
            prefix = f"leg/{leg_name}/{joint_name}"
            position = self.dof_pos[:, dof_index]
            target = self.joint_pos_target[:, dof_index]
            action = self.actions[:, dof_index]
            default = self.default_dof_pos[:, dof_index]
            default_deviation = position - default

            hard_limits = self.dof_pos_hard_limits[dof_index]
            has_limit = torch.isfinite(hard_limits).all() & (
                hard_limits[1] > hard_limits[0]
            )
            distance_to_limit = torch.minimum(
                position - hard_limits[0],
                hard_limits[1] - position,
            )
            distance_to_limit = torch.where(
                has_limit,
                distance_to_limit,
                torch.full_like(position, float("inf")),
            )
            hard_limit_dwell = has_limit & (distance_to_limit <= limit_margin)

            metrics[f"{prefix}/position_rad"] = position
            metrics[f"{prefix}/target_rad"] = target
            metrics[f"{prefix}/action"] = action
            metrics[f"{prefix}/default_deviation_rad"] = default_deviation
            metrics[f"{prefix}/default_deviation_abs_rad"] = (
                default_deviation.abs()
            )
            metrics[f"{prefix}/distance_to_hard_limit_rad"] = (
                distance_to_limit
            )
            metrics[f"{prefix}/hard_limit_dwell"] = hard_limit_dwell.float()
            metrics[f"{prefix}/action_saturation"] = (
                action.abs() >= clip - 1e-6
            ).float()

        wheel_state = self.get_wheel_kinematics()
        contact = wheel_state["contact"]
        normal_forces = wheel_state["normal_forces"]
        wheel_vel_base = wheel_state["velocities_base"]
        wheel_pos_base = wheel_state["positions_base"]
        angular_speeds = wheel_state["angular_speeds"]
        rolling_speeds = wheel_state["rolling_speeds"]
        rolling_residual = wheel_state["rolling_residual"].abs()
        base_speed_residual = (
            rolling_speeds - self.base_lin_vel[:, 0:1]
        ).abs()

        for wheel_index, dof_name in enumerate(self.cfg.asset.wheel_dof_names):
            wheel_name = dof_name[: -len("_FOOT_JOINT")]
            prefix = f"wheel/{wheel_name}"
            metrics[f"{prefix}/contact"] = contact[:, wheel_index].float()
            metrics[f"{prefix}/normal_force_n"] = normal_forces[:, wheel_index]
            metrics[f"{prefix}/longitudinal_velocity_mps"] = (
                wheel_vel_base[:, wheel_index, 0]
            )
            metrics[f"{prefix}/lateral_velocity_mps"] = (
                wheel_vel_base[:, wheel_index, 1]
            )
            metrics[f"{prefix}/lateral_velocity_abs_mps"] = (
                wheel_vel_base[:, wheel_index, 1].abs()
            )
            metrics[f"{prefix}/angular_speed_radps"] = (
                angular_speeds[:, wheel_index]
            )
            metrics[f"{prefix}/rolling_speed_mps"] = rolling_speeds[:, wheel_index]
            metrics[f"{prefix}/rolling_residual_mps"] = (
                rolling_residual[:, wheel_index]
            )
            metrics[f"{prefix}/rolling_residual_contact_mps"] = torch.where(
                contact[:, wheel_index],
                rolling_residual[:, wheel_index],
                torch.full_like(rolling_residual[:, wheel_index], float("nan")),
            )
            metrics[f"{prefix}/base_speed_residual_mps"] = (
                base_speed_residual[:, wheel_index]
            )
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                metrics[f"{prefix}/base_position_{axis_name}_m"] = (
                    wheel_pos_base[:, wheel_index, axis_index]
                )

        return metrics

    def _semantic_contact_reset_mask(self):
        """Debounce low-force contacts on bodies that semantically mean failure."""
        indices = self.termination_contact_indices
        if not hasattr(self, "semantic_contact_counter"):
            self.semantic_contact_counter = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        if indices.numel() == 0:
            self.semantic_contact_counter.zero_()
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        magnitudes = torch.linalg.norm(
            self.contact_forces[:, indices, :], dim=-1
        )
        contact_now = torch.any(
            magnitudes > self.cfg.rewards.terminal_contact_force, dim=1
        )
        self.semantic_contact_counter = torch.where(
            contact_now,
            self.semantic_contact_counter + 1,
            torch.zeros_like(self.semantic_contact_counter),
        )
        return self.semantic_contact_counter >= int(
            self.cfg.rewards.terminal_contact_debounce_steps
        )

    def _safety_reset_masks(self):
        """Identify finite but pathological states before the next solve."""
        velocity_ratio = self.cfg.control.safety_dof_velocity_ratio
        velocity_reset = torch.any(
            torch.abs(self.dof_vel)
            > self.dof_vel_limits.unsqueeze(0) * velocity_ratio,
            dim=1,
        )

        limited = self.dof_limited_indices
        position_reset = torch.zeros_like(velocity_reset)
        if limited.numel() > 0:
            margin = self.cfg.control.safety_dof_position_margin
            positions = self.dof_pos[:, limited]
            lower = self.dof_pos_hard_limits[limited, 0] - margin
            upper = self.dof_pos_hard_limits[limited, 1] + margin
            position_reset = torch.any(
                (positions < lower) | (positions > upper), dim=1
            )

        # Foot impacts can legitimately be large. Contacts on any other body
        # at this magnitude indicate a trapped/fallen articulation and have
        # preceded the observed PhysX GPU solver failures.
        nonfoot_mask = torch.ones(
            self.num_bodies, dtype=torch.bool, device=self.device
        )
        nonfoot_mask[self.feet_indices] = False
        nonfoot_contact_magnitudes = torch.linalg.norm(
            self.contact_forces[:, nonfoot_mask, :], dim=-1
        )
        contact_reset = torch.any(
            nonfoot_contact_magnitudes
            > self.cfg.control.safety_nonfoot_contact_force,
            dim=1,
        )
        return velocity_reset, position_reset, contact_reset

    def check_termination(self):
        # The shared implementation subtracts measured_heights.  Populate it
        # with the height directly below the moving base so boxes_tm termination
        # is terrain-relative rather than a world-Z comparison.
        self.measured_heights = self._terrain_height_at_base().unsqueeze(1)
        super().check_termination()
        self.semantic_contact_reset_buf = self._semantic_contact_reset_mask()
        (
            self.dof_velocity_safety_reset_buf,
            self.dof_position_safety_reset_buf,
            self.nonfoot_contact_safety_reset_buf,
        ) = self._safety_reset_masks()
        self.contact_buf |= self.semantic_contact_reset_buf
        self.reset_buf |= self.semantic_contact_reset_buf
        self.reset_buf |= self.dof_velocity_safety_reset_buf
        self.reset_buf |= self.dof_position_safety_reset_buf
        self.reset_buf |= self.nonfoot_contact_safety_reset_buf

        cuda_debugger = getattr(self, "_cuda_debugger", None)
        if cuda_debugger is not None:
            cuda_debugger.report_safety_resets(
                {
                    "semantic_contact": self.semantic_contact_reset_buf,
                    "body_height": self.body_height_buf,
                    "body_orientation": self.body_ori_buf,
                    "velocity": self.dof_velocity_safety_reset_buf,
                    "position": self.dof_position_safety_reset_buf,
                    "nonfoot_contact": self.nonfoot_contact_safety_reset_buf,
                },
                step=int(self.common_step_counter),
            )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, "semantic_contact_counter") and len(env_ids) > 0:
            self.semantic_contact_counter[env_ids] = 0


__all__ = ["ZGWSARMComplianceEnv"]
