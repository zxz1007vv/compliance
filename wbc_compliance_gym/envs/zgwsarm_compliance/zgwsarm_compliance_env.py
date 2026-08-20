"""ZGWSARM compliance environment with real wheel-control semantics."""

import torch

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
        actions_scaled[:, self.hip_dof_indices] *= (
            self.cfg.control.hip_scale_reduction
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
