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

        self.joint_pos_target = actions_post + self.default_dof_pos

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


__all__ = ["ZGWSARMComplianceEnv"]
