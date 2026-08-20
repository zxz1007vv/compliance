import torch
import numpy as np
from wbc_compliance_gym.commands import (
    INDEX_EE_POS_PITCH_CMD,
    INDEX_EE_POS_RADIUS_CMD,
    INDEX_EE_POS_YAW_CMD,
    INDEX_EE_ROLL_CMD,
    INDEX_EE_YAW_CMD,
)
from wbc_compliance_gym.utils.math_utils import quat_apply_yaw
from wbc_compliance_gym.rewards.common import WholeBodyComplianceRewards
from isaacgym.torch_utils import *

TRANSFORM_BASE_ARM_X = 0.2
TRANSFORM_BASE_ARM_Z = 0.1585
DEFAULT_BASE_HEIGHT = 0.6 # 0.78

class B1Z1Rewards(WholeBodyComplianceRewards):

    ###########################
    ########## ARM ############
    ###########################

    def _reward_manip_pos_tracking(self):
        '''
        Reward for manipulation tracking (EE positon)
        '''
        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.env.num_envs, 1) 
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.env.num_envs, 1) 
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(self.env.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.sin(yaw_cmd)
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the base frame
        x_cmd_base = x_cmd_arm.add_(TRANSFORM_BASE_ARM_X)
        y_cmd_base = y_cmd_arm
        z_cmd_base = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z)
        ee_position_cmd_base = torch.cat((x_cmd_base, y_cmd_base, z_cmd_base), dim=1)

        # Commands in world frame
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.env.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.env.base_pos[:, 0].view(self.env.num_envs, 1) 
        y_base_pos_world = self.env.base_pos[:, 1].view(self.env.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.env.base_pos[:, 2].view(self.env.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world


        # Get current ee position in world frame 
        ee_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator")
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,ee_idx,0:3].view(self.env.num_envs,3)

        # print("p", ee_pos_world, ee_position_cmd_world)
        # ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)
        # ee_position_error = torch.norm(ee_position_cmd_world - ee_pos_world, dim=1)
        ee_position_error = torch.sum(torch.square(ee_position_cmd_world - ee_pos_world), dim=1)

        ee_position_coeff = 15.0

        pos_rew =  torch.exp(-ee_position_coeff*ee_position_error)
        pos_rew = pos_rew * (1 - self.env.force_or_position_control)
        # position_or_freed = torch.logical_or(self.env.force_or_position_control == 0,
        #                                        self.env.freed_envs == 1)
        # pos_rew = pos_rew * position_or_freed.float()

        # print("p", ee_pos_world, ee_position_cmd_world)

        # print("eeposiitno error: ",torch.exp(-ee_position_coeff*ee_position_error)) 
        return pos_rew
    
    def _reward_manip_combo_tracking(self):
        '''
        Reward for manipulation tracking (EE positon)
        '''
        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.env.num_envs, 1) 
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.env.num_envs, 1) 
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(self.env.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.sin(yaw_cmd)
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the base frame
        x_cmd_base = x_cmd_arm.add_(TRANSFORM_BASE_ARM_X)
        y_cmd_base = y_cmd_arm
        z_cmd_base = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z)
        ee_position_cmd_base = torch.cat((x_cmd_base, y_cmd_base, z_cmd_base), dim=1)

        # Commands in world frame
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.env.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.env.base_pos[:, 0].view(self.env.num_envs, 1) 
        y_base_pos_world = self.env.base_pos[:, 1].view(self.env.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.env.base_pos[:, 2].view(self.env.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world


        # Get current ee position in world frame 
        ee_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator")
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,ee_idx,0:3].view(self.env.num_envs,3)

        # print("p", ee_pos_world, ee_position_cmd_world)
        # ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)
        ee_position_error = torch.norm(ee_position_cmd_world - ee_pos_world, dim=1)
        # ee_position_error = torch.sum(torch.square(ee_position_cmd_world - ee_pos_world), dim=1)

        ee_position_coeff = self.env.cfg.rewards.manip_pos_tracking_coef

        ee_rpy_yrf = self.env.get_measured_ee_rpy_yrf()

        ee_ori_cmd = self.env.commands[:, INDEX_EE_ROLL_CMD:INDEX_EE_YAW_CMD+1].clone()



        roll_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:,0]))
        pitch_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]))
        yaw_error = torch.minimum(torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]), 
                                        2*np.pi - torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]))

        assert not (torch.any(torch.logical_or(roll_error < 0, roll_error > np.pi)))

        tracking_coef_manip_ori = self.env.cfg.rewards.manip_ori_tracking_coef

        ee_ori_tracking_error = roll_error + pitch_error + yaw_error
        # ee_ori_tracking_error = roll_error**2 + pitch_error**2 + yaw_error**2

        return torch.exp(-ee_position_coeff*ee_position_error - ee_ori_tracking_error * tracking_coef_manip_ori) 
    
    def _reward_torque_limits_arm(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(
            (torch.abs(self.env.torques[:,12:19]) - self.env.torque_limits[12:19] * self.env.cfg.rewards.soft_torque_limit_arm).clip(min=0.)), dim=1)

    def _reward_dof_vel_arm(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel[:,12:18]), dim=1)

    def _reward_dof_acc_arm(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel[:,12:18] - self.env.dof_vel[:,12:18]) / self.env.dt), dim=1)

    def _reward_action_rate_arm(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions[:,12:18] - self.env.actions[:,12:18]), dim=1)

    def _reward_action_smoothness_1_arm(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target[:, 12:18] - self.env.last_joint_pos_target[:, 12:18])
        diff = diff * (self.env.last_actions[:,12:18] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_arm(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target[:, 12:18] - 2 * self.env.last_joint_pos_target[:, 12:18] + self.env.last_last_joint_pos_target[:, 12:18])
        diff = diff * (self.env.last_actions[:, 12:18] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, 12:18] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_base_height(self):
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1), dim=1)
        return torch.square(base_height - self.env.cfg.rewards.base_height_target)

    def _reward_dof_pos_limits_arm(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos[:, 12:18] - self.env.dof_pos_limits[12:18, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos[:, 12:18] - self.env.dof_pos_limits[12:18, 1]).clip(min=0.)
        out_of_limits = -(self.env.joint_pos_target[:, 12:18] - self.env.dof_pos_limits[12:18, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.joint_pos_target[:, 12:18] - self.env.dof_pos_limits[12:18, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)


    ###########################
    ########## LEG ############
    ###########################

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_swing = (self.env.clock_inputs < self.env.cfg.rewards.swing_ratio).float()
        
        reward = 0
        for i in range(4):
            reward += (desired_swing[:, i]) * (
                        (foot_forces[:, i] < 1.0).float())
                        # torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4
    
    

    def _reward_tracking_contacts_shaped_vel(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        # foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = (self.env.clock_inputs > (1-self.env.cfg.rewards.stance_ratio)).float()
        reward = 0
        for i in range(4):
            reward += (desired_contact[:, i]) * (
                        (foot_forces[:, i] > 1.0).float())
                        # torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
                        # torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        # print("reward: ", reward, " foot_velocities: ", foot_velocities, " desired_contact: ", desired_contact, " foot indices: ", self.env.feet_indices, " gait indices: ", self.env.gait_indices)
        return reward / 4
    
    def _reward_feet_clearance_cmd(self):
        foot_heights = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1)
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)

        desired_swing = (self.env.clock_inputs < self.env.cfg.rewards.swing_ratio).float()

        swing_progress = torch.clamp(-1 * self.env.clock_inputs, 0, 1)
        
        foot_target_height = self.env.cfg.rewards.footswing_height * desired_swing + 0.02

        # return torch.sum((foot_heights - foot_target_height)**2 * desired_swing, dim=-1)
        return torch.sum((foot_heights - foot_target_height)**2 * swing_progress, dim=-1)
    
    def _reward_torque_limits_leg(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(
            (torch.abs(self.env.torques[:,:12]) - self.env.torque_limits[:12] * self.env.cfg.rewards.soft_torque_limit_leg).clip(min=0.)), dim=1)

    def _reward_torques(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(self.env.torques[:,:12]), dim=1)
    
    def _reward_torques_arm(self):
        # penalize torques too close to the limit
        return torch.sum(torch.square(self.env.torques[:,12:]), dim=1)

    def _reward_dof_pos_limits_leg(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos[:, :12] - self.env.dof_pos_limits[:12, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos[:, :12] - self.env.dof_pos_limits[:12, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_leg(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel[:,:12]), dim=1)

    def _reward_dof_acc_leg(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel[:,:12] - self.env.dof_vel[:,:12]) / self.env.dt), dim=1)

    def _reward_action_rate_leg(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions[:,:12] - self.env.actions[:,:12]), dim=1)

    def _reward_action_smoothness_1_leg(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target[:, :12] - self.env.last_joint_pos_target[:, :12])
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2_leg(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target[:, :12] - 2 * self.env.last_joint_pos_target[:, :12] + self.env.last_last_joint_pos_target[:, :12])
        diff = diff * (self.env.last_actions[:, :12] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :12] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        # k_q = -0.75
        return torch.sum(torch.square(self.env.dof_pos[:, :12] - self.env.default_dof_pos[:, :12]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # # nominal positions: [FR, FL, RR, RL]
        # if self.env.cfg.commands.num_commands >= 13:
        #     desired_stance_width = self.env.commands[:, 12:13]
        #     desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        # else:
        # desired_stance_width = 0.55
        desired_stance_width = self.env.cfg.rewards.stance_width
        desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        # if self.env.cfg.commands.num_commands >= 14:
        #     desired_stance_length = self.env.commands[:, 13:14]
        #     desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        # else:
        # desired_stance_length = 0.85
        desired_stance_length = self.env.cfg.rewards.stance_length
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward


# Historical config files use this container name. Keep it as an exact alias
# so old configs and imports resolve to the canonical task reward class.
B1LocoZ1GaitfreeRewards = B1Z1Rewards
