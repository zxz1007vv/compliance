"""B1+Z1 command-vector contract and control-mode sampling.

The 23-dimensional command layout is consumed by the environment, sensors,
rewards, and evaluation scripts.  Keeping the indices here prevents those
components from silently drifting apart while preserving the legacy layout.
"""

from dataclasses import dataclass

import numpy as np
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    torch_rand_float,
)
import torch

from wbc_compliance_gym.utils.math_utils import plus_2pi_wrap_to_pi


INDEX_LIN_VEL_X = 0
INDEX_LIN_VEL_Y = 1
INDEX_ANG_VEL_YAW = 2
INDEX_BODY_HEIGHT = 3
INDEX_GAIT_FREQUENCY = 4
INDEX_GAIT_PHASE = 5
INDEX_GAIT_OFFSET = 6
INDEX_GAIT_BOUND = 7
INDEX_GAIT_DURATION = 8
INDEX_FOOTSWING_HEIGHT = 9
INDEX_BODY_PITCH = 10
INDEX_BODY_ROLL = 11
INDEX_EE_FORCE_X = 12
INDEX_EE_FORCE_Y = 13
INDEX_EE_FORCE_Z = 14
INDEX_EE_POS_RADIUS_CMD = 15
INDEX_EE_POS_PITCH_CMD = 16
INDEX_EE_POS_YAW_CMD = 17
INDEX_EE_POS_TIMING_CMD = 18
INDEX_EE_TIMING_CMD = INDEX_EE_POS_TIMING_CMD  # historical reward name
INDEX_EE_ROLL_CMD = 19
INDEX_EE_PITCH_CMD = 20
INDEX_EE_YAW_CMD = 21
INDEX_FORCE_OR_POSITION_INDICATOR = 22

COMMAND_DIMENSION = 23
VALID_CONTROL_MODES = ("position", "force", "binary", "mixed")

TRANSFORM_BASE_ARM_X = 0.2
TRANSFORM_BASE_ARM_Z = 0.1585
DEFAULT_BASE_HEIGHT = 0.6


@dataclass(frozen=True)
class CommandDimension:
    """One immutable row in the legacy command-distribution layout."""

    index: int
    curriculum_key: str
    curriculum_range_attr: str = None
    bins_attr: str = None
    active_range_attr: str = None
    fixed_curriculum_range: tuple = None
    fixed_active_range: tuple = None

    def curriculum_range(self, command_cfg):
        if self.fixed_curriculum_range is not None:
            return self.fixed_curriculum_range
        value_range = getattr(command_cfg, self.curriculum_range_attr)
        return (value_range[0], value_range[1], getattr(command_cfg, self.bins_attr))

    def active_range(self, command_cfg):
        if self.fixed_active_range is not None:
            return self.fixed_active_range
        value_range = getattr(command_cfg, self.active_range_attr)
        return (value_range[0], value_range[1])


# The names in positions 12-14 are historical curriculum-grid names.  Their
# grids intentionally remain fixed at zero while the force subsystem writes
# XYZ commands into those vector slots during a rollout.
COMMAND_SCHEMA = (
    CommandDimension(0, "x_vel", "limit_vel_x", "num_bins_vel_x", "lin_vel_x"),
    CommandDimension(1, "y_vel", "limit_vel_y", "num_bins_vel_y", "lin_vel_y"),
    CommandDimension(2, "yaw_vel", "limit_vel_yaw", "num_bins_vel_yaw", "ang_vel_yaw"),
    CommandDimension(3, "body_height", "limit_body_height", "num_bins_body_height", "body_height_cmd"),
    CommandDimension(4, "gait_frequency", "limit_gait_frequency", "num_bins_gait_frequency", "gait_frequency_cmd_range"),
    CommandDimension(5, "gait_phase", "limit_gait_phase", "num_bins_gait_phase", "gait_phase_cmd_range"),
    CommandDimension(6, "gait_offset", "limit_gait_offset", "num_bins_gait_offset", "gait_offset_cmd_range"),
    CommandDimension(7, "gait_bounds", "limit_gait_bound", "num_bins_gait_bound", "gait_bound_cmd_range"),
    CommandDimension(8, "gait_duration", "limit_gait_duration", "num_bins_gait_duration", "gait_duration_cmd_range"),
    CommandDimension(9, "footswing_height", "limit_footswing_height", "num_bins_footswing_height", "footswing_height_range"),
    CommandDimension(10, "body_pitch", "limit_body_pitch", "num_bins_body_pitch", "body_pitch_range"),
    CommandDimension(11, "body_roll", "limit_body_roll", "num_bins_body_roll", "body_roll_range"),
    CommandDimension(12, "stance_width", active_range_attr="ee_force_magnitude", fixed_curriculum_range=(0, 0, 1)),
    CommandDimension(13, "stance_length", active_range_attr="ee_force_direction_angle", fixed_curriculum_range=(0, 0, 1)),
    CommandDimension(14, "aux_reward_coef", active_range_attr="ee_force_z", fixed_curriculum_range=(0, 0, 1)),
    CommandDimension(15, "ee_sphe_radius", "limit_ee_sphe_radius", "num_bins_ee_sphe_radius", "ee_sphe_radius"),
    CommandDimension(16, "ee_sphe_pitch", "limit_ee_sphe_pitch", "num_bins_ee_sphe_pitch", "ee_sphe_pitch"),
    CommandDimension(17, "ee_sphe_yaw", "limit_ee_sphe_yaw", "num_bins_ee_sphe_yaw", "ee_sphe_yaw"),
    CommandDimension(18, "ee_timing", "limit_ee_timing", "num_bins_ee_timing", "ee_timing"),
    CommandDimension(19, "end_effector_roll", "limit_end_effector_roll", "num_bins_end_effector_roll", "end_effector_roll"),
    CommandDimension(20, "end_effector_pitch", "limit_end_effector_pitch", "num_bins_end_effector_pitch", "end_effector_pitch"),
    CommandDimension(21, "end_effector_yaw", "limit_end_effector_yaw", "num_bins_end_effector_yaw", "end_effector_yaw"),
    CommandDimension(22, "force_or_position_mode", fixed_curriculum_range=(0, 1, 1), fixed_active_range=(0, 1)),
)


def command_curriculum_ranges(command_cfg):
    """Return ordered keyword ranges for the curriculum constructor."""
    return {
        dimension.curriculum_key: dimension.curriculum_range(command_cfg)
        for dimension in COMMAND_SCHEMA
    }


def command_active_bounds(command_cfg):
    """Return the legacy low/high arrays used to activate curriculum bins."""
    active_ranges = [
        dimension.active_range(command_cfg) for dimension in COMMAND_SCHEMA
    ]
    return (
        np.array([value_range[0] for value_range in active_ranges]),
        np.array([value_range[1] for value_range in active_ranges]),
    )


def build_command_curricula(command_cfg):
    """Construct and initialize command curricula without owning env state."""
    if command_cfg.curriculum_type != "RewardThresholdCurriculum":
        raise ValueError(
            f"Unknown curriculum type {command_cfg.curriculum_type!r}; "
            "available: RewardThresholdCurriculum"
        )

    from wbc_compliance_gym.curriculum import RewardThresholdCurriculum

    category_names = ["nominal"]
    if command_cfg.gaitwise_curricula:
        category_names = ["pronk", "trot", "pace", "bound"]

    ranges = command_curriculum_ranges(command_cfg)
    curricula = [
        RewardThresholdCurriculum(seed=command_cfg.curriculum_seed, **ranges)
        for _ in category_names
    ]
    low, high = command_active_bounds(command_cfg)
    for curriculum in curricula:
        curriculum.set_to(low=low, high=high)
    return category_names, curricula


def validate_control_mode(mode):
    if mode not in VALID_CONTROL_MODES:
        choices = ", ".join(VALID_CONTROL_MODES)
        raise ValueError(f"Unknown control mode {mode!r}; choose one of: {choices}")
    return mode


def sample_control_modes(mode, count, device):
    """Return the legacy position/force indicator samples for ``mode``.

    Semantics intentionally match ``LeggedRobot`` before extraction:
    position=0, force=1, binary samples either endpoint, and mixed samples a
    continuous value in [0, 1).
    """
    validate_control_mode(mode)
    if mode == "mixed":
        return torch.rand(count, device=device)
    if mode == "binary":
        return torch.randint(0, 2, (count,), device=device).float()
    if mode == "force":
        return torch.ones(count, device=device)
    return torch.zeros(count, device=device)


class CommandLifecycleMixin:
    """Own command sampling, curriculum updates, and gait target generation.

    The mixin deliberately has no constructor and stores no state of its own.
    It operates on the existing environment buffers so extraction does not
    change random-number order, tensor allocation, or the physics-step path.
    """

    def _end_effector_body_name(self):
        return getattr(self.cfg.asset, "end_effector_name", "gripperStator")

    def _arm_mount_parameters(self):
        translation = getattr(
            self.cfg.commands,
            "arm_mount_translation",
            [TRANSFORM_BASE_ARM_X, 0.0, TRANSFORM_BASE_ARM_Z],
        )
        yaw = getattr(self.cfg.commands, "arm_mount_yaw", 0.0)
        base_height = getattr(
            self.cfg.commands, "command_base_height", DEFAULT_BASE_HEIGHT
        )
        return translation, yaw, base_height

    def _arm_position_to_base(self, position_arm):
        """Transform arm-frame XYZ positions to the robot base frame."""
        translation, yaw, _ = self._arm_mount_parameters()
        if yaw == 0.0:
            position_base = torch.zeros_like(position_arm)
            position_base[:, 0] = position_arm[:, 0] + translation[0]
            position_base[:, 1] = position_arm[:, 1] + translation[1]
            position_base[:, 2] = position_arm[:, 2] + translation[2]
            return position_base
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        position_base = torch.zeros_like(position_arm)
        position_base[:, 0] = (
            cos_yaw * position_arm[:, 0]
            - sin_yaw * position_arm[:, 1]
            + translation[0]
        )
        position_base[:, 1] = (
            sin_yaw * position_arm[:, 0]
            + cos_yaw * position_arm[:, 1]
            + translation[1]
        )
        position_base[:, 2] = position_arm[:, 2] + translation[2]
        return position_base

    def _base_position_to_arm(self, position_base):
        """Transform robot-base-frame XYZ positions to the arm frame."""
        translation, yaw, _ = self._arm_mount_parameters()
        translated = position_base.clone()
        translated[:, 0] -= translation[0]
        translated[:, 1] -= translation[1]
        translated[:, 2] -= translation[2]
        if yaw == 0.0:
            return translated
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        position_arm = torch.zeros_like(translated)
        position_arm[:, 0] = (
            cos_yaw * translated[:, 0] + sin_yaw * translated[:, 1]
        )
        position_arm[:, 1] = (
            -sin_yaw * translated[:, 0] + cos_yaw * translated[:, 1]
        )
        position_arm[:, 2] = translated[:, 2]
        return position_arm

    def _command_base_position_world(self):
        """Return the yaw-following, height-independent command-frame origin."""
        _, _, base_height = self._arm_mount_parameters()
        return torch.cat(
            (
                self.base_pos[:, 0:1],
                self.base_pos[:, 1:2],
                torch.ones_like(self.base_pos[:, 2:3]) * base_height,
            ),
            dim=1,
        )

    # Public forwarding points are needed because Gym wrappers intentionally
    # block access to private attributes from task reward containers.
    def arm_position_to_base(self, position_arm):
        return self._arm_position_to_base(position_arm)

    def command_base_position_world(self):
        return self._command_base_position_world()

    def is_ee_cmd_feasible(self, radius_cmd, pitch_cmd):
        commands_feasible = True
        z_cmd_arm = -radius_cmd * torch.sin(pitch_cmd)
        translation, _, base_height = self._arm_mount_parameters()
        z_cmd_world = z_cmd_arm.add_(
            translation[2] + base_height
        )

        env_ids = torch.arange(radius_cmd.shape[0], device=self.device)
        minimum_world_height = getattr(
            self.cfg.commands, "ee_min_world_height", 0.05
        )
        env_ids_resample = env_ids[z_cmd_world < minimum_world_height]
        if env_ids_resample.nelement() > 0:
            commands_feasible = False
        return commands_feasible, env_ids_resample

    def get_measured_ee_pos_spherical(self) -> torch.Tensor:
        """Return measured EE position in arm-frame spherical coordinates."""
        ee_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0],
            self.robot_actor_handles[0],
            self._end_effector_body_name(),
        )
        ee_position_world = self.rigid_body_state[
            :, ee_idx, 0:3
        ].view(self.num_envs, 3)

        base_quat_world = self.base_quat.view(self.num_envs, 4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(
            0 * base_rpy_world[:, 0],
            0 * base_rpy_world[:, 1],
            base_rpy_world[:, 2],
        )

        base_position_world = self._command_base_position_world()

        ee_position_base = quat_rotate_inverse(
            base_quat_world_indep, ee_position_world - base_position_world
        ).view(self.num_envs, 3)
        ee_position_arm = self._base_position_to_arm(ee_position_base)

        radius = torch.norm(ee_position_arm, dim=1).view(self.num_envs, 1)
        pitch = -torch.asin(
            ee_position_arm[:, 2].view(self.num_envs, 1) / radius
        ).view(self.num_envs, 1)
        yaw = torch.atan2(
            ee_position_arm[:, 1].view(self.num_envs, 1),
            ee_position_arm[:, 0].view(self.num_envs, 1),
        ).view(self.num_envs, 1)
        ee_pos_sphe_arm = torch.cat((radius, pitch, yaw), dim=1).view(
            self.num_envs, 3
        )

        radius_cmd = self.commands[:, INDEX_EE_POS_RADIUS_CMD].view(
            self.num_envs, 1
        )
        pitch_cmd = self.commands[:, INDEX_EE_POS_PITCH_CMD].view(
            self.num_envs, 1
        )
        yaw_cmd = self.commands[:, INDEX_EE_POS_YAW_CMD].view(
            self.num_envs, 1
        )
        x_cmd_arm = radius_cmd * torch.cos(pitch_cmd) * torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd * torch.cos(pitch_cmd) * torch.sin(yaw_cmd)
        z_cmd_arm = -radius_cmd * torch.sin(pitch_cmd)
        ee_position_cmd_base = self._arm_position_to_base(
            torch.cat((x_cmd_arm, y_cmd_arm, z_cmd_arm), dim=1)
        )
        ee_position_cmd_world = quat_rotate_inverse(
            quat_conjugate(base_quat_world_indep), ee_position_cmd_base
        ) + base_position_world
        self.gripper_pos_tracking_error_buf = torch.norm(
            ee_position_cmd_world - ee_position_world, dim=1
        )
        return ee_pos_sphe_arm

    def get_measured_ee_rpy_yrf(self) -> torch.Tensor:
        ee_quat_yrf = self.get_measured_ee_quat_yrf()
        nominal_rpy = getattr(
            self.cfg.commands, "ee_nominal_orientation_rpy", None
        )
        if nominal_rpy is not None:
            nominal_rpy = torch.as_tensor(
                nominal_rpy,
                dtype=ee_quat_yrf.dtype,
                device=ee_quat_yrf.device,
            )
            nominal_quat = quat_from_euler_xyz(
                nominal_rpy[0].expand(self.num_envs),
                nominal_rpy[1].expand(self.num_envs),
                nominal_rpy[2].expand(self.num_envs),
            )
            ee_quat_command_frame = quat_mul(
                quat_conjugate(nominal_quat), ee_quat_yrf
            )
        else:
            ee_quat_command_frame = ee_quat_yrf
        ee_rpy_yrf = torch.stack(
            get_euler_xyz(ee_quat_command_frame), dim=1
        )
        ee_rpy_yrf = plus_2pi_wrap_to_pi(ee_rpy_yrf)

        ee_ori_cmd = self.commands[
            :, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
        ].clone()
        if nominal_rpy is None:
            roll_error = torch.minimum(
                torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:, 0]),
                2 * np.pi - torch.abs(ee_rpy_yrf[:, 0] - ee_ori_cmd[:, 0]),
            )
            pitch_error = torch.minimum(
                torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]),
                2 * np.pi - torch.abs(ee_rpy_yrf[:, 1] - ee_ori_cmd[:, 1]),
            )
            yaw_error = torch.minimum(
                torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]),
                2 * np.pi - torch.abs(ee_rpy_yrf[:, 2] - ee_ori_cmd[:, 2]),
            )
            self.gripper_ori_tracking_error_buf = torch.norm(
                torch.stack((roll_error, pitch_error, yaw_error), dim=1), dim=1
            )
        else:
            command_quat = quat_from_euler_xyz(
                ee_ori_cmd[:, 0], ee_ori_cmd[:, 1], ee_ori_cmd[:, 2]
            )
            error_quat = quat_mul(
                quat_conjugate(ee_quat_command_frame), command_quat
            )
            error_quat = error_quat / torch.linalg.norm(
                error_quat, dim=1, keepdim=True
            ).clamp(min=1e-8)
            self.gripper_ori_tracking_error_buf = 2.0 * torch.atan2(
                torch.linalg.norm(error_quat[:, :3], dim=1),
                torch.abs(error_quat[:, 3]),
            )
        return ee_rpy_yrf

    def get_measured_ee_quat_yrf(self) -> torch.Tensor:
        """Return the LINK7 quaternion relative to the base-yaw frame."""
        ee_idx = self.gym.find_actor_rigid_body_handle(
            self.envs[0],
            self.robot_actor_handles[0],
            self._end_effector_body_name(),
        )
        ee_quat = self.rigid_body_state[:, ee_idx, 3:7].view(
            self.num_envs, 4
        )

        base_quat_world = self.base_quat.view(self.num_envs, 4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        quat_yrf = quat_from_euler_xyz(
            torch.zeros_like(
                base_rpy_world[:, 0], dtype=torch.float, device=self.device
            ),
            torch.zeros_like(
                base_rpy_world[:, 1], dtype=torch.float, device=self.device
            ),
            base_rpy_world[:, 2],
        )
        ee_quat_yrf = quat_mul(quat_conjugate(quat_yrf), ee_quat)
        return ee_quat_yrf / torch.linalg.norm(
            ee_quat_yrf, dim=1, keepdim=True
        ).clamp(min=1e-8)

    def set_gripper_teleop_value(self, value: float):
        self.teleop_gripper_value = value * torch.ones_like(
            self.teleop_gripper_value
        )

    def set_joint6_teleop_value(self, value: float):
        self.teleop_joint6_value = value * torch.ones_like(
            self.teleop_joint6_value
        )

    def set_trajectory_time(self, value: float):
        self.trajectory_time = value * torch.ones_like(self.trajectory_time)

    def set_initial_ee_pos(self):
        self.initial_ee_pos = self.get_measured_ee_pos_spherical()

    def set_initial_ee_rpy(self):
        self.initial_ee_rpy = self.get_measured_ee_rpy_yrf()

    def set_target_joint_angles(self, target_joints):
        self.target_joint_values = target_joints

    def compute_intermediate_ee_pos_command(self, env_ids):
        """Advance the legacy EE command trajectory without changing timing."""
        ee_pos_meas = self.get_measured_ee_pos_spherical()
        ee_rpy_meas = self.get_measured_ee_rpy_yrf()

        if self.init_training:
            self.init_training = False
            self.ee_target_pos_cmd = self.commands[
                :, INDEX_EE_POS_RADIUS_CMD : INDEX_EE_POS_YAW_CMD + 1
            ].view(self.num_envs, 3).clone()
            self.ee_target_rpy_cmd = self.commands[
                :, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
            ].view(self.num_envs, 3).clone()
            self.first_ee_pos = ee_pos_meas
            self.initial_ee_pos[:] = self.first_ee_pos[:]
            self.first_ee_rpy = ee_rpy_meas
            self.initial_ee_rpy[:] = self.first_ee_rpy[:]

        if len(env_ids) > 0:
            self.trajectory_time[env_ids] = 0.0
            self.ee_target_pos_cmd[env_ids] = self.commands[
                env_ids,
                INDEX_EE_POS_RADIUS_CMD : INDEX_EE_POS_YAW_CMD + 1,
            ].view(len(env_ids), 3)
            self.ee_target_rpy_cmd[env_ids] = self.commands[
                env_ids, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
            ].view(len(env_ids), 3)
            self.initial_ee_pos[env_ids] = ee_pos_meas[env_ids, :]
            self.initial_ee_rpy[env_ids] = ee_rpy_meas[env_ids, :]
            self._resample_force_or_position_control(env_ids)

        t_traj = self.commands[:, INDEX_EE_POS_TIMING_CMD]
        env_ids_inter = (
            self.trajectory_time.view(self.num_envs) < t_traj
        ).nonzero(as_tuple=False).flatten()

        self.commands[
            :, INDEX_EE_POS_RADIUS_CMD : INDEX_EE_POS_YAW_CMD + 1
        ] = self.ee_target_pos_cmd.view(self.num_envs, 3)
        self.commands[
            :, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
        ] = self.ee_target_rpy_cmd.view(self.num_envs, 3)
        self.commands[:, INDEX_FORCE_OR_POSITION_INDICATOR] = (
            self.force_or_position_control.view(self.num_envs)
        )

        if self.cfg.commands.interpolate_ee_cmds and len(env_ids_inter):
            time_fraction = self.trajectory_time.view(
                self.num_envs, 1
            ) / t_traj.view(self.num_envs, 1)
            new_command = (
                time_fraction * self.ee_target_pos_cmd.view(self.num_envs, 3)
                + (1 - time_fraction)
                * self.initial_ee_pos.view(self.num_envs, 3)
            )
            self.commands[
                env_ids_inter,
                INDEX_EE_POS_RADIUS_CMD : INDEX_EE_POS_YAW_CMD + 1,
            ] = new_command[env_ids_inter, :]

            drpy = plus_2pi_wrap_to_pi(
                self.ee_target_rpy_cmd - self.initial_ee_rpy
            )
            assert not torch.any(torch.abs(drpy) > np.pi)
            new_rpy_command = plus_2pi_wrap_to_pi(
                self.initial_ee_rpy + time_fraction * drpy
            )
            new_rpy_command[:, 0] = torch.clamp(
                new_rpy_command[:, 0],
                self.cfg.commands.limit_end_effector_roll[0],
                self.cfg.commands.limit_end_effector_roll[1],
            )
            new_rpy_command[:, 1] = torch.clamp(
                new_rpy_command[:, 1],
                self.cfg.commands.limit_end_effector_pitch[0],
                self.cfg.commands.limit_end_effector_pitch[1],
            )
            new_rpy_command[:, 2] = torch.clamp(
                new_rpy_command[:, 2],
                self.cfg.commands.limit_end_effector_yaw[0],
                self.cfg.commands.limit_end_effector_yaw[1],
            )
            self.commands[
                env_ids_inter, INDEX_EE_ROLL_CMD : INDEX_EE_YAW_CMD + 1
            ] = new_rpy_command[env_ids_inter, :]

        env_ids_resample = (
            self.trajectory_time.view(self.num_envs)
            > (t_traj + self.cfg.commands.settle_time)
        ).nonzero(as_tuple=False).flatten()
        if self.cfg.commands.interpolate_ee_cmds and len(env_ids_resample):
            self.trajectory_time[env_ids_resample] = 0.0
            new_radius_cmd = torch_rand_float(
                self.cfg.commands.ee_sphe_radius[0],
                self.cfg.commands.ee_sphe_radius[1],
                (len(env_ids_resample), 1),
                device=self.device,
            ).view(len(env_ids_resample))
            new_pitch_cmd = torch_rand_float(
                self.cfg.commands.ee_sphe_pitch[0],
                self.cfg.commands.ee_sphe_pitch[1],
                (len(env_ids_resample), 1),
                device=self.device,
            ).view(len(env_ids_resample))
            new_yaw_cmd = torch_rand_float(
                self.cfg.commands.ee_sphe_yaw[0],
                self.cfg.commands.ee_sphe_yaw[1],
                (len(env_ids_resample), 1),
                device=self.device,
            ).view(len(env_ids_resample))
            new_roll_ori_cmd = torch_rand_float(
                self.cfg.commands.end_effector_roll[0],
                self.cfg.commands.end_effector_roll[1],
                (len(env_ids_resample), 1),
                device=self.device,
            ).view(len(env_ids_resample))
            new_pitch_ori_cmd = torch_rand_float(
                self.cfg.commands.end_effector_pitch[0],
                self.cfg.commands.end_effector_pitch[1],
                (len(env_ids_resample), 1),
                device=self.device,
            ).view(len(env_ids_resample))
            new_yaw_ori_cmd = torch_rand_float(
                self.cfg.commands.end_effector_yaw[0],
                self.cfg.commands.end_effector_yaw[1],
                (len(env_ids_resample), 1),
                device=self.device,
            ).view(len(env_ids_resample))

            commands_feasible, env_ids_reresample = self.is_ee_cmd_feasible(
                new_radius_cmd, new_pitch_cmd
            )
            while not commands_feasible:
                new_radius_cmd[env_ids_reresample] = torch_rand_float(
                    self.cfg.commands.ee_sphe_radius[0],
                    self.cfg.commands.ee_sphe_radius[1],
                    (len(env_ids_reresample), 1),
                    device=self.device,
                ).view(len(env_ids_reresample))
                new_pitch_cmd[env_ids_reresample] = torch_rand_float(
                    self.cfg.commands.ee_sphe_pitch[0],
                    self.cfg.commands.ee_sphe_pitch[1],
                    (len(env_ids_reresample), 1),
                    device=self.device,
                ).view(len(env_ids_reresample))
                commands_feasible, env_ids_reresample = self.is_ee_cmd_feasible(
                    new_radius_cmd, new_pitch_cmd
                )

            self.initial_ee_pos[env_ids_resample] = self.ee_target_pos_cmd[
                env_ids_resample, :
            ]
            self.initial_ee_rpy[env_ids_resample] = self.ee_target_rpy_cmd[
                env_ids_resample, :
            ]
            self.ee_target_pos_cmd[env_ids_resample, 0] = new_radius_cmd
            self.ee_target_pos_cmd[env_ids_resample, 1] = new_pitch_cmd
            self.ee_target_pos_cmd[env_ids_resample, 2] = new_yaw_cmd
            self.ee_target_rpy_cmd[env_ids_resample, 0] = new_roll_ori_cmd
            self.ee_target_rpy_cmd[env_ids_resample, 1] = new_pitch_ori_cmd
            self.ee_target_rpy_cmd[env_ids_resample, 2] = new_yaw_ori_cmd

        self.trajectory_time += self.dt

    def _init_command_distribution(self, env_ids):
        self.category_names, self.curricula = build_command_curricula(
            self.cfg.commands
        )
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int32)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int32)

        # 0 = position, 1 = force
        self.force_or_position_control = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _resample_force_or_position_control(self, env_ids):
        self.force_or_position_control[env_ids] = sample_control_modes(
            self.cfg.commands.hybrid_mode,
            len(env_ids),
            self.device,
        )
        self.commands[
            env_ids, INDEX_FORCE_OR_POSITION_INDICATOR
        ] = self.force_or_position_control[env_ids]

    def _update_command_ranges(self, env_ids):
        constrict_indices = self.cfg.rewards.constrict_indices
        constrict_ranges = self.cfg.rewards.constrict_ranges

        if (
            self.cfg.rewards.constrict
            and self.common_step_counter >= self.cfg.rewards.constrict_after
        ):
            for idx, value_range in zip(constrict_indices, constrict_ranges):
                self.commands[env_ids, idx] = value_range[0]

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # Update curricula based on terminated environment bins/categories.
        for i, (category, curriculum) in enumerate(
            zip(self.category_names, self.curricula)
        ):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor(
                    [env_ids_in_category], dtype=torch.bool
                )
            elif len(env_ids_in_category) == 0:
                continue

            # Preserve the historical indexing behavior exactly.
            env_ids_in_category = env_ids
            task_rewards, success_thresholds = [], []
            for key in [
                "tracking_lin_vel",
                "tracking_ang_vel",
                "tracking_contacts_shaped_force",
                "tracking_contacts_shaped_vel",
            ]:
                if key in self.command_sums.keys():
                    task_rewards.append(
                        self.command_sums[key][env_ids_in_category] / ep_len
                    )
                    success_thresholds.append(
                        self.curriculum_thresholds[key] * self.reward_scales[key]
                    )

            old_bins = self.env_command_bins[
                env_ids_in_category.cpu().numpy()
            ]
            if len(success_thresholds) > 0:
                curriculum.update(
                    old_bins,
                    task_rewards,
                    success_thresholds,
                    local_range=np.array(
                        [
                            0.55,
                            0.55,
                            0.55,
                            0.55,
                            0.35,
                            0.25,
                            0.25,
                            0.25,
                            0.25,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            0.5,
                            0.5,
                            0.5,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ]
                    ),
                )

        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1.0 / len(self.category_names)
        category_env_ids = [
            env_ids[
                torch.logical_and(
                    probability_per_category * i <= random_env_floats,
                    random_env_floats < probability_per_category * (i + 1),
                )
            ]
            for i in range(len(self.category_names))
        ]

        for i, (category, env_ids_in_category, curriculum) in enumerate(
            zip(self.category_names, category_env_ids, self.curricula)
        ):
            batch_size = len(env_ids_in_category)
            if batch_size == 0:
                continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)
            self.env_command_bins[
                env_ids_in_category.cpu().numpy()
            ] = new_bin_inds
            self.env_command_categories[
                env_ids_in_category.cpu().numpy()
            ] = i
            self.commands[env_ids_in_category, :] = torch.Tensor(
                new_commands[:, : self.cfg.commands.num_commands]
            ).to(self.device)

        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(
                    zip(self.category_names, category_env_ids)
                ):
                    if category == "pronk":
                        self.commands[env_ids_in_category, 5] = (
                            self.commands[env_ids_in_category, 5] / 2 - 0.25
                        ) % 1
                        self.commands[env_ids_in_category, 6] = (
                            self.commands[env_ids_in_category, 6] / 2 - 0.25
                        ) % 1
                        self.commands[env_ids_in_category, 7] = (
                            self.commands[env_ids_in_category, 7] / 2 - 0.25
                        ) % 1
                    elif category == "trot":
                        self.commands[env_ids_in_category, 5] = (
                            self.commands[env_ids_in_category, 5] / 2 + 0.25
                        )
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = (
                            self.commands[env_ids_in_category, 6] / 2 + 0.25
                        )
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = (
                            self.commands[env_ids_in_category, 7] / 2 + 0.25
                        )

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[
                    torch.logical_and(
                        0.34 <= random_env_floats,
                        random_env_floats < 0.67,
                    )
                ]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[
                    torch.logical_and(
                        0.25 <= random_env_floats,
                        random_env_floats < 0.50,
                    )
                ]
                pacing_envs = env_ids[
                    torch.logical_and(
                        0.50 <= random_env_floats,
                        random_env_floats < 0.75,
                    )
                ]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (
                    self.commands[pronking_envs, 5] / 2 - 0.25
                ) % 1
                self.commands[pronking_envs, 6] = (
                    self.commands[pronking_envs, 6] / 2 - 0.25
                ) % 1
                self.commands[pronking_envs, 7] = (
                    self.commands[pronking_envs, 7] / 2 - 0.25
                ) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = (
                    self.commands[trotting_envs, 5] / 2 + 0.25
                )
                self.commands[pacing_envs, 6] = (
                    self.commands[pacing_envs, 6] / 2 + 0.25
                )
                self.commands[bounding_envs, 7] = (
                    self.commands[bounding_envs, 7] / 2 + 0.25
                )

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (
                    torch.round(2 * self.commands[env_ids, 5]) / 2.0 % 1
                )
                self.commands[env_ids, 6] = (
                    torch.round(2 * self.commands[env_ids, 6]) / 2.0 % 1
                )
                self.commands[env_ids, 7] = (
                    torch.round(2 * self.commands[env_ids, 7]) / 2.0 % 1
                )

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.0

        self._update_command_ranges(env_ids)

        if self.cfg.commands.heading_command:
            self.heading_commands[env_ids] = torch_rand_float(
                self.cfg.commands.heading[0],
                self.cfg.commands.heading[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # Preserve the original last-category indexing for fixed gait commands.
        if (
            self.cfg.commands.gait_phase_cmd_range[0]
            == self.cfg.commands.gait_phase_cmd_range[1]
        ):
            self.commands[env_ids_in_category, 5] = (
                self.cfg.commands.gait_phase_cmd_range[0]
            )
        if (
            self.cfg.commands.gait_offset_cmd_range[0]
            == self.cfg.commands.gait_offset_cmd_range[1]
        ):
            self.commands[env_ids_in_category, 6] = (
                self.cfg.commands.gait_offset_cmd_range[0]
            )
        if (
            self.cfg.commands.gait_bound_cmd_range[0]
            == self.cfg.commands.gait_bound_cmd_range[1]
        ):
            self.commands[env_ids_in_category, 7] = (
                self.cfg.commands.gait_bound_cmd_range[0]
            )

    def _step_contact_targets(self):
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        if self.cfg.commands.pacing_offset:
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + bounds,
                self.gait_indices + offsets,
                self.gait_indices + phases,
            ]
        else:
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            ]

        self.foot_indices = torch.remainder(
            torch.cat(
                [foot_indices[i].unsqueeze(1) for i in range(4)], dim=1
            ),
            1.0,
        )

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(
                idxs[stance_idxs], 1
            ) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (
                torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]
            ) * (0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])
        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        def smoothing_multiplier(index):
            wrapped = torch.remainder(index, 1.0)
            return (
                smoothing_cdf_start(wrapped)
                * (1 - smoothing_cdf_start(wrapped - 0.5))
                + smoothing_cdf_start(wrapped - 1)
                * (1 - smoothing_cdf_start(wrapped - 1.5))
            )

        self.desired_contact_states[:, 0] = smoothing_multiplier(foot_indices[0])
        self.desired_contact_states[:, 1] = smoothing_multiplier(foot_indices[1])
        self.desired_contact_states[:, 2] = smoothing_multiplier(foot_indices[2])
        self.desired_contact_states[:, 3] = smoothing_multiplier(foot_indices[3])

        env_ids = torch.arange(self.num_envs, device=self.device)
        static_env_ids = env_ids[
            torch.logical_and(
                torch.logical_and(
                    torch.abs(self.commands[:, 0]) < 0.2,
                    torch.abs(self.commands[:, 1]) < 0.2,
                ),
                torch.abs(self.commands[:, 2]) < 0.2,
            )
        ]
        self.desired_contact_states[static_env_ids, :] = 1.0
        self.clock_inputs[static_env_ids, :] = 1.0

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]
