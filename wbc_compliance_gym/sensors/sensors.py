"""B1+Z1 observation sensors and the stable configuration catalog.

The task uses many small observation adapters.  Their implementations live in
this single file so observation composition can be reviewed in one place.  The
historical per-sensor modules re-export these exact class objects.
"""

import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
)
import torch

from wbc_compliance_gym.utils.math_utils import get_scale_shift, quat_apply_yaw, wrap_to_pi


class Sensor:
    def __init__(self, env):
        self.env = env

    def get_observation(self):
        raise NotImplementedError

    def get_noise_vec(self):
        raise NotImplementedError

    def get_dim(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


class AttachedCameraSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def initialize(
        self, camera_label, camera_pose, camera_rpy, camera_gimbal, env_ids=None
    ):
        if env_ids is None:
            env_ids = range(self.env.num_envs)

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.env.cfg.perception.image_width
        camera_props.height = self.env.cfg.perception.image_height
        camera_props.horizontal_fov = self.env.cfg.perception.image_horizontal_fov
        self.cams = []

        for env_id in env_ids:
            cam = self.env.gym.create_camera_sensor(
                self.env.envs[env_id], camera_props
            )
            trans_pos = gymapi.Vec3(
                camera_pose[0], camera_pose[1], camera_pose[2]
            )
            quat_pitch = quat_from_angle_axis(
                torch.Tensor([-camera_rpy[1]]), torch.Tensor([0, 1, 0])
            )[0]
            quat_yaw = quat_from_angle_axis(
                torch.Tensor([camera_rpy[2]]), torch.Tensor([0, 0, 1])
            )[0]
            quat = quat_mul(quat_yaw, quat_pitch)
            trans_quat = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
            transform = gymapi.Transform(trans_pos, trans_quat)
            if camera_gimbal:
                follow_mode = gymapi.CameraFollowMode.FOLLOW_POSITION
            else:
                follow_mode = gymapi.CameraFollowMode.FOLLOW_TRANSFORM
            self.env.gym.attach_camera_to_body(
                cam, self.env.envs[env_id], 0, transform, follow_mode
            )
            self.cams.append(cam)
        return self.cams

    def get_observation(self, env_ids=None):
        raise NotImplementedError

    def get_depth_images(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.env.num_envs)
        depth_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(
                self.env.sim,
                self.env.envs[env_id],
                self.cams[env_id],
                gymapi.IMAGE_DEPTH,
            )
            width, height = img.shape
            depth_images.append(
                torch.from_numpy(img.reshape([1, width, height])).to(
                    self.env.device
                )
            )
        return torch.cat(depth_images, dim=0)

    def get_rgb_images(self, env_ids):
        if env_ids is None:
            env_ids = range(self.env.num_envs)
        rgb_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(
                self.env.sim,
                self.env.envs[env_id],
                self.cams[env_id],
                gymapi.IMAGE_COLOR,
            )
            width, height = img.shape
            rgb_images.append(
                torch.from_numpy(
                    img.reshape([1, width, height // 4, 4]).astype(np.int32)
                ).to(self.env.device)
            )
        return torch.cat(rgb_images, dim=0)

    def get_segmentation_images(self, env_ids):
        if env_ids is None:
            env_ids = range(self.env.num_envs)
        segmentation_images = []
        for env_id in env_ids:
            img = self.env.gym.get_camera_image(
                self.env.sim,
                self.env.envs[env_id],
                self.cams[env_id],
                gymapi.IMAGE_SEGMENTATION,
            )
            width, height = img.shape
            segmentation_images.append(
                torch.from_numpy(
                    img.reshape([1, width, height]).astype(np.int32)
                ).to(self.env.device)
            )
        return torch.cat(segmentation_images, dim=0)


class FloatingCameraSensor(Sensor):
    def __init__(self, env, env_idx=0):
        super().__init__(env)
        self.env = env
        self.env_idx = env_idx

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.env.cfg.env.recording_width_px
        camera_props.height = self.env.cfg.env.recording_height_px
        self.rendering_camera = self.env.gym.create_camera_sensor(
            self.env.envs[self.env_idx], camera_props
        )
        self.env.gym.set_camera_location(
            self.rendering_camera,
            self.env.envs[self.env_idx],
            gymapi.Vec3(1.5, 1, 3.0),
            gymapi.Vec3(0, 0, 0),
        )

    def set_position(self, target_loc=None, cam_distance=None):
        if cam_distance is None:
            cam_distance = [0, -1.0, 1.0]
        if target_loc is None:
            bx = self.env.root_states[self.env_idx, 0]
            by = self.env.root_states[self.env_idx, 1]
            bz = self.env.root_states[self.env_idx, 2]
            target_loc = [bx, by, bz]
        self.env.gym.set_camera_location(
            self.rendering_camera,
            self.env.envs[self.env_idx],
            gymapi.Vec3(
                target_loc[0] + cam_distance[0],
                target_loc[1] + cam_distance[1],
                target_loc[2] + cam_distance[2],
            ),
            gymapi.Vec3(target_loc[0], target_loc[1], target_loc[2]),
        )

    def get_observation(self, env_ids=None):
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        img = self.env.gym.get_camera_image(
            self.env.sim,
            self.env.envs[self.env_idx],
            self.rendering_camera,
            gymapi.IMAGE_COLOR,
        )
        width, height = img.shape
        return img.reshape([width, height // 4, 4])


# ---------------------------------------------------------------------------
# Policy observations
# ---------------------------------------------------------------------------


class JointPositionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        obs = (
            self.env.dof_pos[:, : self.env.num_actuated_dof]
            - self.env.default_dof_pos[:, : self.env.num_actuated_dof]
        ) * self.env.cfg.obs_scales.dof_pos
        if self.env.cfg.commands.control_only_z1:
            obs[:, :12] = 0.0
        return obs

    def get_noise_vec(self):
        noise_vec = (
            torch.ones(self.env.num_actuated_dof, device=self.env.device)
            * self.env.cfg.noise_scales.dof_pos
            * self.env.cfg.noise.noise_level
            * self.env.cfg.obs_scales.dof_pos
        )
        if self.env.cfg.commands.control_only_z1:
            noise_vec[:12] = 0.0
        return noise_vec

    def get_dim(self):
        return self.env.num_actuated_dof


class JointVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        obs = (
            self.env.dof_vel[:, : self.env.num_actuated_dof]
            * self.env.cfg.obs_scales.dof_vel
        )
        if self.env.cfg.commands.control_only_z1:
            obs[:, :12] = 0.0
        return obs

    def get_noise_vec(self):
        noise_vec = (
            torch.ones(self.env.num_actuated_dof, device=self.env.device)
            * self.env.cfg.noise_scales.dof_vel
            * self.env.cfg.noise.noise_level
            * self.env.cfg.obs_scales.dof_vel
        )
        if self.env.cfg.commands.control_only_z1:
            noise_vec[:12] = 0.0
        return noise_vec

    def get_dim(self):
        return self.env.num_actuated_dof


class JointPositionTargetSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        obs = self.env.joint_pos_target
        if self.env.cfg.commands.control_only_z1:
            obs[:, :12] = 0.0
        return obs

    def get_noise_vec(self):
        return torch.zeros(self.env.num_actuated_dof, device=self.env.device)

    def get_dim(self):
        return self.env.num_actuated_dof


class OrientationSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        return self.env.projected_gravity

    def get_noise_vec(self):
        return (
            torch.ones(3, device=self.env.device)
            * self.env.cfg.noise_scales.gravity
            * self.env.cfg.noise.noise_level
        )

    def get_dim(self):
        return 3


class HeightmapSensor(Sensor):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.env.device)
        if self.env.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.env.num_envs,
                self.env.cfg.perception.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        if self.env.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(
            self.base_quat[env_ids].repeat(
                1, self.env.cfg.perception.num_height_points
            ),
            self.height_points[env_ids],
        ) + self.env.root_states[
            self.robot_actor_idxs[env_ids], :3
        ].unsqueeze(1)
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.env.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.env.height_samples.shape[1] - 2)
        heights1 = self.env.height_samples[px, py]
        heights2 = self.env.height_samples[px + 1, py]
        heights3 = self.env.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        return heights.view(len(env_ids), -1) * self.env.terrain.cfg.vertical_scale


class RCSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        if (
            self.env.cfg.commands.gait_phase_cmd_range[0]
            == self.env.cfg.commands.gait_phase_cmd_range[1]
        ):
            self.env.commands[:, 5] = (
                self.env.cfg.commands.gait_phase_cmd_range[0]
            )
        if (
            self.env.cfg.commands.gait_offset_cmd_range[0]
            == self.env.cfg.commands.gait_offset_cmd_range[1]
        ):
            self.env.commands[:, 6] = (
                self.env.cfg.commands.gait_offset_cmd_range[0]
            )
        if (
            self.env.cfg.commands.gait_bound_cmd_range[0]
            == self.env.cfg.commands.gait_bound_cmd_range[1]
        ):
            self.env.commands[:, 7] = (
                self.env.cfg.commands.gait_bound_cmd_range[0]
            )

        force_control_envs = self.env.force_or_position_control == 1
        obs_commands = self.env.commands * self.env.commands_scale
        obs_commands[force_control_envs, 15:18] = 0
        return obs_commands

    def get_noise_vec(self):
        return torch.zeros(
            self.env.cfg.commands.num_commands, device=self.env.device
        )

    def get_dim(self):
        return self.env.cfg.commands.num_commands


class ActionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.delay = delay

    def get_observation(self, env_ids=None):
        if self.delay == 0:
            return self.env.actions
        if self.delay == 1:
            return self.env.last_actions
        raise NotImplementedError(
            "Action delay of {} not implemented".format(self.delay)
        )

    def get_noise_vec(self):
        return torch.zeros(self.env.num_actions, device=self.env.device)

    def get_dim(self):
        return self.env.num_actions


class LastActionSensor(ActionSensor):
    """Historical duplicate of :class:`ActionSensor`."""


class ClockSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        return self.env.clock_inputs

    def get_noise_vec(self):
        return torch.zeros(4, device=self.env.device)

    def get_dim(self):
        return 4


class YawSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        return wrap_to_pi(heading - self.env.heading_offsets.unsqueeze(1))

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 1


class ObjectSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        return self.env.object_local_pos * self.env.cfg.obs_scales.ball_pos

    def get_noise_vec(self):
        return (
            torch.ones(3, device=self.env.device)
            * self.env.cfg.noise_scales.ball_pos
            * self.env.cfg.noise.noise_level
            * self.env.cfg.obs_scales.ball_pos
        )

    def get_dim(self):
        return 3


class TimingSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return self.env.gait_indices.unsqueeze(1)

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 1


# ---------------------------------------------------------------------------
# Privileged observations
# ---------------------------------------------------------------------------


class BodyVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        return self.env.base_lin_vel

    def get_dim(self):
        return 3


class ObjectVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return self.env.object_lin_vel

    def get_dim(self):
        return 3


class RestitutionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        scale, shift = get_scale_shift(
            self.env.cfg.normalization.restitution_range
        )
        return (self.env.restitutions[:, 0].unsqueeze(1) - shift) * scale

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 1


class FrictionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        scale, shift = get_scale_shift(
            self.env.cfg.normalization.friction_range
        )
        return (self.env.friction_coeffs[:, 0].unsqueeze(1) - shift) * scale

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 1


class GroundFrictionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        ground_friction_coeffs = self.env.get_ground_frictions(
            range(self.env.num_envs)
        )
        scale, shift = get_scale_shift(
            self.env.cfg.normalization.ground_friction_range
        )
        return (ground_friction_coeffs.unsqueeze(1) - shift) * scale

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)


class GroundRoughnessSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        ground_roughness = self.env.get_ground_roughness(
            range(self.env.num_envs)
        )
        scale, shift = get_scale_shift(
            self.env.cfg.normalization.roughness_range
        )
        return (ground_roughness.unsqueeze(1) - shift) * scale

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)


class EgomotionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        motion_scale, motion_shift = get_scale_shift(
            self.env.cfg.normalization.motion
        )
        global_body_motion = self.env.base_pos - self.env.prev_base_pos
        global_body_motion = quat_apply_yaw(
            quat_from_angle_axis(
                -1 * self.env.heading_offsets.unsqueeze(1),
                torch.Tensor([0, 0, 1]).to(self.env.device),
            )[:, 0, :],
            global_body_motion,
        )
        reset_env_ids = self.env.reset_buf.nonzero(as_tuple=False).flatten()
        global_body_motion[reset_env_ids] = 0.0
        global_body_motion[global_body_motion > 0.5] = 0.0
        global_body_motion[global_body_motion < -0.5] = 0.0
        return (global_body_motion - motion_shift) * motion_scale

    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)


class JointDynamicsSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids=None):
        return torch.cat(
            (
                self.env.Kp_factors[:, 0:1],
                self.env.Kd_factors[:, 0:1],
                self.env.motor_strengths[:, 0:1],
            ),
            dim=1,
        )

    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)


# ---------------------------------------------------------------------------
# End-effector and force observations
# ---------------------------------------------------------------------------


class EeGripperForceSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        forces_global = self.env.forces[
            :, self.env.gripper_stator_index, 0:3
        ]
        base_quat_world = self.env.base_quat.view(self.env.num_envs, 4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(
            0 * base_rpy_world[:, 0],
            0 * base_rpy_world[:, 1],
            base_rpy_world[:, 2],
        )
        forces_local = quat_rotate_inverse(
            base_quat_world_indep, forces_global
        )
        return forces_local.view(self.env.num_envs, 3)

    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)

    def get_dim(self):
        return 3


class EeGripperForceMagnSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return torch.norm(
            self.env.forces[:, self.env.gripper_stator_index, :2], dim=1
        ).view(self.env.num_envs, 1)

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 1


class EeGripperForceDirSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return torch.atan2(
            self.env.forces[:, self.env.gripper_stator_index, 1],
            self.env.forces[:, self.env.gripper_stator_index, 0],
        ).view(self.env.num_envs, 1)

    def get_noise_vec(self):
        return torch.zeros(1, device=self.env.device)

    def get_dim(self):
        return 1


class EeBaseForceSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return self.env.forces[:, self.env.robot_base_index, :3]

    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)

    def get_dim(self):
        return 1


class EeGripperPositionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return self.env.get_measured_ee_pos_spherical()

    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)

    def get_dim(self):
        return 3


class EeGripperTargetPositionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env

    def get_observation(self, env_ids=None):
        return self.env.commands[:, 15:18]

    def get_noise_vec(self):
        return torch.zeros(3, device=self.env.device)

    def get_dim(self):
        return 3


SENSOR_TYPES = {
    sensor_type.__name__: sensor_type
    for sensor_type in (
        AttachedCameraSensor,
        FloatingCameraSensor,
        JointPositionSensor,
        JointVelocitySensor,
        JointPositionTargetSensor,
        OrientationSensor,
        HeightmapSensor,
        RCSensor,
        ActionSensor,
        LastActionSensor,
        ClockSensor,
        YawSensor,
        ObjectSensor,
        TimingSensor,
        BodyVelocitySensor,
        ObjectVelocitySensor,
        RestitutionSensor,
        FrictionSensor,
        GroundFrictionSensor,
        GroundRoughnessSensor,
        EgomotionSensor,
        EeGripperForceSensor,
        EeGripperForceMagnSensor,
        EeGripperForceDirSensor,
        EeBaseForceSensor,
        JointDynamicsSensor,
        EeGripperPositionSensor,
        EeGripperTargetPositionSensor,
    )
}
ALL_SENSORS = SENSOR_TYPES


def make_sensor(name, env, args=None):
    try:
        sensor_type = SENSOR_TYPES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SENSOR_TYPES))
        raise ValueError(
            f"Sensor {name!r} not found; available: {available}"
        ) from exc
    return sensor_type(env, **(args or {}))


__all__ = ["ALL_SENSORS", "SENSOR_TYPES", "Sensor", "make_sensor"] + sorted(
    SENSOR_TYPES
)
