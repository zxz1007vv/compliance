"""Isaac Gym asset loader for the ZGWSARM wheel-legged manipulator."""

import os

from isaacgym import gymapi

from wbc_compliance_gym import WBC_COMPLIANCE_ROOT_DIR

from .robot import Robot


EXPECTED_DOF_NAMES = (
    "FAR_ABAD_JOINT",
    "FAR_HIP_JOINT",
    "FAR_KNEE_JOINT",
    "FAR_FOOT_JOINT",
    "FBL_ABAD_JOINT",
    "FBL_HIP_JOINT",
    "FBL_KNEE_JOINT",
    "FBL_FOOT_JOINT",
    "RAR_ABAD_JOINT",
    "RAR_HIP_JOINT",
    "RAR_KNEE_JOINT",
    "RAR_FOOT_JOINT",
    "RBL_ABAD_JOINT",
    "RBL_HIP_JOINT",
    "RBL_KNEE_JOINT",
    "RBL_FOOT_JOINT",
    "ROBOT_ARM_JOINT1",
    "ROBOT_ARM_JOINT2",
    "ROBOT_ARM_JOINT3",
    "ROBOT_ARM_JOINT4",
    "ROBOT_ARM_JOINT5",
    "ROBOT_ARM_JOINT6",
)


class ZGWSArm(Robot):
    def initialize(self):
        configured_path = self.env.cfg.asset.file.replace(
            "{WBC_COMPLIANCE_ROOT_DIR}", WBC_COMPLIANCE_ROOT_DIR
        )
        asset_path = os.path.abspath(configured_path)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = (
            self.env.cfg.asset.default_dof_drive_mode
        )
        asset_options.collapse_fixed_joints = self.env.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.env.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = (
            self.env.cfg.asset.flip_visual_attachments
        )
        asset_options.fix_base_link = self.env.cfg.asset.fix_base_link
        asset_options.density = self.env.cfg.asset.density
        asset_options.angular_damping = self.env.cfg.asset.angular_damping
        asset_options.linear_damping = self.env.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.env.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.env.cfg.asset.max_linear_velocity
        asset_options.armature = self.env.cfg.asset.armature
        asset_options.thickness = self.env.cfg.asset.thickness
        asset_options.disable_gravity = self.env.cfg.asset.disable_gravity
        asset_options.vhacd_enabled = False

        asset = self.env.gym.load_asset(
            self.env.sim, asset_root, asset_file, asset_options
        )
        if asset is None:
            raise RuntimeError(f"Isaac Gym failed to load ZGWSARM asset: {asset_path}")

        actual_names = tuple(self.env.gym.get_asset_dof_names(asset))
        if actual_names != EXPECTED_DOF_NAMES:
            raise ValueError(
                "ZGWSARM URDF DOF order changed; policy action order would be unsafe. "
                f"expected={EXPECTED_DOF_NAMES}, actual={actual_names}"
            )

        self.num_dof = self.env.gym.get_asset_dof_count(asset)
        self.num_actuated_dof = len(EXPECTED_DOF_NAMES)
        self.num_bodies = self.env.gym.get_asset_rigid_body_count(asset)
        dof_props_asset = self.env.gym.get_asset_dof_properties(asset)
        rigid_shape_props_asset = self.env.gym.get_asset_rigid_shape_properties(asset)
        return asset, dof_props_asset, rigid_shape_props_asset


__all__ = ["EXPECTED_DOF_NAMES", "ZGWSArm"]
