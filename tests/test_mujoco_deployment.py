import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from wbc_compliance_rl.utils.deployment_bundle import (
    COMMAND_NAMES,
    _format_flat,
    _parse_urdf_contract,
)


ROOT = Path(__file__).resolve().parents[1]


class MujocoDeploymentTest(unittest.TestCase):
    def test_command_schema_is_the_training_schema(self):
        self.assertEqual(len(COMMAND_NAMES), 23)
        self.assertEqual(COMMAND_NAMES[12:15], (
            "ee_force_x", "ee_force_y", "ee_force_z"
        ))
        self.assertEqual(COMMAND_NAMES[15:18], (
            "ee_pos_radius", "ee_pos_pitch", "ee_pos_yaw"
        ))
        self.assertEqual(COMMAND_NAMES[22], "force_or_position")
        self.assertEqual(_format_flat([1.0, float("nan"), "wheel"]),
                         "1,nan,wheel")

    def test_b1_urdf_policy_joint_order(self):
        path = ROOT / "resources/robots/b1_z1/urdf/b1_plus_z1.urdf"
        joints = _parse_urdf_contract(path, 19)
        names = [joint["name"] for joint in joints]
        self.assertEqual(names[:3], [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"
        ])
        self.assertEqual(names[-7:], [
            "joint1", "joint2", "joint3", "joint4", "joint5",
            "joint6", "jointGripper",
        ])

    def test_generated_scenes_keep_both_action_contracts(self):
        expected = {"zgwsarm": 22, "b1_z1": 19}
        end_effector = {
            "zgwsarm": "ROBOT_ARM_LINK7",
            "b1_z1": "gripperStator",
        }
        variants = {
            "zgwsarm": ("flat", "wall", "block", "terrain"),
            "b1_z1": ("flat", "wall", "block"),
        }
        for task, actions in expected.items():
            for variant in variants[task]:
                path = ROOT / f"mujoco/models/{task}/scene_{variant}.xml"
                root = ET.parse(path).getroot()
                included = None
                include = root.find("include")
                if include is not None:
                    included = ET.parse(path.parent / include.attrib["file"]).getroot()
                contract_root = included if included is not None else root
                motors = contract_root.findall("./actuator/motor")
                self.assertEqual(len(motors), actions, path)
                self.assertIsNotNone(
                    contract_root.find(f".//body[@name='{end_effector[task]}']"), path
                )
                self.assertIsNotNone(
                    root.find("./worldbody/geom[@name='floor']"), path
                )
                if variant in {"wall", "block"}:
                    self.assertIsNotNone(
                        root.find("./worldbody/geom[@name='contact_wall']"),
                        path,
                    )

    def test_zgwsarm_terrain_assets_are_repository_local(self):
        path = ROOT / "mujoco/models/zgwsarm/scene_terrain.xml"
        root = ET.parse(path).getroot()
        self.assertIsNotNone(root.find("./asset/texture[@type='skybox']"))
        self.assertIsNotNone(root.find("./worldbody/light"))
        hfields = root.findall("./asset/hfield")
        self.assertEqual(len(hfields), 2)
        for hfield in hfields:
            image = Path(hfield.attrib["file"])
            self.assertFalse(image.is_absolute())
            self.assertNotIn("..", image.parts)
            self.assertTrue((path.parent / image).is_file())


if __name__ == "__main__":
    unittest.main()
