"""Generate deterministic MuJoCo scenes for both C++ sim2sim tasks.

ZGWSARM starts from its checked-in MJCF because it already carries the exact
mesh/inertial data.  B1+Z1 is converted from the training URDF; all active
collision geometry in that URDF is primitive, so no visual-mesh conversion is
needed for dynamics/contact fidelity.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "mujoco" / "models"


def indent_xml(element, level=0):
    """Indent an ElementTree on both the Python 3.8 training env and newer Python."""
    if hasattr(ET, "indent"):
        ET.indent(element, space="  ")
        return
    whitespace = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = whitespace + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = whitespace
    if level and (not element.tail or not element.tail.strip()):
        element.tail = whitespace


def vector(node, attribute, default):
    if node is None:
        return default
    return node.attrib.get(attribute, default)


def set_origin(element, origin):
    if origin is None:
        return
    xyz = origin.attrib.get("xyz")
    rpy = origin.attrib.get("rpy")
    if xyz and any(abs(float(value)) > 1e-14 for value in xyz.split()):
        element.set("pos", xyz)
    if rpy and any(abs(float(value)) > 1e-14 for value in rpy.split()):
        element.set("euler", rpy)


def add_visual_environment(model):
    """Add a consistent lit sky and checkerboard ground material."""
    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "headlight",
        diffuse="0.7 0.7 0.7",
        ambient="0.35 0.35 0.35",
        specular="0.1 0.1 0.1",
    )
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="-130", elevation="-20")
    ET.SubElement(visual, "map", znear="0.01", zfar="100")

    option_index = next(
        (index for index, child in enumerate(model) if child.tag == "option"),
        0,
    )
    model.insert(option_index + 1, visual)

    asset = model.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = next(
            index for index, child in enumerate(model) if child.tag == "worldbody"
        )
        model.insert(worldbody_index, asset)
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.3 0.5 0.7",
        rgb2="0.02 0.02 0.03",
        width="512",
        height="3072",
    )
    ET.SubElement(
        asset,
        "texture",
        type="2d",
        name="groundplane_texture",
        builtin="checker",
        mark="edge",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        markrgb="0.8 0.8 0.8",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset,
        "material",
        name="groundplane_material",
        texture="groundplane_texture",
        texuniform="true",
        texrepeat="5 5",
        reflectance="0.2",
    )


def add_scene_geometry(worldbody, variant):
    ET.SubElement(
        worldbody,
        "light",
        pos="0 0 3",
        dir="0 0 -1",
        directional="true",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="8 8 0.1",
        material="groundplane_material",
        friction="0.9 0.02 0.01",
        condim="3",
    )
    if variant in {"wall", "block"}:
        ET.SubElement(
            worldbody,
            "geom",
            name="contact_wall",
            type="box",
            pos="0.82 0 0.55",
            size="0.035 0.60 0.55",
            rgba="0.38 0.43 0.52 1",
            friction="0.8 0.02 0.01",
        )
    if variant == "block":
        ET.SubElement(
            worldbody,
            "body",
            name="manipulation_block",
            pos="0.67 0 0.12",
        ).extend(
            [
                ET.Element(
                    "freejoint", name="manipulation_block_free"
                ),
                ET.Element(
                    "geom",
                    name="manipulation_block_geom",
                    type="box",
                    size="0.10 0.10 0.10",
                    mass="1.0",
                    rgba="0.85 0.42 0.12 1",
                    friction="0.8 0.02 0.01",
                ),
            ]
        )


def write_tree(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    indent_xml(root)
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")


def generate_zgwsarm():
    source = ROOT / "resources" / "robots" / "zgwsarm" / "zgwsarm.xml"
    base = ET.parse(source).getroot()
    base.set("model", "zgwsarm_sim2sim")
    compiler = base.find("compiler")
    # Resolve all assets from the generated XML directory.  Explicit mesh
    # paths keep MuJoCo's meshdir from being applied to terrain hfield images
    # declared by a scene that includes this robot file.
    compiler.set("meshdir", ".")
    compiler.set("autolimits", "true")
    for mesh in base.findall("./asset/mesh"):
        mesh.set(
            "file",
            f"../../../resources/robots/zgwsarm/assets/{mesh.attrib['file']}",
        )
    option = base.find("option")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    link6 = base.find(".//body[@name='ROBOT_ARM_LINK6']")
    if link6 is None:
        raise RuntimeError("ZGWSARM MJCF is missing ROBOT_ARM_LINK6")
    tip = ET.SubElement(link6, "body", name="ROBOT_ARM_LINK7", pos="0 0 0.025")
    ET.SubElement(
        tip,
        "geom",
        name="end_effector_contact",
        type="sphere",
        size="0.035",
        density="0",
        rgba="0.95 0.25 0.12 0.75",
        friction="0.8 0.02 0.01",
    )
    ET.SubElement(tip, "site", name="end_effector_site", size="0.012", rgba="1 0 0 1")

    # Runtime reset writes the exported training defaults by joint name.  Keep
    # the keyframe useful for standalone inspection too.
    key = base.find("./keyframe/key[@name='home']")
    if key is not None:
        qpos = [float(value) for value in key.attrib["qpos"].split()]
        qpos[-6:] = [0.0, 0.8, -1.5, 0.0, 0.0, 0.0]
        key.set("qpos", " ".join(format(value, ".9g") for value in qpos))

    # scene_terrain.xml includes this robot-only file.  Keeping it generated
    # from the same source as the built-in scenes prevents the terrain scene
    # from drifting away from the runtime's joint and end-effector contract.
    write_tree(base, OUTPUT / "zgwsarm" / "zgwsarm.xml")

    terrain_root = ET.parse(source.parent / "scene_terrain.xml").getroot()
    write_tree(terrain_root, OUTPUT / "zgwsarm" / "scene_terrain.xml")
    for image_name in ("height_field.png", "unitree_hfield.png"):
        shutil.copyfile(
            source.parent / image_name,
            OUTPUT / "zgwsarm" / image_name,
        )

    for variant in ("flat", "wall", "block"):
        root = copy.deepcopy(base)
        add_visual_environment(root)
        worldbody = root.find("worldbody")
        # World geoms must precede bodies in MJCF's schema.
        robot_body = list(worldbody)
        worldbody.clear()
        add_scene_geometry(worldbody, variant)
        worldbody.extend(robot_body)
        write_tree(root, OUTPUT / "zgwsarm" / f"scene_{variant}.xml")


def parse_urdf(path):
    # See deployment_bundle.py: the already-expanded file has one harmless
    # legacy xacro prefix without a namespace declaration.
    return ET.fromstring(path.read_text(encoding="utf-8").replace("xacro:", "xacro_"))


def inertia_element(link, body):
    inertial = link.find("inertial")
    if inertial is None:
        return
    mass = inertial.find("mass")
    tensor = inertial.find("inertia")
    if mass is None or tensor is None:
        return
    output = ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=mass.attrib["value"],
        fullinertia=" ".join(
            tensor.attrib[name]
            for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    )
    set_origin(output, inertial.find("origin"))


def collision_elements(link, body):
    for index, collision in enumerate(link.findall("collision")):
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1:
            continue
        shape = geometry[0]
        attributes = {
            "name": f"{link.attrib['name']}_collision_{index}",
            "rgba": "0.55 0.58 0.64 1",
            "friction": "0.8 0.02 0.01",
            "condim": "3",
        }
        if shape.tag == "box":
            size = [0.5 * float(value) for value in shape.attrib["size"].split()]
            attributes.update(type="box", size=" ".join(format(value, ".9g") for value in size))
        elif shape.tag == "sphere":
            attributes.update(type="sphere", size=shape.attrib["radius"])
        elif shape.tag == "cylinder":
            attributes.update(
                type="cylinder",
                size=f"{shape.attrib['radius']} {0.5 * float(shape.attrib['length']):.9g}",
            )
        else:
            raise RuntimeError(
                f"Unsupported active B1+Z1 collision geometry {shape.tag} on {link.attrib['name']}"
            )
        output = ET.SubElement(body, "geom", attributes)
        set_origin(output, collision.find("origin"))


def generate_b1_z1():
    source = ROOT / "resources" / "robots" / "b1_z1" / "urdf" / "b1_plus_z1.urdf"
    urdf = parse_urdf(source)
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    joints = [joint for joint in urdf.findall("joint") if joint.find("parent") is not None]
    children = {joint.find("child").attrib["link"] for joint in joints}
    roots = set(links) - children
    if roots != {"base"}:
        raise RuntimeError(f"Unexpected B1+Z1 root links: {sorted(roots)}")
    by_parent = {}
    for joint in joints:
        by_parent.setdefault(joint.find("parent").attrib["link"], []).append(joint)

    model = ET.Element("mujoco", model="b1_z1_sim2sim")
    ET.SubElement(model, "compiler", angle="radian", autolimits="true", balanceinertia="true")
    ET.SubElement(
        model,
        "option",
        timestep="0.005",
        gravity="0 0 -9.81",
        integrator="implicitfast",
        solver="Newton",
    )
    default = ET.SubElement(model, "default")
    ET.SubElement(default, "joint", damping="0", armature="0", frictionloss="0")
    ET.SubElement(default, "geom", margin="0.001")
    worldbody = ET.SubElement(model, "worldbody")
    actuator = ET.SubElement(model, "actuator")

    def add_link(link_name, parent_body, incoming=None):
        link = links[link_name]
        body = ET.SubElement(parent_body, "body", name=link_name)
        if incoming is None:
            body.set("pos", "0 0 0.6")
            ET.SubElement(body, "freejoint", name="base_free")
            ET.SubElement(body, "site", name="imu", size="0.01")
        else:
            set_origin(body, incoming.find("origin"))
            joint_type = incoming.attrib.get("type")
            if joint_type in {"revolute", "continuous"}:
                attributes = {"name": incoming.attrib["name"], "type": "hinge"}
                axis = incoming.find("axis")
                attributes["axis"] = vector(axis, "xyz", "1 0 0")
                limit = incoming.find("limit")
                if joint_type == "revolute" and limit is not None:
                    attributes["range"] = f"{limit.attrib['lower']} {limit.attrib['upper']}"
                if limit is not None and "effort" in limit.attrib:
                    effort = float(limit.attrib["effort"])
                    attributes["actuatorfrcrange"] = f"{-effort:.9g} {effort:.9g}"
                dynamics = incoming.find("dynamics")
                if dynamics is not None:
                    if "damping" in dynamics.attrib:
                        attributes["damping"] = dynamics.attrib["damping"]
                    if "friction" in dynamics.attrib:
                        attributes["frictionloss"] = dynamics.attrib["friction"]
                ET.SubElement(body, "joint", attributes)
                motor_attributes = {"name": f"{incoming.attrib['name']}_motor", "joint": incoming.attrib["name"]}
                if limit is not None and "effort" in limit.attrib:
                    effort = float(limit.attrib["effort"])
                    motor_attributes["ctrlrange"] = f"{-effort:.9g} {effort:.9g}"
                ET.SubElement(actuator, "motor", motor_attributes)
        inertia_element(link, body)
        collision_elements(link, body)
        if link_name == "gripperStator":
            ET.SubElement(
                body,
                "geom",
                name="end_effector_contact",
                type="sphere",
                pos="0.10 0 0",
                size="0.035",
                density="0",
                rgba="0.95 0.25 0.12 0.75",
            )
            ET.SubElement(body, "site", name="end_effector_site", pos="0.10 0 0", size="0.012")
        for child_joint in by_parent.get(link_name, []):
            add_link(child_joint.find("child").attrib["link"], body, child_joint)

    add_link("base", worldbody)
    sensor = ET.SubElement(model, "sensor")
    ET.SubElement(sensor, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensor, "accelerometer", name="imu_acc", site="imu")

    for variant in ("flat", "wall", "block"):
        root = copy.deepcopy(model)
        add_visual_environment(root)
        variant_world = root.find("worldbody")
        bodies = list(variant_world)
        variant_world.clear()
        add_scene_geometry(variant_world, variant)
        variant_world.extend(bodies)
        write_tree(root, OUTPUT / "b1_z1" / f"scene_{variant}.xml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("all", "zgwsarm", "b1_z1"), default="all")
    args = parser.parse_args()
    if args.task in {"all", "zgwsarm"}:
        generate_zgwsarm()
    if args.task in {"all", "b1_z1"}:
        generate_b1_z1()
    print(f"Generated MuJoCo scenes under {OUTPUT}")


if __name__ == "__main__":
    main()
