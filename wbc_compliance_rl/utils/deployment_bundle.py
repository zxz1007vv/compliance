"""Export simulator-independent policy bundles for the C++ MuJoCo runner.

The JSON manifest is intended for humans and tooling.  ``runtime.cfg`` carries
the same latency-critical values in a deliberately small ``key=value`` format
so the C++ runner does not need a third-party JSON dependency.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from wbc_compliance_gym.utils.artifacts import resolve_run_dir, resolve_run_task
from wbc_compliance_rl.utils.policy_export import export_policy_as_jit


DEPLOYMENT_SCHEMA_VERSION = 1

COMMAND_NAMES = (
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_yaw",
    "body_height",
    "gait_frequency",
    "gait_phase",
    "gait_offset",
    "gait_bound",
    "gait_duration",
    "footswing_height",
    "body_pitch",
    "body_roll",
    "ee_force_x",
    "ee_force_y",
    "ee_force_z",
    "ee_pos_radius",
    "ee_pos_pitch",
    "ee_pos_yaw",
    "ee_pos_timing",
    "ee_roll",
    "ee_pitch",
    "ee_yaw",
    "force_or_position",
)

COMMAND_RANGE_FIELDS = (
    ("lin_vel_x", "limit_vel_x"),
    ("lin_vel_y", "limit_vel_y"),
    ("ang_vel_yaw", "limit_vel_yaw"),
    ("body_height_cmd", "limit_body_height"),
    ("gait_frequency_cmd_range", "limit_gait_frequency"),
    ("gait_phase_cmd_range", "limit_gait_phase"),
    ("gait_offset_cmd_range", "limit_gait_offset"),
    ("gait_bound_cmd_range", "limit_gait_bound"),
    ("gait_duration_cmd_range", "limit_gait_duration"),
    ("footswing_height_range", "limit_footswing_height"),
    ("body_pitch_range", "limit_body_pitch"),
    ("body_roll_range", "limit_body_roll"),
    ("ee_force_x", "limit_ee_force_x"),
    ("ee_force_y", "limit_ee_force_y"),
    ("ee_force_z", "limit_ee_force_z"),
    ("ee_sphe_radius", "limit_ee_sphe_radius"),
    ("ee_sphe_pitch", "limit_ee_sphe_pitch"),
    ("ee_sphe_yaw", "limit_ee_sphe_yaw"),
    ("ee_timing", "limit_ee_timing"),
    ("end_effector_roll", "limit_end_effector_roll"),
    ("end_effector_pitch", "limit_end_effector_pitch"),
    ("end_effector_yaw", "limit_end_effector_yaw"),
    (None, None),
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_asset_path(value, project_root):
    value = str(value)
    for token in ("{WBC_COMPLIANCE_ROOT_DIR}", "{MINI_GYM_ROOT_DIR}"):
        value = value.replace(token, str(project_root))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _parse_urdf_contract(path, num_actions):
    if not path.is_file():
        raise FileNotFoundError(f"Robot URDF not found: {path}")
    # The historical combined B1+Z1 URDF contains one leftover ``xacro:include``
    # without declaring the xacro namespace. Isaac Gym ignores it, but a strict
    # XML parser correctly rejects the unbound prefix.  Treat it as an ordinary
    # extension tag while extracting the already-expanded joint contract.
    source = path.read_text(encoding="utf-8").replace("xacro:", "xacro_")
    root = ET.fromstring(source)
    joints = []
    for node in root.findall("joint"):
        joint_type = node.attrib.get("type")
        if joint_type not in {"revolute", "continuous"}:
            continue
        limit = node.find("limit")
        limit_values = limit.attrib if limit is not None else {}

        def numeric(name, default=None):
            value = limit_values.get(name)
            return default if value is None else float(value)

        joints.append(
            {
                "name": node.attrib["name"],
                "type": joint_type,
                "lower": numeric("lower") if joint_type != "continuous" else None,
                "upper": numeric("upper") if joint_type != "continuous" else None,
                "velocity": numeric("velocity", math.inf),
                "effort": numeric("effort", math.inf),
            }
        )
    if len(joints) != int(num_actions):
        raise ValueError(
            f"URDF action contract mismatch for {path}: "
            f"expected {num_actions} movable joints, found {len(joints)}"
        )
    return joints


def _command_ranges(cfg):
    command_cfg = cfg["commands"]
    legacy_force_range = cfg["domain_rand"]["max_push_force_xyz_gripper"]
    active, limits = [], []
    for index, (active_name, limit_name) in enumerate(COMMAND_RANGE_FIELDS):
        if 12 <= index <= 14:
            # New bundles expose independent Cartesian target ranges.  The
            # fallback keeps export compatible with saved pre-migration configs.
            active_value = command_cfg.get(active_name, legacy_force_range)
            limit_value = command_cfg.get(limit_name, active_value)
        elif index == 22:
            active_value = limit_value = [0.0, 1.0]
        else:
            active_value = command_cfg[active_name]
            limit_value = command_cfg.get(limit_name, active_value)
        active.append([float(active_value[0]), float(active_value[1])])
        limits.append([float(limit_value[0]), float(limit_value[1])])
    return active, limits


def _command_scales(cfg):
    scales = cfg["obs_scales"]
    return [
        scales["lin_vel"],
        scales["lin_vel"],
        scales["ang_vel"],
        scales["body_height_cmd"],
        scales["gait_freq_cmd"],
        scales["gait_phase_cmd"],
        scales["gait_phase_cmd"],
        scales["gait_phase_cmd"],
        scales["gait_phase_cmd"],
        scales["footswing_height_cmd"],
        scales["body_pitch_cmd"],
        scales["body_roll_cmd"],
        scales.get("ee_force_x", scales["ee_force_magnitude"]),
        scales.get("ee_force_y", scales["ee_force_magnitude"]),
        scales["ee_force_z"],
        scales["ee_sphe_radius_cmd"],
        scales["ee_sphe_pitch_cmd"],
        scales["ee_sphe_yaw_cmd"],
        scales["ee_timing_cmd"],
        scales["end_effector_roll_cmd"],
        scales["end_effector_pitch_cmd"],
        scales["end_effector_yaw_cmd"],
        1.0,
    ]


def _command_defaults(cfg, active_ranges):
    defaults = [0.5 * (bounds[0] + bounds[1]) for bounds in active_ranges]
    configured_ee = cfg["commands"].get("default_ee_position_spherical")
    if configured_ee is not None:
        defaults[15:18] = [float(value) for value in configured_ee]
    defaults[12:15] = [0.0, 0.0, 0.0]
    defaults[22] = 0.0
    return defaults


def _task_groups(task_name, cfg, joints):
    names = [joint["name"] for joint in joints]
    if task_name == "zgwsarm_compliance":
        asset = cfg["asset"]
        leg_names = list(asset["leg_dof_names"])
        wheel_names = list(asset["wheel_dof_names"])
        arm_names = list(asset["arm_dof_names"])
        abad_names = list(asset["abad_dof_names"])
        direct_wrist = "ROBOT_ARM_JOINT6"
        direct_gripper = ""
    elif task_name == "b1_z1_ik":
        leg_names = names[:12]
        wheel_names = []
        arm_names = names[12:]
        abad_names = [names[index] for index in (0, 3, 6, 9)]
        direct_wrist = "joint6"
        direct_gripper = "jointGripper"
    else:
        raise ValueError(f"Unsupported deployment task: {task_name}")
    grouped = leg_names + wheel_names + arm_names
    if set(grouped) != set(names) or len(grouped) != len(names):
        raise ValueError(
            f"DOF groups do not cover task action contract for {task_name}"
        )
    return {
        "leg_dof_names": leg_names,
        "wheel_dof_names": wheel_names,
        "arm_dof_names": arm_names,
        "abad_dof_names": abad_names,
        "direct_wrist_dof": direct_wrist,
        "direct_gripper_dof": direct_gripper,
    }


def _per_dof(values, names, selected_names):
    mapping = dict(zip(selected_names, values))
    return [float(mapping.get(name, 0.0)) for name in names]


def _control_contract(task_name, cfg, joints, groups):
    control = cfg["control"]
    commands = cfg["commands"]
    names = [joint["name"] for joint in joints]
    global_scale = float(control["action_scale"])
    action_scales = [global_scale for _ in names]
    for name in groups["abad_dof_names"]:
        index = names.index(name)
        if task_name == "zgwsarm_compliance":
            action_scales[index] *= float(control["abad_scale_reduction"])
        else:
            action_scales[index] *= float(control["hip_scale_reduction"])
    for name in groups["arm_dof_names"]:
        action_scales[names.index(name)] *= float(control["arm_scale_reduction"])

    p_gains = [0.0 for _ in names]
    d_gains = [0.0 for _ in names]
    for group_name, p_key, d_key in (
        ("leg_dof_names", "p_gains_legs", "d_gains_legs"),
        ("wheel_dof_names", "p_gains_wheels", "d_gains_wheels"),
        ("arm_dof_names", "p_gains_arm", "d_gains_arm"),
    ):
        selected = groups[group_name]
        if not selected:
            continue
        p_values = commands[p_key]
        d_values = commands[d_key]
        if len(selected) != len(p_values) or len(selected) != len(d_values):
            raise ValueError(f"Gain count mismatch for {task_name} {group_name}")
        for name, p_value, d_value in zip(selected, p_values, d_values):
            index = names.index(name)
            p_gains[index] = float(p_value)
            d_gains[index] = float(d_value)

    kinds = []
    wheel_names = set(groups["wheel_dof_names"])
    arm_names = set(groups["arm_dof_names"])
    for name in names:
        if name in wheel_names:
            kinds.append("wheel_torque")
        elif task_name == "zgwsarm_compliance" and name in arm_names:
            kinds.append("arm_position_pd")
        else:
            kinds.append("position_pd")

    return {
        "physics_dt": float(cfg["sim"]["dt"]),
        "decimation": int(control["decimation"]),
        "control_dt": float(cfg["sim"]["dt"]) * int(control["decimation"]),
        "action_clip": float(cfg["normalization"]["clip_actions"]),
        "observation_clip": float(cfg["normalization"]["clip_observations"]),
        "action_scale": global_scale,
        "action_scale_per_dof": action_scales,
        "p_gains": p_gains,
        "d_gains": d_gains,
        "control_kind": kinds,
        "arm_target_velocity_limit_scale": float(
            control.get("arm_target_velocity_limit_scale", 0.0)
        ),
    }


def build_deployment_manifest(run_dir, checkpoint_path, checkpoint_number, config):
    run_dir = resolve_run_dir(run_dir)
    project_root = Path(__file__).resolve().parents[2]
    task_name = resolve_run_task(run_dir, config)
    cfg = config["Cfg"]
    env = cfg["env"]
    num_actions = int(env["num_actions"])
    frame_dim = int(env["num_observations"])
    history_length = int(env["num_observation_history"])
    history_frame_skip = int(env.get("history_frame_skip", 1))
    input_dim = frame_dim * history_length
    if history_frame_skip != 1:
        raise ValueError("C++ deployment currently requires history_frame_skip=1")

    asset_path = _resolve_asset_path(cfg["asset"]["file"], project_root)
    joints = _parse_urdf_contract(asset_path, num_actions)
    names = [joint["name"] for joint in joints]
    groups = _task_groups(task_name, cfg, joints)
    default_angles = cfg["init_state"]["default_joint_angles"]
    default_positions = [float(default_angles[name]) for name in names]
    active_ranges, limit_ranges = _command_ranges(cfg)
    command_scales = [float(value) for value in _command_scales(cfg)]
    command_defaults = _command_defaults(cfg, active_ranges)
    control = _control_contract(task_name, cfg, joints, groups)

    zero_position_names = list(
        cfg["asset"].get("zero_position_observation_dof_names", [])
    )
    zero_position_indices = [names.index(name) for name in zero_position_names]
    force_frame = cfg["rewards"].get("force_command_frame", "yaw")
    base_name = cfg["asset"].get("base_name", "trunk")
    end_effector_name = cfg["asset"].get(
        "end_effector_name", "gripperStator"
    )
    initial_rotation_xyzw = cfg["init_state"]["rot"]
    model_relative = (
        "models/zgwsarm/scene_flat.xml"
        if task_name == "zgwsarm_compliance"
        else "models/b1_z1/scene_flat.xml"
    )
    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "task_name": task_name,
        "run_dir": str(run_dir),
        "checkpoint": {
            "number": int(checkpoint_number),
            "file": str(Path(checkpoint_path).resolve()),
            "sha256": _sha256(checkpoint_path),
        },
        "policy": {
            "file": "policy.pt",
            "input_dim": input_dim,
            "output_dim": num_actions,
            "latent_dim": int(env["num_privileged_obs"]),
            "dtype": "float32",
        },
        "observation": {
            "frame_dim": frame_dim,
            "history_length": history_length,
            "history_frame_skip": history_frame_skip,
            "layout": [
                {"name": "projected_gravity", "offset": 0, "dim": 3},
                {"name": "commands", "offset": 3, "dim": 23},
                {"name": "joint_position", "offset": 26, "dim": num_actions},
                {
                    "name": "joint_velocity",
                    "offset": 26 + num_actions,
                    "dim": num_actions,
                },
                {
                    "name": "action",
                    "offset": 26 + 2 * num_actions,
                    "dim": num_actions,
                },
                {
                    "name": "clock",
                    "offset": 26 + 3 * num_actions,
                    "dim": 4,
                },
            ],
            "dof_position_scale": float(cfg["obs_scales"]["dof_pos"]),
            "dof_velocity_scale": float(cfg["obs_scales"]["dof_vel"]),
            "zero_position_dof_indices": zero_position_indices,
        },
        "commands": {
            "dimension": len(COMMAND_NAMES),
            "names": list(COMMAND_NAMES),
            "scales": command_scales,
            "active_ranges": active_ranges,
            "limit_ranges": limit_ranges,
            "defaults": command_defaults,
            "force_frame": force_frame,
            "position_command_indices": [15, 16, 17],
            "force_command_indices": [12, 13, 14],
            "mode_index": 22,
        },
        "robot": {
            "model": model_relative,
            "source_urdf": str(asset_path),
            "base_body": base_name,
            "end_effector_body": end_effector_name,
            "initial_base_position": [
                float(value) for value in cfg["init_state"]["pos"]
            ],
            "initial_base_quaternion_wxyz": [
                float(initial_rotation_xyzw[3]),
                float(initial_rotation_xyzw[0]),
                float(initial_rotation_xyzw[1]),
                float(initial_rotation_xyzw[2]),
            ],
            "initial_base_linear_velocity": [
                float(value) for value in cfg["init_state"]["lin_vel"]
            ],
            "initial_base_angular_velocity": [
                float(value) for value in cfg["init_state"]["ang_vel"]
            ],
            "dof_names": names,
            "default_dof_positions": default_positions,
            "joints": joints,
            **groups,
            "wheel_radius": float(cfg["asset"].get("wheel_radius", 0.0)),
            "arm_mount_translation": [
                float(value)
                for value in cfg["commands"].get(
                    "arm_mount_translation", [0.2, 0.0, 0.1585]
                )
            ],
            "arm_mount_yaw": float(cfg["commands"].get("arm_mount_yaw", 0.0)),
            "command_base_height": float(
                cfg["commands"].get("command_base_height", 0.6)
            ),
        },
        "control": control,
        "teleoperation": {
            "force_limit": max(abs(active_ranges[12][0]), abs(active_ranges[12][1])),
            "deadzone": 0.10,
            "precision_scale": 0.25,
            "position_rates": [0.20, 0.75, 0.75],
            "wrist_rate": 0.75,
            "gripper_rate": 0.75,
        },
    }


def _flat_values(manifest):
    observation = manifest["observation"]
    commands = manifest["commands"]
    robot = manifest["robot"]
    control = manifest["control"]
    teleop = manifest["teleoperation"]
    joint_lower = [
        "nan" if joint["lower"] is None else joint["lower"]
        for joint in robot["joints"]
    ]
    joint_upper = [
        "nan" if joint["upper"] is None else joint["upper"]
        for joint in robot["joints"]
    ]
    return {
        "schema_version": manifest["schema_version"],
        "task_name": manifest["task_name"],
        "checkpoint_number": manifest["checkpoint"]["number"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "golden_absolute_tolerance": manifest["golden"][
            "absolute_tolerance"
        ],
        "policy_file": manifest["policy"]["file"],
        "policy_input_dim": manifest["policy"]["input_dim"],
        "policy_output_dim": manifest["policy"]["output_dim"],
        "frame_dim": observation["frame_dim"],
        "history_length": observation["history_length"],
        "physics_dt": control["physics_dt"],
        "decimation": control["decimation"],
        "action_clip": control["action_clip"],
        "observation_clip": control["observation_clip"],
        "dof_position_scale": observation["dof_position_scale"],
        "dof_velocity_scale": observation["dof_velocity_scale"],
        "zero_position_dof_indices": observation["zero_position_dof_indices"],
        "command_names": commands["names"],
        "command_scales": commands["scales"],
        "command_defaults": commands["defaults"],
        "command_active_low": [value[0] for value in commands["active_ranges"]],
        "command_active_high": [value[1] for value in commands["active_ranges"]],
        "command_limit_low": [value[0] for value in commands["limit_ranges"]],
        "command_limit_high": [value[1] for value in commands["limit_ranges"]],
        "force_frame": commands["force_frame"],
        "model": robot["model"],
        "base_body": robot["base_body"],
        "end_effector_body": robot["end_effector_body"],
        "initial_base_position": robot["initial_base_position"],
        "initial_base_quaternion_wxyz": robot[
            "initial_base_quaternion_wxyz"
        ],
        "initial_base_linear_velocity": robot[
            "initial_base_linear_velocity"
        ],
        "initial_base_angular_velocity": robot[
            "initial_base_angular_velocity"
        ],
        "dof_names": robot["dof_names"],
        "default_dof_positions": robot["default_dof_positions"],
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
        "joint_velocity": [joint["velocity"] for joint in robot["joints"]],
        "joint_effort": [joint["effort"] for joint in robot["joints"]],
        "leg_dof_names": robot["leg_dof_names"],
        "wheel_dof_names": robot["wheel_dof_names"],
        "arm_dof_names": robot["arm_dof_names"],
        "abad_dof_names": robot["abad_dof_names"],
        "direct_wrist_dof": robot["direct_wrist_dof"],
        "direct_gripper_dof": robot["direct_gripper_dof"],
        "wheel_radius": robot["wheel_radius"],
        "arm_mount_translation": robot["arm_mount_translation"],
        "arm_mount_yaw": robot["arm_mount_yaw"],
        "command_base_height": robot["command_base_height"],
        "action_scale_per_dof": control["action_scale_per_dof"],
        "p_gains": control["p_gains"],
        "d_gains": control["d_gains"],
        "control_kind": control["control_kind"],
        "arm_target_velocity_limit_scale": control[
            "arm_target_velocity_limit_scale"
        ],
        "teleop_force_limit": teleop["force_limit"],
        "teleop_deadzone": teleop["deadzone"],
        "teleop_precision_scale": teleop["precision_scale"],
        "teleop_position_rates": teleop["position_rates"],
        "teleop_wrist_rate": teleop["wrist_rate"],
        "teleop_gripper_rate": teleop["gripper_rate"],
    }


def _format_flat(value):
    if isinstance(value, list):
        return ",".join(_format_flat(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".17g")
    return str(value)


def _write_float32(path, tensor):
    values = tensor.detach().cpu().contiguous().to(torch.float32).view(-1)
    Path(path).write_bytes(values.numpy().tobytes(order="C"))


def export_deployment_bundle(
    actor_critic,
    run_dir,
    checkpoint_path,
    checkpoint_number,
    config,
    output_dir=None,
):
    """Export one self-contained C++ inference contract and golden vectors."""
    run_dir = resolve_run_dir(run_dir)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else run_dir / "exported" / "deployment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_deployment_manifest(
        run_dir, checkpoint_path, checkpoint_number, config
    )
    policy_path = export_policy_as_jit(actor_critic, output_dir, "policy.pt")

    input_dim = manifest["policy"]["input_dim"]
    with torch.no_grad():
        ramp = torch.linspace(-0.75, 0.75, input_dim, dtype=torch.float32)
        phase = torch.arange(input_dim, dtype=torch.float32) * 0.017
        golden_inputs = torch.stack(
            (torch.zeros(input_dim), ramp, torch.sin(phase)), dim=0
        )
        golden_outputs = torch.jit.load(str(policy_path)).eval()(golden_inputs)
    _write_float32(output_dir / "golden_inputs.f32", golden_inputs)
    _write_float32(output_dir / "golden_outputs.f32", golden_outputs)
    manifest["golden"] = {
        "cases": int(golden_inputs.shape[0]),
        "inputs_file": "golden_inputs.f32",
        "outputs_file": "golden_outputs.f32",
        # LibTorch and Python Torch may choose different CPU GEMM kernels.
        # The observed differences are a few 1e-5 for these networks; 1e-4 is
        # still orders of magnitude below actuator/action significance.
        "absolute_tolerance": 1e-4,
    }
    manifest["policy"]["sha256"] = _sha256(policy_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    flat = _flat_values(manifest)
    (output_dir / "runtime.cfg").write_text(
        "\n".join(f"{key}={_format_flat(value)}" for key, value in flat.items())
        + "\n",
        encoding="utf-8",
    )
    source_config = run_dir / "config.json"
    if source_config.is_file():
        shutil.copy2(source_config, output_dir / "training_config.json")
    checksum_paths = [
        output_dir / "policy.pt",
        output_dir / "manifest.json",
        output_dir / "runtime.cfg",
        output_dir / "golden_inputs.f32",
        output_dir / "golden_outputs.f32",
    ]
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in checksum_paths)
        + "\n",
        encoding="utf-8",
    )
    return output_dir, manifest


__all__ = [
    "COMMAND_NAMES",
    "DEPLOYMENT_SCHEMA_VERSION",
    "build_deployment_manifest",
    "export_deployment_bundle",
]
