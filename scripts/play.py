import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import isaacgym
assert isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_from_euler_xyz,
    quat_rotate_inverse,
)
from tqdm import tqdm

from wbc_compliance_gym.commands import (
    INDEX_EE_FORCE_X,
    INDEX_EE_FORCE_Z,
    VALID_CONTROL_MODES,
)
from wbc_compliance_gym.envs import register_tasks
from wbc_compliance_gym.envs.zgwsarm_compliance.zgwsarm_compliance_config import (
    ZGWSARM_DIAGNOSTIC_SCENARIOS,
)
from wbc_compliance_rl.logging.experiment_logger import safe_name
from utils import (
    load_env,
    load_run_config,
    resolve_local_checkpoint,
    resolve_latest_run,
    resolve_run_task,
)


RESET_REASON_NAMES = (
    "timeout",
    "body_height",
    "body_orientation",
    "semantic_contact",
    "other_contact",
    "leg_torque_limit",
    "arm_torque_limit",
    "ee_position_limit",
    "dof_velocity_safety",
    "dof_position_safety",
    "nonfoot_contact_safety",
    "unknown",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play a registered whole-body compliance task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "examples:\n"
            "  python scripts/play.py --task b1_z1_ik\n"
            "  python scripts/play.py --task b1_z1_ik --checkpoint latest\n"
            "  python scripts/play.py --run-dir logs/b1_z1_ik/<run> "
            "--checkpoint 5000"
        ),
    )
    parser.add_argument(
        "--task",
        help=(
            "Registered task; selects its latest run when --run-dir is omitted. "
            "With only --run-dir, the task is read from config.json"
        ),
    )
    parser.add_argument(
        "--list-tasks", action="store_true", help="List registered tasks and exit"
    )
    parser.add_argument(
        "--run-dir",
        help="Training run directory; when omitted, --task selects its latest run",
    )
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint iteration or 'latest'")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--control-mode",
        choices=VALID_CONTROL_MODES,
        help="Evaluation control mode; defaults to the mode saved by training",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Evaluation RNG seed (default: 1)"
    )
    parser.add_argument(
        "--force-amplitude",
        type=float,
        help="Override the force-target sampling range with [-A, A] N",
    )
    parser.add_argument(
        "--output",
        help="Metrics JSON path; defaults to <run-dir>/evaluations/<timestamp>_*.json",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print cumulative rollout metrics every N steps; use 0 for summary only",
    )
    parser.add_argument(
        "--diagnostic-scenario",
        choices=ZGWSARM_DIAGNOSTIC_SCENARIOS,
        help=(
            "Apply a deterministic ZGWSARM-only A/B scenario without changing "
            "the saved training configuration"
        ),
    )
    parser.add_argument(
        "--diagnostic-lin-vel-x",
        type=float,
        help=(
            "Fixed forward command for a diagnostic scenario; "
            "velocity_arm_fixed defaults to 0.5 m/s"
        ),
    )
    parser.add_argument(
        "--diagnostic-ang-vel-yaw",
        type=float,
        help="Fixed yaw-rate command for a diagnostic scenario (default: 0)",
    )
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--viewer", dest="viewer", action="store_true", help="Open the Isaac Gym viewer (default)"
    )
    viewer_group.add_argument(
        "--headless", dest="viewer", action="store_false", help="Run without an Isaac Gym window"
    )
    parser.set_defaults(viewer=True)
    parser.add_argument("--record-video", action="store_true", help="Save an MP4 under the run directory")
    args = parser.parse_args()
    if not args.list_tasks and args.task is None and args.run_dir is None:
        parser.error("one of --task or --run-dir is required")
    return args


class RolloutMetrics:
    def __init__(self):
        self.totals = {}
        self.square_totals = {}
        self.counts = {}
        self.minima = {}
        self.maxima = {}

    def update(self, name, values, mask=None):
        values = values.detach().reshape(-1)
        if mask is not None:
            values = values[mask.detach().reshape(-1)]
        if values.numel() == 0:
            return

        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return

        value_sum = values.sum()
        value_square_sum = torch.square(values).sum()
        value_min = values.min()
        value_max = values.max()
        if name in self.totals:
            self.totals[name] += value_sum
            self.square_totals[name] += value_square_sum
            self.minima[name] = torch.minimum(self.minima[name], value_min)
            self.maxima[name] = torch.maximum(self.maxima[name], value_max)
        else:
            self.totals[name] = value_sum
            self.square_totals[name] = value_square_sum
            self.minima[name] = value_min
            self.maxima[name] = value_max
            self.counts[name] = 0
        self.counts[name] += values.numel()

    def mean(self, name):
        if self.counts.get(name, 0) == 0:
            return None
        return (self.totals[name] / self.counts[name]).item()

    def maximum(self, name):
        if name not in self.maxima:
            return None
        return self.maxima[name].item()

    def minimum(self, name):
        if name not in self.minima:
            return None
        return self.minima[name].item()

    def standard_deviation(self, name):
        count = self.counts.get(name, 0)
        if count == 0:
            return None
        mean = self.totals[name] / count
        variance = self.square_totals[name] / count - torch.square(mean)
        return torch.sqrt(torch.clamp(variance, min=0.0)).item()

    def count(self, name):
        return self.counts.get(name, 0)

    def as_dict(self):
        return {
            name: {
                "mean": self.mean(name),
                "standard_deviation": self.standard_deviation(name),
                "minimum": self.minimum(name),
                "maximum": self.maximum(name),
                "samples": self.count(name),
            }
            for name in sorted(self.totals)
        }


def update_rollout_metrics(metrics, env, rewards, dones):
    dones_bool = dones.bool()
    valid_state = ~dones_bool
    force_mode = env.force_or_position_control > 0.5
    position_mode = (~force_mode) & valid_state
    valid_force_mode = force_mode & valid_state
    force_commands = env.commands[:, INDEX_EE_FORCE_X:INDEX_EE_FORCE_Z + 1]
    applied_forces_world = env.forces[:, env.gripper_stator_index, :3]
    force_command_frame = getattr(
        env.cfg.rewards, "force_command_frame", "yaw"
    )
    if force_command_frame == "world":
        applied_forces = applied_forces_world
    elif force_command_frame == "yaw":
        base_rpy_world = torch.stack(get_euler_xyz(env.base_quat), dim=1)
        zeros = torch.zeros_like(base_rpy_world[:, 2])
        yaw_quat_world = quat_from_euler_xyz(
            zeros, zeros, base_rpy_world[:, 2]
        )
        applied_forces = quat_rotate_inverse(
            yaw_quat_world, applied_forces_world
        )
    else:
        raise ValueError(
            f"unsupported force command frame {force_command_frame!r}"
        )
    force_abs_error = torch.abs(applied_forces - force_commands)
    active_force_command = torch.norm(force_commands, dim=1) > 1.0
    valid_active_force = valid_force_mode & active_force_command
    base_xy_error = torch.norm(
        env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1
    )
    yaw_rate_error = torch.abs(env.commands[:, 2] - env.base_ang_vel[:, 2])
    torque_rms = torch.sqrt(torch.mean(torch.square(env.torques), dim=1))

    metrics.update("reward", rewards)
    metrics.update("reset", dones.float())
    reset_reason_fn = getattr(env, "get_reset_reason_tensors", None)
    reason_tensors = (
        reset_reason_fn()
        if reset_reason_fn is not None
        else {"timeout": getattr(env, "time_out_buf", torch.zeros_like(dones_bool))}
    )
    explained_resets = torch.zeros_like(dones_bool)
    for reason_name, reason_tensor in reason_tensors.items():
        active_reason = dones_bool & reason_tensor.bool()
        explained_resets |= active_reason
        metrics.update(
            f"reset_reason/{reason_name}", active_reason.float()
        )
    metrics.update(
        "reset_reason/unknown",
        (dones_bool & ~explained_resets).float(),
    )
    metrics.update("force_mode", force_mode.float())
    metrics.update("base_xy_error", base_xy_error, valid_state)
    metrics.update("yaw_rate_error", yaw_rate_error, valid_state)
    metrics.update(
        "ee_position_error", env.gripper_pos_tracking_error_buf, position_mode
    )
    metrics.update(
        "ee_orientation_error", env.gripper_ori_tracking_error_buf, valid_state
    )
    metrics.update(
        "ee_xy_force_error",
        env.gripper_xy_force_tracking_error_buf,
        valid_force_mode,
    )
    metrics.update(
        "ee_z_force_error", env.gripper_z_force_tracking_error_buf, valid_force_mode
    )
    metrics.update(
        "active_force_command", active_force_command.float(), valid_force_mode
    )
    for axis, index in zip(("x", "y", "z"), range(3)):
        metrics.update(
            f"ee_{axis}_force_abs_error_active",
            force_abs_error[:, index],
            valid_active_force,
        )
        metrics.update(
            f"ee_{axis}_force_command_abs_active",
            torch.abs(force_commands[:, index]),
            valid_active_force,
        )
        metrics.update(
            f"ee_{axis}_applied_force_abs_active",
            torch.abs(applied_forces[:, index]),
            valid_active_force,
        )
    metrics.update("joint_torque_rms", torque_rms)
    metrics.update("joint_torque_abs", torch.abs(env.torques))
    diagnostic_fn = getattr(env, "get_zgwsarm_diagnostic_tensors", None)
    if diagnostic_fn is not None:
        for name, values in diagnostic_fn().items():
            metrics.update(name, values, valid_state)


def format_metric(value, unit="", precision=3):
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{precision}f}{suffix}"


def metric_total_count(metrics, name):
    total = metrics.totals.get(name)
    return 0 if total is None else int(round(total.item()))


def print_rollout_metrics(metrics, step, steps, final=False):
    label = "Rollout summary" if final else f"Cumulative metrics {step}/{steps}"
    reset_count = metric_total_count(metrics, "reset")
    reset_rate = metrics.mean("reset")
    force_mode_fraction = metrics.mean("force_mode")
    active_force_fraction = metrics.mean("active_force_command")
    reset_rate_text = "n/a" if reset_rate is None else f"{100 * reset_rate:.2f}%"
    force_mode_text = (
        "n/a" if force_mode_fraction is None else f"{100 * force_mode_fraction:.1f}%"
    )
    active_force_text = (
        "n/a" if active_force_fraction is None else f"{100 * active_force_fraction:.1f}%"
    )

    lines = [
        f"[{label}]",
        f"  reward/env-step: {format_metric(metrics.mean('reward'), precision=4)}",
        f"  resets: {reset_count} ({reset_rate_text} of env-steps)",
        "  reset causes (overlap allowed): "
        + ", ".join(
            f"{name}={metric_total_count(metrics, f'reset_reason/{name}')}"
            for name in RESET_REASON_NAMES
        ),
        f"  base XY velocity error: {format_metric(metrics.mean('base_xy_error'), 'm/s')}",
        f"  base yaw-rate error: {format_metric(metrics.mean('yaw_rate_error'), 'rad/s')}",
        f"  EE position error (position mode): {format_metric(metrics.mean('ee_position_error'), 'm')}",
        f"  EE orientation error: {format_metric(metrics.mean('ee_orientation_error'), 'rad')}",
        f"  EE XY force error (force mode): {format_metric(metrics.mean('ee_xy_force_error'), 'N')}",
        f"  EE Z force error (force mode): {format_metric(metrics.mean('ee_z_force_error'), 'N')}",
        "  active-command force |error| X/Y/Z: "
        f"{format_metric(metrics.mean('ee_x_force_abs_error_active'), 'N')} / "
        f"{format_metric(metrics.mean('ee_y_force_abs_error_active'), 'N')} / "
        f"{format_metric(metrics.mean('ee_z_force_abs_error_active'), 'N')}",
        f"  joint torque RMS: {format_metric(metrics.mean('joint_torque_rms'), 'N m')}",
        f"  force-mode samples: {force_mode_text}",
        f"  active force commands (>1 N): {active_force_text}",
    ]
    if final:
        lines.append(
            f"  peak absolute joint torque: {format_metric(metrics.maximum('joint_torque_abs'), 'N m')}"
        )
    tqdm.write("\n".join(lines))


def print_zgwsarm_diagnostics(metrics):
    """Print the compact view; the JSON retains every diagnostic statistic."""
    if not any(name.startswith("leg/") for name in metrics.totals):
        return

    lines = [
        "[ZGWSARM leg diagnostics: cumulative mean]",
        "  leg joint   pos(rad) target(rad) action default-delta(rad) limit(%) sat(%)",
    ]
    for leg_name in ("FAR", "FBL", "RAR", "RBL"):
        for joint_name in ("ABAD", "HIP", "KNEE"):
            prefix = f"leg/{leg_name}/{joint_name}"
            limit_rate = metrics.mean(f"{prefix}/hard_limit_dwell")
            saturation_rate = metrics.mean(f"{prefix}/action_saturation")
            lines.append(
                f"  {leg_name:3s} {joint_name:4s} "
                f"{format_metric(metrics.mean(f'{prefix}/position_rad'), precision=3):>8s} "
                f"{format_metric(metrics.mean(f'{prefix}/target_rad'), precision=3):>11s} "
                f"{format_metric(metrics.mean(f'{prefix}/action'), precision=3):>6s} "
                f"{format_metric(metrics.mean(f'{prefix}/default_deviation_rad'), precision=3):>18s} "
                f"{format_metric(None if limit_rate is None else 100 * limit_rate, precision=2):>8s} "
                f"{format_metric(None if saturation_rate is None else 100 * saturation_rate, precision=2):>6s}"
            )

    lines.extend(
        [
            "[ZGWSARM wheel diagnostics: cumulative mean]",
            "  wheel contact(%) normal(N) vx(m/s) |vy|(m/s) omega(rad/s) roll-res(m/s) base-res(m/s) pos-base[x,y,z](m)",
        ]
    )
    for wheel_name in ("FAR", "FBL", "RAR", "RBL"):
        prefix = f"wheel/{wheel_name}"
        contact_rate = metrics.mean(f"{prefix}/contact")
        position = [
            format_metric(metrics.mean(f"{prefix}/base_position_{axis}_m"), precision=3)
            for axis in ("x", "y", "z")
        ]
        lines.append(
            f"  {wheel_name:3s} "
            f"{format_metric(None if contact_rate is None else 100 * contact_rate, precision=1):>10s} "
            f"{format_metric(metrics.mean(f'{prefix}/normal_force_n'), precision=1):>9s} "
            f"{format_metric(metrics.mean(f'{prefix}/longitudinal_velocity_mps'), precision=3):>7s} "
            f"{format_metric(metrics.mean(f'{prefix}/lateral_velocity_abs_mps'), precision=3):>9s} "
            f"{format_metric(metrics.mean(f'{prefix}/angular_speed_radps'), precision=3):>12s} "
            f"{format_metric(metrics.mean(f'{prefix}/rolling_residual_contact_mps'), precision=3):>13s} "
            f"{format_metric(metrics.mean(f'{prefix}/base_speed_residual_mps'), precision=3):>13s} "
            f"[{', '.join(position)}]"
        )
    tqdm.write("\n".join(lines))


def play(task=None, run_dir=None, checkpoint="latest", device="cuda:0", num_envs=1, steps=2000,
         viewer=True, record_video=False, print_every=100, control_mode=None,
         seed=1, force_amplitude=None, output=None, diagnostic_scenario=None,
         diagnostic_lin_vel_x=None, diagnostic_ang_vel_yaw=None):
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    registry = register_tasks()
    if run_dir is None:
        if task is None:
            raise ValueError("task is required when run_dir is not provided")
        registry.get_spec(task)
        run_dir = resolve_latest_run(task_name=task)
        print(f"Automatically selected latest run for task {task!r}: {run_dir}")
    run_dir, _, checkpoint_number = resolve_local_checkpoint(run_dir, checkpoint)
    run_config = load_run_config(run_dir)
    saved_task = resolve_run_task(run_dir, run_config)
    if task is not None and task != saved_task:
        raise ValueError(
            f"Requested task {task!r} does not match run task {saved_task!r}: "
            f"{run_dir}"
        )
    task = task or saved_task
    registry.get_spec(task)
    if diagnostic_scenario is not None and task != "zgwsarm_compliance":
        raise ValueError(
            "--diagnostic-scenario is available only for zgwsarm_compliance"
        )
    run_name = safe_name(
        run_config.get("RunCfg", {}).get("training_name", run_dir.name)
    )
    policy_id = f"{run_name}_{checkpoint_number}"

    env, policy = load_env(
        local_run_dir=str(run_dir),
        checkpoint=checkpoint_number,
        sim_device=device,
        num_envs=num_envs,
        headless=not viewer,
        control_mode=control_mode,
        seed=seed,
        force_amplitude=force_amplitude,
        diagnostic_scenario=diagnostic_scenario,
        diagnostic_lin_vel_x=diagnostic_lin_vel_x,
        diagnostic_ang_vel_yaw=diagnostic_ang_vel_yaw,
        task_name=task,
    )
    effective_mode = env.cfg.commands.hybrid_mode
    print(
        f"Evaluation settings: mode={effective_mode}, seed={seed}, "
        f"num_envs={num_envs}, steps={steps}, "
        f"diagnostic_scenario={diagnostic_scenario or 'none'}"
    )
    cameras = []
    video_writer = None
    video_path = None
    if record_video:
        import imageio
        from wbc_compliance_gym.sensors import FloatingCameraSensor

        cameras = [FloatingCameraSensor(env, env_idx=i) for i in range(num_envs)]
        play_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = run_dir / "play_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"play_{play_timestamp}_{policy_id}.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=30)
        print(f"Play video: {video_path}")

    metrics = RolloutMetrics()
    obs = env.reset()
    for step in tqdm(range(1, steps + 1), desc=f"Playing {policy_id}"):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)
        update_rollout_metrics(metrics, env, rewards, dones)

        if print_every > 0 and step % print_every == 0 and step < steps:
            print_rollout_metrics(metrics, step, steps)

        if video_writer is not None:
            frames = []
            for camera in cameras:
                camera.set_position()
                frames.append(camera.get_observation()[:, :, :3])
            video_writer.append_data(np.concatenate(frames, axis=0))

    if video_writer is not None:
        video_writer.close()
    print_rollout_metrics(metrics, steps, steps, final=True)
    print_zgwsarm_diagnostics(metrics)

    evaluation = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": checkpoint_number,
        "task": task,
        "policy_id": policy_id,
        "settings": {
            "control_mode": effective_mode,
            "seed": seed,
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "force_amplitude": force_amplitude,
            "diagnostic_scenario": diagnostic_scenario,
            "diagnostic_lin_vel_x": diagnostic_lin_vel_x,
            "diagnostic_ang_vel_yaw": diagnostic_ang_vel_yaw,
            "force_target_range": list(
                env.cfg.domain_rand.max_push_force_xyz_gripper
            ),
        },
        "config_load": getattr(env, "_config_load_info", {}),
        "reset_causes": {
            name: metric_total_count(metrics, f"reset_reason/{name}")
            for name in RESET_REASON_NAMES
        },
        "metrics": metrics.as_dict(),
        "video": str(video_path.resolve()) if video_path is not None else None,
    }
    if output is None:
        evaluation_dir = run_dir / "evaluations"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = evaluation_dir / (
            f"{evaluation_timestamp}_{policy_id}_{effective_mode}_seed{seed}.json"
        )
    else:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Evaluation metrics: {output_path}")
    cameras.clear()
    env.close()


if __name__ == "__main__":
    args = parse_args()
    if args.list_tasks:
        print("\n".join(register_tasks().names()))
    else:
        play(
            task=args.task,
            run_dir=args.run_dir,
            checkpoint=args.checkpoint,
            device=args.device,
            num_envs=args.num_envs,
            steps=args.steps,
            viewer=args.viewer,
            record_video=args.record_video,
            print_every=args.print_every,
            control_mode=args.control_mode,
            seed=args.seed,
            force_amplitude=args.force_amplitude,
            diagnostic_scenario=args.diagnostic_scenario,
            diagnostic_lin_vel_x=args.diagnostic_lin_vel_x,
            diagnostic_ang_vel_yaw=args.diagnostic_ang_vel_yaw,
            output=args.output,
        )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
