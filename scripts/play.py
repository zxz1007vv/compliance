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

from b1_gym_learn.ppo_cse.experiment_logger import safe_name
from utils import (
    export_policy_as_jit,
    load_env,
    load_run_config,
    resolve_local_checkpoint,
    resolve_latest_run,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Play a locally trained compliance policy")
    parser.add_argument(
        "--run-dir",
        help="Training run directory; defaults to the most recently saved run",
    )
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint iteration or 'latest'")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--control-mode",
        choices=("position", "force", "binary", "mixed"),
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
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument(
        "--viewer", dest="viewer", action="store_true", help="Open the Isaac Gym viewer (default)"
    )
    viewer_group.add_argument(
        "--headless", dest="viewer", action="store_false", help="Run without an Isaac Gym window"
    )
    parser.set_defaults(viewer=True)
    parser.add_argument("--record-video", action="store_true", help="Save an MP4 under the run directory")
    return parser.parse_args()


class RolloutMetrics:
    def __init__(self):
        self.totals = {}
        self.counts = {}
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
        value_max = values.max()
        if name in self.totals:
            self.totals[name] += value_sum
            self.maxima[name] = torch.maximum(self.maxima[name], value_max)
        else:
            self.totals[name] = value_sum
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

    def count(self, name):
        return self.counts.get(name, 0)

    def as_dict(self):
        return {
            name: {
                "mean": self.mean(name),
                "maximum": self.maximum(name),
                "samples": self.count(name),
            }
            for name in sorted(self.totals)
        }


def update_rollout_metrics(metrics, env, rewards, dones):
    valid_state = ~dones.bool()
    force_mode = env.force_or_position_control > 0.5
    position_mode = (~force_mode) & valid_state
    valid_force_mode = force_mode & valid_state
    force_commands = env.commands[:, 12:15]
    applied_forces_world = env.forces[:, env.gripper_stator_index, :3]
    base_rpy_world = torch.stack(get_euler_xyz(env.base_quat), dim=1)
    zeros = torch.zeros_like(base_rpy_world[:, 2])
    yaw_quat_world = quat_from_euler_xyz(zeros, zeros, base_rpy_world[:, 2])
    applied_forces = quat_rotate_inverse(yaw_quat_world, applied_forces_world)
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


def format_metric(value, unit="", precision=3):
    if value is None:
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{precision}f}{suffix}"


def print_rollout_metrics(metrics, step, steps, final=False):
    label = "Rollout summary" if final else f"Cumulative metrics {step}/{steps}"
    reset_count = int(round(metrics.totals.get("reset", torch.tensor(0.0)).item()))
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


def play(run_dir=None, checkpoint="latest", device="cuda:0", num_envs=1, steps=2000,
         viewer=True, record_video=False, print_every=100, control_mode=None,
         seed=1, force_amplitude=None, output=None):
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if run_dir is None:
        run_dir = resolve_latest_run()
        print(f"Automatically selected latest run: {run_dir}")
    run_dir, _, checkpoint_number = resolve_local_checkpoint(run_dir, checkpoint)
    run_config = load_run_config(run_dir)
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
    )
    effective_mode = env.cfg.commands.hybrid_mode
    print(
        f"Evaluation settings: mode={effective_mode}, seed={seed}, "
        f"num_envs={num_envs}, steps={steps}"
    )
    export_path = export_policy_as_jit(
        policy.actor_critic,
        run_dir / "exported" / "policies",
        filename=f"{policy_id}.pt",
    )
    print(f"Exported policy as jit script to: {export_path}")

    cameras = []
    video_writer = None
    video_path = None
    if record_video:
        import imageio
        from b1_gym.sensors.floating_camera_sensor import FloatingCameraSensor

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

    evaluation = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": checkpoint_number,
        "policy_id": policy_id,
        "settings": {
            "control_mode": effective_mode,
            "seed": seed,
            "device": device,
            "num_envs": num_envs,
            "steps": steps,
            "force_amplitude": force_amplitude,
            "force_target_range": list(
                env.cfg.domain_rand.max_push_force_xyz_gripper
            ),
        },
        "config_load": getattr(env, "_config_load_info", {}),
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
    play(
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
        output=args.output,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
