import argparse
import os
import sys
from datetime import datetime

import isaacgym
assert isaacgym
import numpy as np
import torch
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


def play(run_dir=None, checkpoint="latest", device="cuda:0", num_envs=1, steps=2000,
         viewer=True, record_video=False):
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
    )
    export_path = export_policy_as_jit(
        policy.actor_critic,
        run_dir / "exported" / "policies",
        filename=f"{policy_id}.pt",
    )
    print(f"Exported policy as jit script to: {export_path}")

    cameras = []
    video_writer = None
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

    obs = env.reset()
    for _ in tqdm(range(steps), desc=f"Playing {policy_id}"):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

        if video_writer is not None:
            frames = []
            for camera in cameras:
                camera.set_position()
                frames.append(camera.get_observation()[:, :, :3])
            video_writer.append_data(np.concatenate(frames, axis=0))

    if video_writer is not None:
        video_writer.close()
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
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
