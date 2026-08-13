"""Export a training checkpoint without constructing an Isaac Gym environment."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from b1_gym.utils.artifacts import load_run_config, resolve_local_checkpoint
from b1_gym_learn.modules.actor_critic import AC_Args, ActorCritic
from b1_gym_learn.utils.policy_export import export_policy_as_jit


def parse_args():
    parser = argparse.ArgumentParser(description="Export a policy checkpoint")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--output-dir")
    parser.add_argument("--filename")
    return parser.parse_args()


def policy_config(values):
    defaults = {
        key: copy.deepcopy(value)
        for key, value in vars(AC_Args).items()
        if not key.startswith("_") and not callable(value)
    }
    defaults.update(copy.deepcopy(values))
    return SimpleNamespace(**defaults)


def export_run_policy(run_dir, checkpoint="latest", output_dir=None, filename=None):
    run_dir, checkpoint_path, checkpoint_number = resolve_local_checkpoint(
        run_dir, checkpoint
    )
    config = load_run_config(run_dir)
    env_cfg = config["Cfg"]["env"]
    actor_critic = ActorCritic(
        num_obs=env_cfg["num_observations"],
        num_privileged_obs=env_cfg["num_privileged_obs"],
        num_obs_history=(
            env_cfg["num_observations"] * env_cfg["num_observation_history"]
        ),
        num_actions=env_cfg["num_actions"],
        cfg=policy_config(config.get("AC_Args", {})),
    )
    loaded = torch.load(str(checkpoint_path), map_location="cpu")
    actor_critic.load_state_dict(loaded.get("model_state_dict", loaded), strict=True)
    actor_critic.eval()

    output_dir = Path(output_dir) if output_dir else run_dir / "exported" / "policies"
    if filename is None:
        suffix = "latest" if checkpoint_number < 0 else f"{checkpoint_number:06d}"
        filename = f"policy_{suffix}.pt"
    output_path = export_policy_as_jit(actor_critic, output_dir, filename)
    print(f"Exported policy: {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    args = parse_args()
    export_run_policy(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        filename=args.filename,
    )
