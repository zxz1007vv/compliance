"""Export a checkpoint and its exact C++ MuJoCo deployment contract."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from wbc_compliance_gym.utils.artifacts import (
    load_run_config,
    resolve_local_checkpoint,
)
from wbc_compliance_rl.modules.actor_critic import AC_Args, ActorCritic
from wbc_compliance_rl.utils.deployment_bundle import export_deployment_bundle


def policy_config(values):
    defaults = {
        key: copy.deepcopy(value)
        for key, value in vars(AC_Args).items()
        if not key.startswith("_") and not callable(value)
    }
    defaults.update(copy.deepcopy(values))
    return SimpleNamespace(**defaults)


def export_run_bundle(run_dir, checkpoint="latest", output_dir=None):
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
    actor_critic.load_state_dict(
        loaded.get("model_state_dict", loaded), strict=True
    )
    actor_critic.eval()
    bundle_dir, _ = export_deployment_bundle(
        actor_critic,
        run_dir,
        checkpoint_path,
        checkpoint_number,
        config,
        output_dir=output_dir,
    )
    print(f"Exported deployment bundle: {bundle_dir.resolve()}")
    return bundle_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a TorchScript policy and exact C++ runtime contract"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_run_bundle(args.run_dir, args.checkpoint, args.output_dir)
