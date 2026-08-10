import os
import sys
import glob
import copy
import json
import re
from pathlib import Path
import yaml
import wandb
import torch
from typing import Callable, Tuple 
import pickle as pkl

from isaacgym.torch_utils import *
from isaacgym import gymapi, gymutil
from typing import List

sys.path.append('../')

from b1_gym.envs import *
from b1_gym.envs.base.legged_robot_config import Cfg
from b1_gym.envs.b1.b1_plus_z1_config import config_b1_plus_z1
from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from b1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from b1_gym_learn.ppo_cse.actor_critic import ActorCritic
from b1_gym_learn.ppo_cse.actor_critic import AC_Args


def _apply_class_dict(target, values):
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)


def _apply_cfg_dict(cfg_dict):
    for section_name, section_values in cfg_dict.items():
        if not hasattr(Cfg, section_name) or not isinstance(section_values, dict):
            continue
        section = getattr(Cfg, section_name)
        for key, value in section_values.items():
            if hasattr(section, key):
                setattr(section, key, value)


def _resolve_run_dir(run_dir):
    requested = Path(run_dir).expanduser()
    if requested.is_absolute() or requested.exists():
        return requested.resolve()
    project_relative = Path(__file__).resolve().parents[1] / requested
    if project_relative.exists():
        return project_relative.resolve()
    return requested.resolve()


def resolve_latest_run(log_root=None, task_name=None):
    project_root = Path(__file__).resolve().parents[1]
    if log_root is None:
        log_root = Path(os.environ.get("COMPLIANCE_LOG_DIR", project_root / "logs"))
    else:
        log_root = Path(log_root).expanduser()
    task_name = task_name or os.environ.get("COMPLIANCE_TASK_NAME", "b1_z1_ik")
    task_dir = log_root / task_name

    candidates = []
    if task_dir.is_dir():
        for run_dir in task_dir.iterdir():
            if not run_dir.is_dir() or not (run_dir / "config.json").is_file():
                continue
            try:
                _, checkpoint_path, checkpoint_number = resolve_local_checkpoint(
                    run_dir, "latest"
                )
            except FileNotFoundError:
                continue
            candidates.append(
                (checkpoint_path.stat().st_mtime, checkpoint_number, run_dir)
            )

    if not candidates:
        raise FileNotFoundError(
            f"No trained runs with checkpoints found under {task_dir}"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2].resolve()


def resolve_local_checkpoint(run_dir, checkpoint="latest"):
    run_dir = _resolve_run_dir(run_dir)
    # New runs follow HIMLoco and store model_<iteration>.pt in the run root.
    # The checkpoints/ fallback keeps existing compliance runs playable.
    checkpoint_dirs = (run_dir, run_dir / "checkpoints")
    numbered = []
    for checkpoint_dir in checkpoint_dirs:
        for path in checkpoint_dir.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match:
                numbered.append((int(match.group(1)), path))

    if str(checkpoint).lower() == "latest":
        if not numbered:
            searched = ", ".join(str(path) for path in checkpoint_dirs)
            raise FileNotFoundError(f"No model checkpoints found in: {searched}")
        checkpoint_number, checkpoint_path = max(
            numbered,
            key=lambda item: (item[0], item[1].parent == run_dir),
        )
    else:
        checkpoint_number = int(checkpoint)
        candidates = [path for checkpoint_dir in checkpoint_dirs for path in (
            checkpoint_dir / f"model_{checkpoint_number}.pt",
            checkpoint_dir / f"model_{checkpoint_number:06d}.pt",
        )
        ]
        checkpoint_path = next((path for path in candidates if path.is_file()), None)
        if checkpoint_path is None:
            searched = ", ".join(str(path) for path in checkpoint_dirs)
            raise FileNotFoundError(
                f"No model checkpoint {checkpoint_number} in: {searched}"
            )
    return run_dir, checkpoint_path, checkpoint_number


def load_run_config(run_dir):
    config_path = Path(run_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


class LocalPolicy:
    def __init__(self, actor_critic, device):
        self.actor_critic = actor_critic
        self.device = device

    def __call__(self, obs, info=None):
        obs_history = obs["obs_history"].to(self.device)
        latent = self.actor_critic.adaptation_module(obs_history)
        actions = self.actor_critic.actor_body(
            torch.cat((obs_history, latent), dim=-1)
        )
        if info is not None:
            info["latent"] = latent
        return actions


class PolicyExporter(torch.nn.Module):
    """Single-file inference graph matching the policy used by play."""

    def __init__(self, actor_critic):
        super().__init__()
        self.adaptation_module = copy.deepcopy(actor_critic.adaptation_module)
        self.actor_body = copy.deepcopy(actor_critic.actor_body)

    def forward(self, obs_history):
        latent = self.adaptation_module(obs_history)
        return self.actor_body(torch.cat((obs_history, latent), dim=-1))


def export_policy_as_jit(actor_critic, path, filename):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / filename
    exporter = PolicyExporter(actor_critic).cpu().eval()
    torch.jit.script(exporter).save(str(output_path))
    return output_path


def load_local_policy(run_dir, checkpoint, env, device="cuda:0"):
    run_dir, checkpoint_path, checkpoint_number = resolve_local_checkpoint(
        run_dir, checkpoint
    )
    local_config = load_run_config(run_dir)
    _apply_class_dict(AC_Args, local_config.get("AC_Args", {}))
    actor_critic = ActorCritic(
        env.num_obs,
        env.num_privileged_obs,
        env.num_obs_history,
        env.num_actions,
    ).to(device)
    loaded = torch.load(str(checkpoint_path), map_location=device)
    state_dict = loaded.get("model_state_dict", loaded)
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return LocalPolicy(actor_critic, device), checkpoint_number

def load_policy(run, run_path: str, weights_path: str) -> Callable:
    '''
    1. Loads the latest policy and adaptation module weights in the temporary directory /tmp/legged_data
    2. Initialize these networks with these weights 

    Arguments:
        - runpath:      path of the run in wandb (click on info for a run to retrieve it) 
        - weights_path: local path where the weights are stored locally

    Returns: 
        - The function used to generate actions given the obs history
    '''
    wandb_path = 'tmp/legged_data/'
    # replace file locally if it already exists, root: A string specifying the root directory where the downloaded file should be stored. 
    body_file = run.file(wandb_path + 'body_latest.jit').download(replace=True, root=weights_path) 
    body = torch.jit.load(body_file.name)

    #adaptation_module_file = wandb.restore(weights_path + 'adaptation_module_latest.jit', run_path=run_path)
    adaptation_module_file = run.file(wandb_path + 'adaptation_module_latest.jit').download(replace=True, root=weights_path) 
    adaptation_module = torch.jit.load(adaptation_module_file.name)

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_env(run_path: str = None, weights_path: str = None, sim_device: str = 'cuda:0',
             num_envs: int = 1, headless: bool = False, fix_base: bool = False,
             teleop: bool = False, interpolate_ee_cmds: bool = True,
             sample_feasible_commands: bool = False, control_only_z1: bool = False,
             local_run_dir: str = None, checkpoint="latest"):
    '''
    1. Load the parameters and weights of a wandb run
    2. Initialize the simulation parameters with these weights.
    3. Turn off all the domain randomization parameters.
    4. Create the environment using these params.
    5. Load the policy.

    Arguments:
        - run_path: Path of the run in wandb (click on info for a run to retrieve it) 
        - num_envs: Number of environments to create 
        - headless: Boolean to specify if rendering should be on (False = rendering on)
    
    Returns: 
        - The function used to generate actions given the obs history
    '''
    
    if run_path is not None and local_run_dir is not None:
        raise ValueError("Choose either a W&B run_path or a local_run_dir, not both")

    if local_run_dir is not None:
        local_run_dir = _resolve_run_dir(local_run_dir)
        config_b1_plus_z1(Cfg)
        local_config = load_run_config(local_run_dir)
        _apply_cfg_dict(local_config["Cfg"])
    elif run_path is not None:
        # test mode
        api = wandb.Api()
        run = api.run(run_path)

        # Default config for all robots
        config_b1_plus_z1(Cfg)

        all_cfg = run.config
        cfg = all_cfg["Cfg"]

        _apply_cfg_dict(cfg)
                    
    else:
        # play mode
        config_b1_plus_z1(Cfg)


    # # # Turn off DR for evaluation script
    # Cfg.domain_rand.push_robots = False
    # Cfg.domain_rand.push_gripper_stators = False
    # Cfg.domain_rand.push_robot_base = False
    # Cfg.domain_rand.randomize_friction = False
    # Cfg.domain_rand.randomize_gravity = False
    # Cfg.domain_rand.randomize_restitution = False
    # Cfg.domain_rand.randomize_motor_offset = False
    # Cfg.domain_rand.randomize_motor_strength = False
    # Cfg.domain_rand.randomize_friction_indep = False
    # Cfg.domain_rand.randomize_ground_friction = False
    # Cfg.domain_rand.randomize_base_mass = False
    # Cfg.domain_rand.randomize_Kd_factor = False
    # Cfg.domain_rand.randomize_Kp_factor = False
    # Cfg.domain_rand.randomize_joint_friction = False
    # Cfg.domain_rand.randomize_com_displacement = False
    # Cfg.domain_rand.randomize_tile_roughness = True
    # Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]
    # Cfg.domain_rand.ground_friction_range = [2.0, 2.01]
    # Cfg.robot.name = "b1_plus_dismounted_z1"

    # Cfg.noise.noise_level = 0

    # Define env params 
    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = num_envs
    Cfg.env.episode_length_s = 10000
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.terrain.border_size = 0
    Cfg.terrain.num_border_boxes = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True
    # Cfg.terrain.mesh_type = "boxes_tm"

    Cfg.asset.fix_base_link = fix_base
    Cfg.commands.teleop_occulus = teleop
    Cfg.commands.interpolate_ee_cmds = interpolate_ee_cmds
    Cfg.commands.control_only_z1 = control_only_z1

    Cfg.env.recording_height_px = 720
    Cfg.env.recording_width_px = 1280
    
    Cfg.env.record_video = True
    Cfg.env.send_eval_data = True

    # Create env
    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    if local_run_dir is not None:
        policy, _ = load_local_policy(
            local_run_dir,
            checkpoint=checkpoint,
            env=env,
            device=sim_device,
        )
    elif run_path is not None:
        # Load policy
        policy = load_policy(run, 
                             run_path=run_path,
                             weights_path=weights_path)
    else:
        # set the dummy policy
        policy = lambda x: torch.zeros((num_envs, 19), device=sim_device)

    return env, policy
