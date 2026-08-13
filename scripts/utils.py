import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

import isaacgym
import numpy as np
import torch

sys.path.append('../')

from b1_gym.envs.b1_z1.b1_z1 import B1Z1Env
from b1_gym.envs.b1_z1.b1_z1_config import B1Z1Cfg, B1Z1CfgPPO
from b1_gym.commands import validate_control_mode
from b1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from b1_gym.utils.artifacts import (
    load_run_config,
    resolve_latest_run,
    resolve_local_checkpoint,
    resolve_run_dir,
)
from b1_gym.utils.config_utils import apply_config, normalize_saved_env_config
from b1_gym_learn.modules.actor_critic import ActorCritic
from b1_gym_learn.utils.policy_export import export_policy_as_jit


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


def load_local_policy(run_dir, checkpoint, env, device="cuda:0"):
    run_dir, checkpoint_path, checkpoint_number = resolve_local_checkpoint(
        run_dir, checkpoint
    )
    local_config = load_run_config(run_dir)
    policy = _policy_from_checkpoint(
        checkpoint_path, local_config, env, device
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    return policy, checkpoint_number


def _policy_from_checkpoint(checkpoint_path, config, env, device):
    train_cfg = B1Z1CfgPPO()
    apply_config(train_cfg.policy, config.get("AC_Args", {}))
    actor_critic = ActorCritic(
        env.num_obs,
        env.num_privileged_obs,
        env.num_obs_history,
        env.num_actions,
        cfg=train_cfg.policy,
    ).to(device)
    loaded = torch.load(str(checkpoint_path), map_location=device)
    state_dict = loaded.get("model_state_dict", loaded)
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()
    return LocalPolicy(actor_critic, device)


def _load_legacy_split_policy(run, weights_path):
    """Load pre-refactor W&B exports without making them the active layout."""
    def download_modules(download_root):
        legacy_root = "tmp/legged_data/"
        body_file = run.file(legacy_root + "body_latest.jit").download(
            replace=True, root=str(download_root)
        )
        body = torch.jit.load(body_file.name)
        adaptation_file = run.file(
            legacy_root + "adaptation_module_latest.jit"
        ).download(replace=True, root=str(download_root))
        adaptation_module = torch.jit.load(adaptation_file.name)
        return body, adaptation_module

    if weights_path:
        download_root = Path(weights_path).expanduser()
        download_root.mkdir(parents=True, exist_ok=True)
        body, adaptation_module = download_modules(download_root)
    else:
        with tempfile.TemporaryDirectory(
            prefix="learning-compliance-wandb-legacy-"
        ) as temp_dir:
            body, adaptation_module = download_modules(temp_dir)

    def policy(obs, info=None):
        latent = adaptation_module(obs["obs_history"].to("cpu"))
        action = body(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        if info is not None:
            info["latent"] = latent
        return action

    return policy


def load_policy(
    run,
    run_path: str,
    weights_path: str,
    env=None,
    device="cuda:0",
    checkpoint="latest",
) -> Callable:
    '''
    Load a canonical run-owned checkpoint from W&B and initialize its policy.

    Historical split JIT artifacts remain readable when a run predates the
    canonical ``checkpoints/model_*.pt`` layout.

    Arguments:
        - runpath:      path of the run in wandb (click on info for a run to retrieve it) 
        - weights_path: local path where the weights are stored locally

    Returns: 
        - The function used to generate actions given the obs history
    '''
    if env is None:
        return _load_legacy_split_policy(run, weights_path)

    remote_files = {remote_file.name: remote_file for remote_file in run.files()}
    numbered = []
    for name in remote_files:
        match = re.fullmatch(r"checkpoints/model_(\d+)\.pt", name)
        if match:
            numbered.append((int(match.group(1)), name))

    if str(checkpoint).lower() == "latest":
        checkpoint_name = max(numbered)[1] if numbered else None
        if checkpoint_name is None and "checkpoints/model_latest.pt" in remote_files:
            checkpoint_name = "checkpoints/model_latest.pt"
    else:
        checkpoint_number = int(checkpoint)
        candidates = (
            f"checkpoints/model_{checkpoint_number:06d}.pt",
            f"checkpoints/model_{checkpoint_number}.pt",
        )
        checkpoint_name = next(
            (name for name in candidates if name in remote_files), None
        )

    if checkpoint_name is None:
        return _load_legacy_split_policy(run, weights_path)

    def download_and_load(download_root):
        downloaded = remote_files[checkpoint_name].download(
            replace=True, root=str(download_root)
        )
        policy = _policy_from_checkpoint(
            downloaded.name, dict(run.config), env, device
        )
        print(f"Loaded W&B checkpoint: {run_path}/{checkpoint_name}")
        return policy

    if weights_path:
        download_root = Path(weights_path).expanduser()
        download_root.mkdir(parents=True, exist_ok=True)
        return download_and_load(download_root)
    with tempfile.TemporaryDirectory(prefix="learning-compliance-wandb-") as temp_dir:
        return download_and_load(temp_dir)

def load_env(run_path: str = None, weights_path: str = None, sim_device: str = 'cuda:0',
             num_envs: int = 1, headless: bool = False, fix_base: bool = False,
             teleop: bool = False, interpolate_ee_cmds: bool = True,
             sample_feasible_commands: bool = False, control_only_z1: bool = False,
             local_run_dir: str = None, checkpoint="latest",
             control_mode: str = None, seed: int = 1,
             force_amplitude: float = None):
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
    if control_mode is not None:
        validate_control_mode(control_mode)
    if force_amplitude is not None and force_amplitude < 0:
        raise ValueError("force_amplitude must be non-negative")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    config_load_info = {
        "legacy_reward_scale_repaired": False,
        "control_dt": None,
        "matching_scaled_coefficients": 0,
        "compared_coefficients": 0,
    }

    if local_run_dir is not None:
        local_run_dir = resolve_run_dir(local_run_dir)
        cfg = B1Z1Cfg()
        local_config = load_run_config(local_run_dir)
        normalized_cfg, config_load_info = normalize_saved_env_config(
            cfg, local_config["Cfg"]
        )
        apply_config(cfg, normalized_cfg)
        if config_load_info["legacy_reward_scale_repaired"]:
            print(
                "Detected and repaired a legacy runtime-scaled reward config "
                f"(divided by dt={config_load_info['control_dt']:.6g})."
            )
    elif run_path is not None:
        # test mode
        import wandb

        api = wandb.Api()
        run = api.run(run_path)

        # Default config for all robots
        cfg = B1Z1Cfg()

        all_cfg = run.config
        normalized_cfg, config_load_info = normalize_saved_env_config(
            cfg, all_cfg["Cfg"]
        )
        apply_config(cfg, normalized_cfg)
                    
    else:
        # play mode
        cfg = B1Z1Cfg()


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
    cfg.env.num_recording_envs = 1
    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = 10000
    cfg.terrain.num_rows = 10
    cfg.terrain.num_cols = 10
    cfg.terrain.border_size = 0
    cfg.terrain.num_border_boxes = 0
    cfg.terrain.center_robots = True
    cfg.terrain.center_span = 1
    cfg.terrain.teleport_robots = False
    cfg.terrain.mesh_type = "plane"  # boxes_tm; plane requires teleport_robots=False



    if control_mode is not None:
        cfg.commands.hybrid_mode = control_mode
    if seed is not None:
        cfg.commands.curriculum_seed = seed
    if force_amplitude is not None:
        force_range = [-float(force_amplitude), float(force_amplitude)]
        cfg.domain_rand.max_push_force_xyz_gripper = force_range

    cfg.commands.lin_vel_x = [0.0, 0.0]
    cfg.commands.limit_vel_x = [0.0, 0.0]

    cfg.commands.lin_vel_y = [0.0, 0.0]
    cfg.commands.limit_vel_y = [0.0, 0.0]

    cfg.commands.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.limit_vel_yaw = [0.0, 0.0]


    #末端位置球坐标命令
    cfg.commands.ee_sphe_radius = [0.55, 0.55]  #末端球坐标半径
    cfg.commands.limit_ee_sphe_radius = [0.55, 0.55]
    cfg.commands.ee_sphe_pitch = [0.0, 0.0]
    cfg.commands.limit_ee_sphe_pitch = [0.0, 0.0]
    cfg.commands.ee_sphe_yaw = [0.0, 0.0]
    cfg.commands.limit_ee_sphe_yaw = [0.0, 0.0]


    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_tile_roughness = False


    cfg.asset.fix_base_link = fix_base
    cfg.commands.teleop_occulus = teleop
    cfg.commands.interpolate_ee_cmds = interpolate_ee_cmds
    cfg.commands.control_only_z1 = control_only_z1

    cfg.env.recording_height_px = 720
    cfg.env.recording_width_px = 1280
    
    cfg.env.record_video = True
    cfg.env.send_eval_data = True

    # Create env
    env = B1Z1Env(sim_device=sim_device, headless=headless, cfg=cfg)
    env = HistoryWrapper(env)
    env._config_load_info = config_load_info
    env._evaluation_settings = {
        "control_mode": cfg.commands.hybrid_mode,
        "seed": seed,
        "force_amplitude": force_amplitude,
        "force_target_range": list(cfg.domain_rand.max_push_force_xyz_gripper),
    }

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
                             weights_path=weights_path,
                             env=env,
                             device=sim_device,
                             checkpoint=checkpoint)
    else:
        # set the dummy policy
        policy = lambda x: torch.zeros((num_envs, 19), device=sim_device)

    return env, policy
