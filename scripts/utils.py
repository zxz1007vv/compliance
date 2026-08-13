import random
import sys
from typing import Callable

import isaacgym
import numpy as np
import torch

sys.path.append('../')

from b1_gym.envs.b1_z1.b1_z1 import B1Z1Env
from b1_gym.envs.b1_z1.b1_z1_config import B1Z1Cfg, B1Z1CfgPPO
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
    train_cfg = B1Z1CfgPPO()
    apply_config(train_cfg.policy, local_config.get("AC_Args", {}))
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
    valid_control_modes = {"position", "force", "binary", "mixed"}
    if control_mode is not None and control_mode not in valid_control_modes:
        available = ", ".join(sorted(valid_control_modes))
        raise ValueError(
            f"Unknown control mode {control_mode!r}; choose one of: {available}"
        )
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
                             weights_path=weights_path)
    else:
        # set the dummy policy
        policy = lambda x: torch.zeros((num_envs, 19), device=sim_device)

    return env, policy
