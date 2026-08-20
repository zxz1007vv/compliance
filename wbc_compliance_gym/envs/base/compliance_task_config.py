"""Shared, robot-neutral builders for whole-body compliance tasks.

This module owns construction mechanics only.  Robot geometry, command ranges,
reward weights, termination semantics, and domain-randomization ranges belong
to each concrete task configuration.
"""

from __future__ import annotations

import os

from wbc_compliance_gym.envs.base.legged_robot_config import Cfg
from wbc_compliance_gym.utils.config_utils import ConfigNode, clone_config
from wbc_compliance_rl.algorithms.ppo_cse import PPO_Args
from wbc_compliance_rl.modules.actor_critic import AC_Args
from wbc_compliance_rl.runners.on_policy_runner import RunnerArgs


def new_compliance_env_config(cfg=None):
    """Return an isolated generic environment config unless one was supplied."""
    return clone_config(Cfg) if cfg is None else cfg


def disable_all_reward_scales(cfg):
    """Start a task reward definition from an explicit all-disabled state."""
    for name, value in vars(cfg.reward_scales).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            setattr(cfg.reward_scales, name, 0.0)


def apply_reward_scales(cfg, reward_scales):
    """Apply the complete active reward manifest for a concrete task."""
    disable_all_reward_scales(cfg)
    for name, scale in reward_scales.items():
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise TypeError(f"reward scale {name!r} must be numeric, got {scale!r}")
        if scale == 0:
            raise ValueError(
                f"active reward manifest must not contain zero scale {name!r}"
            )
        setattr(cfg.reward_scales, name, scale)
    return cfg


def active_reward_scales(cfg):
    """Return the non-zero reward scales without mutating the config."""
    return {
        name: value
        for name, value in vars(cfg.reward_scales).items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value != 0
    }


def validate_active_reward_scales(cfg, expected):
    """Reject accidental reward inheritance or missing task rewards."""
    active = active_reward_scales(cfg)
    if active != dict(expected):
        missing = sorted(set(expected) - set(active))
        unexpected = sorted(set(active) - set(expected))
        changed = sorted(
            name
            for name in set(active) & set(expected)
            if active[name] != expected[name]
        )
        raise ValueError(
            "active reward manifest mismatch: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )
    return cfg


def build_compliance_ppo_config(task_name):
    """Build the currently shared PPO-CSE setup without task inheritance."""
    policy = clone_config(AC_Args)
    policy.init_noise_std = 1.0
    policy.adaptation_labels = [
        "motion_loss",
        "dynamics_loss",
        "force_loss",
        "friction_loss",
        "gripper_pos_loss",
        "gripper_target_pos_loss",
    ]
    policy.adaptation_dims = [3, 3, 3, 1, 3, 3]
    policy.adaptation_weights = [1, 1, 0.05, 1, 10, 1]

    algorithm = clone_config(PPO_Args)
    algorithm.entropy_coef = 0.005

    runner = clone_config(RunnerArgs)
    runner.num_steps_per_env = 48
    runner.save_video_interval = 0

    run = ConfigNode(
        task_name=os.environ.get("COMPLIANCE_TASK_NAME", task_name),
        training_name=os.environ.get("COMPLIANCE_TRAINING_NAME", "wbc_release"),
        experiment_group="wbc",
        experiment_job_type="release",
    )
    return ConfigNode(policy=policy, algorithm=algorithm, runner=runner, run=run)


__all__ = [
    "active_reward_scales",
    "apply_reward_scales",
    "build_compliance_ppo_config",
    "disable_all_reward_scales",
    "new_compliance_env_config",
    "validate_active_reward_scales",
]
