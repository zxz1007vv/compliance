"""Training entry point for registered learning-compliance tasks."""

import os
import sys
from types import SimpleNamespace

import isaacgym  # Isaac Gym must be imported before torch.

from b1_gym.envs import register_tasks
from b1_gym.envs.b1_z1.b1_z1_config import B1Z1Cfg
from b1_gym.utils.helpers import get_args
from b1_gym.utils.task_registry import task_registry


def configure_env():
    """Compatibility entry point returning the isolated B1+Z1 task config."""
    return B1Z1Cfg()


def train(args):
    """Compose the selected task and run its registered on-policy runner."""
    if getattr(args, "logger", None):
        os.environ["COMPLIANCE_LOGGER"] = args.logger

    register_tasks()
    env, _ = task_registry.make_env(args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(env, args.task, args=args)
    try:
        runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        env.close()
    return runner


def train_b1_z1_IK(
    headless=True,
    max_iterations=None,
    save_interval=None,
    training_name=None,
    resume_run_dir=None,
    checkpoint="latest",
    **deps,
):
    """Backward-compatible programmatic entry point used by older workflows."""
    args = SimpleNamespace(
        task="b1_z1_ik",
        sim_device=deps.pop("sim_device", "cuda:0"),
        rl_device=deps.pop("rl_device", "cuda:0"),
        physics_engine=deps.pop("physics_engine", "SIM_PHYSX"),
        num_envs=deps.pop("num_envs", None),
        max_iterations=max_iterations,
        save_interval=save_interval,
        run_name=training_name,
        resume_run_dir=resume_run_dir,
        checkpoint=checkpoint,
        logger=deps.pop("logger", None),
        headless=headless,
    )
    if deps:
        names = ", ".join(sorted(deps))
        raise TypeError(f"Unexpected training options: {names}")
    return train(args)


if __name__ == "__main__":
    train(get_args())
    # Isaac Gym's native runtime may fault during interpreter teardown after
    # the simulator has already been destroyed. All logs/checkpoints are closed
    # above, so use the same clean process exit strategy as scripts/play.py.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
