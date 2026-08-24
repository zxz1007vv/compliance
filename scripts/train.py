"""Training entry point for registered learning-compliance tasks."""

import os
import sys
from types import SimpleNamespace

import isaacgym  # Isaac Gym must be imported before torch.

from wbc_compliance_gym.envs import register_tasks
from wbc_compliance_gym.utils.artifacts import (
    load_run_config,
    resolve_latest_run,
    resolve_run_task,
)
from wbc_compliance_gym.utils.helpers import get_args
from wbc_compliance_gym.utils.task_registry import task_registry


def configure_env(task_name):
    """Return a fresh environment config for a registered task."""
    register_tasks()
    return task_registry.get_spec(task_name).env_cfg_factory()


def apply_task_run_defaults(args, train_cfg):
    """Apply task-local resume defaults without overriding explicit CLI input."""
    run_cfg = train_cfg.run
    explicit_resume = bool(
        getattr(args, "resume", False)
        or getattr(args, "resume_run_dir", None)
    )
    if not explicit_resume and getattr(run_cfg, "resume", False):
        args.resume = True
        args.resume_run_dir = getattr(run_cfg, "resume_run_dir", None)
        if getattr(args, "checkpoint", None) is None:
            args.checkpoint = getattr(run_cfg, "resume_checkpoint", "latest")

    if getattr(args, "checkpoint", None) is None:
        args.checkpoint = "latest"
    return args


def train(args):
    """Compose the selected task and run its registered on-policy runner."""
    if getattr(args, "logger", None):
        os.environ["COMPLIANCE_LOGGER"] = args.logger

    registry = register_tasks()
    if getattr(args, "list_tasks", False):
        print("\n".join(registry.names()))
        return None
    task_spec = registry.get_spec(args.task)
    train_cfg = task_spec.train_cfg_factory()
    apply_task_run_defaults(args, train_cfg)

    resume_run_dir = getattr(args, "resume_run_dir", None)
    if getattr(args, "resume", False) and not resume_run_dir:
        resume_run_dir = str(resolve_latest_run(task_name=args.task))
        args.resume_run_dir = resume_run_dir
        print(f"Automatically selected latest run to resume: {resume_run_dir}")
    if resume_run_dir:
        saved_config = load_run_config(resume_run_dir)
        saved_task = resolve_run_task(resume_run_dir, saved_config)
        if saved_task != args.task:
            raise ValueError(
                f"Requested task {args.task!r} does not match resume run task "
                f"{saved_task!r}: {resume_run_dir}"
            )
    env, _ = task_registry.make_env(args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(
        env, args.task, args=args, train_cfg=train_cfg
    )
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
        resume=resume_run_dir is not None,
        list_tasks=False,
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
