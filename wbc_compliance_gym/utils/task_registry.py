"""Task/config/runner composition inspired by legged_gym's TaskRegistry."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

from wbc_compliance_gym.utils.config_utils import config_fingerprint, config_to_dict


@dataclass(frozen=True)
class TaskSpec:
    env_class: type
    env_cfg_factory: Callable
    train_cfg_factory: Callable
    runner_class: type
    wrappers: Tuple[Callable, ...] = ()
    play_cfg_hook: Optional[Callable] = None


class TaskRegistry:
    def __init__(self):
        self._specs: Dict[str, TaskSpec] = {}

    def register(
        self,
        name,
        env_class,
        env_cfg_factory,
        train_cfg_factory,
        runner_class,
        wrappers: Iterable[Callable] = (),
        play_cfg_hook=None,
    ):
        spec = TaskSpec(
            env_class=env_class,
            env_cfg_factory=env_cfg_factory,
            train_cfg_factory=train_cfg_factory,
            runner_class=runner_class,
            wrappers=tuple(wrappers),
            play_cfg_hook=play_cfg_hook,
        )
        previous = self._specs.get(name)
        if previous is not None and previous != spec:
            raise ValueError(f"Task {name!r} is already registered with another spec")
        self._specs[name] = spec

    def names(self):
        return tuple(sorted(self._specs))

    def get_spec(self, name):
        try:
            return self._specs[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"Task {name!r} is not registered; available tasks: {available}"
            ) from exc

    def get_cfgs(self, name):
        spec = self.get_spec(name)
        return spec.env_cfg_factory(), spec.train_cfg_factory()

    @staticmethod
    def _apply_overrides(env_cfg, train_cfg, args):
        if args is None:
            return
        if getattr(args, "num_envs", None) is not None:
            env_cfg.env.num_envs = args.num_envs
        if getattr(args, "max_iterations", None) is not None:
            train_cfg.runner.max_iterations = args.max_iterations
        if getattr(args, "save_interval", None) is not None:
            train_cfg.runner.save_interval = args.save_interval
        if getattr(args, "run_name", None):
            train_cfg.run.training_name = args.run_name

    def make_env(self, name, args=None, env_cfg=None):
        spec = self.get_spec(name)
        if env_cfg is None:
            env_cfg = spec.env_cfg_factory()
        if args is not None and getattr(args, "num_envs", None) is not None:
            env_cfg.env.num_envs = args.num_envs

        # Capture the resolved experiment configuration before the environment
        # derives runtime fields or prepares timestep-scaled reward weights.
        resolved_env_config = config_to_dict(env_cfg)

        sim_device = getattr(args, "sim_device", "cuda:0")
        headless = getattr(args, "headless", True)
        physics_engine = getattr(args, "physics_engine", "SIM_PHYSX")
        env = spec.env_class(
            sim_device=sim_device,
            headless=headless,
            cfg=env_cfg,
            physics_engine=physics_engine,
        )
        for wrapper in spec.wrappers:
            env = wrapper(env)
        env._resolved_env_config = resolved_env_config
        return env, env_cfg

    def make_alg_runner(self, env, name, args=None, train_cfg=None):
        spec = self.get_spec(name)
        if train_cfg is None:
            train_cfg = spec.train_cfg_factory()
        # Environment overrides are applied by make_env; runner overrides must
        # also be applied when this method is called independently.
        env_cfg = getattr(env, "cfg", None)
        self._apply_overrides(env_cfg, train_cfg, args)

        resolved_env_config = getattr(env, "_resolved_env_config", None)
        config_stage = "pre_environment_init"
        if resolved_env_config is None:
            resolved_env_config = config_to_dict(env_cfg)
            config_stage = "runner_creation"

        log_config = {
            "ConfigMeta": {
                "schema_version": 2,
                "env_config_stage": config_stage,
                "env_config_sha256": config_fingerprint(resolved_env_config),
            },
            "RunCfg": config_to_dict(train_cfg.run),
            "AC_Args": config_to_dict(train_cfg.policy),
            "PPO_Args": config_to_dict(train_cfg.algorithm),
            "RunnerArgs": config_to_dict(train_cfg.runner),
            "Cfg": resolved_env_config,
        }
        runner = spec.runner_class(
            env,
            device=getattr(args, "rl_device", "cuda:0"),
            task_name=train_cfg.run.task_name,
            run_name=train_cfg.run.training_name,
            train_cfg=train_cfg,
            log_config=log_config,
            wandb_init_kwargs={
                "project": "b1-loco-z1-manip",
                "group": train_cfg.run.experiment_group,
                "job_type": train_cfg.run.experiment_job_type,
            },
        )

        resume_run_dir = getattr(args, "resume_run_dir", None)
        if resume_run_dir:
            from wbc_compliance_gym.utils.artifacts import resolve_local_checkpoint

            _, checkpoint_path, _ = resolve_local_checkpoint(
                resume_run_dir, getattr(args, "checkpoint", "latest")
            )
            warm_start = getattr(
                train_cfg.run, "reset_progress_on_load", False
            )
            runner.load(
                checkpoint_path,
                restore_runner_state=not warm_start,
                restore_rng_state=not warm_start,
            )
            if warm_start:
                print(
                    "Warm-starting a new run at iteration 0 from: "
                    f"{checkpoint_path}"
                )
            else:
                print(f"Resuming training from: {checkpoint_path}")
        return runner, train_cfg


task_registry = TaskRegistry()
