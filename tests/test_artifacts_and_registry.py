import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import isaacgym  # noqa: F401 - must precede torch imports in this project.

from scripts.train import apply_task_run_defaults
from wbc_compliance_gym.envs import register_tasks
from wbc_compliance_gym.utils.artifacts import (
    load_run_config,
    resolve_latest_run,
    resolve_local_checkpoint,
    resolve_run_task,
)
from wbc_compliance_gym.utils.config_utils import ConfigNode
from wbc_compliance_gym.utils.task_registry import TaskRegistry
from wbc_compliance_rl.logging.experiment_logger import ExperimentLogger


class DummyEnv:
    def __init__(self, sim_device, headless, cfg, physics_engine):
        self.sim_device = sim_device
        self.headless = headless
        self.cfg = cfg
        self.physics_engine = physics_engine
        self.cfg.runtime_marker = "derived-by-environment"


class DummyRunner:
    def __init__(self, env, **kwargs):
        self.env = env
        self.kwargs = kwargs

    def load(self, path):
        self.loaded_path = Path(path)
        self.current_learning_iteration = 3500
        self.tot_timesteps = 123456
        self.tot_time = 789.0


def env_cfg_factory():
    return ConfigNode(env=ConfigNode(num_envs=4096))


def train_cfg_factory():
    return ConfigNode(
        policy=ConfigNode(width=64),
        algorithm=ConfigNode(learning_rate=1e-3),
        runner=ConfigNode(
            max_iterations=100,
            save_interval=10,
        ),
        run=ConfigNode(
            task_name="dummy",
            training_name="baseline",
            experiment_group="tests",
            experiment_job_type="unit",
        ),
    )


class ArtifactsAndRegistryTests(unittest.TestCase):
    def test_task_resume_defaults_do_not_override_explicit_cli_values(self):
        train_cfg = ConfigNode(
            run=ConfigNode(
                resume=True,
                resume_run_dir="configured/run",
                resume_checkpoint=1200,
            )
        )
        configured_args = SimpleNamespace(
            resume=False,
            resume_run_dir=None,
            checkpoint=None,
        )
        apply_task_run_defaults(configured_args, train_cfg)
        self.assertTrue(configured_args.resume)
        self.assertEqual("configured/run", configured_args.resume_run_dir)
        self.assertEqual(1200, configured_args.checkpoint)

        cli_args = SimpleNamespace(
            resume=True,
            resume_run_dir="cli/run",
            checkpoint=800,
        )
        apply_task_run_defaults(cli_args, train_cfg)
        self.assertEqual("cli/run", cli_args.resume_run_dir)
        self.assertEqual(800, cli_args.checkpoint)

    def test_checkpoint_resolution_supports_new_and_historical_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            (run_dir / "model_9.pt").touch()
            (checkpoint_dir / "model_000009.pt").touch()
            (checkpoint_dir / "model_000012.pt").touch()

            _, latest_path, latest_number = resolve_local_checkpoint(run_dir)
            self.assertEqual(12, latest_number)
            self.assertEqual(checkpoint_dir / "model_000012.pt", latest_path)

            _, exact_path, exact_number = resolve_local_checkpoint(run_dir, 9)
            self.assertEqual(9, exact_number)
            self.assertEqual(checkpoint_dir / "model_000009.pt", exact_path)

    def test_latest_run_and_config_are_local_only(self):
        with tempfile.TemporaryDirectory() as directory:
            log_root = Path(directory)
            task_dir = log_root / "b1_z1_ik"
            older = task_dir / "older"
            newer = task_dir / "newer"
            for index, run_dir in enumerate((older, newer), start=1):
                (run_dir / "checkpoints").mkdir(parents=True)
                (run_dir / "config.json").write_text(
                    json.dumps({"index": index}), encoding="utf-8"
                )
                checkpoint = run_dir / "checkpoints" / f"model_{index:06d}.pt"
                checkpoint.touch()
                os.utime(checkpoint, (index, index))

            self.assertEqual(newer, resolve_latest_run(log_root, "b1_z1_ik"))
            self.assertEqual({"index": 2}, load_run_config(newer))
            self.assertEqual("b1_z1_ik", resolve_run_task(newer))

    def test_run_task_prefers_saved_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "folder_task" / "run"
            run_dir.mkdir(parents=True)
            config = {"RunCfg": {"task_name": "saved_task"}}
            (run_dir / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            self.assertEqual("saved_task", resolve_run_task(run_dir))

    def test_registry_returns_isolated_configs_and_applies_cli_overrides(self):
        registry = TaskRegistry()
        registry.register(
            "dummy",
            DummyEnv,
            env_cfg_factory,
            train_cfg_factory,
            DummyRunner,
            play_cfg_hook=lambda cfg, **kwargs: cfg,
        )
        self.assertIsNotNone(registry.get_spec("dummy").play_cfg_hook)
        first_env_cfg, first_train_cfg = registry.get_cfgs("dummy")
        second_env_cfg, second_train_cfg = registry.get_cfgs("dummy")
        first_env_cfg.env.num_envs = 1
        first_train_cfg.runner.max_iterations = 1
        self.assertEqual(4096, second_env_cfg.env.num_envs)
        self.assertEqual(100, second_train_cfg.runner.max_iterations)

        args = SimpleNamespace(
            num_envs=32,
            max_iterations=7,
            save_interval=3,
            run_name="override",
            sim_device="cpu",
            rl_device="cpu",
            headless=True,
            physics_engine="SIM_PHYSX",
            resume_run_dir=None,
        )
        env, env_cfg = registry.make_env("dummy", args)
        runner, train_cfg = registry.make_alg_runner(env, "dummy", args)
        self.assertEqual(32, env_cfg.env.num_envs)
        self.assertEqual(7, train_cfg.runner.max_iterations)
        self.assertEqual(3, train_cfg.runner.save_interval)
        self.assertEqual("override", train_cfg.run.training_name)
        self.assertIs(train_cfg, runner.kwargs["train_cfg"])
        logged = runner.kwargs["log_config"]
        self.assertEqual("pre_environment_init", logged["ConfigMeta"]["env_config_stage"])
        self.assertNotIn("runtime_marker", logged["Cfg"])
        self.assertEqual(32, logged["Cfg"]["env"]["num_envs"])

    def test_registry_can_warm_start_with_fresh_progress_numbering(self):
        registry = TaskRegistry()
        registry.register(
            "dummy",
            DummyEnv,
            env_cfg_factory,
            train_cfg_factory,
            DummyRunner,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "model_003500.pt"
            checkpoint.touch()
            args = SimpleNamespace(
                num_envs=None,
                max_iterations=None,
                save_interval=None,
                run_name=None,
                sim_device="cpu",
                rl_device="cpu",
                headless=True,
                physics_engine="SIM_PHYSX",
                resume_run_dir=directory,
                checkpoint="3500",
            )
            env, _ = registry.make_env("dummy", args)
            train_cfg = train_cfg_factory()
            train_cfg.run.reset_progress_on_load = True
            runner, _ = registry.make_alg_runner(
                env, "dummy", args, train_cfg=train_cfg
            )

        self.assertEqual(checkpoint, runner.loaded_path)
        self.assertEqual(0, runner.current_learning_iteration)
        self.assertEqual(0, runner.tot_timesteps)
        self.assertEqual(0.0, runner.tot_time)

    def test_builtin_registration_is_idempotent(self):
        first = register_tasks()
        second = register_tasks()
        self.assertIs(first, second)
        self.assertIn("b1_z1_ik", first.names())

    def test_experiment_logger_creates_canonical_layout(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"COMPLIANCE_LOG_DIR": directory, "COMPLIANCE_LOGGER": "none"},
        ):
            logger = ExperimentLogger("task", "run", config={"value": 1})
            try:
                self.assertTrue((logger.run_dir / "config.json").is_file())
                self.assertTrue(logger.checkpoint_dir.is_dir())
                self.assertTrue(logger.tensorboard_dir.is_dir())
                self.assertTrue(logger.export_dir.is_dir())
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()
