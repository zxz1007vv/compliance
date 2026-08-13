"""Local run and checkpoint discovery with legacy layout compatibility."""

import json
import os
import re
from pathlib import Path

from wbc_compliance_gym.envs import DEFAULT_TASK


def resolve_run_dir(run_dir):
    """Resolve an absolute, cwd-relative, or project-relative run directory."""
    requested = Path(run_dir).expanduser()
    if requested.is_absolute() or requested.exists():
        return requested.resolve()
    project_relative = Path(__file__).resolve().parents[2] / requested
    if project_relative.exists():
        return project_relative.resolve()
    return requested.resolve()


def checkpoint_directories(run_dir):
    """Yield the canonical directory first and the historical run root second."""
    run_dir = resolve_run_dir(run_dir)
    return run_dir / "checkpoints", run_dir


def resolve_local_checkpoint(run_dir, checkpoint="latest"):
    """Resolve model checkpoints from both current and historical run layouts."""
    run_dir = resolve_run_dir(run_dir)
    checkpoint_dirs = checkpoint_directories(run_dir)
    numbered = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.is_dir():
            continue
        for path in checkpoint_dir.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match:
                numbered.append((int(match.group(1)), path))

    if str(checkpoint).lower() == "latest":
        if numbered:
            checkpoint_number, checkpoint_path = max(
                numbered,
                key=lambda item: (
                    item[0],
                    item[1].parent == run_dir / "checkpoints",
                ),
            )
        else:
            latest_candidates = [
                directory / "model_latest.pt" for directory in checkpoint_dirs
            ]
            checkpoint_path = next(
                (path for path in latest_candidates if path.is_file()), None
            )
            if checkpoint_path is None:
                searched = ", ".join(str(path) for path in checkpoint_dirs)
                raise FileNotFoundError(
                    f"No model checkpoints found in: {searched}"
                )
            checkpoint_number = -1
    else:
        checkpoint_number = int(checkpoint)
        names = (
            f"model_{checkpoint_number:06d}.pt",
            f"model_{checkpoint_number}.pt",
        )
        candidates = [
            directory / name
            for directory in checkpoint_dirs
            for name in names
        ]
        checkpoint_path = next(
            (path for path in candidates if path.is_file()), None
        )
        if checkpoint_path is None:
            searched = ", ".join(str(path) for path in checkpoint_dirs)
            raise FileNotFoundError(
                f"No model checkpoint {checkpoint_number} in: {searched}"
            )
    return run_dir, checkpoint_path, checkpoint_number


def resolve_latest_run(log_root=None, task_name=None):
    project_root = Path(__file__).resolve().parents[2]
    if log_root is None:
        log_root = Path(
            os.environ.get("COMPLIANCE_LOG_DIR", project_root / "logs")
        )
    else:
        log_root = Path(log_root).expanduser()
    task_name = task_name or os.environ.get("COMPLIANCE_TASK_NAME", DEFAULT_TASK)
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


def load_run_config(run_dir):
    config_path = resolve_run_dir(run_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_run_task(run_dir, config=None):
    """Return the task recorded by a run, falling back to its parent folder."""
    run_dir = resolve_run_dir(run_dir)
    config = load_run_config(run_dir) if config is None else config
    task_name = config.get("RunCfg", {}).get("task_name")
    return task_name or run_dir.parent.name
