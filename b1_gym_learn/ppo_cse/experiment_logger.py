import json
import numbers
import os
import re
from datetime import datetime
from pathlib import Path

import torch


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("._-") or "unnamed"


class ExperimentLogger:
    """Own one run directory and mirror metrics to the selected backends."""

    VALID_BACKENDS = {"tensorboard", "wandb", "both", "none"}

    def __init__(self, task_name, run_name, config=None, wandb_init_kwargs=None):
        self.backend = os.environ.get("COMPLIANCE_LOGGER", "tensorboard").strip().lower()
        if self.backend not in self.VALID_BACKENDS:
            choices = ", ".join(sorted(self.VALID_BACKENDS))
            raise ValueError(f"Invalid COMPLIANCE_LOGGER={self.backend!r}; choose one of: {choices}")

        self.task_name = safe_name(task_name)
        self.training_name = safe_name(run_name)
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = f"{self.timestamp}_{self.training_name}"

        project_root = Path(__file__).resolve().parents[2]
        log_root = Path(os.environ.get("COMPLIANCE_LOG_DIR", project_root / "logs"))
        self.run_dir = log_root / self.task_name / self.run_id
        self.checkpoint_dir = self.run_dir
        self.tensorboard_dir = self.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.writer = None
        self.wandb = None

        if config is not None:
            config_text = json.dumps(config, indent=2, default=str)
            (self.run_dir / "config.json").write_text(config_text + "\n", encoding="utf-8")
        else:
            config_text = None

        if self.backend in {"tensorboard", "both"}:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "TensorBoard is the default logger but is not installed. "
                    "Install it with: python -m pip install tensorboard"
                ) from exc

            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir), flush_secs=10)
            if config_text is not None:
                self.writer.add_text("config", f"```json\n{config_text}\n```", 0)

        if self.backend in {"wandb", "both"}:
            import wandb

            init_kwargs = dict(wandb_init_kwargs or {})
            init_kwargs.setdefault("name", self.run_id)
            init_kwargs.setdefault("config", config)
            wandb.init(**init_kwargs)
            self.wandb = wandb

        print(f"Run directory: {self.run_dir.resolve()}")
        if self.writer is not None:
            print(f"TensorBoard directory: {self.tensorboard_dir.resolve()}")
        print(f"Experiment logger: {self.backend}")

    @staticmethod
    def _materialize(metrics):
        """Convert scalar tensors in one batch per device to avoid repeated CUDA syncs."""
        result = {}
        tensor_groups = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    tensor_groups.setdefault(value.device, []).append((name, value.detach().float().reshape(())))
            elif isinstance(value, numbers.Number):
                result[name] = value
            else:
                try:
                    if getattr(value, "size", None) == 1:
                        result[name] = value.item()
                except (AttributeError, ValueError):
                    pass

        for items in tensor_groups.values():
            values = torch.stack([value for _, value in items]).cpu().tolist()
            result.update({name: value for (name, _), value in zip(items, values)})
        return result

    def log(self, metrics, step):
        scalar_metrics = self._materialize(metrics)
        if self.writer is not None:
            for name, value in scalar_metrics.items():
                self.writer.add_scalar(name, value, step)
        if self.wandb is not None:
            self.wandb.log(scalar_metrics, step=step)
        return scalar_metrics

    def watch(self, model, log_freq):
        if self.wandb is not None:
            self.wandb.watch(model, log=None, log_freq=log_freq)

    def save(self, path):
        if self.wandb is not None:
            self.wandb.save(str(path), base_path=str(self.run_dir))

    def log_video(self, video_array, step, fps):
        if self.writer is not None:
            video = torch.as_tensor(video_array).unsqueeze(0)
            self.writer.add_video("video", video, step, fps=fps)
        if self.wandb is not None:
            self.wandb.log(
                {"video": self.wandb.Video(video_array, fps=fps)},
                step=step,
            )

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()
