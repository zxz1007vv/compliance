"""Opt-in diagnostics for asynchronous CUDA/PhysX failures.

The checks in this module intentionally synchronize CUDA and inspect complete
simulation tensors.  They are therefore disabled unless
``COMPLIANCE_CUDA_DEBUG`` is explicitly enabled.
"""

import os
import sys

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _read_bool(environ, name, default=False):
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    choices = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"{name} must be one of: {choices}; got {value!r}")


def _read_positive_int(environ, name, default):
    value = environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed


class CudaPhysicsDebugger:
    """Synchronize stages and validate physics tensors when explicitly enabled."""

    @classmethod
    def from_environment(cls, device, num_envs, environ=None):
        environ = os.environ if environ is None else environ
        if not _read_bool(environ, "COMPLIANCE_CUDA_DEBUG"):
            return None
        return cls(
            device=device,
            num_envs=num_envs,
            stats_interval=_read_positive_int(
                environ, "COMPLIANCE_CUDA_DEBUG_INTERVAL", 100
            ),
            synchronize_cuda=_read_bool(
                environ, "COMPLIANCE_CUDA_DEBUG_SYNC", default=True
            ),
            launch_blocking=environ.get("CUDA_LAUNCH_BLOCKING", "0"),
        )

    def __init__(
        self,
        device,
        num_envs,
        stats_interval,
        synchronize_cuda,
        launch_blocking,
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.stats_interval = stats_interval
        self.synchronize_cuda = synchronize_cuda
        self.launch_blocking = launch_blocking
        self.last_stage = None
        self.last_tensor_summary = None
        self.last_action_summary = None
        self.last_dof_summary = None
        self.last_contact_summary = None
        self.last_safety_summary = None

    def _context(self, stage, step, substep):
        context = f"stage={stage} step={step}"
        if substep is not None:
            context += f" substep={substep}"
        return context

    def _should_report_stats(self, step, substep):
        return step % self.stats_interval == 0 and substep in (None, 0)

    def log_startup(self, cfg):
        physx = cfg.sim.physx
        fields = {
            "device": str(self.device),
            "num_envs": self.num_envs,
            "dt": cfg.sim.dt,
            "substeps": cfg.sim.substeps,
            "control_decimation": cfg.control.decimation,
            "self_collisions": cfg.asset.self_collisions,
            "max_gpu_contact_pairs": physx.max_gpu_contact_pairs,
            "default_buffer_size_multiplier": physx.default_buffer_size_multiplier,
            "contact_collection": physx.contact_collection,
            "stats_interval": self.stats_interval,
            "synchronize_cuda": self.synchronize_cuda,
            "CUDA_LAUNCH_BLOCKING": self.launch_blocking,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                fields["gpu"] = torch.cuda.get_device_name(self.device)
                fields["torch_cuda"] = torch.version.cuda
            except RuntimeError as exc:
                fields["gpu_query_error"] = repr(exc)
        rendered = " ".join(f"{name}={value}" for name, value in fields.items())
        print(f"[cuda-debug] enabled {rendered}", file=sys.stderr, flush=True)

    def stage(self, stage, step, substep=None, synchronize=False):
        """Record a stage and optionally force asynchronous CUDA errors to surface."""
        context = self._context(stage, step, substep)
        self.last_stage = context
        if self._should_report_stats(step, substep):
            print(f"[cuda-debug] reached {context}", file=sys.stderr, flush=True)

        if not (
            synchronize
            and self.synchronize_cuda
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            return
        try:
            torch.cuda.synchronize(self.device)
        except RuntimeError as exc:
            print(
                f"[cuda-debug] CUDA failure first observed while synchronizing "
                f"{context}: {exc} last_valid_tensors={self.last_tensor_summary!r} "
                f"last_action_diagnostics={self.last_action_summary!r} "
                f"last_dof_diagnostics={self.last_dof_summary!r} "
                f"last_contact_diagnostics={self.last_contact_summary!r} "
                f"last_safety_resets={self.last_safety_summary!r}",
                file=sys.stderr,
                flush=True,
            )
            raise

    def check_tensors(self, stage, tensors, step, substep=None):
        """Raise immediately on NaN/Inf and periodically report tensor ranges."""
        report_stats = self._should_report_stats(step, substep)
        stats = []
        for name, tensor in tensors.items():
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.numel() == 0:
                if report_stats:
                    stats.append(f"{name}=empty")
                continue

            finite = torch.isfinite(tensor)
            if not bool(finite.all().item()):
                bad_indices = (~finite).nonzero(as_tuple=False)
                bad_count = int(bad_indices.shape[0])
                samples = bad_indices[:8].detach().cpu().tolist()
                env_ids = []
                if tensor.ndim > 0 and tensor.shape[0] == self.num_envs:
                    env_ids = sorted({index[0] for index in samples})
                context = self._context(stage, step, substep)
                print(
                    f"[cuda-debug] non-finite tensor detected {context} "
                    f"tensor={name} shape={tuple(tensor.shape)} bad_count={bad_count} "
                    f"sample_indices={samples} env_ids={env_ids}",
                    file=sys.stderr,
                    flush=True,
                )
                raise FloatingPointError(
                    f"Non-finite values in {name} at {context}; "
                    f"sample indices: {samples}"
                )

            minimum = tensor.min().item()
            maximum = tensor.max().item()
            max_abs = tensor.abs().max().item()
            stats.append(
                f"{name}[min={minimum:.5g},max={maximum:.5g},abs={max_abs:.5g}]"
            )

        context = self._context(stage, step, substep)
        self.last_tensor_summary = f"{context} {' '.join(stats)}"

        if not report_stats:
            return

        memory = ""
        if self.device.type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            memory = f" memory_mib[allocated={allocated:.1f},reserved={reserved:.1f}]"
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                memory += (
                    " device_memory_mib"
                    f"[free={free / (1024 ** 2):.1f},"
                    f"total={total / (1024 ** 2):.1f}]"
                )
            except (AttributeError, RuntimeError):
                # Older PyTorch versions may not expose the CUDA driver-level
                # free/total query. Tensor diagnostics remain usable there.
                pass
        print(
            f"[cuda-debug] tensors {self.last_tensor_summary}{memory}",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _topk_named(values, names, limit=5):
        if values.ndim != 2:
            raise ValueError(f"Expected a rank-2 diagnostic tensor, got {values.shape}")
        if values.numel() == 0:
            return "none"

        count = min(limit, values.numel())
        top_values, top_indices = torch.topk(values.reshape(-1), count)
        top_values = top_values.detach().cpu().tolist()
        top_indices = top_indices.detach().cpu().tolist()
        item_count = values.shape[1]
        rendered = []
        for value, flat_index in zip(top_values, top_indices):
            if value <= 0:
                continue
            env_id, item_index = divmod(flat_index, item_count)
            item_name = names[item_index] if item_index < len(names) else item_index
            rendered.append(
                f"env={env_id},item={item_name},value={value:.5g}"
            )
        return ";".join(rendered) if rendered else "none"

    def report_dof_diagnostics(
        self,
        positions,
        velocities,
        requested_targets,
        hard_clamped_targets,
        applied_targets,
        hard_limits,
        velocity_limits,
        dof_names,
        step,
        substep=None,
    ):
        """Retain separate hard-clamp, slew-limit, and velocity outliers."""
        lower = hard_limits[:, 0].unsqueeze(0)
        upper = hard_limits[:, 1].unsqueeze(0)
        limited = (
            torch.isfinite(hard_limits).all(dim=1)
            & (hard_limits[:, 1] > hard_limits[:, 0])
        ).unsqueeze(0)
        position_violation = torch.maximum(
            (lower - positions).clamp(min=0),
            (positions - upper).clamp(min=0),
        )
        position_violation = torch.where(
            limited, position_violation, torch.zeros_like(position_violation)
        )

        hard_target_clamp = torch.zeros_like(applied_targets)
        if requested_targets is not None:
            hard_target_clamp = (
                requested_targets - hard_clamped_targets
            ).abs()
        target_slew_limit = (
            hard_clamped_targets - applied_targets
        ).abs()
        hard_clamp_rate = 100.0 * float(
            (hard_target_clamp > 1e-6).count_nonzero().item()
        ) / max(hard_target_clamp.numel(), 1)
        slew_limit_rate = 100.0 * float(
            (target_slew_limit > 1e-6).count_nonzero().item()
        ) / max(target_slew_limit.numel(), 1)

        valid_velocity_limit = (
            torch.isfinite(velocity_limits) & (velocity_limits > 0)
        ).unsqueeze(0)
        velocity_excess = (velocities.abs() - velocity_limits.unsqueeze(0)).clamp(
            min=0
        )
        velocity_excess = torch.where(
            valid_velocity_limit,
            velocity_excess,
            torch.zeros_like(velocity_excess),
        )

        context = self._context("dof_diagnostics", step, substep)
        self.last_dof_summary = (
            f"{context} hard_limit_violation="
            f"[{self._topk_named(position_violation, dof_names)}] "
            f"hard_target_clamp_rate={hard_clamp_rate:.3f}% "
            f"hard_target_clamp=[{self._topk_named(hard_target_clamp, dof_names)}] "
            f"target_slew_limit_rate={slew_limit_rate:.3f}% "
            f"target_slew_limit=[{self._topk_named(target_slew_limit, dof_names)}] "
            f"velocity_excess=[{self._topk_named(velocity_excess, dof_names)}]"
        )
        if self._should_report_stats(step, substep):
            print(
                f"[cuda-debug] {self.last_dof_summary}",
                file=sys.stderr,
                flush=True,
            )

    def report_action_diagnostics(self, actions, dof_names, clip, step):
        """Retain aggregate and per-DOF policy action saturation rates."""
        saturated = torch.abs(actions) >= (float(clip) - 1e-6)
        per_dof_rate = saturated.float().mean(dim=0) * 100.0
        overall_rate = 100.0 * float(saturated.count_nonzero().item()) / max(
            saturated.numel(), 1
        )
        context = self._context("action_diagnostics", step, None)
        self.last_action_summary = (
            f"{context} saturation_rate={overall_rate:.3f}% "
            f"per_dof_saturation="
            f"[{self._topk_named(per_dof_rate.unsqueeze(0), dof_names)}]"
        )
        if self._should_report_stats(step, None):
            print(
                f"[cuda-debug] {self.last_action_summary}",
                file=sys.stderr,
                flush=True,
            )

    def report_contact_diagnostics(
        self,
        contact_forces,
        body_names,
        step,
        substep=None,
        ignored_body_indices=None,
    ):
        """Retain contact outliers and low-force non-foot contact occupancy."""
        magnitudes = torch.linalg.norm(contact_forces, dim=-1)
        included = torch.ones(
            magnitudes.shape[1], dtype=torch.bool, device=magnitudes.device
        )
        if ignored_body_indices is not None:
            included[ignored_body_indices] = False
        included_magnitudes = magnitudes[:, included]
        active_1n = included_magnitudes > 1.0
        active_10n = included_magnitudes > 10.0
        envs_1n = torch.any(active_1n, dim=1)
        context = self._context("contact_diagnostics", step, substep)
        self.last_contact_summary = (
            f"{context} nonignored_bodies_gt1N="
            f"{int(active_1n.count_nonzero().item())} "
            f"nonignored_bodies_gt10N="
            f"{int(active_10n.count_nonzero().item())} "
            f"envs_with_nonignored_contact="
            f"{int(envs_1n.count_nonzero().item())} "
            f"top_contacts=[{self._topk_named(magnitudes, body_names)}]"
        )
        if self._should_report_stats(step, substep):
            print(
                f"[cuda-debug] {self.last_contact_summary}",
                file=sys.stderr,
                flush=True,
            )

    def report_safety_resets(self, reset_masks, step):
        """Retain task safety-reset counts for later CUDA failure reports."""
        counts = {
            name: int(mask.count_nonzero().item())
            for name, mask in reset_masks.items()
        }
        context = self._context("safety_resets", step, None)
        rendered = " ".join(
            f"{name}={count}" for name, count in counts.items()
        )
        self.last_safety_summary = f"{context} {rendered}"
        if self._should_report_stats(step, None):
            print(
                f"[cuda-debug] {self.last_safety_summary}",
                file=sys.stderr,
                flush=True,
            )

    def report_exception(self, stage, step, substep, exc):
        """Report only CPU-side context; the CUDA context may already be poisoned."""
        context = self._context(stage, step, substep)
        print(
            f"[cuda-debug] exception {context} last_stage={self.last_stage!r} "
            f"last_valid_tensors={self.last_tensor_summary!r} "
            f"last_action_diagnostics={self.last_action_summary!r} "
            f"last_dof_diagnostics={self.last_dof_summary!r} "
            f"last_contact_diagnostics={self.last_contact_summary!r} "
            f"last_safety_resets={self.last_safety_summary!r} "
            f"type={type(exc).__name__} message={exc}",
            file=sys.stderr,
            flush=True,
        )
