"""Continuous, mode-structured planar velocity command curriculum."""

from dataclasses import dataclass

import torch


LOCOMOTION_MODE_NAMES = (
    "stand", "pure_x", "pure_y", "pure_yaw", "xy", "x_yaw", "y_yaw", "full",
)
LOCOMOTION_MODE_TO_ID = {
    name: mode_id for mode_id, name in enumerate(LOCOMOTION_MODE_NAMES)
}
LOCOMOTION_MODE_ACTIVE_AXES = torch.tensor(
    (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ),
    dtype=torch.bool,
)
PURE_MODE_BY_AXIS = ("pure_x", "pure_y", "pure_yaw")
AXIS_NAMES = ("vx", "vy", "yaw")


def resolve_yaw_gait_phase_slots(wheel_dof_names, phase_order):
    """Map each configured wheel to its slot in a four-beat yaw crawl."""
    wheel_names = [
        name[: -len("_FOOT_JOINT")]
        if name.endswith("_FOOT_JOINT")
        else name
        for name in wheel_dof_names
    ]
    phase_order = list(phase_order)
    if len(wheel_names) != 4 or len(set(wheel_names)) != 4:
        raise ValueError("yaw gait requires four uniquely named wheels")
    if len(phase_order) != 4 or len(set(phase_order)) != 4:
        raise ValueError("yaw_gait_phase_order must contain four unique legs")
    if set(phase_order) != set(wheel_names):
        raise ValueError(
            "yaw_gait_phase_order must match the configured wheel names"
        )
    slot_by_name = {name: slot for slot, name in enumerate(phase_order)}
    return tuple(slot_by_name[name] for name in wheel_names)


def yaw_swing_envelope(swing_progress, transition_fraction):
    """Smoothly unload and reload the scheduled wheel within one gait beat."""
    transition_fraction = float(transition_fraction)
    if not 0.0 < transition_fraction < 0.5:
        raise ValueError("yaw gait transition fraction must lie in (0, 0.5)")
    progress = torch.remainder(swing_progress, 1.0)
    unload = (progress / transition_fraction).clip(0.0, 1.0)
    reload = ((1.0 - progress) / transition_fraction).clip(0.0, 1.0)
    unload = torch.square(unload) * (3.0 - 2.0 * unload)
    reload = torch.square(reload) * (3.0 - 2.0 * reload)
    return unload * reload


def yaw_swing_trajectory(swing_progress, transition_fraction):
    """Return smooth swing progress restricted to the middle of a gait beat."""
    transition_fraction = float(transition_fraction)
    if not 0.0 < transition_fraction < 0.5:
        raise ValueError("yaw gait transition fraction must lie in (0, 0.5)")
    progress = torch.remainder(swing_progress, 1.0)
    motion_duration = 1.0 - 2.0 * transition_fraction
    motion_progress = (
        (progress - transition_fraction) / motion_duration
    ).clip(0.0, 1.0)
    return torch.square(motion_progress) * (3.0 - 2.0 * motion_progress)


def _pair(value, name):
    if len(value) != 2 or float(value[0]) > float(value[1]):
        raise ValueError(f"commands.{name} must be an ordered [min, max] pair")
    return [float(value[0]), float(value[1])]


def validate_locomotion_mode_mixture(mixture, ranges):
    unknown = set(mixture) - set(LOCOMOTION_MODE_NAMES)
    if unknown:
        raise ValueError(f"unknown locomotion modes: {sorted(unknown)}")
    probabilities = [float(mixture.get(name, 0.0)) for name in LOCOMOTION_MODE_NAMES]
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("locomotion mode probabilities must be non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("locomotion mode probabilities must have a positive sum")
    for mode_id, probability in enumerate(probabilities):
        if probability == 0.0:
            continue
        for axis, active in enumerate(LOCOMOTION_MODE_ACTIVE_AXES[mode_id]):
            if active and ranges[axis][0] == 0.0 and ranges[axis][1] == 0.0:
                raise ValueError(
                    f"locomotion mode {LOCOMOTION_MODE_NAMES[mode_id]!r} activates "
                    f"{AXIS_NAMES[axis]}, but its current range is fixed at zero"
                )
    return [probability / total for probability in probabilities]


@dataclass
class _AxisStatistics:
    pure_error_sum: float = 0.0
    pure_sample_count: int = 0
    active_error_sum: float = 0.0
    active_sample_count: int = 0
    pure_mae: float = float("nan")
    active_mae: float = float("nan")
    pure_mae_ema: float = float("nan")
    consecutive_successes: int = 0
    expansion_count: int = 0


class ContinuousVelocityCurriculum:
    """Own runtime ranges, structured sampling, and independent expansion."""

    def __init__(self, command_cfg):
        initial = (
            _pair(command_cfg.lin_vel_x, "lin_vel_x"),
            _pair(command_cfg.lin_vel_y, "lin_vel_y"),
            _pair(command_cfg.ang_vel_yaw, "ang_vel_yaw"),
        )
        self.limits = (
            _pair(command_cfg.limit_vel_x, "limit_vel_x"),
            _pair(command_cfg.limit_vel_y, "limit_vel_y"),
            _pair(command_cfg.limit_vel_yaw, "limit_vel_yaw"),
        )
        for axis, (current, limit) in enumerate(zip(initial, self.limits)):
            if current[0] < limit[0] or current[1] > limit[1]:
                raise ValueError(
                    f"initial {AXIS_NAMES[axis]} range {current} must lie within {limit}"
                )
        self.enabled = bool(command_cfg.command_curriculum)
        source_ranges = initial if self.enabled else self.limits
        self.current_ranges = [list(value) for value in source_ranges]
        minimum_magnitudes = getattr(
            command_cfg, "min_active_command_magnitude", (0.0, 0.0, 0.0)
        )
        if len(minimum_magnitudes) != len(AXIS_NAMES):
            raise ValueError(
                "commands.min_active_command_magnitude must contain vx, vy, yaw"
            )
        self.minimum_active_magnitudes = tuple(
            float(value) for value in minimum_magnitudes
        )
        if any(value < 0.0 for value in self.minimum_active_magnitudes):
            raise ValueError(
                "commands.min_active_command_magnitude must be non-negative"
            )
        self.steps = (
            float(command_cfg.vx_curriculum_step),
            float(command_cfg.vy_curriculum_step),
            float(command_cfg.yaw_curriculum_step),
        )
        self.success_thresholds = (
            float(command_cfg.vx_success_threshold),
            float(command_cfg.vy_success_threshold),
            float(command_cfg.yaw_success_threshold),
        )
        self.minimum_samples = int(command_cfg.velocity_curriculum_min_samples)
        self.required_successes = int(command_cfg.velocity_curriculum_required_successes)
        self.ema_alpha = float(command_cfg.velocity_curriculum_ema_alpha)
        if any(step <= 0.0 for step in self.steps):
            raise ValueError("velocity curriculum steps must be positive")
        if any(value <= 0.0 for value in self.success_thresholds):
            raise ValueError("velocity curriculum success thresholds must be positive")
        if self.minimum_samples <= 0 or self.required_successes <= 0:
            raise ValueError("velocity curriculum sample/success counts must be positive")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("velocity_curriculum_ema_alpha must lie in (0, 1]")

        mixture = command_cfg.planar_command_mixture
        if hasattr(mixture, "items"):
            mixture = dict(mixture.items())
        self.mode_probabilities = validate_locomotion_mode_mixture(
            mixture, self.current_ranges
        )
        for axis, (value_range, minimum) in enumerate(
            zip(self.current_ranges, self.minimum_active_magnitudes)
        ):
            axis_is_active = any(
                probability > 0.0
                and bool(LOCOMOTION_MODE_ACTIVE_AXES[mode_id, axis])
                for mode_id, probability in enumerate(self.mode_probabilities)
            )
            if (
                axis_is_active
                and minimum > 0.0
                and max(abs(value_range[0]), abs(value_range[1])) < minimum
            ):
                raise ValueError(
                    f"commands.{AXIS_NAMES[axis]} range cannot satisfy minimum "
                    f"active magnitude {minimum}"
                )
        self.statistics = [_AxisStatistics() for _ in AXIS_NAMES]
        self.last_expanded = [False, False, False]

    def sample(self, count, device, *, generator=None):
        probabilities = torch.tensor(
            self.mode_probabilities, dtype=torch.float, device=device
        )
        mode_ids = torch.multinomial(
            probabilities, count, replacement=True, generator=generator
        )
        commands = torch.zeros(count, 3, dtype=torch.float, device=device)
        active_axes = LOCOMOTION_MODE_ACTIVE_AXES.to(device)[mode_ids]
        for axis, value_range in enumerate(self.current_ranges):
            active = active_axes[:, axis]
            active_count = int(active.sum().item())
            if active_count:
                commands[active, axis] = self._sample_active_values(
                    active_count,
                    value_range,
                    self.minimum_active_magnitudes[axis],
                    device,
                    generator,
                )
        return commands, mode_ids

    @staticmethod
    def _sample_active_values(count, value_range, minimum, device, generator):
        """Sample uniformly while excluding a near-zero inactive band."""
        low, high = value_range
        if minimum <= 0.0:
            unit = torch.rand(count, device=device, generator=generator)
            return low + unit * (high - low)

        negative_high = min(high, -minimum)
        positive_low = max(low, minimum)
        negative_length = max(0.0, negative_high - low)
        positive_length = max(0.0, high - positive_low)
        total_length = negative_length + positive_length
        if total_length <= 0.0:
            fixed_candidates = [
                value for value in (low, high) if abs(value) >= minimum
            ]
            if not fixed_candidates:
                raise ValueError(
                    f"range {value_range} cannot satisfy minimum magnitude {minimum}"
                )
            return torch.full(
                (count,), fixed_candidates[0], dtype=torch.float, device=device
            )

        union_coordinate = torch.rand(
            count, device=device, generator=generator
        ) * total_length
        negative = union_coordinate < negative_length
        values = positive_low + (union_coordinate - negative_length)
        values[negative] = low + union_coordinate[negative]
        return values

    def record(self, pure_error_sums, pure_counts, active_error_sums, active_counts):
        """Consume subset aggregates and independently evaluate each axis."""
        self.last_expanded = [False, False, False]
        for axis, stats in enumerate(self.statistics):
            stats.pure_error_sum += float(pure_error_sums[axis])
            stats.pure_sample_count += int(pure_counts[axis])
            stats.active_error_sum += float(active_error_sums[axis])
            stats.active_sample_count += int(active_counts[axis])
            if stats.active_sample_count:
                stats.active_mae = stats.active_error_sum / stats.active_sample_count
            if stats.pure_sample_count < self.minimum_samples:
                continue
            stats.pure_mae = stats.pure_error_sum / stats.pure_sample_count
            if stats.pure_mae_ema != stats.pure_mae_ema:
                stats.pure_mae_ema = stats.pure_mae
            else:
                stats.pure_mae_ema = self.ema_alpha * stats.pure_mae + (
                    1.0 - self.ema_alpha
                ) * stats.pure_mae_ema
            if stats.pure_mae_ema < self.success_thresholds[axis]:
                stats.consecutive_successes += 1
            else:
                stats.consecutive_successes = 0
            stats.pure_error_sum = 0.0
            stats.pure_sample_count = 0
            stats.active_error_sum = 0.0
            stats.active_sample_count = 0
            if (
                self.enabled
                and stats.consecutive_successes >= self.required_successes
                and self.expand(axis)
            ):
                stats.consecutive_successes = 0

    def expand(self, axis):
        current = self.current_ranges[axis]
        limit = self.limits[axis]
        step = self.steps[axis]
        expanded = [
            max(limit[0], current[0] - step),
            min(limit[1], current[1] + step),
        ]
        changed = expanded != current
        if changed:
            self.current_ranges[axis] = expanded
            self.statistics[axis].expansion_count += 1
            self.last_expanded[axis] = True
        return changed

    def state_dict(self):
        return {
            "current_ranges": [list(value) for value in self.current_ranges],
            "statistics": [vars(stats).copy() for stats in self.statistics],
        }

    def load_state_dict(self, state):
        if not self.enabled:
            return
        ranges = state.get("current_ranges")
        if ranges is not None:
            for axis, value in enumerate(ranges):
                value = _pair(value, f"runtime_{AXIS_NAMES[axis]}_range")
                limit = self.limits[axis]
                if value[0] < limit[0] or value[1] > limit[1]:
                    raise ValueError(f"restored {AXIS_NAMES[axis]} range exceeds limits")
                self.current_ranges[axis] = value
        for stats, restored in zip(self.statistics, state.get("statistics", ())):
            for name in vars(stats):
                if name in restored:
                    setattr(stats, name, restored[name])
