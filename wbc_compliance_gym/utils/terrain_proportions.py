"""Shared terrain-probability definitions and validation."""

import math


HEIGHTFIELD_TERRAIN_TYPES = (
    "smooth_pyramid_slope",
    "rough_pyramid_slope",
    "negative_height_stairs",
    "positive_height_stairs",
    "discrete_obstacles",
    "stepping_stones",
    "reserved_flat_1",
    "reserved_flat_2",
    "uniform_roughness",
    "half_flat_half_rough",
)

BOX_TERRAIN_TYPES = (
    "flat",
    "downstairs",
    "upstairs",
    "slope",
    "pit",
)


def cumulative_terrain_proportions(values, terrain_types, config_name):
    """Validate a terrain probability vector and return its cumulative form."""
    probabilities = [float(value) for value in values]
    expected_count = len(terrain_types)
    if len(probabilities) != expected_count:
        raise ValueError(
            f"{config_name} must contain {expected_count} probabilities in this "
            f"order: {', '.join(terrain_types)}; got {len(probabilities)}"
        )
    if any(value < 0.0 for value in probabilities):
        raise ValueError(f"{config_name} probabilities must be non-negative")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{config_name} probabilities must sum to 1.0; got {total}")

    cumulative = []
    running_total = 0.0
    for probability in probabilities:
        running_total += probability
        cumulative.append(running_total)
    cumulative[-1] = 1.0
    return cumulative
