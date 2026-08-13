from .helpers import get_args
from .task_registry import TaskRegistry, task_registry

_MATH_EXPORTS = {
    "get_scale_shift",
    "plus_2pi_wrap_to_pi",
    "quat_apply_yaw",
    "torch_rand_sqrt_float",
    "wrap_to_pi",
}


def __getattr__(name):
    """Keep historical utility exports lazy so non-simulation tools stay light."""
    if name == "Terrain":
        from .terrain import Terrain

        return Terrain
    if name in _MATH_EXPORTS:
        from . import math_utils

        return getattr(math_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TaskRegistry",
    "Terrain",
    "get_args",
    "task_registry",
    *_MATH_EXPORTS,
]
