"""Small helpers for copying, serializing, and comparing nested configs."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


class ConfigNode(SimpleNamespace):
    """Attribute-based config node with an explicit, isolated object graph."""

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()


def _public_items(value):
    if isinstance(value, type):
        namespace = {
            key: getattr(value, key)
            for key in vars(value)
            if not key.startswith("_")
        }
    elif hasattr(value, "__dict__"):
        namespace = vars(value)
        if not namespace:
            namespace = vars(type(value))
    else:
        return None
    return {
        key: item
        for key, item in namespace.items()
        if not key.startswith("_")
        and (not callable(item) or isinstance(item, type))
    }


def clone_config(value):
    """Clone class-style ParamsProto configs without sharing nested classes."""
    items = _public_items(value)
    if items is not None:
        return ConfigNode(**{key: clone_config(item) for key, item in items.items()})
    if isinstance(value, dict):
        return {key: clone_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_config(item) for item in value)

    return copy.deepcopy(value)


def config_to_dict(value):
    """Convert a nested class/object config into JSON-compatible containers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)

    items = _public_items(value)
    if items is not None:
        return {
            key: config_to_dict(item)
            for key, item in sorted(items.items())
        }
    if isinstance(value, dict):
        return {
            str(key): config_to_dict(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [config_to_dict(item) for item in value]

    if hasattr(value, "tolist"):
        return config_to_dict(value.tolist())
    return str(value)


def config_fingerprint(value):
    """Return a stable SHA-256 digest for a nested config."""
    payload = json.dumps(
        config_to_dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_config(target, values):
    """Recursively apply config values to objects and nested dictionaries."""
    source = config_to_dict(values)
    for key, item in source.items():
        if isinstance(target, dict):
            current = target.get(key)
        else:
            current = getattr(target, key, None)

        if isinstance(item, dict) and current is not None:
            apply_config(current, item)
        elif isinstance(target, dict):
            target[key] = copy.deepcopy(item)
        else:
            setattr(target, key, copy.deepcopy(item))
    return target


def normalize_saved_env_config(default_config, saved_config):
    """Normalize historical environment configs before applying them.

    Some runs created during the first refactor saved ``cfg`` after environment
    initialization.  At that point the legacy environment had removed zero
    reward scales and multiplied every active scale by the control timestep.
    Loading that config into a fresh environment applied the timestep a second
    time.  Detect that distinctive all-at-once scaling and restore the raw
    coefficients without changing valid historical configs.

    Returns a detached config dictionary and a small, JSON-compatible report.
    """
    normalized = copy.deepcopy(config_to_dict(saved_config))
    defaults = config_to_dict(default_config)
    report = {
        "legacy_reward_scale_repaired": False,
        "control_dt": None,
        "matching_scaled_coefficients": 0,
        "compared_coefficients": 0,
    }

    saved_scales = normalized.get("reward_scales")
    default_scales = defaults.get("reward_scales")
    try:
        control_dt = (
            float(defaults["control"]["decimation"])
            * float(defaults["sim"]["dt"])
        )
    except (KeyError, TypeError, ValueError):
        return normalized, report

    report["control_dt"] = control_dt
    if not isinstance(saved_scales, dict) or not isinstance(default_scales, dict):
        return normalized, report
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        return normalized, report

    compared = 0
    scaled_matches = 0
    raw_matches = 0
    for name, default_value in default_scales.items():
        saved_value = saved_scales.get(name)
        if (
            isinstance(default_value, bool)
            or isinstance(saved_value, bool)
            or not isinstance(default_value, (int, float))
            or not isinstance(saved_value, (int, float))
            or default_value == 0
        ):
            continue
        ratio = float(saved_value) / float(default_value)
        if not math.isfinite(ratio):
            continue
        compared += 1
        if math.isclose(ratio, control_dt, rel_tol=1e-6, abs_tol=1e-12):
            scaled_matches += 1
        if math.isclose(ratio, 1.0, rel_tol=1e-6, abs_tol=1e-12):
            raw_matches += 1

    report["matching_scaled_coefficients"] = scaled_matches
    report["compared_coefficients"] = compared
    # Require several independent coefficients and a strong majority.  The
    # raw-match guard keeps ordinary saved configs from being "repaired".
    looks_runtime_scaled = (
        compared >= 3
        and scaled_matches / compared >= 0.8
        and scaled_matches > raw_matches
    )
    if not looks_runtime_scaled:
        return normalized, report

    for name, value in saved_scales.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            restored_value = value / control_dt
            default_value = default_scales.get(name)
            if (
                isinstance(default_value, (int, float))
                and not isinstance(default_value, bool)
                and math.isclose(
                    restored_value,
                    float(default_value),
                    rel_tol=1e-6,
                    abs_tol=1e-12,
                )
            ):
                restored_value = default_value
            saved_scales[name] = restored_value
    report["legacy_reward_scale_repaired"] = True
    return normalized, report
