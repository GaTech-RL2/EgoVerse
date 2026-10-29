"""Hardware-free validation of opt-in rollout IK settings."""

import math
from collections.abc import Mapping


def validate_ik_relaxation(config=None):
    defaults = {
        "enabled": False,
        "position_tolerance_m": 0.005,
        "rotation_tolerance_deg": 3.0,
    }
    if config is None:
        config = {}
    if not isinstance(config, Mapping) or set(config) - defaults.keys():
        raise ValueError("ik_relaxation must contain only the documented options")
    result = {**defaults, **config}
    if type(result["enabled"]) is not bool:
        raise ValueError("ik_relaxation.enabled must be a boolean")
    for name, upper in (
        ("position_tolerance_m", 0.02),
        ("rotation_tolerance_deg", 15.0),
    ):
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= upper
        ):
            raise ValueError(f"ik_relaxation.{name} must be finite and in (0, {upper}]")
        result[name] = float(value)
    return result


def validate_ik_solver(config=None):
    """Endpoint solver selection; independent of optional relaxation/recovery."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("ik_solver must be a mapping")
    settings = {"backend": "mujoco", "timeout_s": 0.005, "solver_type": "Distance"}
    supplied = dict(config or {})
    if supplied.keys() - settings.keys():
        raise ValueError("Unknown ik_solver option")
    settings.update(supplied)
    if settings["backend"] not in ("mujoco", "pytracik"):
        raise ValueError("ik_solver.backend must be mujoco or pytracik")
    if settings["solver_type"] not in ("Speed", "Distance", "Manip1", "Manip2"):
        raise ValueError("Unknown TRAC-IK solver_type")
    timeout = settings["timeout_s"]
    if (
        type(timeout) not in (float, int)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 0.02
    ):
        raise ValueError("ik_solver.timeout_s must be in (0, 0.02] seconds per arm")
    return settings
