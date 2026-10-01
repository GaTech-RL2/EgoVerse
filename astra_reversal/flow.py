"""Directional ODE solves: action at t=0, noise at t=1."""

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .records import to_numpy


@dataclass
class FlowResult:
    value: Any
    grid: list[float]
    velocity_evaluations: int
    latency_seconds: float
    states: list[np.ndarray] | None


def integrate(
    velocity: Callable,
    initial,
    *,
    start: float,
    end: float,
    steps: int,
    solver: str = "euler",
    save_trace: bool = False,
    time_power: float = 1.0,
) -> FlowResult:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not (0 <= start <= 1 and 0 <= end <= 1) or start == end:
        raise ValueError("Distinct integration endpoints must lie in [0, 1]")
    if solver not in ("euler", "heun", "rk4"):
        raise ValueError(f"Unsupported solver: {solver}")
    if not np.isfinite(time_power) or time_power <= 0:
        raise ValueError("time_power must be finite and positive")
    array = to_numpy(initial)
    if (
        array.ndim != 3
        or not all(array.shape)
        or not np.issubdtype(array.dtype, np.floating)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Flow state must be finite floating [B, H, D_model]")
    # Multiplication copies numpy, torch, and JAX arrays without device transfers.
    x = initial * 1.0
    dt = (end - start) / steps
    grid = [start + (end - start) * j / steps for j in range(steps + 1)]
    if time_power != 1.0:
        left, right = start ** (1 / time_power), end ** (1 / time_power)
        grid = [
            (left + (right - left) * j / steps) ** time_power for j in range(steps + 1)
        ]
        grid[0], grid[-1] = start, end
    states = [array.copy()] if save_trace else None
    began = time.perf_counter()
    evaluations = 0
    for index in range(steps):
        if time_power != 1.0:
            dt = grid[index + 1] - grid[index]
        v = velocity(x, grid[index])
        if v.shape != x.shape:
            raise ValueError("Velocity changed the full internal tensor shape")
        evaluations += 1
        candidate = x + dt * v
        if solver == "heun":
            next_v = velocity(candidate, grid[index + 1])
            if next_v.shape != x.shape:
                raise ValueError("Velocity changed the full internal tensor shape")
            candidate = x + (dt / 2) * (v + next_v)
            evaluations += 1
        elif solver == "rk4":
            middle = grid[index] + dt / 2
            k2 = velocity(x + (dt / 2) * v, middle)
            k3 = velocity(x + (dt / 2) * k2, middle)
            k4 = velocity(x + dt * k3, grid[index + 1])
            if any(k.shape != x.shape for k in (k2, k3, k4)):
                raise ValueError("Velocity changed the full internal tensor shape")
            candidate = x + (dt / 6) * (v + 2 * k2 + 2 * k3 + k4)
            evaluations += 3
        x = candidate
        if states is not None:
            states.append(to_numpy(x).copy())
    # Also synchronizes GPU/JAX work before reporting elapsed time.
    if not np.isfinite(to_numpy(x)).all():
        raise FloatingPointError("Non-finite flow endpoint")
    return FlowResult(x, grid, evaluations, time.perf_counter() - began, states)


def invert(velocity, actions, **kwargs):
    return integrate(velocity, actions, start=0.0, end=1.0, **kwargs)


def generate(velocity, noise, **kwargs):
    return integrate(velocity, noise, start=1.0, end=0.0, **kwargs)


def mix_noise(recovered, independent, rho: float):
    if not 0 <= rho <= 1 or recovered.shape != independent.shape:
        raise ValueError("Noise mixing needs equal shapes and rho in [0, 1]")
    return math.sqrt(1 - rho * rho) * recovered + rho * independent


def error_metrics(reference, actual) -> dict:
    reference, actual = to_numpy(reference), to_numpy(actual)
    if (
        reference.shape != actual.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Reconstruction arrays must have matching finite shapes")
    delta = actual.astype(np.float64) - reference
    return {
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def noise_statistics(noise):
    x = to_numpy(noise).astype(np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "l2": float(np.linalg.norm(x)),
        "max_abs": float(np.abs(x).max()),
    }
