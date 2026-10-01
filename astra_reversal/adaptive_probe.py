"""Adaptive reference numerics, separate from the fixed-step runtime sampler."""

import time

import numpy as np

from .flow import FlowResult


def integrate_reference(
    velocity,
    initial,
    *,
    start,
    end,
    rtol,
    atol,
    method="RK45",
    max_evaluations=20000,
    save_trace=False,
):
    """Accumulate in float64; the supplied velocity retains its own precision.

    This is a numerical diagnostic, not an alias for the native Euler sampler.
    ``velocity`` receives and returns NumPy arrays with full [B, H, D] shape.
    """
    from scipy.integrate import solve_ivp

    initial = np.asarray(initial)
    if (
        initial.ndim != 3
        or not all(initial.shape)
        or not np.issubdtype(initial.dtype, np.floating)
        or not np.isfinite(initial).all()
    ):
        raise ValueError("Expected finite floating [B, H, D] flow state")
    if not (0 <= start <= 1 and 0 <= end <= 1) or start == end:
        raise ValueError("Distinct endpoints must lie in [0, 1]")
    if any(not np.isfinite(x) or x <= 0 for x in (rtol, atol)):
        raise ValueError("Adaptive tolerances must be finite and positive")
    if method not in {"RK45", "DOP853"}:
        raise ValueError("Unsupported reference solver")
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise ValueError("A positive evaluation limit is required")
    shape, evaluations = initial.shape, 0

    def evaluate(t, flat):
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise RuntimeError("Adaptive velocity evaluation limit reached")
        result = np.asarray(velocity(flat.reshape(shape), float(t)))
        evaluations += 1
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError("Velocity must preserve finite full flow shape")
        return result.astype(np.float64).ravel()

    began = time.perf_counter()
    solution = solve_ivp(
        evaluate,
        (start, end),
        initial.astype(np.float64).ravel(),
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f"Adaptive solve failed: {solution.message}")
    return FlowResult(
        solution.y[:, -1].reshape(shape),
        solution.t.tolist(),
        evaluations,
        time.perf_counter() - began,
        [column.reshape(shape).copy() for column in solution.y.T]
        if save_trace
        else None,
    )
