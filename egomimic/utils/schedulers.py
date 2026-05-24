"""Pure-Python scalar schedulers, used by Lightning callbacks that anneal
something over training steps (residual scale, KL beta, dropout rate, etc.)."""

from __future__ import annotations


def piecewise_linear(step: int, schedule):
    """Piecewise-linear interpolation. ``schedule`` is a list of
    ``(step, value)`` knots (sorted by step). Returns a float.

    - For ``step <= schedule[0][0]``: returns ``schedule[0][1]``.
    - For ``step >= schedule[-1][0]``: returns ``schedule[-1][1]``.
    - Otherwise: linear interpolation between the two surrounding knots.
    """
    schedule = list(schedule)
    if not schedule:
        raise ValueError("schedule must be non-empty")
    if step <= schedule[0][0]:
        return float(schedule[0][1])
    if step >= schedule[-1][0]:
        return float(schedule[-1][1])
    for (s0, v0), (s1, v1) in zip(schedule[:-1], schedule[1:]):
        if s0 <= step <= s1:
            if s1 == s0:
                return float(v1)
            t = (step - s0) / (s1 - s0)
            return float(v0 + t * (v1 - v0))
    return float(schedule[-1][1])
