"""Validate official state setters without changing simulator state.

MuJoCo forward normalizes free/ball-joint quaternions. Accept that exact native
normalization only for already-unit quaternions; all other state stays exact.
"""

import numpy as np


def validate_setter(initial, actual, model, *, normalize=None):
    initial = np.asarray(initial, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    expected_size = 1 + model.nq + model.nv + model.na
    if (
        initial.shape != (expected_size,)
        or actual.shape != initial.shape
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(actual))
    ):
        raise ValueError("official_state_setter_mismatch")
    if np.array_equal(actual, initial):
        return {"match": "exact", "normalized_joints": [], "max_abs_difference": 0.0}
    if normalize is None:
        from mujoco import mju_normalize4

        normalize = mju_normalize4
    expected = initial.copy()
    changed = []
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind not in (0, 1):  # MuJoCo free and ball joints.
            continue
        start = 1 + int(model.jnt_qposadr[joint]) + (3 if kind == 0 else 0)
        quaternion = expected[start : start + 4]
        if np.array_equal(actual[start : start + 4], quaternion):
            continue
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 8 * np.finfo(float).eps:
            raise ValueError("official_state_setter_mismatch")
        normalize(quaternion)
        changed.append(int(joint))
    # No allclose across the full state and no tolerance on positions, velocities,
    # time or actuator state. Even quaternion output must match MuJoCo bitwise.
    if not changed or not np.array_equal(actual, expected):
        raise ValueError("official_state_setter_mismatch")
    return {
        "match": "native_unit_quaternion_normalization",
        "normalized_joints": changed,
        "max_abs_difference": float(np.max(np.abs(actual - initial))),
    }
