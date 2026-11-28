"""Offline-only adapter: enumerate unclipped analytic branches, reject infeasible IK.

Never install in shared code or hardware. Invalid solves may be held by the
upstream converter; EVENTS must therefore be propagated to release rejection.
"""
import inspect
import numpy as np
from scipy.spatial.transform import Rotation

EVENTS = []


def equivalent_in_limits(q, lower, upper, reference):
    """Choose equivalent 2*pi angles; never project a pose onto a joint limit."""
    q = np.asarray(q, float)
    if not np.isfinite(q).all():
        return None
    low_k = np.ceil((lower - q) / (2 * np.pi))
    high_k = np.floor((upper - q) / (2 * np.pi))
    if np.any(low_k > high_k):
        return None
    k = np.minimum(np.maximum(np.round((reference - q) / (2 * np.pi)), low_k), high_k)
    return q + 2 * np.pi * k


def install(frame_getter=lambda: None):
    from projects.rby1_teleop import geometric_kinematics_rby1 as geo
    original = geo.SEW_IK_SP_2_1_1_Euler_3
    source = inspect.getsource(original)
    selection = "i = np.argmin(np.abs(q1_diff))"
    assert source.count(selection) == 1, "Upstream changed: review adapter before use"
    branches = []
    for index in (0, 1):
        namespace = dict(original.__globals__)
        exec(compile(source.replace(selection, f"i = {index}"), "<unclipped-arm-branch>", "exec"), namespace)
        branches.append(namespace[original.__name__])
    signature = inspect.signature(original)

    def solve(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = dict(bound.arguments)
        if not p["limited"]:
            return original(*args, **kwargs)
        model = p["model_transforms"]
        # Model transforms also include the fixed end-link after seven arm joints.
        lower = np.asarray(model["joint_lower"], float)[:7]
        upper = np.asarray(model["joint_upper"], float)[:7]
        assert lower.shape == upper.shape == (7,)
        reference = np.zeros(7) if p["q_prev"] is None else np.asarray(p["q_prev"], float)
        p["limited"] = False
        candidates = []
        for branch in branches:
            try:
                raw, ls, vectors = branch(**p)
            except IndexError:  # sp_2 can have only one solution.
                continue
            # Analytic subproblems can report LS for nominal robot geometry.
            # Preserve that flag; final FK residual, not this flag alone, gates data.
            variants = [np.asarray(raw, float)]
            if p["R_0_7"] is not None:
                alternative = variants[0].copy()
                alternative[4:] = [raw[4] + np.pi, np.pi - raw[5], raw[6] + np.pi]
                np.testing.assert_allclose(Rotation.from_euler("ZYX", raw[4:]).as_matrix(),
                                           Rotation.from_euler("ZYX", alternative[4:]).as_matrix(), atol=1e-10)
                variants.append(alternative)
            for raw_q in variants:
                q = equivalent_in_limits(raw_q, lower, upper, reference)
                if q is not None:
                    candidates.append((float(np.sum((q - reference) ** 2)), q, ls, vectors))
        event = {"frame": frame_getter(), "joints": list(model.get("joint_names", [])),
                 "feasible_candidates": len(candidates), "valid": bool(candidates)}
        EVENTS.append(event)
        if not candidates:
            raise ValueError("NO_LIMIT_FEASIBLE_ARM_BRANCH: offline frame must remain invalid")
        _, q, ls, vectors = min(candidates, key=lambda item: item[0])
        assert np.all(q >= lower) and np.all(q <= upper)
        return q, ls, vectors

    geo.SEW_IK_SP_2_1_1_Euler_3 = solve
    return original
