"""Local, bounded differential IK for rollout recovery; never opens hardware.

The pinned Quest streaming solver is deliberately independent of this module.
Each call solves one box-constrained, posture-regularized least-squares step,
then checks the nonlinear FK error and total command displacement.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


def validate_ik_recovery(config=None):
    defaults = {
        "enabled": False,
        "max_joint_velocity": 1.0,
        "position_scale_m": 0.05,
        "rotation_scale_rad": 0.5,
        "damping": 0.01,
        "posture_weight": 0.05,
        "max_ticks_per_target": 30,
    }
    if config is None:
        config = {}
    if not isinstance(config, Mapping) or set(config) - defaults.keys():
        raise ValueError("ik_recovery must contain only the documented options")
    result = {**defaults, **config}
    if type(result["enabled"]) is not bool:
        raise ValueError("ik_recovery.enabled must be a boolean")
    ticks = result["max_ticks_per_target"]
    if type(ticks) is not int or not 1 <= ticks <= 300:
        raise ValueError("ik_recovery.max_ticks_per_target must be in [1, 300]")
    for name in defaults.keys() - {"enabled", "max_ticks_per_target"}:
        value = result[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"ik_recovery.{name} must be finite and positive")
        result[name] = float(value)
    return result


@dataclass(frozen=True)
class RecoveryStep:
    joints: np.ndarray
    reached: bool
    position_error_m: float
    rotation_error_rad: float


def differential_ik_step(
    kinematics, target, measured, last_command, max_step, settings
):
    """Find one improving local step inside measured AND command rate bounds.

    The six-joint box also contains the model joint limits. The posture
    objective prefers the previous command, with 3x regularization on YAM's
    parallel-axis chain. No random seeds or distant IK branches are used.
    """
    target = np.asarray(target, dtype=float)
    measured = np.asarray(measured, dtype=float)
    last_command = np.asarray(last_command, dtype=float)
    if (
        target.shape != (4, 4)
        or not np.isfinite(target).all()
        or measured.shape != (6,)
        or not np.isfinite(measured).all()
        or last_command.shape != (6,)
        or not np.isfinite(last_command).all()
        or not np.isfinite(max_step)
        or max_step <= 0
    ):
        raise ValueError(
            "Differential IK requires finite poses, joints and a positive step bound"
        )
    options = validate_ik_recovery(settings)
    lo = np.maximum(measured - max_step, last_command - max_step)
    hi = np.minimum(measured + max_step, last_command + max_step)
    for i, joint in enumerate(kinematics.joints):
        if kinematics.model.jnt_limited[joint]:
            qmin, qmax = kinematics.model.jnt_range[joint]
            lo[i], hi[i] = max(lo[i], qmin), min(hi[i], qmax)
    if np.any(lo > hi):
        raise ValueError("Tracking lag leaves no joint step inside both rate bounds")

    def residual(q):
        pose = kinematics.fk(q)
        return np.r_[
            target[:3, 3] - pose[:3, 3],
            Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec(),
        ]

    def result(q, error):
        pe, re = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        return RecoveryStep(
            q.copy(),
            pe <= kinematics.position_tolerance and re <= kinematics.rotation_tolerance,
            float(pe),
            float(re),
        )

    error = residual(measured)
    if (
        result(measured, error).reached
        and np.all(measured >= lo)
        and np.all(measured <= hi)
    ):
        return result(measured, error)
    jp = np.zeros((3, kinematics.model.nv))
    jr = np.zeros_like(jp)
    kinematics.mj.mj_jacSite(kinematics.model, kinematics.data, jp, jr, kinematics.site)
    scale = np.r_[
        np.full(3, 1 / options["position_scale_m"]),
        np.full(3, 1 / options["rotation_scale_rad"]),
    ]
    jac = scale[:, None] * np.vstack([jp[:, kinematics.didx], jr[:, kinematics.didx]])
    weighted_error = scale * error
    # Damping is in scaled task coordinates; increase it near rank deficiency.
    sigma_min = np.linalg.svd(jac, compute_uv=False)[-1]
    damping = options["damping"] * (1 + max(0.0, 1 - sigma_min / 0.1) * 9)
    posture = np.full(6, options["posture_weight"])
    posture[1:4] *= 3
    matrix = np.vstack([jac, np.diag(np.sqrt(posture)), damping * np.eye(6)])
    rhs = np.r_[
        weighted_error, np.sqrt(posture) * (last_command - measured), np.zeros(6)
    ]
    lower, upper = lo - measured, hi - measured
    # scipy requires strictly separated bounds; exactly fixed joints are
    # eliminated from the optimization rather than widened past a limit.
    free = upper - lower > 1e-12
    delta = lower.copy()
    try:
        if np.any(free):
            solution = lsq_linear(
                matrix[:, free],
                rhs - matrix[:, ~free] @ delta[~free],
                bounds=(lower[free], upper[free]),
                method="bvls",
                max_iter=20,
                tol=1e-8,
            )
            if not solution.success:
                raise ValueError("Differential IK exceeded its linear solver budget")
            delta[free] = solution.x
    except np.linalg.LinAlgError as exc:
        raise ValueError("Differential IK numerical failure") from exc
    if not np.isfinite(delta).all():
        raise ValueError("Differential IK produced a nonfinite step")
    before = np.dot(weighted_error, weighted_error)
    for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        candidate = measured + fraction * delta
        if np.any(candidate < lo - 1e-12) or np.any(candidate > hi + 1e-12):
            continue
        after_error = residual(candidate)
        after = np.dot(scale * after_error, scale * after_error)
        if after < before - max(1e-12, before * 1e-8):
            return result(candidate, after_error)
    raise ValueError("Differential IK cannot make bounded Cartesian progress")
