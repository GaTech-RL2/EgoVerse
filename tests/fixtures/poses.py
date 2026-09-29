"""Pose-test helpers: random xyz+quat(wxyz) poses and chunks, their SE(3)
matrices, a transform-list runner, and a MultiDataset shell for _check_bounds."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rand_pose(rng, scale=1.0):
    q = R.random(random_state=int(rng.integers(1 << 31))).as_quat()  # xyzw
    return np.concatenate([rng.uniform(-scale, scale, 3), q[[3, 0, 1, 2]]])


def rand_chunk(rng, start, n):
    """Smooth random walk of ``n`` poses starting AT ``start``."""
    out = np.zeros((n, 7))
    p, r = start[:3].copy(), R.from_quat(start[[4, 5, 6, 3]])
    for t in range(n):
        if t:
            p = p + rng.normal(0, 0.01, 3)
            r = R.from_rotvec(rng.normal(0, 0.05, 3)) * r
        out[t] = np.concatenate([p, r.as_quat()[[3, 0, 1, 2]]])
    return out


def pose_matrix(p7):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(p7[[4, 5, 6, 3]]).as_matrix()
    M[:3, 3] = p7[:3]
    return M


def apply_transforms(transform_list, sample):
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    for t in transform_list:
        s = t.transform(s)
    return s


def bounds_shell(key, stats=None, *, width=None, lo=-1.0, hi=1.0, mode="quantile"):
    """MultiDataset with just enough state for ``_check_bounds`` /
    ``_apply_norm_one`` on ``key`` (embodiment 0). ``stats`` defaults to
    constant ``[lo, hi]`` quantile bounds of ``width`` channels."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    if stats is None:
        stats = {
            "quantile_1": np.full(width, lo, dtype=np.float32),
            "quantile_99": np.full(width, hi, dtype=np.float32),
        }
    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = mode
    md.norm_stats = {0: {key: stats}}
    md.zarr_keys = {0: {key: key}}
    md._warned_violations = set()
    return md
