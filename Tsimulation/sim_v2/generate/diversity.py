"""Diversity machinery: perturbation, novelty filtering, coverage.

WHY THIS EXISTS. The first generator produced 854 umi demos whose intrinsic
dimensionality was FIVE: a PCA over object paths needed 5 components for 95%
of the variance, and the top 3 already covered 88%. That is not a coincidence
-- every demo was a rigid SE(2) retarget of one of ~15 seeds, and SE(2) has 3
degrees of freedom plus goal placement. The manifold is bounded by the
TRANSFORM GROUP, so no amount of extra sampling adds information. Scaling that
dataset cannot improve a policy, which is exactly what was observed.

Three things break that ceiling, in increasing order of effect:

  1. RANDOMISE THE SEEDS. A deterministic planner emits one trajectory shape.
     Sampling grasp target, standoff, approach side and speed makes the seed
     set itself diverse, which multiplies through every retarget.
  2. PERTURB THE RETARGET. Waypoint noise and time warping move a demo OFF the
     rigid-transform manifold, so a retarget is no longer a pure isometry of
     its source.
  3. FILTER ON NOVELTY. Accepting every success is what fills a dataset with
     near-duplicates. Requiring each new demo to be far enough from everything
     already accepted turns generation from "collect N successes" into "cover
     the space", and is the only one of the three that gives a guarantee.
"""

from __future__ import annotations

import math

import numpy as np


def trajectory_signature(object_xy: np.ndarray, n_points: int = 24) -> np.ndarray:
    """Shape descriptor of an object path, invariant to where it happened.

    Translation is removed so two identical manoeuvres in different corners of
    the arena count as the SAME behaviour -- which is the redundancy we are
    trying to measure. Absolute placement is already covered by layout
    sampling and should not be allowed to masquerade as behavioural variety.
    """
    obj = np.asarray(object_xy, dtype=np.float64)
    if len(obj) < 2:
        return np.zeros(n_points * 2)
    idx = np.linspace(0, len(obj) - 1, n_points).astype(int)
    return (obj[idx] - obj[0]).ravel()


class NoveltyFilter:
    """Accept a candidate only if it is far from everything already kept.

    Uses raw nearest-neighbour distance in signature space rather than a
    clustering method: it needs no fitted model, stays valid as the set grows,
    and the threshold is in world units so it can be reasoned about directly.
    """

    def __init__(self, min_distance: float = 60.0, n_points: int = 24):
        self.min_distance = float(min_distance)
        self.n_points = int(n_points)
        self._sigs: list[np.ndarray] = []
        self.rejected = 0

    def distance_to_set(self, sig: np.ndarray) -> float:
        if not self._sigs:
            return math.inf
        M = np.asarray(self._sigs)
        return float(np.linalg.norm(M - sig[None, :], axis=1).min())

    def offer(self, object_xy: np.ndarray) -> bool:
        sig = trajectory_signature(object_xy, self.n_points)
        if self.distance_to_set(sig) < self.min_distance:
            self.rejected += 1
            return False
        self._sigs.append(sig)
        return True

    def __len__(self) -> int:
        return len(self._sigs)

    def intrinsic_dim(self, var: float = 0.95) -> int:
        """PCs needed to explain `var` of the variance -- the redundancy metric."""
        if len(self._sigs) < 3:
            return len(self._sigs)
        A = np.asarray(self._sigs)
        s = np.linalg.svd(A - A.mean(0), compute_uv=False)
        ev = np.cumsum((s ** 2) / (s ** 2).sum())
        return int(np.searchsorted(ev, var) + 1)


def perturb_actions(actions: np.ndarray, rng, *, pos_sigma: float = 4.0,
                    angle_sigma: float = 0.06, warp: float = 0.18,
                    smooth: int = 7) -> np.ndarray:
    """Move a retargeted trajectory OFF the rigid-transform manifold.

    Noise is SMOOTHED before it is added: per-step white noise is rejected by
    the controller and just makes the command jitter without changing the
    path, so it costs success rate and buys no diversity. A low-frequency
    offset instead bends the whole approach, which is a genuinely different
    trajectory.

    Time warping resamples the trajectory at a non-uniform rate, so the same
    path is executed with a different speed profile -- free variation for a
    controller whose behaviour depends on rate.
    """
    a = np.asarray(actions, dtype=np.float64).copy()
    n = len(a)
    if n < 4:
        return a

    # --- smoothed positional offset
    noise = rng.normal(0.0, 1.0, size=(n, 2))
    k = np.ones(smooth) / smooth
    for c in range(2):
        noise[:, c] = np.convolve(noise[:, c], k, mode="same")
    # taper to zero at both ends so the start pose and final placement survive
    taper = np.sin(np.linspace(0, math.pi, n))[:, None]
    a[:, :2] += noise * pos_sigma * taper

    if a.shape[1] >= 3:
        an = np.convolve(rng.normal(0.0, 1.0, n), k, mode="same")
        a[:, 2] += an * angle_sigma * taper[:, 0]

    # --- time warp: resample on a monotone but uneven grid
    if warp > 0.0:
        t = np.linspace(0.0, 1.0, n)
        bend = rng.uniform(-warp, warp)
        tw = np.clip(t + bend * np.sin(math.pi * t), 0.0, 1.0)
        tw = np.maximum.accumulate(tw)          # keep it monotone
        src = tw * (n - 1)
        lo = np.floor(src).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        f = (src - lo)[:, None]
        a = a[lo] * (1 - f) + a[hi] * f
    return a
