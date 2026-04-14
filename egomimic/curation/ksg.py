"""
Kraskov-Stögbauer-Grassberger (KSG) mutual information estimator.

Implements Algorithm 1 from:
    Kraskov, Stögbauer, Grassberger (2004).
    "Estimating mutual information."
    Physical Review E, 69(6).

"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

logger = logging.getLogger(__name__)

# Memory budget for batched queries (bytes). Stay well under typical RAM.
_MEMORY_BUDGET_BYTES = 4 * 1024**3  # 4 GB
_BATCH_THRESHOLD = 10_000  # Points above this → batch queries


# ---------------------------------------------------------------------------
# Core KSG estimator
# ---------------------------------------------------------------------------


def ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    noise_scale: float = 1e-10,
) -> np.ndarray:
    """
    Estimate per-sample mutual information contributions using KSG Algorithm 1.

    For each point i:
        1. Find distance to the k-th nearest neighbour in JOINT (x, y) space
           using the Chebyshev (L-inf) metric.  Call this ε_i.
        2. Count n_x(i) = #{j ≠ i : ‖x_i − x_j‖_∞ < ε_i}
        3. Count n_y(i) = #{j ≠ i : ‖y_i − y_j‖_∞ < ε_i}
        4. mi_i = ψ(k) − ψ(n_x+1) − ψ(n_y+1) + ψ(N)

    where ψ is the digamma function.

    Args:
        x: (N, dx) array — state embeddings.
        y: (N, dy) array — action embeddings.
        k: Number of nearest neighbours (must be ≥ 1).
        noise_scale: Tiny jitter to break exact ties (avoids ε=0 queries).

    Returns:
        mi: (N,) float64 array of per-sample MI contributions.
            The mean approximates I(X; Y) in nats.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    N = len(x)
    if N <= k:
        raise ValueError(f"k={k} must be < N={N}")

    # Add tiny noise to break exact ties (common in quantised / normalised data)
    rng = np.random.default_rng(seed=0)
    x = x + rng.standard_normal(x.shape) * noise_scale
    y = y + rng.standard_normal(y.shape) * noise_scale

    # Joint space (Chebyshev distance)
    z = np.hstack([x, y])
    tree_z = cKDTree(z)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # Find k-th NN distance in joint space (index k because index 0 = self)
    # workers=-1 uses all available CPU threads
    dists, _ = tree_z.query(z, k=k + 1, p=np.inf, workers=-1)
    eps = dists[:, k]  # (N,) — distance to k-th NN, excluding self

    # Count neighbours in marginal spaces strictly within ε
    # Using eps * (1 - tiny) ensures strict inequality despite floating point.
    eps_strict = eps * (1.0 - 1e-10)

    if N <= _BATCH_THRESHOLD:
        n_x = _count_neighbours(tree_x, x, eps_strict)
        n_y = _count_neighbours(tree_y, y, eps_strict)
    else:
        n_x = _count_neighbours_batched(tree_x, x, eps_strict)
        n_y = _count_neighbours_batched(tree_y, y, eps_strict)

    # KSG formula: ψ(k) - ψ(n_x+1) - ψ(n_y+1) + ψ(N)
    mi = digamma(k) - digamma(n_x + 1) - digamma(n_y + 1) + digamma(N)
    return mi.astype(np.float64)


def _count_neighbours(
    tree: cKDTree,
    points: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """
    For each point i, count neighbours within radii[i] (excluding self).

    Uses return_length=True (SciPy ≥ 1.9) to return integer counts directly,
    avoiding materialisation of the full per-point neighbour-index lists which
    causes OOM for large N.
    """
    # return_length=True returns an int array of counts (includes self)
    counts = tree.query_ball_point(points, radii, p=np.inf, return_length=True)
    counts = np.asarray(counts, dtype=np.float64) - 1.0  # subtract self
    return np.maximum(counts, 0.0)


def _count_neighbours_batched(
    tree: cKDTree,
    points: np.ndarray,
    radii: np.ndarray,
    batch_size: int = 10_000,
) -> np.ndarray:
    """Batch version of _count_neighbours for very large N."""
    N = len(points)
    counts = np.empty(N, dtype=np.float64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_counts = tree.query_ball_point(
            points[start:end], radii[start:end], p=np.inf, return_length=True
        )
        counts[start:end] = np.maximum(
            np.asarray(batch_counts, dtype=np.float64) - 1.0, 0.0
        )
    return counts


# ---------------------------------------------------------------------------
# Averaged KSG (more robust to choice of k)
# ---------------------------------------------------------------------------


def ksg_mi_averaged(
    x: np.ndarray,
    y: np.ndarray,
    k_range: tuple[int, int] = (3, 7),
    noise_scale: float = 1e-10,
) -> np.ndarray:
    """
    Average per-sample MI estimates over a range of k values.

    DemInf found results generally robust to k, but averaging over a small
    range of k values reduces variance without significant compute overhead.

    Args:
        x: (N, dx) state embeddings.
        y: (N, dy) action embeddings.
        k_range: (k_min, k_max) inclusive range of k values to average.
        noise_scale: Jitter for tie-breaking.

    Returns:
        mi: (N,) float64 per-sample MI contributions, averaged over k values.
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))
    if not k_values:
        raise ValueError(f"Invalid k_range: {k_range}")

    estimates = np.stack(
        [ksg_mi(x, y, k=k, noise_scale=noise_scale) for k in k_values],
        axis=0,
    )  # (num_k, N)

    return estimates.mean(axis=0)  # (N,)
