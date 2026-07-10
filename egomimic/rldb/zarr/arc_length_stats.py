"""Data-driven statistics for the arc-length tokenizer's trans_vel channels.

The bimanual arc-length tokenizer synthesizes per-arm translational-velocity
channels online from EEF action chunks, so they are not present in any
precomputed dataset statistics. This module computes their normalization
stats by tokenizing a sample of raw (T, 14) bimanual cartesian chunks and
reducing the velocity columns into the standard
``{min, max, mean, std, q01, q99}`` dict (JSON-serializable lists).

Ported from the GR00T arc-length stats module
(gr00t branch ``rpunamiya/arc-length-tokenizer``,
``groot/core/data/state_action/arc_length_stats.py``), adapted to consume
canonical bimanual chunks instead of per-arm EEF/gripper iterables and to
reuse the production tokenize path rather than re-deriving the velocity
computation. Chunks containing the >=1e8 invalid-pose sentinel are skipped
so they cannot poison the statistics.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARM_DIM,
    INVALID_POSE_THRESHOLD,
    BimanualArcLengthConfig,
    BimanualArcLengthTokenizer,
)

# Stats-dict key names for the synthesized velocity channels (one per arm).
TRANS_VEL_LEFT_KEY = "trans_vel_left"
TRANS_VEL_RIGHT_KEY = "trans_vel_right"


def collect_trans_vel_samples(
    chunks: Iterable[np.ndarray],
    config: BimanualArcLengthConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize (T, 14) chunks and return the stacked per-arm velocity samples.

    args:
        chunks: iterable of (T, 14) bimanual cartesian chunks. Chunks with any
            value >= the invalid-pose sentinel threshold are skipped.
        config: arc-length tokenizer config (sets M, mode, distance unit, dt).
    returns:
        (left (N*M, velocity_dim), right (N*M, velocity_dim)) — concatenated
        velocity waypoints across all tokenized chunks.
    """
    tokenizer = BimanualArcLengthTokenizer(config)
    vd = tokenizer.velocity_dim
    arm_dim = tokenizer.arc_arm_dim
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float64)
        if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
            continue
        arc = tokenizer.tokenize(chunk)
        lefts.append(arc[:, ARM_DIM : ARM_DIM + vd])
        rights.append(arc[:, arm_dim + ARM_DIM : arm_dim + ARM_DIM + vd])
    if not lefts:
        empty = np.zeros((0, vd), dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(lefts, axis=0), np.concatenate(rights, axis=0)


def trans_vel_stats_from_samples(samples: np.ndarray) -> dict[str, list[float]]:
    """Reduce (N, D) velocity samples to the standard stats dict.

    Keys match what the dataset stats pipeline produces for normal action
    groups: {min, max, mean, std, q01, q99}. Values are JSON-serializable
    lists of length D.

    raises: ValueError if samples is empty — the caller must ensure at least
    one chunk was tokenized.
    """
    if samples.size == 0:
        raise ValueError(
            "trans_vel_stats_from_samples got 0 samples — caller must pass at "
            "least one tokenized chunk."
        )
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def compute_trans_vel_stats(
    chunks: Iterable[np.ndarray],
    config: BimanualArcLengthConfig,
) -> dict[str, dict[str, list[float]]]:
    """End-to-end: bimanual chunks -> tokenized velocities -> stats dicts.

    returns: ``{trans_vel_left: {min, max, ...}, trans_vel_right: {...}}``
    suitable for direct injection into a merged-stats action dict.
    """
    left, right = collect_trans_vel_samples(chunks, config)
    return {
        TRANS_VEL_LEFT_KEY: trans_vel_stats_from_samples(left),
        TRANS_VEL_RIGHT_KEY: trans_vel_stats_from_samples(right),
    }
