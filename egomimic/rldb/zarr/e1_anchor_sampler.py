"""Progress-weighted anchor sampling for the E1 fold rows (#3 of the
tempo-invariance ablation — Ideas note *Arc Tokenizer Tempo Invariance Changes*).

The standard chunk sampler anchors a training sample at every frame with equal
probability, so a slow demonstration contributes more tokens per metre of path
than a fast one (≈ 2.5× across the fold tertiles): geometry supervision is
skewed slow at equal episode count. Here every frame is weighted by the
protocol's mixed start measure

    w_t = alpha + (1 - alpha) * v_t / v_pool

with ``v_t`` the wrist speed at the anchor (max over arms, 3 Hz zero-phase
Butterworth on the episode's ``obs_wrist_pose``, the Step 0 signal) and
``v_pool`` its mean over the training pool — so the second term samples
uniformly in progress ``ds`` while the first keeps holds at ``alpha`` of their
uniform mass. ``alpha = 0`` is pure progress sampling; ``alpha = 1`` is the
uniform sampler. Built once per training run from the leaf episodes'
``obs_wrist_pose`` (a few seconds for 220 episodes); the sampler draws with
replacement, which at ≈ 2 % of the pool per epoch is indistinguishable from
the shuffle it replaces.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import zarr
from torch.utils.data import WeightedRandomSampler

from egomimic.rldb.zarr.e1_arc_tokenizer import lowpass_positions

logger = logging.getLogger(__name__)
HANDS = ("left", "right")


def episode_frame_speed(path: str, fc_hz: float = 3.0, fs_hz: float = 30.0, pose_key: str = "wrist") -> np.ndarray:
    """Per-frame wrist speed (m/s), max over arms, low-passed at ``fc_hz``."""
    g = zarr.open_group(str(path), mode="r")
    speeds = []
    for h in HANDS:
        p = np.asarray(g[f"{h}.obs_{pose_key}_pose"][:, :3], dtype=np.float64)
        p = lowpass_positions(p, fc_hz, fs_hz)
        if len(p) < 2:
            speeds.append(np.zeros(len(p)))
            continue
        step = np.linalg.norm(np.diff(p, axis=0), axis=1) * fs_hz
        speeds.append(np.concatenate([step, step[-1:]]))
    return np.maximum(speeds[0], speeds[1])


def anchor_weights(dataset, alpha: float, fc_hz: float, fs_hz: float, pose_key: str) -> np.ndarray:
    """One weight per ``dataset.index_map`` entry (a MultiDataset over leaf episodes)."""
    per_ep: dict[str, np.ndarray] = {}
    for name, leaf in dataset.datasets.items():
        if not hasattr(leaf, "episode_path"):
            raise TypeError(f"anchor sampler needs leaf episode datasets, got {type(leaf).__name__} for {name}")
        v = episode_frame_speed(leaf.episode_path, fc_hz, fs_hz, pose_key)
        n = len(leaf)
        if len(v) < n:  # keep the two in step if the episode metadata disagrees with the array
            v = np.concatenate([v, np.full(n - len(v), float(v[-1]) if len(v) else 0.0)])
        per_ep[name] = v[:n]
    v_pool = float(np.mean(np.concatenate(list(per_ep.values()))))
    v_pool = max(v_pool, 1e-6)
    w = np.empty(len(dataset.index_map), dtype=np.float64)
    for gidx, (name, local) in enumerate(dataset.index_map):
        w[gidx] = alpha + (1.0 - alpha) * per_ep[name][local] / v_pool
    ess = float(w.sum() ** 2 / np.sum(w**2))
    logger.info(
        "anchor sampler: %d episodes, %d frames, v_pool %.3f m/s, alpha %.2f, weight q10/q50/q90 = %.2f/%.2f/%.2f, ESS %.0f (%.0f%%)",
        len(per_ep), len(w), v_pool, alpha, *np.quantile(w, [0.1, 0.5, 0.9]), ess, 100 * ess / len(w),
    )
    print(f"ANCHOR_SAMPLER episodes={len(per_ep)} frames={len(w)} v_pool={v_pool:.3f} alpha={alpha} "
          f"w_q10={np.quantile(w, .1):.2f} w_q50={np.quantile(w, .5):.2f} w_q90={np.quantile(w, .9):.2f} ess_frac={ess / len(w):.2f}", flush=True)
    return w


def build_anchor_sampler(dataset, alpha: float = 0.2, fc_hz: float = 3.0, fs_hz: float = 30.0,
                         pose_key: str = "wrist", num_samples: int | None = None, seed: int = 0) -> WeightedRandomSampler:
    w = anchor_weights(dataset, float(alpha), float(fc_hz), float(fs_hz), pose_key)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=int(num_samples or len(w)),
                                 replacement=True, generator=gen)
