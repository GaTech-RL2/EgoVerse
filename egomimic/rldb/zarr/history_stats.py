"""History norm stats: per-lag stats for proprio observation history.

A history step ``lag`` frames old is expressed in the CURRENT step's head
frame, so its wrist translation spreads wider than the current step's (the
past wrist carries all the head motion since). Current-step stats then clip or
reject old steps. The table holds stats per lag ``0..max_lag`` for the
channels that depend on it (the wrist / ee translation); every other channel
keeps the current-step stats. ``gather`` picks the rows for a given
``(K, stride)`` and returns ordinary ``(K, D)`` per-cell stats, so normalize,
the bounds check and checkpoints need nothing new.

Keypoints are excluded: each history step's keypoints are in that step's own
wrist frame, so their distribution does not depend on the lag. Rotation
channels are excluded because rot6d is left unnormalized.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset

from egomimic.utils.pose_utils import (
    bimanual_cartesian_layout,
    bimanual_keypoint_layout,
)

logger = logging.getLogger(__name__)

HISTORY_STATS_FILE = "history_stats.json"
HISTORY_MASK_KEY = "proprio_history_mask"


def lag_channels(zarr_key: str, width: int) -> list[int] | None:
    """Channels of a proprio key whose distribution depends on the lag: the
    head-frame wrist translation. ``None`` for keys without one."""
    if zarr_key == "observations.state.keypoints":
        layout = bimanual_keypoint_layout(width)
        return None if layout is None else sorted(layout["wrist_xyz"])
    if zarr_key == "observations.state.ee_pose":
        layout = bimanual_cartesian_layout(width)
        return None if layout is None else sorted(layout["xyz"])
    return None


def history_keymap(key_map: dict, max_lag: int) -> dict:
    """The norm-mode keymap config reading a ``max_lag + 1`` step history at
    stride 1, so one sample carries every lag."""
    km = dict(key_map)
    km["proprio_history"] = int(max_lag) + 1
    km["history_stride"] = 1
    km["history_stride_s"] = None
    return km


class _LagSlice(Dataset):
    """Worker-side slice to the lag channels and the history mask, so a
    ``(max_lag + 1, D)`` window does not cross the worker pipe whole."""

    def __init__(self, dataset, zarr_keys: dict[str, str], channels: dict[str, list]):
        self.dataset = dataset
        self.zarr_keys = zarr_keys
        self.channels = channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        data = self.dataset[i]
        out = {HISTORY_MASK_KEY: np.asarray(data[HISTORY_MASK_KEY], dtype=np.float32)}
        for name, zarr_key in self.zarr_keys.items():
            v = np.asarray(data[zarr_key], dtype=np.float32)
            out[name] = v[..., self.channels[name]]
        return out


def compute(
    norm_stats,
    lag_dataset,
    embodiment_id: int,
    max_lag: int,
    n_anchors: int,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 8,
) -> dict:
    """Per-lag stats for every history proprio key of ``lag_dataset``.

    ``lag_dataset`` is a norm-mode MultiDataset built from ``history_keymap``;
    ``norm_stats`` supplies the key names. Each lag only counts anchors whose
    step at that lag lies inside the episode (history mask 1), so the
    front-padding copies never shape the stats.
    """
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    probe = lag_dataset[0]
    if HISTORY_MASK_KEY not in probe:
        raise ValueError(
            "lag dataset has no proprio history; build it with history_keymap"
        )
    L = int(max_lag) + 1
    zarr_keys, channels = {}, {}
    for name in norm_stats.keys_of_type("proprio_keys", embodiment_id):
        zk = norm_stats.keyname_to_zarr_key(name, embodiment_id)
        v = probe.get(zk)
        if v is None or np.ndim(v) != 2 or np.shape(v)[0] != L:
            continue
        ch = lag_channels(zk, np.shape(v)[-1])
        if ch:
            zarr_keys[name], channels[name] = zk, ch
    if not zarr_keys:
        raise ValueError("no history proprio key with lag-dependent channels")

    n = min(int(n_anchors), len(lag_dataset))
    idx = np.random.default_rng(seed).choice(len(lag_dataset), size=n, replace=False)
    loader = DataLoader(
        Subset(_LagSlice(lag_dataset, zarr_keys, channels), idx.tolist()),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    vals = {name: [] for name in zarr_keys}
    masks = []
    for batch in loader:
        masks.append(batch[HISTORY_MASK_KEY].numpy())
        for name in zarr_keys:
            vals[name].append(batch[name].numpy())
    mask = np.concatenate(masks) > 0  # (N, L), oldest first

    out = {}
    for name in zarr_keys:
        X = np.concatenate(vals[name])  # (N, L, C)
        per_lag = {}
        count = np.zeros(L, dtype=np.int64)
        for lag in range(L):
            j = L - 1 - lag
            rows = X[mask[:, j], j]
            rows = rows[np.isfinite(rows).all(axis=1)]
            count[lag] = len(rows)
            if len(rows) == 0:
                raise ValueError(
                    f"{name}: no anchor has a real step at lag {lag}; lower max_lag"
                )
            for stat, arr in MultiDataset._compute_stats_for_array(rows).items():
                per_lag.setdefault(stat, []).append(arr)
        out[name] = {
            "zarr_key": zarr_keys[name],
            "channels": channels[name],
            "count": count,
            "stats": {s: np.stack(a).astype(np.float32) for s, a in per_lag.items()},
        }
        logger.info(
            f"[history_stats] {name}: {n} anchors, lag {max_lag} has {count[-1]} real steps"
        )
    return out


def save(path: str, embodiment_id: int, table: dict, max_lag: int) -> str:
    """Write/merge ``table`` for ``embodiment_id`` into ``path`` (a file or a
    norm_stats dir)."""
    if os.path.isdir(path) or not path.endswith(".json"):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, HISTORY_STATS_FILE)
    payload = {"max_lag": int(max_lag), "stats": {}}
    if os.path.isfile(path):
        with open(path) as f:
            payload = json.load(f)
        if payload["max_lag"] != int(max_lag):
            raise ValueError(f"{path} has max_lag={payload['max_lag']}, not {max_lag}")
    payload["stats"][str(embodiment_id)] = {
        name: {
            "zarr_key": e["zarr_key"],
            "channels": list(map(int, e["channels"])),
            "count": np.asarray(e["count"]).tolist(),
            "stats": {s: np.asarray(a).tolist() for s, a in e["stats"].items()},
        }
        for name, e in table.items()
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def load(path: str, embodiment_id: int) -> tuple[dict, int]:
    """``(table, max_lag)`` for ``embodiment_id`` from a file or norm_stats dir."""
    if os.path.isdir(path):
        path = os.path.join(path, HISTORY_STATS_FILE)
    with open(path) as f:
        payload = json.load(f)
    raw = payload["stats"].get(str(embodiment_id))
    if raw is None:
        raise ValueError(f"{path} has no lag stats for embodiment id {embodiment_id}")
    table = {
        name: {
            "zarr_key": e["zarr_key"],
            "channels": e["channels"],
            "count": np.asarray(e["count"]),
            "stats": {
                s: np.asarray(a, dtype=np.float32) for s, a in e["stats"].items()
            },
        }
        for name, e in raw.items()
    }
    return table, int(payload["max_lag"])


def gather(base: dict, entry: dict, history: int, stride: int) -> dict:
    """``(K, D)`` stats for a K-step history ``stride`` frames apart (oldest
    first): the current-step ``(D,)`` stats tiled, with the lag channels of
    step k taken from lag ``stride * (K - 1 - k)``."""
    lags = stride * (history - 1 - np.arange(history))
    max_lag = entry["stats"]["mean"].shape[0] - 1
    if lags[0] > max_lag:
        raise ValueError(
            f"history {history} x stride {stride} reaches lag {lags[0]}, past the "
            f"table's max_lag {max_lag}"
        )
    ch = np.asarray(entry["channels"])
    out = {}
    for stat, arr in base.items():
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"expected current-step (D,) stats, got {arr.shape}")
        tiled = np.tile(arr, (history, 1))
        if stat in entry["stats"]:
            tiled[:, ch] = entry["stats"][stat][lags]
        out[stat] = tiled
    return out


def apply(norm_stats, embodiment_id: int, path: str, history: int, stride: int) -> list:
    """Replace the current-step stats of each history key in ``norm_stats``
    with its gathered ``(K, D)`` stats. Keys already at ``(K, D)`` (stats loaded
    from a run that gathered them) are left as they are. Returns the keys
    changed."""
    table, _ = load(path, embodiment_id)
    per_emb = norm_stats.norm_stats.get(embodiment_id, {})
    changed = []
    for name, entry in table.items():
        base = per_emb.get(name)
        if base is None:
            continue
        if np.ndim(base["mean"]) == 2:
            if np.shape(base["mean"])[0] != history:
                raise ValueError(
                    f"{name}: stats are {np.shape(base['mean'])}, not a "
                    f"{history}-step history"
                )
            continue
        per_emb[name] = gather(base, entry, history, stride)
        changed.append(name)
    return changed


def history_stride_frames(key_map: dict, dataset) -> int:
    """The history stride in frames: ``history_stride`` or
    ``history_stride_s`` x fps, which must then be the same for every
    episode (the stats have one row per lag in frames)."""
    stride_s = key_map.get("history_stride_s")
    if stride_s is None:
        return int(key_map.get("history_stride", 1) or 1)
    strides = set()
    stack = [dataset]
    while stack:
        ds = stack.pop()
        if hasattr(ds, "datasets"):
            stack.extend(ds.datasets.values())
        elif hasattr(ds, "_seconds_to_frames"):
            strides.add(ds._seconds_to_frames(stride_s))
    if len(strides) != 1:
        raise ValueError(
            f"history_stride_s={stride_s} is {sorted(strides)} frames across "
            "episodes; per-lag stats need one stride in frames (set history_stride)"
        )
    return strides.pop()
