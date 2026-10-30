"""Per-episode norm-stat samples.

Each episode's sampled proprio/action rows are cached on their own, keyed by
what a sample is (resolver keymap + transforms + reader code) but not by which
split the episode is in. A split's stats are computed from the union of its
episodes' rows, so a new split (20 % -> full, an operator hold-out, newly
collected data) only reads the episodes it has not seen before.

Frames are taken on a power-of-two stride with a fixed per-episode phase, so
the rows at stride 2s are a subset of the rows at stride s: one cached file
serves every coarser stride, and a split only resamples an episode when it needs
it denser than it was ever cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from egomimic.rldb.multi import worker_context
from egomimic.rldb.norm import norm_cache
from egomimic.utils.gpu_orphans import die_with_parent
from egomimic.utils.type_utils import _to_numpy

logger = logging.getLogger(__name__)

# Bump when the frame selection or the stored layout changes.
VERSION = 1


def recipe_key(dataset_name: str, dataset_cfg) -> str:
    """Identity of a sample: the resolver's keymap and transforms, the dataset
    class it loads and the reader code. Filters, mode and split ratios only pick
    episodes, so they are left out."""
    raw = norm_cache._to_dict(dataset_cfg)
    resolver = {"resolver": raw.get("resolver") or {}}
    inputs = {
        "v": VERSION,
        "dataset": dataset_name,
        "resolver": norm_cache.recipe_inputs(resolver)["resolver"],
        "code": norm_cache.code_hash(resolver),
    }
    blob = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_root(cache_dir, dataset_name: str, dataset_cfg) -> Path:
    return (
        Path(cache_dir)
        / "episodes"
        / dataset_name
        / recipe_key(dataset_name, dataset_cfg)[:16]
    )


def stride_for(total_frames: int, n_target: int) -> int:
    """Largest power of two that still yields at least ``n_target`` frames."""
    s = 1
    while total_frames // (2 * s) >= max(n_target, 1):
        s *= 2
    return s


def _phase(episode_hash: str) -> int:
    return int(hashlib.sha256(episode_hash.encode("utf-8")).hexdigest()[:15], 16)


def frame_indices(episode_hash: str, n_frames: int, stride: int) -> np.ndarray:
    return np.arange(_phase(episode_hash) % stride, n_frames, stride, dtype=np.int64)


def _file(root: Path, episode_hash: str, fingerprint: str, stride: int) -> Path:
    return root / f"{episode_hash}.{fingerprint[:16]}.s{stride}.npz"


def find(root: Path, episode_hash: str, fingerprint: str, stride: int) -> Path | None:
    """The coarsest cached file whose rows include every frame at ``stride``."""
    best = None
    for p in root.glob(f"{episode_hash}.{fingerprint[:16]}.s*.npz"):
        try:
            s = int(p.name.rsplit(".s", 1)[1][: -len(".npz")])
        except ValueError:
            continue
        if stride % s == 0 and (best is None or s > best[0]):
            best = (s, p)
    return None if best is None else best[1]


def _write(path: Path, idx: list[int], rows: dict[str, list[np.ndarray]]) -> bool:
    keys = sorted(rows)
    arrays = {"idx": np.asarray(idx, dtype=np.int64), "keys": np.asarray(keys)}
    for i, k in enumerate(keys):
        arrays[f"k{i}"] = np.stack(rows[k])
    # Several jobs (and every DDP rank) can share the cache, so each writer
    # gets its own tmp file.
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("episode norm cache write failed (%s): %s", path, e)
        tmp.unlink(missing_ok=True)
        return False
    return True


# An episode's rows: a cache file, or (idx, {key: rows}) held in memory when it
# must not be cached (see _sample_episodes).
_Source = Path | tuple[np.ndarray, dict[str, np.ndarray]]


def _in_memory(idx: list[int], rows: dict[str, list[np.ndarray]]) -> _Source:
    return np.asarray(idx, dtype=np.int64), {k: np.stack(v) for k, v in rows.items()}


def _indices(src: _Source) -> np.ndarray:
    if isinstance(src, tuple):
        return src[0]
    with np.load(src) as z:
        return z["idx"]


def _read(src: _Source, rows: np.ndarray) -> dict[str, np.ndarray]:
    if isinstance(src, tuple):
        return {k: v[rows] for k, v in src[1].items()}
    with np.load(src) as z:
        return {str(k): z[f"k{i}"][rows] for i, k in enumerate(z["keys"])}


class _Frames(torch.utils.data.Dataset):
    def __init__(self, leaves, tasks, zarr_keys: Mapping[str, str]):
        self.leaves = leaves
        self.tasks = tasks
        self.zarr_keys = dict(zarr_keys)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, i):
        h, idx = self.tasks[i]
        try:
            data = self.leaves[h][idx]
        except Exception as e:
            # ZarrDataset already retried inside the episode; this one is bad.
            logger.warning(f"norm sample failed ({type(e).__name__}: {e}) {h}[{idx}]")
            return h, idx, None
        return (
            h,
            idx,
            {k: _to_numpy(data[z]) for k, z in self.zarr_keys.items() if z in data},
        )


def _as_list(batch):
    return batch


def _sample_episodes(
    root: Path,
    leaves: Mapping,
    fingerprints: Mapping[str, str | None],
    zarr_keys: Mapping[str, str],
    stride: int,
    num_workers: int,
) -> dict[str, _Source]:
    """Sample ``leaves`` at ``stride``; returns where each episode's rows are.

    An episode is cached only when every one of its samples loaded and it has a
    fingerprint. A failed sample may be transient (S3, a worker OOM-retry), and
    a file keyed by recipe and fingerprint would keep the partial rows for every
    later run -- and DDP ranks that disagree on them would compute different
    stats. Such an episode's rows are used from memory and sampled again next
    time."""
    sources: dict[str, _Source] = {}
    failed: set[str] = set()

    def _finish(h, idx, rows):
        fp = fingerprints.get(h)
        path = None if fp is None else _file(root, h, fp, stride)
        if h not in failed and path is not None and _write(path, idx, rows):
            sources[h] = path
        else:
            sources[h] = _in_memory(idx, rows)

    tasks = [
        (h, int(i))
        for h in sorted(leaves)
        for i in frame_indices(h, len(leaves[h]), stride)
    ]
    pending = {h: 0 for h in leaves}
    for h, _ in tasks:
        pending[h] += 1
    # An episode with no frames at this stride still gets an (empty) file.
    for h in [h for h, n in pending.items() if n == 0]:
        _finish(h, [], {})
        del pending[h]
    if not tasks:
        return sources

    loader = torch.utils.data.DataLoader(
        _Frames(leaves, tasks, zarr_keys),
        batch_size=64,
        num_workers=num_workers,
        collate_fn=_as_list,
        worker_init_fn=die_with_parent if num_workers > 0 else None,
        multiprocessing_context=(
            worker_context(leaves.values()) if num_workers > 0 else None
        ),
    )
    idx_buf: dict[str, list[int]] = {}
    row_buf: dict[str, dict[str, list[np.ndarray]]] = {}
    with tqdm(total=len(tasks), unit="sample", desc="norm samples") as pbar:
        for batch in loader:
            for h, idx, sample in batch:
                if sample is not None:
                    idx_buf.setdefault(h, []).append(idx)
                    for k, v in sample.items():
                        row_buf.setdefault(h, {}).setdefault(k, []).append(v)
                else:
                    failed.add(h)
                pending[h] -= 1
                if pending[h] == 0:
                    _finish(h, idx_buf.pop(h, []), row_buf.pop(h, {}))
            pbar.update(len(batch))
    return sources


def collect(
    root: Path,
    leaves: Mapping,
    zarr_keys: Mapping[str, str],
    n_target: int,
    *,
    seed: int = 42,
    num_workers: int = 4,
    fingerprints: Mapping[str, str | None] | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict]:
    """``n_target`` rows per norm key (fewer if the episodes hold fewer frames),
    drawn uniformly from ``leaves`` (episode hash -> ZarrDataset). Episodes
    missing from ``root`` at the needed density are sampled and cached first.

    ``fingerprints``: ``norm_cache.leaf_fingerprint`` per episode hash, when
    the caller already has them (trainHydra's split key); the rest are computed.
    An episode without one is never cached."""
    root = Path(root)
    fingerprints = dict(fingerprints or {})
    for h, ds in leaves.items():
        if h not in fingerprints:
            fingerprints[h] = norm_cache.leaf_fingerprint(ds)
    lengths = {h: len(ds) for h, ds in leaves.items()}
    stride = stride_for(sum(lengths.values()), n_target)

    files: dict[str, _Source | None] = {
        h: None if fingerprints[h] is None else find(root, h, fingerprints[h], stride)
        for h in leaves
    }
    missing = {h: leaves[h] for h, p in files.items() if p is None}
    logger.info(
        f"[norm samples] stride {stride}: {len(leaves) - len(missing)}/{len(leaves)} "
        f"episodes cached under {root}, sampling {len(missing)}"
    )
    if missing:
        files.update(
            _sample_episodes(
                root, missing, fingerprints, zarr_keys, stride, num_workers
            )
        )

    order = sorted(leaves)
    at_stride = {}
    for h in order:
        phase = _phase(h) % stride
        at_stride[h] = np.flatnonzero((_indices(files[h]) - phase) % stride == 0)
    total = sum(len(v) for v in at_stride.values())
    if total > n_target:
        keep = np.zeros(total, bool)
        keep[np.random.default_rng(seed).choice(total, n_target, replace=False)] = True
        offset = 0
        for h in order:
            n = len(at_stride[h])
            at_stride[h] = at_stride[h][keep[offset : offset + n]]
            offset += n

    collected: dict[str, list[np.ndarray]] = {k: [] for k in zarr_keys}
    for h in order:
        if not len(at_stride[h]):
            continue
        for k, v in _read(files[h], at_stride[h]).items():
            if k in collected:
                collected[k].append(v)
    meta = {
        "stride": stride,
        "episodes": len(leaves),
        "episodes_sampled": len(missing),
        "frames": int(min(total, n_target)),
    }
    return collected, meta
