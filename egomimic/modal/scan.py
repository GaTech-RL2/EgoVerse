"""Standalone Modal app for parallel zarr metadata scanning.

Deploy once:
    modal deploy --env robotics egomimic/modal/scan.py

Then `egomimic.rldb.zarr.zarr_dataset_multi._modal_fanout_scan` will look up
`egomimic-scan::scan_shard` and fan out filter scans across many small
CPU-only containers, each mounting the zarr volume read-only.

Kept separate from `run.py` so the training app and the scan utility can be
deployed independently — and so the scan workers use a minimal image (no
PyTorch / ML stack) for fast cold starts.

Self-contained: no egomimic imports. The function mirrors
`_normalize_filter_row` + `DatasetFilter.matches` from
egomimic/rldb/filters.py — keep in sync if those change.
"""

from __future__ import annotations

import os

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

VOLUME_MOUNT_PATH = "/mnt/zarr-data"
ZARR_VOLUME_NAME = "mecka_data_v2"

image = modal.Image.debian_slim(python_version="3.11").pip_install("zarr==3.1.5")
zarr_volume = modal.Volume.from_name(ZARR_VOLUME_NAME)
app = modal.App("egomimic-scan", image=image)


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=900,  # 15 min per shard
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    max_containers=100,  # cap parallel workers — too many saturates parent's control-plane connection
)
def scan_shard(
    episode_names: list[str],
    filter_lambdas: list[str],
) -> list[tuple[str, str]]:
    """Scan a shard of zarr episode dirs and return matched (path, episode_hash) pairs."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    zarr_volume.reload()

    predicates = [eval(expr) for expr in filter_lambdas]
    base = Path(VOLUME_MOUNT_PATH)

    def _read_metadata(name: str):
        p = base / name
        zattrs = p / ".zattrs"
        try:
            if zattrs.is_file():
                with zattrs.open("rb") as f:
                    return json.load(f)
            import zarr

            store = zarr.open_group(str(p), mode="r")
            return dict(store.attrs)
        except Exception:
            return None

    def _matches(metadata: dict, episode_hash: str) -> bool:
        row = dict(metadata)
        row["episode_hash"] = episode_hash
        v = row.get("is_deleted")
        if v is None or v == "":
            row["is_deleted"] = False
        if row.get("is_deleted"):
            return False
        for pred in predicates:
            if not pred(row):
                return False
        return True

    def _process(name: str):
        episode_hash = name[:-5] if name.endswith(".zarr") else name
        metadata = _read_metadata(name)
        if metadata is None:
            return None
        if _matches(metadata, episode_hash):
            return (str(base / name), episode_hash)
        return None

    matched: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for result in executor.map(_process, episode_names):
            if result is not None:
                matched.append(result)

    return matched


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=900,
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    max_containers=100,
)
def load_shard(
    episode_names: list[str],
) -> list[tuple[str, str, dict]]:
    """Read .zattrs for each name and return (full_path, episode_hash, metadata).

    The caller can then construct ZarrDataset objects locally without re-opening
    every zarr group serially — the slow step on Modal volume metadata.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    zarr_volume.reload()
    base = Path(VOLUME_MOUNT_PATH)

    def _read_metadata(name: str):
        p = base / name
        zattrs = p / ".zattrs"
        try:
            if zattrs.is_file():
                with zattrs.open("rb") as f:
                    return json.load(f)
            import zarr

            store = zarr.open_group(str(p), mode="r")
            return dict(store.attrs)
        except Exception:
            return None

    def _process(name: str):
        episode_hash = name[:-5] if name.endswith(".zarr") else name
        metadata = _read_metadata(name)
        if metadata is None:
            return None
        return (str(base / name), episode_hash, metadata)

    results: list[tuple[str, str, dict]] = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for r in executor.map(_process, episode_names):
            if r is not None:
                results.append(r)

    return results


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=900,
    volumes={VOLUME_MOUNT_PATH: zarr_volume},
    max_containers=100,
)
def pause_precompute_shard(
    episodes: list[tuple[str, str]],
    epsilon: float,
) -> list[tuple[str, int, list[int]]]:
    """Compute pause keep_indices for a shard of episodes.

    Args:
        episodes: list of (episode_hash, episode_path_str).
        epsilon: L2 threshold for the "frame is paused" test (m, per hand).

    Returns:
        list of (episode_hash, raw_total_frames, keep_indices_list). For
        episodes missing the obs_ee_pose keys or that fail to open, raw is 0
        and indices is empty — the caller treats this as "skip".
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import zarr

    LEFT_KEY = "left.obs_ee_pose"
    RIGHT_KEY = "right.obs_ee_pose"

    zarr_volume.reload()

    def _mask(left_pose: "np.ndarray", right_pose: "np.ndarray") -> "np.ndarray":
        T = len(left_pose)
        if T < 2:
            return np.ones(T, dtype=bool)
        left_d = np.linalg.norm(np.diff(left_pose, axis=0), axis=-1)
        right_d = np.linalg.norm(np.diff(right_pose, axis=0), axis=-1)
        is_paused = (left_d < epsilon) & (right_d < epsilon)
        keep = np.ones(T, dtype=bool)
        in_pause = False
        for t in range(1, T):
            if is_paused[t - 1]:
                if in_pause:
                    keep[t] = False
                else:
                    in_pause = True
            else:
                in_pause = False
        return keep

    def _one(item: tuple[str, str]) -> tuple[str, int, list[int]]:
        episode_hash, path_str = item
        try:
            store = zarr.open_group(path_str, mode="r")
            left = np.asarray(store[LEFT_KEY][:])
            right = np.asarray(store[RIGHT_KEY][:])
        except Exception:
            return (episode_hash, 0, [])
        try:
            keep = _mask(left, right)
            indices = np.flatnonzero(keep).astype(np.int64).tolist()
            return (episode_hash, int(left.shape[0]), indices)
        except Exception:
            return (episode_hash, 0, [])

    out: list[tuple[str, int, list[int]]] = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for r in executor.map(_one, episodes):
            out.append(r)
    return out
