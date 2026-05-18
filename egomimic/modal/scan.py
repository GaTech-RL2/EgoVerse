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
