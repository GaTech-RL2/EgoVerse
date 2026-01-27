"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observation.images.{cam}  (JPEG compressed)
        ├── observation.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations

import logging
import os
import random
import time
from functools import cached_property
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr

from lerobot.common.datasets.utils import (
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    get_delta_indices,
    check_delta_timestamps,
    load_json,
    load_jsonlines,
    flatten_dict,
    unflatten_dict,
)

from egomimic.rldb.data_utils import (
    _ypr_to_quat,
    _quat_to_ypr,
    _slow_down_slerp_quat,
)

logger = logging.getLogger(__name__)

SEED = 42


class ZarrProfiler:
    """Lightweight profiler for ZarrDataset hot paths."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: float, count: int = 1) -> None:
        self.totals[key] = self.totals.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + count

    def inc(self, key: str, count: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + count

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {"totals": dict(self.totals), "counts": dict(self.counts)}


def decode_jpeg(data) -> np.ndarray:
    """Decode JPEG image to numpy array using OpenCV.

    Args:
        data: Compressed JPEG bytes, may be wrapped in numpy array from zarr

    Returns:
        RGB image array of shape (H, W, C) with dtype uint8
    """
    # Extract bytes from numpy array wrapper (zarr VariableLengthBytes returns 0-dim array)
    while isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    # Convert bytes to numpy array for OpenCV
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)

    # Decode with OpenCV (returns BGR)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode JPEG image")

    # Convert BGR to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_info(local_dir: Path) -> dict:
    """Load info.json and convert shape lists to tuples.

    Supports both legacy global metadata and per-episode metadata structures.
    """
    # Try legacy global metadata first
    global_info_path = local_dir / INFO_PATH
    if global_info_path.exists():
        info = load_json(global_info_path)
        for ft in info.get("features", {}).values():
            if "shape" in ft:
                ft["shape"] = tuple(ft["shape"])
        return info

    # Fall back to per-episode metadata - aggregate from first episode
    return load_info_from_episodes(local_dir)


def load_info_from_episodes(local_dir: Path) -> dict:
    """Load and aggregate info from per-episode metadata."""
    import json

    # Find episode directories
    episode_dirs = sorted([
        p for p in local_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and p.name.endswith(".zarr")
    ])

    if not episode_dirs:
        return {"fps": 30, "features": {}, "total_episodes": 0, "total_frames": 0}

    # Load info from first episode for common fields
    base_info = {}
    first_ep_zarr_json = episode_dirs[0] / "zarr.json"
    first_ep_info_path = episode_dirs[0] / "meta" / "info.json"
    if first_ep_zarr_json.exists():
        with open(first_ep_zarr_json, "r") as f:
            base_info = json.load(f).get("attributes", {})
    elif first_ep_info_path.exists():
        with open(first_ep_info_path, "r") as f:
            base_info = json.load(f)

    # Aggregate totals across all episodes
    total_frames = 0
    tasks = set()

    for ep_dir in episode_dirs:
        ep_zarr_json = ep_dir / "zarr.json"
        ep_info_path = ep_dir / "meta" / "info.json"
        if ep_zarr_json.exists():
            with open(ep_zarr_json, "r") as f:
                ep_info = json.load(f).get("attributes", {})
        elif ep_info_path.exists():
            with open(ep_info_path, "r") as f:
                ep_info = json.load(f)
        else:
            ep_info = {}

        if ep_info:
            total_frames += ep_info.get("total_frames", 0)
            if ep_info.get("task"):
                tasks.add(ep_info["task"])

    # Build aggregated info
    info = {
        "fps": base_info.get("fps", 30),
        "robot_type": base_info.get("robot_type"),
        "features": base_info.get("features", {}),
        "total_episodes": len(episode_dirs),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
    }

    # Convert shape lists to tuples
    for ft in info.get("features", {}).values():
        if "shape" in ft:
            ft["shape"] = tuple(ft["shape"])

    return info


def load_stats(local_dir: Path) -> dict | None:
    """Load stats.json if it exists."""
    stats_path = local_dir / STATS_PATH
    if not stats_path.exists():
        return None
    stats = load_json(stats_path)
    stats = {key: torch.tensor(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_tasks(local_dir: Path) -> dict:
    """Load tasks.jsonl."""
    tasks_path = local_dir / TASKS_PATH
    if not tasks_path.exists():
        return {}
    tasks = load_jsonlines(tasks_path)
    return {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}


class ZarrDatasetMetadata:
    """
    Metadata handler for ZarrDataset.
    Mirrors LeRobotDatasetMetadata interface.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        local_files_only: bool = True,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else Path.cwd() / repo_id
        self.local_files_only = local_files_only

        # Load metadata
        self.info = load_info(self.root)
        self.stats = load_stats(self.root)
        self.tasks = load_tasks(self.root)

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info.get("robot_type")

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info.get("features", {})

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as compressed images."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "jpeg"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "jpeg"]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft.get("names") for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items() if "shape" in ft}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info.get("total_episodes", 0)

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info.get("total_frames", 0)

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info.get("total_tasks", len(self.tasks))

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info.get("total_chunks", 0)

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info.get("chunks_size", 1000)

    @property
    def task_to_task_index(self) -> dict:
        return {task: task_idx for task_idx, task in self.tasks.items()}

    def get_task_index(self, task: str) -> int:
        """Given a task in natural language, returns its task_index."""
        task_index = self.task_to_task_index.get(task, None)
        return task_index if task_index is not None else self.total_tasks

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode."""
        return ep_index // self.chunks_size

    def _update_splits(self, seed: int = SEED, valid_ratio: float = 0.2) -> None:
        """Updates self.info['splits'] with episode indices."""
        total_episodes = self.info.get("total_episodes", len(self.episodes))

        if total_episodes == 0:
            return

        all_indices = list(range(total_episodes))
        random.seed(seed)
        random.shuffle(all_indices)

        valid_size = max(1, min(int(valid_ratio * total_episodes), total_episodes - 1))

        self.info["splits"] = {
            "train": all_indices[valid_size:],
            "valid": all_indices[:valid_size],
        }

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "}})"
        )


class ZarrDatasetView:
    """
    Compatibility shim that makes Zarr data look like hf_dataset.
    Used for code that expects hf_dataset interface.
    """

    def __init__(self, zarr_dataset: "ZarrDataset"):
        self._zarr_dataset = zarr_dataset

    def __len__(self):
        return len(self._zarr_dataset)

    def __getitem__(self, idx):
        """Get item (for compatibility with hf_dataset interface)."""
        return self._zarr_dataset._get_zarr_item(idx)

    def select(self, indices: list[int]) -> list[dict]:
        """Return data for multiple indices."""
        return [self._zarr_dataset._get_zarr_item(i) for i in indices]


class ZarrDataset(torch.utils.data.Dataset):
    """
    Dataset class that reads from Zarr arrays.
    API-compatible with LeRobotDataset for drop-in replacement.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        local_files_only: bool = True,
        profile: bool | None = None,
        skip_keys: set[str] | list[str] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else Path.cwd() / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.local_files_only = local_files_only
        self.delta_indices = None
        self.skip_keys: set[str] = set(skip_keys) if skip_keys else set()
        if profile is None:
            profile = os.getenv("EGOMIMIC_ZARR_PROFILE", "0") == "1"
        self._profiler = ZarrProfiler() if profile else None

        # Load metadata
        self.meta = ZarrDatasetMetadata(repo_id, self.root, local_files_only)

        # Discover episodes
        self._episode_paths: dict[int, Path] = {}
        self._episode_stores: dict[int, zarr.Group] = {}
        self._discover_episodes()

        # Build episode data index (global frame -> episode/chunk/local mapping)
        self.episode_data_index = self._build_episode_data_index()

        # Build cached lookup structures for O(log n) episode lookup
        self._ep_list: list[int] = []
        self._ep_bounds: dict[int, tuple[int, int]] = {}
        self._ep_ends: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._build_lookup_cache()

        # Build frame index for episode filtering
        self._frame_indices: list[int] | None = None
        self._num_frames: int = 0
        self._build_frame_index()

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _discover_episodes(self) -> None:
        """Find all episode zarr stores.

        Supports both layouts:
        - New: episodes directly in root (episode_*.zarr/)
        - Legacy: episodes in data/ subdirectory (data/episode_*.zarr/)
        """
        # Try new layout first (episodes directly in root)
        search_dir = self.root

        # Check if using legacy layout (data/ subdirectory)
        data_dir = self.root / "data"
        if data_dir.exists() and any(
            p.name.startswith("episode_") and p.name.endswith(".zarr")
            for p in data_dir.iterdir() if p.is_dir()
        ):
            search_dir = data_dir

        if not search_dir.exists():
            logger.warning(f"Episode directory not found: {search_dir}")
            return

        for ep_path in sorted(search_dir.iterdir()):
            if not ep_path.is_dir():
                continue
            # Parse episode index from folder name: episode_{ep_idx}.zarr
            name = ep_path.name
            if not name.startswith("episode_") or not name.endswith(".zarr"):
                continue
            try:
                ep_idx = int(name.replace("episode_", "").replace(".zarr", ""))
            except ValueError:
                continue

            self._episode_paths[ep_idx] = ep_path

    def _get_episode_store(self, ep_idx: int) -> zarr.Group | None:
        """Open/cache episode Zarr store on demand."""
        if ep_idx in self._episode_stores:
            return self._episode_stores[ep_idx]

        ep_path = self._episode_paths.get(ep_idx)
        if ep_path is None:
            return None

        if not ep_path.exists():
            return None

        start = time.perf_counter() if self._profiler else None
        store = zarr.open_group(str(ep_path), mode="r")
        if self._profiler and start is not None:
            self._profiler.add("open_store_sec", time.perf_counter() - start)
            self._profiler.inc("open_store_count")
        self._episode_stores[ep_idx] = store
        return store

    def _build_episode_data_index(self) -> dict[str, torch.Tensor]:
        """Build from/to indices for each episode."""
        episode_lengths = {}

        for ep_idx in sorted(self._episode_paths.keys()):
            if self.episodes is not None and ep_idx not in self.episodes:
                continue

            # Calculate length from zarr store (use first available array)
            store = self._get_episode_store(ep_idx)
            if store is not None:
                keys = list(store.array_keys())
                if keys:
                    episode_lengths[ep_idx] = store[keys[0]].shape[0]
                else:
                    episode_lengths[ep_idx] = 0
            else:
                episode_lengths[ep_idx] = 0

        # Build cumulative indices
        from_indices = []
        to_indices = []
        cumulative = 0

        for ep_idx in sorted(episode_lengths.keys()):
            from_indices.append(cumulative)
            cumulative += episode_lengths[ep_idx]
            to_indices.append(cumulative)

        return {
            "from": torch.LongTensor(from_indices),
            "to": torch.LongTensor(to_indices),
        }

    def _build_lookup_cache(self) -> None:
        """Build cached structures for O(log n) episode lookup.

        Creates:
            _ep_list: Sorted list of episode indices (filtered if self.episodes is set)
            _ep_bounds: Dict mapping ep_idx -> (ep_start, ep_end) for quick bounds lookup
            _ep_ends: Tensor of cumulative end indices for binary search with searchsorted
        """
        # Build sorted episode list (filtered if needed)
        ep_list = sorted(self._episode_paths.keys())
        if self.episodes is not None:
            ep_list = [e for e in ep_list if e in self.episodes]
        self._ep_list = ep_list

        # Build bounds dict and ends tensor
        self._ep_bounds = {}
        ends = []
        for i, ep_idx in enumerate(self._ep_list):
            ep_start = int(self.episode_data_index["from"][i])
            ep_end = int(self.episode_data_index["to"][i])
            self._ep_bounds[ep_idx] = (ep_start, ep_end)
            ends.append(ep_end)

        self._ep_ends = torch.tensor(ends, dtype=torch.long) if ends else torch.empty(0, dtype=torch.long)

    def _build_frame_index(self) -> None:
        """Build mapping from dataset index to global frame index."""
        if self.episodes is None:
            self._frame_indices = None
            self._num_frames = int(self.episode_data_index["to"][-1]) if len(self.episode_data_index["to"]) > 0 else 0
        else:
            # Filter to only selected episodes
            # Note: episode_data_index only contains entries for episodes matching the filter
            indices = []
            for i in range(len(self.episode_data_index["from"])):
                start = int(self.episode_data_index["from"][i])
                end = int(self.episode_data_index["to"][i])
                indices.extend(range(start, end))
            self._frame_indices = indices
            self._num_frames = len(indices)

    def _global_idx_to_episode_local(self, global_idx: int) -> tuple[int, int]:
        """Map global frame index to (episode_idx, local_idx).

        Uses binary search via torch.searchsorted for O(log n) lookup instead of O(n) linear scan.
        """
        if len(self._ep_ends) == 0:
            raise IndexError(f"Global index {global_idx} out of range (no episodes)")

        # Binary search: find the first episode where ep_end > global_idx
        # searchsorted with side="right" finds insertion point after equal values
        ep_list_idx = torch.searchsorted(self._ep_ends, global_idx, side="right").item()

        if ep_list_idx >= len(self._ep_list):
            raise IndexError(f"Global index {global_idx} out of range")

        ep_idx = self._ep_list[ep_list_idx]
        ep_start = int(self.episode_data_index["from"][ep_list_idx])
        local_idx = global_idx - ep_start

        return ep_idx, local_idx

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @cached_property
    def hf_dataset(self) -> ZarrDatasetView:
        """Compatibility shim - returns a zarr-backed dataset view."""
        return ZarrDatasetView(self)

    def _get_zarr_item(self, idx: int) -> dict:
        """Get a single frame from Zarr."""
        start_total = time.perf_counter() if self._profiler else None
        global_idx = self._frame_indices[idx] if self._frame_indices else idx
        ep_idx, local_idx = self._global_idx_to_episode_local(global_idx)

        store = self._get_episode_store(ep_idx)
        if store is None:
            raise ValueError(f"Could not load store for episode {ep_idx}")

        item = {}
        for key in store.array_keys():
            # Skip keys that are explicitly excluded
            if key in self.skip_keys:
                continue

            arr = store[key]
            if not hasattr(arr, "dtype") or not hasattr(arr, "shape"):
                # Skip non-array nodes (e.g., metadata groups)
                continue
            if local_idx >= arr.shape[0]:
                continue

            is_image_key = key in self.meta.camera_keys
            is_encoded_bytes = arr.dtype.kind == "O"  # Variable-length bytes = JPEG compressed images
            if not is_image_key and key.startswith("observation.images.") and is_encoded_bytes:
                # Fallback: treat variable-length byte arrays as images even if metadata is missing
                is_image_key = True

            if is_image_key:
                read_start = time.perf_counter() if self._profiler else None
                raw_data = arr[local_idx]
                if self._profiler and read_start is not None:
                    self._profiler.add("zarr_read_sec", time.perf_counter() - read_start)
                    self._profiler.inc("image_read_count")

                if is_encoded_bytes:
                    # Encoded bytes: decode JPEG
                    decode_start = time.perf_counter() if self._profiler else None
                    img_np = decode_jpeg(raw_data)  # (H, W, C) uint8
                    if self._profiler and decode_start is not None:
                        self._profiler.add("image_decode_sec", time.perf_counter() - decode_start)
                        self._profiler.inc("image_decode_count")
                else:
                    # Raw uint8 array stored directly
                    img_np = raw_data

                # Convert to (C, H, W) float tensor normalized to [0, 1]
                convert_start = time.perf_counter() if self._profiler else None
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                if self._profiler and convert_start is not None:
                    self._profiler.add("image_to_tensor_sec", time.perf_counter() - convert_start)
                    self._profiler.inc("image_tensor_count")
                item[key] = img_tensor
            else:
                if arr.dtype.kind == "O":
                    raise TypeError(
                        f"Unsupported object dtype for key '{key}'. "
                        "If this is an image array, add it to meta/info.json features with dtype='jpeg'."
                    )
                read_start = time.perf_counter() if self._profiler else None
                data = arr[local_idx]
                if self._profiler and read_start is not None:
                    self._profiler.add("zarr_read_sec", time.perf_counter() - read_start)
                    self._profiler.inc("non_image_read_count")

                convert_start = time.perf_counter() if self._profiler else None
                item[key] = torch.as_tensor(data)
                if self._profiler and convert_start is not None:
                    self._profiler.add("non_image_to_tensor_sec", time.perf_counter() - convert_start)
                    self._profiler.inc("non_image_tensor_count")

        # Compute metadata fields on-the-fly
        item["episode_index"] = torch.tensor(ep_idx)
        item["frame_index"] = torch.tensor(local_idx)
        item["timestamp"] = torch.tensor(local_idx / self.fps, dtype=torch.float32)

        if self._profiler and start_total is not None:
            self._profiler.add("getitem_total_sec", time.perf_counter() - start_total)
            self._profiler.inc("getitem_count")

        return item

    def __len__(self):
        return self.num_frames

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Get indices for delta_timestamps queries."""
        # Use cached episode bounds for O(1) lookup instead of O(n) recomputation
        ep_start, ep_end = self._ep_bounds[ep_idx]
        global_idx = self._frame_indices[idx] if self._frame_indices else idx

        query_indices = {
            key: [max(ep_start, min(ep_end - 1, global_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(global_idx + delta < ep_start) | (global_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_zarr_data(self, query_indices: dict[str, list[int]]) -> dict:
        """Query Zarr arrays for multiple indices with optimized batch reads."""
        result = {}
        for key, q_indices in query_indices.items():
            if key in self.skip_keys:
                continue
            if key in self.meta.camera_keys:
                continue

            if not q_indices:
                continue

            # Group indices by episode for batch reads
            # groups: {ep_idx: [(local_idx, output_position), ...]}
            groups: dict[int, list[tuple[int, int]]] = {}
            for out_pos, q_idx in enumerate(q_indices):
                ep_idx, local_idx = self._global_idx_to_episode_local(q_idx)
                if ep_idx not in groups:
                    groups[ep_idx] = []
                groups[ep_idx].append((local_idx, out_pos))

            # Get sample shape from first available store
            first_ep_idx = next(iter(groups.keys()))
            sample_store = self._get_episode_store(first_ep_idx)
            if sample_store is None or key not in sample_store:
                continue

            if sample_store[key].dtype.kind == "O":
                raise TypeError(
                    f"Unsupported object dtype for key '{key}'. "
                    "If this is an image array, add it to meta/info.json features with dtype='jpeg'."
                )

            sample_shape = sample_store[key].shape[1:]  # Shape excluding time dimension

            # Pre-allocate output tensor
            frames = torch.empty((len(q_indices), *sample_shape), dtype=torch.float32)

            # Batch read from each episode
            for ep_idx, idx_pairs in groups.items():
                store = self._get_episode_store(ep_idx)
                if store is None or key not in store:
                    continue

                local_indices = [p[0] for p in idx_pairs]
                out_positions = [p[1] for p in idx_pairs]

                # Sort by local index to check contiguity
                sorted_pairs = sorted(zip(local_indices, out_positions), key=lambda x: x[0])
                sorted_local = [p[0] for p in sorted_pairs]
                sorted_out = [p[1] for p in sorted_pairs]

                # Check if indices are contiguous
                is_contiguous = (
                    len(sorted_local) > 1 and
                    sorted_local == list(range(sorted_local[0], sorted_local[0] + len(sorted_local)))
                )

                arr = store[key]
                if is_contiguous:
                    # Single slice read - much faster for contiguous data
                    start_idx = sorted_local[0]
                    end_idx = sorted_local[-1] + 1
                    if end_idx <= arr.shape[0]:
                        data = torch.as_tensor(arr[start_idx:end_idx])
                        for i, out_pos in enumerate(sorted_out):
                            frames[out_pos] = data[i]
                else:
                    # Non-contiguous: use individual reads (rare case at episode boundaries)
                    for local_idx, out_pos in zip(local_indices, out_positions):
                        if local_idx < arr.shape[0]:
                            frames[out_pos] = torch.as_tensor(arr[local_idx])

            result[key] = frames

        return result

    def _get_zarr_item_by_global(self, global_idx: int) -> dict:
        """Get item by global index (not filtered index)."""
        ep_idx, local_idx = self._global_idx_to_episode_local(global_idx)

        store = self._get_episode_store(ep_idx)
        if store is None:
            return {}

        item = {}
        for key in store.array_keys():
            # Skip keys that are explicitly excluded
            if key in self.skip_keys:
                continue

            arr = store[key]
            if not hasattr(arr, "dtype") or not hasattr(arr, "shape"):
                continue
            if local_idx >= arr.shape[0]:
                continue

            is_image_key = key in self.meta.camera_keys
            is_encoded_bytes = arr.dtype.kind == "O"
            if not is_image_key and key.startswith("observation.images.") and is_encoded_bytes:
                is_image_key = True

            if is_image_key:
                raw_data = arr[local_idx]
                if is_encoded_bytes:
                    img_np = decode_jpeg(raw_data)  # (H, W, C) uint8
                else:
                    img_np = raw_data
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                item[key] = img_tensor
            else:
                if arr.dtype.kind == "O":
                    raise TypeError(
                        f"Unsupported object dtype for key '{key}'. "
                        "If this is an image array, add it to meta/info.json features with dtype='jpeg'."
                    )
                item[key] = torch.as_tensor(arr[local_idx])

        # Compute metadata fields on-the-fly
        item["episode_index"] = torch.tensor(ep_idx)
        item["frame_index"] = torch.tensor(local_idx)
        item["timestamp"] = torch.tensor(local_idx / self.fps, dtype=torch.float32)

        return item

    def _query_images(self, query_indices: dict[str, list[int]]) -> dict:
        """Query images from Zarr for multiple indices."""
        result = {}

        for key in self.meta.camera_keys:
            if key in self.skip_keys:
                continue
            if key not in query_indices:
                continue

            q_indices = query_indices[key]
            if not q_indices:
                continue

            frames_list = []
            for q_idx in q_indices:
                ep_idx, local_idx = self._global_idx_to_episode_local(q_idx)
                store = self._get_episode_store(ep_idx)
                if store is None or key not in store:
                    continue
                arr = store[key]
                if local_idx < arr.shape[0]:
                    raw_data = arr[local_idx]
                    if arr.dtype.kind == "O":
                        img_np = decode_jpeg(raw_data)
                    else:
                        img_np = raw_data
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
                    frames_list.append(img_tensor)

            if frames_list:
                result[key] = torch.stack(frames_list, dim=0)

        return result

    def __getitem__(self, idx) -> dict:
        """Get a single frame with all features."""
        item = self._get_zarr_item(idx)

        query_indices = None
        if self.delta_indices is not None:
            ep_idx = int(item["episode_index"])
            query_indices, padding = self._get_query_indices(idx, ep_idx)

            # Query non-image data
            query_result = self._query_zarr_data(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

            # Query images if delta_timestamps includes camera keys
            image_result = self._query_images(query_indices)
            for key, val in image_result.items():
                item[key] = val

        # Apply image transforms
        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "}})"
        )

    def reset_profile(self) -> None:
        if self._profiler is not None:
            self._profiler.reset()

    def get_profile_summary(self) -> dict[str, dict[str, float | int]]:
        if self._profiler is None:
            return {}
        return self._profiler.summary()


# Import EMBODIMENT enum and helper functions
try:
    from egomimic.rldb.utils import EMBODIMENT, get_embodiment_id, AnnotationLoader
except ImportError:
    # Fallback definitions if utils not available
    from enum import Enum

    class EMBODIMENT(Enum):
        EVE_RIGHT_ARM = 0
        EVE_LEFT_ARM = 1
        EVE_BIMANUAL = 2
        ARIA_RIGHT_ARM = 3
        ARIA_LEFT_ARM = 4
        ARIA_BIMANUAL = 5
        EVA_RIGHT_ARM = 6
        EVA_LEFT_ARM = 7
        EVA_BIMANUAL = 8
        MECKA_BIMANUAL = 9
        MECKA_RIGHT_ARM = 10
        MECKA_LEFT_ARM = 11

    def get_embodiment_id(embodiment_name: str) -> int:
        embodiment_name = embodiment_name.upper()
        return EMBODIMENT[embodiment_name].value

    AnnotationLoader = None


class RLDBZarrDataset(ZarrDataset):
    """
    RLDB-specific Zarr dataset with embodiment tracking,
    task injection, and action slow-down support.

    Mirrors RLDBDataset functionality but reads from Zarr arrays.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        local_files_only: bool = True,
        episodes: list[int] | None = None,
        percent: float = 0.1,
        mode: str = "train",
        valid_ratio: float = 0.2,
        **kwargs,
    ):
        # Initialize metadata first to get splits
        meta = ZarrDatasetMetadata(repo_id=repo_id, root=root, local_files_only=True)
        meta._update_splits(valid_ratio=valid_ratio)

        # Get embodiment from robot_type
        robot_type = meta.robot_type or "EVA_BIMANUAL"
        self.embodiment = get_embodiment_id(robot_type)

        # Task string injection
        self.use_task_string = kwargs.get("use_task_string", False)
        self.task_string = kwargs.get("task_string", "") if self.use_task_string else ""

        # Action slow-down parameters
        self.slow_down_factor = float(kwargs.get("slow_down_factor", 1.0))
        self._parse_slow_down_params(kwargs)

        # Determine episodes based on mode
        dataset_splits = meta.info.get("splits", {})
        train_indices = dataset_splits.get("train", list(range(meta.total_episodes)))

        if mode == "train":
            selected_episodes = train_indices
        elif mode == "valid":
            if "valid" not in dataset_splits:
                raise ValueError(
                    f"Validation split not found. Please update dataset metadata in {meta.root}/meta/info.json"
                )
            selected_episodes = dataset_splits["valid"]
        elif mode == "sample" and episodes is not None:
            selected_episodes = episodes
        elif mode == "percent" and percent is not None:
            selected_episodes = train_indices
        else:
            selected_episodes = None

        # Remove kwargs that aren't for parent class
        parent_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ["use_task_string", "task_string", "slow_down_factor",
                        "slow_down_ac_keys", "slow_down_rot_specs"]
        }

        super().__init__(
            repo_id=repo_id,
            root=root,
            local_files_only=local_files_only,
            episodes=selected_episodes,
            **parent_kwargs,
        )

        # Handle percent mode - sample a percentage of frames
        self.sampled_indices: list[int] | None = None
        if mode == "percent" and percent is not None:
            if not (0 < percent <= 1):
                raise ValueError("Percent should be a value between 0 and 1.")
            total_frames = len(self)
            num_sampled = int(percent * total_frames)
            self.sampled_indices = sorted(
                random.sample(range(total_frames), num_sampled)
            )

    def _parse_slow_down_params(self, kwargs: dict) -> None:
        """Parse slow-down action keys and rotation specs."""
        from collections.abc import Sequence

        # Parse slow_down_ac_keys
        raw_keys = kwargs.get("slow_down_ac_keys", None)
        if raw_keys is None:
            self.slow_down_ac_keys = []
        elif isinstance(raw_keys, str):
            self.slow_down_ac_keys = [raw_keys]
        elif isinstance(raw_keys, Sequence) and not isinstance(raw_keys, (str, bytes)):
            self.slow_down_ac_keys = list(raw_keys)
        else:
            raise ValueError(
                f"slow_down_ac_keys must be str, sequence, or None; got {type(raw_keys)}"
            )

        # Parse slow_down_rot_specs
        raw_rot_specs = kwargs.get("slow_down_rot_specs", None)
        if raw_rot_specs is None:
            self.slow_down_rot_specs = {}
        else:
            self.slow_down_rot_specs = dict(raw_rot_specs)

        # Validate rotation specs
        for k, v in self.slow_down_rot_specs.items():
            if not (isinstance(v, Sequence) and not isinstance(v, (str, bytes)) and len(v) == 2):
                raise ValueError(
                    f"slow_down_rot_specs['{k}'] must be (rot_type, index_ranges), got {type(v)}"
                )

            rot_type, ranges = v
            if rot_type not in ("quat_wxyz", "ypr"):
                raise ValueError(
                    f"Rotation type for key '{k}' must be 'quat_wxyz' or 'ypr', got {rot_type}"
                )

            if not (isinstance(ranges, Sequence) and not isinstance(ranges, (str, bytes))):
                raise ValueError(
                    f"Index ranges for slow_down_rot_specs['{k}'] must be a sequence of (start, end) pairs"
                )

            for pair in ranges:
                if not (isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2):
                    raise ValueError(
                        f"Each index range for slow_down_rot_specs['{k}'] must be a (start, end) sequence"
                    )

    def __len__(self):
        """Return the total number of sampled frames if in 'percent' mode."""
        if self.sampled_indices is not None:
            return len(self.sampled_indices)
        return super().__len__()

    def __getitem__(self, idx):
        """Fetch frames with RLDB-specific processing."""
        # Map index if in percent mode
        if self.sampled_indices is not None:
            idx = self.sampled_indices[idx]

        item = super().__getitem__(idx)

        # Add embodiment (computed from robot_type, not stored per-frame)
        item["metadata.embodiment"] = torch.tensor(self.embodiment)

        # Inject task string
        if self.use_task_string:
            item["high_level_language_prompt"] = self.task_string

        # Apply action slow-down
        if self.slow_down_ac_keys and self.slow_down_factor > 1.0:
            for key in self.slow_down_ac_keys:
                if key in item:
                    rot_spec = self.slow_down_rot_specs.get(key, None)
                    item[key] = self._slow_down_sequence(item[key], rot_spec)

        # Load annotations
        item = self._load_annotations(item, idx)

        return item

    def _slow_down_sequence(self, seq: torch.Tensor, rot_spec=None) -> torch.Tensor:
        """
        Slow down a sequence of shape (S, D) along the time dimension S.

        Steps:
        1. Take first S / slow_down_factor steps (shortened trajectory).
        2. Linearly upsample back to length S.
        3. For rotation slices, use SLERP interpolation.
        """
        alpha = self.slow_down_factor
        if alpha is None or alpha <= 1.0:
            return seq

        if seq.ndim != 2:
            raise ValueError(
                f"_slow_down_sequence expects seq of shape (S, D). Got shape {seq.shape}"
            )

        S, D = seq.shape
        S_short = max(1, min(S, int(S / alpha)))

        if S_short == S:
            return seq

        # Base: linear interpolation over full feature dimension
        seq_short = seq[:S_short]  # (S_short, D)

        x = seq_short.transpose(0, 1).unsqueeze(0)  # (1, D, S_short)
        x_interp = F.interpolate(x, size=S, mode="linear", align_corners=True)  # (1, D, S)
        out = x_interp.squeeze(0).transpose(0, 1)  # (S, D)

        # If we have rotation specs, overwrite specified feature slices with SLERP output
        if rot_spec is not None:
            rot_type, index_ranges = rot_spec

            for start, end in index_ranges:
                if not (0 <= start < end <= D):
                    raise ValueError(f"Invalid rotation slice [{start}:{end}] for seq with D={D}")

                rot_short = seq_short[:, start:end]  # (S_short, k)
                k = end - start

                if rot_type == "quat_wxyz":
                    if k != 4:
                        raise ValueError(f"quat slice must have length 4, got {k}")
                    rot_interp = _slow_down_slerp_quat(rot_short, S)
                    out[:, start:end] = rot_interp

                elif rot_type == "ypr":
                    if k != 3:
                        raise ValueError(f"ypr slice must have length 3, got {k}")
                    # ypr -> quat -> slerp -> ypr
                    quat_short = _ypr_to_quat(rot_short)
                    quat_interp = _slow_down_slerp_quat(quat_short, S)
                    ypr_interp = _quat_to_ypr(quat_interp)
                    out[:, start:end] = ypr_interp
                else:
                    raise ValueError(f"Unknown rotation type: {rot_type}")

        return out

    def _load_annotations(self, item: dict, _idx: int) -> dict:
        """Load annotations from CSV if available."""
        # Check if annotations directory exists
        annotation_path = Path(self.root) / "annotations"
        if not annotation_path.is_dir():
            item["annotations"] = ""
            return item

        if AnnotationLoader is None:
            item["annotations"] = ""
            return item

        try:
            annotations = AnnotationLoader(root=self.root)
            df = annotations.df

            ep_idx = int(item.get("episode_index", 0))
            frame_time = float(item.get("timestamp", 0))
            df_episode = df.loc[df["idx"].astype(int) == ep_idx]

            if df_episode.empty:
                item["annotations"] = ""
                return item

            frame_annotations = df_episode[
                (df_episode["start_time"] <= frame_time)
                & (df_episode["end_time"] >= frame_time)
            ]

            if frame_annotations.empty:
                next_ann = df_episode[df_episode["start_time"] > frame_time]
                if next_ann.empty:
                    annotation = df_episode.tail(1)["Labels"].iloc[0]
                else:
                    next_pos = df_episode.index.get_loc(next_ann.index[0])
                    prev_pos = next_pos - 1
                    if prev_pos >= 0:
                        annotation = df_episode.iloc[prev_pos]["Labels"]
                    else:
                        annotation = ""
                item["annotations"] = annotation
            else:
                item["annotations"] = frame_annotations["Labels"].iloc[0]

        except Exception as e:
            logger.debug(f"Error loading annotations: {e}")
            item["annotations"] = ""

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Embodiment: '{self.embodiment}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{len(self)}',\n"
            f"    Features: '{feature_keys}',\n"
            "}})"
        )
