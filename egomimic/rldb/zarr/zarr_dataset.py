"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure:
    dataset_root/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── episodes.jsonl
    │   └── tasks.jsonl
    ├── data/
    │   └── episode_{ep_idx}/
    │       ├── meta/
    │       │   └── episode_info.json
    │       └── chunk-{chunk_idx}.zarr/
    │           ├── timestamp
    │           ├── frame_index
    │           ├── action
    │           ├── observation.state
    │           └── ...
    └── videos/
        └── episode_{ep_idx}/
            └── observation.images.{cam}/
                └── chunk-{chunk_idx}.mp4
"""

from __future__ import annotations

import json
import logging
import random
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from lerobot.common.datasets.utils import (
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    EPISODES_PATH,
    get_delta_indices,
    check_delta_timestamps,
    load_json,
    load_jsonlines,
    flatten_dict,
    unflatten_dict,
)
from lerobot.common.datasets.video_utils import decode_video_frames_torchvision

from egomimic.rldb.data_utils import (
    _ypr_to_quat,
    _quat_to_ypr,
    _slow_down_slerp_quat,
)

logger = logging.getLogger(__name__)

SEED = 42


def load_info(local_dir: Path) -> dict:
    """Load info.json and convert shape lists to tuples."""
    info = load_json(local_dir / INFO_PATH)
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


def load_episodes(local_dir: Path) -> list[dict]:
    """Load episodes.jsonl."""
    episodes_path = local_dir / EPISODES_PATH
    if not episodes_path.exists():
        return []
    return load_jsonlines(episodes_path)


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
        self.episodes = load_episodes(self.root)

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
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft.get("dtype") == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of storage method)."""
        return [key for key, ft in self.features.items() if ft.get("dtype") in ["video", "image"]]

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
        return self.info.get("total_episodes", len(self.episodes))

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
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info.get("video_path")

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

    def get_video_file_path(self, ep_index: int, vid_key: str, chunk_idx: int = 0) -> Path:
        """Get path to video file for an episode."""
        # videos/episode_{ep_idx}/observation.images.{cam}/chunk-{chunk_idx}.mp4
        return Path(f"videos/episode_{ep_index:06d}/{vid_key}/chunk-{chunk_idx:03d}.mp4")

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
        """Get item without video loading (for compatibility)."""
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
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else Path.cwd() / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or "pyav"
        self.local_files_only = local_files_only
        self.delta_indices = None

        # Load metadata
        self.meta = ZarrDatasetMetadata(repo_id, self.root, local_files_only)

        # Discover episodes and chunks
        self._episode_paths: dict[int, Path] = {}
        self._episode_chunks: dict[int, list[int]] = {}
        self._chunk_stores: dict[tuple[int, int], zarr.Group] = {}
        self._discover_episodes()

        # Build episode data index (global frame -> episode/chunk/local mapping)
        self.episode_data_index = self._build_episode_data_index()

        # Build frame index for episode filtering
        self._frame_indices: list[int] | None = None
        self._num_frames: int = 0
        self._build_frame_index()

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _discover_episodes(self) -> None:
        """Find all episode folders in data/."""
        data_dir = self.root / "data"
        if not data_dir.exists():
            logger.warning(f"Data directory not found: {data_dir}")
            return

        for ep_dir in sorted(data_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            # Parse episode index from folder name: episode_{ep_idx}
            name = ep_dir.name
            if not name.startswith("episode_"):
                continue
            try:
                ep_idx = int(name.replace("episode_", ""))
            except ValueError:
                continue

            self._episode_paths[ep_idx] = ep_dir
            self._episode_chunks[ep_idx] = self._discover_chunks(ep_idx)

    def _discover_chunks(self, ep_idx: int) -> list[int]:
        """Find all chunk Zarr stores within an episode."""
        ep_dir = self._episode_paths.get(ep_idx)
        if ep_dir is None:
            return []

        chunks = []
        for item in sorted(ep_dir.iterdir()):
            # Look for chunk-{idx}.zarr directories
            name = item.name
            if not name.startswith("chunk-") or not name.endswith(".zarr"):
                continue
            try:
                chunk_idx = int(name.replace("chunk-", "").replace(".zarr", ""))
                chunks.append(chunk_idx)
            except ValueError:
                continue

        return sorted(chunks)

    def _get_chunk_store(self, ep_idx: int, chunk_idx: int) -> zarr.Group | None:
        """Open/cache chunk Zarr store on demand."""
        key = (ep_idx, chunk_idx)
        if key in self._chunk_stores:
            return self._chunk_stores[key]

        ep_dir = self._episode_paths.get(ep_idx)
        if ep_dir is None:
            return None

        chunk_path = ep_dir / f"chunk-{chunk_idx:03d}.zarr"
        if not chunk_path.exists():
            return None

        store = zarr.open(str(chunk_path), mode="r")
        self._chunk_stores[key] = store
        return store

    def _build_episode_data_index(self) -> dict[str, torch.Tensor]:
        """Build from/to indices for each episode."""
        # Get episode lengths from metadata or by scanning chunks
        episode_lengths = {}

        for ep_idx in sorted(self._episode_paths.keys()):
            if self.episodes is not None and ep_idx not in self.episodes:
                continue

            # Try to get length from metadata
            if ep_idx < len(self.meta.episodes):
                episode_lengths[ep_idx] = self.meta.episodes[ep_idx].get("length", 0)
            else:
                # Calculate from chunks
                total_len = 0
                for chunk_idx in self._episode_chunks.get(ep_idx, []):
                    store = self._get_chunk_store(ep_idx, chunk_idx)
                    if store is not None and "timestamp" in store:
                        total_len += store["timestamp"].shape[0]
                episode_lengths[ep_idx] = total_len

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

    def _global_idx_to_episode_chunk_local(self, global_idx: int) -> tuple[int, int, int]:
        """Map global frame index to (episode_idx, chunk_idx, local_idx)."""
        # Find which episode this frame belongs to
        ep_list = sorted(self._episode_paths.keys())
        if self.episodes is not None:
            ep_list = [e for e in ep_list if e in self.episodes]

        for i, ep_idx in enumerate(ep_list):
            ep_start = int(self.episode_data_index["from"][i])
            ep_end = int(self.episode_data_index["to"][i])

            if ep_start <= global_idx < ep_end:
                # Found the episode, now find the chunk
                local_in_episode = global_idx - ep_start
                cumulative = 0

                for chunk_idx in self._episode_chunks.get(ep_idx, []):
                    store = self._get_chunk_store(ep_idx, chunk_idx)
                    if store is None:
                        continue
                    chunk_len = store["timestamp"].shape[0]

                    if cumulative + chunk_len > local_in_episode:
                        local_in_chunk = local_in_episode - cumulative
                        return ep_idx, chunk_idx, local_in_chunk

                    cumulative += chunk_len

        raise IndexError(f"Global index {global_idx} out of range")

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
        """Get a single frame from Zarr (without video loading)."""
        global_idx = self._frame_indices[idx] if self._frame_indices else idx
        ep_idx, chunk_idx, local_idx = self._global_idx_to_episode_chunk_local(global_idx)

        store = self._get_chunk_store(ep_idx, chunk_idx)
        if store is None:
            raise ValueError(f"Could not load chunk store for episode {ep_idx}, chunk {chunk_idx}")

        item = {}
        for key in store.keys():
            if key in self.meta.camera_keys:
                continue  # Skip video keys
            arr = store[key]
            if local_idx < arr.shape[0]:
                item[key] = torch.from_numpy(np.array(arr[local_idx]))

        # Add episode_index if not in data
        if "episode_index" not in item:
            item["episode_index"] = torch.tensor(ep_idx)

        return item

    def __len__(self):
        return self.num_frames

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Get indices for delta_timestamps queries."""
        # Find episode bounds in our filtered dataset
        ep_list = sorted(self._episode_paths.keys())
        if self.episodes is not None:
            ep_list = [e for e in ep_list if e in self.episodes]

        try:
            ep_list_idx = ep_list.index(ep_idx)
        except ValueError:
            ep_list_idx = 0

        ep_start = int(self.episode_data_index["from"][ep_list_idx])
        ep_end = int(self.episode_data_index["to"][ep_list_idx])

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
        """Query Zarr arrays for multiple indices."""
        result = {}
        for key, q_indices in query_indices.items():
            if key in self.meta.camera_keys:
                continue

            frames = []
            for q_idx in q_indices:
                item = self._get_zarr_item(q_idx) if self._frame_indices is None else self._get_zarr_item_by_global(q_idx)
                if key in item:
                    frames.append(item[key])

            if frames:
                result[key] = torch.stack(frames)

        return result

    def _get_zarr_item_by_global(self, global_idx: int) -> dict:
        """Get item by global index (not filtered index)."""
        ep_idx, chunk_idx, local_idx = self._global_idx_to_episode_chunk_local(global_idx)

        store = self._get_chunk_store(ep_idx, chunk_idx)
        if store is None:
            return {}

        item = {}
        for key in store.keys():
            if key in self.meta.camera_keys:
                continue
            arr = store[key]
            if local_idx < arr.shape[0]:
                item[key] = torch.from_numpy(np.array(arr[local_idx]))

        if "episode_index" not in item:
            item["episode_index"] = torch.tensor(ep_idx)

        return item

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        """Get timestamps for video frame queries."""
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = []
                for q_idx in query_indices[key]:
                    item = self._get_zarr_item_by_global(q_idx)
                    if "timestamp" in item:
                        timestamps.append(float(item["timestamp"]))
                    else:
                        timestamps.append(current_ts)
                query_timestamps[key] = timestamps
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """Decode video frames."""
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Find which chunk(s) contain these timestamps
            # For now, assume chunk 0 (can be extended for multi-chunk support)
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key, chunk_idx=0)

            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                continue

            frames = decode_video_frames_torchvision(
                video_path, query_ts, self.tolerance_s, self.video_backend
            )
            item[vid_key] = frames.squeeze(0)

        return item

    def __getitem__(self, idx) -> dict:
        """Get a single frame with all features."""
        item = self._get_zarr_item(idx)
        ep_idx = int(item["episode_index"])

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_zarr_data(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        # Load videos
        if len(self.meta.video_keys) > 0:
            current_ts = float(item["timestamp"]) if "timestamp" in item else 0.0
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

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

    def _load_annotations(self, item: dict, idx: int) -> dict:
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
            current_ts = float(item.get("timestamp", 0))
            fps = float(self.fps)

            frame_time = current_ts
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
