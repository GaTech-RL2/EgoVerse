"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observations.images.{cam}  (JPEG compressed)
        ├── observations.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations
import json
import logging
import os
import random
import time
from functools import cached_property
from pathlib import Path
from typing import Callable
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
import boto3
import subprocess
import tempfile
import tqdm
from datasets import DatasetDict, concatenate_datasets
from enum import Enum

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    text,
)

from egomimic.utils.aws.aws_sql import (
    TableRow,
    add_episode,
    create_default_engine,
    delete_all_episodes,
    delete_episodes,
    episode_hash_to_table_row,
    episode_table_to_df,
    update_episode,
)


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


SEED = 42
EMBODIMENT_ID_TO_KEY = {
    member.value: key for key, member in EMBODIMENT.__members__.items()
}

def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """
    Split a list of dataset names into train/valid sets.


    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


def get_embodiment(index):
    return EMBODIMENT_ID_TO_KEY.get(index, None)


def get_embodiment_id(embodiment_name):
    embodiment_name = embodiment_name.upper()
    return EMBODIMENT[embodiment_name].value



class EpisodeResolver:
    """
    Filters SQL table for zarr episode paths/ downloads from S3.
    resolve returns processed_path. 
    """
    def __init__(
        self,
        folder_path,
        bucket_name="rldb",
        main_prefix="processed_v2",
    ):
        self.folder_path = folder_path
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix

    def resolve(
        self,
        embodiment,
        sync_from_s3 = False,
        filters={},
    ) -> list[tuple[str, str]]:
        """
        Outputs a list of ZarrDatasets with relevant filters. 
        If sync_from_s3 is True, sync S3 paths to local_root before indexing.
        If not True, assuming that folders already exist within folder.
        """
        
        if sync_from_s3:
            filters["robot_name"] = embodiment
            filters["is_deleted"] = False
      
            if self.folder_path.is_dir():
                logger.info(f"Using existing directory: {self.folder_path}")
            if not self.folder_path.is_dir():
                self.folder_path.mkdir()

            logger.info(f"Filters: {filters}")

            datasets = {}
            skipped = []
            filtered_paths = self.sync_from_filters(
                bucket_name= self.bucket_name,
                filters=filters,
                local_dir=self.folder_path,
            )
        else:
            filtered_paths = self._get_filtered_paths(filters)

        valid_hashes = set()
        for _, hashes in filtered_paths:
            valid_hashes.add(hashes)
        if not valid_hashes:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        datasets = self._load_zarr_datasets(
            search_path = self.folder_path, 
            valid_hashes = valid_hashes,
        )

        return datasets

    @classmethod
    def _load_zarr_datasets(cls, *, search_path: Path, valid_hashes: set[str]):
        
        """
        Loads multiple Zarr datasets from the specified folder path, filtering only those whose hashes
        are present in the valid_hashes set.

        Args:
            folder_path (Path): The root folder containing candidate dataset subfolders.
            valid_hashes (set[str]): Set of valid dataset hashes (names) to load.

        Returns:
            dict[str, ZarrDataset]: a dictionary mapping string keys to constructed zarr datasets from valid filters.
        """
        all_paths = sorted(search_path.iterdir())
        datasets: dict[str, ZarrDataset] = {}
        skipped: list[str] = []
        for p in all_paths:
            if not p.is_dir():
                logger.info(f"{p} is not a valid directory")
                skipped.append(p.name)
                continue
            if p.name not in valid_hashes:
                logger.info(f"{p} is not in the list of filtered paths") 
                skipped.append(p.name)
                continue     
            try:
                ds_obj = ZarrDataset(p)
                datasets[p.name] = ds_obj
            except Exception as e:
                logger.error(f"Failed to load dataset at {p}: {e}")
                skipped.append(p.name)
            
        return datasets, skipped

    @staticmethod
    def _get_filtered_paths(filters):
        """
        Filters episodes from the SQL episode table according to the criteria specified in `filters`
        and returns a list of (processed_path, episode_hash) tuples for episodes that match and have
        a non-null processed_path.

        Args:
            filters (dict): Dictionary of filter key-value pairs to apply on the episode table.

        Returns:
            list[tuple[str, str]]: List of tuples, each containing (processed_path, episode_hash)
                                   for episodes passing the filter criteria.
        """
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        series = pd.Series(filters)

        output = df.loc[
            (df[list(filters)] == series).all(axis=1),
            ["processed_path", "episode_hash"],
        ]
        skipped = df[df["processed_path"].isnull()]["episode_hash"].tolist()
        logger.info(
            f"Skipped {len(skipped)} episodes with null processed_path: {skipped}"
        )
        output = output[~output["episode_hash"].isin(skipped)]

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths


    @classmethod
    def _sync_s3_to_local(cls, bucket_name, s3_paths, local_dir: Path):
        if not s3_paths:
            return

        # 0) Skip episodes already present locally
        to_sync = []
        already = []
        for processed_path, episode_hash in s3_paths:
            if cls._episode_already_present(local_dir, episode_hash):
                already.append(episode_hash)
            else:
                to_sync.append((processed_path, episode_hash))

        if already:
            logger.info("Skipping %d episodes already present locally.", len(already))

        if not to_sync:
            logger.info("Nothing to sync from S3 (all episodes already present).")
            return

        # 1) Build s5cmd batch script (one line per episode)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_sync_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            batch_path = Path(tmp_file.name)

        lines = []
        for processed_path, episode_hash in to_sync:
            # processed_path like: s3://rldb/processed_v2/eva/<hash>/
            if processed_path.startswith("s3://"):
                src_prefix = processed_path.rstrip("/") + "/*"
            else:
                src_prefix = (
                    f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"
                    + "/*"
                )

            # Destination is the root local_dir; s5cmd will preserve <hash>/... under it
            dst = local_dir / episode_hash
            lines.append(f'sync "{src_prefix}" "{str(dst)}/"')

        try:
            batch_path.write_text("\n".join(lines) + "\n")

            cmd = ["s5cmd", "run", str(batch_path)]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            subprocess.run(cmd, check=True)

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)


    """
    TODO: add more robust logic when the full folder logic is fleshed out
    """
    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        ep = local_dir / episode_hash

        if not ep.isdir():
            return False

        return True

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: dict,
        local_dir: Path,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir


        Returns:
            List[(processed_path, episode_hash)]
        """

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_filtered_paths(filters)
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        # 2) Logging
        logger.info(
            f"Syncing S3 datasets with filters {filters} to local directory {local_dir}..."
        )

        # 3) Sync
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=filtered_paths,
            local_dir=local_dir,
        )

        return filtered_paths


class MultiDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiDataset, can wrap zarr or multi dataset. 
    note: I am not adding embodiments yet because to match would require something beyond current zarr dataset lazy loading
    """
    def __init__(self, 
        datasets,
        embodiment,
        mode="train",
        percent=0.1,
        key_map=None,
        valid_ratio=0.2,
        **kwargs,):
        """
        Args:
            datasets (dict): Dictionary mapping unique dataset hashes (str) to dataset objects. Datasets can be individual Zarr datasets or other multi-datasets; mixing different types is supported.
            embodiment (str or int): The embodiment type or ID associated with all datasets (e.g., robot name or numeric ID).
            mode (str, optional): Split mode to use (e.g., "train", "valid"). Defaults to "train".
            percent (float, optional): Fraction of the dataset to use from each underlying dataset. Defaults to 0.1.
            key_map (dict, optional): If provided, a dictionary of per-dataset key mapping dicts for remapping source dataset keys to unified keys. Keyed by dataset hash.
            valid_ratio (float, optional): Validation split ratio for datasets that support a train/valid split.
            **kwargs: Additional keyword arguments passed to underlying dataset constructors if needed.
        """
        self.datasets = datasets
        self.key_map = key_map

        self.index_map = []
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_name, local_idx))

        #self.zarr_dataset = self._merge_datasets()
        self.train_collections, self.valid_collections = split_dataset_names(
            datasets.keys(), valid_ratio=valid_ratio, seed=SEED
        )

        if mode == "train":
            chosen = self.train_collections
        elif mode == "valid":
            chosen = self.valid_collections
        elif mode == "total":
            chosen = set(datasets.keys())
        elif mode == "percent":
            all_names = sorted(datasets.keys())
            rng = random.Random(SEED)
            rng.shuffle(all_names)

            n_keep = int(len(all_names) * percent)
            if percent > 0.0:
                n_keep = max(1, n_keep)
            chosen = set(all_names[:n_keep])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}
        assert datasets, "No datasets left after applying mode split."

        self.key_map = (
            {repo_id: key_map for repo_id in datasets} if key_map else None
        )


        super().__init__()

    def __len__(self) -> int:
        return len(self.index_map)


    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_map[idx]
        data = self.datasets[dataset_name][local_idx]

        if self.key_map and dataset_name in self.key_map:
            key_map = self.key_map[dataset_name]
            data = {key_map.get(k, k): v for k, v in data.items()}

        return data

    def _merge_datasets(self):
        """
        Merge zarrdatasets from multiple RLDBDataset instances while remapping keys.


        Returns:
            A unified MultiDataset
        """
        dataset_list = []

        for dataset_name, sub_dataset in self.datasets.items():
            # Apply key mapping if available
            if self.key_map and dataset_name in self.key_map:
                key_map = self.key_map[dataset_name]
                sub_dataset = sub_dataset.rename_columns(key_map)

            dataset_list.append(sub_dataset)

        merged_dataset = concatenate_datasets(dataset_list)

        return merged_dataset


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path: str,
        action_horizon: int | None = None,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            action_horizon: Number of future timesteps to load for action chunking.
                If specified, actions_base_cartesian and actions_joints will be loaded
                as sequences of shape (action_horizon, action_dim) instead of single frames.
                If None, actions are loaded as single frames (action_dim,).
        """
        self.episode_path = Episode_path
        self.metadata = None
        self.action_horizon = action_horizon
        self.action_keys = {"actions_cartesian", "actions_joints"}
        self._image_keys = None  # Lazy-loaded set of JPEG-encoded keys
        self.init_episode()
        super().__init__()

    def init_episode(self):
        """
        inits the zarr episode and all the metadata associated, as well as total_frames for len
        """
        self.episode_reader = ZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}
        self.embodiment = int(get_embodiment_id(self.metadata["robot_type"]))

        # Detect JPEG-encoded image keys from metadata
        self._image_keys = self._detect_image_keys()

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    def __len__(self) -> int:
        if hasattr(self, 'total_frames'):
            return self.total_frames
        else:
            self.init_episode()
            return self.total_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not hasattr(self, 'episode_reader'):
            self.init_episode()

        # Build keys_dict with ranges based on whether action chunking is enabled
        keys_dict = {}
        for k in self.episode_reader._collect_keys():
            # Apply action horizon to action keys
            if self.action_horizon is not None and k in self.action_keys:
                # Load action sequence from idx to idx + action_horizon
                end_idx = min(idx + self.action_horizon, self.total_frames)
                keys_dict[k] = (idx, end_idx)
            else:
                # Load single frame
                keys_dict[k] = (idx, None)

        data = self.episode_reader.read(keys_dict)

        # Pad action sequences to fixed length if needed
        if self.action_horizon is not None:
            for k in self.action_keys:
                if k in data and isinstance(data[k], np.ndarray):
                    seq_len = data[k].shape[0]
                    if seq_len < self.action_horizon:
                        # Pad by repeating the last frame
                        pad_len = self.action_horizon - seq_len
                        last_frame = data[k][-1:]  # Keep dims: (1, action_dim)
                        padding = np.repeat(last_frame, pad_len, axis=0)
                        data[k] = np.concatenate([data[k], padding], axis=0)

        # Decode JPEG-encoded image data and normalize to [0, 1]
        import simplejpeg
        for key in self._image_keys:
            if key in data:
                jpeg_bytes = data[key]
                # Decode JPEG bytes to numpy array (H, W, 3)
                decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace='RGB')
                data[key] = torch.from_numpy(np.transpose(decoded, (2, 0, 1))).to(torch.float32) / 255.0
                   
                

        # Convert all numpy arrays in data to torch tensors
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

            
        
        # Add embodiment id
        data["metadata.embodiment"] = self.embodiment

        return data




class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.
    Designed for efficient PyTorch DataLoader usage with direct store access.
    """
    __slots__ = (
        "_path",
        "_store",
        "metadata",
        "keys",
    )
    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.
        Args:
            path: Path to the .zarr episode directory
        """
        self._path = Path(path)
        self._store = zarr.open_group(str(self._path), mode='r')
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]
        
    def read(self, keys_with_ranges: dict[str, tuple[int, int | None]]) -> dict[str, np.ndarray]:
        """
        Read data for specified keys, each with their own index or range.
        Args:
            keys_with_ranges: Dictionary mapping keys to (start, end) tuples.
                - start: Starting frame index
                - end: Ending frame index (exclusive). If None, reads single frame at start.
        Returns:
            Dictionary mapping keys to numpy arrays
        Example:
            >>> episode.read({
            ...     "obs/image": (0, 10),      # Read frames 0-10
            ...     "actions": (5, 15),        # Read frames 5-15
            ...     "rewards": (20, None),     # Read single frame at index 20
            ... })
        """
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            arr = self._store[key]
            if end is not None:
                data = arr[start:end]
            else:
                # Single frame read - use slicing to avoid 0D array issues with VariableLengthBytes
                # arr[start:start+1] gives us a 1D array, then [0] extracts the actual object
                data = arr[start:start+1][0]
            result[key] = data
        return result
    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.
        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        keys = []
        for name in self._store.array_keys():
            keys.append(name)
        return keys
    def __len__(self) -> int:
        """
        Get total number of frames in the episode.
        Returns:
            Number of frames
        """
        return self.metadata['total_frames']
    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"