import ast
import logging
import os
import random
import shutil
from datetime import datetime, timezone
from enum import Enum
from multiprocessing.dummy import connection
from pathlib import Path
from unittest import result

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from datasets import DatasetDict, concatenate_datasets
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
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
from torch.utils.data import Subset
from collections.abc import Sequence

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

logger = logging.getLogger(__name__)

logging.getLogger("huggingface_hub._snapshot_download").setLevel(logging.ERROR)

import torch.nn.functional as F

from egomimic.rldb.data_utils import _ypr_to_quat, _slerp, _quat_to_ypr, _slow_down_slerp_quat
import subprocess
from urllib.parse import urlparse
from collections import defaultdict

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


def nds(nested_ds, tab_level=0):
    """
    Print the structure of a nested dataset.
    nested_ds: a series of nested dictionaries and iterables.  If a dictionary, print the key and recurse on the value.  If a list, print the length of the list and recurse on just the first index.  For other types, just print the shape.
    """

    def is_key(x):
        return hasattr(x, "keys") and callable(x.keys)

    def is_listy(x):
        return isinstance(x, list)

    # print('--' * tab_level, end='')
    if is_key(nested_ds):
        print("dict with keys: ", nested_ds.keys())
    elif is_listy(nested_ds):
        print("list of len: ", len(nested_ds))
    elif nested_ds is None:
        print("None")
    else:
        # print('\t' * (tab_level), end='')
        print(nested_ds.shape)

    if is_key(nested_ds):
        for key, value in nested_ds.items():
            print("\t" * (tab_level), end="")
            print(f"{key}: ", end="")
            nds(value, tab_level + 1)
    elif isinstance(nested_ds, list):
        print("\t" * tab_level, end="")
        print("Index[0]", end="")
        nds(nested_ds[0], tab_level + 1)


class RLDBDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        root,
        local_files_only=False,
        episodes=None,
        percent=0.1,
        mode="train",
        valid_ratio: float = 0.2,
        **kwargs,
    ):
        dataset_meta = LeRobotDatasetMetadata(
            repo_id=repo_id, root=root, local_files_only=local_files_only
        )
        dataset_meta._update_splits(valid_ratio=valid_ratio)

        dataset_splits = dataset_meta.info["splits"]
        train_indices = dataset_splits["train"]

        self.embodiment = get_embodiment_id(dataset_meta.robot_type)
        self.sampled_indices = None

        self.use_task_string = kwargs.get("use_task_string", False)
        if self.use_task_string:
            self.task_string = kwargs.get("task_string", "")

        self.slow_down_factor = float(kwargs.get("slow_down_factor", 1.0))
        raw_keys = kwargs.get("slow_down_ac_keys", None)
        raw_rot_specs = kwargs.get("slow_down_rot_specs", None)
        
        if raw_rot_specs is None:
            self.slow_down_rot_specs = {}
        else:
            self.slow_down_rot_specs = dict(raw_rot_specs)
            
        for k, v in self.slow_down_rot_specs.items():
            # v should be a 2-tuple-like: (rot_type, index_ranges)
            if not (isinstance(v, Sequence) and not isinstance(v, (str, bytes)) and len(v) == 2):
                raise ValueError(
                    f"slow_down_rot_specs['{k}'] must be (rot_type, index_ranges), got {type(v)} with value {v}"
                )

            rot_type, ranges = v

            if rot_type not in ("quat_wxyz", "ypr"):
                raise ValueError(
                    f"Rotation type for key '{k}' must be 'quat_wxyz' or 'ypr', got {rot_type}"
                )

            if not (isinstance(ranges, Sequence) and not isinstance(ranges, (str, bytes))):
                raise ValueError(
                    f"Index ranges for slow_down_rot_specs['{k}'] must be a sequence of (start, end) pairs, got {type(ranges)}"
                )

            for pair in ranges:
                if not (isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2):
                    raise ValueError(
                        f"Each index range for slow_down_rot_specs['{k}'] must be a (start, end) sequence, got {pair}"
                    )
        
        if raw_keys is None:
            self.slow_down_ac_keys = []
        elif isinstance(raw_keys, str):
            # single key as string
            self.slow_down_ac_keys = [raw_keys]
        elif isinstance(raw_keys, Sequence) and not isinstance(raw_keys, (str, bytes)):
            # list, tuple, Hydra ListConfig, etc.
            self.slow_down_ac_keys = list(raw_keys)
        else:
            raise ValueError(
                f"slow_down_ac_keys must be str, sequence, or None; got {type(raw_keys)}"
            )
        
        if mode == "train":
            super().__init__(
                repo_id=repo_id,
                root=root,
                local_files_only=local_files_only,
                episodes=train_indices,
            )

        elif mode == "valid":
            assert "valid" in dataset_splits, (
                "Validation split not found in dataset_splits. "
                f"Please include a 'valid' key by updating your dataset metadata in {dataset_meta.root}.info.json ."
            )
            valid_indices = dataset_splits["valid"]
            super().__init__(
                repo_id=repo_id,
                root=root,
                local_files_only=local_files_only,
                episodes=valid_indices,
            )

        elif mode == "sample" and episodes is not None:
            super().__init__(
                repo_id=repo_id,
                root=root,
                local_files_only=local_files_only,
                episodes=episodes,
            )

        elif mode == "percent" and percent is not None:
            assert 0 < percent <= 1, "Percent should be a value between 0 and 1."

            # Load full dataset first
            super().__init__(
                repo_id=repo_id,
                root=root,
                local_files_only=local_files_only,
                episodes=train_indices,
            )

            # Sample a percentage of frames
            total_frames = len(self)
            num_sampled_frames = int(percent * total_frames)
            self.sampled_indices = sorted(
                random.sample(range(total_frames), num_sampled_frames)
            )

        else:
            super().__init__(
                repo_id=repo_id, root=root, local_files_only=local_files_only
            )

    def __len__(self):
        """Return the total number of sampled frames if in 'percent' mode, otherwise the full dataset size."""
        if self.sampled_indices is not None:
            return len(self.sampled_indices)
        return super().__len__()

    def __getitem__(self, idx):
        """Fetch frames based on sampled indices in 'percent' mode, otherwise default to full dataset."""
        if self.sampled_indices is not None:
            idx = self.sampled_indices[idx]  # Map index to sampled frames
        item = super().__getitem__(idx)

        if self.use_task_string:
            item["high_level_language_prompt"] = self.task_string

        if self.slow_down_ac_keys and self.slow_down_factor > 1.0:
            for key in self.slow_down_ac_keys:
                if key in item:
                    rot_spec = self.slow_down_rot_specs.get(key, None)
                    item[key] = self._slow_down_sequence(item[key], rot_spec=rot_spec)
        return item

    def _slow_down_sequence(self, seq, rot_spec=None):
        """
        Slow down a sequence of shape (S, D) along the time dimension S.

        - S: time steps
        - D: feature dimension, with any rotation sub-blocks living in slices
             along D (e.g., [:, 0:4] for quats, [:, 3:6] for ypr).

        Steps:
        1. Take first S / slow_down_factor steps (shortened trajectory).
        2. Linearly upsample back to length S.
        3. For any rotation slices specified in rot_spec, overwrite the
           linearly interpolated slices with SLERP-based interpolation.
        """
        alpha = self.slow_down_factor
        if alpha is None or alpha <= 1.0:  # no-op
            return seq

        if seq.ndim != 2:
            raise ValueError(
                f"_slow_down_sequence expects seq of shape (S, D). "
                f"Got shape {seq.shape} with dim={seq.ndim}"
            )

        S, D = seq.shape
        S_short = max(1, min(S, int(S / alpha)))

        if S_short == S:
            return seq  # nothing to do

        # Base: linear interpolation over full feature dimension
        seq_short = seq[:S_short]  # (S_short, D)

        x = seq_short.transpose(0, 1).unsqueeze(0)  # (1, D, S_short)
        x_interp = F.interpolate(
            x, size=S, mode="linear", align_corners=True
        )  # (1, D, S)
        out = x_interp.squeeze(0).transpose(0, 1)  # (S, D)

        # If we have rotation specs, overwrite specified feature slices with SLERP output
        if rot_spec is not None:
            rot_type, index_ranges = rot_spec

            for (start, end) in index_ranges:
                if not (0 <= start < end <= D):
                    raise ValueError(
                        f"Invalid rotation slice [{start}:{end}] for seq with D={D}"
                    )

                rot_short = seq_short[:, start:end]  # (S_short, k)
                k = end - start

                if rot_type == "quat_wxyz":
                    if k != 4:
                        raise ValueError(
                            f"quat slice must have length 4, got {k} for slice [{start}:{end}]"
                        )
                    rot_interp = _slow_down_slerp_quat(rot_short, S)  # (S, 4)
                    out[:, start:end] = rot_interp

                elif rot_type == "ypr":
                    if k != 3:
                        raise ValueError(
                            f"ypr slice must have length 3, got {k} for slice [{start}:{end}]"
                        )
                    # ypr -> quat -> slerp -> ypr
                    quat_short = _ypr_to_quat(rot_short)                # (S_short, 4)
                    quat_interp = _slow_down_slerp_quat(quat_short, S)  # (S, 4)
                    ypr_interp = _quat_to_ypr(quat_interp)              # (S, 3)
                    out[:, start:end] = ypr_interp
                else:
                    raise ValueError(f"Unknown rotation type: {rot_type}")

        return out
        
class MultiRLDBDataset(torch.utils.data.Dataset):
    
    def __init__(self, datasets, embodiment, key_map=None):
        self.datasets = datasets
        self.key_map = key_map

        self.embodiment = get_embodiment_id(embodiment)
        for dataset_name, dataset in self.datasets.items():
            assert dataset.embodiment == self.embodiment, (
                f"Dataset {dataset_name} has embodiment {dataset.embodiment}, expected {self.embodiment}."
            )

        self.index_map = []
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_name, local_idx))

        self.hf_dataset = self._merge_hf_datasets()

        super().__init__()

    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_map[idx]
        data = self.datasets[dataset_name][local_idx]

        if self.key_map and dataset_name in self.key_map:
            key_map = self.key_map[dataset_name]
            data = {key_map.get(k, k): v for k, v in data.items()}

        return data

    def __len__(self):
        return len(self.index_map)

    def _merge_hf_datasets(self):
        """
        Merge hf_dataset from multiple RLDBDataset instances while remapping keys.

        Returns:
            A unified Hugging Face Dataset object.
        """
        dataset_list = []

        for dataset_name, sub_dataset in self.datasets.items():
            hf_dataset = sub_dataset.hf_dataset  # This is a Hugging Face Dataset

            # Apply key mapping if available
            if self.key_map and dataset_name in self.key_map:
                key_map = self.key_map[dataset_name]
                hf_dataset = hf_dataset.rename_columns(key_map)

            dataset_list.append(hf_dataset)

        merged_dataset = concatenate_datasets(dataset_list)

        return merged_dataset


# TODO: add S3 mode where it directly downloads dataset folder from S3
class FolderRLDBDataset(MultiRLDBDataset):
    def __init__(
        self,
        folder_path,
        embodiment,
        mode="train",
        percent=0.1,
        local_files_only=True,
        key_map=None,
        valid_ratio=0.2,
        **kwargs,
    ):
        folder_path = Path(folder_path)
        assert folder_path.is_dir(), f"{folder_path} is not a valid directory."
        assert mode in ["train", "valid", "percent", "total"], f"Invalid mode: {mode}"
        assert embodiment is not None, "embodiment should not be None"

        datasets = {}
        skipped = []

        subdirs = sorted([p for p in folder_path.iterdir() if p.is_dir()])
        logger.info(
            f"Found {len(subdirs)} subfolders. Attempting to load valid RLDB datasets..."
        )

        for subdir in subdirs:
            info_json = subdir / "meta" / "info.json"
            if not info_json.exists():
                logger.warning(f"Skipping {subdir.name}: missing meta/info.json")
                skipped.append(subdir.name)
                continue

            try:
                repo_id = subdir.name
                dataset = RLDBDataset(
                    repo_id=repo_id,
                    root=subdir,
                    local_files_only=local_files_only,
                    mode=mode,
                    percent=percent,
                    valid_ratio=valid_ratio,
                    **kwargs,
                )
                if dataset.embodiment != get_embodiment_id(embodiment):
                    logger.warning(
                        f"Skipping {repo_id}: embodiment mismatch {dataset.embodiment} != {embodiment}"
                    )
                    skipped.append(repo_id)
                    continue

                datasets[repo_id] = dataset
                logger.info(f"Loaded: {repo_id}")

            except Exception as e:
                logger.error(f"Failed to load {subdir.name}: {e}")
                skipped.append(subdir.name)
        assert len(datasets) > 0, "No valid RLDB datasets found!"

        key_map_per_dataset = (
            {repo_id: key_map for repo_id in datasets} if key_map else None
        )

        super().__init__(
            datasets=datasets,
            embodiment=embodiment,
            key_map=key_map_per_dataset,
        )

        if skipped:
            logger.warning(f"Skipped {len(skipped)} datasets: {skipped}")


class S3RLDBDataset(MultiRLDBDataset):
    """
    A dataset class that downloads datasets from AWS S3 and instantiates them as RLDBDataset objects

    Args:
        embodiment (str): The embodiment type (e.g., "EVE_RIGHT_ARM", "ARIA_BIMANUAL").
        mode (str): Dataset mode - "train", "valid".
        bucket_name (str): AWS S3 bucket name containing the datasets.
        main_prefix (str): S3 prefix path to datasets.
        percent (float): fraction of data to use.
        local_files_only (bool): Whether to use only local files.
        key_map (dict): Optional mapping to rename dataset keys.
        valid_ratio (float): Validation split ratio for train/valid split
        temp_root (str): Absolute path for temporary download storage.
                        e.g.: "/coc/cedarp-dxu345-0/datasets/egoverse"
        filters (dict): Filtering criteria for S3 datasets. e.g.,:
                       {"task": "fold_cloth/", "lab": "eth", "scene": "scene_10", "recording": "recording_1"}
                       Only datasets matching ALL non-empty filter values will be loaded.

    Example:
        # Download and load fold_cloth datasets from ETH lab, scene_10, recording_1
        dataset = S3RLDBDataset(
            embodiment="EVE_RIGHT_ARM",
            mode="train",
            filters={"task": "fold_cloth/", "lab": "eth", "scene": "scene_10", "recording": "recording_1"}
        )
    """

    def __init__(
        self,
        embodiment,
        mode,
        bucket_name="rldb",
        main_prefix="processed_v2",
        percent=0.1,
        local_files_only=True,
        key_map=None,
        valid_ratio=0.2,
        temp_root="/coc/flash7/scratch/egoverseS3Dataset",  # "/coc/flash7/scratch/rldb_temp"
        filters={},
        **kwargs,
    ):
        temp_root += "/S3_rldb_data"
        filters["robot_name"] = embodiment
        filters["is_deleted"] = False

        if temp_root[0] != "/":
            temp_root = "/" + temp_root
        temp_root = Path(temp_root)

        if temp_root.is_dir():
            logger.info(f"Using existing temp_root directory: {temp_root}")
        if not temp_root.is_dir():
            temp_root.mkdir()

        logger.info(f"Filters: {filters}")
        datasets = {}
        skipped = []
        
        filtered_paths = self.sync_from_filters(
            bucket_name=bucket_name,
            filters=filters,
            local_dir=temp_root,
        )

        search_path = temp_root

        valid_collection_names = set()
        for _, hashes in filtered_paths:
            valid_collection_names.add(hashes)

        for collection_path in sorted(search_path.iterdir()):
            if not collection_path.is_dir():
                continue

            if collection_path.name not in valid_collection_names:
                logger.warning(
                    f"Skipping {collection_path.name}: not in filtered S3 paths"
                )
                skipped.append(collection_path.name)
                continue
                    
            try:
                repo_id = collection_path.name
                dataset = RLDBDataset(
                    repo_id=repo_id,
                    root=collection_path,
                    local_files_only=local_files_only,
                    mode="total",
                    percent=percent,
                    valid_ratio=valid_ratio,
                    **kwargs,
                )

                if dataset.embodiment != get_embodiment_id(embodiment):
                    logger.warning(
                        f"Skipping {repo_id}: embodiment mismatch {dataset.embodiment} != {embodiment}"
                    )
                    skipped.append(repo_id)
                    continue

                datasets[repo_id] = dataset

            except Exception as e:
                logger.error(f"Failed to load {repo_id} as RLDBDataset: {e}")
                skipped.append(repo_id)
        assert datasets, "No valid RLDB datasets found! Check your S3 path and filters."
        
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

        key_map_per_dataset = (
            {repo_id: key_map for repo_id in datasets} if key_map else None
        )

        super().__init__(
            datasets=datasets,
            embodiment=embodiment,
            key_map=key_map_per_dataset,
        )

        if skipped:
            logger.warning(f"Skipped {len(skipped)} datasets: {skipped}")
            
    @staticmethod
    def _get_processed_path(filters):
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        series = pd.Series(filters)

        output = df.loc[
            (df[list(filters)] == series).all(axis=1),
            ["processed_path", "episode_hash"],
        ]
        skipped = df[df["processed_path"].isnull()]["episode_hash"].tolist()
        logger.info(f"Skipped {len(skipped)} episodes with null processed_path: {skipped}")
        output = output[~output["episode_hash"].isin(skipped)]

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @staticmethod
    def _download_files(bucket_name, s3_prefix, local_dir):
        """
        Downloads all files from a specific S3 prefix to a local directory.

        This method lists all objects under the given S3 prefix and downloads
        each file to the local directory, preserving just the filename (not
        the full S3 path structure).

        Args:
            bucket_name (str): The AWS S3 bucket name
            s3_prefix (str): The S3 prefix path to download from (e.g., "processed/fold_cloth/dataset1/meta/")
            local_dir (Path): The local directory to save files to
        """
        s3 = boto3.client("s3")

        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
        objects = response.get("Contents", [])

        if not objects:
            logger.warning(f"No objects found for prefix: {s3_prefix}")
            return

        for obj in objects:
            key = obj["Key"]

            if key.endswith("/"):
                logger.debug(f"Skipping directory: {key}")
                continue

            if key == s3_prefix or key == s3_prefix.rstrip("/"):
                logger.debug(f"Skipping prefix path: {key}")
                continue

            try:
                local_file_path = local_dir / Path(key).name
                s3.download_file(bucket_name, key, str(local_file_path))
                logger.debug(f"Successfully downloaded: {key}")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")

    @classmethod
    def _sync_s3_to_local(cls, bucket_name, s3_paths, local_dir):
        if not s3_paths:
            return

        folders = []
        for folder, _ in s3_paths:
            if folder.startswith("s3://"):
                folder = urlparse(folder).path.lstrip("/")
            else:
                folder = folder.lstrip("/")
            if not folder.endswith("/"):
                folder += "/"
            folders.append(folder)

        groups = defaultdict(list)
        for f in folders:
            episode_dir = f.rstrip("/")                    # processed_v2/eva/<hash>
            episode_parent = os.path.dirname(episode_dir)  # processed_v2/eva
            root_prefix = episode_parent + "/"             # processed_v2/eva/
            groups[root_prefix].append(f)

        for root_prefix, group_folders in groups.items():
            include_args = []
            for f in group_folders:
                rel = f[len(root_prefix):]                # <hash>/
                include_args += ["--include", f"{rel}*"]   # <hash>/*
            
            cmd = [
                "aws", "s3", "sync",
                f"s3://{bucket_name}/{root_prefix}",
                str(local_dir),
                "--exclude", "*",
                "--only-show-errors",
            ] + include_args

            subprocess.run(cmd, check=True)
    
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
        filtered_paths = cls._get_processed_path(filters)
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
class DataSchematic(object):
    def __init__(self, schematic_dict, viz_img_key, norm_mode="zscore"):
        """
        Initialize with a schematic dictionary and create a DataFrame.

        Args:
            schematic_dict:
                {embodiment_name}:
                    front_img_1_line:
                        key_type: camera_keys
                        lerobot_key: observations.images.cam_high
                    right_wrist_img:
                        key_type: camera_keys
                        lerobot_key: observations.images.right_wrist
                    joint_positions:
                        key_type: proprio_keys
                        lerobot_key: observations.qpos
                    actions_joints_act:
                        key_type: action_keys
                        lerobot_key: actions.joints_act
                    .
                    .
                    .
                    .

        Attributes:
            df (pd.DataFrame): Columns include 'key_name', 'key_type', and 'shape', 'embodiment'.
        """

        rows = []
        self.embodiments = set()

        for embodiment, schematic in schematic_dict.items():
            embodiment_id = get_embodiment_id(embodiment)
            self.embodiments.add(embodiment_id)
            for key_name, key_info in schematic.items():
                rows.append(
                    {
                        "key_name": key_name,
                        "key_type": key_info["key_type"],
                        "lerobot_key": key_info["lerobot_key"],
                        "shape": None,
                        "embodiment": embodiment_id,
                    }
                )

        self.df = pd.DataFrame(rows)
        self._viz_img_key = {get_embodiment_id(k): v for k, v in viz_img_key.items()}
        self.shapes_infered = False
        self.norm_mode = norm_mode
        self.norm_stats = {emb: {} for emb in self.embodiments}

    def lerobot_key_to_keyname(self, lerobot_key, embodiment):
        """
        Get the key name from the Lerobot key.

        Args:
            lerobot_key (str): Lerobot key, e.g., "observations.images.cam_high".
            embodiment (int): int id corresponding to embodiment


        Returns:
            str: Key name, e.g., "front_img_1_line".
        """
        df_filtered = self.df[
            (self.df["lerobot_key"] == lerobot_key)
            & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            return None
        return df_filtered["key_name"].item()

    def keyname_to_lerobot_key(self, key_name, embodiment):
        """
        Get the Lerobot key from the key name.

        Args:
            key_name (str): Key name, e.g., "front_img_1_line".
            embodiment (int): int id corresponding to embodiment

        Returns:
            str: Lerobot key, e.g., "observations.images.cam_high".
        """
        df_filtered = self.df[
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            return None
        return df_filtered["lerobot_key"].item()

    def infer_shapes_from_batch(self, batch):
        """
        Update shapes in the DataFrame based on a batch.

        Args:
            batch (dict): Maps key names (str) to tensors with shapes, e.g.,
                {"key": tensor of shape (3, 480, 640, 3)}.

        Updates:
            The 'shape' column in the DataFrame is updated to match the inferred shapes (stored as tuples).
        """
        embodiment_id = int(batch["metadata.embodiment"])
        for key, tensor in batch.items():
            if hasattr(tensor, "shape"):
                shape = tuple(tensor.shape)
            elif isinstance(tensor, int):
                shape = (1,)
            else:
                shape = None
            key = self.lerobot_key_to_keyname(key, embodiment_id)
            if key in self.df["key_name"].values:
                self.df.loc[self.df["key_name"] == key, "shape"] = str(shape)

        self.shapes_infered = True

    def infer_norm_from_dataset(self, dataset):
        """
        dataset: huggingface dataset backed by pyarrow
        returns: dictionary of means and stds for proprio and action keys
        """
        norm_columns = []

        embodiment = dataset.embodiment

        norm_columns.extend(self.keys_of_type("proprio_keys"))
        norm_columns.extend(self.keys_of_type("action_keys"))

        logger.info(
            f"[NormStats] Starting norm inference for embodiment={embodiment}, "
            f"{len(norm_columns)} columns"
        )

        for column in norm_columns:
            if not self.is_key_with_embodiment(column, embodiment):
                continue

            column_name = self.keyname_to_lerobot_key(column, embodiment)
            logger.info(f"[NormStats] Processing column={column_name}")

            # Arrow → NumPy (fast path, preserves shape)
            column_data = (
                dataset.hf_dataset
                .with_format("numpy", columns=[column_name])[:][column_name]
            )

            if column_data.ndim not in (2, 3):
                raise ValueError(
                    f"Column {column} has shape {column_data.shape}, "
                    "expected 2 or 3 dims"
                )

            mean = np.mean(column_data, axis=0)
            std = np.std(column_data, axis=0)
            minv = np.min(column_data, axis=0)
            maxv = np.max(column_data, axis=0)
            median = np.median(column_data, axis=0)
            q1 = np.percentile(column_data, 1, axis=0)
            q99 = np.percentile(column_data, 99, axis=0)

            self.norm_stats[embodiment][column] = {
                "mean": torch.from_numpy(mean).float(),
                "std": torch.from_numpy(std).float(),
                "min": torch.from_numpy(minv).float(),
                "max": torch.from_numpy(maxv).float(),
                "median": torch.from_numpy(median).float(),
                "quantile_1": torch.from_numpy(q1).float(),
                "quantile_99": torch.from_numpy(q99).float(),
            }

        logger.info("[NormStats] Finished norm inference")


    def viz_img_key(self):
        """
        Get the key that should be used for offline visualization
        """
        return self._viz_img_key

    def all_keys(self):
        """
        Get all key names.

        Returns:
            list: Key names (str).
        """
        return self.df["key_name"].tolist()

    def is_key_with_embodiment(self, key_name, embodiment):
        """
        Check if a key_name exists with a given embodiment

        Args:
            key_name (str): name of key, e.g. actions_joints
            embodiment (int): integer id of embodiment

        Returns:
            bool: if the key exists.
        """
        return (
            (self.df["key_name"] == key_name) & (self.df["embodiment"] == embodiment)
        ).any()

    def keys_of_type(self, key_type):
        """
        Get keys of a specific type.

        Args:
            key_type (str): Type of keys, e.g., "camera_keys", "proprio_keys", "action_keys", "metadata_keys".

        Returns:
            list: Key names (str) of the given type.
        """
        return self.df[self.df["key_type"] == key_type]["key_name"].tolist()

    def action_keys(self):
        return self.keys_of_type("action_keys")

    def key_shape(self, key, embodiment):
        """
        Get the shape of a specific key.

        Args:
            key (str): Name of the key.
            embodiment (int): integer id of embodiment

        Returns:
            tuple or None: Shape as a tuple, or None if not found.
        """
        if key not in self.df["key_name"].values:
            raise ValueError(f"Keyname '{key}' is not in the schematic")

        df_filtered = self.df[
            (self.df["key_name"] == key) & (self.df["embodiment"] == embodiment)
        ]

        if df_filtered.empty:
            raise ValueError(f"Keyname '{key}' with embodiment {embodiment} not found.")

        shape = df_filtered["shape"].item()
        return ast.literal_eval(shape)

    def normalize_data(self, data, embodiment):
        """
        Normalize data using the stored normalization statistics.

        Args:
            data (dict): Maps key names to tensors.
                joint_positions: tensor of shape (B, S, 7)
            embodiment (int): Id of the embodiment.

        Returns:
            dict: Maps key names to normalized tensors.
        """
        if self.norm_stats is None:
            raise ValueError(
                "Normalization statistics not set. Call infer_norm_from_dataset() first."
            )

        norm_data = {}
        for key, tensor in data.items():
            if key in self.keys_of_type("proprio_keys") or key in self.keys_of_type(
                "action_keys"
            ):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                if self.norm_mode == "zscore":
                    mean = stats["mean"].to(tensor.device)
                    std = stats["std"].to(tensor.device)
                    norm_data[key] = (tensor - mean) / (std + 1e-6)
                elif self.norm_mode == "minmax":
                    min = stats["min"].to(tensor.device)
                    max = stats["max"].to(tensor.device)
                    ndata = (tensor - min) / (max - min + 1e-6)
                    norm_data[key] = 2.0 * ndata - 1.0
                elif self.norm_mode == "quantile":
                    quantile_1 = stats["quantile_1"].to(tensor.device)
                    quantile_99 = stats["quantile_99"].to(tensor.device)
                    ndata = (tensor - quantile_1) / (quantile_99 - quantile_1 + 1e-6)
                    norm_data[key] = 2.0 * ndata - 1.0
                else:
                    raise ValueError(f"Invalid normalization mode: {self.norm_mode}")
            else:
                norm_data[key] = tensor

        return norm_data

    def unnormalize_data(self, data, embodiment):
        """
        Unnormalize data using the stored normalization statistics.

        Args:
            data (dict): Maps key names to tensors.
                joint_positions: tensor of shape (B, S, 7)
            embodiment (int): Id of the embodiment.

        Returns:
            dict: Maps key names to denormalized tensors.
        """
        if self.norm_stats is None:
            raise ValueError(
                "Normalization statistics not set. Call infer_norm_from_dataset() first."
            )

        denorm_data = {}
        for key, tensor in data.items():
            if key in self.keys_of_type("proprio_keys") or key in self.keys_of_type(
                "action_keys"
            ):
                if (
                    embodiment not in self.norm_stats
                    or key not in self.norm_stats[embodiment]
                ):
                    raise ValueError(
                        f"Missing normalization stats for key {key} and embodiment {embodiment}."
                    )

                stats = self.norm_stats[embodiment][key]
                if self.norm_mode == "zscore":
                    mean = stats["mean"].to(tensor.device)
                    std = stats["std"].to(tensor.device)
                    denorm_data[key] = tensor * (std + 1e-6) + mean

                elif self.norm_mode == "minmax":
                    min_val = stats["min"].to(tensor.device)
                    max_val = stats["max"].to(tensor.device)
                    denorm_data[key] = (tensor + 1) * 0.5 * (
                        max_val - min_val + 1e-6
                    ) + min_val

                elif self.norm_mode == "quantile":
                    quantile_1 = stats["quantile_1"].to(tensor.device)
                    quantile_99 = stats["quantile_99"].to(tensor.device)
                    denorm_data[key] = (tensor + 1) * 0.5 * (
                        quantile_99 - quantile_1 + 1e-6
                    ) + quantile_1

                else:
                    raise ValueError(f"Invalid normalization mode: {self.norm_mode}")
            else:
                denorm_data[key] = tensor

        return denorm_data
