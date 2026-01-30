import ast
import logging
import os
import random
import subprocess
import tempfile
import traceback
import psutil

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Sequence, Dict, Set, Tuple, Optional

import boto3
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import huggingface_hub
import datasets.config as ds_cfg
from datasets import DatasetDict, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

# Local EgoMimic Imports
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)
from egomimic.rldb.data_utils import (
    _ypr_to_quat,
    _slerp,
    _quat_to_ypr,
    _slow_down_slerp_quat,
)

# Logging Setup
logger = logging.getLogger(__name__)
disable_progress_bar()
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub._snapshot_download").setLevel(logging.ERROR)


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
        use_annotations=False,
        **kwargs,
    ):
        logger.info(f"Loading RLDB dataset from {root}")
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
            if not (
                isinstance(v, Sequence)
                and not isinstance(v, (str, bytes))
                and len(v) == 2
            ):
                raise ValueError(
                    f"slow_down_rot_specs['{k}'] must be (rot_type, index_ranges), got {type(v)} with value {v}"
                )

            rot_type, ranges = v

            if rot_type not in ("quat_wxyz", "ypr"):
                raise ValueError(
                    f"Rotation type for key '{k}' must be 'quat_wxyz' or 'ypr', got {rot_type}"
                )

            if not (
                isinstance(ranges, Sequence) and not isinstance(ranges, (str, bytes))
            ):
                raise ValueError(
                    f"Index ranges for slow_down_rot_specs['{k}'] must be a sequence of (start, end) pairs, got {type(ranges)}"
                )

            for pair in ranges:
                if not (
                    isinstance(pair, Sequence)
                    and not isinstance(pair, (str, bytes))
                    and len(pair) == 2
                ):
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

        annotation_path = Path(root) / "annotations"
        if annotation_path.is_dir() and use_annotations:
            self.annotations = AnnotationLoader(root=root)
            self.annotation_df = self.annotations.df
        else:
            self.annotations = None
            self.annotation_df = None

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
                    item[key] = self._slow_down_sequence(item[key])

        ep_idx = int(item["episode_index"])
        frame_idx = (
            self.sampled_indices[idx] if self.sampled_indices is not None else idx
        )

        frame_item = self.hf_dataset[frame_idx]
        frame_time = float(frame_item["timestamp"])

        if self._get_frame_annotation is not None:
            frame_item["annotations"] = self._get_frame_annotation(
                episode_idx=ep_idx,
                frame_time=frame_time,
            )

        return frame_item

    def _get_frame_annotation(
        self,
        episode_idx: int,
        frame_time: float,
    ) -> str:
        """
        Return the annotation string for a given episode index and timestamp.
        Returns empty string if annotations are unavailable or no match is found.
        """
        if self.annotation_df is None:
            return ""

        df_episode = self.annotation_df.loc[
            self.annotation_df["idx"].astype(int) == episode_idx
        ]

        if df_episode.empty:
            return ""

        # Active annotation
        active = df_episode[
            (df_episode["start_time"] <= frame_time)
            & (df_episode["end_time"] >= frame_time)
        ]

        if not active.empty:
            return active["Labels"].iloc[0]

        # Fallback: previous annotation
        future = df_episode[df_episode["start_time"] > frame_time]
        if future.empty:
            return df_episode.tail(1)["Labels"].iloc[0]

        next_pos = df_episode.index.get_loc(future.index[0])
        prev_pos = next_pos - 1
        if prev_pos >= 0:
            return df_episode.iloc[prev_pos]["Labels"]

        return ""

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

            for start, end in index_ranges:
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
                    quat_short = _ypr_to_quat(rot_short)  # (S_short, 4)
                    quat_interp = _slow_down_slerp_quat(quat_short, S)  # (S, 4)
                    ypr_interp = _quat_to_ypr(quat_interp)  # (S, 3)
                    out[:, start:end] = ypr_interp
                else:
                    raise ValueError(f"Unknown rotation type: {rot_type}")

        return out


class AnnotationLoader:
    df = None

    def __init__(self, root):
        root = Path(root)
        self.annotation_path = root / "annotations"

        if not self.annotation_path.is_dir():
            raise ValueError(f"Annotation {self.annotation_path} path does not exist.")

        self.df = self.load_annotations()

    def load_annotations(self):
        frames = []
        for file in sorted(self.annotation_path.iterdir()):
            if not file.is_file():
                continue

            temp_df = pd.read_csv(file)
            parts = file.name.split("_")
            temp_df["idx"] = parts[1]
            frames.append(temp_df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
        
        try:
            merged_dataset = concatenate_datasets(dataset_list)
        except Exception as e:
            logger.error(f"Failed to merge datasets: {e}")
            return None

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
                expected_embodiment_id = get_embodiment_id(embodiment)
                if dataset.embodiment != expected_embodiment_id:
                    dataset_emb_name = EMBODIMENT_ID_TO_KEY.get(
                        dataset.embodiment, f"unknown({dataset.embodiment})"
                    )
                    expected_emb_name = EMBODIMENT_ID_TO_KEY.get(
                        expected_embodiment_id, f"unknown({expected_embodiment_id})"
                    )
                    logger.warning(
                        f"Skipping {repo_id}: embodiment mismatch {dataset_emb_name} ({dataset.embodiment}) != {expected_emb_name} ({expected_embodiment_id})"
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


_PROCESS = psutil.Process()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}PB"


def _log_mem(tag: str):
    """Logs CPU + (optional) CUDA memory safely."""
    try:
        mi = _PROCESS.memory_info()
        msg = f"[MEM] {tag} | RSS={_fmt_bytes(mi.rss)} | VMS={_fmt_bytes(mi.vms)}"

        if torch.cuda.is_available():
            msg += (
                f" | CUDA_alloc={_fmt_bytes(torch.cuda.memory_allocated())}"
                f" | CUDA_reserved={_fmt_bytes(torch.cuda.memory_reserved())}"
            )
        logger.info(msg)
    except Exception as e:
        logger.warning(f"[MEM] failed at {tag}: {e}")


class S3RLDBDataset(MultiRLDBDataset):
    def __init__(
        self,
        embodiment,
        mode,
        local_mode=False,
        use_annotations=False,
        bucket_name="rldb",
        main_prefix="processed_v2",
        sample_percent=1.0,  # Samples % of total available collection paths
        percent=1.0,  # Samples % of frames within each loaded RLDBDataset
        local_files_only=True,
        key_map=None,
        valid_ratio=0.2,
        temp_root="/storage/project/r-dxu345-0/shared/egoverse_datasets/S3_rldb_data/S3_rldb_data",  # "/coc/flash7/scratch/rldb_temp"
        cache_root="/storage/project/r-dxu345-0/shared/.cache",
        filters={},
        debug=False,
        **kwargs,
    ):
        _log_mem("init:start")
        logger.info(f"Summary Dataset and S3RLDBDataset instantiation: {filters} {embodiment} {mode} {local_mode} {use_annotations} {bucket_name} {main_prefix} {sample_percent} {percent} {local_files_only} {key_map} {valid_ratio} {temp_root} {cache_root} {debug} {kwargs}")
        filters = filters or {}
        filters["robot_name"] = embodiment
        filters["is_deleted"] = False

        # Environment & Cache Setup
        os.environ["HF_HOME"] = cache_root
        os.environ["HF_DATASETS_CACHE"] = f"{cache_root}/datasets"
        ds_cfg.HF_DATASETS_CACHE = os.environ["HF_DATASETS_CACHE"]
        huggingface_hub.constants.HF_HOME = os.environ["HF_HOME"]

        temp_root_path = Path(temp_root)
        temp_root_path.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # 1. Query Metadata & Subsample FIRST (No S3 Sync yet)
        # ------------------------------------------------------------
        _log_mem("init:query_metadata")
        all_filtered_paths = self._get_processed_path(filters)

        if not all_filtered_paths:
            raise ValueError(f"No episodes matched filters: {filters}")

        if local_mode:
            local_paths = []
            for ep in all_filtered_paths:
                if self._episode_already_present(temp_root_path, ep[1]):
                    local_paths.append(ep)
            all_filtered_paths = local_paths
        # Deterministic shuffle for disjoint train/valid splits
        random.Random(6).shuffle(all_filtered_paths)

        # Apply sample_percent to the list of paths
        total_to_sample = (
            max(1, int(len(all_filtered_paths) * sample_percent))
            if sample_percent > 0
            else 0
        )
        sampled_paths = all_filtered_paths[:total_to_sample]
        logger.info(f"Sampled {len(sampled_paths)} paths from {len(all_filtered_paths)} total paths.")

        # Split into disjoint subsets
        num_valid = int(len(sampled_paths) * valid_ratio)
        valid_paths_subset = sampled_paths[:num_valid]
        train_paths_subset = sampled_paths[num_valid:]

        if len(valid_paths_subset) == 0 or len(train_paths_subset) == 0:
            raise ValueError(f"Not enough paths to split for mode: {mode}. Valid paths: {valid_paths_subset}, Train paths: {train_paths_subset}")

        # Select which subset we actually need to download and load
        if mode == "train":
            paths_to_process = train_paths_subset
        elif mode == "valid":
            paths_to_process = valid_paths_subset
        elif mode in ["total", "percent"]:
            paths_to_process = sampled_paths
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ------------------------------------------------------------
        # 2. Sync ONLY the required paths from S3
        # ------------------------------------------------------------
        _log_mem("init:before_sync")
        logger.info(
            f"Syncing {len(paths_to_process)} sampled episodes for mode '{mode}' to {temp_root_path}"
        )

        if mode == "local":
            pass
        else:
            self._sync_s3_to_local(
                bucket_name=bucket_name,
                s3_paths=paths_to_process,
                local_dir=temp_root_path,
            )
            _log_mem("init:after_sync")

        # ------------------------------------------------------------
        # 3. Parallel Load
        # ------------------------------------------------------------
        valid_collection_names = {h for _, h in paths_to_process}
        max_workers = int(os.environ.get("RLDB_LOAD_WORKERS", "10"))

        _log_mem("init:before_parallel_load")
        datasets, skipped = self._load_rldb_datasets_parallel(
            search_path=temp_root_path,
            embodiment=embodiment,
            valid_collection_names=valid_collection_names,
            local_files_only=local_files_only,
            percent=percent,  # Frame-level sampling
            valid_ratio=valid_ratio,
            max_workers=max_workers,
            debug=debug,
            use_annotations=use_annotations,
            kwargs=kwargs,
        )
        _log_mem("init:after_parallel_load")

        if datasets is None:
            raise ValueError("No datasets loaded during parallel load.")

        key_map = {repo_id: key_map for repo_id in datasets} if key_map else None

        super().__init__(
            datasets=datasets,
            embodiment=embodiment,
            key_map=key_map,
        )

        if skipped:
            logger.warning(f"Skipped {len(skipped)} datasets during parallel load.")
        _log_mem("init:done")

    @classmethod
    def _load_rldb_dataset_one(
        cls,
        *,
        collection_path: Path,
        embodiment: str,
        local_files_only: bool,
        percent: float,
        valid_ratio: float,
        kwargs: dict,
        use_annotations: bool,
    ):
        repo_id = collection_path.name

        if not collection_path.is_dir():
            return repo_id, None, "not_a_dir", None

        try:
            ds_obj = RLDBDataset(
                repo_id=repo_id,
                root=collection_path,
                local_files_only=local_files_only,
                mode="total",
                percent=percent,
                valid_ratio=valid_ratio,
                use_annotations=use_annotations,
                **kwargs,
            )

            expected = get_embodiment_id(embodiment)
            if ds_obj.embodiment != expected:
                return (
                    repo_id,
                    None,
                    "embodiment_mismatch",
                    f"{ds_obj.embodiment} != {expected}",
                )

            return repo_id, ds_obj, None, None

        except Exception as e:
            return (
                repo_id,
                None,
                "exception",
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )


    @classmethod
    def _load_rldb_datasets_parallel(
        cls,
        *,
        search_path: Path,
        embodiment: str,
        valid_collection_names: set[str],
        local_files_only: bool,
        percent: float,
        valid_ratio: float,
        max_workers: int,
        debug: bool = False,
        use_annotations: bool,
        kwargs: dict,
        batch_size: int = 10,
    ):
        _log_mem("parallel_load:start")
        max_workers = max(1, int(max_workers))

        if debug:
            logger.info("Debug mode: limiting to 10 datasets.")
            valid_collection_names = set(list(valid_collection_names)[:10])

        valid_paths = [
            search_path / name
            for name in valid_collection_names
            if (search_path / name).is_dir()
        ]

        total = len(valid_paths)
        datasets: dict[str, RLDBDataset] = {}
        skipped: list[str] = []

        logger.info(
            f"Starting parallel RLDB load: {total} datasets | workers={max_workers}"
        )

        with (
            tqdm(total=total, desc="Loading RLDBDataset") as dataset_bar,
            tqdm(
                total=1, bar_format="RSS Mem: {bar} {n:.1f}MB", position=1, leave=True
            ) as rss_bar,
            tqdm(
                total=1, bar_format="VMS Mem: {bar} {n:.1f}MB", position=2, leave=True
            ) as vms_bar,
        ):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                in_flight = set()
                path_iter = iter(valid_paths)

                def _submit(p: Path):
                    return executor.submit(
                        cls._load_rldb_dataset_one,
                        collection_path=p,
                        embodiment=embodiment,
                        local_files_only=local_files_only,
                        percent=percent,
                        valid_ratio=valid_ratio,
                        use_annotations=use_annotations,
                        kwargs=kwargs,
                    )

                # Prime pipeline
                for _ in range(min(batch_size, total)):
                    try:
                        in_flight.add(_submit(next(path_iter)))
                    except StopIteration:
                        break

                while in_flight:
                    done, in_flight = concurrent.futures.wait(
                        in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for fut in done:
                        try:
                            repo_id, ds_obj, reason, err = fut.result()
                        except Exception as e:
                            msg = (
                                "[FUTURE FAILURE]\n"
                                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                            )
                            tqdm.write(msg)
                            logger.exception(msg)
                            dataset_bar.update(1)
                            continue

                        if ds_obj:
                            datasets[repo_id] = ds_obj
                        else:
                            msg = (
                                "\n[DATASET SKIPPED]\n"
                                f"repo_id: {repo_id}\n"
                                f"reason: {reason}\n"
                                f"error:\n{err if err else 'None'}\n"
                            )
                            tqdm.write(msg)
                            logger.error(msg)

                            if reason != "not_a_dir":
                                skipped.append(repo_id)

                        dataset_bar.update(1)

                        try:
                            in_flight.add(_submit(next(path_iter)))
                        except StopIteration:
                            pass

                    # Memory monitoring
                    mi = _PROCESS.memory_info()
                    rss_bar.n, vms_bar.n = mi.rss / 1e6, mi.vms / 1e6
                    rss_bar.refresh()
                    vms_bar.refresh()

        _log_mem("parallel_load:end")
        return datasets, skipped


    @staticmethod
    def _get_processed_path(filters):
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        series = pd.Series(filters)

        mask = (df[list(filters)] == series).all(axis=1)
        output = df.loc[mask, ["processed_path", "episode_hash"]].dropna(
            subset=["processed_path"]
        )
        return list(output.itertuples(index=False, name=None))

    @classmethod
    def _sync_s3_to_local(cls, bucket_name, s3_paths, local_dir: Path):
        _log_mem("s3_sync:start")
        to_sync = [
            p for p in s3_paths if not cls._episode_already_present(local_dir, p[1])
        ]
        if not to_sync:
            logger.info("Nothing to sync from S3.")
            return

        local_dir.mkdir(parents=True, exist_ok=True)

        # Create the full batch file as before for maximum s5cmd efficiency
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_", suffix=".txt", delete=False
        ) as f:
            lines = []
            for src_path, episode_hash in to_sync:
                src = (
                    src_path
                    if src_path.startswith("s3://")
                    else f"s3://{bucket_name}/{src_path.lstrip('/')}"
                )
                lines.append(
                    f'sync "{src.rstrip("/")}/*" "{local_dir / episode_hash}/"'
                )
            f.write("\n".join(lines).encode())
            batch_path = Path(f.name)

        logger.info(f"Syncing {len(to_sync)} episodes via s5cmd...")

        # Execute s5cmd and stream output to tqdm
        # We use a simple counter for files synced since we don't know the exact file count per episode easily
        with tqdm(total=len(to_sync), desc="S3 Sync Progress", unit="ep") as pbar:
            process = subprocess.Popen(
                ["s5cmd", "run", str(batch_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # We look for the completion of a folder/sync operation in the logs
            # s5cmd logs individual files, but we'll increment the bar based on unique episode hashes seen
            completed_hashes = set()
            for line in process.stdout:
                # s5cmd output usually contains the destination path
                # We check which episode hash is mentioned in the current log line
                for _, episode_hash in to_sync:
                    if episode_hash in line and episode_hash not in completed_hashes:
                        completed_hashes.add(episode_hash)
                        pbar.update(1)
                        break

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

        batch_path.unlink(missing_ok=True)
        _log_mem("s3_sync:after_s5cmd")

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        ep = local_dir / episode_hash
        meta, chunk0 = ep / "meta", ep / "data" / "chunk-000"
        try:
            return all(d.is_dir() and any(d.iterdir()) for d in [meta, chunk0])
        except (FileNotFoundError, StopIteration):
            return False

    @classmethod
    def sync_from_filters(cls, *, bucket_name: str, filters: dict, local_dir: Path):
        _log_mem("sync_from_filters:start")
        filtered_paths = cls._get_processed_path(filters)
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        logger.info(f"Syncing S3 datasets with filters {filters} to {local_dir}")
        cls._sync_s3_to_local(
            bucket_name=bucket_name, s3_paths=filtered_paths, local_dir=local_dir
        )
        _log_mem("sync_from_filters:end")
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

            memory_usage = psutil.Process().memory_info().rss / (1024**2)
            logger.info(f"Memory usage before column processing: {memory_usage:.2f} MB")

            memory_usage = psutil.Process().memory_info().rss / (1024**2)
            logger.info(f"Memory usage before column processing: {memory_usage:.2f} MB")

            column_name = self.keyname_to_lerobot_key(column, embodiment)
            logger.info(f"[NormStats] Processing column={column_name}")

            # Arrow → NumPy (fast path, preserves shape)
            column_data = dataset.hf_dataset.with_format(
                "numpy", columns=[column_name]
            )[:][column_name]

            memory_usage = psutil.Process().memory_info().rss / (1024**2)
            logger.info(f"Memory usage before mean calculation: {memory_usage:.2f} MB")

            memory_usage = psutil.Process().memory_info().rss / (1024**2)
            logger.info(f"Memory usage before mean calculation: {memory_usage:.2f} MB")

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
