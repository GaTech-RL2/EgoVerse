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

SEED = 42




class EpisodeResolver:
    """
    Filters SQL table for zarr episode paths/ downloads from S3.
    resolve returns processed_path. 
    """
    def __init__(
        self,
        folder_path = "/coc/flash7/scratch/egoverseS3Dataset",
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
            temp_root = self.folder_path + "/S3_rldb_data"
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
                bucket_name= self.bucket_name,
                filters=filters,
                local_dir=temp_root,
            )
            search_path = temp_root

            valid_collection_names = set()
            for _, hashes in filtered_paths:
                valid_collection_names.add(hashes)
            max_workers = int(os.environ.get("RLDB_LOAD_WORKERS", "10"))
            datasets = self._load_zarr_dataset_parallel(
                search_path = search_path, 
                valid_collection_names = valid_collection_names,
                max_workers=max_workers,
            )
            return datasets

        else:
            paths = self._get_processed_path(filters)
            valid_collection_names = set()
            for processed_path, episode_hash in paths:
                valid_collection_names.add(episode_hash)
            if not valid_collection_names:
                raise ValueError(
                    "No valid collection names from _get_processed_path: "
                    "filters matched no episodes in the SQL table."
                )
            max_workers = int(os.environ.get("RLDB_LOAD_WORKERS", "10"))
            datasets = self._load_zarr_dataset_parallel(
                search_path = search_path, 
                valid_collection_names = valid_collection_names,
                max_workers=max_workers,
            )
            return datasets

    @classmethod
    def _load_zarr_dataset_one(
        cls,
        *,
        collection_path: Path,
        valid_collection_names: set[str],
    ):
        """
        Attempt to construct one RLDBDataset from a local folder.


        Returns:
            (repo_id, dataset_or_None, skip_reason_or_None, err_str_or_None)
        """
        repo_id = collection_path.name

        if not collection_path.is_dir():
            return repo_id, None, "not_a_dir", None

        if repo_id not in valid_collection_names:
            return repo_id, None, "not_in_filtered_paths", None

        try:
            ds_obj = ZarrDataset(collection_path)

            #Have not added embodiment logic yet    
            # expected = get_embodiment_id(embodiment)
            # if ds_obj.embodiment != expected:
            #     return (
            #         repo_id,
            #         None,
            #         f"embodiment_mismatch {ds_obj.embodiment} != {expected}",
            #         None,
            #     )

            return repo_id, ds_obj, None, None

        except Exception as e:
            return repo_id, None, "exception", f"{e}\n{traceback.format_exc()}"

    @classmethod
    def _load_zarr_datasets_parallel(
        cls,
        *,
        search_path: Path,
        valid_collection_names: set[str],
        max_workers: int,
    ):
        """
        Parallelize RLDBDataset instantiation over folders in search_path.


        Returns:
            datasets: dict[str, RLDBDataset]
            skipped: list[str]
        """
        all_paths = sorted(search_path.iterdir())
        max_workers = max(1, int(max_workers))

        datasets: dict[str, RLDBDataset] = {}
        skipped: list[str] = []

        def _submit_arg(p: Path):
            return dict(
                collection_path=p,
                valid_collection_names=valid_collection_names,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(cls._load_zarr_dataset_one, **_submit_arg(p))
                for p in all_paths
            ]

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading RLDBDataset",
            ):
                repo_id, ds_obj, reason, err = fut.result()

                if ds_obj is not None:
                    datasets[repo_id] = ds_obj
                    continue

                if reason == "not_a_dir":
                    continue

                skipped.append(repo_id)

                if reason == "not_in_filtered_paths":
                    logger.warning(f"Skipping {repo_id}: not in filtered S3 paths")
                elif reason and reason.startswith("embodiment_mismatch"):
                    logger.warning(f"Skipping {repo_id}: {reason}")
                else:
                    logger.error(f"Failed to load {repo_id} as RLDBDataset:\n{err}")

        return datasets, skipped

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
        logger.info(
            f"Skipped {len(skipped)} episodes with null processed_path: {skipped}"
        )
        output = output[~output["episode_hash"].isin(skipped)]

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @staticmethod
    def _download_files(bucket_name, s3_prefix, local_dir):
        """
        Downloads all files from a specific S3 prefix to a local directory.
        """

    def decode_jpeg(data) -> np.ndarray:
        """Decode JPEG image to numpy array using OpenCV.

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

            local_file_path = local_dir / Path(key).name

            # Check if file already exists and is not empty, solves race condition of multiple processes downloading the same file
            try:
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    logger.debug(
                        f"File already exists, skipping: {key} -> {local_file_path}"
                    )
                    continue

                s3.download_file(bucket_name, key, str(local_file_path))
                logger.debug(f"Successfully downloaded: {key}")
            except FileNotFoundError as e:
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    logger.debug(f"File downloaded by another process, skipping: {key}")
                else:
                    logger.error(f"Failed to download {key}: {e}")
            except Exception as e:
                if local_file_path.exists() and local_file_path.stat().st_size > 0:
                    logger.debug(
                        f"File downloaded by another process after error: {key}"
                    )
                else:
                    logger.error(f"Failed to download {key}: {e}")

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

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        ep = local_dir / episode_hash
        meta = ep / "meta"
        chunk0 = ep / "data" / "chunk-000"

        if not meta.is_dir() or not chunk0.is_dir():
            return False

        try:
            if not any(meta.iterdir()):
                return False
            if not any(chunk0.iterdir()):
                return False
        except FileNotFoundError:
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


class MultiZarrDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiZarr Dataset, can wrap zarr or multi dataset. 
    note: I am not adding embodiments yet because to match would require something beyond current zarr dataset lazy loading
    """
    def __init__(self, datasets, key_map=None):
        """
        Args:
            datasets: either multi or zarr datasets, can mix and match
            key_map (dict, optional): Mapping from source dataset keys to unified keys, if remapping is needed.
        """
        self.datasets = datasets
        self.key_map = key_map

        self.index_map = []
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                self.index_map.append((dataset_name, local_idx))

    def __len__(self) -> int:
        return len(self.index_map)


    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_map[idx]
        data = self.datasets[dataset_name][local_idx]

        if self.key_map and dataset_name in self.key_map:
            key_map = self.key_map[dataset_name]
            data = {key_map.get(k, k): v for k, v in data.items()}

        return data



class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(self, Episode_path:str):
        """
        Args:
            episode_path: just a path to the designated zarr episode
        """
        self.episode_path = Episode_path
        # should probably initialize embodiment here but I'm just lazy loading path and don't want to read yet
        super().__init__()


    def __len__(self) -> int:
        if self.total_frames.exists():
            return self.total_frames
        else:
            json_path = self.episode_path / "zarr.json"
            if json_path.exists():
                with open(json_path, "r") as f:
                    ep_info = json.load(f).get("attributes", {})
                    self.total_frames = ep_info.get("total_frames", 0)


    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_reader = ZarrEpisode(self.episode_path)
        data = episode_reader.read(idx)
        # Squeeze batch dim to single frame
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out



