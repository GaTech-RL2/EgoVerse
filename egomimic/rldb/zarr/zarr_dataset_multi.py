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

import copy
import json
import logging
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr
from tqdm import tqdm

from egomimic.rldb.embodiment.embodiment import get_embodiment_id

# from action_chunk_transforms import Transform
from egomimic.rldb.filters import DatasetFilter
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)

if TYPE_CHECKING:
    # Annotation-only import — avoids a runtime circular import with
    # zarr_dataset_action_expert (which itself imports MultiDataset from here).
    from egomimic.rldb.zarr.zarr_dataset_action_expert import (
        ZarrActionExpertDataset,
    )

logger = logging.getLogger(__name__)

SEED = 42


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


def _ensure_dataset_filter(filters: DatasetFilter | None) -> DatasetFilter:
    if filters is None:
        return DatasetFilter()
    if isinstance(filters, DatasetFilter):
        return filters
    raise TypeError(
        "filters must be a DatasetFilter or None in the zarr resolver path. "
        "Plain dict filters are no longer supported."
    )


def _is_missing_filter_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""

    try:
        missing = pd.isna(value)
    except Exception:
        return False

    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _first_present(*values: object) -> object | None:
    for value in values:
        if not _is_missing_filter_value(value):
            return value
    return None


def _normalize_filter_row(
    row: Mapping[str, Any],
    *,
    episode_hash: str | None = None,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["episode_hash"] = (
        episode_hash if episode_hash is not None else normalized.get("episode_hash")
    )

    if _is_missing_filter_value(normalized.get("is_deleted")):
        normalized["is_deleted"] = False

    robot_name = _first_present(
        normalized.get("robot_name"),
        normalized.get("robot_type"),
        normalized.get("embodiment"),
    )
    if robot_name is not None:
        normalized["robot_name"] = robot_name

    return normalized


def _infer_key_type(key_name: str) -> str | None:
    """
    Heuristically infer a key_type for a post-transform key that wasn't
    declared in the leaf's key_map (typically produced by a transform like
    ``ConcatKeys``). Returns one of the recognised key_type strings or None
    when the key looks like metadata that NormStats shouldn't normalize.

    Recognition is conservative — only positive matches return a type.
    """
    name = key_name.lower()
    # Camera image keys
    if "image" in name or "/img" in name or name.endswith("_img") or ".img" in name:
        return "camera_keys"
    # Action keys (model outputs)
    if name.startswith("actions") or "/cmd_" in name or ".cmd_" in name:
        return "action_keys"
    # Proprioception (observed state)
    if (
        name.startswith("observations.state")
        or "/obs_" in name
        or ".obs_" in name
        or "joint_positions" in name
        or "ee_pose" in name
        or "keypoints" in name
    ):
        return "proprio_keys"
    # Language / annotations
    if "annotation" in name or "tokenized" in name or "lang" in name:
        return "annotation_keys"
    return None


def get_fallback_idx(
    idx: int,
    candidates: Iterable[int],
    _attempts: int | None,
    max_attempts: int,
    exhausted_error: str,
) -> tuple[int, int]:
    attempts = (_attempts or 0) + 1
    valid_candidates = [
        candidate_idx for candidate_idx in candidates if candidate_idx != idx
    ]
    if attempts >= max_attempts or not valid_candidates:
        raise RuntimeError(exhausted_error)
    return random.choice(valid_candidates), attempts


class EpisodeResolver:
    """
    Base class for episode resolution utilities.
    Provides shared static/class helpers; subclasses implement resolve().
    """

    _dataset_class = None  # set to ZarrDataset after that class is defined

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.key_map = key_map
        self.transform_list = transform_list

    def _load_zarr_datasets(self, search_path: Path, valid_folder_names: set[str]):
        """
        Loads multiple Zarr datasets from the specified folder path, filtering only those whose hashes
        are present in the valid_folder_names set.

        Args:
            search_path (Path): The root directory to search for Zarr datasets.
            valid_folder_names (set[str]): A set of valid folder names (episode hashes without ".zarr") to filter datasets.
        Returns:
            dict[str, ZarrDataset]: a dictionary mapping string keys to constructed zarr datasets from valid filters.
        """
        dataset_class = self._dataset_class or ZarrDataset
        all_paths = sorted(search_path.iterdir())
        datasets: dict[str, ZarrDataset] = {}
        skipped: list[str] = []
        for p in all_paths:
            if not p.is_dir():
                logger.info(f"{p} is not a valid directory")
                skipped.append(p.name)
                continue
            name = p.name
            if name.endswith(".zarr"):
                name = name[: -len(".zarr")]
            if name not in valid_folder_names:
                skipped.append(p.name)
                continue
            try:
                ds_obj = dataset_class(
                    p,
                    key_map=self.key_map,
                    transform_list=self.transform_list,
                )
                datasets[name] = ds_obj
            except Exception as e:
                logger.error(f"Failed to load dataset at {p}: {e}")
                skipped.append(p.name)

        return datasets

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        direct = local_dir / episode_hash
        if direct.is_dir():
            return True


class S3EpisodeResolver(EpisodeResolver):
    """
    Resolves episodes via SQL table and optionally syncs from S3.
    """

    def __init__(
        self,
        folder_path: Path,
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v3",
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        norm_stats: dict | None = None,
    ):
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix
        self.debug = debug
        super().__init__(
            folder_path,
            key_map=key_map,
            transform_list=transform_list,
        )

    def resolve(
        self,
        filters: DatasetFilter | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters.
        Syncs S3 paths to local_root before indexing.
        """
        filters = _ensure_dataset_filter(filters)

        if self.folder_path.is_dir():
            logger.info(f"Using existing directory: {self.folder_path}")
        if not self.folder_path.is_dir():
            self.folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Filters: {filters}")

        filtered_paths = self.sync_from_filters(
            bucket_name=self.bucket_name,
            filters=filters,
            local_dir=self.folder_path,
            debug=self.debug,
        )

        valid_hashes = {hashes for _, hashes in filtered_paths}
        if not valid_hashes:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path,
            valid_folder_names=valid_hashes,
        )

        return datasets

    @staticmethod
    def _get_filtered_paths(
        filters: DatasetFilter | None = None, debug: int | bool | None = None
    ) -> list[tuple[str, str]]:
        """
        Filters episodes from the SQL episode table according to the criteria specified in `filters`
        and returns a list of (zarr_processed_path, episode_hash) tuples for episodes that match and
        have a non-null zarr_processed_path.

        Args:
            filters (DatasetFilter | None): Filter object applied row-by-row to the
                episode table.

        Returns:
            list[tuple[str, str]]: List of tuples, each containing (zarr_processed_path, episode_hash)
                                   for episodes passing the filter criteria.
        """
        filters = _ensure_dataset_filter(filters)
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        if df.empty:
            logger.info("Episode table is empty.")
            return []

        mask = df.apply(
            lambda row: filters.matches(_normalize_filter_row(row.to_dict())),
            axis=1,
        )
        output = df.loc[mask, ["zarr_processed_path", "episode_hash"]]
        n_matched_sql = len(output)

        output = output[
            output["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
        ]
        n_skipped_null = n_matched_sql - len(output)
        if n_skipped_null:
            logger.info(
                "Skipped %d episodes with null/empty zarr_processed_path.",
                n_skipped_null,
            )

        if debug is not None and debug is not False:
            k = min(10 if debug is True else int(debug), len(output))
            if k < len(output):
                logger.info("Debug mode: limiting to %d datasets.", k)
            output = output.iloc[:k]

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @classmethod
    def _sync_s3_to_local(
        cls,
        bucket_name: str,
        s3_paths: list[tuple[str, str]],
        local_dir: Path,
        numworkers: int = 10,
    ):
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

            load_env()
            rl2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
            access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
            secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            if not all([rl2_endpoint_url, access_key_id, secret_access_key]):
                raise ValueError(
                    "R2 credentials missing. Ensure ~/.egoverse_env has "
                    "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
                )
            s5cmd_env = os.environ.copy()
            s5cmd_env["AWS_ACCESS_KEY_ID"] = access_key_id
            s5cmd_env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
            s5cmd_env["AWS_DEFAULT_REGION"] = "auto"
            s5cmd_env["AWS_REGION"] = "auto"
            cmd = [
                "s5cmd",
                "--endpoint-url",
                rl2_endpoint_url,
                "--numworkers",
                str(numworkers),
                "run",
                str(batch_path),
            ]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            subprocess.run(cmd, check=True, env=s5cmd_env)

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: DatasetFilter | None = None,
        local_dir: Path,
        numworkers: int = 10,
        debug: int | bool | None = None,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir

        Args:
            numworkers: Number of parallel workers for s5cmd.

        Returns:
            List[(processed_path, episode_hash)]
        """
        filters = _ensure_dataset_filter(filters)

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_filtered_paths(filters, debug=debug)
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
            numworkers=numworkers,
        )

        return filtered_paths


# ---------------------------------------------------------------------------
# Safer variant: S3EpisodeResolver subclass that filters out unusable episodes
# (missing required keymap keys, corrupt JPEGs, or — optionally — episodes
# without a language-annotation field).
# ---------------------------------------------------------------------------
def _jpeg_probe_failed(ds_obj: "ZarrDataset") -> tuple[str, str] | None:
    """Try decoding 5 sampled frames per image key. If every probe fails
    for a key, return (key, reason); else None."""
    image_keys = getattr(ds_obj, "_image_keys", None) or set()
    for img_key in image_keys:
        try:
            arr = ds_obj.episode_reader._store[img_key]
            n = arr.shape[0] if hasattr(arr, "shape") else len(arr)
            if n == 0:
                return (img_key, "empty")
            probe_idx = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
            ok_any = False
            last_err = ""
            for i in probe_idx:
                try:
                    simplejpeg.decode_jpeg(arr[i : i + 1][0], colorspace="RGB")
                    ok_any = True
                    break
                except Exception as e:
                    last_err = str(e)
            if not ok_any:
                return (img_key, f"all probes failed: {last_err}")
        except Exception as e:
            return (img_key, str(e))
    return None


def _has_annotation(ds_obj: "ZarrDataset", annotation_key: str = "annotations") -> bool:
    try:
        arr = ds_obj.episode_reader._store[annotation_key]
    except Exception:
        return False
    try:
        n = arr.shape[0] if hasattr(arr, "shape") else len(arr)
    except Exception:
        return False
    return n > 0


class SafeS3EpisodeResolver(S3EpisodeResolver):
    """Drop-in replacement for `S3EpisodeResolver` that filters unusable
    episodes after the underlying resolver has discovered them. Skips
    episodes that:
      - are missing any keymap-required key,
      - have unrecoverably-corrupt JPEG image streams (verified by
        spot-decoding a handful of frames), or
      - (optionally) lack a non-empty language-annotation field."""

    def __init__(
        self,
        *args,
        require_annotations: bool = False,
        annotation_key: str = "annotations",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.require_annotations = require_annotations
        self.annotation_key = annotation_key

    def resolve(self, filters=None) -> dict[str, "ZarrDataset"]:
        datasets = super().resolve(filters=filters)
        if self.key_map is None:
            return datasets

        required = {
            spec["zarr_key"]
            for spec in self.key_map.values()
            if isinstance(spec, dict)
            and spec.get("key_type") != "annotation_keys"
            and spec.get("zarr_key") is not None
        }

        kept: dict[str, "ZarrDataset"] = {}
        for ep_hash, ds_obj in datasets.items():
            available = set(getattr(ds_obj, "keys_dict", {}))
            missing = required - available
            if missing:
                logger.warning(
                    "SafeS3EpisodeResolver: skipping %s (missing %s)",
                    ep_hash,
                    sorted(missing),
                )
                continue
            bad = _jpeg_probe_failed(ds_obj)
            if bad is not None:
                logger.warning(
                    "SafeS3EpisodeResolver: skipping %s (JPEG probe failed for %s: %s)",
                    ep_hash,
                    bad[0],
                    bad[1],
                )
                continue
            if self.require_annotations and not _has_annotation(
                ds_obj, self.annotation_key
            ):
                logger.warning(
                    "SafeS3EpisodeResolver: skipping %s (no '%s' field — episode has no language annotation)",
                    ep_hash,
                    self.annotation_key,
                )
                continue
            kept[ep_hash] = ds_obj

        if not kept:
            logger.warning(
                "SafeS3EpisodeResolver: every episode was filtered out — "
                "the keymap may demand a key your data does not have, or "
                "require_annotations=true is too strict."
            )
        return kept


class LocalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from local Zarr stores, filtering via local metadata.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug=False,
    ):
        super().__init__(folder_path, key_map, transform_list)
        self.debug = debug

    @staticmethod
    def _local_filters_match(
        metadata: dict,
        episode_hash: str,
        filters: DatasetFilter,
    ) -> bool:
        return filters.matches(
            _normalize_filter_row(metadata, episode_hash=episode_hash)
        )

    @classmethod
    def _get_local_filtered_paths(
        cls,
        search_path: Path,
        filters: DatasetFilter | None = None,
        debug: int | bool | None = None,
    ):
        filters = _ensure_dataset_filter(filters)
        if not search_path.is_dir():
            logger.warning("Local path does not exist: %s", search_path)
            return []

        filtered = []
        for p in sorted(search_path.iterdir()):
            if not p.is_dir():
                continue

            episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name

            try:
                store = zarr.open_group(str(p), mode="r")
                metadata = dict(store.attrs)
            except Exception as e:
                logger.warning("Failed to read metadata for %s: %s", p, e)
                continue

            if cls._local_filters_match(metadata, episode_hash, filters):
                filtered.append((str(p), episode_hash))

        if debug is not None and debug is not False:
            k = min(10 if debug is True else int(debug), len(filtered))
            if k < len(filtered):
                logger.info("Debug mode: limiting to %d datasets.", k)
            filtered = filtered[:k]

        logger.info("Local filtered paths: %s", filtered)
        return filtered

    def resolve(
        self,
        sync_from_s3=False,
        filters: DatasetFilter | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters from local data.
        """
        if sync_from_s3:
            logger.warning(
                "LocalEpisodeResolver does not sync from S3; ignoring sync_from_s3=True."
            )

        filters = _ensure_dataset_filter(filters)

        filtered_paths = self._get_local_filtered_paths(
            self.folder_path, filters, debug=self.debug
        )

        valid_folder_names = {folder_name for _, folder_name in filtered_paths}
        logger.info(f"Valid folder names: {valid_folder_names}")
        if not valid_folder_names:
            raise ValueError(
                "No valid collection names from local filtering: "
                "filters matched no episodes in the local directory."
            )

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path, valid_folder_names=valid_folder_names
        )

        return datasets


class MultiDataset(torch.utils.data.Dataset):
    """
    Wraps a dict of child datasets (Zarr leaves or other MultiDatasets) and
    also owns the normalization-stats descriptor and helpers that used to live
    in separate ``NormStats`` / ``NormalizingMultiDataset`` classes.

    Two construction modes:
      - **Data mode** (default): pass ``datasets`` to wrap a real dataset graph
        (existing behaviour, used during training). Stats fields start empty;
        call ``populate_from_datasets()`` and ``infer_norm_from_dataset(...)``
        to fill them in, then ``attach_normalize_transforms()`` to wire
        normalize/reject transforms onto each leaf's ``transform_list``.
      - **State mode** (``state=...``, ``datasets=None``): construct a
        stats-only instance for deploy/eval where the dataset graph isn't
        available. ``self.datasets`` is empty; only the stats fields are
        populated. Used by checkpoint reconstruction.
    """

    NORMALIZE_KEY_TYPES = ("proprio_keys", "action_keys")

    def __init__(
        self,
        datasets: (
            dict[str, MultiDataset | ZarrDataset | ZarrActionExpertDataset] | None
        ) = None,
        mode: str = "train",
        percent: float = 0.1,
        valid_ratio: float = 0.2,
        norm_mode: str = "zscore",
        state: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            datasets: Dict of child datasets. None for state-only construction.
            mode: One of "train", "valid", "total", "percent" — which split to keep.
            percent: Fraction (when mode="percent").
            valid_ratio: Train/valid split ratio.
            norm_mode: One of "zscore", "minmax", "quantile".
            state: If provided, populate stats fields from this dict (deploy mode).
        """
        super().__init__()

        # ---- Stats fields (always present, may be empty) ----
        self.norm_mode = norm_mode
        self.embodiments: set[int] = set()
        self.key_types: dict[int, dict[str, str]] = {}
        self.zarr_keys: dict[int, dict[str, str]] = {}
        self.shapes: dict[int, dict[str, tuple]] = {}
        self.norm_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        self._norm_run_metadata: dict[str, float | int | None] | None = None

        # ---- Dataset graph fields ----
        self.datasets: dict = {}
        self.index_map: list = []
        self._global_indices_by_dataset: dict[str, list[int]] = {}
        # Dedup bounds-check warnings: keyed by f"bounds:{episode}:{zarr_key}"
        self._warned_violations: set[str] = set()
        self.train_collections: set = set()
        self.valid_collections: set = set()

        if state is not None:
            # Deploy / state-only construction — no dataset graph.
            self._load_state(state)
            return

        if datasets is None:
            raise ValueError("MultiDataset requires either `datasets` or `state`.")

        # ---- Normal data-mode construction ----
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

        self.datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}
        assert self.datasets, "No datasets left after applying mode split."

        self._global_indices_by_dataset = {n: [] for n in self.datasets}
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                global_idx = len(self.index_map)
                self.index_map.append((dataset_name, local_idx))
                self._global_indices_by_dataset[dataset_name].append(global_idx)

    def __len__(self) -> int:
        return len(self.index_map)

    @staticmethod
    def _episode_name_for_dataset(dataset, dataset_name: str) -> str:
        episode_path = getattr(dataset, "episode_path", None)
        if episode_path is None:
            return dataset_name
        return Path(episode_path).name

    def set_norm_stats_from(self, source: "MultiDataset") -> None:
        """Share stats with this dataset (and its nested MultiDatasets) by
        reference. After this call ``__getitem__`` will bounds-check + normalize
        each sample using ``source``'s ``norm_stats``/``key_types``/``zarr_keys``.

        Use this *instead of* ``attach_normalize_transforms`` — it doesn't mutate
        any leaf-level ``transform`` list, so it can't accumulate duplicate
        passes when leaves share a transform list reference.
        """
        self.norm_stats = source.norm_stats
        self.key_types = source.key_types
        self.zarr_keys = source.zarr_keys
        self.shapes = source.shapes
        self.embodiments = source.embodiments
        self.norm_mode = source.norm_mode
        # Each MultiDataset keeps its own warning-dedup state.
        self._warned_violations = set()
        for ds in self.datasets.values():
            if isinstance(ds, MultiDataset):
                ds.set_norm_stats_from(source)

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        """Return a violation message if any tracked key in ``data`` has NaN/Inf
        or values outside per-key quantile bounds. ``None`` means the sample
        passes. Logs each (episode, key) violation once.
        """
        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            return None
        per_emb_stats = self.norm_stats.get(embodiment_id, {})
        if not per_emb_stats:
            return None

        episode_name = self._episode_name_for_dataset(dataset, dataset_name)

        for key_name, stats in per_emb_stats.items():
            zarr_key = self.zarr_keys.get(embodiment_id, {}).get(key_name)
            if zarr_key is None or zarr_key not in data:
                continue
            v = data[zarr_key]
            if isinstance(v, torch.Tensor):
                arr = v.float()
            elif isinstance(v, np.ndarray):
                arr = torch.from_numpy(v).float()
            else:
                continue

            q_low = stats.get(
                "quantile_0_01", stats.get("quantile_0_1", stats["quantile_1"])
            )
            q_high = stats.get(
                "quantile_99_99", stats.get("quantile_99_9", stats["quantile_99"])
            )
            q_low = torch.as_tensor(q_low, device=arr.device, dtype=torch.float32)
            q_high = torch.as_tensor(q_high, device=arr.device, dtype=torch.float32)
            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                prefix = f"NaN/Inf in {zarr_key} ep={episode_name} frame={idx}"
                warn_key = f"naninf:{episode_name}:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    logger.warning(prefix)
                return prefix

            below = arr < q_low
            above = arr > q_high
            if torch.any(below) or torch.any(above):
                prefix = f"Bounds violation in {zarr_key} ep={episode_name} frame={idx}"
                warn_key = f"bounds:{episode_name}:{zarr_key}"
                if warn_key not in self._warned_violations:
                    self._warned_violations.add(warn_key)
                    n_below = int(below.sum().item())
                    n_above = int(above.sum().item())
                    logger.warning(
                        f"{prefix} | n_below={n_below} n_above={n_above} "
                        f"arr_range=[{arr.min().item():.4f}, {arr.max().item():.4f}]"
                    )
                return prefix
        return None

    def __getitem__(self, idx, _attempts: int | None = None):
        attempts = _attempts
        while True:
            dataset_name, local_idx = self.index_map[idx]
            dataset = self.datasets[dataset_name]
            try:
                data = dataset[local_idx]
            except Exception as e:
                next_idx, attempts = self._next_after_failure(
                    idx,
                    dataset_name,
                    attempts,
                    reason=f"Sample failed ({type(e).__name__}: {e}) at "
                    f"{dataset_name}[{local_idx}]",
                )
                idx = next_idx
                continue

            # If this leaf is itself a MultiDataset, it already ran bounds +
            # normalize for its returned sample. Pass through unchanged.
            if isinstance(dataset, MultiDataset):
                return data

            violation = self._check_bounds(data, dataset, local_idx, dataset_name)
            if violation is not None:
                next_idx, attempts = self._next_after_failure(
                    idx,
                    dataset_name,
                    attempts,
                    reason=violation,
                )
                idx = next_idx
                continue

            # Bounds passed — normalize and return.
            if self.norm_stats and data.get("embodiment") in self.norm_stats:
                data = self.normalize(data, data["embodiment"])
            return data

    def _next_after_failure(
        self, idx: int, dataset_name: str, attempts: int | None, *, reason: str
    ) -> tuple[int, int]:
        global_candidates = self._global_indices_by_dataset[dataset_name]
        next_idx, attempts = get_fallback_idx(
            idx=idx,
            candidates=global_candidates,
            _attempts=attempts,
            max_attempts=len(global_candidates),
            exhausted_error=(
                f"Entire dataset bad (no valid indices): dataset={dataset_name}"
            ),
        )
        next_dataset_name, next_local_idx = self.index_map[next_idx]
        logger.warning(
            f"{reason} | attempt {attempts}, "
            f"trying {next_dataset_name}[{next_local_idx}]"
        )
        return next_idx, attempts

    @classmethod
    def _from_resolver(cls, resolver: EpisodeResolver, **kwargs):
        """create a MultiDataset from an EpisodeResolver."""
        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", None)

        if isinstance(resolver, LocalEpisodeResolver):
            resolved = resolver.resolve(sync_from_s3=sync_from_s3, filters=filters)
        else:
            resolved = resolver.resolve(filters=filters)

        return cls(datasets=resolved, **kwargs)

    # =====================================================================
    # Stats / normalization
    # =====================================================================

    @staticmethod
    def _iter_leaves(ds):
        """Yield non-MultiDataset leaves from possibly nested wrappers."""
        if isinstance(ds, MultiDataset):
            for child in ds.datasets.values():
                yield from MultiDataset._iter_leaves(child)
        else:
            yield ds

    def populate_from_datasets(self, datasets: dict | None = None) -> None:
        """
        Populate per-embodiment key inventory by walking leaves and probing
        one post-transform sample per leaf. ``datasets`` defaults to
        ``self.datasets`` so the typical call is just ``mds.populate_from_datasets()``.
        """
        graph = datasets if datasets is not None else self.datasets
        for ds in graph.values():
            for leaf in self._iter_leaves(ds):
                emb = getattr(leaf, "embodiment", None)
                key_map = getattr(leaf, "key_map", None)
                if emb is None or key_map is None:
                    continue
                emb_id = emb if isinstance(emb, int) else get_embodiment_id(emb)
                self.embodiments.add(emb_id)
                self.key_types.setdefault(emb_id, {})
                self.zarr_keys.setdefault(emb_id, {})
                self.shapes.setdefault(emb_id, {})
                self.norm_stats.setdefault(emb_id, {})

                sample_keys: set | None = None
                try:
                    sample = leaf[0]
                    if isinstance(sample, dict):
                        sample_keys = set(sample.keys())
                except Exception as e:
                    logger.warning(
                        f"[MultiDataset] Could not probe leaf for post-transform "
                        f"keys (emb={emb_id}): {e}. Falling back to raw key_map."
                    )

                if sample_keys is None:
                    for key_name, info in key_map.items():
                        self.key_types[emb_id][key_name] = info.get(
                            "key_type", "metadata_keys"
                        )
                        self.zarr_keys[emb_id][key_name] = info["zarr_key"]
                    continue

                # Identity zarr_keys map (data_key is the algo-side name).
                for data_key in sample_keys:
                    if data_key in key_map:
                        info = key_map[data_key]
                        self.key_types[emb_id][data_key] = info.get(
                            "key_type", "metadata_keys"
                        )
                    else:
                        inferred = _infer_key_type(data_key)
                        if inferred is None:
                            continue
                        self.key_types[emb_id][data_key] = inferred
                    self.zarr_keys[emb_id][data_key] = data_key

    # ---- key lookups ----

    def keys_of_type(self, key_type: str, embodiment_id: int) -> list[str]:
        return [
            k for k, t in self.key_types.get(embodiment_id, {}).items() if t == key_type
        ]

    def is_key_with_embodiment(self, key_name: str, embodiment_id: int) -> bool:
        return key_name in self.key_types.get(embodiment_id, {})

    def keyname_to_zarr_key(self, key_name: str, embodiment_id: int) -> str | None:
        return self.zarr_keys.get(embodiment_id, {}).get(key_name)

    def zarr_key_to_keyname(self, zarr_key: str, embodiment_id: int) -> str | None:
        for k, v in self.zarr_keys.get(embodiment_id, {}).items():
            if v == zarr_key:
                return k
        return None

    def key_shape(self, key_name: str, embodiment_id: int) -> tuple:
        if key_name not in self.shapes.get(embodiment_id, {}):
            raise ValueError(
                f"Shape for key {key_name!r} on embodiment {embodiment_id} not inferred yet."
            )
        return self.shapes[embodiment_id][key_name]

    # ---- shape & norm inference ----

    def infer_shapes_from_batch(self, batch: dict) -> None:
        for emb_id, per_emb in self.zarr_keys.items():
            for key_name, zarr_key in per_emb.items():
                if zarr_key in batch:
                    val = batch[zarr_key]
                    if hasattr(val, "shape"):
                        self.shapes.setdefault(emb_id, {})[key_name] = tuple(val.shape)
                    elif isinstance(val, int):
                        self.shapes.setdefault(emb_id, {})[key_name] = (1,)

    def infer_norm_from_dataset(
        self,
        dataset,
        dataset_name,
        sample_frac: float = 0.10,
        seed: int = 42,
        max_samples: int | None = None,
        batch_size: int = 512,
        num_workers: int = 4,
        precomputed_norm_path: str | None = None,
    ):
        embodiment = dataset_name
        if isinstance(embodiment, str):
            embodiment = get_embodiment_id(embodiment)

        norm_keys = list(self.keys_of_type("proprio_keys", embodiment))
        norm_keys.extend(self.keys_of_type("action_keys", embodiment))
        if not norm_keys:
            logger.warning(
                f"[MultiDataset] No proprio/action keys for embodiment={embodiment}"
            )
            return

        self.norm_stats.setdefault(embodiment, {})

        if precomputed_norm_path is not None:
            if os.path.isdir(precomputed_norm_path):
                precomputed_file = os.path.join(
                    precomputed_norm_path, "norm_stats.json"
                )
            elif os.path.isfile(precomputed_norm_path):
                precomputed_file = precomputed_norm_path
            else:
                logger.warning(
                    f"[MultiDataset] precomputed_norm_path={precomputed_norm_path} is not valid"
                )
                return
            if os.path.isfile(precomputed_file):
                with open(precomputed_file, "r") as f:
                    payload = json.load(f)
                self.norm_stats[embodiment] = payload["stats"].get(str(embodiment), {})
                self._norm_run_metadata = payload.get("norm_run_metadata", None)
                logger.info(
                    f"[MultiDataset] Loaded precomputed stats for embodiment={embodiment}"
                )
                return

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        N = len(dataset)
        if N <= 0:
            raise ValueError("Dataset is empty")
        n_samples = int(math.ceil(sample_frac * N))
        n_samples = max(1, min(n_samples, N))
        if max_samples is not None:
            n_samples = min(n_samples, max_samples)

        logger.info(f"[MultiDataset] embodiment={embodiment} norm_keys={norm_keys}")
        logger.info(
            f"[MultiDataset] sampling {n_samples}/{N} (~{100 * sample_frac:.1f}%)"
        )

        loading_start = time.time()
        collected = self._collect_norm_samples(
            loader, norm_keys, embodiment, n_samples, batch_size, num_workers
        )
        for k in [k for k, v in collected.items() if not v]:
            del collected[k]
            norm_keys.remove(k)
        loading_time = time.time() - loading_start

        computing_start = time.time()
        for k in norm_keys:
            collected[k] = np.concatenate(collected[k], axis=0)
            stats_np = self._compute_stats_for_array(collected[k])
            self.norm_stats[embodiment][k] = {
                name: np.asarray(arr, dtype=np.float32)
                for name, arr in stats_np.items()
            }
            logger.info(
                f"[MultiDataset] key={k} samples={collected[k].shape[0]} stat_shape={stats_np['mean'].shape}"
            )
        computing_time = time.time() - computing_start

        self._norm_run_metadata = {
            "loading_time": loading_time,
            "computing_time": computing_time,
            "frames": n_samples,
        }
        logger.info(
            f"[MultiDataset] Finished norm inference, loading={loading_time:.2f}s, computing={computing_time:.2f}s"
        )

    def _collect_norm_samples(
        self, loader, norm_keys, embodiment, n_samples, batch_size, num_workers
    ):
        collected = {k: [] for k in norm_keys}
        cur = 0
        with tqdm(total=n_samples, unit="sample") as pbar:
            for batch in loader:
                remaining = n_samples - cur
                if remaining <= 0:
                    break
                batch_len = None
                for value in batch.values():
                    if hasattr(value, "shape") and len(value.shape) > 0:
                        batch_len = int(value.shape[0])
                        break
                if batch_len is None:
                    raise ValueError(
                        "[MultiDataset] Could not infer batch size from DataLoader batch"
                    )
                take = min(remaining, batch_len)
                for k in norm_keys:
                    zarr_key = self.keyname_to_zarr_key(k, embodiment)
                    if zarr_key is None or zarr_key not in batch:
                        continue
                    x = batch[zarr_key][:take]
                    if hasattr(x, "detach"):
                        x = x.detach().cpu().numpy()
                    collected[k].append(x)
                cur += take
                pbar.update(take)
        return collected

    @staticmethod
    def _compute_stats_for_array(X):
        return {
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0),
            "min": np.min(X, axis=0),
            "max": np.max(X, axis=0),
            "median": np.median(X, axis=0),
            "quantile_1": np.percentile(X, 1, axis=0),
            "quantile_99": np.percentile(X, 99, axis=0),
            "quantile_0_01": np.percentile(X, 0.01, axis=0),
            "quantile_99_99": np.percentile(X, 99.99, axis=0),
        }

    def cache_stats(self, save_cache_dir: str):
        cache_dir = os.path.join(save_cache_dir, "norm_stats")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, "norm_stats.json")

        stats_out: dict[str, dict[str, dict[str, list]]] = {}
        for emb, keys_dict in self.norm_stats.items():
            stats_out[str(emb)] = {
                k: {name: np.asarray(arr).tolist() for name, arr in stat_dict.items()}
                for k, stat_dict in keys_dict.items()
            }
        payload = {
            "stats": stats_out,
            "loading_time": None,
            "computing_time": None,
            "frames": None,
        }
        if self._norm_run_metadata is not None:
            for k in ("loading_time", "computing_time", "frames"):
                if k in self._norm_run_metadata:
                    payload[k] = self._norm_run_metadata[k]
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"[MultiDataset] Cached stats to {out_path}")

    # ---- normalize / unnormalize ----

    def _apply_norm_one(self, tensor, stats):
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(
                stats["mean"], device=tensor.device, dtype=torch.float32
            )
            std = torch.as_tensor(
                stats["std"], device=tensor.device, dtype=torch.float32
            )
            return (tensor - mean) / (std + 1e-6)
        if self.norm_mode == "minmax":
            mn = torch.as_tensor(
                stats["min"], device=tensor.device, dtype=torch.float32
            )
            mx = torch.as_tensor(
                stats["max"], device=tensor.device, dtype=torch.float32
            )
            return 2.0 * ((tensor - mn) / (mx - mn + 1e-6)) - 1.0
        if self.norm_mode == "quantile":
            q1 = torch.as_tensor(
                stats["quantile_1"], device=tensor.device, dtype=torch.float32
            )
            q99 = torch.as_tensor(
                stats["quantile_99"], device=tensor.device, dtype=torch.float32
            )
            return 2.0 * ((tensor - q1) / (q99 - q1 + 1e-6)) - 1.0
        raise ValueError(f"Invalid normalization mode: {self.norm_mode}")

    def _apply_unnorm_one(self, tensor, stats):
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(
                stats["mean"], device=tensor.device, dtype=torch.float32
            )
            std = torch.as_tensor(
                stats["std"], device=tensor.device, dtype=torch.float32
            )
            return tensor * (std + 1e-6) + mean
        if self.norm_mode == "minmax":
            mn = torch.as_tensor(
                stats["min"], device=tensor.device, dtype=torch.float32
            )
            mx = torch.as_tensor(
                stats["max"], device=tensor.device, dtype=torch.float32
            )
            return (tensor + 1) * 0.5 * (mx - mn + 1e-6) + mn
        if self.norm_mode == "quantile":
            q1 = torch.as_tensor(
                stats["quantile_1"], device=tensor.device, dtype=torch.float32
            )
            q99 = torch.as_tensor(
                stats["quantile_99"], device=tensor.device, dtype=torch.float32
            )
            return (tensor + 1) * 0.5 * (q99 - q1 + 1e-6) + q1
        raise ValueError(f"Invalid normalization mode: {self.norm_mode}")

    def normalize(self, data: dict, embodiment_id: int) -> dict:
        if not self.norm_stats.get(embodiment_id):
            return data
        out = dict(data)
        for key_name, key_type in self.key_types.get(embodiment_id, {}).items():
            if key_type not in self.NORMALIZE_KEY_TYPES:
                continue
            stats = self.norm_stats[embodiment_id].get(key_name)
            if stats is None:
                continue
            zarr_key = self.zarr_keys[embodiment_id][key_name]
            if zarr_key not in out:
                continue
            tensor = out[zarr_key]
            if not isinstance(tensor, torch.Tensor):
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor).float()
                else:
                    continue
            out[zarr_key] = self._apply_norm_one(tensor, stats)
        return out

    def unnormalize(self, data: dict, embodiment_id: int) -> dict:
        if not self.norm_stats.get(embodiment_id):
            return data
        out = dict(data)
        zk_to_kn = {v: k for k, v in self.zarr_keys.get(embodiment_id, {}).items()}
        for data_key, value in list(data.items()):
            key_name = (
                data_key
                if data_key in self.norm_stats[embodiment_id]
                else zk_to_kn.get(data_key)
            )
            if key_name is None:
                continue
            stats = self.norm_stats[embodiment_id].get(key_name)
            if stats is None:
                continue
            if not isinstance(value, torch.Tensor):
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                else:
                    continue
            out[data_key] = self._apply_unnorm_one(value, stats)
        return out

    # ---- transform attachment ----

    def attach_normalize_transforms(
        self, datasets: dict | None = None, reject_outliers: bool = True
    ) -> None:
        """Deprecated. Use ``set_norm_stats_from`` on each training/valid
        MultiDataset instead. Bounds-check + normalize now run at the
        MultiDataset level in ``__getitem__``, not as per-leaf transforms.

        Kept as a thin shim that calls ``set_norm_stats_from(self)`` on each
        MultiDataset in ``datasets`` so existing callers keep working. The
        ``reject_outliers`` flag is no longer honored — bounds checking is
        always on when stats are populated. To disable, clear ``norm_stats``.
        """
        del reject_outliers  # unused
        graph = datasets if datasets is not None else self.datasets
        for ds in graph.values():
            if isinstance(ds, MultiDataset):
                ds.set_norm_stats_from(self)

    # ---- serialization (checkpoint roundtrip) ----

    @staticmethod
    def _clone_norm_stats(norm_stats):
        out = {}
        for emb, per_emb in (norm_stats or {}).items():
            out[emb] = {
                key: {
                    name: (
                        v.detach().cpu().clone()
                        if torch.is_tensor(v)
                        else copy.deepcopy(v)
                    )
                    for name, v in stats.items()
                }
                for key, stats in per_emb.items()
            }
        return out

    def to_state(self) -> dict:
        """Serialize stats only (not the dataset graph). Suitable for checkpoint."""
        return {
            "norm_mode": self.norm_mode,
            "embodiments": sorted(self.embodiments),
            "key_types": copy.deepcopy(self.key_types),
            "zarr_keys": copy.deepcopy(self.zarr_keys),
            "shapes": copy.deepcopy(self.shapes),
            "norm_stats": self._clone_norm_stats(self.norm_stats),
        }

    @classmethod
    def from_state(cls, state: dict) -> "MultiDataset":
        """Reconstruct a stats-only MultiDataset (no dataset graph) from state."""
        if state is None:
            raise ValueError("MultiDataset state must be provided for reconstruction.")
        return cls(state=state)

    def _load_state(self, state: dict) -> None:
        """Populate stats fields from a state dict. Used by state-only construction."""
        self.norm_mode = state.get("norm_mode", self.norm_mode)
        self.embodiments = set(state.get("embodiments", []))
        self.key_types = copy.deepcopy(state.get("key_types", {}))
        self.zarr_keys = copy.deepcopy(state.get("zarr_keys", {}))
        self.shapes = copy.deepcopy(state.get("shapes", {}))
        self.norm_stats = self._clone_norm_stats(state.get("norm_stats", {}))
        for emb in self.embodiments:
            self.key_types.setdefault(emb, {})
            self.zarr_keys.setdefault(emb, {})
            self.shapes.setdefault(emb, {})
            self.norm_stats.setdefault(emb, {})


# ---------------------------------------------------------------------------
# Per-episode subsampling wrapper around MultiDataset (K evenly-spaced
# frames OR every Sth frame within each episode).
# ---------------------------------------------------------------------------
def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n // 2]
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def _strided_indices(n: int, stride: int) -> list[int]:
    if n <= 0 or stride <= 0:
        return []
    return list(range(0, n, stride))


class EvenStrideDataset(MultiDataset):
    """Wraps a `MultiDataset` to subsample frames per underlying episode.
    Pass exactly one of `frames_per_episode` (K evenly-spaced) or
    `stride` (every Sth frame). Subclasses `MultiDataset` so trainHydra's
    isinstance check passes; we skip the parent `__init__` and adopt its
    class identity only, delegating to `self.base`."""

    def __init__(
        self, base, frames_per_episode: int | None = None, stride: int | None = None
    ):
        if (frames_per_episode is None) == (stride is None):
            raise ValueError(
                "EvenStrideDataset requires exactly ONE of "
                "`frames_per_episode` or `stride` to be set "
                f"(got frames_per_episode={frames_per_episode}, stride={stride})."
            )
        gibd = getattr(base, "_global_indices_by_dataset", None)
        if gibd is None:
            raise ValueError(
                f"EvenStrideDataset requires a MultiDataset exposing "
                f"_global_indices_by_dataset, got {type(base).__name__}"
            )
        torch.utils.data.Dataset.__init__(self)
        self.base = base
        self.frames_per_episode = frames_per_episode
        self.stride = stride
        self.datasets = base.datasets

        keep: list[int] = []
        for ep_name, idxs in gibd.items():
            n = len(idxs)
            if stride is not None:
                picks = _strided_indices(n, stride)
                rule = f"stride={stride}"
            else:
                picks = _evenly_spaced_indices(n, frames_per_episode)
                rule = f"k={frames_per_episode}"
            chosen = [idxs[i] for i in picks]
            actual_stride = n / max(1, len(chosen))
            logger.info(
                "EvenStrideDataset: %s -> %d / %d frames (%s, actual stride ~%.1f)",
                ep_name,
                len(chosen),
                n,
                rule,
                actual_stride,
            )
            keep.extend(chosen)
        self.indices = sorted(keep)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def set_data_schematic(self, data_schematic) -> None:
        self.base.set_data_schematic(data_schematic)
        self.data_schematic = data_schematic

    def __getattr__(self, item):
        if item == "base":
            raise AttributeError(item)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(item)
        return getattr(base, item)


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path: Path,
        key_map: dict,
        transform_list: list | None = None,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            key_map: dict mapping from dataset keys to zarr keys and horizon info, e.g. {"obs/image/front": {"zarr_key": "observations.images.front", "horizon": 4}, ...}
            transform_list: list of Transform objects to apply to the data after loading, e.g. for action chunk transformations. Should be in order of application.
        """
        self.episode_path = Episode_path
        self.metadata = None
        self._image_keys = None  # Lazy-loaded set of JPEG-encoded keys
        self._json_keys = None  # Lazy-loaded set of JSON-encoded keys
        self._annotations = None
        self.init_episode()

        self.key_map = key_map
        self.transform = transform_list
        super().__init__()

    def init_episode(self):
        """
        inits the zarr episode and all the metadata associated, as well as total_frames for len
        """
        self.episode_reader = ZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.embodiment = self.metadata["embodiment"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}

        # Detect JPEG-encoded image keys from metadata
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    def _detect_json_keys(self) -> set[str]:
        """
        Detect keys containing JSON-encoded bytes from metadata.

        Returns:
            Set of keys containing JSON payloads.
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "json"}

    @staticmethod
    def _decode_json_entry(value):
        if isinstance(value, np.void):
            value = value.item()
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _load_annotations(self) -> list[dict]:
        """
        Load and cache decoded language annotations.

        Expected format per entry:
            {"text": str, "start_idx": int, "end_idx": int}
        """
        if self._annotations is not None:
            return self._annotations

        raw = self.episode_reader._store["annotations"][:]

        decoded = [self._decode_json_entry(x) for x in raw]
        self._annotations = [d for d in decoded if isinstance(d, dict)]
        return self._annotations

    def _annotation_text_for_frame(self, frame_idx: int) -> str:
        """
        Resolve language annotation text for a frame from span annotations.
        """
        annotations = self._load_annotations()
        valid_annotations = []
        for ann in annotations:
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx <= frame_idx < end_idx:
                valid_annotations.append(ann.get("text", ""))
        return valid_annotations

    def __len__(self) -> int:
        return self.total_frames

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        """End index (exclusive) for a windowed read starting at ``start_idx``.

        Subclasses can override to add per-key-type clamping (e.g., annotation EOS).
        """
        return min(start_idx + horizon, self.total_frames)

    def _pad_sequences(self, data, horizon: int | None) -> dict:
        if horizon is None:
            return data

        # Note that k is zarr key
        for k in data:
            if isinstance(data[k], np.ndarray):
                seq_len = data[k].shape[0]
                if seq_len < horizon:
                    # Pad by repeating the last frame
                    pad_len = horizon - seq_len
                    last_frame = data[k][-1:]  # Keep dims: (1, action_dim)
                    padding = np.repeat(last_frame, pad_len, axis=0)
                    data[k] = np.concatenate([data[k], padding], axis=0)

        return data

    def __getitem__(
        self,
        idx: int,
        _fallback_origin: int | None = None,
        _attempts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        # Build keys_dict with ranges based on whether action chunking is enabled
        """
        ZarrDataset handles jpeg decoding and transform function errors, and
        retries on a different random index (bounded loop, no recursion).
        """
        origin = _fallback_origin if _fallback_origin is not None else idx
        attempts = _attempts

        def _next(reason: str, key: str = "") -> int:
            nonlocal attempts
            next_idx, attempts = get_fallback_idx(
                idx=idx,
                candidates=range(self.total_frames),
                _attempts=attempts,
                max_attempts=self.total_frames,
                exhausted_error=(
                    f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
                ),
            )
            logger.warning(
                f"{reason} ep={Path(self.episode_path).name} frame={idx}"
                + (f" key={key}" if key else "")
                + f" | attempt {attempts}, trying random idx {next_idx}"
            )
            return next_idx

        while True:
            data = {}
            retry = False
            for k in self.key_map:
                zarr_key = self.key_map[k]["zarr_key"]
                key_type = self.key_map[k].get("key_type", None)
                horizon = self.key_map[k].get("horizon", None)

                if key_type == "annotation_keys":
                    data[k] = self._annotation_text_for_frame(idx)
                    continue

                if horizon is not None:
                    end_idx = self._chunk_end_idx(idx, horizon, key_type)
                    read_interval = (idx, end_idx)
                else:
                    read_interval = (idx, None)
                read_dict = {zarr_key: read_interval}
                raw_data = self.episode_reader.read(read_dict)
                self._pad_sequences(raw_data, horizon)  # should be able to pad images
                data[k] = raw_data[zarr_key]

                if zarr_key in self._image_keys:
                    jpeg_bytes = data[k]
                    try:
                        decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
                    except Exception:
                        idx = _next("JPEG decode failed", key=k)
                        retry = True
                        break
                    data[k] = np.transpose(decoded, (2, 0, 1)) / 255.0
                elif zarr_key in self._json_keys:
                    if isinstance(data[k], np.ndarray):
                        data[k] = [self._decode_json_entry(v) for v in data[k]]
                    else:
                        data[k] = self._decode_json_entry(data[k])
            if retry:
                continue

            if self.transform:
                for transform in self.transform or []:
                    data = transform.transform(data)

            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(torch.float32)

            data["metadata.robot_name"] = get_embodiment_id(self.embodiment)
            data["embodiment"] = get_embodiment_id(self.embodiment)
            ep_name = Path(self.episode_path).name
            data["episode_hash"] = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            _ = origin  # preserved for symmetry with prior API
            return data


class ZarrAnnotationCutoffDataset(ZarrDataset):
    """ZarrDataset that clamps action chunks at the end of the enclosing annotation.

    Standard chunking from the start frame, but action reads stop at EOS+1 of the
    annotation span containing the start frame. The chunk is then padded out to
    ``horizon`` via the base ``_pad_sequences`` (repeat-last), so frames beyond
    EOS become the last action of the interval rather than crossing into the
    next annotation.

    If the start frame is not inside any annotation, behaves like the base class.
    """

    def init_episode(self):
        super().init_episode()
        self._frame_to_ann_end: dict[int, int] | None = None

    def _build_frame_to_ann_end(self) -> dict[int, int]:
        """Map ``frame_idx -> ann_end`` (exclusive) for every frame inside an
        annotation span. Annotations use half-open ``[start_idx, end_idx)``.
        """
        mapping: dict[int, int] = {}
        for ann in self._load_annotations():
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx < 0 or end_idx <= start_idx:
                continue
            for idx in range(start_idx, end_idx):
                mapping[idx] = end_idx
        return mapping

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        end_idx = super()._chunk_end_idx(start_idx, horizon, key_type)
        if key_type != "action_keys":
            return end_idx
        if self._frame_to_ann_end is None:
            self._frame_to_ann_end = self._build_frame_to_ann_end()
        ann_end = self._frame_to_ann_end.get(start_idx)
        if ann_end is None:
            return end_idx
        return min(end_idx, ann_end)


class S3AnnotationCutoffEpisodeResolver(S3EpisodeResolver):
    """S3EpisodeResolver that loads ZarrAnnotationCutoffDataset instances."""

    _dataset_class = ZarrAnnotationCutoffDataset


class LocalAnnotationCutoffEpisodeResolver(LocalEpisodeResolver):
    """LocalEpisodeResolver that loads ZarrAnnotationCutoffDataset instances."""

    _dataset_class = ZarrAnnotationCutoffDataset


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
        self._store = zarr.open_group(str(self._path), mode="r")
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]

    def read(
        self, keys_with_ranges: dict[str, tuple[int, int | None]]
    ) -> dict[str, np.ndarray]:
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
                data = arr[start : start + 1][0]
            result[key] = data
        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.
        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        if isinstance(self.keys, dict):
            return list(self.keys.keys())
        return list(self.keys)

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.
        Returns:
            Number of frames
        """
        return self.metadata["total_frames"]

    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"
