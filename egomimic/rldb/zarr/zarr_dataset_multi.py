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
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr

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


def _import_real_modal():
    """Return the installed Modal SDK, working around egomimic.modal shadowing.

    When trainHydra.py runs as `python /root/EgoVerse/egomimic/trainHydra.py`,
    Python puts /root/EgoVerse/egomimic on sys.path[0], which makes a bare
    `import modal` resolve to the egomimic.modal subpackage. This helper:
      1. Returns the cached real SDK if one is already in sys.modules.
      2. Otherwise strips egomimic from sys.path and re-imports.

    Important: we do NOT pop `modal` from sys.modules once the real SDK is
    cached there — re-importing produces a fresh top-level module that no
    longer carries the lazily-attached submodules (modal._grpc_client, etc.),
    breaking subsequent fn.map() calls.
    """
    import sys

    existing = sys.modules.get("modal")
    if existing is not None and hasattr(existing, "Function"):
        return existing

    _orig_path = list(sys.path)
    sys.path = [p for p in sys.path if Path(p).name != "egomimic"]
    sys.modules.pop("modal", None)
    try:
        import modal as _modal
    finally:
        sys.path = _orig_path

    if not hasattr(_modal, "Function"):
        raise RuntimeError(
            "Imported `modal` has no `Function` attribute — "
            "egomimic.modal subpackage is still shadowing the installed Modal SDK"
        )
    return _modal


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
        norm_stats: dict | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.key_map = key_map
        self.transform_list = transform_list
        self.norm_stats = norm_stats

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
        import time

        dataset_class = self._dataset_class or ZarrDataset

        # Modal fan-out: when inside Modal with the volume mounted at the standard
        # path, shard the per-episode .zattrs reads across many small containers.
        # Each worker returns (path, hash, metadata) tuples; ZarrDataset is then
        # constructed locally with precomputed_metadata, skipping the slow open.
        # Disable with EGOMIMIC_DISABLE_MODAL_LOAD=1.
        inside_modal = os.environ.get("MODAL_IS_REMOTE") == "1" or bool(
            os.environ.get("MODAL_TASK_ID")
        )
        if (
            inside_modal
            and str(search_path) == "/mnt/zarr-data"
            and os.environ.get("EGOMIMIC_DISABLE_MODAL_LOAD") != "1"
        ):
            try:
                return self._modal_fanout_load(
                    search_path, valid_folder_names, dataset_class
                )
            except Exception as e:
                logger.warning(
                    "Modal fan-out load failed (%s) — falling back to serial loop",
                    e,
                )

        all_paths = sorted(search_path.iterdir())
        datasets: dict[str, ZarrDataset] = {}
        skipped: list[str] = []
        total = len(all_paths)
        log_every = max(1, total // 20)
        start_time = time.monotonic()
        logger.info(f"Loading {total} Zarr datasets from {search_path}...")
        for i, p in enumerate(all_paths, start=1):
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
                    norm_stats=self.norm_stats,
                )
                datasets[name] = ds_obj
            except Exception as e:
                logger.error(f"Failed to load dataset at {p}: {e}")
                skipped.append(p.name)

            if i % log_every == 0 or i == total:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else float("inf")
                logger.info(
                    f"Loading Zarr datasets: {i}/{total} ({100 * i / total:.1f}%) "
                    f"| loaded={len(datasets)} skipped={len(skipped)} "
                    f"| {rate:.1f} ep/s | ETA {eta:.0f}s"
                )

        logger.info(
            f"Loaded {len(datasets)} datasets, skipped {len(skipped)} "
            f"(total scanned {total}) in {time.monotonic() - start_time:.1f}s"
        )
        return datasets

    def _modal_fanout_load(
        self,
        search_path: Path,
        valid_folder_names: set[str],
        dataset_class,
    ) -> dict:
        """Fan out per-episode .zattrs reads across Modal containers via load_shard.

        Worker returns picklable (path, hash, metadata) tuples; ZarrDataset is
        constructed locally with precomputed_metadata so its eager zarr open is
        skipped (deferred until first __getitem__).
        """
        import time

        modal = _import_real_modal()

        start_time = time.monotonic()
        logger.info(f"Modal fan-out load: listing {search_path} (single readdir)...")
        t0 = time.monotonic()
        on_disk_names = []
        for raw_name in os.listdir(search_path):
            if raw_name.startswith("."):
                continue
            hash_name = raw_name[:-5] if raw_name.endswith(".zarr") else raw_name
            if hash_name in valid_folder_names:
                on_disk_names.append(raw_name)
        total = len(on_disk_names)
        logger.info(
            f"Modal fan-out load: {total} target entries (of {len(valid_folder_names)} requested) "
            f"in {time.monotonic() - t0:.1f}s"
        )
        if total == 0:
            return {}

        n_shards = min(int(os.environ.get("EGOMIMIC_LOAD_SHARDS", "100")), total)
        shards = [on_disk_names[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]
        total_shards = len(shards)

        logger.info(
            f"Modal fan-out load: {total} entries across {total_shards} shards "
            f"(~{total // total_shards} per shard). Looking up load_shard function..."
        )

        fn = modal.Function.from_name(
            "egomimic-scan", "load_shard", environment_name="robotics"
        )
        logger.info("Modal fan-out load: function lookup OK; launching .map()...")

        datasets: dict = {}
        skipped: list[str] = []
        completed = 0
        log_every = max(1, total_shards // 20)
        for shard_result in fn.map(shards):
            completed += 1
            for path_str, episode_hash, metadata in shard_result:
                if episode_hash not in valid_folder_names:
                    continue
                try:
                    datasets[episode_hash] = dataset_class(
                        Path(path_str),
                        key_map=self.key_map,
                        transform_list=self.transform_list,
                        norm_stats=self.norm_stats,
                        precomputed_metadata=metadata,
                    )
                except Exception as e:
                    logger.error("Failed to construct dataset for %s: %s", path_str, e)
                    skipped.append(episode_hash)
            if completed % log_every == 0 or completed == total_shards:
                elapsed = time.monotonic() - start_time
                logger.info(
                    f"Modal load: shard {completed}/{total_shards} done "
                    f"| loaded={len(datasets)} skipped={len(skipped)} "
                    f"| elapsed {elapsed:.0f}s"
                )

        logger.info(
            f"Modal fan-out load complete: {len(datasets)} datasets "
            f"(skipped {len(skipped)}) in {time.monotonic() - start_time:.1f}s "
            f"({total_shards} shards)"
        )
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
            norm_stats=norm_stats,
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


class LocalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from local Zarr stores, filtering via local metadata.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        norm_stats: dict | None = None,
        allowed_episode_ids: list[str] | None = None,
    ):
        super().__init__(folder_path, key_map, transform_list, norm_stats=norm_stats)
        self.debug = debug
        self.allowed_episode_ids = (
            set(allowed_episode_ids) if allowed_episode_ids else None
        )

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

        if filters.is_empty():
            filtered = []
            for p in search_path.iterdir():
                if not p.is_dir():
                    continue
                episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name
                filtered.append((str(p), episode_hash))
            logger.info("Local paths (no filter): %d episodes", len(filtered))
        else:
            filtered = []
            for p in search_path.iterdir():
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

            logger.info("Local filtered paths: %d episodes", len(filtered))

        if debug:
            k = 10 if debug is True else int(debug)
            filtered = filtered[:k]
            logger.info("Debug mode: using first %d episodes", len(filtered))

        return filtered

    @classmethod
    def _modal_fanout_scan(
        cls,
        search_path: Path,
        filters: DatasetFilter,
        start_time: float,
    ) -> list[tuple[str, str]]:
        """Fan out the filter scan across Modal containers.

        Lists candidate dir names locally (one fast syscall), shards them, and
        invokes `egomimic-training::scan_shard` in parallel. Each worker mounts
        the same zarr volume read-only and runs a thread-pooled .zattrs scan.
        """
        import time

        modal = _import_real_modal()

        # Single readdir syscall — no per-entry stat. Drop hidden files cheaply
        # (string compare). Workers handle non-dir entries gracefully via
        # exception-swallowing in _read_metadata, so we don't need is_dir() here.
        logger.info(f"Modal fan-out scan: listing {search_path} (single readdir)...")
        t0 = time.monotonic()
        names = [n for n in os.listdir(search_path) if not n.startswith(".")]
        total = len(names)
        logger.info(
            f"Modal fan-out scan: listed {total} entries in {time.monotonic() - t0:.1f}s"
        )
        if total == 0:
            return []

        n_shards = min(int(os.environ.get("EGOMIMIC_SCAN_SHARDS", "100")), total)
        shards = [names[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]
        total_shards = len(shards)

        logger.info(
            f"Modal fan-out scan: {total} entries across {total_shards} shards "
            f"(~{total // total_shards} per shard). Looking up scan_shard function..."
        )

        fn = modal.Function.from_name(
            "egomimic-scan", "scan_shard", environment_name="robotics"
        )
        logger.info("Modal fan-out scan: function lookup OK; launching .map()...")
        filter_lambdas = list(filters.filter_lambdas)

        matched: list[tuple[str, str]] = []
        completed = 0
        log_every = max(1, total_shards // 20)
        for shard_result in fn.map(shards, [filter_lambdas] * total_shards):
            completed += 1
            matched.extend(shard_result)
            if completed % log_every == 0 or completed == total_shards:
                elapsed = time.monotonic() - start_time
                logger.info(
                    f"Modal scan: shard {completed}/{total_shards} done "
                    f"| matched={len(matched)} | elapsed {elapsed:.0f}s"
                )

        logger.info(
            f"Modal fan-out scan complete: {len(matched)} matches from {total} "
            f"entries in {time.monotonic() - start_time:.1f}s "
            f"({total_shards} shards)"
        )
        logger.info("Local filtered paths: %s", matched)
        return matched

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

        if self.allowed_episode_ids is not None:
            # Skip the full directory scan — directly resolve and load only the allowed paths.
            dataset_class = self._dataset_class or ZarrDataset
            datasets = {}
            for episode_hash in self.allowed_episode_ids:
                for candidate in (
                    self.folder_path / f"{episode_hash}.zarr",
                    self.folder_path / episode_hash,
                ):
                    if candidate.is_dir():
                        try:
                            datasets[episode_hash] = dataset_class(
                                candidate,
                                key_map=self.key_map,
                                transform_list=self.transform_list,
                                norm_stats=self.norm_stats,
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to load dataset at %s: %s", candidate, e
                            )
                        break
                else:
                    logger.warning(
                        "allowed_episode_id not found on disk: %s", episode_hash
                    )
            if not datasets:
                raise ValueError(
                    "No valid episodes found for allowed_episode_ids in the local directory."
                )
            return datasets

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


class LocalSQLEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes by querying the SQL episode table and loading matching
    zarr stores from a local volume directory.

    Filtering happens on the SQL table (fast — no per-file zarr reads).
    Episodes present in the table but absent from the local volume are skipped
    with a warning, so a partially-synced volume still works.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        norm_stats: dict | None = None,
        exclude_hashes: list[str] | None = None,
    ):
        super().__init__(folder_path, key_map, transform_list, norm_stats=norm_stats)
        self.debug = debug
        self.exclude_hashes: set[str] = set(exclude_hashes) if exclude_hashes else set()

    def resolve(
        self,
        filters: DatasetFilter | None = None,
        **kwargs,
    ) -> dict[str, "ZarrDataset"]:
        filters = _ensure_dataset_filter(filters)

        engine = create_default_engine()
        df = episode_table_to_df(engine)

        if df.empty:
            raise ValueError("SQL episode table is empty.")

        # Always exclude deleted episodes before any other filtering
        if "is_deleted" in df.columns:
            df = df[df["is_deleted"] != True]  # noqa: E712 — handles non-bool True values

        if self.exclude_hashes:
            before = len(df)
            df = df[~df["episode_hash"].isin(self.exclude_hashes)]
            logger.info("Excluded %d episodes via exclude_hashes", before - len(df))

        mask = df.apply(
            lambda row: filters.matches(_normalize_filter_row(row.to_dict())),
            axis=1,
        )
        matched = df.loc[mask, ["episode_hash", "num_frames", "robot_name"]]
        episode_hashes = matched["episode_hash"].tolist()
        logger.info("SQL filter matched %d episodes", len(episode_hashes))

        if not episode_hashes:
            raise ValueError("SQL filter matched no episodes.")

        if self.debug:
            k = 10 if self.debug is True else int(self.debug)
            matched = matched.iloc[:k]
            episode_hashes = matched["episode_hash"].tolist()
            logger.info("Debug mode: using first %d episodes", len(episode_hashes))

        dataset_class = self._dataset_class or ZarrDataset

        # Build a lookup from episode_hash → (num_frames, robot_name) from SQL
        meta_lookup = {
            row["episode_hash"]: (int(row["num_frames"]), str(row["robot_name"]))
            for _, row in matched.iterrows()
        }

        datasets: dict[str, ZarrDataset] = {}
        n_missing = 0
        for episode_hash, (num_frames, robot_name) in meta_lookup.items():
            local_path = next(
                (
                    p
                    for p in (
                        self.folder_path / episode_hash,
                        self.folder_path / f"{episode_hash}.zarr",
                    )
                    if p.is_dir()
                ),
                None,
            )
            if local_path is None:
                n_missing += 1
                logger.warning("Episode not found locally, skipping: %s", episode_hash)
                continue
            datasets[episode_hash] = dataset_class(
                local_path,
                key_map=self.key_map,
                transform_list=self.transform_list,
                norm_stats=self.norm_stats,
                _total_frames=num_frames,
                _embodiment=robot_name,
            )

        if n_missing:
            logger.info("Skipped %d episodes not present in local volume", n_missing)

        if not datasets:
            raise ValueError(
                "No episodes loaded — all SQL-matched episodes are missing from the local volume."
            )

        logger.info("Loaded %d episodes from local volume", len(datasets))
        return datasets


class MultiDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiDataset, can wrap zarr or multi dataset.

    """

    def __init__(
        self,
        datasets: dict[str, MultiDataset | ZarrDataset | ZarrActionExpertDataset],
        mode="train",
        percent=0.1,
        valid_ratio=0.2,
        **kwargs,
    ):
        """
        Args:
            datasets (dict): Dictionary mapping unique dataset hashes (str) to dataset objects. Datasets can be individual Zarr datasets or other multi-datasets; mixing different types is supported.
            mode (str, optional): Split mode to use (e.g., "train", "valid"). Defaults to "train".
            percent (float, optional): Fraction of the dataset to use from each underlying dataset. Defaults to 0.1.
            valid_ratio (float, optional): Validation split ratio for datasets that support a train/valid split.
            **kwargs: Additional keyword arguments passed to underlying dataset constructors if needed.
        """
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

        self.index_map = []
        self._global_indices_by_dataset: dict[str, list[int]] = {
            dataset_name: [] for dataset_name in self.datasets
        }
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                global_idx = len(self.index_map)
                self.index_map.append((dataset_name, local_idx))
                self._global_indices_by_dataset[dataset_name].append(global_idx)

        self.data_schematic = None
        self._n_samples_checked = 0
        self._n_violation_samples = 0
        self._violation_log_every = 100

        super().__init__()

    def __len__(self) -> int:
        return len(self.index_map)

    @staticmethod
    def _episode_name_for_dataset(dataset, dataset_name: str) -> str:
        episode_path = getattr(dataset, "episode_path", None)
        if episode_path is None:
            return dataset_name
        return Path(episode_path).name

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        if self.data_schematic is None:
            return None

        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            raise ValueError("data has no embodiment metadata")

        norm_stats = self.data_schematic.norm_stats.get(embodiment_id, {})
        if not norm_stats:
            return None

        self._n_samples_checked += 1
        prefix: str | None = None

        for key_name, stats in norm_stats.items():
            zarr_key = self.data_schematic.keyname_to_zarr_key(key_name, embodiment_id)
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
                "quantile_0_01",
                stats.get("quantile_0_1", stats["quantile_1"]),
            )
            q_high = stats.get(
                "quantile_99_99",
                stats.get("quantile_99_9", stats["quantile_99"]),
            )
            q_low = torch.as_tensor(q_low, device=arr.device, dtype=torch.float32)
            q_high = torch.as_tensor(q_high, device=arr.device, dtype=torch.float32)

            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                shape_warn_key = (str(zarr_key), tuple(arr.shape), tuple(q_low.shape))
                if shape_warn_key not in getattr(self, "_shape_mismatch_warned", set()):
                    if not hasattr(self, "_shape_mismatch_warned"):
                        self._shape_mismatch_warned = set()
                    self._shape_mismatch_warned.add(shape_warn_key)
                    logger.warning(
                        "Skipping bounds check for key=%s due to incompatible shapes: value=%s q_low=%s",
                        zarr_key,
                        tuple(arr.shape),
                        tuple(q_low.shape),
                    )
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                episode_name = self._episode_name_for_dataset(dataset, dataset_name)
                prefix = (
                    f"NaN/Inf violation ep={episode_name} frame={idx} key={zarr_key}"
                )
                break

            if torch.any(arr < q_low) or torch.any(arr > q_high):
                episode_name = self._episode_name_for_dataset(dataset, dataset_name)
                prefix = (
                    f"Bounds violation ep={episode_name} frame={idx} key={zarr_key}"
                )
                break

        if prefix is not None:
            self._n_violation_samples += 1

        if (
            self._n_samples_checked == 1
            or self._n_samples_checked % self._violation_log_every == 0
        ):
            pct = 100 * self._n_violation_samples / max(self._n_samples_checked, 1)
            print(
                f"[BoundsCheck] {self._n_samples_checked} samples checked: "
                f"{self._n_violation_samples} violations ({pct:.1f}%)"
            )

        return prefix

    def __getitem__(self, idx, _attempts: int | None = None):
        """
        Multidataset handles outlier rejection so that you don't need to propagate the norm stats down to every sub dataset.
        """
        dataset_name, local_idx = self.index_map[idx]
        dataset = self.datasets[dataset_name]
        data = dataset[local_idx]

        if isinstance(dataset, MultiDataset):
            return data

        violation = self._check_bounds(data, dataset, local_idx, dataset_name)
        if violation is not None:
            next_idx, attempts = get_fallback_idx(
                idx=idx,
                candidates=self._global_indices_by_dataset[dataset_name],
                _attempts=_attempts,
                max_attempts=len(self._global_indices_by_dataset[dataset_name]),
                exhausted_error=(
                    f"Entire dataset bad (no valid indices): dataset={dataset_name}"
                ),
            )
            return self.__getitem__(next_idx, _attempts=attempts)

        return data

    def set_data_schematic(self, data_schematic) -> None:
        """
        Set the data schematic used for top-level bounds checking.

        When child datasets are themselves MultiDatasets, recursively assign the
        same schematic so each wrapper can validate its own returned samples.
        """
        self.data_schematic = data_schematic
        for ds in self.datasets.values():
            if isinstance(ds, MultiDataset):
                ds.set_data_schematic(data_schematic)
        logger.info(
            f"Set data_schematic on MultiDataset with {len(self.datasets)} child datasets"
        )

    @classmethod
    def _from_resolver(cls, resolver: EpisodeResolver, **kwargs):
        """
        create a MultiDataset from an EpisodeResolver.

        Args:
            resolver (EpisodeResolver): The resolver instance to use for loading datasets.
            embodiment: The embodiment identifier to use for resolving datasets.
            **kwargs: Keyword args forwarded to resolver (e.g., filters,
                sync_from_s3) and MultiDataset constructor (e.g., mode, percent,
                key_map, valid_ratio).
        Returns:
            MultiDataset: The constructed multi-dataset.
        """
        # TODO add key_map and transform pass to children

        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", None)

        if isinstance(resolver, LocalEpisodeResolver):
            resolved = resolver.resolve(
                sync_from_s3=sync_from_s3,
                filters=filters,
            )
        else:
            resolved = resolver.resolve(filters=filters)

        return cls(datasets=resolved, **kwargs)


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object, Just intializes as pass through to read from zarr episode
    """

    def __init__(
        self,
        Episode_path: Path,
        key_map: dict,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        _total_frames: int | None = None,
        _embodiment: str | None = None,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            key_map: dict mapping from dataset keys to zarr keys and horizon info, e.g. {"obs/image/front": {"zarr_key": "observations.images.front", "horizon": 4}, ...}
            transform_list: list of Transform objects to apply to the data after loading, e.g. for action chunk transformations. Should be in order of application.
            norm_stats: optional dict mapping dataset key names (same keys as key_map) to
                {"quantile_1": tensor, "quantile_99": tensor} bounds. When provided, any
                loaded sample whose values fall outside [quantile_1, quantile_99] for any
                tracked key triggers the random index fallback.
            _total_frames: if provided (e.g. from SQL), skip zarr open at init and defer to first __getitem__.
            _embodiment: if provided alongside _total_frames, also skip zarr open at init.
        """
        self.episode_path = Episode_path
        self.metadata = None
        self._image_keys = None
        self._json_keys = None
        self._annotations = None
        self.episode_reader = None

        if _total_frames is not None and _embodiment is not None:
            # Fast path: metadata supplied externally — defer zarr open to first access
            self.total_frames = _total_frames
            self.embodiment = _embodiment
            self.keys_dict = {}
        else:
            self.init_episode()

        self.key_map = key_map
        self.transform = transform_list
        self.norm_stats = norm_stats or {}
        super().__init__()

    def _ensure_episode_reader(self):
        """Open the zarr store on first access if it was deferred at init."""
        if self.episode_reader is None:
            self.init_episode()

    def init_episode(self):
        """Eagerly open the zarr group and populate all metadata-derived fields."""
        self.episode_reader = ZarrEpisode(self.episode_path)
        self._init_from_metadata(self.episode_reader.metadata)

    def _init_from_metadata(self, metadata: dict) -> None:
        """Populate metadata-derived fields from a pre-loaded `.zattrs` dict."""
        self.metadata = metadata
        self.total_frames = metadata["total_frames"]
        self.embodiment = metadata["embodiment"]
        features = metadata.get("features", {})
        keys_list = (
            list(features.keys()) if isinstance(features, dict) else list(features)
        )
        self.keys_dict = {k: (0, None) for k in keys_list}
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
        """
        ZarrDataset handles jpeg decoding and transform function errors, and triggers resample on dataset level.
        """
        self._ensure_episode_reader()
        data = {}
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

            # Decode JPEG-encoded image data and normalize to [0, 1]
            # print(f"Print the image_keys: {self._image_keys}")
            if zarr_key in self._image_keys:
                jpeg_bytes = data[k]
                # Decode JPEG bytes to numpy array (H, W, 3)
                try:
                    decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
                except Exception:
                    origin = _fallback_origin if _fallback_origin is not None else idx
                    next_idx, attempts = get_fallback_idx(
                        idx=idx,
                        candidates=range(self.total_frames),
                        _attempts=_attempts,
                        max_attempts=self.total_frames,
                        exhausted_error=(
                            f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
                        ),
                    )
                    logger.warning(
                        f"JPEG decode failed ep={Path(self.episode_path).name} frame={idx} key={k} | "
                        f"attempt {attempts}, trying random idx {next_idx}"
                    )
                    result = self.__getitem__(
                        next_idx, _fallback_origin=origin, _attempts=attempts
                    )
                    return result
                data[k] = np.transpose(decoded, (2, 0, 1)) / 255.0
            elif zarr_key in self._json_keys:
                if isinstance(data[k], np.ndarray):
                    data[k] = [self._decode_json_entry(v) for v in data[k]]
                else:
                    data[k] = self._decode_json_entry(data[k])

        # Convert all numpy arrays in data to torch tensors

        # TODO add the transform list code here
        if self.transform:
            for transform in self.transform or []:
                try:
                    data = transform.transform(data)
                except Exception as e:
                    origin = _fallback_origin if _fallback_origin is not None else idx
                    next_idx, attempts = get_fallback_idx(
                        idx=idx,
                        candidates=range(self.total_frames),
                        _attempts=_attempts,
                        max_attempts=self.total_frames,
                        exhausted_error=(
                            f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
                        ),
                    )
                    logger.warning(
                        f"Transform failed ep={Path(self.episode_path).name} frame={idx} ({type(e).__name__}: {e}) | "
                        f"attempt {attempts}, trying random idx {next_idx}"
                    )
                    result = self.__getitem__(
                        next_idx, _fallback_origin=origin, _attempts=attempts
                    )
                    return result

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        data["metadata.robot_name"] = get_embodiment_id(self.embodiment)
        data["embodiment"] = get_embodiment_id(self.embodiment)
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

    def __init__(self, *args, **kwargs):
        # Set before super().__init__ so it exists regardless of whether the
        # parent goes through init_episode (eager) or _init_from_metadata (lazy).
        self._frame_to_ann_end: dict[int, int] | None = None
        super().__init__(*args, **kwargs)

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
