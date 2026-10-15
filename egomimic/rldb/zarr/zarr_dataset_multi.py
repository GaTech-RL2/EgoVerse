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
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr

from egomimic.rldb.embodiment.embodiment import (
    canonical_embodiment_name,
)
from egomimic.rldb.filters import DatasetFilter, PinError, _ensure_dataset_filter
from egomimic.rldb.leaf import EpisodeLeaf

# Re-exported: Hydra `_target_`s (MultiDataset._from_resolver) and older imports
# name these here.
from egomimic.rldb.multi import (  # noqa: F401
    SEED,
    EvenStrideDataset,
    MultiDataset,
    _IndexMap,
    pinned_episode_subset,
    split_dataset_names,
    worker_context,
)
from egomimic.rldb.resolve_memo import memoized
from egomimic.utils.env import load_env


def create_default_engine():
    from egomimic.utils.aws.aws_sql import create_default_engine as _f

    return _f()


def episode_table_to_df(engine):
    from egomimic.utils.aws.aws_sql import episode_table_to_df as _f

    return _f(engine)


logger = logging.getLogger(__name__)


def _episode_table() -> pd.DataFrame:
    return memoized(
        ("episode_table",), lambda: episode_table_to_df(create_default_engine())
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


def _embodiment_matches(name: object, expected: str) -> bool:
    """Compare an episode's embodiment string with a dataset's expected name,
    honouring the legacy vendor aliases (MECKA_BIMANUAL == human_bimanual)."""
    try:
        return canonical_embodiment_name(str(name)) == canonical_embodiment_name(
            expected
        )
    except AttributeError:
        return False


def _check_pins(
    df: pd.DataFrame, filters: DatasetFilter, expected_embodiment: str | None
) -> None:
    pins = filters.episode_hashes
    if not pins:
        return
    # First row per pinned hash, as dicts: a per-pin `.loc` took ~17 s for 41k
    # pins.
    by_hash = {}
    if "episode_hash" in df.columns:
        rows = df[df["episode_hash"].astype(str).isin(pins)]
        by_hash = {
            str(r["episode_hash"]): r
            for r in rows.drop_duplicates("episode_hash").to_dict("records")
        }
    problems: list[str] = []
    for h in sorted(pins):
        if h not in by_hash:
            problems.append(f"{h}: not in app.episodes")
            continue
        row = by_hash[h]
        if not _is_missing_filter_value(row.get("is_deleted")) and bool(
            row["is_deleted"]
        ):
            problems.append(f"{h}: is_deleted")
        if (
            _is_missing_filter_value(row.get("zarr_processed_path"))
            or not str(row["zarr_processed_path"]).strip()
        ):
            problems.append(f"{h}: empty zarr_processed_path")
        if expected_embodiment is not None:
            emb = str(row.get("embodiment", ""))
            if not _embodiment_matches(emb, expected_embodiment):
                problems.append(
                    f"{h}: embodiment '{emb}' != dataset '{expected_embodiment}'"
                )
    if problems:
        name = expected_embodiment or "<unnamed>"
        raise PinError(
            f"{len(problems)} pinned episode(s) invalid for dataset '{name}':\n  "
            + "\n  ".join(problems)
        )


def _warn_embodiment_mismatch(
    rows: pd.DataFrame, expected_embodiment: str | None, source: str
) -> None:
    if expected_embodiment is None or "embodiment" not in rows:
        return
    mismatched = rows.loc[
        ~rows["embodiment"].map(lambda e: _embodiment_matches(e, expected_embodiment))
    ]
    if len(mismatched):
        details = sorted(
            f"{h} ({e})"
            for h, e in zip(
                mismatched["episode_hash"], mismatched["embodiment"].astype(str)
            )
        )
        logger.warning(
            "%s: %d resolved episode(s) have embodiment != dataset '%s': %s",
            source,
            len(mismatched),
            expected_embodiment,
            details,
        )


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

    def _memo_fields(self) -> tuple:
        """Resolver state that affects which paths ``_compute_paths`` returns."""
        raise NotImplementedError

    def _compute_paths(
        self, filters: DatasetFilter, expected_embodiment: str | None
    ) -> list[tuple[str, str]]:
        raise NotImplementedError

    def resolve_paths(
        self,
        filters: DatasetFilter | None = None,
        expected_embodiment: str | None = None,
    ) -> list[tuple[str, str]]:
        """(path, episode_hash) pairs matching `filters`; memoized inside
        resolve_once(). Datasets are not memoized: each resolver carries its own
        keymap, so load() always runs."""
        filters = _ensure_dataset_filter(filters)
        filter_key = filters.cache_key()
        # Keyed on the _compute_paths function itself: subclasses that only
        # change loading share entries, one that overrides path finding gets its own.
        key = (
            None
            if filter_key is None
            else (
                type(self)._compute_paths,
                *self._memo_fields(),
                filter_key,
                expected_embodiment,
            )
        )
        return list(
            memoized(
                key, lambda: tuple(self._compute_paths(filters, expected_embodiment))
            )
        )

    def load(self, paths: list[tuple[str, str]]) -> dict[str, "ZarrDataset"]:
        valid = {h for _, h in paths}
        return self._load_zarr_datasets(
            search_path=self.folder_path, valid_folder_names=valid
        )

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
        validate_wrist_convention: bool = True,
    ):
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix
        self.debug = debug
        self.validate_wrist_convention = validate_wrist_convention
        super().__init__(
            folder_path,
            key_map=key_map,
            transform_list=transform_list,
        )

    # Deliberately keyed on "S3" rather than type(self).__name__: every
    # subclass shares this exact path resolution and differs only in load().
    def _memo_fields(self) -> tuple:
        return (str(self.folder_path), self.bucket_name, self.main_prefix, self.debug)

    def _compute_paths(
        self, filters: DatasetFilter, expected_embodiment: str | None
    ) -> list[tuple[str, str]]:
        self.folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Filters: {filters}")
        return self.sync_from_filters(
            bucket_name=self.bucket_name,
            filters=filters,
            local_dir=self.folder_path,
            debug=self.debug,
            expected_embodiment=expected_embodiment,
        )

    def resolve(
        self,
        filters: DatasetFilter | None = None,
        expected_embodiment: str | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters.
        Syncs S3 paths to local_root before indexing.
        """
        paths = self.resolve_paths(filters, expected_embodiment)
        if not paths:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )
        datasets = self.load(paths)
        if self.validate_wrist_convention:
            self._check_left_wrist_convention(datasets)
        return datasets

    WRIST_CONVENTION_SAMPLE = 3
    _WRIST_KEYS = (
        "left.obs_keypoints",
        "left.obs_wrist_pose",
        "right.obs_keypoints",
        "right.obs_wrist_pose",
    )

    def _check_left_wrist_convention(self, datasets: dict) -> None:
        """Cross-check ``fix_left_wrist_convention`` against the episodes.

        The flag is a property of the data, not of the code: it is right today
        only because nobody has reconverted, and a wrong setting produces a
        plausible mirrored frame rather than an error. A few sampled episodes
        re-derive the convention from their own keypoints.

        Only the double-flip raises. Enabled-but-post-fix means the correction
        is being applied to data that was already corrected, which is a fresh
        error introduced by a stale setting. Disabled-but-pre-fix is the state
        every run was in before this commit -- wrong, but not newly wrong, and
        the generic human configs (aria, human, the cotrains) have never set
        the flag; turning their startup into a crash is not this correction's
        business. It warns, naming the knob. An episode that matches neither
        convention (a handful of 2025 ones do) warns too.
        """
        from egomimic.rldb.transforms import RotateLocalFrame
        from egomimic.scripts.mecka_process.mecka_to_zarr import left_wrist_convention

        fix_enabled = any(
            isinstance(t, RotateLocalFrame) for t in (self.transform_list or [])
        )
        for name, ds in list(datasets.items())[: self.WRIST_CONVENTION_SAMPLE]:
            try:
                store = zarr.open_group(str(ds.episode_path), mode="r")
                if any(k not in store for k in self._WRIST_KEYS):
                    continue
                verdict = left_wrist_convention(store)
            except Exception as e:  # a probe must never take a run down
                logger.warning(f"[wrist-convention] {name}: probe failed ({e})")
                continue
            if verdict == "unknown":
                logger.warning(
                    f"[wrist-convention] {name}: neither convention matches its own "
                    "keypoints (right-hand control failed); cannot check "
                    f"fix_left_wrist_convention={fix_enabled} against it"
                )
                continue
            if verdict == "post_fix" and fix_enabled:
                raise ValueError(
                    f"[wrist-convention] {name} was written post_fix, but this "
                    "dataset runs with fix_left_wrist_convention=true: the "
                    "Rz(180) correction would double-flip data the converter "
                    "already fixed. Drop the flag for reconverted data."
                )
            if verdict == "pre_fix" and not fix_enabled:
                logger.warning(
                    f"[wrist-convention] {name} was written pre_fix (its LEFT "
                    "wrist is double-mirrored) and this dataset does not set "
                    "fix_left_wrist_convention: it trains on the uncorrected "
                    "frame. Set the flag on this data config unless you mean to."
                )

    @staticmethod
    def _get_filtered_paths(
        filters: DatasetFilter | None = None,
        debug: int | bool | None = None,
        expected_embodiment: str | None = None,
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
        df = _episode_table()
        # Before the empty-table early return: pins against an empty table are
        # still invalid and must raise rather than silently resolve to nothing.
        _check_pins(df, filters, expected_embodiment)
        if df.empty:
            logger.info("Episode table is empty.")
            return []

        if filters.episode_hashes:
            # matches() rejects every other row anyway; the table has ~10x
            # more rows than a full-lab pin list and 1e5x more than a val pin.
            df = df[df["episode_hash"].astype(str).isin(filters.episode_hashes)]
        # records, not a row-wise df.apply(row.to_dict()): ~20x faster on the
        # full table, where the apply cost ~50 s per resolve.
        mask = [
            filters.matches(_normalize_filter_row(row)) for row in df.to_dict("records")
        ]
        matched = df.loc[mask] if mask else df
        n_matched_sql = len(matched)

        matched = matched[
            matched["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
        ]
        n_skipped_null = n_matched_sql - len(matched)
        if n_skipped_null:
            logger.info(
                "Skipped %d episodes with null/empty zarr_processed_path.",
                n_skipped_null,
            )

        # Warn only about episodes that actually make it into the dataset.
        _warn_embodiment_mismatch(matched, expected_embodiment, "S3EpisodeResolver")
        output = matched[["zarr_processed_path", "episode_hash"]]

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
        expected_embodiment: str | None = None,
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
        filtered_paths = cls._get_filtered_paths(
            filters, debug=debug, expected_embodiment=expected_embodiment
        )
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

    def resolve(
        self, filters=None, expected_embodiment=None
    ) -> dict[str, "ZarrDataset"]:
        datasets = super().resolve(
            filters=filters, expected_embodiment=expected_embodiment
        )
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

    @classmethod
    def _get_local_filtered_paths(
        cls,
        search_path: Path,
        filters: DatasetFilter | None = None,
        debug: int | bool | None = None,
        expected_embodiment: str | None = None,
    ):
        filters = _ensure_dataset_filter(filters)
        if not search_path.is_dir():
            logger.warning("Local path does not exist: %s", search_path)
            return []

        rows: list[dict] = []  # one normalized attrs row per episode dir
        paths_by_hash: dict[str, str] = {}
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
            rows.append(_normalize_filter_row(metadata, episode_hash=episode_hash))
            paths_by_hash[episode_hash] = str(p)

        if filters.episode_hashes:
            present = {r["episode_hash"]: r for r in rows}
            problems = []
            for h in sorted(filters.episode_hashes):
                if h not in present:
                    problems.append(f"{h}: not in local directory {search_path}")
                elif expected_embodiment is not None:
                    emb = str(present[h].get("embodiment", ""))
                    if not _embodiment_matches(emb, expected_embodiment):
                        problems.append(
                            f"{h}: embodiment '{emb}' != dataset '{expected_embodiment}'"
                        )
            if problems:
                raise PinError(
                    f"{len(problems)} pinned episode(s) invalid for dataset "
                    f"'{expected_embodiment or '<unnamed>'}':\n  "
                    + "\n  ".join(problems)
                )

        # Evaluate the filters once; both the path list and the mismatch warning
        # below are derived from the same matched rows.
        matched = [r for r in rows if filters.matches(r)]
        filtered = [
            (paths_by_hash[r["episode_hash"]], r["episode_hash"]) for r in matched
        ]
        if expected_embodiment is not None:
            bad = sorted(
                (r["episode_hash"], str(r.get("embodiment", "")))
                for r in matched
                if not _embodiment_matches(r.get("embodiment", ""), expected_embodiment)
            )
            if bad:
                details = [f"{h} ({e})" for h, e in bad]
                logger.warning(
                    "LocalEpisodeResolver: %d resolved episode(s) have embodiment != "
                    "dataset '%s': %s",
                    len(bad),
                    expected_embodiment,
                    details,
                )

        if debug is not None and debug is not False:
            k = min(10 if debug is True else int(debug), len(filtered))
            if k < len(filtered):
                logger.info("Debug mode: limiting to %d datasets.", k)
            filtered = filtered[:k]

        logger.info("Local filtered paths: %s", filtered)
        return filtered

    # Deliberately keyed on "Local" rather than type(self).__name__: every
    # subclass shares this exact path resolution and differs only in load().
    def _memo_fields(self) -> tuple:
        return (str(self.folder_path), self.debug)

    def _compute_paths(
        self, filters: DatasetFilter, expected_embodiment: str | None
    ) -> list[tuple[str, str]]:
        return self._get_local_filtered_paths(
            self.folder_path,
            filters,
            debug=self.debug,
            expected_embodiment=expected_embodiment,
        )

    def resolve(
        self,
        sync_from_s3=False,
        filters: DatasetFilter | None = None,
        expected_embodiment: str | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters from local data.
        """
        if sync_from_s3:
            logger.warning(
                "LocalEpisodeResolver does not sync from S3; ignoring sync_from_s3=True."
            )

        paths = self.resolve_paths(filters, expected_embodiment)
        if not paths:
            raise ValueError(
                "No valid collection names from local filtering: "
                "filters matched no episodes in the local directory."
            )

        return self.load(paths)


class ZarrDataset(EpisodeLeaf):
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
        super().__init__(key_map=key_map, transform_list=transform_list)

    def init_episode(self):
        """
        inits the zarr episode and all the metadata associated, as well as total_frames for len
        """
        self.episode_reader = ZarrEpisode(self.episode_path)
        self.metadata = self.episode_reader.metadata
        self.total_frames = self.metadata["total_frames"]
        self.embodiment = self.metadata["embodiment"]
        self.keys_dict = {k: (0, None) for k in self.episode_reader._collect_keys()}
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()

    @property
    def intrinsics(self) -> np.ndarray | dict[str, np.ndarray] | None:
        """Pass-through to ZarrEpisode.intrinsics (read from zarr metadata)."""
        return self.episode_reader.intrinsics

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
            windows = self._read_windows(idx)
            # One zarr call per key, not one per keymap entry: an action chunk
            # and the proprio history of the same pose name the same key with
            # different windows, and zarr's per-call metadata hop costs more
            # than the handful of extra rows a merged read carries. Windows are
            # merged only across rows in chunks the read fetches anyway (see
            # ZarrEpisode.plan_reads), so two frames far apart on a JPEG key
            # stay two rows.
            intervals: dict[str, list[tuple[int, int]]] = {}
            for zarr_key, window in windows.values():
                if isinstance(window, np.ndarray):
                    spans = [(int(i), int(i) + 1) for i in np.unique(window)]
                else:
                    start, end = window
                    spans = [(start, start + 1 if end is None else end)]
                intervals.setdefault(zarr_key, []).extend(spans)
            groups = self.episode_reader.read_intervals(intervals)

            def _rows(zarr_key: str, start: int, stop: int):
                for lo, rows in groups[zarr_key]:
                    if lo <= start and stop <= lo + len(rows):
                        return rows[start - lo : stop - lo]
                # A StopIteration escaping __getitem__ reads as "epoch over"
                # to the DataLoader; a short array must fail loudly.
                raise ValueError(
                    f"ep={Path(self.episode_path).name} key={zarr_key}: "
                    f"window [{start}, {stop}) is past the rows read "
                    f"{[(lo, lo + len(rows)) for lo, rows in groups[zarr_key]]}"
                )

            data = {}
            retry = False
            for k in self.key_map:
                zarr_key = self.key_map[k]["zarr_key"]
                key_type = self.key_map[k].get("key_type", None)
                horizon = self.key_map[k].get("horizon", None)

                if key_type == "annotation_keys":
                    data[k] = self._annotation_text_for_frame(idx)
                    continue

                window = windows[k][1]
                if isinstance(window, np.ndarray):
                    # (K,) rows; np.stack copies, so nothing aliases the read.
                    rows = [_rows(zarr_key, int(i), int(i) + 1)[0] for i in window]
                    chunk = rows if zarr_key in self._image_keys else np.stack(rows)
                else:
                    start, end = window
                    chunk = _rows(zarr_key, start, start + 1 if end is None else end)
                    chunk = chunk[0] if end is None else chunk
                    if isinstance(chunk, np.ndarray):
                        # A slice of a shared read is a view; every keymap entry
                        # owned its own array before this coalescing and
                        # transforms are free to write into it.
                        chunk = chunk.copy()
                data[k] = self._pad_to_horizon(chunk, horizon)

                if zarr_key in self._image_keys:
                    try:
                        if isinstance(data[k], list):
                            # A history window, (K, H, W, 3): decode each
                            # distinct frame once (front padding repeats one).
                            window = windows[k][1]
                            unique = {
                                int(i): simplejpeg.decode_jpeg(b, colorspace="RGB")
                                for i, b in zip(window, data[k])
                            }
                            decoded = np.stack([unique[int(i)] for i in window])
                        else:
                            decoded = simplejpeg.decode_jpeg(data[k], colorspace="RGB")
                    except Exception:
                        idx = _next("JPEG decode failed", key=k)
                        retry = True
                        break
                    # float32, not numpy's float64 default for `uint8 / float`:
                    # process_batch_for_training casts to float32 on the GPU
                    # anyway, so the extra 4 bytes a pixel buy nothing and
                    # double the collate, the pin and the H2D copy. At 360x640
                    # and batch 64 that is 338 MB a camera a batch instead of
                    # 169. The result is bit-identical: there are only 256
                    # possible values and float32(u)/float32(255) equals
                    # float32(u / 255.0) for every one of them (see
                    # tests/unit/test_image_decode_dtype.py).
                    data[k] = np.moveaxis(decoded, -1, -3).astype(
                        np.float32
                    ) / np.float32(255.0)
                elif zarr_key in self._json_keys:
                    if isinstance(data[k], np.ndarray):
                        data[k] = [self._decode_json_entry(v) for v in data[k]]
                    else:
                        data[k] = self._decode_json_entry(data[k])
            if retry:
                continue

            _ = origin  # preserved for symmetry with prior API
            return self._finish(idx, data)


class ZarrAnnotationCutoffDataset(ZarrDataset):
    """ZarrDataset that clamps action chunks at the end of the enclosing annotation.

    Standard chunking from the start frame, but action reads stop at EOS+1 of the
    annotation span containing the start frame. The chunk is then padded out to
    ``horizon`` via the base ``_pad_to_horizon`` (repeat-last), so frames beyond
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
        n_spans = 0
        for ann in self._load_annotations():
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx < 0 or end_idx <= start_idx:
                continue
            n_spans += 1
            for idx in range(start_idx, end_idx):
                mapping[idx] = end_idx
        # Visibility into annotation-cutoff usage: spans/frames_covered of 0
        # means the cutoff is a no-op for this episode. One line per episode
        # per rank is thousands of lines on a real split, so this is DEBUG and
        # the resolver's kept N/M line carries the aggregate.
        logger.debug(
            "[AnnotationCutoff] ep=%s spans=%d frames_covered=%d/%d",
            Path(self.episode_path).name,
            n_spans,
            len(mapping),
            self.total_frames,
        )
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


def _episode_has_annotation_spans(ds: "ZarrDataset") -> bool:
    """True if the episode has at least one usable ``[start_idx, end_idx)`` span.

    Many Scale-"completed" episodes have an empty (or span-less) zarr
    ``annotations`` array because the annotation-injection step lagged; the
    AnnotationCutoff is a no-op for those, so they should be dropped when the
    point of the run is to clamp chunks at annotation boundaries.
    """
    try:
        anns = ds._load_annotations()
    except Exception:
        return False
    return any(
        isinstance(a, dict)
        and 0 <= int(a.get("start_idx", -1)) < int(a.get("end_idx", -1))
        for a in anns
    )


class S3AnnotationCutoffEpisodeResolver(S3EpisodeResolver):
    """S3EpisodeResolver that loads ZarrAnnotationCutoffDataset instances.

    When ``require_annotations`` is set (default), episodes whose zarr
    ``annotations`` array has no usable span are dropped; otherwise the
    annotation cutoff would silently no-op on them.
    """

    _dataset_class = ZarrAnnotationCutoffDataset

    def __init__(self, *args, require_annotations: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.require_annotations = require_annotations

    def resolve(self, filters=None, expected_embodiment=None):
        datasets = super().resolve(
            filters=filters, expected_embodiment=expected_embodiment
        )
        if not self.require_annotations:
            return datasets
        kept = {
            h: ds for h, ds in datasets.items() if _episode_has_annotation_spans(ds)
        }
        dropped = sorted(set(datasets) - set(kept))
        if dropped:
            logger.warning(
                "[AnnotationCutoff] dropped %d/%d episodes with no usable "
                "annotation spans (e.g. %s)",
                len(dropped),
                len(datasets),
                dropped[:5],
            )
        logger.info(
            "[AnnotationCutoff] kept %d/%d episodes with usable annotations",
            len(kept),
            len(datasets),
        )
        if not kept:
            raise ValueError(
                "[AnnotationCutoff] no resolved episodes contain usable annotation "
                "spans; check the filter / annotation injection for this dataset."
            )
        return kept


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

    @property
    def intrinsics(self) -> np.ndarray | dict[str, np.ndarray] | None:
        """
        Camera intrinsics persisted in zarr metadata, deserialized to ndarray(s).

        Returns:
            - np.ndarray for a single-camera episode,
            - dict[str, np.ndarray] for multi-camera episodes,
            - None if no intrinsics were written.
        """
        raw = self.metadata.get("intrinsics")
        if raw is None:
            return None
        if isinstance(raw, dict):
            return {k: np.asarray(v) for k, v in raw.items()}
        return np.asarray(raw)

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

    @staticmethod
    def plan_reads(
        intervals: list[tuple[int, int]], chunk_len: int
    ) -> list[tuple[int, int]]:
        """Merge half-open ``[start, end)`` intervals on one key into read groups.

        Two merge only when every row between them lies in a chunk one of them
        already touches, so the bridge costs no extra chunk fetch: the windows
        on a numeric key (100-row chunks) collapse into one read, while two
        frames far apart on a JPEG key (one frame per chunk) stay two rows
        rather than every frame in between."""
        groups: list[list[int]] = []
        for start, end in sorted(intervals):
            if groups and start // chunk_len <= (groups[-1][1] - 1) // chunk_len + 1:
                groups[-1][1] = max(groups[-1][1], end)
            else:
                groups.append([start, end])
        return [(lo, hi) for lo, hi in groups]

    def read_intervals(
        self, intervals: dict[str, list[tuple[int, int]]]
    ) -> dict[str, list[tuple[int, np.ndarray]]]:
        """One zarr call per key for all its ``[start, end)`` intervals.

        Returns ``{key: [(group_start, rows), ...]}``, one entry per group of
        ``plan_reads``; several groups are fetched with a single orthogonal
        index."""
        result = {}
        for key, ivs in intervals.items():
            arr = self._store[key]
            groups = self.plan_reads(ivs, arr.chunks[0])
            if len(groups) == 1:
                lo, hi = groups[0]
                result[key] = [(lo, arr[lo:hi])]
                continue
            rows = arr.oindex[np.concatenate([np.arange(lo, hi) for lo, hi in groups])]
            out, offset = [], 0
            for lo, hi in groups:
                out.append((lo, rows[offset : offset + hi - lo]))
                offset += hi - lo
            result[key] = out
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
