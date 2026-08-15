from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver

logger = logging.getLogger(__name__)


class ManifestEpisodeResolver(S3EpisodeResolver):
    """Resolve and sync episodes listed in a frozen CSV manifest.

    Skips the SQL episode table so a hackathon / Modal job can train from
    `zarr_processed_path` + `episode_hash` alone.
    """

    def __init__(
        self,
        folder_path: Path,
        manifest_csv: str | Path,
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v3",
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        norm_stats: dict | None = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        super().__init__(
            folder_path,
            bucket_name=bucket_name,
            main_prefix=main_prefix,
            key_map=key_map,
            transform_list=transform_list,
            debug=debug,
            norm_stats=norm_stats,
        )

    def _paths_from_manifest(self) -> list[tuple[str, str]]:
        if not self.manifest_csv.is_file():
            raise FileNotFoundError(f"Missing training manifest: {self.manifest_csv}")

        df = pd.read_csv(self.manifest_csv)
        required = {"episode_hash", "zarr_processed_path"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                f"{self.manifest_csv} missing columns: {sorted(missing)}"
            )
        if df["episode_hash"].duplicated().any():
            raise RuntimeError(f"{self.manifest_csv} has duplicate episode_hash values")

        paths = list(
            zip(
                df["zarr_processed_path"].astype(str).tolist(),
                df["episode_hash"].astype(str).tolist(),
            )
        )
        if self.debug is not None and self.debug is not False:
            k = min(10 if self.debug is True else int(self.debug), len(paths))
            paths = paths[:k]
        return paths

    def resolve(
        self,
        filters: DatasetFilter | None = None,
    ) -> dict[str, object]:
        if filters is not None:
            logger.info(
                "ManifestEpisodeResolver ignores DatasetFilter; "
                "the CSV is the episode set."
            )

        if not self.folder_path.is_dir():
            self.folder_path.mkdir(parents=True, exist_ok=True)

        filtered_paths = self._paths_from_manifest()
        self.sync_from_paths(
            bucket_name=self.bucket_name,
            s3_paths=filtered_paths,
            local_dir=self.folder_path,
        )

        valid_hashes = {episode_hash for _, episode_hash in filtered_paths}
        if not valid_hashes:
            raise ValueError(f"Manifest listed no episodes: {self.manifest_csv}")

        return self._load_zarr_datasets(
            search_path=self.folder_path,
            valid_folder_names=valid_hashes,
        )

    @classmethod
    def sync_from_paths(
        cls,
        bucket_name: str,
        s3_paths: list[tuple[str, str]],
        local_dir: Path,
        numworkers: int = 10,
    ) -> list[tuple[str, str]]:
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=s3_paths,
            local_dir=local_dir,
            numworkers=numworkers,
        )
        return s3_paths
