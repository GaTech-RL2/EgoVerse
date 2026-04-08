"""
Sync-only entry point that reuses Hydra training configs.

Instantiates the S3EpisodeResolver(s) and DatasetFilter(s) defined under
`cfg.data.train_datasets[*]` / `cfg.data.valid_datasets[*]` and calls
`S3EpisodeResolver.sync_from_filters(...)`. No model / trainer / Modal
allocation happens.

Example:
    python egomimic/scripts/data_download/sync_from_config.py \\
        --config-name train_zarr_cartesian_pi \\
        data=mecka_pi
"""

import logging
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _sync_split(split_name: str, datasets_cfg: DictConfig, numworkers: int) -> None:
    seen: set[tuple[str, str]] = set()
    for ds_name, ds_cfg in datasets_cfg.items():
        resolver: S3EpisodeResolver = hydra.utils.instantiate(ds_cfg.resolver)
        if not isinstance(resolver, S3EpisodeResolver):
            log.info(
                f"[{split_name}/{ds_name}] resolver is {type(resolver).__name__}, "
                "not S3EpisodeResolver — skipping sync."
            )
            continue

        filters = (
            hydra.utils.instantiate(ds_cfg.filters)
            if "filters" in ds_cfg and ds_cfg.filters is not None
            else None
        )

        local_dir = Path(resolver.folder_path)
        key = (resolver.bucket_name, str(local_dir))
        if key in seen:
            log.info(
                f"[{split_name}/{ds_name}] already synced "
                f"bucket={resolver.bucket_name} -> {local_dir}; skipping."
            )
            continue
        seen.add(key)

        log.info(
            f"[{split_name}/{ds_name}] syncing bucket={resolver.bucket_name} "
            f"-> {local_dir} with filters={filters}"
        )
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            S3EpisodeResolver.sync_from_filters(
                bucket_name=resolver.bucket_name,
                filters=filters,
                local_dir=local_dir,
                numworkers=numworkers,
                debug=resolver.debug,
            )
        except subprocess.CalledProcessError as e:
            # s5cmd `run` processes every line in the batch and only exits non-zero
            # when at least one failed (e.g. "no object found" for a stale DB row).
            # Tolerate that here: the successful episodes are still on disk, and we
            # want the next split / dataset to keep going.
            log.warning(
                f"[{split_name}/{ds_name}] s5cmd reported failures (exit={e.returncode}); "
                "continuing. Inspect the ERROR lines above to see which episode prefixes "
                "are missing in S3."
            )


@hydra.main(
    version_base="1.3",
    config_path="../../hydra_configs",
    config_name="train_zarr_cartesian_pi",
)
def main(cfg: DictConfig) -> None:
    load_env()

    numworkers = int(cfg.get("sync_workers", 128))

    if "train_datasets" in cfg.data and cfg.data.train_datasets:
        _sync_split("train", cfg.data.train_datasets, numworkers)
    if "valid_datasets" in cfg.data and cfg.data.valid_datasets:
        _sync_split("valid", cfg.data.valid_datasets, numworkers)


if __name__ == "__main__":
    main()
