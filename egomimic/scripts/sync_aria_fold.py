"""Sync the aria fold_clothes episodes (709 in SQL, processed_v3/aria/) into
the shared PACE mirror. Skips episodes already present; safe to re-run."""

from pathlib import Path

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

MIRROR = Path("/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets")


def main():
    load_env()
    filters = DatasetFilter(
        filter_lambdas=[
            "lambda row: '/aria/' in str(row.get('zarr_processed_path', ''))",
            "lambda row: str(row.get('task', '')) == 'fold_clothes'",
        ]
    )
    paths = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=filters,
        local_dir=MIRROR,
        numworkers=20,
    )
    print(f"[sync_aria_fold] synced/verified {len(paths)} episodes", flush=True)


if __name__ == "__main__":
    main()
