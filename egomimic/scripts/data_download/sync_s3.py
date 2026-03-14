"""
Sync EgoVerse data from S3/R2 to a local directory.

Example:
    python egomimic/scripts/data_download/sync_s3.py --local-dir /tmp/egoverse \
        --filters aria-fold-clothes
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

logging.basicConfig(level=logging.INFO, format="%(message)s")


DEFAULT_FILTERS = {
    "aria-fold-clothes": DatasetFilter(
        filter_lambdas=[
            "lambda row: row.get('embodiment') == 'aria'",
            "lambda row: row.get('task') == 'fold_clothes'",
        ]
    ),
}


def parse_dataset_filter_key(filter_key: str) -> DatasetFilter:
    try:
        return DEFAULT_FILTERS[filter_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown filter key {filter_key!r}. "
            f"Available filter keys: {sorted(DEFAULT_FILTERS)}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Sync EgoVerse data from S3/R2 to a local directory."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local directory to sync into.",
    )
    parser.add_argument(
        "--workers", type=int, default=128, help="s5cmd parallel workers."
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=True,
        help=(
            "Named DatasetFilter preset key. "
            f"Available keys: {', '.join(sorted(DEFAULT_FILTERS))}"
        ),
    )
    args = parser.parse_args()

    filters = parse_dataset_filter_key(args.filters)

    load_env()
    S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=filters,
        local_dir=Path(args.local_dir),
        numworkers=args.workers,
    )


if __name__ == "__main__":
    main()
