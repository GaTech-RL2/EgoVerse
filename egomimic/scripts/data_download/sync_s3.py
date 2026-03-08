"""
Sync EgoVerse data from S3/R2 to a local directory.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Sync EgoVerse data from S3/R2 to a local directory."
    )
    parser.add_argument(
        "--local-dir", type=str, required=True, help="Local directory to sync into."
    )
    parser.add_argument(
        "--workers", type=int, default=128, help="s5cmd parallel workers."
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=True,
        help='JSON dict of SQL filters, e.g. \'{"lab": "mecka"}\' or \'{"episode_hash": "h1"}\'.',
    )
    args = parser.parse_args()

    filters = json.loads(args.filters)
    if not isinstance(filters, dict):
        raise ValueError("--filters must be a JSON object (dict).")

    load_env()
    S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=filters,
        local_dir=Path(args.local_dir),
        numworkers=args.workers,
    )


if __name__ == "__main__":
    main()
