#!/usr/bin/env python3
"""
Conversion script that updates zarr dataset attributes from the SQL table.

For each matched episode this script:
1. Reads the remote zarr.json (zarr v3 root metadata) from R2/S3
2. Sets the 'embodiment' attribute to the SQL row's robot_name value
3. Removes 'robot_name' from the zarr attributes if present
4. Adds all other SQL table fields that are not yet present in the zarr attributes
5. Writes the updated zarr.json back to R2/S3

Usage:
    python update_zarr_attrs_from_sql.py \
        --filters '{"lab": "rl2", "task": "object in container"}' \
        --dry-run
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_table_row,
    episode_table_to_df,
)

log = logging.getLogger(__name__)

# SQL fields that are infrastructure/path fields and should not be written into
# the zarr attributes.
_SKIP_SQL_FIELDS = {
    "zarr_processed_path",
    "zarr_mp4_path",
    "processed_path",
    "mp4_path",
    "processing_error",
    "zarr_processing_error",
    # handled separately: robot_name value → embodiment
    "robot_name",
    "embodiment",
}


def parse_filters(filters_str: str) -> dict:
    try:
        return json.loads(filters_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in filters: {e}")


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) from an s3://bucket/key URL."""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def download_zarr_json(s3, bucket: str, key_prefix: str) -> dict | None:
    """Download and parse zarr.json from the zarr store root."""
    zarr_json_key = key_prefix.rstrip("/") + "/zarr.json"
    try:
        response = s3.get_object(Bucket=bucket, Key=zarr_json_key)
        return json.loads(response["Body"].read())
    except Exception as e:
        log.error(f"Error downloading zarr.json ({zarr_json_key}): {e}")
        return None


def upload_zarr_json(s3, bucket: str, key_prefix: str, zarr_doc: dict) -> bool:
    """Upload updated zarr.json back to the zarr store root."""
    zarr_json_key = key_prefix.rstrip("/") + "/zarr.json"
    try:
        s3.put_object(
            Bucket=bucket,
            Key=zarr_json_key,
            Body=json.dumps(zarr_doc, indent=2).encode(),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        log.error(f"Error uploading zarr.json ({zarr_json_key}): {e}")
        return False


def build_updated_attrs(attrs: dict, sql_row_dict: dict) -> tuple[dict, list[str]]:
    """
    Return (updated_attrs, change_log) applying the conversion rules:
      - embodiment  ← sql robot_name value
      - robot_name  removed if present
      - all other non-skip SQL fields added only if not already in attrs
    """
    updated = dict(attrs)
    changes = []

    # Set embodiment from robot_name
    robot_name = sql_row_dict.get("robot_name", "")
    old_embodiment = updated.get("embodiment")
    if old_embodiment != robot_name:
        updated["embodiment"] = robot_name
        changes.append(f"embodiment: {old_embodiment!r} → {robot_name!r}")

    # Remove robot_name
    if "robot_name" in updated:
        del updated["robot_name"]
        changes.append("removed robot_name")

    # Add missing SQL fields
    for field, value in sql_row_dict.items():
        if field in _SKIP_SQL_FIELDS:
            continue
        if field not in updated:
            updated[field] = value
            changes.append(f"added {field}={value!r}")

    return updated, changes


def process_episode(
    s3,
    zarr_processed_path: str,
    sql_row_dict: dict,
    dry_run: bool,
) -> bool:
    """Process a single episode. Returns True on success."""
    bucket, key_prefix = parse_s3_url(zarr_processed_path)

    # Download zarr.json
    zarr_doc = download_zarr_json(s3, bucket, key_prefix)
    if zarr_doc is None:
        return False

    if "attributes" not in zarr_doc:
        log.warning("zarr.json has no 'attributes' key — initialising empty")
        zarr_doc["attributes"] = {}

    attrs = zarr_doc["attributes"]
    updated_attrs, changes = build_updated_attrs(attrs, sql_row_dict)

    if not changes:
        log.info("No changes needed")
        return True

    for c in changes:
        log.info(c)

    if dry_run:
        log.info("[DRY RUN] Would upload updated zarr.json")
        return True

    zarr_doc["attributes"] = updated_attrs
    return upload_zarr_json(s3, bucket, key_prefix, zarr_doc)


def main():
    parser = argparse.ArgumentParser(
        description="Update zarr dataset attributes from SQL table"
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=True,
        help='JSON filter dict for SQL table, e.g. \'{"lab": "rl2"}\'',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of episodes to process (for testing)",
    )
    args = parser.parse_args()

    try:
        filters = parse_filters(args.filters)
    except ValueError as e:
        log.error(f"Error: {e}")
        return 1

    log.info(f"Filters: {filters}")
    log.info(f"Dry run: {args.dry_run}")

    # Connect to database
    log.info("Connecting to database...")
    engine = create_default_engine()

    # Query and filter episodes
    log.info("Querying episodes from SQL table...")
    df = episode_table_to_df(engine)

    for key, value in filters.items():
        if key not in df.columns:
            log.warning(f"Filter key '{key}' not found in table columns")
            continue
        df = df[df[key] == value]

    if df.empty:
        log.info("No episodes found matching filters")
        return 0

    # Only process episodes that have a zarr_processed_path
    df = df[
        df["zarr_processed_path"].notna()
        & (df["zarr_processed_path"].str.strip() != "")
    ]
    if df.empty:
        log.info("No episodes with zarr_processed_path found")
        return 0

    log.info(f"Found {len(df)} episodes with zarr_processed_path")

    if args.limit:
        df = df.head(args.limit)
        log.info(f"Limited to {len(df)} episodes")

    s3 = get_boto3_s3_client()
    success_count = 0
    error_count = 0

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        episode_hash = row["episode_hash"]
        zarr_processed_path = row["zarr_processed_path"]
        log.info(f"[{idx}/{len(df)}] {episode_hash}")
        log.info(f"  zarr path: {zarr_processed_path}")

        # Fetch the full SQL row as a dict
        table_row = episode_hash_to_table_row(engine, episode_hash)
        if table_row is None:
            log.warning("Episode not found in SQL table")
            error_count += 1
            continue
        sql_row_dict = asdict(table_row)

        ok = process_episode(s3, zarr_processed_path, sql_row_dict, args.dry_run)
        if ok:
            success_count += 1
        else:
            error_count += 1

    log.info(f"Total: {len(df)}  |  Success: {success_count}  |  Errors: {error_count}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
