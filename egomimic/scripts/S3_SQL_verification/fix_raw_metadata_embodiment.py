#!/usr/bin/env python3
"""
Fix embodiment/robot_name in raw metadata JSON files stored in raw_v2/ on R2/S3.

For each matched episode this script:
1. Locates <timestamp>_metadata.json in raw_v2/<embodiment>/ via SQL episode_hash
2. Downloads the JSON
3. Sets 'embodiment' to the SQL row's embodiment value (already migrated from robot_name)
4. Removes 'robot_name' from the JSON if present
5. Adds any other SQL fields not yet present in the JSON
6. Uploads the updated JSON back to R2/S3

Usage:
    python3 fix_raw_metadata_embodiment.py \
        --filters '{"lab": "rl2"}' \
        --dry-run
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_table_row,
    episode_hash_to_timestamp_ms,
    episode_table_to_df,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# SQL fields that should not be written into raw metadata JSONs
_SKIP_SQL_FIELDS = {
    "zarr_processed_path",
    "zarr_mp4_path",
    "processed_path",
    "mp4_path",
    "processing_error",
    "zarr_processing_error",
    # handled separately
    "robot_name",
    "embodiment",
}


def parse_filters(filters_str: str) -> dict:
    try:
        return json.loads(filters_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in filters: {e}")


def find_raw_metadata_key(
    s3, bucket: str, embodiment: str, timestamp_ms: int
) -> str | None:
    """
    Find the _metadata.json key in raw_v2/<embodiment>/ matching the given timestamp.
    Tries an exact match first, then scans within ±1 second in 100ms steps.
    """
    prefix = f"raw_v2/{embodiment.lower()}/"
    paginator = s3.get_paginator("list_objects_v2")

    candidates: dict[int, str] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("_metadata.json"):
                continue
            stem = Path(key).stem.replace("_metadata", "")
            try:
                candidates[int(stem)] = key
            except ValueError:
                pass

    if timestamp_ms in candidates:
        return candidates[timestamp_ms]

    for delta in range(-1000, 1001, 100):
        ts = timestamp_ms + delta
        if ts in candidates:
            return candidates[ts]

    return None


def build_updated_json(raw_json: dict, sql_row_dict: dict) -> tuple[dict, list[str]]:
    """
    Apply fixes to a raw metadata JSON dict.
    Returns (updated_dict, change_log).
    """
    updated = dict(raw_json)
    changes = []

    # Set embodiment from robot_name in the JSON itself, then remove robot_name
    robot_name = updated.get("robot_name", "")
    old_embodiment = updated.get("embodiment")
    if robot_name and old_embodiment != robot_name:
        updated["embodiment"] = robot_name
        changes.append(f"embodiment: {old_embodiment!r} → {robot_name!r}")

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
    bucket: str,
    episode_hash: str,
    sql_row_dict: dict,
    dry_run: bool,
) -> bool:
    embodiment = sql_row_dict.get("embodiment", "")
    try:
        timestamp_ms = episode_hash_to_timestamp_ms(episode_hash)
    except Exception as e:
        print(f"  Error converting episode_hash to timestamp: {e}")
        return False

    key = find_raw_metadata_key(s3, bucket, embodiment, timestamp_ms)
    if key is None:
        print(
            f"  Warning: no _metadata.json found in raw_v2/{embodiment.lower()}/ for ts={timestamp_ms}"
        )
        return False

    print(f"  metadata key: {key}")

    # Download
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        raw_json = json.loads(response["Body"].read())
    except Exception as e:
        print(f"  Error downloading {key}: {e}")
        return False

    updated_json, changes = build_updated_json(raw_json, sql_row_dict)

    if not changes:
        print("  No changes needed")
        return True

    for c in changes:
        print(f"  {c}")

    if dry_run:
        print("  [DRY RUN] Would upload updated metadata JSON")
        return True

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(updated_json, indent=2).encode(),
            ContentType="application/json",
        )
    except Exception as e:
        print(f"  Error uploading {key}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix embodiment/robot_name in raw_v2 metadata JSONs on R2/S3"
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=True,
        help='JSON filter dict for SQL table, e.g. \'{"lab": "rl2"}\'',
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="rldb",
        help="R2/S3 bucket name (default: rldb)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        filters = parse_filters(args.filters)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Filters: {filters}")
    print(f"Bucket:  {args.bucket}")
    print(f"Dry run: {args.dry_run}")

    print("\nConnecting to database...")
    engine = create_default_engine()

    print("\nQuerying episodes from SQL table...")
    df = episode_table_to_df(engine)

    for key, value in filters.items():
        if key not in df.columns:
            print(f"Warning: filter key '{key}' not found in table columns")
            continue
        df = df[df[key] == value]

    if df.empty:
        print("No episodes found matching filters")
        return 0

    print(f"Found {len(df)} episodes")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} episodes")

    s3 = get_boto3_s3_client()
    success_count = 0
    error_count = 0

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        episode_hash = row["episode_hash"]
        print(f"\n[{idx}/{len(df)}] {episode_hash}")

        table_row = episode_hash_to_table_row(engine, episode_hash)
        if table_row is None:
            print("  Warning: episode not found in SQL table")
            error_count += 1
            continue

        ok = process_episode(
            s3, args.bucket, episode_hash, asdict(table_row), args.dry_run
        )
        if ok:
            success_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 60)
    print(f"Total: {len(df)}  |  Success: {success_count}  |  Errors: {error_count}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
