#!/usr/bin/env python3
"""
Download zarr episodes locally and apply attribute conversions from the SQL table.

For each matched episode this script:
1. Downloads the full zarr store from R2/S3 to <output-dir>/<episode_hash>.zarr/
2. Sets the 'embodiment' attribute to the SQL row's robot_name value
3. Removes 'robot_name' from the zarr attributes if present
4. Adds all other SQL table fields that are not yet present in the zarr attributes
5. Writes the updated zarr.json to the local copy only (nothing is written back to R2/S3)

Usage:
    python download_zarr_episodes.py \
        --episode-hashes 2026-01-12-03-47-29-664000 2026-01-13-10-22-01-000000 \
        --output-dir /tmp/zarr_episodes \
        --dry-run
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from egomimic.utils.aws.aws_data_utils import get_boto3_s3_client, s3_sync_to_local
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_table_row,
)

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


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    """Return (bucket, key_prefix) from an s3://bucket/key URL."""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def build_updated_attrs(attrs: dict, sql_row_dict: dict) -> tuple[dict, list[str]]:
    """
    Return (updated_attrs, change_log) applying the conversion rules:
      - embodiment  ← sql robot_name value
      - robot_name  removed if present
      - all other non-skip SQL fields added only if not already in attrs
    """
    updated = dict(attrs)
    changes = []

    # 1. Set embodiment from robot_name
    robot_name = sql_row_dict.get("robot_name", "")
    old_embodiment = updated.get("embodiment")
    if old_embodiment != robot_name:
        updated["embodiment"] = robot_name
        changes.append(f"embodiment: {old_embodiment!r} → {robot_name!r}")

    # 2. Remove robot_name if present
    if "robot_name" in updated:
        del updated["robot_name"]
        changes.append("removed robot_name")

    # 3. Add missing SQL fields
    for field, value in sql_row_dict.items():
        if field in _SKIP_SQL_FIELDS:
            continue
        if field not in updated:
            updated[field] = value
            changes.append(f"added {field}={value!r}")

    return updated, changes


def download_and_convert_episode(
    s3,
    episode_hash: str,
    zarr_processed_path: str,
    sql_row_dict: dict,
    output_dir: Path,
    dry_run: bool,
) -> bool:
    """Download a zarr episode, apply attr conversions, save locally. Returns True on success."""
    bucket, key_prefix = parse_s3_url(zarr_processed_path)
    local_episode_dir = output_dir / f"{episode_hash}.zarr"

    if dry_run:
        print(
            f"  [DRY RUN] Would download s3://{bucket}/{key_prefix} → {local_episode_dir}"
        )
        # Still compute the attr diff so the user can see what would change
        zarr_json_key = key_prefix.rstrip("/") + "/zarr.json"
        try:
            response = s3.get_object(Bucket=bucket, Key=zarr_json_key)
            zarr_doc = json.loads(response["Body"].read())
            attrs = zarr_doc.get("attributes", {})
            _, changes = build_updated_attrs(attrs, sql_row_dict)
            if changes:
                for c in changes:
                    print(f"  [DRY RUN] attr change: {c}")
            else:
                print("  [DRY RUN] No attr changes needed")
        except Exception as e:
            print(f"  [DRY RUN] Could not fetch zarr.json to preview changes: {e}")
        return True

    # Download the full zarr store
    print(f"  Downloading s3://{bucket}/{key_prefix} → {local_episode_dir}")
    try:
        s3_sync_to_local(bucket, key_prefix, local_episode_dir)
    except Exception as e:
        print(f"  Error downloading zarr store: {e}")
        return False

    # Read the local zarr.json
    zarr_json_path = local_episode_dir / "zarr.json"
    if not zarr_json_path.exists():
        print(f"  Error: zarr.json not found at {zarr_json_path}")
        return False

    try:
        zarr_doc = json.loads(zarr_json_path.read_text())
    except Exception as e:
        print(f"  Error reading local zarr.json: {e}")
        return False

    if "attributes" not in zarr_doc:
        print("  Warning: zarr.json has no 'attributes' key — initialising empty")
        zarr_doc["attributes"] = {}

    attrs = zarr_doc["attributes"]
    updated_attrs, changes = build_updated_attrs(attrs, sql_row_dict)

    if not changes:
        print("  No attr changes needed")
    else:
        for c in changes:
            print(f"  {c}")

    # Write updated zarr.json locally
    zarr_doc["attributes"] = updated_attrs
    try:
        zarr_json_path.write_text(json.dumps(zarr_doc, indent=2))
    except Exception as e:
        print(f"  Error writing local zarr.json: {e}")
        return False

    print(f"  Saved to {local_episode_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download zarr episodes locally and apply SQL attribute conversions"
    )
    parser.add_argument(
        "--episode-hashes",
        type=str,
        nargs="+",
        required=True,
        help="One or more episode hashes to download, e.g. 2026-01-12-03-47-29-664000",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local directory to store downloaded zarr episodes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded/changed without writing anything",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Episode hashes: {args.episode_hashes}")
    print(f"Output dir: {output_dir}")
    print(f"Dry run: {args.dry_run}")

    print("\nConnecting to database...")
    engine = create_default_engine()

    s3 = get_boto3_s3_client()
    success_count = 0
    error_count = 0
    total = len(args.episode_hashes)

    for idx, episode_hash in enumerate(args.episode_hashes, 1):
        print(f"\n[{idx}/{total}] {episode_hash}")

        table_row = episode_hash_to_table_row(engine, episode_hash)
        if table_row is None:
            print("  Warning: episode not found in SQL table")
            error_count += 1
            continue

        zarr_processed_path = table_row.zarr_processed_path
        if not zarr_processed_path or not zarr_processed_path.strip():
            print("  Warning: no zarr_processed_path set in SQL table")
            error_count += 1
            continue

        print(f"  zarr path: {zarr_processed_path}")
        sql_row_dict = asdict(table_row)

        ok = download_and_convert_episode(
            s3,
            episode_hash,
            zarr_processed_path,
            sql_row_dict,
            output_dir,
            args.dry_run,
        )
        if ok:
            success_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 60)
    print(f"Total: {total}  |  Success: {success_count}  |  Errors: {error_count}")
    if not args.dry_run:
        print(f"Episodes saved to: {output_dir}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
