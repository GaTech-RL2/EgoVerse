#!/usr/bin/env python3
"""
Backfill script to update metadata.json files in S3 based on SQL table filters.

This script:
1. Queries the SQL table to filter which episodes to process
2. Downloads metadata.json files from S3 for filtered episodes
3. Updates metadata.json with specified changes
4. Uploads updated metadata.json back to S3
5. Updates the corresponding SQL table entries

Usage:
    python update_metadata_from_sql.py \
        --filters '{"lab": "rl2", "task": "object in container"}' \
        --metadata-changes '{"operator": "NewOperator", "scene": "scene_5"}' \
        --dry-run
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import pandas as pd

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from egomimic.utils.aws.aws_sql import (
    TableRow,
    create_default_engine,
    episode_table_to_df,
    update_episode,
    episode_hash_to_table_row,
)


def parse_filters(filters_str: str) -> Dict:
    """Parse filters from JSON string."""
    try:
        return json.loads(filters_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in filters: {e}")


def parse_metadata_changes(changes_str: str) -> Dict:
    """Parse metadata changes from JSON string."""
    try:
        return json.loads(changes_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in metadata changes: {e}")


def get_metadata_path_from_episode_hash(episode_hash: str, bucket_name: str = "rldb") -> Optional[str]:
    """
    Construct the S3 path to metadata.json file from episode_hash.
    
    The metadata.json is typically stored in raw_v2/<embodiment>/<timestamp>_metadata.json
    We need to find it by searching or using the episode_hash pattern.
    """
    # Episode hash format is typically YYYY-MM-DD-HH-MM-SS-ffffff
    # The original timestamp might be in milliseconds
    # Try to find the metadata file in raw_v2
    s3 = boto3.client("s3")
    
    # Search in common locations
    search_prefixes = [
        f"raw_v2/",
    ]
    
    for prefix in search_prefixes:
        # List objects with the episode_hash in the name
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                # Look for metadata files that might match
                if episode_hash in key and "_metadata.json" in key:
                    return key
    
    return None


def episode_hash_to_timestamp_ms(episode_hash: str) -> Optional[int]:
    """
    Convert episode_hash (YYYY-MM-DD-HH-MM-SS-ffffff) back to timestamp in milliseconds.
    
    The episode_hash format is: YYYY-MM-DD-HH-MM-SS-ffffff
    This was created from: datetime.fromtimestamp(timestamp_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    """
    try:
        # Parse the episode_hash back to datetime
        # Format: YYYY-MM-DD-HH-MM-SS-ffffff
        parts = episode_hash.split("-")
        if len(parts) < 6:
            return None
        
        # Reconstruct datetime string
        dt_str = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
        if len(parts) > 6:
            # Add microseconds
            dt_str += f".{parts[6]}"
        
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
        dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to milliseconds timestamp
        timestamp_ms = int(dt.timestamp() * 1000)
        return timestamp_ms
    except Exception as e:
        print(f"Error converting episode_hash {episode_hash} to timestamp: {e}")
        return None


def find_metadata_files_for_episodes(
    df: pd.DataFrame, bucket_name: str = "rldb"
) -> Dict[str, str]:
    """
    Find metadata.json S3 paths for episodes in the dataframe.
    
    The episode_hash in SQL table is in format: YYYY-MM-DD-HH-MM-SS-ffffff
    The actual files in S3 use timestamp in milliseconds as filename: <timestamp>_metadata.json
    
    Returns:
        Dict mapping episode_hash to S3 metadata.json path
    """
    s3 = boto3.client("s3")
    metadata_paths = {}
    
    # Group by embodiment to search more efficiently
    for embodiment in df["embodiment"].unique():
        prefix = f"raw_v2/{embodiment.lower()}/"
        
        print(f"  Searching in {prefix}...")
        
        # Get all metadata files in this prefix
        paginator = s3.get_paginator("list_objects_v2")
        metadata_files = {}  # Map timestamp_ms to S3 key
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("_metadata.json"):
                    # Extract timestamp from filename
                    # Format: <timestamp>_metadata.json where timestamp is in milliseconds
                    filename = Path(key).stem.replace("_metadata", "")
                    try:
                        timestamp_ms = int(filename)
                        metadata_files[timestamp_ms] = key
                    except ValueError:
                        # If it's not a pure integer, try to match by episode_hash conversion
                        pass
        
        print(f"  Found {len(metadata_files)} metadata files in {prefix}")
        
        # Match episode hashes to metadata files
        for _, row in df[df["embodiment"] == embodiment].iterrows():
            episode_hash = row["episode_hash"]
            
            # Convert episode_hash to timestamp_ms
            timestamp_ms = episode_hash_to_timestamp_ms(episode_hash)
            if timestamp_ms is None:
                print(f"  Warning: Could not convert episode_hash {episode_hash} to timestamp")
                continue
            
            # Look for exact match
            if timestamp_ms in metadata_files:
                metadata_paths[episode_hash] = metadata_files[timestamp_ms]
            else:
                # Try nearby timestamps (within 1 second) in case of rounding issues
                found = False
                for ts in range(timestamp_ms - 1000, timestamp_ms + 1000, 100):
                    if ts in metadata_files:
                        metadata_paths[episode_hash] = metadata_files[ts]
                        found = True
                        break
                
                if not found:
                    print(f"  Warning: Could not find metadata file for episode_hash {episode_hash} (timestamp_ms: {timestamp_ms})")
    
    return metadata_paths


def download_metadata_from_s3(
    s3_path: str, bucket_name: str, local_path: Path
) -> bool:
    """Download metadata.json from S3 to local path."""
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_path, str(local_path))
        return True
    except Exception as e:
        print(f"Error downloading {s3_path}: {e}")
        return False


def upload_metadata_to_s3(
    local_path: Path, s3_path: str, bucket_name: str
) -> bool:
    """Upload metadata.json to S3."""
    s3 = boto3.client("s3")
    try:
        s3.upload_file(str(local_path), bucket_name, s3_path)
        return True
    except Exception as e:
        print(f"Error uploading {s3_path}: {e}")
        return False


def update_metadata_json(
    metadata_path: Path, changes: Dict
) -> Dict:
    """
    Update metadata.json file with specified changes.
    
    Returns:
        Updated metadata dictionary
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Apply changes
    for key, value in changes.items():
        metadata[key] = value
    
    # Write back
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def update_sql_table(engine, episode_hash: str, metadata: Dict):
    """
    Update SQL table entry with values from metadata.
    
    Maps metadata.json fields to TableRow fields.
    Uses episode_hash_to_table_row and update_episode from aws_sql.py.
    """
    # Get current row using the helper function from aws_sql.py
    current_row = episode_hash_to_table_row(engine, episode_hash)
    
    if current_row is None:
        print(f"Warning: Episode {episode_hash} not found in SQL table")
        return False
    
    # Map metadata fields to TableRow fields (including all updateable fields)
    field_mapping = {
        "operator": "operator",
        "lab": "lab",
        "task": "task",
        "embodiment": "embodiment",
        "robot_name": "robot_name",
        "task_description": "task_description",
        "scene": "scene",
        "objects": "objects",
        "processed_path": "processed_path",
        "mp4_path": "mp4_path",
        "num_frames": "num_frames",
    }
    
    # Update fields from metadata
    for metadata_key, table_key in field_mapping.items():
        if metadata_key in metadata:
            value = metadata[metadata_key]
            
            # Handle empty strings appropriately
            if value == "":
                if table_key == "num_frames":
                    # Empty string for num_frames should be converted to -1 (default)
                    value = -1
                elif table_key in ["processed_path", "mp4_path"]:
                    # Empty string is valid for these fields
                    value = ""
                # For other string fields, empty string is also valid
            
            # Handle type conversion for num_frames
            if table_key == "num_frames" and isinstance(value, str):
                if value == "" or value.lower() == "none":
                    value = -1
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        print(f"  Warning: Could not convert num_frames '{value}' to int, skipping")
                        continue
            
            setattr(current_row, table_key, value)
    
    # Update in database using the helper function from aws_sql.py
    update_episode(engine, current_row)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update metadata.json files in S3 based on SQL table filters"
    )
    parser.add_argument(
        "--filters",
        type=str,
        required=True,
        help='JSON string of filters to apply to SQL table, e.g., \'{"lab": "rl2", "task": "object in container"}\'',
    )
    parser.add_argument(
        "--metadata-changes",
        type=str,
        default=None,
        help='JSON string of changes to apply to metadata.json, e.g., \'{"operator": "NewOperator", "scene": "scene_5"}\'. Required when not using --sql-only.',
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="rldb",
        help="S3 bucket name (default: rldb)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of episodes to process (for testing)",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temporary directory for downloading files (default: system temp)",
    )
    parser.add_argument(
        "--sql-only",
        action="store_true",
        help="Update SQL table only, skip S3 metadata.json files. Use --sql-changes instead of --metadata-changes to specify SQL field updates.",
    )
    parser.add_argument(
        "--sql-changes",
        type=str,
        default=None,
        help='JSON string of changes to apply directly to SQL table fields, e.g., \'{"operator": "NewOperator", "scene": "scene_5"}\'. Required when using --sql-only.',
    )

    args = parser.parse_args()

    # Parse filters and changes
    try:
        filters = parse_filters(args.filters)
        if args.sql_only:
            if not args.sql_changes:
                print("Error: --sql-changes is required when using --sql-only")
                return 1
            sql_changes = parse_metadata_changes(args.sql_changes)  # Reuse parser, same format
            metadata_changes = None
        else:
            if not args.metadata_changes:
                print("Error: --metadata-changes is required when not using --sql-only")
                return 1
            metadata_changes = parse_metadata_changes(args.metadata_changes)
            sql_changes = None
    except ValueError as e:
        print(f"Error parsing arguments: {e}")
        return 1

    print(f"Filters: {filters}")
    if args.sql_only:
        print(f"SQL changes: {sql_changes}")
        print("Mode: SQL-only (will not modify S3 metadata.json files)")
    else:
        print(f"Metadata changes: {metadata_changes}")
        print("Mode: Full update (will update both S3 metadata.json and SQL table)")
    print(f"Dry run: {args.dry_run}")

    # Connect to database
    print("\nConnecting to database...")
    engine = create_default_engine()

    # Query episodes based on filters
    print("\nQuerying episodes from SQL table...")
    df = episode_table_to_df(engine)

    # Apply filters
    for key, value in filters.items():
        if key not in df.columns:
            print(f"Warning: Filter key '{key}' not found in table columns")
            continue
        df = df[df[key] == value]

    if df.empty:
        print("No episodes found matching filters")
        return 0

    print(f"Found {len(df)} episodes matching filters")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to {len(df)} episodes")

    # SQL-only mode: skip S3 operations
    if args.sql_only:
        print("\n[SQL-ONLY MODE] Skipping S3 metadata.json operations")
        metadata_paths = {row["episode_hash"]: None for _, row in df.iterrows()}
    else:
        # Find metadata files for these episodes
        print("\nFinding metadata.json files in S3...")
        
        # First, try to use processed_path if available (for processed datasets)
        # Otherwise, search in raw_v2
        metadata_paths = {}
        
        # Check if we have processed_path entries
        if "processed_path" in df.columns and df["processed_path"].notna().any():
            print("  Some episodes have processed_path - these are processed datasets")
            print("  Note: This script currently focuses on raw_v2 metadata files")
        
        # Find metadata files in raw_v2
        raw_metadata_paths = find_metadata_files_for_episodes(df, args.bucket)
        metadata_paths.update(raw_metadata_paths)

        if not metadata_paths:
            print("No metadata.json files found for the filtered episodes")
            return 1

        print(f"Found {len(metadata_paths)} metadata.json files")

    # Create temp directory (only needed for non-SQL-only mode)
    if not args.sql_only:
        if args.temp_dir:
            temp_dir = Path(args.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp())
        print(f"Using temp directory: {temp_dir}")
    else:
        temp_dir = None

    # Process each episode
    success_count = 0
    error_count = 0

    for idx, (episode_hash, s3_metadata_path) in enumerate(metadata_paths.items(), 1):
        print(f"\n[{idx}/{len(metadata_paths)}] Processing episode: {episode_hash}")
        
        if args.sql_only:
            if args.dry_run:
                print(f"  [DRY RUN] Would update SQL table with: {sql_changes}")
                continue
            
            # SQL-only mode: update SQL table directly
            try:
                # Create a metadata dict from sql_changes for update_sql_table
                # The function expects metadata dict format
                update_sql_table(engine, episode_hash, sql_changes)
                print(f"  Updated SQL table entry with: {sql_changes}")
                success_count += 1
            except Exception as e:
                print(f"  Error updating SQL table: {e}")
                error_count += 1
        else:
            # Full mode: update both S3 and SQL
            print(f"  S3 path: s3://{args.bucket}/{s3_metadata_path}")

            if args.dry_run:
                print("  [DRY RUN] Would update metadata and SQL table")
                continue

            # Download metadata
            local_metadata_path = temp_dir / f"{episode_hash}_metadata.json"
            if not download_metadata_from_s3(s3_metadata_path, args.bucket, local_metadata_path):
                print(f"  Error: Failed to download metadata")
                error_count += 1
                continue

            # Update metadata
            try:
                updated_metadata = update_metadata_json(local_metadata_path, metadata_changes)
                print(f"  Updated metadata: {metadata_changes}")
            except Exception as e:
                print(f"  Error updating metadata: {e}")
                error_count += 1
                continue

            # Upload back to S3
            if not upload_metadata_to_s3(local_metadata_path, s3_metadata_path, args.bucket):
                print(f"  Error: Failed to upload metadata")
                error_count += 1
                continue

            print(f"  Uploaded updated metadata to S3")

            # Update SQL table
            try:
                update_sql_table(engine, episode_hash, updated_metadata)
                print(f"  Updated SQL table entry")
                success_count += 1
            except Exception as e:
                print(f"  Error updating SQL table: {e}")
                error_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total episodes processed: {len(metadata_paths)}")
    if not args.dry_run:
        print(f"  Successful: {success_count}")
        print(f"  Errors: {error_count}")

    # Cleanup
    if not args.sql_only and not args.temp_dir and temp_dir:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory: {temp_dir}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

