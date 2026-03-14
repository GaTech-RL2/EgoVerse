#!/usr/bin/env python3
"""
Interactive script to resolve mismatches between SQL table and metadata.json files.

This script:
1. Reads mismatches.csv file
2. Groups mismatches by episode_hash
3. For each episode, interactively allows the user to choose which value to keep (SQL or metadata)
4. Updates both SQL table and metadata.json file in S3
5. If robot_name changes, automatically resets processed_path, num_frames, and mp4_path

Usage:
    python resolve_mismatches.py --csv mismatches.csv --bucket rldb
"""

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from egomimic.utils.aws.aws_sql import (
    create_default_engine,
)
from sqlalchemy import MetaData, Table, select, update


def parse_value(value_str: str):
    """Parse a value from CSV, handling JSON strings and None values."""
    if value_str == "" or value_str == "nan" or value_str == "None":
        return None
    try:
        # Try to parse as JSON (for lists/dicts)
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        # Return as string
        return value_str


def format_value(value) -> str:
    """Format a value for display."""
    if value is None:
        return "None"
    elif isinstance(value, (list, dict)):
        return json.dumps(value, indent=2)
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return str(value)


def display_mismatch(
    episode_hash: str,
    s3_key: str,
    field_name: str,
    sql_value,
    metadata_value,
    mismatch_num: int,
    total_mismatches: int,
):
    """Display a mismatch in a user-friendly format."""
    print("\n" + "=" * 80)
    print(f"MISMATCH {mismatch_num}/{total_mismatches}")
    print("=" * 80)
    print(f"Episode Hash: {episode_hash}")
    print(f"S3 Key:       {s3_key}")
    print(f"Field:        {field_name}")
    print("-" * 80)
    print("SQL Table Value:")
    print(f"  {format_value(sql_value)}")
    print("-" * 80)
    print("Metadata.json Value:")
    print(f"  {format_value(metadata_value)}")
    print("=" * 80)


def get_user_choice() -> str:
    """Get user's choice for which value to keep."""
    while True:
        choice = input("\nChoose value to keep:\n  [s] SQL value\n  [m] Metadata value\n  [k] Skip this mismatch\n  [q] Quit\nEnter choice (s/m/k/q): ").strip().lower()
        if choice in ["s", "sql"]:
            return "sql"
        elif choice in ["m", "metadata"]:
            return "metadata"
        elif choice in ["k", "skip"]:
            return "skip"
        elif choice in ["q", "quit"]:
            return "quit"
        else:
            print("Invalid choice. Please enter 's', 'm', 'k', or 'q'.")


def update_metadata_in_s3(s3_client, bucket: str, s3_key: str, updates: Dict) -> bool:
    """Update metadata.json file in S3."""
    try:
        # Download metadata.json to temp file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Download from S3
            s3_client.download_file(bucket, s3_key, str(tmp_path))
            
            # Read and update
            with open(tmp_path, "r") as f:
                metadata = json.load(f)
            
            # Apply updates
            for key, value in updates.items():
                metadata[key] = value
            
            # Write back
            with open(tmp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Upload back to S3
            s3_client.upload_file(str(tmp_path), bucket, s3_key)
            return True
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
    except Exception as e:
        print(f"Error updating metadata.json in S3: {e}")
        return False


def get_sql_row_dict(engine, episode_hash: str) -> Optional[Dict]:
    """Get SQL row as a dictionary by querying the table directly."""
    try:
        metadata = MetaData()
        episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")
        
        stmt = select(episodes_tbl).where(episodes_tbl.c.episode_hash == episode_hash).limit(1)
        with engine.connect() as conn:
            rec = conn.execute(stmt).mappings().first()
        
        return dict(rec) if rec else None
    except Exception as e:
        print(f"Error querying SQL row: {e}")
        return None


def update_sql_row(engine, episode_hash: str, field_name: str, value) -> bool:
    """Update a single field in the SQL table using direct SQL update."""
    try:
        # Get table reference
        metadata = MetaData()
        episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")
        
        # Check if field exists in the table
        if field_name not in episodes_tbl.columns:
            print(f"Warning: Field '{field_name}' does not exist in SQL table")
            return False
        
        # Handle type conversions and special cases
        if field_name == "num_frames":
            if value is None or value == "" or value == "nan":
                value = -1
            elif isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    print(f"Warning: Could not convert num_frames '{value}' to int")
                    return False
            elif not isinstance(value, int):
                value = int(value)
        elif field_name in ["processed_path", "mp4_path"]:
            if value is None:
                value = ""
            else:
                value = str(value)
        elif field_name == "objects":
            # Handle objects field (can be string or list)
            if isinstance(value, list):
                value = json.dumps(value)
            elif value is None:
                value = ""
            else:
                value = str(value)
        else:
            # For other string fields
            if value is None:
                value = ""
            else:
                value = str(value)
        
        # Build and execute update statement directly
        stmt = (
            update(episodes_tbl)
            .where(episodes_tbl.c.episode_hash == episode_hash)
            .values(**{field_name: value})
        )
        
        with engine.begin() as conn:
            result = conn.execute(stmt)
            if result.rowcount == 0:
                print(f"Warning: Episode {episode_hash} not found in SQL table")
                return False
        
        return True
    except Exception as e:
        print(f"Error updating SQL row '{field_name}' for episode {episode_hash}: {e}")
        import traceback
        traceback.print_exc()
        return False


def handle_robot_name_change(engine, episode_hash: str, s3_client, bucket: str, s3_key: str):
    """Reset processed_path, num_frames, and mp4_path when robot_name changes."""
    print("\n⚠️  Robot name changed! Resetting processed_path, num_frames, and mp4_path...")
    
    # Update SQL table
    update_sql_row(engine, episode_hash, "processed_path", "")
    update_sql_row(engine, episode_hash, "mp4_path", "")
    update_sql_row(engine, episode_hash, "num_frames", -1)
    
    # Update metadata.json
    metadata_updates = {
        "processed_path": "",
        "mp4_path": "",
        "num_frames": -1,
    }
    update_metadata_in_s3(s3_client, bucket, s3_key, metadata_updates)
    
    print("✓ Reset processed_path, num_frames, and mp4_path")


def main():
    parser = argparse.ArgumentParser(
        description="Interactively resolve mismatches between SQL table and metadata.json files"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to mismatches.csv file",
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
        help="Show what would be changed without actually making changes",
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")
    
    # Read CSV
    print(f"Reading mismatches from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if df.empty:
        print("No mismatches found in CSV file.")
        return 0
    
    # Group mismatches by episode_hash
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        episode_hash = row["episode_hash"]
        grouped[episode_hash].append({
            "s3_key": row["s3_key"],
            "field_name": row["field_name"],
            "sql_value": parse_value(row["sql_value"]),
            "metadata_value": parse_value(row["metadata_value"]),
        })
    
    print(f"Found {len(grouped)} episodes with mismatches")
    print(f"Total {len(df)} field mismatches\n")
    
    # Connect to database and S3
    print("Connecting to database...")
    engine = create_default_engine()
    
    print("Initializing S3 client...")
    s3_client = boto3.client("s3")
    
    # Process each episode
    total_episodes = len(grouped)
    processed_episodes = 0
    
    for episode_hash, mismatches in grouped.items():
        processed_episodes += 1
        print(f"\n{'='*80}")
        print(f"EPISODE {processed_episodes}/{total_episodes}")
        print(f"{'='*80}")
        print(f"Episode Hash: {episode_hash}")
        print(f"Number of field mismatches: {len(mismatches)}")
        
        # Get s3_key (should be the same for all mismatches in an episode)
        s3_key = mismatches[0]["s3_key"]
        
        # Track if robot_name is being changed
        robot_name_changed = False
        robot_name_new_value = None
        
        # Get current SQL row to check for robot_name changes
        current_sql_row_dict = get_sql_row_dict(engine, episode_hash)
        current_robot_name = current_sql_row_dict.get("robot_name") if current_sql_row_dict else None
        
        # Process each mismatch for this episode
        episode_updates_sql = {}
        episode_updates_metadata = {}
        
        for i, mismatch in enumerate(mismatches, 1):
            field_name = mismatch["field_name"]
            sql_value = mismatch["sql_value"]
            metadata_value = mismatch["metadata_value"]
            
            display_mismatch(
                episode_hash,
                s3_key,
                field_name,
                sql_value,
                metadata_value,
                i,
                len(mismatches),
            )
            
            choice = get_user_choice()
            
            if choice == "quit":
                print("\nExiting...")
                return 0
            elif choice == "skip":
                print("Skipping this mismatch...")
                continue
            elif choice == "sql":
                chosen_value = sql_value
                print(f"✓ Will use SQL value (will update metadata.json to match SQL)")
                # Update metadata.json to match SQL (SQL is already correct)
                episode_updates_metadata[field_name] = chosen_value
            elif choice == "metadata":
                chosen_value = metadata_value
                print(f"✓ Will use metadata value (will update SQL to match metadata.json)")
                # Update SQL to match metadata.json (metadata.json is already correct)
                episode_updates_sql[field_name] = chosen_value
            
            # Track robot_name changes - reset fields if robot_name is changing on either side
            if field_name == "robot_name":
                if choice == "metadata":
                    # Updating SQL robot_name - check if it's different from current SQL value
                    if current_robot_name != chosen_value:
                        robot_name_changed = True
                        robot_name_new_value = chosen_value
                elif choice == "sql":
                    # Updating metadata.json robot_name - check if it's different from current metadata value
                    # If metadata_value != sql_value, then metadata.json robot_name is changing
                    if metadata_value != sql_value:
                        robot_name_changed = True
                        robot_name_new_value = chosen_value
        
        # Apply updates if any were made
        if episode_updates_sql or episode_updates_metadata:
            if args.dry_run:
                print("\n🔍 DRY RUN - Would apply the following updates:")
                if episode_updates_sql:
                    print("  SQL updates:", episode_updates_sql)
                if episode_updates_metadata:
                    print("  Metadata updates:", episode_updates_metadata)
                if robot_name_changed:
                    print("  Would reset processed_path, num_frames, mp4_path in both SQL and metadata.json")
            else:
                print("\nApplying updates...")
                
                # Update SQL table fields (only if user chose metadata value)
                for field_name, value in episode_updates_sql.items():
                    success = update_sql_row(engine, episode_hash, field_name, value)
                    if success:
                        print(f"  ✓ Updated SQL field '{field_name}'")
                    else:
                        print(f"  ✗ Failed to update SQL field '{field_name}'")
                
                # Prepare metadata.json updates (merge robot_name update with reset fields if needed)
                metadata_json_updates = episode_updates_metadata.copy() if episode_updates_metadata else {}
                
                # If robot_name changed, add reset fields to metadata.json updates
                if robot_name_changed:
                    metadata_json_updates.update({
                        "processed_path": "",
                        "mp4_path": "",
                        "num_frames": -1,
                    })
                
                # Update metadata.json (merge robot_name and reset fields into single update)
                if metadata_json_updates:
                    success = update_metadata_in_s3(s3_client, args.bucket, s3_key, metadata_json_updates)
                    if success:
                        print(f"  ✓ Updated metadata.json in S3")
                    else:
                        print(f"  ✗ Failed to update metadata.json in S3")
                
                # Handle robot_name change - reset processed_path, num_frames, mp4_path in SQL
                # (metadata.json reset is already handled above)
                if robot_name_changed:
                    print("\n⚠️  Robot name changed! Resetting processed_path, num_frames, and mp4_path in SQL...")
                    reset_success = True
                    if not update_sql_row(engine, episode_hash, "processed_path", ""):
                        print(f"  ✗ Failed to reset processed_path")
                        reset_success = False
                    if not update_sql_row(engine, episode_hash, "mp4_path", ""):
                        print(f"  ✗ Failed to reset mp4_path")
                        reset_success = False
                    if not update_sql_row(engine, episode_hash, "num_frames", -1):
                        print(f"  ✗ Failed to reset num_frames")
                        reset_success = False
                    if reset_success:
                        print("✓ Reset processed_path, num_frames, and mp4_path in SQL")
                    else:
                        print("✗ Some fields failed to reset in SQL")
                
                print("✓ Episode updates completed")
        else:
            print("No updates to apply for this episode.")
    
    print(f"\n{'='*80}")
    print(f"COMPLETED")
    print(f"{'='*80}")
    print(f"Processed {processed_episodes} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

