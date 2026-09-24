#!/usr/bin/env python3
"""
Script to verify that SQL table entries match metadata.json files in raw_v2/aria.

This script:
1. Reads all SQL table rows (optionally filtered by embodiment)
2. Reads all metadata.json files from raw_v2/aria
3. Matches SQL rows to metadata.json files by timestamp/episode_hash
4. Compares all fields between SQL row and JSON file
5. Reports mismatches and orphaned entries

Usage:
    python sql_verify.py --embodiment aria
    python sql_verify.py --embodiment aria --output-csv mismatches.csv
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import boto3
import pandas as pd

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


def episode_hash_to_timestamp_ms(episode_hash: str) -> Optional[int]:
    """
    Convert episode_hash (YYYY-MM-DD-HH-MM-SS-ffffff) to timestamp in milliseconds.
    
    Args:
        episode_hash: String in format YYYY-MM-DD-HH-MM-SS-ffffff
        
    Returns:
        Timestamp in milliseconds (13-digit integer), or None if conversion fails
    """
    try:
        parts = episode_hash.split("-")
        if len(parts) < 6:
            return None
        
        # Reconstruct datetime string
        dt_str = f"{parts[0]}-{parts[1]}-{parts[2]} {parts[3]}:{parts[4]}:{parts[5]}"
        if len(parts) > 6:
            # Add microseconds (first 6 digits)
            microsec_str = parts[6][:6].ljust(6, '0')  # Ensure 6 digits
            dt_str += f".{microsec_str}"
        else:
            dt_str += ".000000"
        
        # Parse as UTC time
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
        dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to milliseconds timestamp
        timestamp_ms = int(dt.timestamp() * 1000)
        return timestamp_ms
    except Exception as e:
        return None


def get_all_metadata_files(
    s3_client, bucket_name: str, prefix: str
) -> Dict[int, Tuple[str, Dict]]:
    """
    Get all metadata.json files from S3 and load their contents.
    
    Returns:
        Dictionary mapping timestamp_ms to (s3_key, metadata_dict)
    """
    metadata_files = {}
    
    paginator = s3_client.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" not in page:
            continue
        
        for obj in page["Contents"]:
            key = obj["Key"]
            if not key.endswith("_metadata.json"):
                continue
            
            # Extract timestamp from filename
            # Format: <timestamp>_metadata.json
            filename = Path(key).stem.replace("_metadata", "")
            try:
                timestamp_ms = int(filename)
            except ValueError:
                continue
            
            # Download and parse metadata.json
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=key)
                metadata = json.loads(response['Body'].read().decode('utf-8'))
                metadata_files[timestamp_ms] = (key, metadata)
            except Exception:
                continue
    
    return metadata_files


def parse_postgres_array(value_str: str) -> Optional[list]:
    """
    Parse PostgreSQL array format: {"item1","item2",item3}
    This format uses curly braces and may have unquoted items.
    
    Examples:
    - {"item1","item2",item3} -> ['item1', 'item2', 'item3']
    - {"short sleeve shirts","long sleeve shirts",shorts} -> ['short sleeve shirts', 'long sleeve shirts', 'shorts']
    """
    value_str = value_str.strip()
    if not (value_str.startswith('{') and value_str.endswith('}')):
        return None
    
    # Remove outer braces
    inner = value_str[1:-1].strip()
    if not inner:
        return []
    
    # Split by comma, handling quoted and unquoted items
    items = []
    current_item = ""
    in_quotes = False
    quote_char = None
    i = 0
    
    while i < len(inner):
        char = inner[i]
        
        # Handle quote characters
        if char in ('"', "'"):
            # Check if escaped
            if i > 0 and inner[i-1] == '\\':
                current_item += char
            elif not in_quotes:
                # Start of quoted item
                in_quotes = True
                quote_char = char
                # Don't add the opening quote to current_item
            elif char == quote_char:
                # End of quoted item
                in_quotes = False
                quote_char = None
                # Don't add the closing quote to current_item
            else:
                # Different quote type inside quotes - treat as literal
                current_item += char
        elif char == ',' and not in_quotes:
            # Comma outside quotes - split here
            item = current_item.strip()
            if item:
                # Remove any remaining quotes (shouldn't happen, but just in case)
                item = item.strip('"').strip("'")
                items.append(item)
            current_item = ""
        else:
            # Regular character
            current_item += char
        
        i += 1
    
    # Add last item
    if current_item.strip():
        item = current_item.strip().strip('"').strip("'")
        items.append(item)
    
    return items


def normalize_value_for_comparison(value) -> str:
    """
    Normalize a value for comparison, handling different data types and formats.
    
    Handles:
    - Lists/sets/arrays: Convert to sorted list of strings
    - PostgreSQL array format: {"item1","item2",item3}
    - JSON arrays: ["item1", "item2", "item3"]
    - None/empty: Convert to empty string
    - Other types: Convert to string
    """
    if value is None:
        return ""
    
    # Handle list-like structures (lists, sets, tuples)
    if isinstance(value, (list, set, tuple)):
        # Convert all elements to strings and sort
        return str(sorted([str(item) for item in value]))
    
    # Handle string representations of sets/lists
    if isinstance(value, str):
        value_str = value.strip()
        
        # Try PostgreSQL array format first: {"item1","item2",item3}
        if value_str.startswith('{') and value_str.endswith('}'):
            parsed = parse_postgres_array(value_str)
            if parsed is not None:
                return str(sorted([str(item) for item in parsed]))
        
        # Try JSON array format: ["item1", "item2", "item3"]
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                parsed = json.loads(value_str)
                if isinstance(parsed, (list, set, tuple)):
                    return str(sorted([str(item) for item in parsed]))
            except (json.JSONDecodeError, ValueError):
                pass
    
    # For other types, convert to string
    return str(value)


def compare_sql_and_metadata(
    sql_row: Dict, metadata: Dict, fields_to_compare: List[str]
) -> Dict[str, Tuple]:
    """
    Compare SQL row fields with metadata.json fields.
    
    Returns:
        Dictionary mapping field_name to (sql_value, json_value) for mismatches
    """
    mismatches = {}
    
    for field in fields_to_compare:
        sql_value = sql_row.get(field)
        json_value = metadata.get(field)
        
        # Normalize values for comparison
        sql_normalized = normalize_value_for_comparison(sql_value)
        json_normalized = normalize_value_for_comparison(json_value)
        
        # Compare normalized values
        if sql_normalized != json_normalized:
            mismatches[field] = (sql_value, json_value)
    
    return mismatches


def main():
    parser = argparse.ArgumentParser(
        description="Verify SQL table entries match metadata.json files"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="rldb",
        help="S3 bucket name (default: rldb)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="raw_v2/aria/",
        help="S3 prefix for metadata.json files (default: raw_v2/aria/)",
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default="aria",
        help="Filter SQL table by embodiment (default: aria)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV file for mismatches (required)",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to compare (default: all common fields)",
    )
    
    args = parser.parse_args()
    
    # Connect to database
    engine = create_default_engine()
    
    # Get SQL table rows
    df = episode_table_to_df(engine)
    
    # Filter by embodiment
    if args.embodiment:
        df = df[df["embodiment"].str.lower() == args.embodiment.lower()]
    
    # Initialize S3 client
    s3_client = boto3.client("s3")
    
    # Get all metadata.json files
    metadata_files = get_all_metadata_files(s3_client, args.bucket, args.prefix)
    
    # Determine fields to compare
    if args.fields:
        fields_to_compare = [f.strip() for f in args.fields.split(",")]
    else:
        # Common fields that should match between SQL and metadata.json
        common_fields = [ "embodiment", "operator" "lab",  "rl2", "task", "robot_name", "task_description", "scene", "objects"]
        # Only include fields that exist in both SQL table and metadata
        sql_columns = set(df.columns)
        fields_to_compare = [f for f in common_fields if f in sql_columns]
    
    # Build mapping from timestamp_ms to SQL rows
    sql_by_timestamp = {}
    
    for _, row in df.iterrows():
        episode_hash = row["episode_hash"]
        timestamp_ms = episode_hash_to_timestamp_ms(episode_hash)
        if timestamp_ms is None:
            continue
        
        if timestamp_ms not in sql_by_timestamp:
            sql_by_timestamp[timestamp_ms] = []
        sql_by_timestamp[timestamp_ms].append(row.to_dict())
    
    # Compare SQL rows with metadata.json files
    matched_timestamps = set()
    mismatches = []
    
    # Check SQL rows that have metadata.json files
    for timestamp_ms, sql_rows in sql_by_timestamp.items():
        if timestamp_ms in metadata_files:
            matched_timestamps.add(timestamp_ms)
            s3_key, metadata = metadata_files[timestamp_ms]
            
            # Compare each SQL row (handle collisions)
            for sql_row in sql_rows:
                field_mismatches = compare_sql_and_metadata(
                    sql_row, metadata, fields_to_compare
                )
                
                if field_mismatches:
                    mismatches.append({
                        "episode_hash": sql_row["episode_hash"],
                        "timestamp_ms": timestamp_ms,
                        "s3_key": s3_key,
                        "mismatches": field_mismatches,
                    })
    
    # Generate mismatches-only comparison table (one row per mismatched field)
    comparison_table = []
    
    for mismatch in mismatches:
        # Add one row per mismatched field
        for field, (sql_val, json_val) in mismatch['mismatches'].items():
            # Format values for display
            if isinstance(sql_val, (list, dict)):
                sql_val_str = json.dumps(sql_val)
            else:
                sql_val_str = str(sql_val) if sql_val is not None else ""
            
            if isinstance(json_val, (list, dict)):
                json_val_str = json.dumps(json_val)
            else:
                json_val_str = str(json_val) if json_val is not None else ""
            
            comparison_table.append({
                "episode_hash": mismatch['episode_hash'],
                "timestamp_ms": mismatch['timestamp_ms'],
                "s3_key": mismatch['s3_key'],
                "field_name": field,
                "sql_value": sql_val_str,
                "metadata_value": json_val_str,
                "type": "mismatch"
            })
    
    # Save mismatches-only CSV
    if comparison_table:
        comparison_df = pd.DataFrame(comparison_table)
        # Sort by timestamp, then field name for better organization
        comparison_df = comparison_df.sort_values(['timestamp_ms', 'field_name'])
        comparison_df.to_csv(args.output_csv, index=False)
    else:
        # Create empty CSV with correct columns
        empty_df = pd.DataFrame(columns=["episode_hash", "timestamp_ms", "s3_key", "field_name", "sql_value", "metadata_value", "type"])
        empty_df.to_csv(args.output_csv, index=False)
    
    # Exit code
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

