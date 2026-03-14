#!/usr/bin/env python3
"""
Script to verify which episodes in the SQL table actually exist in S3 bucket and vice versa.

For each episode_hash in the SQL table, this script:
1. Converts the episode_hash (YYYY-MM-DD-HH-MM-SS-ffffff) to a 13-digit timestamp in milliseconds
2. Uses UTC timezone for the conversion (matching original file timestamp conversion)
3. Checks if all 3 required files exist in S3:
   - <timestamp>_metadata.json
   - <timestamp>.json
   - <timestamp>.vrs
4. Also finds files in S3 that don't have entries in the SQL table
5. Outputs comprehensive summary of discrepancies in both directions
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import boto3

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path to import egomimic modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


def episode_hash_to_timestamp_ms(episode_hash: str) -> Optional[int]:
    """
    Convert episode_hash (YYYY-MM-DD-HH-MM-SS-ffffff) to timestamp in milliseconds.
    
    The episode_hash format is: YYYY-MM-DD-HH-MM-SS-ffffff
    This was created from: datetime.fromtimestamp(timestamp_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    
    Uses UTC timezone to match the original file timestamp conversion:
    timestamp_ms = int(stats.st_mtime * 1000)
    
    Args:
        episode_hash: String in format YYYY-MM-DD-HH-MM-SS-ffffff
        
    Returns:
        Timestamp in milliseconds (13-digit integer), or None if conversion fails
    """
    try:
        # Parse the episode_hash
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
        
        # Parse as UTC time (matching the original conversion)
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
        dt = dt.replace(tzinfo=timezone.utc)
        
        # Convert to milliseconds timestamp
        timestamp_ms = int(dt.timestamp() * 1000)
        return timestamp_ms
    except Exception as e:
        print(f"Error converting episode_hash {episode_hash} to timestamp: {e}")
        return None


def check_episode_files_in_s3(
    s3_client, bucket_name: str, prefix: str, timestamp_ms: int, embodiment: str = "aria"
) -> Dict[str, bool]:
    """
    Check if all required files exist in S3 for a given timestamp.
    
    For aria embodiment, required files:
    - <timestamp>_metadata.json
    - <timestamp>.json
    - <timestamp>.vrs
    
    For eva embodiment, required files:
    - <timestamp>.hdf5
    - <timestamp>_metadata.json
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name (e.g., "rldb")
        prefix: S3 prefix (e.g., "raw_v2/aria/")
        timestamp_ms: Timestamp in milliseconds to search for
        embodiment: "aria" or "eva" to determine which files to check
        
    Returns:
        Dictionary mapping file type to existence status
    """
    timestamp_str = str(timestamp_ms)
    
    if embodiment.lower() == "aria":
        required_files = {
            "metadata": f"{timestamp_str}_metadata.json",
            "json": f"{timestamp_str}.json",
            "vrs": f"{timestamp_str}.vrs"
        }
    elif embodiment.lower() == "eva":
        required_files = {
            "hdf5": f"{timestamp_str}.hdf5",
            "metadata": f"{timestamp_str}_metadata.json"
        }
    else:
        raise ValueError(f"Unsupported embodiment: {embodiment}. Must be 'aria' or 'eva'")
    
    results = {}
    
    for file_type, filename in required_files.items():
        s3_key = f"{prefix}{filename}"
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_key)
            results[file_type] = True
        except Exception as e:
            # Check if it's a 404 error (file not found)
            if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == '404':
                results[file_type] = False
            elif '404' in str(e) or 'Not Found' in str(e):
                results[file_type] = False
            else:
                # Other error (permissions, etc.)
                print(f"Error checking {s3_key}: {e}")
                results[file_type] = False
    
    return results


def get_all_s3_timestamps(
    s3_client, bucket_name: str, prefix: str, embodiment: str = "aria"
) -> Tuple[Dict[int, List[str]], List[str]]:
    """
    Get all timestamps from S3 files in the prefix.
    
    Uses pagination to handle S3's 1000 objects per request limit.
    The boto3 paginator automatically handles all pages, so there's no
    hard limit on the total number of objects that can be retrieved.
    
    Note: S3 list_objects_v2 returns max 1000 objects per request, but
    pagination handles this automatically. For very large prefixes (millions
    of objects), this may take significant time and API calls.
    
    Returns a tuple of:
    - Dictionary mapping timestamp_ms to list of file names found
    - List of files that don't match the expected pattern
    
    Args:
        s3_client: Boto3 S3 client
        bucket_name: S3 bucket name
        prefix: S3 prefix to search
        embodiment: "aria" or "eva" to determine which file patterns to match
        
    Returns:
        Tuple of (timestamps dict, unmatched_files list)
    """
    print("Scanning S3 bucket for all episode files...")
    print(f"Note: S3 returns max 1000 objects per page; pagination handles this automatically")
    timestamps = {}
    unmatched_files = []
    total_files_processed = 0
    total_pages = 0
    matched_files = 0
    
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in tqdm(page_iterator, desc="Scanning S3 pages"):
        total_pages += 1
        if "Contents" not in page:
            continue
        
        for obj in page["Contents"]:
            total_files_processed += 1
            key = obj["Key"]
            # Remove prefix to get filename
            filename = key.replace(prefix, "")
            
            # Extract timestamp from filename based on embodiment
            timestamp_str = None
            if embodiment.lower() == "aria":
                # Format: <timestamp>_metadata.json, <timestamp>.json, or <timestamp>.vrs
                if "_metadata.json" in filename:
                    timestamp_str = filename.replace("_metadata.json", "")
                elif filename.endswith(".json") and "_metadata" not in filename:
                    timestamp_str = filename.replace(".json", "")
                elif filename.endswith(".vrs"):
                    timestamp_str = filename.replace(".vrs", "")
            elif embodiment.lower() == "eva":
                # Format: <timestamp>.hdf5 or <timestamp>_metadata.json
                if "_metadata.json" in filename:
                    timestamp_str = filename.replace("_metadata.json", "")
                elif filename.endswith(".hdf5"):
                    timestamp_str = filename.replace(".hdf5", "")
            else:
                raise ValueError(f"Unsupported embodiment: {embodiment}. Must be 'aria' or 'eva'")
            
            if timestamp_str is None:
                # File doesn't match expected pattern
                unmatched_files.append(filename)
                continue
            
            try:
                timestamp_ms = int(timestamp_str)
                if timestamp_ms not in timestamps:
                    timestamps[timestamp_ms] = []
                timestamps[timestamp_ms].append(filename)
                matched_files += 1
            except ValueError:
                # Timestamp couldn't be parsed as integer
                unmatched_files.append(filename)
    
    print(f"Processed {total_files_processed} files across {total_pages} pages")
    print(f"  ✓ Matched expected pattern: {matched_files} files")
    print(f"  ✗ Unmatched (unexpected format): {len(unmatched_files)} files")
    print(f"  Found {len(timestamps)} unique timestamps in S3")
    
    # Calculate expected vs actual based on embodiment
    expected_files_per_timestamp = 3 if embodiment.lower() == "aria" else 2
    expected_files = len(timestamps) * expected_files_per_timestamp
    if len(timestamps) > 0:
        avg_files_per_timestamp = matched_files / len(timestamps)
        print(f"  Expected: ~{expected_files} files ({len(timestamps)} timestamps × {expected_files_per_timestamp})")
        print(f"  Actual: {matched_files} files ({avg_files_per_timestamp:.1f} files per timestamp on average)")
        if avg_files_per_timestamp > expected_files_per_timestamp + 0.5:
            print(f"  ⚠ WARNING: Average files per timestamp ({avg_files_per_timestamp:.1f}) exceeds expected ({expected_files_per_timestamp})")
    
    return timestamps, unmatched_files


def verify_episodes_in_s3(
    bucket_name: str = "rldb",
    prefix: str = "raw_v2/aria/",
    embodiment: str = "aria",
    batch_size: int = 100
) -> Dict:
    """
    Verify which episodes from SQL table exist in S3 bucket and vice versa.
    
    Checks:
    1. SQL -> S3: Are all episodes in SQL table present in S3 with all 3 required files?
    2. S3 -> SQL: Are there files in S3 that don't have entries in SQL table?
    
    Args:
        bucket_name: S3 bucket name
        prefix: S3 prefix to search in
        embodiment: Filter episodes by this embodiment
        batch_size: Number of episodes to process at once
        
    Returns:
        Dictionary with summary statistics
    """
    print("=" * 80)
    print("Episode Verification Script")
    print("=" * 80)
    print(f"Bucket: {bucket_name}")
    print(f"Prefix: {prefix}")
    print(f"Embodiment: {embodiment}")
    print("=" * 80)
    print()
    
    # Initialize S3 client
    print("Initializing S3 client...")
    s3_client = boto3.client("s3")
    
    # Connect to SQL database
    print("Connecting to SQL database...")
    engine = create_default_engine()
    
    # Get all episodes from SQL table
    print("Fetching episodes from SQL table...")
    df = episode_table_to_df(engine)
    
    # Filter by embodiment
    if embodiment:
        df = df[df["embodiment"].str.lower() == embodiment.lower()]
    
    total_episodes = len(df)
    print(f"Found {total_episodes} episodes with embodiment='{embodiment}' in SQL table")
    print()
    
    # Build mapping from timestamp_ms to episode_hash for SQL entries
    # Track collisions: multiple episode_hashes mapping to same timestamp
    sql_timestamps = {}  # timestamp_ms -> episode_hash (for first occurrence)
    sql_timestamps_all = {}  # timestamp_ms -> list of episode_hashes (for collision detection)
    sql_episode_hashes = set()  # All episode_hash values for reverse lookup
    
    print("Converting episode hashes to timestamps...")
    for idx, row in tqdm(df.iterrows(), total=total_episodes, desc="Converting hashes"):
        episode_hash = row["episode_hash"]
        sql_episode_hashes.add(episode_hash)
        timestamp_ms = episode_hash_to_timestamp_ms(episode_hash)
        if timestamp_ms is not None:
            if timestamp_ms not in sql_timestamps:
                sql_timestamps[timestamp_ms] = episode_hash
            if timestamp_ms not in sql_timestamps_all:
                sql_timestamps_all[timestamp_ms] = []
            sql_timestamps_all[timestamp_ms].append({
                "episode_hash": episode_hash,
                "operator": row.get("operator", "N/A"),
                "task": row.get("task", "N/A")
            })
    
    # Detect SQL hash collisions
    sql_collisions = []
    for timestamp_ms, episode_list in sql_timestamps_all.items():
        if len(episode_list) > 1:
            sql_collisions.append({
                "timestamp_ms": timestamp_ms,
                "episode_count": len(episode_list),
                "episodes": episode_list
            })
    
    print()
    if sql_collisions:
        print(f"⚠ WARNING: Found {len(sql_collisions)} SQL hash collisions!")
        print(f"   {sum(c['episode_count'] for c in sql_collisions)} episodes share timestamps with other episodes")
    else:
        print("✓ No SQL hash collisions found")
    print()
    
    # Get all S3 timestamps
    s3_timestamps, unmatched_s3_files = get_all_s3_timestamps(s3_client, bucket_name, prefix, embodiment)
    print()
    
    if unmatched_s3_files:
        print(f"⚠ Found {len(unmatched_s3_files)} files in S3 that don't match expected pattern")
        if embodiment.lower() == "aria":
            print(f"   Expected: <timestamp>_metadata.json, <timestamp>.json, or <timestamp>.vrs")
        elif embodiment.lower() == "eva":
            print(f"   Expected: <timestamp>.hdf5 or <timestamp>_metadata.json")
        if len(unmatched_s3_files) <= 20:
            print(f"   Unmatched files: {', '.join(unmatched_s3_files)}")
        else:
            print(f"   First 20 unmatched files: {', '.join(unmatched_s3_files[:20])}")
            print(f"   ... and {len(unmatched_s3_files) - 20} more")
    print()
    
    # Detect S3 hash collisions (multiple files with same timestamp)
    # This could indicate duplicate uploads or file naming issues
    s3_collisions = []
    s3_incomplete = []  # Timestamps with fewer than expected files
    
    # Determine expected files based on embodiment
    if embodiment.lower() == "aria":
        expected_file_count = 3
        def get_expected_files(ts):
            return [
                f"{ts}_metadata.json",
                f"{ts}.json",
                f"{ts}.vrs"
            ]
        def check_file_types(files):
            return {
                "has_metadata": any("_metadata.json" in f for f in files),
                "has_json": any(f.endswith(".json") and "_metadata" not in f for f in files),
                "has_vrs": any(f.endswith(".vrs") for f in files)
            }
    elif embodiment.lower() == "eva":
        expected_file_count = 2
        def get_expected_files(ts):
            return [
                f"{ts}.hdf5",
                f"{ts}_metadata.json"
            ]
        def check_file_types(files):
            return {
                "has_hdf5": any(f.endswith(".hdf5") for f in files),
                "has_metadata": any("_metadata.json" in f for f in files)
            }
    else:
        raise ValueError(f"Unsupported embodiment: {embodiment}")
    
    for timestamp_ms, files in s3_timestamps.items():
        expected_files = get_expected_files(timestamp_ms)
        file_types = check_file_types(files)
        
        if len(files) > expected_file_count:  # More than expected
            s3_collisions.append({
                "timestamp_ms": timestamp_ms,
                "file_count": len(files),
                "files": files,
                **file_types,
                "extra_files": [f for f in files if f not in expected_files],
                "missing_files": [f for f in expected_files if f not in files]
            })
        elif len(files) < expected_file_count:  # Fewer than expected
            s3_incomplete.append({
                "timestamp_ms": timestamp_ms,
                "file_count": len(files),
                "files": files,
                **file_types,
                "missing_files": [f for f in expected_files if f not in files]
            })
    
    expected_file_count = 3 if embodiment.lower() == "aria" else 2
    if s3_collisions:
        print(f"⚠ WARNING: Found {len(s3_collisions)} S3 hash collisions (more than {expected_file_count} files)!")
        print(f"   {sum(c['file_count'] for c in s3_collisions)} total files in collision groups")
    if s3_incomplete:
        print(f"⚠ WARNING: Found {len(s3_incomplete)} S3 timestamps with incomplete file sets (fewer than {expected_file_count} files)!")
    if not s3_collisions and not s3_incomplete:
        print(f"✓ No S3 hash collisions found (all timestamps have exactly {expected_file_count} files)")
    print()
    
    # Track results for SQL -> S3 check
    complete_count = 0  # All 3 files present
    incomplete_count = 0  # Some files missing
    missing_count = 0  # No files found
    conversion_failed_count = 0
    missing_episodes = []
    
    print("Checking SQL episodes in S3...")
    print()
    
    # Check each SQL episode
    for idx, row in tqdm(df.iterrows(), total=total_episodes, desc="Verifying SQL episodes"):
        episode_hash = row["episode_hash"]
        
        # Convert episode_hash to timestamp in UTC (matching file timestamp conversion)
        timestamp_ms = episode_hash_to_timestamp_ms(episode_hash)
        
        if timestamp_ms is None:
            conversion_failed_count += 1
            missing_episodes.append({
                "episode_hash": episode_hash,
                "timestamp_ms": None,
                "operator": row.get("operator", "N/A"),
                "task": row.get("task", "N/A"),
                "missing_files": "Conversion failed",
                "status": "conversion_failed"
            })
            continue
        
        # Check which files exist in S3
        file_status = check_episode_files_in_s3(
            s3_client, bucket_name, prefix, timestamp_ms, embodiment
        )
        
        missing_files = [ftype for ftype, exists in file_status.items() if not exists]
        
        if all(file_status.values()):
            # All required files exist
            complete_count += 1
        elif any(file_status.values()):
            # Some files exist but not all
            incomplete_count += 1
            episode_info = {
                "episode_hash": episode_hash,
                "timestamp_ms": timestamp_ms,
                "operator": row.get("operator", "N/A"),
                "task": row.get("task", "N/A"),
                "missing_files": ", ".join(missing_files),
                "status": "incomplete"
            }
            # Add file status based on embodiment
            if embodiment.lower() == "aria":
                episode_info.update({
                    "has_metadata": file_status.get("metadata", False),
                    "has_json": file_status.get("json", False),
                    "has_vrs": file_status.get("vrs", False)
                })
            elif embodiment.lower() == "eva":
                episode_info.update({
                    "has_hdf5": file_status.get("hdf5", False),
                    "has_metadata": file_status.get("metadata", False)
                })
            missing_episodes.append(episode_info)
        else:
            # No files found
            missing_count += 1
            if embodiment.lower() == "aria":
                missing_files_str = "all (metadata.json, .json, .vrs)"
            elif embodiment.lower() == "eva":
                missing_files_str = "all (hdf5, metadata.json)"
            else:
                missing_files_str = "all files"
            missing_episodes.append({
                "episode_hash": episode_hash,
                "timestamp_ms": timestamp_ms,
                "operator": row.get("operator", "N/A"),
                "task": row.get("task", "N/A"),
                "missing_files": missing_files_str,
                "status": "missing"
            })
    
    # Check S3 -> SQL: Find files in S3 that don't have SQL entries
    print()
    print("Checking for S3 files not in SQL table...")
    orphaned_s3_files = []
    
    for timestamp_ms, files in tqdm(s3_timestamps.items(), desc="Checking S3 orphans"):
        if timestamp_ms not in sql_timestamps:
            # This timestamp is in S3 but not in SQL
            orphaned_s3_files.append({
                "timestamp_ms": timestamp_ms,
                "files": files,
                "missing_in_sql": True
            })
    
    # Print comprehensive summary
    print()
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    expected_file_count = 3 if embodiment.lower() == "aria" else 2
    print("SQL → S3 Check (Episodes in SQL table):")
    print(f"  Total episodes in SQL: {total_episodes}")
    print(f"  ✓ Complete (all {expected_file_count} files present): {complete_count}")
    print(f"  ⚠ Incomplete (some files missing): {incomplete_count}")
    print(f"  ✗ Missing (no files found): {missing_count}")
    print(f"  ⚠ Conversion failed: {conversion_failed_count}")
    print()
    
    print("S3 → SQL Check (Files in S3 bucket):")
    print(f"  Total unique timestamps in S3: {len(s3_timestamps)}")
    print(f"  ✗ Orphaned (not in SQL table): {len(orphaned_s3_files)}")
    print()
    
    expected_file_count = 3 if embodiment.lower() == "aria" else 2
    print("Hash Collisions:")
    print(f"  SQL collisions: {len(sql_collisions)} timestamps with multiple episode_hashes")
    print(f"  S3 collisions: {len(s3_collisions)} timestamps with more than {expected_file_count} files")
    print(f"  S3 incomplete: {len(s3_incomplete)} timestamps with fewer than {expected_file_count} files")
    print(f"  Unmatched files: {len(unmatched_s3_files)} files that don't match expected pattern")
    print()
    
    # Detailed reports
    if missing_episodes:
        print("=" * 80)
        print("SQL EPISODES WITH MISSING FILES")
        print("=" * 80)
        
        # Group by status
        conversion_failed = [e for e in missing_episodes if e["status"] == "conversion_failed"]
        incomplete = [e for e in missing_episodes if e["status"] == "incomplete"]
        missing = [e for e in missing_episodes if e["status"] == "missing"]
        
        if conversion_failed:
            print(f"\nConversion Failed ({len(conversion_failed)}):")
            print("-" * 80)
            for ep in conversion_failed[:10]:
                print(f"  {ep['episode_hash']} (Operator: {ep['operator']}, Task: {ep['task']})")
            if len(conversion_failed) > 10:
                print(f"  ... and {len(conversion_failed) - 10} more")
        
        if incomplete:
            print(f"\nIncomplete Files ({len(incomplete)}):")
            print("-" * 80)
            for ep in incomplete[:20]:
                print(f"  {ep['episode_hash']} (Timestamp: {ep['timestamp_ms']})")
                print(f"    Missing: {ep['missing_files']}")
                if embodiment.lower() == "aria":
                    print(f"    Has metadata.json: {ep.get('has_metadata', False)}, "
                          f"Has .json: {ep.get('has_json', False)}, "
                          f"Has .vrs: {ep.get('has_vrs', False)}")
                elif embodiment.lower() == "eva":
                    print(f"    Has hdf5: {ep.get('has_hdf5', False)}, "
                          f"Has metadata.json: {ep.get('has_metadata', False)}")
            if len(incomplete) > 20:
                print(f"  ... and {len(incomplete) - 20} more")
        
        if missing:
            print(f"\nCompletely Missing ({len(missing)}):")
            print("-" * 80)
            for ep in missing[:20]:
                print(f"  {ep['episode_hash']} (Timestamp: {ep['timestamp_ms']}, "
                      f"Operator: {ep['operator']}, Task: {ep['task']})")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        
        # Save detailed report
        import pandas as pd
        missing_df = pd.DataFrame(missing_episodes)
        report_file = "missing_episodes_report.csv"
        missing_df.to_csv(report_file, index=False)
        print(f"\nDetailed report saved to: {report_file}")
    
    if orphaned_s3_files:
        print("=" * 80)
        print("ORPHANED S3 FILES (Not in SQL table)")
        print("=" * 80)
        print(f"\nFound {len(orphaned_s3_files)} timestamps in S3 that are not in SQL table:\n")
        
        for orphan in orphaned_s3_files[:30]:
            print(f"  Timestamp: {orphan['timestamp_ms']}")
            print(f"    Files: {', '.join(orphan['files'])}")
        if len(orphaned_s3_files) > 30:
            print(f"  ... and {len(orphaned_s3_files) - 30} more")
        
        # Save orphaned files report
        import pandas as pd
        orphaned_df = pd.DataFrame([
            {
                "timestamp_ms": o["timestamp_ms"],
                "files": ", ".join(o["files"]),
                "file_count": len(o["files"])
            }
            for o in orphaned_s3_files
        ])
        orphaned_file = "orphaned_s3_files_report.csv"
        orphaned_df.to_csv(orphaned_file, index=False)
        print(f"\nOrphaned files report saved to: {orphaned_file}")
    
    # Report hash collisions and unmatched files
    if sql_collisions or s3_collisions or s3_incomplete or unmatched_s3_files:
        print("=" * 80)
        print("HASH COLLISIONS & FILE PATTERN REPORT")
        print("=" * 80)
        
        if sql_collisions:
            print(f"\nSQL Hash Collisions ({len(sql_collisions)}):")
            print("-" * 80)
            print("Multiple episode_hashes in SQL table map to the same timestamp_ms")
            print("This indicates potential duplicate timestamps or data entry issues.\n")
            
            for collision in sql_collisions[:20]:  # Show first 20
                print(f"  Timestamp: {collision['timestamp_ms']}")
                print(f"  Episodes with same timestamp ({collision['episode_count']}):")
                for ep in collision['episodes']:
                    print(f"    - {ep['episode_hash']} (Operator: {ep['operator']}, Task: {ep['task']})")
            if len(sql_collisions) > 20:
                print(f"  ... and {len(sql_collisions) - 20} more collisions")
            
            # Save SQL collisions report
            import pandas as pd
            sql_collisions_flattened = []
            for collision in sql_collisions:
                for ep in collision['episodes']:
                    sql_collisions_flattened.append({
                        "timestamp_ms": collision['timestamp_ms'],
                        "episode_hash": ep['episode_hash'],
                        "operator": ep['operator'],
                        "task": ep['task'],
                        "collision_count": collision['episode_count']
                    })
            
            sql_collisions_df = pd.DataFrame(sql_collisions_flattened)
            sql_collisions_file = "sql_hash_collisions_report.csv"
            sql_collisions_df.to_csv(sql_collisions_file, index=False)
            print(f"\nSQL collisions report saved to: {sql_collisions_file}")
        
        expected_file_count = 3 if embodiment.lower() == "aria" else 2
        if s3_collisions:
            print(f"\nS3 Hash Collisions ({len(s3_collisions)}):")
            print("-" * 80)
            print("Timestamps in S3 with unexpected number of files")
            if embodiment.lower() == "aria":
                print("Expected: 3 files per timestamp (_metadata.json, .json, .vrs)")
            elif embodiment.lower() == "eva":
                print("Expected: 2 files per timestamp (.hdf5, _metadata.json)")
            print("Collisions indicate duplicate files or extra files with same timestamp.\n")
            
            for collision in s3_collisions[:20]:  # Show first 20
                print(f"  Timestamp: {collision['timestamp_ms']}")
                print(f"  File count: {collision['file_count']} (expected: {expected_file_count})")
                print(f"  Files: {', '.join(collision['files'])}")
                if collision['extra_files']:
                    print(f"  Extra files: {', '.join(collision['extra_files'])}")
            if len(s3_collisions) > 20:
                print(f"  ... and {len(s3_collisions) - 20} more collisions")
            
            # Save S3 collisions report
            import pandas as pd
            s3_collisions_flattened = []
            for collision in s3_collisions:
                row = {
                    "timestamp_ms": collision['timestamp_ms'],
                    "file_count": collision['file_count'],
                    "files": ", ".join(collision['files']),
                    "has_metadata": collision.get('has_metadata', False),
                    "extra_files": ", ".join(collision['extra_files']) if collision['extra_files'] else ""
                }
                # Add embodiment-specific fields
                if embodiment.lower() == "aria":
                    row.update({
                        "has_json": collision.get('has_json', False),
                        "has_vrs": collision.get('has_vrs', False)
                    })
                elif embodiment.lower() == "eva":
                    row.update({
                        "has_hdf5": collision.get('has_hdf5', False)
                    })
                s3_collisions_flattened.append(row)
            
            s3_collisions_df = pd.DataFrame(s3_collisions_flattened)
            s3_collisions_file = "s3_hash_collisions_report.csv"
            s3_collisions_df.to_csv(s3_collisions_file, index=False)
            print(f"\nS3 collisions report saved to: {s3_collisions_file}")
        
        if s3_incomplete:
            print(f"\nS3 Incomplete File Sets ({len(s3_incomplete)}):")
            print("-" * 80)
            print(f"Timestamps in S3 with fewer than {expected_file_count} expected files")
            if embodiment.lower() == "aria":
                print("Expected: 3 files per timestamp (_metadata.json, .json, .vrs)\n")
            elif embodiment.lower() == "eva":
                print("Expected: 2 files per timestamp (.hdf5, _metadata.json)\n")
            
            for incomplete in s3_incomplete[:20]:  # Show first 20
                print(f"  Timestamp: {incomplete['timestamp_ms']}")
                print(f"  File count: {incomplete['file_count']} (expected: {expected_file_count})")
                print(f"  Files present: {', '.join(incomplete['files'])}")
                print(f"  Missing: {', '.join(incomplete['missing_files'])}")
            if len(s3_incomplete) > 20:
                print(f"  ... and {len(s3_incomplete) - 20} more incomplete sets")
            
            # Save incomplete sets report
            import pandas as pd
            s3_incomplete_df = pd.DataFrame([
                {
                    "timestamp_ms": i['timestamp_ms'],
                    "file_count": i['file_count'],
                    "files": ", ".join(i['files']),
                    "missing_files": ", ".join(i['missing_files']),
                    "has_metadata": i.get('has_metadata', False),
                    "has_json": i.get('has_json', False) if embodiment.lower() == "aria" else None,
                    "has_vrs": i.get('has_vrs', False) if embodiment.lower() == "aria" else None,
                    "has_hdf5": i.get('has_hdf5', False) if embodiment.lower() == "eva" else None
                }
                for i in s3_incomplete
            ])
            incomplete_file = "s3_incomplete_files_report.csv"
            s3_incomplete_df.to_csv(incomplete_file, index=False)
            print(f"\nS3 incomplete sets report saved to: {incomplete_file}")
        
        if unmatched_s3_files:
            print(f"\nUnmatched S3 Files ({len(unmatched_s3_files)}):")
            print("-" * 80)
            print("Files in S3 that don't match expected naming pattern")
            if embodiment.lower() == "aria":
                print("Expected: <timestamp>_metadata.json, <timestamp>.json, or <timestamp>.vrs\n")
            elif embodiment.lower() == "eva":
                print("Expected: <timestamp>.hdf5 or <timestamp>_metadata.json\n")
            
            # Show examples of unmatched files
            print("Sample unmatched files:")
            for filename in unmatched_s3_files[:30]:
                print(f"  - {filename}")
            if len(unmatched_s3_files) > 30:
                print(f"  ... and {len(unmatched_s3_files) - 30} more")
            
            # Save unmatched files report
            import pandas as pd
            unmatched_df = pd.DataFrame({"filename": unmatched_s3_files})
            unmatched_file = "unmatched_s3_files_report.csv"
            unmatched_df.to_csv(unmatched_file, index=False)
            print(f"\nUnmatched files report saved to: {unmatched_file}")
        
        print()
    
    if (not missing_episodes and not orphaned_s3_files and not sql_collisions and 
        not s3_collisions and not s3_incomplete and not unmatched_s3_files):
        print("✓ Perfect match! All SQL episodes have all files in S3, all S3 files have SQL entries, no collisions, and all files match expected patterns!")
    
    print()
    print("=" * 80)
    
    return {
        "sql_total": total_episodes,
        "sql_complete": complete_count,
        "sql_incomplete": incomplete_count,
        "sql_missing": missing_count,
        "sql_conversion_failed": conversion_failed_count,
        "missing_episodes": missing_episodes,
        "s3_total_timestamps": len(s3_timestamps),
        "s3_orphaned_count": len(orphaned_s3_files),
        "orphaned_s3_files": orphaned_s3_files,
        "sql_collisions": sql_collisions,
        "s3_collisions": s3_collisions,
        "s3_incomplete": s3_incomplete,
        "unmatched_s3_files": unmatched_s3_files
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify which episodes in SQL table exist in S3 bucket"
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="S3 bucket name (default: rldb, or enter interactively)"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="S3 prefix to search (default: raw_v2/aria/ or raw_v2/eva/, or enter interactively)"
    )
    parser.add_argument(
        "--embodiment",
        default=None,
        help="Filter by embodiment: 'aria' or 'eva' (default: aria, or enter interactively)"
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive prompts and use defaults/command line args only"
    )
    
    args = parser.parse_args()
    
    # Interactive mode: prompt for overrides if not provided
    bucket_name = args.bucket
    prefix = args.prefix
    embodiment = args.embodiment
    
    if not args.non_interactive:
        print("=" * 80)
        print("Interactive Mode - Enter values or press Enter to use defaults")
        print("=" * 80)
        
        # Get bucket
        if bucket_name is None:
            user_input = input("S3 bucket name [default: rldb]: ").strip()
            bucket_name = user_input if user_input else "rldb"
        else:
            print(f"S3 bucket name: {bucket_name}")
        
        # Get embodiment first (needed for default prefix)
        if embodiment is None:
            user_input = input("Embodiment (aria/eva) [default: aria]: ").strip().lower()
            embodiment = user_input if user_input in ["aria", "eva"] else "aria"
        else:
            print(f"Embodiment: {embodiment}")
        
        # Get prefix with embodiment-specific default
        if prefix is None:
            default_prefix = f"raw_v2/{embodiment}/"
            user_input = input(f"S3 prefix [default: {default_prefix}]: ").strip()
            prefix = user_input if user_input else default_prefix
        else:
            print(f"S3 prefix: {prefix}")
        
        print("=" * 80)
        print()
    else:
        # Non-interactive mode: use defaults if not provided
        if bucket_name is None:
            bucket_name = "rldb"
        if embodiment is None:
            embodiment = "aria"
        if prefix is None:
            prefix = f"raw_v2/{embodiment}/"
    
    try:
        results = verify_episodes_in_s3(
            bucket_name=bucket_name,
            prefix=prefix,
            embodiment=embodiment
        )
        
        # Exit with appropriate code
        if (results["sql_incomplete"] > 0 or 
            results["sql_missing"] > 0 or 
            results["sql_conversion_failed"] > 0 or
            results["s3_orphaned_count"] > 0):
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

