#!/usr/bin/env python3
"""
List timestamps that exist in processed_v2/aria but not in raw_v2/aria.

Assumes S3 bucket layout:
  processed_v2/aria/...
  raw_v2/aria/...

We extract timestamps from filenames by taking the stem up to the first
underscore or dot and keeping numeric-only strings (e.g., 1762544406656).

Outputs the missing timestamps and optionally writes a CSV.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Set, Tuple

import boto3


def extract_timestamp(filename: str) -> str | None:
    """
    Extract a timestamp substring from a filename.
    Strategy: take prefix up to first '_' or '.', ensure it's digits of length >= 8.
    """
    stem = re.split(r"[_.]", filename, maxsplit=1)[0]
    if stem.isdigit() and len(stem) >= 8:
        return stem
    return None


def collect_timestamps(
    bucket: str, prefix: str, s3_client, max_keys: int | None = None
) -> Tuple[Set[str], int]:
    """
    List objects under prefix and return (timestamps, total_files_seen).
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    ts_set: Set[str] = set()
    total = 0
    for page in page_iterator:
        contents = page.get("Contents", [])
        for obj in contents:
            total += 1
            key = obj["Key"]
            # remove prefix
            fname = key[len(prefix) :] if key.startswith(prefix) else key
            ts = extract_timestamp(fname)
            if ts:
                ts_set.add(ts)
            if max_keys and total >= max_keys:
                return ts_set, total
    return ts_set, total


def main():
    parser = argparse.ArgumentParser(
        description="Find timestamps present in processed_v2/aria but missing in raw_v2/aria"
    )
    parser.add_argument(
        "--bucket", default="rldb", help="S3 bucket name (default: rldb)"
    )
    parser.add_argument(
        "--processed-prefix",
        default="processed_v2/aria/",
        help="Processed prefix (default: processed_v2/aria/)",
    )
    parser.add_argument(
        "--raw-prefix",
        default="raw_v2/aria/",
        help="Raw prefix (default: raw_v2/aria/)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save missing timestamps CSV",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=None,
        help="Optional cap on number of S3 objects to scan (for quick tests)",
    )

    args = parser.parse_args()

    s3 = boto3.client("s3")

    print(f"Bucket: {args.bucket}")
    print(f"Processed prefix: {args.processed_prefix}")
    print(f"Raw prefix: {args.raw_prefix}")
    if args.max_keys:
        print(f"Max keys per listing: {args.max_keys}")
    print("Scanning processed...")
    proc_ts, proc_total = collect_timestamps(
        args.bucket, args.processed_prefix, s3, args.max_keys
    )
    print(f"  Processed files scanned: {proc_total}")
    print(f"  Unique processed timestamps: {len(proc_ts)}")

    print("Scanning raw...")
    raw_ts, raw_total = collect_timestamps(
        args.bucket, args.raw_prefix, s3, args.max_keys
    )
    print(f"  Raw files scanned: {raw_total}")
    print(f"  Unique raw timestamps: {len(raw_ts)}")

    missing = sorted(proc_ts - raw_ts)
    print()
    print(f"Timestamps in processed but NOT in raw: {len(missing)}")
    if missing:
        preview = ", ".join(missing[:20])
        print(f"First {min(20, len(missing))}: {preview}")

    if args.output_csv:
        import csv

        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_ms"])
            for ts in missing:
                writer.writerow([ts])
        print(f"Saved CSV: {out_path}")

    # Exit code: 0 if none missing, 1 if missing
    sys.exit(0 if not missing else 1)


if __name__ == "__main__":
    main()

