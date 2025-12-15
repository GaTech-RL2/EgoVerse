#!/usr/bin/env python3
"""
Download only CSV files that correspond to .vrs recordings in S3.

This tool lists .vrs keys under a prefix and attempts to download only the
related CSV files (it will NOT download the .vrs files or .vrs.json files).
CSV name candidates checked: '<vrs>.csv', '<base>.csv', '<base>.vrs.csv'.

Features:
- list/paginate S3 keys under a prefix
- dry-run mode
- interactive selection with preview
- downloads only CSVs when available
"""

import argparse
import boto3
import botocore
import os
import sys
import math
import textwrap
from datetime import datetime
from typing import List, Dict, Tuple

DEFAULT_PREVIEW_CHARS = 400
VERBOSE = False


def human_readable_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit in ['KB', 'MB', 'GB', 'TB']:
        nbytes /= 1024.0
        if nbytes < 1024.0:
            return f"{nbytes:.2f} {unit}"
    return f"{nbytes:.2f} PB"


def get_s3_client(profile: str = None):
    if profile:
        session = boto3.Session(profile_name=profile)
        return session.client('s3')
    return boto3.client('s3')


def list_objects(bucket: str, prefix: str, s3_client) -> List[Dict]:
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iter = paginator.paginate(Bucket=bucket, Prefix=prefix)
    items = []
    for page in page_iter:
        for obj in page.get('Contents', []):
            items.append(obj)
    return items


def choose_candidates_for_vrs(vrs_key: str) -> List[str]:
    # Given 'path/to/file.vrs', common candidate CSV keys:
    # 1) path/to/file.vrs.csv
    # 2) path/to/file.csv  (strip .vrs)
    # 3) path/to/file.csv (redundant but safe)
    # 4) path/to/file_meta.csv (observed pattern in this bucket)
    candidates = []
    # Try a few common patterns for CSV files related to a .vrs key.
    #  - <vrs>.csv
    #  - <base>.csv
    #  - <base>.vrs.csv
    #  - <base>_meta.csv
    #  - <base>_dm2_meta.csv
    candidates.append(vrs_key + '.csv')
    if vrs_key.lower().endswith('.vrs'):
        base = vrs_key[:-4]
        candidates.append(base + '.csv')
        candidates.append(base + '.vrs.csv')
        candidates.append(base + '_meta.csv')
        candidates.append(base + '_dm2_meta.csv')
    return candidates


def key_exists(bucket: str, key: str, s3_client) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        if VERBOSE:
            print(f"FOUND: s3://{bucket}/{key}")
        return True
    except botocore.exceptions.ClientError:
        if VERBOSE:
            print(f"missing: s3://{bucket}/{key}")
        return False


def download_key(bucket: str, key: str, dest_dir: str, s3_client, dry_run: bool = False):
    dest_path = os.path.join(dest_dir, os.path.basename(key))
    os.makedirs(dest_dir, exist_ok=True)
    if dry_run:
        print(f"[DRY] Would download s3://{bucket}/{key} -> {dest_path}")
        return dest_path
    try:
        s3_client.download_file(bucket, key, dest_path)
        print(f"Downloaded s3://{bucket}/{key} -> {dest_path}")
        return dest_path
    except botocore.exceptions.ClientError as e:
        print(f"Failed to download {key}: {e}")
        return None


def interactive_select(vrs_objects: List[Dict], bucket: str, s3_client, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> List[str]:
    """Return list of selected vrs keys (interactive)."""
    keys = [obj['Key'] for obj in vrs_objects]
    print(f"Found {len(keys)} .vrs files under the prefix.")
    for i, k in enumerate(keys[:50], 1):
        size = human_readable_size(vrs_objects[i-1]['Size'])
        lm = vrs_objects[i-1]['LastModified']
        print(f"{i:3d}. {k} ({size}, {lm})")

    print(textwrap.dedent("""
    Enter selections as comma-separated indices or ranges, e.g.:
      1,3-5  -> select items 1,3,4,5
      all    -> select all
      none   -> select none
      q      -> quit
    You can also type 'd' to begin download of currently selected items.
    """))

    selected = set()
    while True:
        resp = input("selection> ").strip().lower()
        if not resp:
            continue
        if resp in ('q', 'quit'):
            return []
        if resp == 'all':
            return keys
        if resp == 'none':
            return []
        if resp == 'd':
            return [keys[i-1] for i in sorted(selected)]
        parts = [p.strip() for p in resp.split(',')]
        for p in parts:
            if '-' in p:
                a, b = p.split('-', 1)
                try:
                    a_i = int(a); b_i = int(b)
                    for idx in range(a_i, b_i+1):
                        if 1 <= idx <= len(keys):
                            selected.add(idx)
                except ValueError:
                    print(f"Invalid range: {p}")
            else:
                try:
                    i = int(p)
                    if 1 <= i <= len(keys):
                        selected.add(i)
                except ValueError:
                    print(f"Unknown token: {p}")

        if selected:
            print("Selected:", ','.join(str(s) for s in sorted(selected)))


def download_csvs_for_vrs(bucket: str, prefix: str, dest: str, s3_client, dry_run: bool = False, interactive: bool = False):
    objs = list_objects(bucket, prefix, s3_client)
    vrs_objs = [o for o in objs if o['Key'].lower().endswith('.vrs')]
    if not vrs_objs:
        print("No .vrs files found under the prefix")
        return

    if interactive:
        selected_keys = interactive_select(vrs_objs, bucket, s3_client)
        if not selected_keys:
            print("No selection made. Exiting.")
            return
        vrs_objs = [o for o in vrs_objs if o['Key'] in selected_keys]

    print(f"Preparing to download CSVs for {len(vrs_objs)} recordings to {dest}")
    total_csvs = 0
    for o in vrs_objs:
        k = o['Key']
        # Only try CSV candidates; do not download .vrs or .json files
        for cand in choose_candidates_for_vrs(k):
            if key_exists(bucket, cand, s3_client):
                download_key(bucket, cand, dest, s3_client, dry_run=dry_run)
                total_csvs += 1
    print(f"Finished. CSVs downloaded (or planned if dry-run): {total_csvs}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Backfill CSVs corresponding to .vrs recordings from S3')
    # Support simple positional invocation: bucket prefix [dest]
    p.add_argument('bucket_pos', nargs='?', default=None, help='S3 bucket name (positional)')
    p.add_argument('prefix_pos', nargs='?', default=None, help='S3 prefix (positional)')
    p.add_argument('dest_pos', nargs='?', default=None, help='Local destination directory (positional)')

    # Keep flags for advanced usage, but they're optional — script will prompt if missing
    p.add_argument('--bucket', help='S3 bucket name')
    p.add_argument('--prefix', help='S3 prefix to search under')
    p.add_argument('--dest', default='.', help='Local destination directory')
    p.add_argument('--profile', default=None, help='AWS profile to use')
    p.add_argument('--dry-run', action='store_true', help='Do not actually download files')
    p.add_argument('--interactive', action='store_true', help='Prompt to interactively select which files to download')
    p.add_argument('--preview-chars', type=int, default=DEFAULT_PREVIEW_CHARS, help='Number of characters to show when previewing files')
    p.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    global VERBOSE
    VERBOSE = bool(args.verbose)

    # Positional overrides: prefer positional args when provided
    bucket = args.bucket_pos if args.bucket_pos else args.bucket
    prefix = args.prefix_pos if args.prefix_pos else args.prefix
    dest = args.dest_pos if args.dest_pos else args.dest

    # Prompt for missing values interactively (no flags required)
    if not bucket:
        bucket = input('S3 bucket: ').strip()
    if not prefix:
        prefix = input('S3 prefix: ').strip()
    if not bucket or not prefix:
        print('Bucket and prefix are required')
        return

    if not dest or dest == '.':
        resp = input(f"Destination directory [{dest}]: ").strip()
        if resp:
            dest = resp

    s3 = get_s3_client(args.profile)
    download_csvs_for_vrs(bucket, prefix, dest, s3, dry_run=args.dry_run, interactive=args.interactive)


if __name__ == '__main__':
    main()
