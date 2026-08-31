#!/usr/bin/env python3
"""
Verify that SQL episode table entries match metadata.json files in S3.

Dry-run by default: NO writes to SQL or S3 unless --apply is passed.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd
import sys
import os

sys.path.append("./egomimic/utils/aws")
from aws_sql import (
    create_default_engine,
    episode_table_to_df,
    TableRow,update_episode,
)


###############################################################################
# Utilities
###############################################################################

def episode_hash_to_timestamp_ms(episode_hash: str) -> Optional[int]:
    """
    Convert episode_hash (YYYY-MM-DD-HH-MM-SS-ffffff) to UTC timestamp in ms.
    """
    try:
        fmt = "%Y-%m-%d-%H-%M-%S-%f"
        dt = datetime.strptime(episode_hash, fmt).replace(tzinfo=timezone.utc)
        milliseconds = int(dt.timestamp() * 1000)
        return milliseconds
    except Exception as e:
        logging.warning(f"Failed to convert episode_hash to timestamp_ms: {e}")
        logging.warning(f"Episode hash: {episode_hash}")
        return None


import re

_SQL_ARRAY_RE = re.compile(r'^\{.*\}$')


def normalize(value):
    if value is None:
        return None

    # ---- Metadata JSON list ----
    if isinstance(value, list):
        return tuple(sorted(str(v).strip() for v in value))

    # ---- SQL array literal stored as string ----
    if isinstance(value, str):
        s = value.strip()

        # Looks like: {"a","b","c"}
        if _SQL_ARRAY_RE.match(s):
            inner = s[1:-1].strip()
            if not inner:
                return ()

            items = []
            buf = ""
            in_quotes = False

            for char in inner:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == "," and not in_quotes:
                    items.append(buf.strip())
                    buf = ""
                else:
                    buf += char

            if buf:
                items.append(buf.strip())

            cleaned = [item.strip('"').strip() for item in items]
            return tuple(sorted(cleaned))

        # Normal string
        return s

    # ---- Everything else ----
    return value



###############################################################################
# S3 Metadata Loading
###############################################################################

def get_all_metadata_files(
    s3_client,
    bucket_name: str,
    prefixes: List[str],
) -> Dict[int, Tuple[str, Dict]]:
    """
    Load all *_metadata.json files and index by timestamp_ms.
    """
    metadata_files: Dict[int, Tuple[str, Dict]] = {}
    paginator = s3_client.get_paginator("list_objects_v2")

    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("_metadata.json"):
                    continue

                filename = Path(key).stem.replace("_metadata", "")
                try:
                    timestamp_ms = int(filename)
                except ValueError:
                    continue

                try:
                    response = s3_client.get_object(Bucket=bucket_name, Key=key)
                    metadata = json.loads(response["Body"].read().decode("utf-8"))
                    metadata_files[timestamp_ms] = (key, metadata)
                except Exception as e:
                    logging.warning(f"Failed to load {key}: {e}")

    return metadata_files


###############################################################################
# Core Comparison Logic
###############################################################################

def compare_sql_and_metadata(
    *,
    s3_client,
    bucket_name: str,
    metadata_files: Dict[int, Tuple[str, Dict]],
    fields_to_compare: List[str],
    dry_run: bool,
):
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    df = df[df["lab"] != "mecka"]

    total_checked = 0
    total_changed = 0
    total_missing_metadata = 0

    for record in df.to_dict(orient="records"):
        episode_hash = record.get("episode_hash")
        timestamp_ms = episode_hash_to_timestamp_ms(episode_hash)

        if timestamp_ms is None:
            logging.warning(f"Bad episode_hash: {episode_hash}")
            continue

        entry = metadata_files.get(timestamp_ms)
        if entry is None:
            logging.warning(f"No metadata.json for {episode_hash}")
            total_missing_metadata += 1
            continue

        s3_key, metadata_original = entry
        metadata_new = dict(metadata_original)  # copy for safety

        diffs = {}
        sql_needs_update = False
        s3_needs_update = False

        for field in fields_to_compare:
            sql_value = normalize(record.get(field))
            json_value = normalize(metadata_original.get(field))

            if sql_value != json_value:
                diffs[field] = {
                    "metadata": json_value,
                    "sql": sql_value,
                }

                if field == "robot_name":
                    sql_needs_update = True
                else:
                    metadata_new[field] = record.get(field)
                    s3_needs_update = True

        total_checked += 1

        if not diffs:
            continue

        total_changed += 1
        tag = "[DRY-RUN]" if dry_run else "[APPLY]"

        logging.info(
            f"{tag} {episode_hash} → {s3_key}\n"
            f"{json.dumps(diffs, indent=2)}"
        )

        # --- SQL UPDATE ---
        if sql_needs_update:
            if dry_run:
                logging.info(f"{tag} WOULD UPDATE SQL episode {episode_hash}")
            else:
                record["processed_path"] = ""
                record["num_frames"] = -1
                record["mp4_path"] = ""
                episode = TableRow(**record)
                update_episode(engine, episode)
                logging.info(f"{tag} UPDATED SQL episode {episode_hash}")

        # --- S3 UPDATE ---
        if s3_needs_update:
            if dry_run:
                logging.info(f"{tag} WOULD UPDATE S3 metadata {s3_key}")
            else:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=json.dumps(metadata_new, indent=2),
                    ContentType="application/json",
                )
                logging.info(f"{tag} UPDATED S3 metadata {s3_key}")

    logging.info(
        f"Summary: checked={total_checked}, "
        f"changed={total_changed}, "
        f"missing_metadata={total_missing_metadata}"
    )


###############################################################################
# CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes (disable dry-run)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )

    dry_run = not args.apply

    s3_client = boto3.client("s3")
    bucket_name = "rldb"
    prefixes = ["raw_v2/aria", "raw_v2/eva"]

    fields_to_compare = [
        "operator",
        "lab",
        "task",
        "embodiment",
        "robot_name",
        "task_description",
        "scene",
        "objects",
    ]

    metadata_files = get_all_metadata_files(
        s3_client=s3_client,
        bucket_name=bucket_name,
        prefixes=prefixes,
    )

    compare_sql_and_metadata(
        s3_client=s3_client,
        bucket_name=bucket_name,
        metadata_files=metadata_files,
        fields_to_compare=fields_to_compare,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
