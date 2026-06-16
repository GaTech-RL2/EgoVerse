from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import PurePosixPath

from sqlalchemy import MetaData, Table, bindparam, update

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


def parse_s3_uri(uri: str | None):
    if not uri:
        return None, None
    text = str(uri)
    if not text.startswith("s3://"):
        return None, None
    rest = text[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def parse_rldb_uri(uri: str | None):
    if not uri:
        return None, None
    text = str(uri)
    if not text.startswith("rldb:/"):
        return None, None
    rest = text[len("rldb:/") :]
    return "rldb", rest.lstrip("/")


def normalize_key_prefix(prefix: str):
    if not prefix:
        return ""
    text = str(prefix).strip().strip("/")
    normalized = PurePosixPath(text).as_posix()
    return normalized + "/"


def collect_required_prefixes(df):
    required_by_bucket: dict[str, set[str]] = {}
    skipped = 0
    rldb_paths = []
    for _, row in df.iterrows():
        processed_path = row.get("processed_path")
        bucket, key = parse_s3_uri(processed_path)
        if not bucket:
            if processed_path and str(processed_path).startswith("rldb:/"):
                rldb_paths.append(str(processed_path))
            bucket, key = parse_rldb_uri(processed_path)
        if not bucket or not key:
            skipped += 1
            continue
        prefix = normalize_key_prefix(key)
        if not prefix:
            skipped += 1
            continue
        required_by_bucket.setdefault(bucket, set()).add(prefix)
    return required_by_bucket, skipped, rldb_paths


def build_missing_updates(df, missing_rows):
    missing_set = {(bucket, prefix) for bucket, prefix in missing_rows}
    updates = []
    for _, row in df.iterrows():
        processed_path = row.get("processed_path")
        bucket, key = parse_s3_uri(processed_path)
        if not bucket:
            bucket, key = parse_rldb_uri(processed_path)
        if not bucket or not key:
            continue
        prefix = normalize_key_prefix(key)
        if (bucket, prefix) not in missing_set:
            continue
        updates.append(
            {
                "b_episode_hash": row.get("episode_hash"),
                "b_processed_path": "",
            }
        )
    return updates


def clear_missing_processed_paths(engine, updates):
    if not updates:
        print("No missing processed_path rows to clear.")
        return
    metadata = MetaData()
    episodes_tbl = Table("episodes", metadata, autoload_with=engine, schema="app")
    stmt = (
        update(episodes_tbl)
        .where(episodes_tbl.c.episode_hash == bindparam("b_episode_hash"))
        .values(processed_path=bindparam("b_processed_path"))
    )
    with engine.begin() as conn:
        conn.execute(stmt, updates)


def parent_prefix(prefix: str):
    if not prefix:
        return ""
    text = prefix.rstrip("/")
    parent = PurePosixPath(text).parent.as_posix()
    if parent == ".":
        return ""
    return parent + "/"


def list_common_prefixes(bucket: str, prefix: str, delimiter: str = "/"):
    cmd = ["aws", "s3api", "list-objects-v2", "--bucket", bucket]
    if delimiter:
        cmd.extend(["--delimiter", delimiter])
    if prefix:
        cmd.extend(["--prefix", prefix])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    payload = json.loads(result.stdout or "{}")
    common = payload.get("CommonPrefixes", [])
    return {item.get("Prefix") for item in common if item.get("Prefix")}


def find_prefixes_by_parent(bucket: str, required: set[str]):
    parent_map: dict[str, set[str]] = {}
    for prefix in required:
        parent_map.setdefault(parent_prefix(prefix), set()).add(prefix)
    found: set[str] = set()
    for parent, children in parent_map.items():
        common = list_common_prefixes(bucket, parent)
        for child in children:
            if child in common:
                found.add(child)
    return found


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check S3 processed_path prefixes against SQL."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-missing",
        action="store_true",
        help="Report processed_path prefixes missing on S3 (default).",
    )
    mode.add_argument(
        "--check-orphans",
        action="store_true",
        help="Report processed_v2 folders not referenced by SQL processed_path.",
    )
    parser.add_argument(
        "--processed-v2-bucket",
        default="rldb",
        help="Bucket to scan for processed_v2 orphans.",
    )
    parser.add_argument(
        "--processed-v2-prefix",
        default="processed_v2/",
        help="Prefix to scan for processed_v2 orphans.",
    )
    parser.add_argument(
        "--csv-dir",
        default="",
        help="Directory to write missing/orphans CSVs.",
    )
    parser.add_argument(
        "--clear-missing",
        action="store_true",
        help="Clear processed_path for rows missing on S3 (missing mode only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without updating SQL (used with --clear-missing).",
    )
    return parser.parse_args()


def load_required_prefixes():
    engine = create_default_engine()
    df = episode_table_to_df(engine)
    return collect_required_prefixes(df)


def report_rldb_paths(rldb_paths):
    if not rldb_paths:
        return
    print(f"ERROR: Found {len(rldb_paths)} rldb:/ processed_path values.")
    for path in rldb_paths[:20]:
        print(f"rldb:/ path: {path}")


def check_missing_prefixes(required_by_bucket):
    missing_rows = []
    for bucket, required in required_by_bucket.items():
        print(f"Listing bucket {bucket} at processed_path depth...")
        found = find_prefixes_by_parent(bucket, required)
        missing = sorted(required - found)
        print(f"{bucket}: {len(found)} found, {len(missing)} missing.")
        for prefix in missing:
            missing_rows.append((bucket, prefix))
        if missing:
            print(f"Sample missing ({min(20, len(missing))}):")
            for prefix in missing[:20]:
                print(f"s3://{bucket}/{prefix}")
    return missing_rows


def check_orphans(required_by_bucket, v2_bucket, v2_prefix):
    v2_prefix = normalize_key_prefix(v2_prefix)
    required = required_by_bucket.get(v2_bucket, set())
    required_v2 = {p for p in required if p.startswith(v2_prefix)}
    print(f"Checking orphans under s3://{v2_bucket}/{v2_prefix}...")
    existing_v2 = list_common_prefixes(v2_bucket, v2_prefix)
    orphans = sorted(existing_v2 - required_v2)
    print(f"Orphaned prefixes: {len(orphans)}")
    if orphans:
        print(f"Sample orphans ({min(20, len(orphans))}):")
        for prefix in orphans[:20]:
            print(f"s3://{v2_bucket}/{prefix}")
    return v2_prefix, orphans


def write_missing_csv(csv_dir, missing_rows):
    missing_path = os.path.join(csv_dir, "missing_processed_paths.csv")
    with open(missing_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "prefix"])
        for bucket, prefix in missing_rows:
            writer.writerow([bucket, prefix])
    print(f"Wrote missing CSV to {missing_path}")


def write_orphans_csv(csv_dir, v2_bucket, orphans):
    orphans_path = os.path.join(csv_dir, "orphaned_processed_v2.csv")
    with open(orphans_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "prefix"])
        for prefix in orphans:
            writer.writerow([v2_bucket, prefix])
    print(f"Wrote orphans CSV to {orphans_path}")


def main():
    args = parse_args()

    engine = create_default_engine()
    df = episode_table_to_df(engine)
    required_by_bucket, skipped, rldb_paths = collect_required_prefixes(df)
    print(
        f"Found {sum(len(v) for v in required_by_bucket.values())} prefixes to check."
    )
    if skipped:
        print(f"Skipped {skipped} rows without a valid processed_path.")
    report_rldb_paths(rldb_paths)

    missing_rows = []
    orphans = []
    v2_bucket = args.processed_v2_bucket
    v2_prefix = args.processed_v2_prefix

    if args.check_orphans:
        v2_prefix, orphans = check_orphans(required_by_bucket, v2_bucket, v2_prefix)
    else:
        missing_rows = check_missing_prefixes(required_by_bucket)

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        if args.check_orphans:
            write_orphans_csv(args.csv_dir, v2_bucket, orphans)
        else:
            write_missing_csv(args.csv_dir, missing_rows)

    if args.clear_missing:
        if args.check_orphans:
            raise ValueError("--clear-missing cannot be used with --check-orphans.")
        updates = build_missing_updates(df, missing_rows)
        print(f"Clearing processed_path for {len(updates)} rows.")
        if args.dry_run:
            for update_row in updates[:20]:
                print(
                    "[DRY RUN] clear processed_path for episode_hash "
                    f"{update_row.get('b_episode_hash')}"
                )
        else:
            clear_missing_processed_paths(engine, updates)


if __name__ == "__main__":
    main()
