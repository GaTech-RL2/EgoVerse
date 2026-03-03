#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from egomimic.scripts.aria_process.aria_helper import zarr_job
from egomimic.utils.aws.aws_data_utils import (
    get_boto3_s3_client,
    load_env,
    s3_sync_to_local,
)
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_hash_to_timestamp_ms,
    episode_table_to_df,
)

RAW_REMOTE_PREFIX = os.environ.get(
    "RAW_REMOTE_PREFIX", "s3://rldb/raw_v2/aria/"
).rstrip("/")
BUCKET = os.environ.get("BUCKET", "rldb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query the SQL episode table, filter rows, download matching Aria bundles "
            "from S3/R2, and convert them into local zarr datasets."
        )
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        required=True,
        help="Local root directory for downloaded raw bundles and converted zarrs.",
    )
    parser.add_argument(
        "--raw-remote-prefix",
        type=str,
        default=RAW_REMOTE_PREFIX,
        help="Remote raw Aria prefix, e.g. s3://rldb/raw_v2/aria",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=BUCKET,
        help="Default bucket to use when --raw-remote-prefix is not a full s3:// URI.",
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Exact-match dataframe filter. Repeat as needed.",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        metavar="COLUMN=TEXT",
        help="Substring dataframe filter. Repeat as needed.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Optional pandas query string applied after the simple filters.",
    )
    parser.add_argument(
        "--episode-hash",
        action="append",
        default=[],
        help="Explicit episode_hash values to keep. Repeat as needed.",
    )
    parser.add_argument(
        "--episode-hash-file",
        type=Path,
        help="Optional text file containing one episode_hash per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of filtered rows to process.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="episode_hash",
        help="Column used to sort the filtered dataframe.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending instead of ascending.",
    )
    parser.add_argument(
        "--arm-override",
        choices=["left", "right", "bimanual"],
        help="Override arm inference from robot_name.",
    )
    parser.add_argument(
        "--print-df",
        action="store_true",
        help="Print the filtered dataframe preview before processing.",
    )
    parser.add_argument(
        "--save-df",
        type=Path,
        help="Optional CSV path for the filtered dataframe.",
    )
    parser.add_argument(
        "--skip-download-existing",
        action="store_true",
        help="Skip downloading a bundle if the expected local files already exist.",
    )
    parser.add_argument(
        "--skip-convert-existing",
        action="store_true",
        help="Skip conversion if the target zarr already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show matches and planned paths; do not download or convert.",
    )
    parser.add_argument(
        "--no-mp4",
        action="store_true",
        help="Skip mp4 preview generation during zarr conversion.",
    )
    parser.add_argument(
        "--remove-downloaded-raw",
        action="store_true",
        help="Remove the downloaded local raw bundle after a successful zarr conversion.",
    )
    return parser.parse_args()


def parse_s3_uri(uri: str, *, default_bucket: str) -> tuple[str, str]:
    uri = uri.strip()
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        return bucket, prefix.strip("/")
    return default_bucket, uri.strip("/")


def parse_assignment(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected COLUMN=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Missing column name in: {raw}")
    return key, value.strip()


def maybe_coerce_scalar(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_episode_hashes(args: argparse.Namespace) -> list[str]:
    hashes = list(args.episode_hash)
    if args.episode_hash_file:
        hashes.extend(
            line.strip()
            for line in args.episode_hash_file.read_text().splitlines()
            if line.strip()
        )
    return hashes


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()

    for raw in args.where:
        column, value = parse_assignment(raw)
        if column not in filtered.columns:
            raise KeyError(f"Unknown column for --where: {column}")
        filtered = filtered[filtered[column] == maybe_coerce_scalar(value)]

    for raw in args.contains:
        column, value = parse_assignment(raw)
        if column not in filtered.columns:
            raise KeyError(f"Unknown column for --contains: {column}")
        filtered = filtered[
            filtered[column].fillna("").astype(str).str.contains(value, regex=False)
        ]

    episode_hashes = read_episode_hashes(args)
    if episode_hashes:
        filtered = filtered[filtered["episode_hash"].isin(episode_hashes)]

    if args.query:
        filtered = filtered.query(args.query)

    if args.sort_by:
        if args.sort_by not in filtered.columns:
            raise KeyError(f"Unknown column for --sort-by: {args.sort_by}")
        filtered = filtered.sort_values(args.sort_by, ascending=not args.descending)

    return filtered.reset_index(drop=True)


def infer_arm(robot_name: str) -> str:
    name = (robot_name or "").lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    return "bimanual"


def expected_bundle_paths(bundle_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        bundle_dir / f"{stem}.vrs",
        bundle_dir / f"{stem}.json",
        bundle_dir / f"mps_{stem}_vrs",
    )


def expected_eye_gaze_path(bundle_dir: Path, stem: str) -> Path:
    return bundle_dir / f"mps_{stem}_vrs" / "eye_gaze" / "general_eye_gaze.csv"


def bundle_exists(bundle_dir: Path, stem: str) -> bool:
    vrs_path, json_path, mps_dir = expected_bundle_paths(bundle_dir, stem)
    return (
        vrs_path.exists()
        and json_path.exists()
        and mps_dir.exists()
        and (mps_dir / "hand_tracking").exists()
        and (mps_dir / "slam").exists()
        and expected_eye_gaze_path(bundle_dir, stem).exists()
    )


def eye_gaze_exists_in_s3(
    *, s3_client, bucket: str, remote_prefix: str, stem: str
) -> bool:
    eye_gaze_key = (
        f"{remote_prefix}/mps_{stem}_vrs/eye_gaze/general_eye_gaze.csv".strip("/")
    )
    try:
        s3_client.head_object(Bucket=bucket, Key=eye_gaze_key)
        return True
    except Exception:
        return False


def download_bundle(
    *,
    s3_client,
    bucket: str,
    remote_prefix: str,
    bundle_dir: Path,
    stem: str,
) -> tuple[Path, Path, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = f"{remote_prefix}/{stem}".strip("/")
    vrs_key = f"{file_prefix}.vrs"
    json_key = f"{file_prefix}.json"
    mps_key = f"{remote_prefix}/mps_{stem}_vrs".strip("/")

    vrs_path, json_path, mps_dir = expected_bundle_paths(bundle_dir, stem)

    s3_client.download_file(bucket, vrs_key, str(vrs_path))
    s3_client.download_file(bucket, json_key, str(json_path))
    s3_sync_to_local(bucket, mps_key, mps_dir)

    return vrs_path, json_path, mps_dir


def dataframe_preview(df: pd.DataFrame) -> str:
    if df.empty:
        return "<empty dataframe>"
    preview = df[
        [
            c
            for c in ["episode_hash", "task", "operator", "robot_name"]
            if c in df.columns
        ]
    ]
    return preview.to_string(index=False)


def process_rows(df: pd.DataFrame, args: argparse.Namespace) -> list[dict]:
    bucket, remote_prefix = parse_s3_uri(
        args.raw_remote_prefix, default_bucket=args.bucket
    )
    raw_root = args.download_dir / "raw"
    zarr_root = args.download_dir / "zarr"
    summary: list[dict] = []
    s3_client = get_boto3_s3_client()
    if not args.dry_run:
        zarr_root.mkdir(parents=True, exist_ok=True)
    accepted_count = 0

    for row in df.itertuples(index=False):
        episode_hash = row.episode_hash
        stem = str(episode_hash_to_timestamp_ms(episode_hash))
        bundle_dir = raw_root / episode_hash
        zarr_path = zarr_root / f"{episode_hash}.zarr"
        mp4_path = zarr_root / f"{episode_hash}.mp4"
        arm = args.arm_override or infer_arm(getattr(row, "robot_name", ""))

        item = {
            "episode_hash": episode_hash,
            "timestamp_ms_stem": stem,
            "raw_bundle_dir": str(bundle_dir),
            "zarr_path": str(zarr_path),
            "mp4_path": str(mp4_path),
            "arm": arm,
            "status": "pending",
        }

        try:
            if not eye_gaze_exists_in_s3(
                s3_client=s3_client,
                bucket=bucket,
                remote_prefix=remote_prefix,
                stem=stem,
            ):
                item["status"] = "skipped_missing_eye_gaze"
                item["error"] = (
                    f"Missing mps_{stem}_vrs/eye_gaze/general_eye_gaze.csv in S3"
                )
                summary.append(item)
                continue

            reuse_existing_bundle = args.skip_download_existing and bundle_exists(
                bundle_dir, stem
            )

            if args.limit is not None and accepted_count >= args.limit:
                break

            if args.dry_run:
                item["status"] = (
                    "dry_run_reuse_existing_bundle"
                    if reuse_existing_bundle
                    else "dry_run_would_download"
                )
                accepted_count += 1
                summary.append(item)
                continue

            if not reuse_existing_bundle:
                download_bundle(
                    s3_client=s3_client,
                    bucket=bucket,
                    remote_prefix=remote_prefix,
                    bundle_dir=bundle_dir,
                    stem=stem,
                )
            accepted_count += 1

            if args.skip_convert_existing and zarr_path.exists():
                item["status"] = "skipped_existing_zarr"
                summary.append(item)
                continue

            zarr_out, mp4_out = zarr_job(
                raw_path=bundle_dir,
                output_dir=zarr_root,
                dataset_name=episode_hash,
                arm=arm,
                description=getattr(row, "task_description", "") or "",
                save_mp4=not args.no_mp4,
            )

            item["zarr_path"] = str(zarr_out)
            item["mp4_path"] = str(mp4_out) if mp4_out is not None else ""
            item["status"] = "converted"
            if args.remove_downloaded_raw:
                shutil.rmtree(bundle_dir, ignore_errors=True)
                item["raw_bundle_removed"] = True
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)

        summary.append(item)

    return summary


def main() -> None:
    load_env()
    args = parse_args()

    engine = create_default_engine()
    df = episode_table_to_df(engine)
    filtered = apply_filters(df, args)

    print(f"Matched {len(filtered)} episode rows.")
    if args.print_df or args.dry_run:
        print(dataframe_preview(filtered))

    if args.save_df:
        args.save_df.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(args.save_df, index=False)
        print(f"Saved filtered dataframe to {args.save_df}")

    episode_hashes = filtered["episode_hash"].tolist() if not filtered.empty else []
    print("Episode hashes:")
    print(json.dumps(episode_hashes, indent=2))

    summary = process_rows(filtered, args)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
