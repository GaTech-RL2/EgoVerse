from __future__ import annotations

import os
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig


def s3_sync_to_local(bucket: str, key_prefix: str, local_dir: str | Path) -> None:
    """
    Rough equivalent of: aws s3 sync s3://bucket/key_prefix/ local_dir/
    Downloads all objects under key_prefix into local_dir, preserving subpaths.
    Skips download if local file exists and size matches S3 object's size.
    """
    key_prefix = key_prefix.lstrip("/")
    if key_prefix and not key_prefix.endswith("/"):
        key_prefix += "/"

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create the client inside the function so this works cleanly on Ray workers.
    s3 = boto3.client("s3")

    config = TransferConfig(
        max_concurrency=16,
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        use_threads=True,
    )

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(key_prefix) :] if key_prefix else key
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            if dest.exists() and dest.stat().st_size == obj["Size"]:
                continue

            s3.download_file(bucket, key, str(dest), Config=config)


def upload_dir_to_s3(
    local_dir: str, bucket: str, prefix: str = "", concurrency: int = 32
):
    s3 = boto3.client("s3")
    cfg = TransferConfig(
        max_concurrency=concurrency,
        multipart_threshold=64 * 1024 * 1024,  # 64MB
        multipart_chunksize=64 * 1024 * 1024,  # 64MB
        use_threads=True,
    )

    local_dir = Path(local_dir).resolve()
    prefix = prefix.strip("/")

    for root, _, files in os.walk(local_dir):
        rootp = Path(root)
        for f in files:
            lp = rootp / f
            rel = lp.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            s3.upload_file(str(lp), bucket, key, Config=cfg)
