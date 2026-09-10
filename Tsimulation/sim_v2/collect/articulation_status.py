"""Count durable verified shards and validate their completion manifests."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

EMBODIMENTS = (
    "u_socket",
    "gripper",
    "chain_gripper",
    "suction",
    "umi",
    "triangle",
    "scoop",
    "flipper",
    "spring",
)
GAPS = ("ideal", "tight", "loose", "laggy", "sticky", "jittery")


def client():
    import boto3

    if os.environ.get("R2_ACCESS_KEY_ID"):
        return boto3.client(
            "s3",
            endpoint_url=os.environ.get(
                "R2_ENDPOINT_URL", os.environ.get("AWS_ENDPOINT_URL_S3")
            ),
            region_name="auto",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            aws_session_token="",
        )
    from .articulation_batch import r2_client

    return r2_client()


def inspect_store(c, prefix, workers=16):
    prefix = prefix.rstrip("/") + "/"
    entries = []
    for page in c.get_paginator("list_objects_v2").paginate(
        Bucket="rldb", Prefix=prefix
    ):
        entries.extend(page.get("Contents", []))
    objects = {entry["Key"]: entry for entry in entries}

    def read_shard(key):
        result = json.loads(c.get_object(Bucket="rldb", Key=key)["Body"].read())
        archive = objects.get(key[:-5])
        if not archive or archive["Size"] != result["archive_bytes"]:
            raise RuntimeError(f"Missing or truncated archive: {key}")
        head = c.head_object(Bucket="rldb", Key=key[:-5])
        if head["Metadata"].get("sha256") != result["archive_sha256"]:
            raise RuntimeError(f"Archive checksum metadata mismatch: {key}")
        if result["archive"] != "s3://rldb/" + key[:-5]:
            raise RuntimeError(f"Archive address mismatch: {key}")
        if result["embodiment"] not in EMBODIMENTS or result["gap"] not in GAPS:
            raise RuntimeError(f"Unexpected collection cell: {key}")
        if not result["complete"] or result["kept"] != 125 or result["target"] != 125:
            raise RuntimeError(f"Incomplete shard published: {key}")
        return result

    keys = sorted(k for k in objects if k.endswith(".tar.json"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shards = list(pool.map(read_shard, keys))
    counts = defaultdict(int)
    seed_ranges = []
    for result in shards:
        counts[(result["embodiment"], result["gap"])] += result["kept"]
        seed_ranges.append((result["seed0"], result["seed0"] + result["max_attempts"]))
    seed_ranges.sort()
    if any(right[0] < left[1] for left, right in zip(seed_ranges, seed_ranges[1:])):
        raise RuntimeError("Collection shards have overlapping reset seed ranges")
    rows = [
        dict(
            embodiment=e,
            gap=g,
            episodes=counts[(e, g)],
            target=3000,
            complete=counts[(e, g)] == 3000,
        )
        for e in EMBODIMENTS
        for g in GAPS
    ]
    batches = {}
    for emb in EMBODIMENTS:
        key = prefix + emb + "/batch_result.json"
        if key in objects:
            result = json.loads(c.get_object(Bucket="rldb", Key=key)["Body"].read())
            batches[emb] = {k: v for k, v in result.items() if k != "shards"}
            if result["complete"] and (
                result["episodes"] != 18000 or len(result["shards"]) != 144
            ):
                raise RuntimeError(f"Incorrect batch completion manifest: {key}")
    complete = (
        all(r["complete"] for r in rows)
        and len(batches) == 9
        and all(b["complete"] for b in batches.values())
    )
    return dict(
        prefix="s3://rldb/" + prefix,
        durable_episodes=sum(counts.values()),
        durable_shards=len(shards),
        durable_bytes=sum(s["archive_bytes"] for s in shards),
        checked_at=datetime.now(timezone.utc).isoformat(),
        cells=rows,
        batches=batches,
        expected_cells=54,
        expected_episodes=162000,
        complete=complete,
        shards=shards,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prefix", default="staged/pushshapes_articulated/articulated-20260909/"
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    summary = inspect_store(client(), a.prefix, a.workers)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
