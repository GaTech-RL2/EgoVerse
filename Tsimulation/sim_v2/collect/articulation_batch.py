"""Bounded process pool for independent, non-overlapping collection shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from ..pushshapes.agents import CONTROL_GAPS
from .articulation_collect import atomic_json, collect
from .articulation_quality import ARTICULATED


def task(args):
    return collect(**args)


def r2_client():
    import boto3

    secret = json.loads(
        boto3.client("secretsmanager", region_name="us-east-2").get_secret_value(
            SecretId="r2/rldb/credentials"
        )["SecretString"]
    )
    return boto3.client(
        "s3",
        endpoint_url=secret["endpoint_url"],
        region_name="auto",
        aws_access_key_id=secret["access_key_id"],
        aws_secret_access_key=secret["secret_access_key"],
        # This secret's session_token is not an AWS STS token.
        # The established R2 path authenticates with the key pair.
        aws_session_token="",
    )


def upload_shard(client, root, out, prefix, result):
    relative = out.relative_to(root)
    archives = root / "_archives"
    archives.mkdir(exist_ok=True)
    archive = archives / ("_".join(relative.parts) + ".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(out, arcname=str(relative), recursive=True)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    key = prefix.rstrip("/") + "/" + str(relative) + ".tar"
    client.upload_file(
        str(archive), "rldb", key, ExtraArgs={"Metadata": {"sha256": digest}}
    )
    head = client.head_object(Bucket="rldb", Key=key)
    if (
        head["ContentLength"] != archive.stat().st_size
        or head["Metadata"]["sha256"] != digest
    ):
        raise RuntimeError(f"Uploaded shard verification failed: {key}")
    info = dict(
        result,
        archive=f"s3://rldb/{key}",
        archive_sha256=digest,
        archive_bytes=archive.stat().st_size,
    )
    client.put_object(Bucket="rldb", Key=key + ".json", Body=json.dumps(info).encode())
    archive.unlink()
    return info


def finalize_partitions(client, prefix, shards=24, partition_size=8):
    """Publish the full batch only after all disjoint partitions are durable."""
    parts = []
    for start in range(0, shards, partition_size):
        key = (
            prefix.rstrip("/")
            + f"/batch_result_s{start:03d}_s{min(start+partition_size,shards):03d}.json"
        )
        try:
            parts.append(
                json.loads(client.get_object(Bucket="rldb", Key=key)["Body"].read())
            )
        except client.exceptions.NoSuchKey:
            return False
    records = [record for part in parts for record in part["shards"]]
    if not all(part["complete"] for part in parts):
        return False
    assert len(records) == 144 and sum(r["kept"] for r in records) == 18000
    assert len({r["archive"] for r in records}) == 144
    assert len({r["seed0"] for r in records}) == 144
    for gap in CONTROL_GAPS:
        assert sum(r["kept"] for r in records if r["gap"] == gap) == 3000
    for record in records:
        head = client.head_object(
            Bucket="rldb", Key=record["archive"].removeprefix("s3://rldb/")
        )
        assert head["ContentLength"] == record["archive_bytes"]
        assert head["Metadata"]["sha256"] == record["archive_sha256"]
    result = dict(
        complete=True,
        episodes=18000,
        shards=records,
        partitioned=True,
        source_capsules=[p.get("source_capsule_sha256") for p in parts],
    )
    client.put_object(
        Bucket="rldb",
        Key=prefix.rstrip("/") + "/batch_result.json",
        Body=json.dumps(result, indent=2).encode(),
    )
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--embodiments",
        nargs="+",
        default=["u_socket", "gripper", "chain_gripper", "umi"],
    )
    ap.add_argument("--gaps", nargs="+", default=list(CONTROL_GAPS))
    ap.add_argument("--target-per-cell", type=int, default=3000)
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-stop", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=2000000000)
    ap.add_argument("--max-attempts", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument(
        "--s3-prefix", help="Persist verified tar shards in this new rldb prefix"
    )
    a = ap.parse_args()
    if a.shard_stop is None:
        a.shard_stop = a.shards
    if not 0 <= a.shard_start < a.shard_stop <= a.shards:
        ap.error("invalid shard interval")
    partitioned = a.shard_start != 0 or a.shard_stop != a.shards
    if a.target_per_cell % a.shards:
        ap.error("target-per-cell must be divisible by shards")
    if a.max_attempts > 1000000:
        ap.error("attempts exceed a shard's disjoint seed range")
    tasks = []
    for emb in a.embodiments:
        ei = list(ARTICULATED).index(emb)
        for gap in a.gaps:
            gi = list(CONTROL_GAPS).index(gap)
            for shard in range(a.shard_start, a.shard_stop):
                tasks.append(
                    dict(
                        out=a.out / gap / emb / f"shard{shard:03d}",
                        emb=emb,
                        gap=gap,
                        target=a.target_per_cell // a.shards,
                        seed0=a.seed0
                        + ei * 1000000000
                        + gi * 100000000
                        + shard * 1000000,
                        max_attempts=a.max_attempts,
                        max_steps=a.max_steps,
                    )
                )
    # Alternate cells to expose failures in every combination early.
    tasks.sort(key=lambda t: (str(t["out"]).rsplit("/", 1)[-1], t["emb"], t["gap"]))
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        a.out / "batch.json",
        dict(
            target_per_cell=a.target_per_cell,
            embodiments=a.embodiments,
            gaps=a.gaps,
            shards=a.shards,
            workers=a.workers,
            shard_start=a.shard_start,
            shard_stop=a.shard_stop,
            expected_episodes=sum(t["target"] for t in tasks),
        ),
    )
    results = []
    client = r2_client() if a.s3_prefix else None
    uploads = []
    with (
        ThreadPoolExecutor(max_workers=4) as upload_pool,
        ProcessPoolExecutor(max_workers=a.workers) as pool,
    ):
        futures = {pool.submit(task, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if client:
                uploads.append(
                    upload_pool.submit(
                        upload_shard,
                        client,
                        a.out,
                        futures[future]["out"],
                        a.s3_prefix,
                        result,
                    )
                )
            print("SHARD_RESULT " + json.dumps(result), flush=True)
            atomic_json(
                a.out / "batch_progress.json",
                dict(
                    shards_finished=len(results),
                    shards_expected=len(tasks),
                    episodes=sum(r["kept"] for r in results),
                    incomplete_shards=sum(not r["complete"] for r in results),
                ),
            )
        if client:
            results = [future.result() for future in uploads]
    complete = all(r["complete"] for r in results)
    atomic_json(
        a.out / "batch_result.json",
        dict(
            complete=complete,
            shards=results,
            episodes=sum(r["kept"] for r in results),
            shard_start=a.shard_start,
            shard_stop=a.shard_stop,
            source_capsule_sha256=os.environ.get("ARTICULATED_SOURCE_SHA256"),
        ),
    )
    if client:
        name = (
            f"batch_result_s{a.shard_start:03d}_s{a.shard_stop:03d}.json"
            if partitioned
            else "batch_result.json"
        )
        client.upload_file(
            str(a.out / "batch_result.json"),
            "rldb",
            a.s3_prefix.rstrip("/") + "/" + name,
        )
        if partitioned and len(a.embodiments) == 1 and len(a.gaps) == 6:
            finalize_partitions(client, a.s3_prefix, a.shards)
    raise SystemExit(0 if complete else 2)


if __name__ == "__main__":
    main()
