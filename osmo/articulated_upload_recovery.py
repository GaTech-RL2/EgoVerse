"""Persist already-running collection jobs whose uploader used a bad token.

Run only inside this task's dedicated OSMO containers. Collection and physics
continue unchanged. Once the final wave starts, pause the batch coordinator
so a deferred upload exception cannot terminate the container before all
completed shards are safely in R2. Worker processes finish their episodes.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/workspace/EgoVerse")


def upload_shard(client, root, out, prefix, result):
    relative = out.relative_to(root)
    archives = root / "_recovery_archives"
    archives.mkdir(exist_ok=True)
    archive = archives / ("_".join(relative.parts) + ".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(out, arcname=str(relative), recursive=True)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    key = prefix.rstrip("/") + "/" + str(relative) + ".tar"
    client.upload_file(
        str(archive), "rldb", key, ExtraArgs={"Metadata": {"sha256": digest}}
    )
    head = client.head_object(Bucket="rldb", Key=key)
    if (
        head["ContentLength"] != archive.stat().st_size
        or head["Metadata"]["sha256"] != digest
    ):
        raise RuntimeError("Uploaded shard verification failed: " + key)
    info = dict(
        result,
        archive="s3://rldb/" + key,
        archive_sha256=digest,
        archive_bytes=archive.stat().st_size,
    )
    client.put_object(Bucket="rldb", Key=key + ".json", Body=json.dumps(info).encode())
    archive.unlink()
    return info


def coordinator():
    processes = {}
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().split(b"\0")
            if b"Tsimulation.sim_v2.collect.articulation_batch" not in argv:
                continue
            pid = int(path.parent.name)
            status = (path.parent / "status").read_text()
            parent = int(
                next(
                    line.split()[1]
                    for line in status.splitlines()
                    if line.startswith("PPid:")
                )
            )
            processes[pid] = (parent, argv)
        except (OSError, ValueError, StopIteration):
            pass
    roots = [
        (pid, argv)
        for pid, (parent, argv) in processes.items()
        if parent not in processes
    ]
    if len(roots) != 1:
        return None
    pid, argv = roots[0]
    marker = argv.index(b"--s3-prefix")
    return pid, argv[marker + 1].decode()


def main():
    import boto3

    root = Path("/workspace/demos")
    while coordinator() is None or not (root / "batch.json").exists():
        time.sleep(2)
    pid, prefix = coordinator()
    secret = json.loads(
        boto3.client("secretsmanager", region_name="us-east-2").get_secret_value(
            SecretId="r2/rldb/credentials"
        )["SecretString"]
    )
    client = boto3.client(
        "s3",
        endpoint_url=secret["endpoint_url"],
        region_name="auto",
        aws_access_key_id=secret["access_key_id"],
        aws_secret_access_key=secret["secret_access_key"],
        aws_session_token="",
    )
    client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1)
    config = json.loads((root / "batch.json").read_text())
    expected = config["shards"] * len(config["gaps"]) * len(config["embodiments"])
    print("RECOVERY_STARTED", pid, prefix, expected, flush=True)
    completed = {}
    pending = {}
    paused = False
    progress_file = root / "upload_recovery.json"
    # A retry after disconnect can reuse the verified sidecar manifests.
    with ThreadPoolExecutor(max_workers=6) as pool:
        while len(completed) < expected:
            started = list(root.glob("*/*/shard*/collection.json"))
            if len(started) == expected and not paused:
                os.kill(pid, signal.SIGSTOP)
                paused = True
                print("COORDINATOR_PAUSED_FOR_FINAL_FLUSH", pid, flush=True)
            for progress in root.glob("*/*/shard*/progress.json"):
                key = str(progress.parent)
                if key in completed or key in pending:
                    continue
                result = json.loads(progress.read_text())
                if not result["complete"]:
                    continue
                pending[key] = pool.submit(
                    upload_shard, client, root, progress.parent, prefix, result
                )
            for key, future in list(pending.items()):
                if not future.done():
                    continue
                del pending[key]
                try:
                    completed[key] = future.result()
                    print("RECOVERED_SHARD", len(completed), expected, key, flush=True)
                except Exception as exc:
                    print("UPLOAD_RETRY", key, type(exc).__name__, str(exc), flush=True)
            progress_file.write_text(
                json.dumps(
                    dict(
                        parent_pid=pid,
                        parent_paused=paused,
                        uploaded_shards=len(completed),
                        expected_shards=expected,
                        durable_episodes=sum(r["kept"] for r in completed.values()),
                    )
                )
            )
            time.sleep(2)
    episodes = sum(r["kept"] for r in completed.values())
    assert episodes == config["expected_episodes"], (episodes, config)
    result = dict(
        complete=True,
        episodes=episodes,
        shards=list(completed.values()),
        upload_recovered=True,
        source_capsule_sha256=hashlib.file_digest(
            open("/workspace/source.tar.gz", "rb"), "sha256"
        ).hexdigest(),
    )
    (root / "batch_result.json").write_text(json.dumps(result, indent=2) + "\n")
    client.upload_file("/workspace/source.tar.gz", "rldb", prefix + "/source.tar.gz")
    client.upload_file(__file__, "rldb", prefix + "/upload_recovery.py")
    for path in Path("/workspace/preflight").glob("*.json"):
        client.upload_file(str(path), "rldb", prefix + "/preflight/" + path.name)
    # Publish completion last, after every object and checksum has been verified.
    client.upload_file(
        str(root / "batch_result.json"), "rldb", prefix + "/batch_result.json"
    )
    print("ALL_EPISODES_DURABLE", episodes, prefix, flush=True)
    if paused:
        os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
