"""Checkpoint owned slow jobs before moving their work to disjoint partitions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/workspace/EgoVerse")


def main():
    from Tsimulation.sim_v2.collect.articulation_batch import r2_client

    root = Path("/workspace/demos")
    marker = root / "checkpoint_complete.json"
    assert not marker.exists(), "Already checkpointed"
    config = json.loads((root / "batch.json").read_text())
    assert len(config["embodiments"]) == 1
    emb = config["embodiments"][0]
    assert emb in ("scoop", "spring", "flipper")
    prefix = "staged/pushshapes_articulated_checkpoints/articulated-20260909/" + emb
    client = r2_client()
    client.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=1)
    processes = {}
    helpers = []
    for f in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = f.read_bytes().split(b"\0")
            if (
                b"/workspace/articulated_upload_recovery.py" in args
                or b"/workspace/articulated_resume.py" in args
            ):
                helpers.append(int(f.parent.name))
            if b"Tsimulation.sim_v2.collect.articulation_batch" not in args:
                continue
            status = (f.parent / "status").read_text()
            parent = int(
                next(x.split()[1] for x in status.splitlines() if x.startswith("PPid:"))
            )
            processes[int(f.parent.name)] = parent
        except (OSError, ValueError, StopIteration):
            pass
    coordinators = [pid for pid, parent in processes.items() if parent not in processes]
    assert coordinators, "No owned coordinator found"
    for pid in coordinators:
        os.kill(pid, signal.SIGSTOP)
    for pid in helpers:
        os.kill(pid, signal.SIGTERM)
    for pid in processes:
        if pid not in coordinators:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)
    for pid in processes:
        if pid not in coordinators:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(1)
    sources = []
    for path in (
        Path("/workspace/source.tar.gz"),
        Path("/workspace/source_after_resume.tar.gz"),
    ):
        if path.exists():
            with path.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            key = prefix + "/sources/" + digest + ".tar.gz"
            client.upload_file(str(path), "rldb", key)
            sources.append(dict(sha256=digest, key=key))
    archive_dir = root / "_checkpoints"
    archive_dir.mkdir(exist_ok=True)

    def upload(folder):
        for path in folder.glob(".pending-*.zarr"):
            shutil.rmtree(path)
        episodes = sorted(folder.glob("episode_*.zarr"))
        preserved = []
        for path in episodes:
            attrs = json.loads((path / "zarr.json").read_text())["attributes"]
            assert (
                attrs["quality"]["success"] and attrs["collector"] == "articulated-v1"
            )
            preserved.append(
                dict(
                    episode=path.name,
                    seed=attrs["reset_seed"],
                    actions_sha256=attrs["actions_sha256"],
                )
            )
        relative = folder.relative_to(root)
        (folder / "checkpoint_source.json").write_text(
            json.dumps(dict(sources=sources, episodes=preserved), indent=2) + "\n"
        )
        archive = archive_dir / ("_".join(relative.parts) + ".tar")
        with tarfile.open(archive, "w") as tar:
            tar.add(folder, arcname=str(relative))
        with archive.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        key = prefix + "/" + str(relative) + ".tar"
        client.upload_file(
            str(archive), "rldb", key, ExtraArgs={"Metadata": {"sha256": digest}}
        )
        head = client.head_object(Bucket="rldb", Key=key)
        assert (
            head["ContentLength"] == archive.stat().st_size
            and head["Metadata"]["sha256"] == digest
        )
        record = dict(
            relative=str(relative),
            key=key,
            sha256=digest,
            bytes=archive.stat().st_size,
            episodes=len(episodes),
        )
        archive.unlink()
        return record

    folders = sorted(path for path in root.glob("*/*/shard*") if path.is_dir())
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(upload, folders))
    result = dict(
        complete=True,
        embodiment=emb,
        prefix=prefix,
        created_at=time.time(),
        episodes=sum(r["episodes"] for r in records),
        shards=records,
        sources=sources,
        paused_coordinators=coordinators,
    )
    marker.write_text(json.dumps(result, indent=2) + "\n")
    client.upload_file(str(marker), "rldb", prefix + "/checkpoint_complete.json")
    print("CHECKPOINT_DURABLE", emb, result["episodes"], prefix, flush=True)
    # Leave the dedicated container paused until replacement jobs restore it.


if __name__ == "__main__":
    main()
