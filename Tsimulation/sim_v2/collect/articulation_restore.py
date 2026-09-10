"""Restore committed checkpoint episodes for one disjoint shard interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from .articulation_batch import r2_client


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-start", type=int, required=True)
    ap.add_argument("--shard-stop", type=int, required=True)
    a = ap.parse_args()
    c = r2_client()
    manifest = json.loads(
        c.get_object(
            Bucket="rldb", Key=a.prefix.rstrip("/") + "/checkpoint_complete.json"
        )["Body"].read()
    )
    assert manifest["complete"]
    a.out.mkdir(parents=True, exist_ok=True)
    selected = [
        r
        for r in manifest["shards"]
        if a.shard_start <= int(Path(r["relative"]).name[5:]) < a.shard_stop
    ]

    def restore(record):
        relative = Path(record["relative"])
        final = a.out / relative
        assert not final.exists(), f"Checkpoint destination already exists: {final}"
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=a.out) as temporary:
            archive = Path(temporary) / "shard.tar"
            c.download_file("rldb", record["key"], str(archive))
            with archive.open("rb") as f:
                digest = hashlib.file_digest(f, "sha256").hexdigest()
            assert (
                digest == record["sha256"] and archive.stat().st_size == record["bytes"]
            )
            extracted = Path(temporary) / "extracted"
            with tarfile.open(archive, "r:") as tar:
                for member in tar.getmembers():
                    path = PurePosixPath(member.name)
                    assert (
                        not path.is_absolute()
                        and ".." not in path.parts
                        and path.parts[:3] == relative.parts
                    )
                    assert member.isdir() or member.isfile()
                tar.extractall(extracted, filter="data")
            folder = extracted / relative
            assert len(list(folder.glob("episode_*.zarr"))) == record["episodes"]
            final.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(final)
        return record["episodes"]

    with ThreadPoolExecutor(max_workers=6) as pool:
        episodes = sum(pool.map(restore, selected))
    receipt = dict(
        checkpoint_prefix=a.prefix,
        preserved_episodes=episodes,
        restored_shards=len(selected),
        shard_start=a.shard_start,
        shard_stop=a.shard_stop,
    )
    (a.out / "restored_checkpoint.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print("RESTORED_CHECKPOINT " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
