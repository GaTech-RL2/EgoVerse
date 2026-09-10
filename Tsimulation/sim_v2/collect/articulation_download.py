"""Download checksum-verified articulated Zarr shards without mixing datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from .articulation_status import EMBODIMENTS, GAPS, client, inspect_store


def flat_view(output, directory):
    """Unique episode names in the flat layout expected by LocalEpisodeResolver."""
    gap, emb, shard = directory.relative_to(output).parts
    view = output / "cells" / gap / emb
    view.mkdir(parents=True, exist_ok=True)
    for episode in directory.glob("episode_*.zarr"):
        link = view / (gap + "_" + shard + "_" + episode.name)
        target = os.path.relpath(episode, view)
        if link.is_symlink() and os.readlink(link) == target:
            continue
        link.symlink_to(target, target_is_directory=True)


def download_shard(c, output, result):
    key = result["archive"].removeprefix("s3://rldb/")
    relative = Path(*PurePosixPath(key).parts[-3:]).with_suffix("")
    if relative.parts[:2] != (result["gap"], result["embodiment"]):
        raise ValueError("Archive path does not match its cell")
    final = output / relative
    if final.exists():
        receipt = final / "download.json"
        if (
            receipt.exists()
            and json.loads(receipt.read_text())["archive_sha256"]
            == result["archive_sha256"]
        ):
            flat_view(output, final)
            return dict(directory=str(final), episodes=result["kept"], cached=True)
        raise FileExistsError(f"Refusing to overwrite an existing shard: {final}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=output) as temporary:
        staging = Path(temporary)
        archive = staging / "shard.tar"
        c.download_file("rldb", key, str(archive))
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (
            digest != result["archive_sha256"]
            or archive.stat().st_size != result["archive_bytes"]
        ):
            raise RuntimeError(f"Archive checksum/size verification failed: {key}")
        extracted = staging / "extracted"
        with tarfile.open(archive, "r:") as tar:
            for member in tar.getmembers():
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.parts[:3] != relative.parts
                ):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                if (
                    member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError(f"Unexpected archive member type: {member.name}")
            tar.extractall(extracted, filter="data")
        shard = extracted / relative
        episodes = sorted(shard.glob("episode_*.zarr"))
        seeds = set()
        for episode in episodes:
            attrs = json.loads((episode / "zarr.json").read_text())["attributes"]
            if (
                attrs["collector"] != "articulated-v1"
                or attrs["control_gap"] != result["gap"]
            ):
                raise ValueError(f"Unexpected episode metadata: {episode}")
            seeds.add(attrs["reset_seed"])
        if len(episodes) != result["kept"] or len(seeds) != len(episodes):
            raise ValueError(f"Episode count/seed uniqueness failed: {key}")
        (shard / "download.json").write_text(json.dumps(result, indent=2) + "\n")
        final.parent.mkdir(parents=True, exist_ok=True)
        shard.rename(final)
    flat_view(output, final)
    return dict(directory=str(final), episodes=len(episodes), cached=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--snapshot", type=Path, help="Previously verified articulation_status JSON"
    )
    ap.add_argument(
        "--prefix", default="staged/pushshapes_articulated/articulated-20260909/"
    )
    ap.add_argument(
        "--embodiments", nargs="+", choices=EMBODIMENTS, default=EMBODIMENTS
    )
    ap.add_argument("--gaps", nargs="+", choices=GAPS, default=GAPS)
    ap.add_argument(
        "--shards-per-cell", type=int, help="Omit to download every available shard"
    )
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    c = client()
    summary = (
        json.loads(a.snapshot.read_text()) if a.snapshot else inspect_store(c, a.prefix)
    )
    selected = []
    for emb in a.embodiments:
        for gap in a.gaps:
            matches = sorted(
                (
                    s
                    for s in summary["shards"]
                    if s["embodiment"] == emb and s["gap"] == gap
                ),
                key=lambda s: s["seed0"],
            )
            selected.extend(
                matches[: a.shards_per_cell] if a.shards_per_cell else matches
            )
    print(
        json.dumps(
            dict(
                selected_shards=len(selected),
                episodes=sum(s["kept"] for s in selected),
                archive_bytes=sum(s["archive_bytes"] for s in selected),
            )
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(lambda s: download_shard(c, a.out, s), selected):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
