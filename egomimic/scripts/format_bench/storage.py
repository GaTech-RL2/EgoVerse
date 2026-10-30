"""Bytes on disk per format for the 48 pilot recordings, split into numeric / video-image /
metadata-index, total and per frame. One JSON line per format variant.

zarr counts its trimmed episodes (manifest num_frames); Lance and LeRobot hold the full
recordings, so per-frame numbers are the comparable ones. Lance video is reported both as the
480p MP4s (local loader) and as the 480p 8-frame chunk table (object-storage loader).
"""

import argparse
import csv
from pathlib import Path

import lance

from egomimic.scripts.format_bench.common import emit


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(
        f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()
    )


def lance_split(table: Path) -> tuple[int, int]:
    """(data file bytes, manifest/version/transaction bytes) of one Lance table."""
    data = size(table / "data") if (table / "data").exists() else 0
    return data, size(table) - data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pilot", type=Path, default=Path("/coc/flash7/scratch/lance_pilot")
    )
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument(
        "--lerobot",
        type=Path,
        nargs="+",
        required=True,
        help="one or more LeRobot roots",
    )
    ap.add_argument("--camera", default="front_1_480")
    ap.add_argument("--out")
    args = ap.parse_args()

    manifest = list(csv.DictReader(open(args.pilot / "manifest.csv")))
    zarr_frames = sum(int(m["num_frames"]) for m in manifest)
    num = img = meta = 0
    for m in manifest:
        store = args.pilot / "zarr" / m["episode_hash"]
        for child in store.iterdir():
            if child.name == "zarr.json":
                meta += size(child)
            elif child.name == "images.front_1":
                img += size(child)
            else:
                num += size(child)
    rows = [
        dict(
            format="zarr",
            variant="jpeg",
            frames=zarr_frames,
            numeric=num,
            video=img,
            metadata=meta,
        )
    ]

    root = args.pilot / "lance"
    rec = lance.dataset(str(root / "recordings.lance")).to_table().to_pylist()
    lance_frames = sum(r["num_frames"] for r in rec)
    s_data, s_meta = lance_split(root / "steps/aria.lance")
    meta = s_meta + sum(
        size(root / t) for t in ("recordings.lance", "aliases.lance", "tasks.lance")
    )
    mp4 = sum(size(root / dict(r["video_path"])[args.camera]) for r in rec)
    c_data, c_meta = lance_split(args.chunks)
    rows.append(
        dict(
            format="lance",
            variant="mp4",
            frames=lance_frames,
            numeric=s_data,
            video=mp4,
            metadata=meta,
        )
    )
    rows.append(
        dict(
            format="lance",
            variant="k8_chunks",
            frames=lance_frames,
            numeric=s_data,
            video=c_data,
            metadata=meta + c_meta,
        )
    )

    import json

    for lr in args.lerobot:
        info = json.loads((lr / "meta/info.json").read_text())
        rows.append(
            dict(
                format="lerobot",
                variant="packed" if "action" in info["features"] else "unpacked",
                frames=info["total_frames"],
                numeric=size(lr / "data"),
                video=size(lr / "videos"),
                metadata=size(lr / "meta"),
            )
        )
    for r in rows:
        r["total"] = r["numeric"] + r["video"] + r["metadata"]
        for k in ("numeric", "video", "metadata", "total"):
            r[f"{k}_mb"] = round(r[k] / 1e6, 1)
            r[f"{k}_bytes_per_frame"] = round(r[k] / r["frames"], 1)
        emit(
            dict(
                label="storage",
                **{
                    k: v
                    for k, v in r.items()
                    if k not in ("numeric", "video", "metadata", "total")
                },
            ),
            args.out,
        )


if __name__ == "__main__":
    main()
