"""Convert raw Aria recordings (VRS + MPS) into the Lance tables, one Ray task per recording.

Workers run aria_to_lance.convert into `<deliveries>/<vendor id>/`; the driver is the only
Lance writer. As conversions finish it validates them (lance_pilot/commit.py), appends the
recordings' video chunk rows when --chunk-k is set, then commits them in batches. Reruns
skip recordings already delivered or committed, so a killed run resumes where it stopped.

    python -m egomimic.scripts.aria_process.aria_to_lance_ray \\
        --raw-dir /coc/flash7/scratch/lance_pilot/raw \\
        --deliveries /coc/flash7/scratch/lance_pilot/deliveries \\
        --lance /coc/flash7/scratch/lance_pilot/lance
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import lance
import pyarrow as pa
import ray
from lance.blob import blob_array

from egomimic.rldb.lance import commit
from egomimic.rldb.lance.schema import recording_id
from egomimic.scripts.lance_pilot.chunk_videos import SCHEMA as CHUNK_SCHEMA
from egomimic.scripts.lance_pilot.commit import load_deliveries, sql_rows
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash

log = logging.getLogger("aria_to_lance_ray")


def discover(raw_dir: Path) -> list[str]:
    """Vendor ids (VRS stems) under `raw_dir` that have both a VRS and an MPS folder."""
    return sorted(
        p.stem
        for p in raw_dir.glob("*.vrs")
        if (raw_dir / f"mps_{p.stem}_vrs").is_dir()
    )


def embodiments(vids: list[str]) -> dict[str, str]:
    sql = sql_rows([timestamp_ms_to_episode_hash(v) for v in vids])
    out = {}
    for v in vids:
        h = timestamp_ms_to_episode_hash(v)
        if h in sql.index and sql.loc[h].embodiment:
            out[v] = str(sql.loc[h].embodiment).lower()
    return out


@ray.remote(max_retries=1)
def convert_one(vid: str, embodiment: str, opts: dict) -> tuple[str, str | None]:
    """(vendor id, None) on success, else (vendor id, traceback)."""
    from egomimic.scripts.aria_process.aria_to_lance import convert

    logging.basicConfig(level=logging.INFO)
    try:
        convert(
            SimpleNamespace(
                vendor_recording_id=vid,
                recording_id=recording_id("aria", vid),
                embodiment=embodiment,
                **opts,
            )
        )
        return vid, None
    except Exception:
        return vid, traceback.format_exc()


@ray.remote
def chunk_one(mp4: str, k: int) -> list[bytes]:
    from egomimic.scripts.lance_pilot.chunk_videos import chunk_mp4

    return chunk_mp4(Path(mp4), k)


def chunked_ids(uri: Path) -> set[str]:
    if not (uri / "_versions").exists():
        return set()
    col = lance.dataset(str(uri)).to_table(columns=["recording_id"]).column(0)
    return set(col.unique().to_pylist())


def append_chunks(uri: Path, deliveries: list, camera: str, k: int) -> None:
    """Chunk rows for deliveries not yet in the chunk table, each recording's rows contiguous."""
    have = chunked_ids(uri)
    todo = [d for d in deliveries if d.recording_id not in have]
    refs = [chunk_one.remote(str(d.videos[camera]), k) for d in todo]
    for d, chunks in zip(todo, ray.get(refs), strict=True):
        nf = d.recording["num_frames"]
        starts = list(range(0, len(chunks) * k, k))
        if sum(min(k, nf - s) for s in starts) != nf:
            raise ValueError(
                f"{d.recording_id}: {len(chunks)} chunks of {k} for {nf} frames"
            )
        tbl = pa.table(
            {
                "recording_id": [d.recording_id] * len(chunks),
                "chunk_index": list(range(len(chunks))),
                "frame_start": starts,
                "num_frames": [min(k, nf - s) for s in starts],
                "data": blob_array(chunks),
            },
            schema=CHUNK_SCHEMA,
        )
        uri.parent.mkdir(parents=True, exist_ok=True)
        lance.write_dataset(
            tbl,
            str(uri),
            mode="append" if (uri / "_versions").exists() else "create",
            max_rows_per_file=1 << 20,
        )


def commit_dirs(args, dirs: list[Path], report: dict) -> None:
    if not dirs:
        return
    deliveries, rejected = load_deliveries(dirs)
    report["rejected"].update(rejected)
    # Chunks go first: chunk rows for an uncommitted recording are ignored by readers,
    # but a committed recording without chunk rows fails the streaming resolver.
    if args.chunk_k:
        uri = args.lance / "video_chunks" / f"{args.camera}_k{args.chunk_k}.lance"
        append_chunks(uri, deliveries, args.camera, args.chunk_k)
    added = commit.append(args.lance, deliveries)
    report["committed"] += added
    log.info("committed %d recordings (%d rejected)", len(added), len(rejected))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--deliveries", type=Path, required=True)
    ap.add_argument("--lance", type=Path, required=True)
    ap.add_argument(
        "--ray-address", help="existing cluster; default starts a local one"
    )
    ap.add_argument("--cpus-per-task", type=float, default=4)
    ap.add_argument("--commit-every", type=int, default=32)
    ap.add_argument(
        "--chunk-k",
        type=int,
        default=8,
        help="frames per video chunk row for streaming reads; 0 skips chunking",
    )
    ap.add_argument("--camera", default="front_1")
    ap.add_argument(
        "--embodiment",
        help="use this for every recording instead of looking it up in SQL",
    )
    ap.add_argument("--reconvert", action="store_true", help="redo existing deliveries")
    ap.add_argument("--limit", type=int, help="convert at most this many recordings")
    ap.add_argument("--mano", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mano-iters", type=int, default=2400)
    ap.add_argument("--ffmpeg", default="/coc/flash7/rco3/ffmpeg-7.1-shared/bin/ffmpeg")
    ap.add_argument("--tag", help="tag every table's version after the last commit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.chunk_k % 2:
        ap.error("--chunk-k must be even: deliveries are encoded with GOP 2")

    have = commit.committed_ids(args.lance)
    vids = [v for v in discover(args.raw_dir) if recording_id("aria", v) not in have]
    delivered = {v for v in vids if (args.deliveries / v / "meta.json").exists()} - (
        set(vids) if args.reconvert else set()
    )
    todo = [v for v in vids if v not in delivered][: args.limit]
    emb = dict.fromkeys(todo, args.embodiment) if args.embodiment else embodiments(todo)
    report = {
        "committed": [],
        "failed": {v: "no embodiment in SQL" for v in todo if v not in emb},
        "rejected": {},
    }
    log.info(
        "%d uncommitted recordings: %d to convert, %d already delivered",
        len(vids), len(emb), len(delivered),
    )  # fmt: skip

    ray.init(address=args.ray_address)
    opts = dict(
        raw_dir=args.raw_dir,
        out_dir=args.deliveries,
        mano=args.mano,
        mano_iters=args.mano_iters,
        ffmpeg=args.ffmpeg,
    )
    task = convert_one.options(num_cpus=args.cpus_per_task)
    pending = [task.remote(v, e, opts) for v, e in emb.items()]
    ready = [args.deliveries / v for v in sorted(delivered)]
    t0, done = time.time(), 0
    while pending:
        finished, pending = ray.wait(pending, num_returns=1)
        vid, err = ray.get(finished[0])
        done += 1
        if err:
            report["failed"][vid] = err
            log.warning("%s failed:\n%s", vid, err)
        else:
            ready.append(args.deliveries / vid)
        log.info("%d/%d converted, %.0fs", done, len(emb), time.time() - t0)
        if len(ready) >= args.commit_every:
            commit_dirs(args, ready, report)
            ready = []
    commit_dirs(args, ready, report)

    if args.tag and report["committed"]:
        report["tag"] = commit.tag(args.lance, args.tag)
    out = args.deliveries / f"ray_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        f"committed {len(report['committed'])}, failed {len(report['failed'])}, "
        f"rejected {len(report['rejected'])}; report at {out}"
    )


if __name__ == "__main__":
    main()
