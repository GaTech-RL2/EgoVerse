"""Convert raw Aria recordings (VRS + MPS) into staged deliveries, one Ray task per recording.

Workers run aria_to_lance.convert into `<deliveries>/<vendor id>/`; the driver stages each
finished delivery under `<stage-to>/aria/<vendor id>/` for the committer
(egomimic/scripts/lance_commit.py), which is the only Lance writer. Reruns skip recordings
already delivered or committed, so a killed run resumes where it stopped.

    python -m egomimic.scripts.aria_process.aria_to_lance_ray \\
        --raw-dir /coc/flash7/scratch/lance_pilot/raw \\
        --deliveries /coc/flash7/scratch/lance_pilot/deliveries \\
        --stage-to s3://rldb/lance_staging --lance s3://rldb/lance
"""

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import ray

from egomimic.rldb.lance import commit
from egomimic.rldb.lance.schema import recording_id
from egomimic.scripts.lance_commit import sql_rows
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


def stage_dir(args, d: Path, report: dict) -> None:
    commit.stage(d, commit.join(args.stage_to, "aria", d.name))
    report["staged"].append(d.name)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--deliveries", type=Path, required=True)
    ap.add_argument("--stage-to", required=True, help="staging root (s3:// or local)")
    ap.add_argument(
        "--lance", required=True, help="Lance root, read to skip committed recordings"
    )
    ap.add_argument(
        "--ray-address", help="existing cluster; default starts a local one"
    )
    ap.add_argument("--cpus-per-task", type=float, default=4)
    ap.add_argument(
        "--embodiment",
        help="use this for every recording instead of looking it up in SQL",
    )
    ap.add_argument("--reconvert", action="store_true", help="redo existing deliveries")
    ap.add_argument("--limit", type=int, help="convert at most this many recordings")
    ap.add_argument("--mano", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mano-iters", type=int, default=2400)
    ap.add_argument("--ffmpeg", default="/coc/flash7/rco3/ffmpeg-7.1-shared/bin/ffmpeg")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    have = commit.committed_ids(args.lance)
    vids = [v for v in discover(args.raw_dir) if recording_id("aria", v) not in have]
    delivered = {v for v in vids if (args.deliveries / v / "meta.json").exists()} - (
        set(vids) if args.reconvert else set()
    )
    todo = [v for v in vids if v not in delivered][: args.limit]
    emb = dict.fromkeys(todo, args.embodiment) if args.embodiment else embodiments(todo)
    report = {
        "staged": [],
        "failed": {v: "no embodiment in SQL" for v in todo if v not in emb},
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
    for v in sorted(delivered):
        stage_dir(args, args.deliveries / v, report)
    t0, done = time.time(), 0
    while pending:
        finished, pending = ray.wait(pending, num_returns=1)
        vid, err = ray.get(finished[0])
        done += 1
        if err:
            report["failed"][vid] = err
            log.warning("%s failed:\n%s", vid, err)
        else:
            stage_dir(args, args.deliveries / vid, report)
        log.info("%d/%d converted, %.0fs", done, len(emb), time.time() - t0)
    out = args.deliveries / f"ray_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(
        f"staged {len(report['staged'])}, failed {len(report['failed'])}; report at {out}"
    )


if __name__ == "__main__":
    main()
