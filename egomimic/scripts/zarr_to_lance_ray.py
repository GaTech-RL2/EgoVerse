"""Convert the zarr-route episodes of a migration manifest (plan_zarr_migration) and stage them, one Ray task each.

Each task downloads one zarr from R2 to local disk, runs zarr_to_lance.convert_zarr_episode,
checks the delivery's recording_id against the manifest, stages the delivery under
`<stage-to>/zarr/<vendor id>/` for `lance_commit --source zarr` (the only Lance writer) and
deletes its local files. With --direct-videos the MP4s go straight to their final place
under --lance and only steps.parquet and meta.json are staged (commit.stage_direct).
Episodes already committed to --lance or already staged (meta.json present) are skipped, so
a killed run resumes where it stopped. Results append to --report (JSON lines).
See docs/zarr_to_lance_migration.md.

    python -m egomimic.scripts.zarr_to_lance_ray \\
        --manifest /coc/flash7/scratch/lance_pilot/migration_plan/manifest.parquet \\
        --stage-to s3://rldb/lance_staging --lance s3://rldb/lance \\
        --scratch /mnt/scratch --report /mnt/scratch/report.jsonl --limit 1
"""

import argparse
import fnmatch
import json
import logging
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import ray

from egomimic.rldb.lance import commit
from egomimic.scripts.zarr_to_lance import (
    FFMPEG,
    STAGING_SOURCE,
    convert_zarr_episode,
)
from egomimic.utils.env import load_env

log = logging.getLogger("zarr_to_lance_ray")

METADATA = (
    "lab",
    "operator",
    "task",
    "task_description",
    "scene",
    "objects",
    "license",
    "rig_name",
)


def download_prefix(uri: str, dst: Path, threads: int = 16) -> int:
    """Copy every object under the s3:// prefix `uri` into `dst`; returns bytes copied."""
    files = commit.list_files(uri)
    if not files:
        raise FileNotFoundError(f"no objects under {uri}")

    def get(rel):
        commit.get_file(commit.join(uri, rel), dst / rel)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(get, files))
    return sum(size for size, _ in files.values())


@ray.remote(max_retries=2, retry_exceptions=False)
def convert_one(row: dict, opts: dict) -> dict:
    """One manifest row to a staged delivery; never raises, the result says what happened."""
    load_env()
    logging.basicConfig(level=logging.INFO)
    work = Path(opts["scratch"]) / row["episode_hash"]
    res = {"episode_hash": row["episode_hash"], "recording_id": row["recording_id"]}
    t0 = time.time()
    try:
        shutil.rmtree(work, ignore_errors=True)
        res["zarr_bytes"] = download_prefix(row["zarr_uri"], work / "zarr")
        res["seconds_download"] = round(time.time() - t0, 1)
        metadata = {k: row.get(k) for k in METADATA}
        delivery = convert_zarr_episode(
            work / "zarr",
            work / "out",
            metadata,
            episode_hash=row["episode_hash"],
            ffmpeg=opts["ffmpeg"],
        )
        meta = json.loads((Path(delivery) / "meta.json").read_text())
        if meta["recording_id"] != row["recording_id"]:
            raise ValueError(
                f"converter recording_id {meta['recording_id']} != manifest {row['recording_id']}"
            )
        if meta["num_frames"] != row["num_frames"]:
            res["frames_mismatch"] = [meta["num_frames"], row["num_frames"]]
        res["num_frames"] = meta["num_frames"]
        res["seconds_convert"] = meta.get("seconds_total")
        if not opts["no_stage"]:
            dst = commit.join(
                opts["stage_to"], STAGING_SOURCE, meta["vendor_recording_id"]
            )
            if opts.get("direct_videos"):
                commit.stage_direct(delivery, dst, opts["lance"])
            else:
                commit.stage(delivery, dst)
        res["status"] = "staged" if not opts["no_stage"] else "converted"
    except Exception:
        res["status"] = "failed"
        res["error"] = traceback.format_exc(limit=8)
    finally:
        if not opts["keep_local"]:
            shutil.rmtree(work, ignore_errors=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def select(args) -> pd.DataFrame:
    m = pd.read_parquet(args.manifest)
    m = m[m.route == "zarr"]
    if args.shard:
        m = m[[any(fnmatch.fnmatch(s, p) for p in args.shard) for s in m.shard]]
    if args.work_unit:
        m = m[m.work_unit.isin(args.work_unit)]
    if args.lance:
        m = m[~m.recording_id.isin(commit.committed_ids(args.lance))]
    if args.stage_to:
        # A delivery is staged once its meta.json (uploaded last) is; a half-staged one is
        # redone.
        files = commit.list_files(commit.join(args.stage_to, STAGING_SOURCE))
        staged = {p.split("/")[0] for p in files if p.endswith("/meta.json")}
        m = m[~m.vendor_recording_id.isin(staged)]
    if args.report and args.report.exists() and not args.retry_failed:
        if args.report.stat().st_size:
            done = pd.read_json(args.report, lines=True)
            m = m[~m.episode_hash.isin(done.episode_hash)]
    return m.sort_values(["work_unit", "episode_hash"]).head(args.limit)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--stage-to", help="staging root (s3:// or local)")
    ap.add_argument("--lance", help="Lance root, read to skip committed recordings")
    ap.add_argument(
        "--scratch", type=Path, required=True, help="worker-local scratch dir"
    )
    ap.add_argument("--report", type=Path, required=True, help="JSON lines, appended")
    ap.add_argument("--limit", type=int, help="convert at most this many episodes")
    ap.add_argument("--shard", nargs="*", help="shard globs, e.g. 'rl2/eva_*'")
    ap.add_argument("--work-unit", nargs="*")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--no-stage", action="store_true", help="convert only")
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument(
        "--ray-address", help="existing cluster; default starts a local one"
    )
    ap.add_argument("--cpus-per-task", type=float, default=4)
    ap.add_argument(
        "--max-in-flight", type=int, default=512, help="tasks submitted at once"
    )
    ap.add_argument("--ffmpeg", default=FFMPEG, help="ffmpeg binary on the workers")
    ap.add_argument(
        "--direct-videos",
        action="store_true",
        help="upload MP4s straight to --lance; the committer only checks their sizes",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the selection and exit"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.no_stage and not args.stage_to and not args.dry_run:
        ap.error("--stage-to is required unless --no-stage")
    if args.direct_videos and not args.lance:
        ap.error("--direct-videos needs --lance")

    todo = select(args)
    log.info(
        "%d episodes, %d frames, %d work units, shards: %s",
        len(todo),
        todo.num_frames.sum(),
        todo.work_unit.nunique(),
        todo.shard.value_counts().to_dict(),
    )
    if args.dry_run:
        return

    ray.init(address=args.ray_address)
    opts = dict(
        scratch=str(args.scratch),
        stage_to=args.stage_to,
        no_stage=args.no_stage,
        keep_local=args.keep_local,
        ffmpeg=args.ffmpeg,
        lance=args.lance,
        direct_videos=args.direct_videos,
    )
    task = convert_one.options(num_cpus=args.cpus_per_task)
    rows = iter(todo.to_dict("records"))
    pending, done, failed, t0 = {}, 0, 0, time.time()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "a") as rep:
        while True:
            while len(pending) < args.max_in_flight and (row := next(rows, None)):
                pending[task.remote(row, opts)] = row
            if not pending:
                break
            (ref,), _ = ray.wait(list(pending), num_returns=1)
            row = pending.pop(ref)
            try:
                res = ray.get(ref)
            except Exception as e:
                # convert_one never raises; this is its worker dying (OOM) past max_retries.
                res = {
                    "episode_hash": row["episode_hash"],
                    "recording_id": row["recording_id"],
                    "status": "failed",
                    "error": repr(e),
                }
            rep.write(json.dumps(res) + "\n")
            rep.flush()
            done += 1
            if res["status"] == "failed":
                failed += 1
                log.warning("%s failed:\n%s", res["episode_hash"], res["error"])
            if done % 50 == 0 or done == len(todo):
                log.info(
                    "%d/%d done, %d failed, %.0fs",
                    done,
                    len(todo),
                    failed,
                    time.time() - t0,
                )
    print(f"{done - failed} ok, {failed} failed; report at {args.report}")


if __name__ == "__main__":
    main()
