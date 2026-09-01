"""
Upload locally converted ABC episodes to R2 and register them in app.episodes.

Built for the `fold_and_stack_the_t_shirts` set, which the existing `lab='abc'`
ingest does not contain at all (110 of 201 Hub tasks were ingested; t-shirt
folding is one of the 91 missing), so these episodes are net-new rather than a
re-upload.

Per episode it:
  1. refreshes `zarr.json` extrinsics to the CURRENT Yam.EXTRINSICS, so what is
     uploaded matches the refined mount transform rather than the stale nominal
     one the converter baked in at conversion time;
  2. uploads the .zarr to s3://<bucket>/processed_v3/abc/<hash>.zarr/ (zarr
     sharding keeps this to ~35 objects, so multipart PUTs, not a small-file
     storm);
  3. inserts one app.episodes row.

Both steps are idempotent and separately skippable, so a partial run resumes.

    # dry run (default): reports what would happen, writes nothing
    python -m egomimic.scripts.abc_process.upload_abc_episodes \
        --zarr-dir /coc/flash7/scratch/acheluva3/abc_data/zarr \
        --manifest /path/upload_manifest.json

    # commit
    ... --apply
"""

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from sqlalchemy import text

from egomimic.rldb.embodiment.yam import Yam
from egomimic.utils.aws.aws_data_utils import _uses_r2_endpoint, load_env
from egomimic.utils.aws.aws_sql import TableRow, add_episode, create_default_engine

logger = logging.getLogger("upload_abc")

PREFIX = "processed_v3/abc"
LAB = "abc"
OPERATOR = "ABC"
# Matches what the existing lab='abc' rows carry (it was bulk-changed from
# 'abc_yam' to 'abc'); keep new rows consistent with the rest of the ingest.
RIG_NAME = "abc"
LICENSE = "apache-2.0"
EMBODIMENT = "yam_bimanual"

_local = threading.local()
_TRANSFER = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                           multipart_chunksize=64 * 1024 * 1024,
                           max_concurrency=4, use_threads=True)


def _s3():
    if not hasattr(_local, "c"):
        ep = os.environ["R2_ENDPOINT_URL"]
        _local.c = boto3.client(
            "s3", endpoint_url=ep,
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            aws_session_token=None if _uses_r2_endpoint(ep) else os.environ.get("R2_SESSION_TOKEN"),
            region_name="auto",
            config=Config(max_pool_connections=64, retries={"max_attempts": 5}),
        )
    return _local.c


def refresh_extrinsics(zarr_path: Path, apply_changes: bool) -> str:
    """Point the store's zarr.json at the current Yam.EXTRINSICS."""
    jf = zarr_path / "zarr.json"
    doc = json.loads(jf.read_text())
    attrs = doc.setdefault("attributes", {})
    want = {"front_1": Yam.TOP_CAMERA_D405.tolist()}
    if attrs.get("extrinsics") == want and attrs.get("embodiment") == EMBODIMENT:
        return "current"
    attrs["extrinsics"] = want
    attrs["embodiment"] = EMBODIMENT
    if apply_changes:
        jf.write_text(json.dumps(doc))
    return "refreshed"


def upload_store(zarr_path: Path, bucket: str, apply_changes: bool) -> tuple[int, int]:
    files = [p for p in zarr_path.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if apply_changes:
        s3 = _s3()
        for p in files:
            key = f"{PREFIX}/{zarr_path.name}/{p.relative_to(zarr_path).as_posix()}"
            s3.upload_file(str(p), bucket, key, Config=_TRANSFER)
    return len(files), total


def already_uploaded(zarr_path: Path, bucket: str) -> bool:
    try:
        _s3().head_object(Bucket=bucket, Key=f"{PREFIX}/{zarr_path.name}/zarr.json")
        return True
    except Exception:
        return False


def process(job) -> dict:
    zarr_path, bucket, apply_changes, registered, force = job
    ep_hash = zarr_path.name[:-5] if zarr_path.name.endswith(".zarr") else zarr_path.name
    rec = {"episode_hash": ep_hash}
    t0 = time.time()
    try:
        attrs = json.loads((zarr_path / "zarr.json").read_text())["attributes"]
        rec.update(task=attrs["task_name"], frames=int(attrs["total_frames"]),
                   fps=attrs.get("fps"))
        rec["extrinsics"] = refresh_extrinsics(zarr_path, apply_changes)

        if already_uploaded(zarr_path, bucket) and not force:
            rec["upload"] = "already_present"
            rec["files"] = rec["bytes"] = 0
        else:
            n, b = upload_store(zarr_path, bucket, apply_changes)
            rec.update(upload="uploaded" if apply_changes else "would_upload",
                       files=n, bytes=b)
        rec["sql"] = "already_registered" if ep_hash in registered else (
            "inserted" if apply_changes else "would_insert")
        rec["status"] = "ok"
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:250])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--zarr-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--force", action="store_true", help="re-upload even if present")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or 8)
    args = p.parse_args()

    load_env()
    bucket = os.environ.get("BUCKET", "rldb")
    stores = sorted(Path(args.zarr_dir).glob("*.zarr"))
    if args.limit:
        stores = stores[: args.limit]
    if not stores:
        raise SystemExit(f"no .zarr stores under {args.zarr_dir}")

    engine = create_default_engine()
    hashes = [s.name[:-5] for s in stores]
    with engine.connect() as c:
        registered = {
            r[0] for r in c.execute(
                text("select episode_hash from app.episodes where episode_hash = any(:h)"),
                {"h": hashes})
        }
    logger.info("%d stores | %d already in app.episodes | %s | workers=%d",
                len(stores), len(registered),
                "APPLY" if args.apply else "DRY RUN", args.workers)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, (s, bucket, args.apply, registered, args.force))
                for s in stores]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            if i % 10 == 0 or i == len(futs):
                gb = sum(x.get("bytes", 0) for x in results) / 1e9
                logger.info("  %d/%d  ok=%d failed=%d  %.1f GB",
                            i, len(futs), sum(x["status"] == "ok" for x in results),
                            sum(x["status"] == "failed" for x in results), gb)

    # SQL rows last, single-threaded: one transaction per row, clear failures.
    inserted = 0
    if args.apply:
        for r in results:
            if r["status"] != "ok" or r["sql"] != "inserted":
                continue
            try:
                add_episode(engine, TableRow(
                    episode_hash=r["episode_hash"], operator=OPERATOR, lab=LAB,
                    task=r["task"], task_description=r["task"],
                    embodiment=EMBODIMENT, rig_name=RIG_NAME,
                    num_frames=r["frames"],
                    zarr_processed_path=f"s3://{bucket}/{PREFIX}/{r['episode_hash']}.zarr",
                ))
                inserted += 1
            except Exception as exc:
                r["sql"] = f"insert_failed: {type(exc).__name__}: {exc}"[:200]
        if inserted:
            # TableRow has no `license` field, so set it to match the ingest.
            with engine.begin() as c:
                c.execute(text("update app.episodes set license=:l "
                               "where lab=:lab and license is null"),
                          {"l": LICENSE, "lab": LAB})
    counts: dict[str, int] = {}
    for r in results:
        counts[r.get("upload", r["status"])] = counts.get(r.get("upload", r["status"]), 0) + 1
    manifest = {
        "applied": args.apply, "bucket": bucket, "prefix": PREFIX,
        "embodiment": EMBODIMENT, "rig_name": RIG_NAME,
        "episodes": len(results), "sql_inserted": inserted,
        "total_bytes": sum(r.get("bytes", 0) for r in results),
        "total_frames": sum(r.get("frames", 0) for r in results),
        "counts": counts,
        "records": sorted(results, key=lambda r: r["episode_hash"]),
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    logger.info("counts=%s  sql_inserted=%d  %.1f GB  %s frames",
                counts, inserted, manifest["total_bytes"] / 1e9, f"{manifest['total_frames']:,}")
    logger.info("manifest -> %s", args.manifest)
    for r in results:
        if r["status"] == "failed":
            logger.warning("  FAILED %s %s", r["episode_hash"][:12], r.get("error"))


if __name__ == "__main__":
    main()
