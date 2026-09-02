#!/usr/bin/env python3
"""Scan a processed_v3/<folder> tree on S3 and stage every zarr's metadata into a
SQL staging table (default ``app.staging_<folder>``) for review before merging
into ``app.episodes``.

For each ``*.zarr`` episode found (recursively — some folders nest by category),
it reads the zarr group metadata (``zarr.json`` / ``.zattrs``) and stages one row
with the columns that map to ``app.episodes`` (``episode_hash``, ``embodiment``,
``task``, ``task_description``, ``num_frames``, ``zarr_processed_path``) plus
staging-only fields (``fps``, ``has_annotations``, ``source_folder``,
``created_at`` parsed from the hash when it is a recording timestamp,
``raw_attrs`` = the full attributes as JSONB, and ``scanned_at``).
``episode_hash`` = the ``.zarr`` dir name and is the table PRIMARY KEY; duplicate
hashes are de-duplicated (INSERT ... ON CONFLICT DO NOTHING).

The metadata read (one S3 GET per episode) is the bottleneck. For large folders
(microagi has ~300k episodes) this is distributed across the Ray cluster's small
CPU workers (``--ray``, the default); ``--no-ray`` falls back to a local thread
pool. The staging table is rebuilt from scratch on each run; it does NOT touch
app.episodes (the merge is a separate step).

Usage (on aria-head-node, Ray cluster up):
    python -m egomimic.scripts.backfill_scripts.stage_processed_folder --folder microagi
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import text

from egomimic.utils.aws.aws_sql import create_default_engine

BUCKET = "rldb"
_TS_HASH = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}$")


def _r2_cfg():
    """R2 credentials from the environment (passed to Ray workers so the task is
    self-contained — no ~/.egoverse_env or egomimic import needed on workers)."""
    return {
        "endpoint": os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("S3_ENDPOINT"),
        "key": os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret": os.environ.get("R2_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY"),
    }


def _s3_client(cfg):
    import boto3

    return boto3.client(
        "s3", endpoint_url=cfg["endpoint"], region_name="auto",
        aws_access_key_id=cfg["key"], aws_secret_access_key=cfg["secret"],
    )


def find_zarr_prefixes(s3, base):
    """Recursively yield every ``<...>.zarr/`` prefix under ``base``."""
    stack = [base]
    while stack:
        prefix = stack.pop()
        token = None
        while True:
            kw = dict(Bucket=BUCKET, Prefix=prefix, Delimiter="/", MaxKeys=1000)
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for cp in resp.get("CommonPrefixes", []):
                sub = cp["Prefix"]
                if sub.rstrip("/").endswith(".zarr"):
                    yield sub
                else:
                    stack.append(sub)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")




def read_batch(prefixes, folder, cfg):
    """Read the zarr.json for a batch of prefixes -> list[row dict]. FULLY
    self-contained (only stdlib + boto3, imported inside) so it runs on any Ray
    worker regardless of whether egomimic / ~/.egoverse_env are present there."""
    import json as _json
    import re as _re
    from datetime import datetime as _dt, timezone as _tz

    import boto3

    bucket = "rldb"
    ts_re = _re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{6}$")
    s3 = boto3.client(
        "s3", endpoint_url=cfg["endpoint"], region_name="auto",
        aws_access_key_id=cfg["key"], aws_secret_access_key=cfg["secret"],
    )
    out = []
    for zp in prefixes:
        episode_hash = zp.rstrip("/").split("/")[-1][:-5]  # strip ".zarr"
        attrs = {}
        for meta_name in ("zarr.json", ".zattrs"):
            try:
                body = s3.get_object(Bucket=bucket, Key=zp + meta_name)["Body"].read()
            except Exception:
                continue
            m = _json.loads(body)
            attrs = m.get("attributes", m) or {}
            break
        created_at = None
        if ts_re.match(episode_hash):
            try:
                created_at = _dt.strptime(
                    episode_hash, "%Y-%m-%d-%H-%M-%S-%f"
                ).replace(tzinfo=_tz.utc).isoformat()
            except ValueError:
                created_at = None
        feats = attrs.get("features")
        out.append({
            "episode_hash": episode_hash,
            "embodiment": attrs.get("embodiment"),
            "task": attrs.get("task_name"),
            "task_description": attrs.get("task_description"),
            "num_frames": attrs.get("total_frames"),
            "zarr_processed_path": f"s3://{bucket}/{zp.rstrip('/')}",
            "fps": attrs.get("fps"),
            "has_annotations": isinstance(feats, dict) and "annotations" in feats,
            "source_folder": folder,
            "created_at": created_at,
            "raw_attrs": _json.dumps(attrs),
        })
    return out


_CREATE_TABLE = """
    CREATE TABLE {table} (
        episode_hash        TEXT PRIMARY KEY,
        embodiment          TEXT,
        task                TEXT,
        task_description    TEXT,
        num_frames          INTEGER,
        zarr_processed_path TEXT,
        fps                 DOUBLE PRECISION,
        has_annotations     BOOLEAN,
        source_folder       TEXT,
        created_at          TIMESTAMPTZ,
        raw_attrs           JSONB,
        scanned_at          TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""

_INSERT = """
    INSERT INTO {table}
        (episode_hash, embodiment, task, task_description, num_frames,
         zarr_processed_path, fps, has_annotations, source_folder, created_at, raw_attrs)
    VALUES
        (:episode_hash, :embodiment, :task, :task_description, :num_frames,
         :zarr_processed_path, :fps, :has_annotations, :source_folder,
         :created_at, CAST(:raw_attrs AS JSONB))
    ON CONFLICT (episode_hash) DO NOTHING
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True,
                    help="Subfolder under processed_v3/ to scan (e.g. microagi).")
    ap.add_argument("--staging-table", default=None,
                    help="Fully-qualified staging table (default app.staging_<folder>).")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N zarrs (testing).")
    ap.add_argument("--batch-size", type=int, default=400,
                    help="Zarrs per Ray task / insert batch (default 400).")
    ap.add_argument("--workers", type=int, default=48,
                    help="Local threads when --no-ray (default 48).")
    ap.add_argument("--ray", dest="use_ray", action="store_true", default=True,
                    help="Distribute the metadata reads across the Ray cluster (default).")
    ap.add_argument("--no-ray", dest="use_ray", action="store_false")
    args = ap.parse_args()

    folder = args.folder.strip("/")
    table = args.staging_table or (
        "app.staging_" + re.sub(r"[^a-z0-9_]", "_", folder.lower())
    )
    base = f"processed_v3/{folder}/"
    cfg = _r2_cfg()
    engine = create_default_engine()

    # 1) list every episode prefix on the head (paginated listing is cheap).
    print(f"Listing .zarr prefixes under s3://{BUCKET}/{base} ...", flush=True)
    prefixes = list(find_zarr_prefixes(_s3_client(cfg), base))
    if args.limit is not None:
        prefixes = prefixes[: args.limit]
    print(f"Found {len(prefixes)} .zarr prefixes.", flush=True)
    batches = [prefixes[i:i + args.batch_size] for i in range(0, len(prefixes), args.batch_size)]

    # 2) create the staging table fresh.
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(text(_CREATE_TABLE.format(table=table)))

    insert_sql = text(_INSERT.format(table=table))
    staged = 0

    def insert_rows(rows):
        nonlocal staged
        if rows:
            with engine.begin() as conn:
                conn.execute(insert_sql, rows)
            staged += len(rows)

    # 3) read metadata (distributed) and stream the rows into the table.
    if args.use_ray:
        import ray
        ray.init(address="auto", ignore_reinit_error=True)
        remote_read = ray.remote(num_cpus=1)(read_batch)
        futures = [remote_read.remote(b, folder, cfg) for b in batches]
        print(f"Submitted {len(futures)} Ray tasks across the cluster; streaming inserts...",
              flush=True)
        pending = list(futures)
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=None)
            for rows in ray.get(ready):
                insert_rows(rows)
            if staged and staged % 20000 < args.batch_size:
                print(f"  staged {staged}/{len(prefixes)} ...", flush=True)
        ray.shutdown()
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for rows in ex.map(lambda b: read_batch(b, folder, cfg), batches):
                insert_rows(rows)
                if staged % 20000 < args.batch_size:
                    print(f"  staged {staged}/{len(prefixes)} ...", flush=True)

    # 4) summary.
    with engine.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        fill = conn.execute(text(f"""
            SELECT
              COUNT(*) FILTER (WHERE embodiment IS NOT NULL)          AS embodiment,
              COUNT(*) FILTER (WHERE task IS NOT NULL)                AS task,
              COUNT(*) FILTER (WHERE task_description IS NOT NULL)    AS task_description,
              COUNT(*) FILTER (WHERE num_frames IS NOT NULL)         AS num_frames,
              COUNT(*) FILTER (WHERE zarr_processed_path IS NOT NULL) AS zarr_path,
              COUNT(*) FILTER (WHERE created_at IS NOT NULL)          AS created_at,
              COUNT(*) FILTER (WHERE has_annotations)                AS has_annotations
            FROM {table}
        """)).mappings().first()
        sample = conn.execute(text(f"""
            SELECT episode_hash, embodiment, task, num_frames, created_at, has_annotations
            FROM {table} ORDER BY episode_hash LIMIT 5
        """)).all()

    print(f"\nStaged {n} rows into {table} (of {len(prefixes)} prefixes scanned)")
    print(f"  column fill counts (of {n} rows):")
    for k, v in fill.items():
        print(f"    {k:20} {v}")
    print("  sample rows:")
    for s in sample:
        print(f"    {s.episode_hash}  emb={s.embodiment} task={s.task!r} "
              f"nf={s.num_frames} created_at={s.created_at} ann={s.has_annotations}")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
