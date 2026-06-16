#!/usr/bin/env python3
"""
End-to-end Scale SFS -> EgoVerse Zarr pipeline.

Downloads tasks from the Scale API, converts to Zarr, uploads to S3,
registers in SQL, and cleans up local files.

Usage:
    # Full pipeline from CSV
    python sfs_zarr_pipeline.py --csv delivery.csv --workers 6 \\
        --upload-s3 --register-sql --delete-local

    # Single task
    python sfs_zarr_pipeline.py --task-ids TASK1 --upload-s3 --register-sql

    # Convert only (no S3/SQL)
    python sfs_zarr_pipeline.py --task-ids TASK1 TASK2

Environment:
    SCALE_API_KEY           Required for downloading tasks from Scale
    R2_ACCESS_KEY_ID        Cloudflare R2 access key (from ~/.egoverse_env)
    R2_SECRET_ACCESS_KEY    Cloudflare R2 secret key
    R2_ENDPOINT_URL         Cloudflare R2 endpoint URL
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sfs_to_egoverse_zarr import convert_task_to_zarr


# ---------------------------------------------------------------------------
# SQL helpers (lazy imports to avoid torch dependency at module level)
# ---------------------------------------------------------------------------

_sql_engine = None


def _get_sql_engine():
    global _sql_engine
    if _sql_engine is None:
        from egomimic.utils.aws.aws_sql import create_default_engine

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _sql_engine = create_default_engine()
        finally:
            sys.stdout = old_stdout
    return _sql_engine


def is_task_already_processed(task_id: str) -> str | None:
    """Returns zarr_processed_path if already done, else None."""
    from egomimic.utils.aws.aws_sql import episode_hash_to_table_row

    engine = _get_sql_engine()
    row = episode_hash_to_table_row(engine, task_id)
    if row and row.zarr_processed_path:
        return row.zarr_processed_path
    return None


def register_in_sql(
    task_id: str,
    task_desc: str,
    total_frames: int,
    s3_path: str,
) -> bool:
    """Register a converted zarr dataset in the SQL episode table."""
    from egomimic.utils.aws.aws_sql import TableRow, add_episode, update_episode

    engine = _get_sql_engine()
    row = TableRow(
        episode_hash=task_id,
        operator="scale",
        lab="scale",
        task=task_desc,
        embodiment="scale",
        robot_name="scale_bimanual",
        num_frames=total_frames,
        task_description=task_desc,
        scene="unknown",
        objects="",
        zarr_processed_path=s3_path,
        is_deleted=False,
        is_eval=False,
        eval_score=-1,
        eval_success=True,
    )

    try:
        add_episode(engine, row)
        return True
    except Exception:
        try:
            update_episode(engine, row)
            return True
        except Exception as exc:
            print(f"[{task_id}] SQL registration failed: {exc}")
            return False


# ---------------------------------------------------------------------------
# S3 upload via aws cli (uses optimised C transfer engine)
# ---------------------------------------------------------------------------


def upload_to_s3(
    local_dir: str,
    bucket: str,
    s3_prefix: str,
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    endpoint_url: str = "",
    delete_after: bool = False,
) -> str:
    """Upload a local directory tree to S3-compatible storage via `aws s3 sync`.

    Returns the full s3:// URI of the uploaded prefix.
    Supports Cloudflare R2 via endpoint_url.
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    s3_uri = f"s3://{bucket}/{s3_prefix}/"
    env = {**os.environ}
    if aws_access_key_id and aws_secret_access_key:
        env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if endpoint_url:
        env["AWS_DEFAULT_REGION"] = "auto"

    cmd = ["aws", "s3", "sync", str(local_path), s3_uri, "--only-show-errors"]
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed: {result.stderr.strip()}")

    if delete_after:
        shutil.rmtree(local_dir, ignore_errors=True)

    return s3_uri


# ---------------------------------------------------------------------------
# Single-task pipeline
# ---------------------------------------------------------------------------


def run_task(
    task_id: str,
    output_dir: str,
    download_dir: str,
    robot_type: str,
    fps: int,
    img_workers: int,
    *,
    is_flagship: bool = False,
    upload_s3: bool = False,
    s3_bucket: str = "rldb",
    aws_key: str = "",
    aws_secret: str = "",
    endpoint_url: str = "",
    do_register_sql: bool = False,
    delete_local: bool = False,
    max_interp_gap: int = 15,
    max_interp_velocity: float = 2.0,
) -> dict[str, Any]:
    """Run the full pipeline for one task: convert -> upload -> register -> cleanup."""
    t0 = time.perf_counter()

    result = convert_task_to_zarr(
        task_id=task_id,
        output_dir=output_dir,
        download_dir=download_dir,
        robot_type=robot_type,
        fps=fps,
        img_workers=img_workers,
        max_interp_gap=max_interp_gap,
        max_interp_velocity=max_interp_velocity,
    )

    episodes = result["episodes"]
    folder = result["folder"]
    task_desc = result["task_desc"]
    total_frames = result["total_frames"]
    local_dir = result["output_dir"]

    s3_full_path = ""
    category = "flagship" if is_flagship else "freeform"

    if upload_s3 and folder:
        s3_prefix = f"processed_v3/scale/{category}/{folder}"
        print(f"[{task_id}] Uploading to s3://{s3_bucket}/{s3_prefix}/ ...")
        s3_full_path = upload_to_s3(
            local_dir,
            s3_bucket,
            s3_prefix,
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            endpoint_url=endpoint_url,
            delete_after=delete_local,
        )
        print(f"[{task_id}] Uploaded -> {s3_full_path}")
    elif delete_local and folder and os.path.exists(local_dir):
        shutil.rmtree(local_dir, ignore_errors=True)

    sql_ok = False
    if do_register_sql and folder and s3_full_path:
        sql_ok = register_in_sql(task_id, task_desc, total_frames, s3_full_path)
        if sql_ok:
            print(f"[{task_id}] Registered in SQL ({category})")

    elapsed = time.perf_counter() - t0
    return {
        "task_id": task_id,
        "episodes": episodes,
        "folder": folder,
        "total_frames": total_frames,
        "s3_path": s3_full_path,
        "sql_registered": sql_ok,
        "elapsed": elapsed,
        "category": category,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def load_tasks_from_csv(csv_path: str) -> list[tuple[str, bool]]:
    """Load (task_id, is_flagship) pairs from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    flagship_col = df["Is Flagship"].astype(str).str.strip().str.upper() == "TRUE"
    return [
        (str(row["TASK_ID"]).strip(), bool(flagship_col.iloc[i]))
        for i, (_, row) in enumerate(df.iterrows())
    ]


def log_task_result(
    progress_file: str,
    task_id: str,
    folder: str,
    s3_path: str,
    episodes: int,
    status: str,
) -> None:
    with open(progress_file, "a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{task_id},{folder},{s3_path},{episodes},{status},{ts}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end Scale SFS -> EgoVerse Zarr pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--task-ids", nargs="+", help="Scale task IDs")
    input_group.add_argument(
        "--csv", help="CSV file with TASK_ID and Is Flagship columns"
    )

    parser.add_argument(
        "--output-dir", default="egoverse_zarr_dataset", help="Output root"
    )
    parser.add_argument(
        "--download-dir", default="scale_data", help="Temp download cache"
    )
    parser.add_argument(
        "--robot-type", default="scale_bimanual", help="Embodiment tag"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel task workers (default: 6)",
    )
    parser.add_argument(
        "--progress-file",
        default="zarr_pipeline_progress.csv",
        help="Log file for processed tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max tasks to process (0 = no limit)",
    )

    s3_group = parser.add_argument_group("S3/R2 upload")
    s3_group.add_argument("--upload-s3", action="store_true", help="Upload to S3/R2")
    s3_group.add_argument("--s3-bucket", default="rldb", help="S3 bucket")
    s3_group.add_argument("--endpoint-url", help="S3-compatible endpoint URL (for R2)")
    s3_group.add_argument(
        "--delete-local",
        action="store_true",
        help="Delete local files after upload",
    )

    sql_group = parser.add_argument_group("SQL registration")
    sql_group.add_argument(
        "--register-sql",
        action="store_true",
        help="Register episodes in SQL database",
    )

    interp_group = parser.add_argument_group("Interpolation")
    interp_group.add_argument(
        "--max-interp-gap", type=int, default=15,
        help="Max gap length (frames) to interpolate (default: 15 = 0.5s@30fps)",
    )
    interp_group.add_argument(
        "--max-interp-velocity", type=float, default=2.0,
        help="Max per-frame displacement (m) for interpolation sanity check (default: 2.0)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve task list with flagship/freeform metadata
    # ------------------------------------------------------------------
    task_entries: list[tuple[str, bool]] = []  # (task_id, is_flagship)
    if args.csv:
        task_entries = load_tasks_from_csv(args.csv)
    else:
        task_entries = [(tid, False) for tid in args.task_ids]

    if args.limit > 0:
        task_entries = task_entries[: args.limit]

    if not task_entries:
        print("No tasks to process.")
        return 0

    aws_key = os.environ.get("R2_ACCESS_KEY_ID", os.environ.get("AWS_ACCESS_KEY_ID", ""))
    aws_secret = os.environ.get("R2_SECRET_ACCESS_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    endpoint_url = args.endpoint_url or os.environ.get("R2_ENDPOINT_URL", "")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    img_workers = max(1, (os.cpu_count() or 4) // max(args.workers, 1))

    # ------------------------------------------------------------------
    # SQL idempotency check: filter out already-processed tasks
    # ------------------------------------------------------------------
    if args.register_sql:
        print(f"Checking SQL for {len(task_entries)} tasks...")
        _get_sql_engine()
        original_count = len(task_entries)
        filtered: list[tuple[str, bool]] = []
        skipped = 0
        for idx, (task_id, is_flagship) in enumerate(task_entries):
            if (idx + 1) % 100 == 0 or idx == 0:
                print(f"  SQL check {idx + 1}/{original_count}...")
            existing = is_task_already_processed(task_id)
            if existing:
                skipped += 1
                print(f"[SKIP] {task_id} already processed -> {existing}")
            else:
                filtered.append((task_id, is_flagship))
        task_entries = filtered
        print(f"SQL check done: {skipped} skipped, {len(task_entries)} to process")
        if skipped:
            print(f"Skipped {skipped}/{original_count} already-processed tasks")

    if not task_entries:
        print("All tasks already processed.")
        return 0

    n_flagship = sum(1 for _, f in task_entries if f)
    n_freeform = len(task_entries) - n_flagship

    # Header
    print()
    print("=" * 60)
    print("  Scale SFS -> EgoVerse Zarr Pipeline")
    print("=" * 60)
    print(f"  Tasks:     {len(task_entries)} ({n_flagship} flagship, {n_freeform} freeform)")
    print(f"  Workers:   {args.workers}")
    print(f"  Img threads/worker: {img_workers}")
    print(f"  Output:    {args.output_dir}/<timestamp>/")
    if args.upload_s3:
        dest = "R2" if endpoint_url else "S3"
        print(f"  {dest}:        s3://{args.s3_bucket}/processed_v3/scale/{{flagship,freeform}}/<timestamp>/")
    if args.register_sql:
        print("  SQL:       enabled (zarr_processed_path)")
    print(f"  Progress:  {args.progress_file}")
    print("=" * 60)
    print()

    total_episodes = 0
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    def _process_one(task_id: str, is_flagship: bool, idx: int) -> dict[str, Any] | None:
        print(f"[{idx}/{len(task_entries)}] {task_id} ({'flagship' if is_flagship else 'freeform'})")
        try:
            res = run_task(
                task_id=task_id,
                output_dir=args.output_dir,
                download_dir=args.download_dir,
                robot_type=args.robot_type,
                fps=args.fps,
                img_workers=img_workers,
                is_flagship=is_flagship,
                upload_s3=args.upload_s3,
                s3_bucket=args.s3_bucket,
                aws_key=aws_key,
                aws_secret=aws_secret,
                endpoint_url=endpoint_url,
                do_register_sql=args.register_sql,
                delete_local=args.delete_local,
                max_interp_gap=args.max_interp_gap,
                max_interp_velocity=args.max_interp_velocity,
            )
            log_task_result(
                args.progress_file,
                task_id,
                res["folder"],
                res["s3_path"],
                res["episodes"],
                "ok",
            )
            print(
                f"  {task_id}: {res['episodes']} eps, "
                f"{res['total_frames']} frames ({res['elapsed']:.1f}s) "
                f"-> {res['category']}"
            )
            return res
        except Exception as exc:
            print(f"[{task_id}] ERROR: {exc}")
            traceback.print_exc()
            log_task_result(args.progress_file, task_id, "", "", 0, f"failed: {str(exc)[:80]}")
            return None

    if args.workers > 1:
        print(
            f"Running with {args.workers} parallel workers "
            f"({img_workers} image threads per worker)\n"
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, tid, is_fl, idx): tid
                for idx, (tid, is_fl) in enumerate(task_entries, start=1)
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    res = future.result()
                    if res:
                        total_episodes += res["episodes"]
                        results.append(res)
                    else:
                        failed.append(tid)
                except Exception as exc:
                    print(f"  {tid}: UNEXPECTED ERROR — {exc}")
                    traceback.print_exc()
                    failed.append(tid)
    else:
        for idx, (task_id, is_flagship) in enumerate(task_entries, start=1):
            res = _process_one(task_id, is_flagship, idx)
            if res:
                total_episodes += res["episodes"]
                results.append(res)
            else:
                failed.append(task_id)

    total_time = time.perf_counter() - start_time

    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(
        f"  Tasks:    {len(task_entries)} total, "
        f"{len(task_entries) - len(failed)} ok, {len(failed)} failed"
    )
    print(f"  Episodes: {total_episodes}")
    print(f"  Time:     {total_time:.1f}s ({total_time / 60:.1f}m)")
    if args.upload_s3:
        n_uploaded = sum(1 for r in results if r.get("s3_path"))
        print(f"  S3:       {n_uploaded} uploaded")
    if args.register_sql:
        n_registered = sum(1 for r in results if r.get("sql_registered"))
        print(f"  SQL:      {n_registered} registered")
    print(f"  Progress: {args.progress_file}")
    if failed:
        print(f"  Failed:   {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print("=" * 60)
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
