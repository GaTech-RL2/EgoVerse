#!/usr/bin/env python3
"""
run_aria_conversion_sql.py – SQL-backed driver (no CSV/S3)

• Walks /mnt/raw for bundles: {name}.vrs, {name}.json , mps_{name}_vrs/
• name == episode_hash (TEXT in DB) – row must already exist in app.episodes; otherwise skipped
• Uses Ray to run conversions with absolute symlinks in per-job tmp dirs
• On success, updates app.episodes.processed_path and num_frames
• Passes SQL task_description into the converter as `description`
• Determines arm automatically from SQL (left, right, or bimanual)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple

import ray
import traceback

import csv

# --- Conversion wrapper ------------------------------------------------------
from aria_helper import lerobot_job

# --- SQL helpers --------------------------------------------------------------
from egomimic.utils.aws.aws_sql import (
    TableRow,
    create_default_engine,
    episode_hash_to_table_row,
    update_episode,
)

# --- Paths -------------------------------------------------------------------
RAW_ROOT = Path("/mnt/raw")
PROCESSED_ROOT = Path("/mnt/processed")
PROCESSED_LOCAL_ROOT = Path(
    os.environ.get("PROCESSED_LOCAL_ROOT", "/mnt/processed")
).resolve()
PROCESSED_REMOTE_PREFIX = os.environ.get(
    "PROCESSED_REMOTE_PREFIX", "rldb:/processed_v2/aria"
).rstrip("/")


# --- Utilities ---------------------------------------------------------------
def ensure_path_ready(p: str | Path, retries: int = 30) -> bool:
    p = Path(p)
    for _ in range(retries):
        try:
            if p.exists():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _map_processed_local_to_remote(p: str | Path) -> str:
    """Map any path under PROCESSED_LOCAL_ROOT → PROCESSED_REMOTE_PREFIX/relative."""
    if not p:
        return ""
    p = Path(p).resolve()
    try:
        rel = p.relative_to(PROCESSED_LOCAL_ROOT)  # raises if not under root
    except Exception:
        return str(p)
    return (
        f"{PROCESSED_REMOTE_PREFIX}/{rel.as_posix()}"
        if PROCESSED_REMOTE_PREFIX
        else str(p)
    )


def iter_vrs_bundles(root: Path) -> Iterator[Tuple[Path, Path, Path]]:
    """
    Yield (vrs_file, json_file, mps_dir) for every valid bundle in `root`.

    Accept bundles containing:
      • {name}.vrs
      • {name}.json
      • mps_{name}_vrs/
    All must be in the SAME directory.
    """
    for vrs in sorted(root.glob("*.vrs")):
        name = vrs.stem
        jsonf = root / f"{name}.json"
        mpsdir = root / f"mps_{name}_vrs"

        # Require all four to exist
        if mpsdir.is_dir():
            yield vrs, jsonf, mpsdir


def infer_arm_from_row(row: TableRow) -> str:
    """
    Infer arm from SQL row.robot_name (e.g., 'aria_left', 'aria_right', 'aria_bimanual').
    Falls back to 'bimanual'.
    """
    emb = (row.robot_name or "").lower()
    if "left" in emb:
        return "left"
    if "right" in emb:
        return "right"
    if "bimanual" in emb:
        return "bimanual"
    return "bimanual"


def _load_episode_hash(episode_hash: Path) -> str | None:
    return datetime.fromtimestamp(float(episode_hash) / 1000.0, timezone.utc).strftime(
        "%Y-%m-%d-%H-%M-%S-%f"
    )


# --- Ray task ----------------------------------------------------------------
@ray.remote(num_cpus=8)
def convert_one_bundle(
    vrs: str,
    jsonf: str,
    mps_dir: str,
    out_dir: str,
    dataset_name: str,
    arm: str,
    description: str,
) -> tuple[str, str, int]:
    """
    Perform symlink-based conversion for a single episode.
    Returns (ds_path, mp4_path, total_frames).
      • ds_path: dataset folder path
      • mp4_path: per-episode MP4 ('' if not created)
      • total_frames: -1 if unknown/failure
    """
    stem = Path(vrs).stem

    tmp_dir = Path.home() / "temp_mps_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        Path(vrs).resolve(strict=True),
        Path(jsonf).resolve(strict=True),
        Path(mps_dir).resolve(strict=True),
    ]

    for t in targets:
        if not ensure_path_ready(t):
            print(f"[ERR] missing {t}", flush=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return "", "", -1
        link = tmp_dir / t.name
        try:
            os.symlink(t, link, target_is_directory=t.is_dir())
        except FileExistsError:
            pass

    ds_parent = Path(out_dir)
    ds_parent.mkdir(parents=True, exist_ok=True)
    ds_path = ds_parent / dataset_name

    try:
        print(f"[INFO] Converting: {stem} → {ds_path} (arm={arm})", flush=True)
        lerobot_job(
            raw_path=str(tmp_dir),
            output_dir=str(ds_parent),
            dataset_name=dataset_name,
            arm=arm,
            description=description or "",
        )

        frames = -1
        info = ds_path / "meta/info.json"
        if info.exists():
            try:
                meta = json.loads(info.read_text())
                frames = int(meta.get("total_frames", -1))
            except Exception:
                frames = -1

        candidate = ds_parent / f"{stem}_video.mp4"
        if candidate.exists():
            mp4_str = str(candidate)
        else:
            matches = list(ds_path.glob(f"*{stem}*_video.mp4"))
            mp4_str = str(matches[0]) if matches else ""

        return str(ds_path), mp4_str, frames

    except Exception as e:
        err_msg = f"[FAIL] {stem}: {e}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        return str(ds_path), "", -1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Driver ------------------------------------------------------------------
def launch(dry: bool = False, skip_if_done: bool = False):
    engine = create_default_engine()
    pending: dict = {}

    benchmark_rows = []
    
    for vrs, jsonf, mps in iter_vrs_bundles(RAW_ROOT):
        name = vrs.stem

        # IMPORTANT: episode_hash is TEXT in DB; do not cast to int
        episode_key = _load_episode_hash(name)

        if not episode_key:
            print(f"[SKIP] {name}: no 'episode_hash' for {name}")
            continue

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)")
            continue

        processed_path = (row.processed_path or "").strip()
        if skip_if_done and len(processed_path) > 0:
            print(f"[SKIP] {name}: already has processed_path='{processed_path}'")
            continue

        arm = infer_arm_from_row(row)
        dataset_name = f"{name}_processed"
        out_dir = PROCESSED_ROOT
        description = row.task_description or ""

        if dry:
            ds_path = (PROCESSED_ROOT / f"{name}_processed").resolve()
            stem = vrs.stem
            mp4_candidate = PROCESSED_ROOT / f"{stem}_video.mp4"

            mapped_ds = _map_processed_local_to_remote(ds_path)
            mapped_mp4 = _map_processed_local_to_remote(mp4_candidate)

            print(
                f"[DRY] {name}: arm={arm} | out_dir={out_dir}/{dataset_name}\n"
                f"      desc='{description[:60]}'\n"
                f"      would write to SQL:\n"
                f"        processed_path={mapped_ds}\n"
                f"        mp4_path={mapped_mp4}"
            )
            continue

        start_time = time.time()
        ref = convert_one_bundle.remote(
            str(vrs),
            str(jsonf),
            str(mps),
            str(out_dir),
            dataset_name,
            arm,
            description,
        )
        pending[ref] = (episode_key, dataset_name, start_time)

    if dry or not pending:
        return

    # Collect and update SQL
    while pending:
        done_refs, _ = ray.wait(list(pending), num_returns=1)
        ref = done_refs[0]
        ds_path, mp4_path, frames = ray.get(ref)
        episode_key, _dataset_name, start_time = pending.pop(ref)

        duration_sec = time.time() - start_time
        
        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(f"[WARN] Episode {episode_key}: row disappeared before update?")
            continue

        row.num_frames = frames

        if row.num_frames > 0:
            row.processed_path = _map_processed_local_to_remote(ds_path)
            row.mp4_path = _map_processed_local_to_remote(mp4_path)
        else:
            row.processed_path = ""
            row.mp4_path = ""

        try:
            update_episode(engine, row)
            print(
                f"[OK] Updated SQL for {episode_key}: "
                f"processed_path={row.processed_path}, num_frames={row.num_frames}"
            )
            
            ## timing bench
            if row.num_frames > 0 and row.processed_path:
                benchmark_rows.append(
                    {
                        "episode_key": episode_key,
                        "processed_path": row.processed_path,
                        "mp4_path": row.mp4_path,
                        "num_frames": row.num_frames,
                        "duration_sec": duration_sec,
                    }
                )
                print(
                    f"[BENCH] episode_key={episode_key} "
                    f"processed_path={row.processed_path} "
                    f"duration_sec={duration_sec:.2f}"
                )
                
        except Exception as e:
            print(f"[ERR] SQL update failed for {episode_key}: {e}")

    if benchmark_rows:
        timing_file = Path("./aria_conversion_timings.csv")
        file_exists = timing_file.exists()
        fieldnames = [
            "episode_key",
            "processed_path",
            "mp4_path",
            "num_frames",
            "duration_sec",
        ]
        try:
            with timing_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for bench_row in benchmark_rows:
                    writer.writerow(bench_row)
            print(f"[BENCH] wrote {len(benchmark_rows)} entries → {timing_file.resolve()}")
        except Exception as e:
            print(f"[ERR] Failed to write benchmark CSV {timing_file}: {e}")

# --- CLI ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip episodes that already have a processed_path in SQL",
    )
    p.add_argument(
        "--ray-address", default="auto", help="Ray cluster address (default: auto)"
    )
    args = p.parse_args()

    ray.init(address=args.ray_address)
    launch(dry=args.dry_run, skip_if_done=args.skip_if_done)


if __name__ == "__main__":
    main()
