#!/usr/bin/env python3
"""
run_aria_conversion_sql.py – SQL-backed driver (no CSV/S3)

• Walks /mnt/raw for bundles: {name}.vrs, {name}.vrs.json, mps_{name}_vrs/
• name == episode_hash (int) – row must already exist in app.episodes; otherwise skipped
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
from pathlib import Path
from typing import Iterator, Tuple

import ray

# --- SQL helpers --------------------------------------------------------------
from egomimic.utils.aws.aws_sql import (
    TableRow,
    create_default_engine,
    episode_hash_to_table_row,
    update_episode,
)

# --- Conversion wrapper ------------------------------------------------------
from aria_helper import lerobot_job

# --- Paths -------------------------------------------------------------------
RAW_ROOT = Path("/mnt/raw")
PROCESSED_ROOT = Path("/mnt/processed")

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


def iter_vrs_bundles(root: Path) -> Iterator[Tuple[Path, Path, Path]]:
    """
    Yield tuples (vrs_file, json_file, mps_dir) for every valid bundle in `root`.
    """
    for vrs in sorted(root.glob("*.vrs")):
        name = vrs.stem
        jsonf = root / f"{name}.vrs.json"
        mpsdir = root / f"mps_{name}_vrs"
        if jsonf.exists() and mpsdir.is_dir():
            yield vrs, jsonf, mpsdir


def infer_arm_from_row(row: TableRow) -> str:
    """
    Infer arm from SQL row.embodiment (e.g., 'aria_left', 'aria_right', 'aria_bimanual').
    """
    emb = (row.embodiment or "").lower()
    if "left" in emb:
        return "left"
    if "right" in emb:
        return "right"
    if "bimanual" in emb or "bi" in emb:
        return "bimanual"
    return "bimanual"  # fallback


# --- Ray task ----------------------------------------------------------------
@ray.remote(num_cpus=24, memory=48 * 1024 ** 3)
def convert_one_bundle(
    vrs: str,
    jsonf: str,
    mps_dir: str,
    out_dir: str,
    dataset_name: str,
    arm: str,
    description: str,
) -> tuple[str, int]:
    """
    Perform symlink-based conversion for a single episode.
    Returns (dataset_path, total_frames). total_frames = -1 if unknown/failure.
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
            return "", -1
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

        return str(ds_path), frames
    except Exception as e:
        print(f"[FAIL] {stem}: {e}", flush=True)
        return str(ds_path), -1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Driver ------------------------------------------------------------------
def launch(dry: bool = False, skip_if_done: bool = False):
    engine = create_default_engine()
    pending: dict = {}

    for vrs, jsonf, mps in iter_vrs_bundles(RAW_ROOT):
        name = vrs.stem

        try:
            episode_hash = int(name)
        except Exception:
            print(f"[SKIP] {name}: filename is not a numeric episode_hash")
            continue

        row = episode_hash_to_table_row(engine, episode_hash)
        if row is None:
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)")
            continue

        if skip_if_done and (row.processed_path or "").strip():
            print(f"[SKIP] {name}: already has processed_path")
            continue

        arm = infer_arm_from_row(row)
        dataset_name = f"{name}_processed"
        out_dir = PROCESSED_ROOT
        description = row.task_description or ""

        if dry:
            print(f"[DRY] {name}: arm={arm} → {out_dir}/{dataset_name} | desc='{description[:60]}'")
            continue

        ref = convert_one_bundle.remote(
            str(vrs),
            str(jsonf),
            str(mps),
            str(out_dir),
            dataset_name,
            arm,
            description,
        )
        pending[ref] = (episode_hash, dataset_name)

    if dry or not pending:
        return

    # Collect and update SQL
    while pending:
        done_refs, _ = ray.wait(list(pending), num_returns=1)
        ref = done_refs[0]
        ds_path, frames = ray.get(ref)
        episode_hash, _dataset_name = pending.pop(ref)

        row = episode_hash_to_table_row(engine, episode_hash)
        if row is None:
            print(f"[WARN] Episode {episode_hash}: row disappeared before update?")
            continue

        row.processed_path = ds_path or ""
        row.num_frames = frames if isinstance(frames, int) else -1

        try:
            update_episode(engine, row)
            print(
                f"[OK] Updated SQL for {episode_hash}: "
                f"processed_path={row.processed_path}, num_frames={row.num_frames}"
            )
        except Exception as e:
            print(f"[ERR] SQL update failed for {episode_hash}: {e}")


# --- CLI ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-if-done", action="store_true",
                   help="Skip episodes that already have a processed_path in SQL")
    p.add_argument("--ray-address", default="auto", help="Ray cluster address (default: auto)")
    args = p.parse_args()

    ray.init(address=args.ray_address)
    launch(dry=args.dry_run, skip_if_done=args.skip_if_done)


if __name__ == "__main__":
    main()
