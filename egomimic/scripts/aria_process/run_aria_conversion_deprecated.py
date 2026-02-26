#!/usr/bin/env python3
"""
run_aria_conversion.py  – ABSOLUTE-symlink edition
"""

import argparse
import csv
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import boto3
import ray
from aria_helper import lerobot_job  # your wrapper around aria_to_lerobot
from filelock import FileLock

RAW_ROOT = Path("/mnt/raw")
PROCESSED_ROOT = Path("/mnt/processed")
TASK_MAP_CSV = RAW_ROOT / "task_map.csv"
GLOBAL_STATUS = PROCESSED_ROOT / "vrs_conversion_status.csv"

LOCAL_STATUS = Path("/tmp/vrs_conversion_status.csv")
LOCK_PATH = Path("/tmp/vrs_status.lock")
lock = FileLock(str(LOCK_PATH))


# ───────────────── helpers ────────────────────────────────────
def ensure_path_ready(p: str, retries: int = 30) -> bool:
    p = Path(p)
    for _ in range(retries):
        try:
            if p.exists():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def load_task_map() -> dict[str, str]:
    with TASK_MAP_CSV.open() as f:
        return {r["task"].strip(): r["arm"].strip().lower() for r in csv.DictReader(f)}


def already_done() -> set[str]:
    if not GLOBAL_STATUS.exists():
        return set()
    with GLOBAL_STATUS.open() as f:
        return {f"{r['task']}/{r['vrs']}" for r in csv.DictReader(f)}


def vrs_bundles(task_dir: Path):
    for vrs in task_dir.glob("*.vrs"):
        stem = vrs.stem
        jsonf = task_dir / f"{stem}.vrs.json"
        mps = task_dir / f"mps_{stem}_vrs"
        if not (
            jsonf.exists()
            and (mps / "hand_tracking/wrist_and_palm_poses.csv").exists()
            and (mps / "slam/closed_loop_trajectory.csv").exists()
        ):
            continue
        yield vrs, jsonf, mps


def load_meta_fields(vrs_file: Path) -> dict:
    """Loads metadata from corresponding {vrs_name}_meta.csv"""
    meta_file = vrs_file.parent / f"{vrs_file.stem}_meta.csv"
    if not meta_file.exists():
        return {}
    with meta_file.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return {}
        return rows[0]


def append_status(row: dict):
    with FileLock(str(LOCK_PATH)):
        if LOCAL_STATUS.exists():
            with LOCAL_STATUS.open("r", newline="") as f:
                reader = list(csv.DictReader(f))
                existing_fieldnames = set(reader[0].keys()) if reader else set()
                rows = reader
        else:
            existing_fieldnames = set()
            rows = []

        all_fields = list(existing_fieldnames.union(row.keys()))

        rows.append(row)
        with LOCAL_STATUS.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        boto3.client("s3").upload_file(
            str(LOCAL_STATUS), "rldb", "processed/vrs_conversion_status.csv"
        )


def extract_lab_name(vrs_stem: str, task_name: str) -> str:
    """
    Extract lab_name from a VRS stem using known task_name.

    Example:
    vrs_stem = 'scoop_granular_rl2_lab_scene_10_recording_3'
    task_name = 'scoop_granular'
    → lab_name = 'rl2_lab'
    """
    try:
        prefix = vrs_stem.split("_scene_")[0]  # 'scoop_granular_rl2_lab'
        expected_prefix = task_name + "_"
        if not prefix.startswith(expected_prefix):
            return "unknown"
        lab_part = prefix[len(expected_prefix) :]
        return lab_part
    except Exception:
        return "unknown"


# ───────────────── Ray task ────────────────────────────────────
@ray.remote(num_cpus=24, memory=48 * 1024**3)
def convert_one(
    vrs: str, jsonf: str, mps_dir: str, out_dir: str, arm: str
) -> tuple[str, int]:
    """Return (output_dataset_path, total_frames).  −1 on failure."""
    stem = Path(vrs).stem
    tmp_dir = Path.home() / "temp_mps_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ◀︎ make ABSOLUTE, verified targets
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
            pass  # another thread created it first

    ds_path = Path(out_dir) / f"{stem}_processed"
    try:
        print(f"[INFO] → {ds_path}", flush=True)
        lerobot_job(
            raw_path=str(tmp_dir),
            output_dir=str(out_dir),
            dataset_name=f"{stem}_processed",
            arm=arm,
            description="",
        )
        frames = -1
        info = ds_path / "meta/info.json"
        if info.exists():
            print("[DEBUG] Found metadata info.json")
            frames = int(json.loads(info.read_text()).get("total_frames", -1))
        return str(ds_path), frames
    except Exception as e:
        print(f"[FAIL] {stem}: {e}", flush=True)
        return str(ds_path), -1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ───────────────── driver ──────────────────────────────────────
def launch(dry: bool = False):
    done = already_done()
    pending = {}
    for task, arm in load_task_map().items():
        for vrs, jsonf, mps in vrs_bundles(RAW_ROOT / task):
            key = f"{task}/{vrs.stem}"
            if key in done:
                continue
            out_dir = PROCESSED_ROOT / task
            if dry:
                print(f"[DRY] {key} → {out_dir}/{vrs.stem}_processed  (arm={arm})")
                continue
            ref = convert_one.remote(str(vrs), str(jsonf), str(mps), str(out_dir), arm)
            pending[ref] = (task, vrs.stem)

    if dry or not pending:
        return

    while pending:
        done_ref, _ = ray.wait(list(pending), num_returns=1)
        ds_path, frames = ray.get(done_ref[0])
        task, vrs_stem = pending.pop(done_ref[0])
        vrs_path = RAW_ROOT / task / f"{vrs_stem}.vrs"
        meta = load_meta_fields(vrs_path)

        resolved_task = meta.get("task", task).strip()
        lab_name = meta.get("lab", extract_lab_name(vrs_stem, resolved_task)).strip()
        task_map = load_task_map()
        resolved_arm = meta.get("arm", task_map.get(resolved_task, "unknown")).strip()
        row = {
            "task": resolved_task,
            "lab": lab_name,
            "vrs": vrs_stem,
            "arm": resolved_arm,
            "total_frames": frames,
            "output_path": ds_path,
            **meta,  # add any other metadata
        }

        row.update(meta)
        append_status(row)


# ───────────────── CLI ──────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    ray.init(address="auto")
    launch(dry=args.dry_run)
