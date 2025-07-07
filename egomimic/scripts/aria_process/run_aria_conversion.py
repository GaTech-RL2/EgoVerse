#!/usr/bin/env python3
import argparse, csv, json, os, shutil, uuid
from pathlib import Path
from filelock import FileLock
import ray
from aria_helper import lerobot_job

# ───────────── paths ──────────────────────────────────────────────
RAW_ROOT        = Path("/mnt/raw")
PROCESSED_ROOT  = Path("/mnt/processed")
TASK_MAP_CSV    = RAW_ROOT / "task_map.csv"          # cols: task,arm
GLOBAL_STATUS   = PROCESSED_ROOT / "vrs_conversion_status.csv"
TMP_ROOT        = Path.home() / "temp_mps_processing"

# ───────────── helpers ────────────────────────────────────────────
def load_task_map() -> dict[str, str]:
    """Read task_map.csv → {task: arm} (arm lowered)."""
    with TASK_MAP_CSV.open() as f:
        return {r["task"]: r["arm"].lower() for r in csv.DictReader(f)}

def already_done() -> set[str]:
    """Return set of ‘task/stem’ keys already in the global CSV."""
    if not GLOBAL_STATUS.exists():
        return set()
    with GLOBAL_STATUS.open() as f:
        return {f"{r['task']}/{r['vrs']}" for r in csv.DictReader(f)}

def vrs_bundles(task_dir: Path):
    """Yield (vrs_path, json_path, mps_dir) triples that pass integrity checks."""
    for vrs in task_dir.glob("*.vrs"):
        stem  = vrs.stem
        jsonf = task_dir / f"{stem}.vrs.json"
        mps   = task_dir / f"mps_{stem}_vrs"
        if not (jsonf.exists() and mps.is_dir() and
                (mps/"hand_tracking/wrist_and_palm_poses.csv").exists() and
                (mps/"slam/closed_loop_trajectory.csv").exists()):
            continue
        yield vrs, jsonf, mps

# ───────────── Ray remote task ────────────────────────────────────
@ray.remote(num_cpus=8, memory=16 * 1024**3)
def convert_one(tmp_dir: str, out_dir: str,
                dataset_name: str, arm: str) -> tuple[str, int]:
    """
    Run conversion; return (dataset_path, total_frames).
    On failure → frames = -1.
    """
    out_dir = Path(out_dir)
    ds_path = out_dir / dataset_name     # we’ll report this
    try:
        print(f"[INFO] Converting → {ds_path}", flush=True)

        lerobot_job(raw_path=tmp_dir,
                    output_dir=str(out_dir),
                    dataset_name=dataset_name,
                    arm=arm,
                    description="")

        info_p = ds_path / "meta/info.json"
        frames = -1
        if info_p.exists():
            frames = int(json.loads(info_p.read_text()).get("total_frames", -1))

        print(f"[INFO] Done   → {ds_path} ({frames} frames)", flush=True)
        return str(ds_path), frames

    except Exception as exc:
        print(f"[ERROR] {ds_path} failed: {exc}", flush=True)
        return str(ds_path), -1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ───────────── main driver ────────────────────────────────────────
def launch(dry_run: bool = False):
    already = already_done()
    pending_jobs: dict[ray.ObjectRef, tuple[str,str,str]] = {}  # ref → (task, vrs_stem, ds_path)

    for task, arm in load_task_map().items():
        task_dir = RAW_ROOT / task
        for vrs, jsonf, mps in vrs_bundles(task_dir):
            key = f"{task}/{vrs.stem}"
            if key in already:
                continue

            # temp workspace with symlinks
            tmp = TMP_ROOT / f"{vrs.stem}-{uuid.uuid4().hex[:6]}"
            tmp.mkdir(parents=True, exist_ok=True)
            for src in (vrs, jsonf, mps):
                os.symlink(src, tmp/src.name, target_is_directory=src.is_dir())

            out_dir   = PROCESSED_ROOT / task
            dataset   = f"{vrs.stem}_processed"
            ds_path   = out_dir / dataset

            if dry_run:
                print(f"[DRY-RUN] {task} → {ds_path}  (arm={arm})")
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                ref = convert_one.remote(str(tmp), str(out_dir), dataset, arm)
                pending_jobs[ref] = (task, vrs.stem, str(ds_path))

    if dry_run or not pending_jobs:
        print("Dry run complete." if dry_run else "Nothing to do.")
        return

    print(f"Submitted {len(pending_jobs)} jobs to Ray…")

    lock = FileLock(str(GLOBAL_STATUS)+".lock")
    while pending_jobs:
        finished, _ = ray.wait(list(pending_jobs.keys()), num_returns=1)
        ref = finished[0]
        ds_path, frames = ray.get(ref)
        task, vrs_stem, _ = pending_jobs.pop(ref)

        with lock:  # append a row under lock
            new_file = not GLOBAL_STATUS.exists()
            with GLOBAL_STATUS.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["task", "vrs", "total_frames", "output_path"])
                if new_file:
                    writer.writeheader()
                writer.writerow({
                    "task": task,
                    "vrs":  vrs_stem,
                    "total_frames": frames,
                    "output_path": ds_path
                })
        print(f"[LOG] {task}/{vrs_stem} → {frames} frames")

# ───────────── CLI entry ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List conversions without running them")
    args = parser.parse_args()

    ray.init(address="auto")
    launch(dry_run=args.dry_run)
