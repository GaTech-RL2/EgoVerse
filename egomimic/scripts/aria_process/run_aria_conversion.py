
import argparse, csv, json, os, shutil, uuid
from pathlib import Path
from filelock import FileLock
import ray
from aria_helper import lerobot_job

RAW_ROOT       = Path("/mnt/raw")
PROCESSED_ROOT = Path("/mnt/processed")
TASK_MAP_CSV   = RAW_ROOT / "task_map.csv"               # cols: task,arm
STATUS_NAME    = "vrs_status.csv"
TMP_ROOT       = Path.home() / "temp_mps_processing"

def load_task_map() -> dict[str, str]:
    with TASK_MAP_CSV.open() as f:
        return {r["task"]: r["arm"].lower() for r in csv.DictReader(f)}

def already_done(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {r["vrs"] for r in csv.DictReader(f)}

def vrs_bundles(task_dir: Path):
    """Yield (vrs_file, json_file, mps_dir) triples that pass integrity checks."""
    for vrs in task_dir.glob("*.vrs"):
        stem = vrs.stem
        jsonf = task_dir / f"{stem}.vrs.json"
        mps   = task_dir / f"mps_{stem}_vrs"
        if not (jsonf.exists() and mps.is_dir()):
            continue
        if not (mps / "hand_tracking/wrist_and_palm_poses.csv").exists():
            continue
        if not (mps / "slam/closed_loop_trajectory.csv").exists():
            continue
        yield vrs, jsonf, mps

@ray.remote(num_cpus=12, memory=16 * 1024**3)
def convert_one(tmp_dir: str, out_dir: str,
                dataset_name: str, arm: str) -> tuple[str, int]:
    """
    Run the conversion inside a temp dir.
    Returns (vrs_stem, total_frames).
    """
    lerobot_job(
        raw_path     = tmp_dir,
        output_dir   = out_dir,
        dataset_name = dataset_name,
        arm          = arm,
        description  = ""
    )
    info_p = Path(out_dir) / dataset_name / "meta/info.json"
    frames = -1
    if info_p.exists():
        frames = int(json.loads(info_p.read_text()).get("total_frames", -1))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return Path(dataset_name).stem, frames

def launch(dry_run: bool = False):
    jobs: list[tuple[ray.ObjectRef, Path]] = []

    for task, arm in load_task_map().items():
        task_dir   = RAW_ROOT / task
        status_csv = task_dir / STATUS_NAME
        done       = already_done(status_csv)

        for vrs, jsonf, mps in vrs_bundles(task_dir):
            if vrs.stem in done:
                continue

            tmp = TMP_ROOT / f"{vrs.stem}-{uuid.uuid4().hex[:6]}"
            tmp.mkdir(parents=True, exist_ok=True)
            for src in (vrs, jsonf, mps):
                os.symlink(src, tmp / src.name,
                           target_is_directory=src.is_dir())

            out_dir  = PROCESSED_ROOT / task
            dataset  = f"{vrs.stem}_processed"

            if dry_run:
                print(f"[DRY-RUN] would process {vrs.name}  →  {dataset}  (arm={arm})")
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                ref = convert_one.remote(str(tmp), str(out_dir), dataset, arm)
                jobs.append((ref, status_csv))

    if dry_run:
        return
    if not jobs:
        print("Nothing to convert – all up-to-date.")
        return

    print(f"Submitted {len(jobs)} conversions to Ray…")

    pending = dict(jobs)
    while pending:
        finished, _ = ray.wait(list(pending.keys()), num_returns=1)
        fut = finished[0]
        vrs_stem, frames = ray.get(fut)
        status_csv = pending.pop(fut)

        lock = FileLock(str(status_csv) + ".lock")
        with lock:
            new_file = not status_csv.exists()
            with status_csv.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["vrs", "total_frames"])
                if new_file:
                    writer.writeheader()
                writer.writerow({"vrs": vrs_stem, "total_frames": frames})
        print(f"{vrs_stem}   ({frames} frames)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List the jobs that would be launched, then exit")
    args = parser.parse_args()

    ray.init(address="auto")
    launch(dry_run=args.dry_run)