#!/usr/bin/env python3
"""
run_daily_mps.py – processes /mnt/raw
  • normal call: runs aria_mps on all *.vrs files in /mnt/raw in batches of 16
  • --debug: dry-run (lists the .vrs files & batch plan, no work)
  • Single-node Ray: spins up one local Ray node, then runs a single remote job
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
import traceback
from pathlib import Path

import ray

RAW_ROOT = Path("/mnt/raw")
ARIA_DIR = RAW_ROOT  # currently /mnt/raw; change to RAW_ROOT / "aria" if you add that subdir

MPS_INI_CONTENT = """[DEFAULT]
log_dir = /tmp/logs/projectaria/mps/ # Path to log directory
status_check_interval = 30 # Status check interval in seconds

[HASH]
concurrent_hashes = 16 # Maximum number of recordings whose hashes will be calculated concurrently
chunk_size = 10485760 # 10 * 2**20 (10MB)

[HEALTH_CHECK]
concurrent_health_checks = 2  # Maximum number of checks that can run concurrently

[ENCRYPTION]
chunk_size = 52428800 # 50 * 2**20 (50MB)
concurrent_encryptions = 5 # Maximum number of recordings that will be encrypted concurrently
delete_encrypted_files = true # Delete encrypted files after upload is done

[UPLOAD]
backoff = 1.5 # Backoff factor for retries
concurrent_uploads = 16 # Maximum number of concurrent uploads
interval = 20 # Interval between runs
max_chunk_size = 104857600 # 100 * 2**20 (100 MB)
min_chunk_size = 5242880 # 5 * 2**20 (5MB)
retries = 10 # Number of times to retry a failed upload
smoothing_window_size = 10 # Size of the smoothing window
target_chunk_upload_secs = 3 # Target duration to upload a chunk

[DOWNLOAD]
backoff = 1.5 # Backoff factor for retries
chunk_size = 10485760 # 10 * 2**20 (10MB)
concurrent_downloads = 10 # Maximum number of concurrent downloads
delete_zip = true # Delete zip files after extracting
interval = 20 # Interval between runs
retries = 10 # Number of times to retry a failed upload

[GRAPHQL]
backoff = 1.5 # Backoff factor for retries
interval = 4 # Interval between runs
retries = 3 # Number of times to retry a failed upload
"""


def ensure_token_and_config():
    """
    Ensure mps.ini exists on the local node (head & workers).
    NOTE: no longer touches auth_token; login is done via CLI username/password.
    """
    projectaria_dir = Path.home() / ".projectaria"
    projectaria_dir.mkdir(parents=True, exist_ok=True)

    ini_path = projectaria_dir / "mps.ini"
    ini_path.write_text(MPS_INI_CONTENT)


def prepare_tmp_on_raw():
    """Force temp files to /mnt/raw/.tmp to avoid cross-device rename/copy metadata issues."""
    tmp = RAW_ROOT / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ["PYTHONPRESERVEFILETIMES"] = "0"


def discover_input_dir() -> str:
    """Return the input directory: currently /mnt/raw (error if missing)."""
    if not RAW_ROOT.exists():
        raise RuntimeError(f"{RAW_ROOT} does not exist or is not mounted?")
    if not ARIA_DIR.exists() or not ARIA_DIR.is_dir():
        raise RuntimeError(f"{ARIA_DIR} does not exist or is not a directory")
    return str(ARIA_DIR)


def discover_vrs_files(input_dir: str) -> list[str]:
    """Find all .vrs files directly under input_dir."""
    p = Path(input_dir)
    vrs_files = sorted(str(f) for f in p.glob("*.vrs"))
    if not vrs_files:
        raise RuntimeError(f"No .vrs files found in {input_dir}")
    return vrs_files


def filter_unprocessed_vrs(vrs_files: list[str]) -> tuple[list[str], list[str]]:
    """
    Split vrs_files into:
      - to_process: VRS that do NOT yet have successful MPS output
      - already_done: VRS that already have mps_{stem}_vrs/hand_tracking and /slam

    A VRS is considered 'done' iff BOTH subfolders exist.
    """
    to_process: list[str] = []
    already_done: list[str] = []

    for f in vrs_files:
        p = Path(f)
        stem = p.stem  # e.g. '1760313742038' from '1760313742038.vrs'
        mps_dir = p.parent / f"mps_{stem}_vrs"
        hand_dir = mps_dir / "hand_tracking"
        slam_dir = mps_dir / "slam"

        if mps_dir.is_dir() and hand_dir.is_dir() and slam_dir.is_dir():
            already_done.append(f)
        else:
            to_process.append(f)

    return to_process, already_done


def chunk_list(seq, chunk_size):
    """Yield successive chunks of size chunk_size from seq."""
    for i in range(0, len(seq), chunk_size):
        yield seq[i : i + chunk_size]


@ray.remote
def run_mps_batches(vrs_files: list[str], chunk_size: int = 16) -> dict:
    ensure_token_and_config()
    prepare_tmp_on_raw()

    ts_start = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    batches = list(chunk_list(vrs_files, chunk_size))
    total_batches = len(batches)

    try:
        for idx, batch in enumerate(batches, start=1):
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Starting batch {idx}/{total_batches} with {len(batch)} files")

            # Build aria_mps command with username/password + batch inputs
            cmd = [
                "aria_mps",
                "single",
                "--username", "georgiat_p4u7t9",
                "--password", "georgiat0001",
            ]
            for f in batch:
                cmd.extend(["--input", f])
            cmd += ["--no-ui", "--retry-failed"]

            print("  CMD:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            results.append(
                {
                    "batch_index": idx,
                    "status": "ok",
                    "num_files": len(batch),
                    "files": batch,
                }
            )

        return {
            "status": "ok",
            "started_at": ts_start,
            "finished_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_files": len(vrs_files),
            "num_batches": total_batches,
            "batches": results,
        }

    except Exception as exc:
        return {
            "status": "err",
            "started_at": ts_start,
            "finished_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_files": len(vrs_files),
            "num_batches": total_batches,
            "err": str(exc),
            "trace": traceback.format_exc(limit=5),
            "partial_results": results,
        }


def launch_job(vrs_files: list[str], chunk_size: int = 16) -> None:
    print(f"Launching MPS on {len(vrs_files)} .vrs files in chunks of {chunk_size} …")
    fut = run_mps_batches.remote(vrs_files, chunk_size)
    ready, _ = ray.wait([fut], num_returns=1)
    res = ray.get(ready[0])
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if res["status"] == "ok":
        print(
            f"[{ts}] ✓ Completed {res['num_files']} files "
            f"in {res['num_batches']} batches"
        )
    else:
        print(f"[{ts}] ✗ Error while running MPS")
        print("  Error:", res.get("err", "unknown"))
        print(res.get("trace", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dry-run: list .vrs files in /mnt/raw and planned batches (no work)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of .vrs files per aria_mps invocation (default: 16)",
    )
    args = parser.parse_args()

    input_dir = discover_input_dir()
    all_vrs = discover_vrs_files(input_dir)
    vrs_files, already_done = filter_unprocessed_vrs(all_vrs)

    if args.debug:
        print("DEBUG MODE – will process .vrs files in batches (no work)")
        print(f"Input dir: {input_dir}")
        print(f"Found {len(all_vrs)} .vrs files total")
        print(f"  {len(already_done)} already have MPS outputs (hand_tracking + slam)")
        print(f"  {len(vrs_files)} remaining to process\n")

        if already_done:
            print("Already done:")
            for f in already_done:
                print(f"  [DONE] {f}")

        print("\nTo process:")
        for f in vrs_files:
            print(f"  [TODO] {f}")

        print("\nPlanned batches:")
        for i, batch in enumerate(chunk_list(vrs_files, args.chunk_size), start=1):
            print(f"  Batch {i}: {len(batch)} files")
            for bf in batch:
                print(f"    {bf}")
        sys.exit(0)

    if not vrs_files:
        print("No new .vrs files to process (all have MPS outputs). Exiting.")
        sys.exit(0)

    # Initialize Ray on single local node
    ray.init()

    # Ensure config on head node too and set tmp on head
    ensure_token_and_config()
    prepare_tmp_on_raw()

    # Launch MPS job on remaining (unprocessed) files only
    launch_job(vrs_files, args.chunk_size)


if __name__ == "__main__":
    main()
