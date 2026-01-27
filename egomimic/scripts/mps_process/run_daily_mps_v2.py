#!/usr/bin/env python3
"""
run_daily_mps.py – processes /mnt/raw/aria
  • normal call: launches one Ray job on /mnt/raw/aria
  • --debug: dry-run (lists the files in /mnt/raw/aria, no work)
  • Single-node Ray: spins up one local Ray node
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
ARIA_DIR = RAW_ROOT

TOKEN = (
    "FRLAbl7Eqw5g4upZCoow1ht3YE9e16ue9iLTv13IpnXXxxt8gR5BXrqkuj6deunEcnDAUMNjylAZAv"
    "SKzZB6PB2amlGFec8dyuvpaZAtZB0hxxzRRHgoy9gdZChM8lUDalGDP1q8VPoszMZBLiYoif0Q9aL49"
    "Ewn0mXUEVd3gHBW74yAwZD"
)

MPS_INI_CONTENT = """[DEFAULT]
log_dir = /tmp/logs/projectaria/mps/ # Path to log directory
status_check_interval = 30 # Status check interval in seconds
concurrent_processing = 25 # Maximum number of recordings being processed concurrently

[HASH]
concurrent_hashes = 4 # Maximum number of recordings whose hashes will be calculated concurrently
chunk_size = 10485760 # 10 * 2**20 (10MB)

[HEALTH_CHECK]
concurrent_health_checks = 4  # Maximum number of checks that can run concurrently

[ENCRYPTION]
chunk_size = 52428800 # 50 * 2**20 (50MB)
concurrent_encryptions = 5 # Maximum number of recordings that will be encrypted concurrently
delete_encrypted_files = true # Delete encrypted files after upload is done

[UPLOAD]
backoff = 1.5 # Backoff factor for retries
concurrent_uploads = 4 # Maximum number of concurrent uploads
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
    """Ensure ProjectAria token and mps.ini exist on the local node (head & workers)."""
    projectaria_dir = Path.home() / ".projectaria"
    projectaria_dir.mkdir(parents=True, exist_ok=True)

    token_path = projectaria_dir / "auth_token"
    if not token_path.exists():
        token_path.write_text(TOKEN)

    ini_path = projectaria_dir / "mps.ini"
    ini_path.write_text(MPS_INI_CONTENT)


def prepare_tmp_on_raw():
    """Force temp files to /mnt/raw/.tmp to avoid cross-device rename/copy metadata issues."""
    tmp = RAW_ROOT / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    # Best-effort: disable preserving file times in stdlib paths that honor it (Py 3.12+)
    os.environ["PYTHONPRESERVEFILETIMES"] = "0"


def discover_input_dir() -> str:
    """Return the single input directory: /mnt/raw/aria (error if missing)."""
    if not RAW_ROOT.exists():
        raise RuntimeError(f"{RAW_ROOT} does not exist or is not mounted?")
    if not ARIA_DIR.exists() or not ARIA_DIR.is_dir():
        raise RuntimeError(f"{ARIA_DIR} does not exist or is not a directory")
    return str(ARIA_DIR)


@ray.remote
def run_mps_on_dir(folder: str) -> dict:
    try:
        ensure_token_and_config()
        prepare_tmp_on_raw()
        subprocess.run(
            ["aria_mps", "single", "-i", folder, "--no-ui", "--retry-failed"],
            check=True,
        )
        return {"folder": folder, "status": "ok"}
    except Exception as exc:
        return {
            "folder": folder,
            "status": "err",
            "err": str(exc),
            "trace": traceback.format_exc(limit=3),
        }


def launch_job(input_dir: str) -> None:
    print(f"Launching MPS on {input_dir} …")
    fut = run_mps_on_dir.remote(input_dir)
    ready, _ = ray.wait([fut], num_returns=1)
    res = ray.get(ready[0])
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if res["status"] == "ok":
        print(f"[{ts}] ✓ {res['folder']}")
    else:
        print(f"[{ts}] ✗ {res['folder']} :: {res['err']}")
        print(res["trace"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Dry-run: list files in /mnt/raw/aria (no work)",
    )
    args = parser.parse_args()

    input_dir = discover_input_dir()

    if args.debug:
        print("DEBUG MODE – listing entries in /mnt/raw/aria (no work)")
        p = Path(input_dir)
        entries = sorted(p.iterdir())
        if not entries:
            print("  (empty)")
        else:
            for e in entries:
                kind = "file " if e.is_file() else "dir  "
                print(f"  {kind} {e}")
        sys.exit(0)

    # Initialize Ray on single local node
    ray.init()

    # Ensure config on head node too and set tmp on head (some tools use it)
    ensure_token_and_config()
    prepare_tmp_on_raw()

    # Launch MPS job
    launch_job(input_dir)


if __name__ == "__main__":
    main()