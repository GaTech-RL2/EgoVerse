#!/usr/bin/env python3
"""
run_daily_mps.py – processes /mnt/raw
  • normal call: launches aria_mps on /mnt/raw
  • --debug: dry-run (lists the files in /mnt/raw, no work)

Uses username/password (no auth_token).
"""

import argparse
import datetime as dt
import os
import subprocess
import sys
import traceback
from pathlib import Path

RAW_ROOT = Path("/mnt/raw")
ARIA_DIR = RAW_ROOT

USERNAME = "georgiat_p4u7t9"
PASSWORD = "georgiat0001"

MPS_INI_CONTENT = """[DEFAULT]
log_dir = /tmp/logs/projectaria/mps/ # Path to log directory
status_check_interval = 30 # Status check interval in seconds
concurrent_processing = 10 # Maximum number of recordings being processed concurrently

[HASH]
concurrent_hashes = 5
chunk_size = 10485760

[HEALTH_CHECK]
concurrent_health_checks = 4

[ENCRYPTION]
chunk_size = 52428800
concurrent_encryptions = 5
delete_encrypted_files = true

[UPLOAD]
backoff = 1.5
concurrent_uploads = 4
interval = 20
max_chunk_size = 104857600
min_chunk_size = 5242880
retries = 10
smoothing_window_size = 10
target_chunk_upload_secs = 3

[DOWNLOAD]
backoff = 1.5
chunk_size = 10485760
concurrent_downloads = 10
delete_zip = true
interval = 20
retries = 10

[GRAPHQL]
backoff = 1.5
interval = 4
retries = 3
"""


def ensure_config():
    """Ensure mps.ini exists (no auth_token used)."""
    projectaria_dir = Path.home() / ".projectaria"
    projectaria_dir.mkdir(parents=True, exist_ok=True)

    ini_path = projectaria_dir / "mps.ini"
    ini_path.write_text(MPS_INI_CONTENT)


def prepare_tmp_on_raw():
    """Force temp files to /mnt/raw/.tmp to avoid cross-device rename issues."""
    tmp = RAW_ROOT / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)
    os.environ["PYTHONPRESERVEFILETIMES"] = "0"


def discover_input_dir() -> str:
    if not RAW_ROOT.exists():
        raise RuntimeError(f"{RAW_ROOT} does not exist or is not mounted?")
    if not ARIA_DIR.exists() or not ARIA_DIR.is_dir():
        raise RuntimeError(f"{ARIA_DIR} does not exist or is not a directory")
    return str(ARIA_DIR)


def run_mps_on_dir(folder: str) -> dict:
    try:
        ensure_config()
        prepare_tmp_on_raw()

        cmd = [
            "aria_mps",
            "single",
            "--username",
            USERNAME,
            "--password",
            PASSWORD,
            "-i",
            folder,
            "--no-ui",
            "--retry-failed",
        ]

        print("CMD:", " ".join(cmd))
        subprocess.run(cmd, check=True)

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
    res = run_mps_on_dir(input_dir)
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
        help="Dry-run: list files in /mnt/raw (no work)",
    )
    args = parser.parse_args()

    input_dir = discover_input_dir()

    if args.debug:
        print("DEBUG MODE – listing entries in /mnt/raw")
        for e in sorted(Path(input_dir).iterdir()):
            kind = "file " if e.is_file() else "dir  "
            print(f"  {kind} {e}")
        sys.exit(0)

    ensure_config()
    prepare_tmp_on_raw()
    launch_job(input_dir)


if __name__ == "__main__":
    main()
