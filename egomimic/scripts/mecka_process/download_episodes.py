"""
Download a list of episodes from S3/R2 to a local directory using s5cmd.

Input file format (one entry per line):
    <episode_hash> <s3_path>

Example:
    696da439fd6a4da2c4f27354 s3://rldb/processed_v3/mecka/freeform/696da439fd6a4da2c4f27354.zarr
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from egomimic.utils.aws.aws_data_utils import load_env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download episodes from S3/R2 to a local directory."
    )
    parser.add_argument(
        "--list",
        required=True,
        type=str,
        help="Path to text file with lines: <episode_hash> <s3_path>",
    )
    parser.add_argument(
        "--dest",
        required=True,
        type=str,
        help="Local destination directory.",
    )
    parser.add_argument(
        "--numworkers",
        default=128,
        type=int,
        help="s5cmd parallel workers (default 128).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip episodes whose destination dir already exists (default true).",
    )
    args = parser.parse_args()

    list_path = Path(args.list)
    if not list_path.is_file():
        print(f"Error: list file not found: {list_path}")
        return 2

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Parse list file
    entries: list[tuple[str, str]] = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            print(f"Warning: skipping malformed line: {line!r}")
            continue
        entries.append((parts[0], parts[1]))

    if not entries:
        print("No entries found in list file.")
        return 2

    # Filter already-downloaded episodes
    to_download = []
    skipped = 0
    for eh, s3_path in entries:
        local = dest / eh
        if args.skip_existing and local.exists():
            skipped += 1
            continue
        to_download.append((eh, s3_path))

    print(f"Total entries : {len(entries)}")
    print(f"Already local : {skipped}")
    print(f"To download   : {len(to_download)}")

    if not to_download:
        print("Nothing to download.")
        return 0

    # Load R2 credentials
    load_env()
    endpoint_url = os.environ.get("R2_ENDPOINT_URL")
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["R2_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["R2_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = "auto"
    os.environ["AWS_REGION"] = "auto"

    # Build s5cmd batch file  (sync <s3_prefix>/* <local_dest>/)
    lines = []
    for eh, s3_path in to_download:
        src = s3_path.rstrip("/") + "/*"
        dst = str(dest / eh) + "/"
        lines.append(f'sync "{src}" "{dst}"')

    with tempfile.NamedTemporaryFile(
        mode="w", prefix="_s5cmd_dl_", suffix=".txt", delete=False
    ) as tmp:
        tmp.write("\n".join(lines) + "\n")
        batch_path = tmp.name

    try:
        cmd = [
            "s5cmd",
            "--endpoint-url", endpoint_url,
            "--numworkers", str(args.numworkers),
            "run", batch_path,
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode
    finally:
        os.unlink(batch_path)


if __name__ == "__main__":
    raise SystemExit(main())
