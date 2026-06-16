#!/usr/bin/env python3
"""
Pull raw annotation JSON from Scale tasks.

Outputs one JSON file per task: <output_dir>/<task_id>.json
Also writes a task_id → episode_hash mapping file for cross-referencing.

Usage:
    export SCALE_API_KEY="live_..."
    python pull_annotations.py                          # all 21 pick_place tasks
    python pull_annotations.py --tasks <ID1> <ID2>      # specific tasks
    python pull_annotations.py --csv tasks.csv           # CSV with TASK_ID column
    python pull_annotations.py -o ./my_annotations       # custom output dir
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import requests
from scaleapi import ScaleClient

PICK_PLACE_TASKS = {
    "69b479c61d4011ae649d9f85": "2026-03-03-05-03-57-023000",
    "69b479c74ecfb2743328a365": "2026-03-03-05-11-26-461000",
    "69b479c87eef031ee4090676": "2026-03-03-05-15-53-096000",
    "69b479c90b09e4834244e43a": "2026-03-03-05-20-29-363000",
    "69b479ca5342ed23630722da": "2026-03-03-05-27-14-876000",
    "69b479cb2feda60369f3d6a4": "2026-03-03-05-29-34-981000",
    "69b479cb3edefa8a2eb5ce7d": "2026-03-03-05-32-38-565000",
    "69b479ccf967765c339baaac": "2026-03-03-05-35-22-081000",
    "69b479cdafcb625912ed442c": "2026-03-03-05-42-25-155000",
    "69b479ce1d4011ae649d9fec": "2026-03-03-05-47-21-437000",
    "69b479cfc196f6ada95bfa01": "2026-03-03-05-50-12-790000",
    "69b479d05e2160cca458e58f": "2026-03-03-05-52-32-417000",
    "69b479d1504fd86b62536c1f": "2026-03-03-06-37-47-140000",
    "69b479d260c2267553ebe05b": "2026-03-03-06-40-10-619000",
    "69b479d2ea069c7c26588ed3": "2026-03-03-06-43-14-995000",
    "69b479d3efbfbb8fe3ca9055": "2026-03-03-06-48-23-857000",
    "69b479d45e2160cca458e5e7": "2026-03-03-06-50-42-588000",
    "69b479d5ea069c7c26588f2a": "2026-03-03-06-53-17-768000",
    "69b479d61390762cd40c5c2f": "2026-03-03-06-56-07-657000",
    "69b479d70b09e4834244e54d": "2026-03-03-06-58-23-150000",
    "69b479d76eeb711b23ba5e41": "2026-03-03-07-01-06-642000",
}

REQUEST_TIMEOUT_S = 60


def fetch_raw_annotations(client: ScaleClient, task_id: str) -> dict:
    """Fetch the raw annotation JSON for a Scale task, untouched."""
    task = client.get_task(task_id)
    url = task.response["annotations"]["url"]
    resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return json.loads(resp.text.rstrip("\x00"))


def resolve_task_ids(args: argparse.Namespace) -> list[str]:
    if args.csv:
        with open(args.csv, newline="") as f:
            reader = csv.DictReader(f)
            fields = {(n or "").strip(): n for n in (reader.fieldnames or [])}
            col = fields.get("TASK_ID") or fields.get("task_id")
            if not col:
                sys.exit(f"CSV must have TASK_ID or task_id column: {args.csv}")
            return list(
                dict.fromkeys(r[col].strip() for r in reader if r.get(col, "").strip())
            )
    if args.tasks:
        return list(dict.fromkeys(args.tasks))
    return list(PICK_PLACE_TASKS.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Pull raw Scale annotation JSON per task"
    )
    parser.add_argument("--tasks", nargs="+", help="Scale task IDs")
    parser.add_argument("--csv", help="CSV with TASK_ID or task_id column")
    parser.add_argument(
        "-o", "--output-dir", default="annotations", help="Output directory"
    )
    parser.add_argument("-s", "--api-key", default=os.environ.get("SCALE_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Set SCALE_API_KEY or pass --api-key")

    task_ids = resolve_task_ids(args)
    if not task_ids:
        sys.exit("No task IDs provided.")

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    client = ScaleClient(args.api_key)

    for i, tid in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {tid} ...", end=" ", flush=True)
        try:
            raw = fetch_raw_annotations(client, tid)
            path = os.path.join(out, f"{tid}.json")
            with open(path, "w") as f:
                json.dump(raw, f, indent=2)
            n_ann = len(raw.get("annotations", []))
            print(f"{n_ann} annotations -> {path}")
        except Exception as e:
            print(f"FAILED: {e}")

    # Write the task_id -> episode_hash mapping so they can cross-reference
    mapping_path = os.path.join(out, "task_id_to_episode_hash.json")
    mapping = {tid: PICK_PLACE_TASKS.get(tid, "") for tid in task_ids}
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nMapping: {mapping_path}")


if __name__ == "__main__":
    main()
