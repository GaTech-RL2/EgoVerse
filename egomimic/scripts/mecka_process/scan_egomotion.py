"""Scan local episodes' ``obs_head_pose`` (Mecka egomotion) for failed tracking.

Writes a per-episode CSV and, with ``--blocklist``, the hashes whose status is
``constant`` (one pose for the whole episode: no tracking) or ``jump`` (a
single-frame head move over ``--jump-m``), in the format
``DatasetFilter(exclude_hashes=...)`` reads.

    python egomimic/scripts/mecka_process/scan_egomotion.py \\
        --dataset-dir $EGOVERSE_DATASET_DIR --csv scan.csv \\
        --blocklist egomimic/resources/blocklists/mecka_bad_egomotion.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import date
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import zarr

BAD = ("constant", "jump")
FIELDS = [
    "episode",
    "status",
    "embodiment",
    "frames",
    "unique_quat",
    "unique_xyz",
    "xyz_std",
    "max_step_m",
    "nonfinite",
]


def scan_episode(root: Path, episode: str, jump_m: float) -> dict:
    path = root / episode
    row = {"episode": episode}
    try:
        attrs = json.loads((path / "zarr.json").read_text()).get("attributes", {})
        row["embodiment"] = attrs.get("embodiment")
        if not (path / "obs_head_pose").is_dir():
            return {**row, "status": "no_head_pose"}
        pose = np.asarray(zarr.open(str(path), mode="r")["obs_head_pose"][:], float)
    except Exception as exc:
        return {**row, "status": f"error:{type(exc).__name__}"}
    xyz, quat = pose[:, :3], pose[:, 3:]
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(pose) > 1 else [0.0]
    row.update(
        frames=len(pose),
        unique_quat=len(np.unique(quat, axis=0)),
        unique_xyz=len(np.unique(xyz, axis=0)),
        xyz_std=float(xyz.std(0).max()),
        max_step_m=float(np.max(step)),
        nonfinite=int((~np.isfinite(pose)).any(1).sum()),
    )
    if row["unique_quat"] == 1 and row["unique_xyz"] == 1:
        status = "constant"
    elif row["unique_quat"] == 1:
        status = "constant_rot"
    elif row["unique_xyz"] == 1:
        status = "constant_xyz"
    elif row["nonfinite"]:
        status = "nonfinite"
    elif row["max_step_m"] > jump_m:
        status = "jump"
    else:
        status = "ok"
    return {**row, "status": status}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--blocklist", type=Path)
    # 30 cm in one frame is 9 m/s at 30 fps: a tracking jump, not a head move.
    ap.add_argument("--jump-m", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    episodes = sorted(
        d.name for d in args.dataset_dir.iterdir() if (d / "zarr.json").is_file()
    )
    scan = partial(scan_episode, args.dataset_dir, jump_m=args.jump_m)
    with Pool(args.workers) as pool:
        rows = sorted(
            pool.imap_unordered(scan, episodes, chunksize=32),
            key=lambda r: r["episode"],
        )

    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(Counter(r["status"] for r in rows))

    if args.blocklist:
        bad = [r for r in rows if r["status"] in BAD]
        header = [
            "# Episodes with failed head tracking (obs_head_pose / vendor egomotion),",
            "# from egomimic/scripts/mecka_process/scan_egomotion.py over the",
            f"# {len(rows)} episodes in the local dataset dir on {date.today()}.",
            "# constant: one pose for the whole episode; jump: a single-frame head",
            f"# move over {args.jump_m} m. {len(bad)} episodes.",
        ]
        lines = [f"{r['episode']}  # {r['status']}" for r in bad]
        args.blocklist.write_text("\n".join(header + lines) + "\n")


if __name__ == "__main__":
    main()
