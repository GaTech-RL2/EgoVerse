#!/usr/bin/env python
"""
Print the language annotations (and the frames they span) for the episode(s)
that a run's validation video was rendered on.

The validation video buffers frames from the *valid* split of a single
``data=`` filter (here: ``lab == 'mecka' and task == 'folding_clothes'``),
capped at ``max_frames_per_task * videos_per_task`` frames. This script
reproduces that exact selection:

  1. apply the run's SQL filter to ``app.episodes`` (lab + task),
  2. take the deterministic valid split (seed 42, valid_ratio 0.2) via
     ``split_dataset_names`` -- the same call the dataloader makes,
  3. for each valid episode, read the zarr ``annotations`` array and print
     each span's text and ``[start_idx, end_idx)`` frame range.

Defaults match logs/.../mecka_pi_fold_clothes_freeform_2026-06-01_19-23-33.
Override with --lab / --task / --dataset-dir if you point it at another run.
"""

import argparse
import json
import os

import numpy as np
import zarr

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import (
    S3EpisodeResolver,
    split_dataset_names,
)
from egomimic.utils.aws.aws_data_utils import load_env

# Defaults pulled from the run's .hydra/config.yaml (data.*.filters +
# evaluator settings). Change these if you target a different run.
DEFAULT_DATASET_DIR = "/storage/project/r-dxu345-0/shared/egoverseS3ZarrDatasets"
DEFAULT_LAB = "mecka"
DEFAULT_TASK = "folding_clothes"
VALID_RATIO = 0.2  # MultiDataset default; the config does not override it
# evaluator.max_frames_per_task * evaluator.videos_per_task from the run config.
DEFAULT_VIDEO_FRAME_CAP = 1000 * 1


def decode_entry(value):
    """Decode one raw annotations[] element into a dict (mirrors
    ZarrDataset._decode_json_entry)."""
    if isinstance(value, np.void):
        value = value.item()
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lab", default=DEFAULT_LAB)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument(
        "--all-splits",
        action="store_true",
        help="Print every matching episode, not just the ones the video shows.",
    )
    ap.add_argument(
        "--valid-split",
        action="store_true",
        help="Print all 92 valid-split episodes (not just those in the video).",
    )
    ap.add_argument(
        "--frame-cap",
        type=int,
        default=DEFAULT_VIDEO_FRAME_CAP,
        help="max_frames_per_task * videos_per_task; the video's frame budget.",
    )
    args = ap.parse_args()

    load_env()  # AWS/SQL credentials, same as trainHydra.py

    filt = DatasetFilter(
        filter_lambdas=[
            f"lambda row: row['lab'] == '{args.lab}' "
            f"and row['task'] == '{args.task}'"
        ]
    )

    # SQL-only lookup: returns [(zarr_processed_path, episode_hash), ...].
    # No download is triggered here.
    paths, hash_to_task = S3EpisodeResolver._get_filtered_paths(filt)
    all_hashes = sorted(h for _, h in paths)
    if not all_hashes:
        print(f"No episodes matched lab={args.lab!r} task={args.task!r}.")
        return

    train_set, valid_set = split_dataset_names(all_hashes, valid_ratio=VALID_RATIO)
    # The val loader is shuffle=False and episodes load in sorted(iterdir())
    # order, i.e. sorted episode_hash. The viz renders one frame per validation
    # sample in dataset order (one sample == one frame index), so the video is
    # the valid episodes in this order, played frame-by-frame, truncated at the
    # frame cap. Walk the sorted valid hashes accumulating frames to find which
    # episode(s) the video actually shows.
    sorted_valid = sorted(valid_set)
    coverage = {}  # hash -> frames shown in video [0, shown)
    remaining = args.frame_cap
    for h in sorted_valid:
        if remaining <= 0:
            break
        ep_path = os.path.join(args.dataset_dir, h)
        if not os.path.isdir(ep_path):
            # Episode not cached locally: we can't read its length to advance
            # the cumulative offset. Flag it and stop -- ordering past this
            # point is unknown.
            coverage[h] = None
            break
        total = int(dict(zarr.open_group(ep_path, mode="r").attrs)["total_frames"])
        shown = min(total, remaining)
        coverage[h] = shown
        remaining -= shown

    if args.all_splits:
        chosen = all_hashes
    elif args.valid_split:
        chosen = sorted_valid
    else:
        chosen = list(coverage.keys())  # only episodes shown in the video

    print(
        f"Filter: lab={args.lab!r} task={args.task!r}\n"
        f"Matched {len(all_hashes)} episode(s); "
        f"{len(valid_set)} in valid split (seed 42, ratio {VALID_RATIO}).\n"
        f"Validation video frame budget: {args.frame_cap} frames "
        f"(shuffle=False, sorted-hash order, one frame per frame-index).\n"
        f"=> Video covers {len(coverage)} episode(s); "
        f"annotations below.\n"
    )

    for h in chosen:
        split = "valid" if h in valid_set else "train"
        shown = coverage.get(h)  # frames shown in video, or None
        ep_path = os.path.join(args.dataset_dir, h)
        print("=" * 78)
        tag = (
            f"  [shows frames 0-{shown - 1} in video]"
            if shown
            else (
                "  [in valid split, NOT reached by video]" if split == "valid" else ""
            )
        )
        print(f"episode_hash: {h}   [{split} split]{tag}")
        if not os.path.isdir(ep_path):
            print(f"  (not in local cache: {ep_path})")
            continue
        g = zarr.open_group(ep_path, mode="r")
        attrs = dict(g.attrs)
        total = attrs.get("total_frames", "?")
        print(
            f"  embodiment={attrs.get('embodiment')!r}  "
            f"task_name={attrs.get('task_name')!r}  total_frames={total}"
        )
        if "annotations" not in g:
            print("  no 'annotations' array in store.")
            continue
        raw = g["annotations"][:]
        anns = [decode_entry(x) for x in raw]
        anns = [a for a in anns if isinstance(a, dict)]
        if not anns:
            print("  annotations array is empty (0 spans).")
            continue
        anns.sort(key=lambda a: int(a.get("start_idx", -1)))
        print(f"  {len(anns)} annotation span(s):")
        for a in anns:
            s, e = int(a.get("start_idx", -1)), int(a.get("end_idx", -1))
            # Flag whether this span is visible within the rendered window.
            if shown is None:
                vis = ""
            elif s >= shown:
                vis = "   (not shown)"
            elif e > shown:
                vis = f"   (partly shown: up to frame {shown - 1})"
            else:
                vis = "   (shown)"
            print(
                f"    frames [{s:>5} , {e:>5})  ({e - s:>5} frames)  "
                f"{a.get('text', '')!r}{vis}"
            )
    print("=" * 78)


if __name__ == "__main__":
    main()
