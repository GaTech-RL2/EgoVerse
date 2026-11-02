#!/usr/bin/env python3
"""Verify the selected destination data and local resolver stage, without training.

--refresh-remote lists object names/sizes in the explicit destination bucket;
it never deletes or uploads objects. --prepare-stage writes an s5cmd download
batch for available selected episodes, with missing episodes reported separately.
Readiness stays blocked until all selected episodes pass local array, image,
metadata, chunk-object and numeric wrist-convention checks. Run the full local
check on a CPU allocation; only a few frames per episode are decoded.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from canonicalize import canonicalize_episode, classify
from launch_pace import BUDGETS, OPERATORS, digest, now, read_json, write_json

ARRAYS = (
    "left.obs_keypoints",
    "right.obs_keypoints",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "obs_head_pose",
    "images.front_1",
    "annotations",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selection(private):
    design = read_json(private / "moving_opscale_design.json")
    ids = set()
    for b in BUDGETS:
        for n in OPERATORS:
            split = read_json(private / "splits" / f"fold_moving_b{b}_op{n}.json")
            for key in ("train", "seen_matched", "seen_perlevel", "unseen"):
                ids.update(split[key])
    require(
        ids <= design["episodes"].keys(),
        "Selected episode missing from design metadata",
    )
    return design, ids


def converted_frame_index(private, design, selected):
    """Bind exact converter output lengths, rather than rounded video clocks.

    Conversion takes the shortest synchronized stream. Valid tracking may end
    before the video timestamp duration, and source frame rates round to 30.
    The converter-produced CSV supplies the actual retained frames; the two
    repaired episodes have a separately validated recovery receipt.
    """
    source = private / "inputs/main_episodes.csv"
    counts = {}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            eid = row["episode_id"]
            if eid not in selected:
                continue
            expected = design["episodes"][eid]
            require(
                eid not in counts, "Duplicate selected episode in converted frame index"
            )
            require(
                row["user_id"] == expected["user_id"]
                and row["task_id"] == "folding_clothes",
                "Converted frame index identity differs",
            )
            require(
                abs(float(row["duration_s"]) - expected["duration"]) <= 1 / 30 + 0.001,
                "Converted frame index duration differs from manifest",
            )
            counts[eid] = int(row["frames"])
    provenance = {"main_episodes_sha256": digest(source)}
    recovery = private / "recovery2/report.json"
    if recovery.exists():
        report = read_json(recovery)
        for row in report["episodes"]:
            eid = row["episode_id"]
            if eid not in selected:
                continue
            require(
                row.get("validated") is True, "Recovery frame count is not validated"
            )
            require(
                eid not in counts, "Recovery episode duplicates converted frame index"
            )
            counts[eid] = int(row["frames"])
        provenance["recovery_report_sha256"] = digest(recovery)
    require(set(counts) == selected, "Missing authoritative converted frame counts")
    require(all(n >= 30 for n in counts.values()), "Invalid converted frame count")
    provenance["selected_frames"] = sum(counts.values())
    return counts, provenance


def check_timing(attrs, expected):
    frames = int(attrs["total_frames"])
    require(frames >= 30, "Episode has fewer than 30 real frames")
    require(
        abs(float(attrs["duration"]) - expected["duration"]) <= 1 / 30 + 0.001,
        "Converted duration and selected manifest disagree by more than one frame",
    )
    require(
        frames == expected["converted_frames"],
        "Frame count differs from authoritative converted frame index",
    )
    return frames


def remote_inventory(prefix, selected, design_sha, existing=None):
    import boto3
    from botocore.config import Config

    uri = urlparse(prefix)
    require(
        uri.scheme == "s3" and uri.netloc and uri.path.strip("/"),
        "Explicit s3://bucket/prefix required",
    )
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )
    base = uri.path.strip("/") + "/"
    found = dict(existing["episodes"]) if existing else {eid: {} for eid in selected}
    prefixes = (
        [base + eid + ".zarr/" for eid in existing["missing"]] if existing else [base]
    )
    if existing:
        for eid in existing["missing"]:
            found[eid] = {}
    pages = itertools.chain.from_iterable(
        client.get_paginator("list_objects_v2").paginate(
            Bucket=uri.netloc, Prefix=prefix
        )
        for prefix in prefixes
    )
    for page_number, page in enumerate(pages, 1):
        if page_number % 50 == 0:
            print(
                json.dumps(
                    {
                        "inventory_pages": page_number,
                        "selected_episode_prefixes_seen": sum(
                            bool(v) for v in found.values()
                        ),
                    }
                ),
                flush=True,
            )
        for obj in page.get("Contents", []):
            relative = obj["Key"][len(base) :]
            directory, separator, key = relative.partition("/")
            eid = directory.removesuffix(".zarr")
            if separator and directory.endswith(".zarr") and eid in found:
                found[eid][key] = {
                    "size": obj["Size"],
                    "etag": obj.get("ETag", "").strip('"'),
                }
    needed = {"zarr.json"} | {key + "/zarr.json" for key in ARRAYS}
    missing = {
        eid: sorted(needed - objects.keys())
        for eid, objects in found.items()
        if not needed <= objects.keys()
    }
    return {
        "schema_version": 1,
        "checked_at": now(),
        "s3_prefix": prefix.rstrip("/") + "/",
        "design_sha256": design_sha,
        "episodes": found,
        "missing": missing,
        "objects": sum(map(len, found.values())),
        "bytes": sum(v["size"] for objects in found.values() for v in objects.values()),
    }


def chunk_keys(meta):
    shape = meta["shape"]
    grid = meta["chunk_grid"]
    require(grid["name"] == "regular", "Unsupported Zarr chunk grid")
    chunks = grid["configuration"]["chunk_shape"]
    enc = meta.get("chunk_key_encoding", {"name": "default"})
    separator = enc.get("configuration", {}).get("separator", "/")
    require(enc["name"] in ("default", "v2"), "Unsupported chunk-key encoding")
    for index in itertools.product(
        *(range(math.ceil(n / c)) for n, c in zip(shape, chunks))
    ):
        parts = list(map(str, index))
        if enc["name"] == "default":
            parts.insert(0, "c")
        yield separator.join(parts)


def inspect_episode(item):
    eid, expected, stage, inventory = item
    path = stage / f"{eid}.zarr"
    result = {"episode_id": eid, "passed": False}
    try:
        require(path.is_dir(), "Local episode directory is missing")
        remote = inventory["episodes"].get(eid, {})
        require(
            eid not in inventory["missing"] and remote,
            "Destination episode is missing/incomplete",
        )
        for key, meta in remote.items():
            local = path / key
            require(
                local.is_file() and local.stat().st_size == meta["size"],
                f"Object missing/size differs: {key}",
            )
        group = read_json(path / "zarr.json")
        require(
            group["zarr_format"] == 3 and group["node_type"] == "group",
            "Expected Zarr v3 episode",
        )
        attrs = group["attributes"]
        require(attrs.get("episode_id") == eid, "Episode identity differs")
        require(
            attrs.get("user_id") == expected["user_id"], "Operator identity differs"
        )
        require(attrs.get("task_id") == "folding_clothes", "Wrong task")
        require(attrs.get("embodiment") == "human_bimanual", "Wrong embodiment")
        require(float(attrs.get("fps", 0)) == 30, "Reference requires 30 fps")
        frames = check_timing(attrs, expected)
        for key in ARRAYS:
            meta = read_json(path / key / "zarr.json")
            require(meta["node_type"] == "array", f"{key} is not an array")
            shape = meta["shape"]
            require(
                shape and shape[0] >= (1 if key == "annotations" else frames),
                f"{key} is too short",
            )
            if key.endswith("obs_keypoints"):
                require(shape[1:] == [63], f"{key} is not 21x3 flattened MANO")
            elif key.endswith("pose"):
                require(shape[1:] == [7], f"{key} is not xyz+wxyz")
            for chunk in chunk_keys(meta):
                require(
                    (path / key / chunk).is_file(), f"Missing shard/chunk {key}/{chunk}"
                )
        import cv2
        import numpy as np
        import zarr

        from egomimic.scripts.mecka_process.mecka_to_zarr import (
            compute_hand_pose_xyzquat,
        )

        arrays = zarr.open_group(str(path), mode="r")
        probes = sorted({0, frames // 2, frames - 1})
        for index in probes:
            image = arrays["images.front_1"][index : index + 1][0]
            decoded = cv2.imdecode(
                np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            require(decoded is not None and decoded.ndim == 3, "JPEG decode failed")
            require(
                np.isfinite(arrays["obs_head_pose"][index]).all(), "Nonfinite head pose"
            )
            for side in ("left", "right"):
                kp = np.asarray(arrays[f"{side}.obs_keypoints"][index]).reshape(21, 3)
                actual = np.asarray(arrays[f"{side}.obs_wrist_pose"][index])
                require(
                    np.isfinite(kp).all() and np.isfinite(actual).all(),
                    "Nonfinite keypoints/wrist",
                )
                # The converter preserves zero-hand frames; their identity pose
                # is valid. The established orientation classifier skips these.
                _, reconstructed = compute_hand_pose_xyzquat(kp, 0)
                require(
                    np.allclose(actual[:3], reconstructed[:3], atol=1e-5),
                    "Wrist position convention differs",
                )
                require(
                    abs(float(np.linalg.norm(actual[3:])) - 1) <= 1e-4,
                    "Non-unit wrist quaternion",
                )
        # Zarr's variable-length-byte scalar indexing returns a wrapper. The
        # slice-then-index path is also the dataset loader's JPEG read path.
        annotations = bytes(arrays["annotations"][0:1][0])
        require(bool(json.loads(annotations)), "Empty annotation record")
        convention = classify(arrays, frames)
        require(convention in ("pre_fix", "post_fix"), "Unknown raw wrist convention")
        record = canonicalize_episode(
            path, stage.parent / "canonical_zarr", frames, convention
        )
        canonical_meta = read_json(
            stage.parent / "canonical_zarr" / path.name / "zarr.json"
        )["attributes"]
        result.update(
            passed=True,
            frames=frames,
            metadata=canonical_meta,
            source_wrist_convention=convention,
            canonical_wrist_convention="post_fix",
            canonical_root_sha256=record["canonical_root_sha256"],
            frame_clock_minus_duration_s=frames / 30 - float(attrs["duration"]),
        )
    except Exception as error:
        # Messages are locally constructed; strip other library error text to avoid URLs/secrets.
        result["error"] = (
            str(error) if isinstance(error, ValueError) else type(error).__name__
        )
        result["error_location"] = [
            {
                "file": Path(frame.filename).name,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(error.__traceback__)[-3:]
        ]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--s3-prefix", default="s3://private-mecka/private_opscale/")
    p.add_argument("--refresh-remote", action="store_true")
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument(
        "--refresh-missing",
        action="store_true",
        help="Re-list only previously missing destination episodes",
    )
    p.add_argument(
        "--stage-missing-only",
        action="store_true",
        help="Prepare commands only for previously missing episodes",
    )
    p.add_argument(
        "--prepare-stage", type=Path, help="Write s5cmd commands for available episodes"
    )
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    require(1 <= args.workers <= 16, "Use between 1 and 16 local verification workers")
    private = args.private_root.resolve()
    design, selected = selection(private)
    frame_counts, frame_provenance = converted_frame_index(private, design, selected)
    for eid in selected:
        design["episodes"][eid] = dict(
            design["episodes"][eid], converted_frames=frame_counts[eid]
        )
    design_sha = digest(private / "moving_opscale_design.json")
    report = {
        "schema_version": 1,
        "status": "blocked",
        "checked_at": now(),
        "design_sha256": design_sha,
        "selected_episodes": len(selected),
        "checks": {
            "transfer_complete": False,
            "selected_episode_objects": False,
            "required_arrays": False,
            "episode_index": False,
            "canonical_view": False,
        },
        "converted_frame_index": frame_provenance,
        "canonical_root": str(private / "canonical_zarr"),
    }
    write_json(args.output, report)
    previous = read_json(args.inventory) if args.inventory.exists() else None
    stage_ids = (
        set(previous["missing"]) if args.stage_missing_only and previous else selected
    )
    require(
        not args.stage_missing_only or previous is not None,
        "Missing-only staging requires prior inventory",
    )
    if args.refresh_remote or args.refresh_missing:
        require(
            not args.refresh_missing or previous is not None,
            "Missing-only refresh requires prior inventory",
        )
        inventory = remote_inventory(
            args.s3_prefix,
            selected,
            design_sha,
            previous if args.refresh_missing else None,
        )
        write_json(args.inventory, inventory)
    else:
        inventory = read_json(args.inventory)
    require(
        inventory["design_sha256"] == design_sha, "Inventory belongs to another design"
    )
    require(
        set(inventory["episodes"]) == selected,
        "Inventory selected episode union differs",
    )
    require(
        inventory["s3_prefix"].rstrip("/") == args.s3_prefix.rstrip("/"),
        "Inventory source differs",
    )
    report.update(
        inventory_sha256=digest(args.inventory),
        s3_prefix=inventory["s3_prefix"],
        destination_bytes=inventory["bytes"],
        destination_objects=inventory["objects"],
        missing_destination_episodes=sorted(inventory["missing"]),
    )
    report["checks"]["selected_episode_objects"] = not inventory["missing"]
    stage = private / "zarr"
    stage.mkdir(exist_ok=True)
    report["free_disk_bytes"] = shutil.disk_usage(stage).free
    if args.prepare_stage:
        missing_bytes = sum(
            meta["size"]
            for eid, entries in inventory["episodes"].items()
            for key, meta in entries.items()
            if not (stage / f"{eid}.zarr" / key).exists()
        )
        require(
            shutil.disk_usage(stage).free >= missing_bytes * 1.2,
            "Insufficient free disk for selected downloads +20% headroom",
        )
        priority_split = read_json(private / "splits/fold_moving_b24_op4.json")
        priority = set().union(
            *(
                priority_split[k]
                for k in ("train", "seen_matched", "seen_perlevel", "unseen")
            )
        )
        ordered = sorted(stage_ids, key=lambda eid: (eid not in priority, eid))
        commands = [
            f'sync "{inventory["s3_prefix"]}{eid}.zarr/*" "{stage}/{eid}.zarr/"'
            for eid in ordered
            if eid not in inventory["missing"]
        ]
        args.prepare_stage.parent.mkdir(parents=True, exist_ok=True)
        args.prepare_stage.write_text("\n".join(commands) + "\n")
        os.chmod(args.prepare_stage, 0o600)
        report["staging_commands"] = len(commands)
        report["missing_download_bytes"] = missing_bytes
    if not args.inventory_only:
        items = [
            (eid, design["episodes"][eid], stage, inventory) for eid in sorted(selected)
        ]
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(inspect_episode, items):
                results.append(result)
                if len(results) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "checked": len(results),
                                "total": len(selected),
                                "failed": sum(not r["passed"] for r in results),
                            }
                        ),
                        flush=True,
                    )
        failures = [
            {k: v for k, v in r.items() if k != "metadata"}
            for r in results
            if not r["passed"]
        ]
        report["failures"] = failures
        report["verified_episodes"] = len(results) - len(failures)
        complete = len(results) == len(selected) and not failures
        report["checks"]["required_arrays"] = complete
        report["checks"]["transfer_complete"] = complete and not inventory["missing"]
        if complete:
            # Build the exact local resolver index only from verified episode attrs.
            index = {r["episode_id"]: r["metadata"] for r in results}
            canonical = private / "canonical_zarr"
            write_json(canonical / "_episode_index.json", index)
            report["checks"]["episode_index"] = (
                set(read_json(canonical / "_episode_index.json")) == selected
            )
            report["checks"]["canonical_view"] = all(
                r["canonical_wrist_convention"] == "post_fix" for r in results
            )
            report["canonical_index_sha256"] = digest(canonical / "_episode_index.json")
            report["source_wrist_conventions"] = {
                key: sum(r["source_wrist_convention"] == key for r in results)
                for key in ("pre_fix", "post_fix")
            }
            report["max_frame_clock_duration_difference_s"] = max(
                abs(r["frame_clock_minus_duration_s"]) for r in results
            )
            report["converted_training_hours"] = {}
            for b in BUDGETS:
                for n in OPERATORS:
                    name = f"fold_moving_b{b}_op{n}"
                    train = read_json(private / "splits" / (name + ".json"))["train"]
                    report["converted_training_hours"][name] = (
                        sum(frame_counts[eid] for eid in train) / 30 / 3600
                    )
        report["status"] = "passed" if all(report["checks"].values()) else "blocked"
    report["checked_at"] = now()
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    return 0 if report["status"] == "passed" or args.inventory_only else 2


if __name__ == "__main__":
    raise SystemExit(main())
