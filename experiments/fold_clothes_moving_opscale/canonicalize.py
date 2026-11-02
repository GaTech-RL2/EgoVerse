"""Immutable local compatibility view of mixed Mecka wrist conventions.

Images and all unchanged arrays remain links to the verified raw download. Only
pre-fix left wrist arrays are copied and transformed, using the same Rz(180)
right multiplication as Human's RotateLocalFrame transform. Provenance embedded
in the copied root metadata and a separate receipt make retries idempotent.
The source and destination bucket objects are never modified.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from launch_pace import digest, read_json, write_json

VERSION = "left-wrist-local-rz180-v1"
LEFT = "left.obs_wrist_pose"
POSE_KEYS = ("left.obs_keypoints", LEFT, "right.obs_keypoints", "right.obs_wrist_pose")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tree_digest(path):
    """Hash relative filenames and file bytes, including array codec metadata."""
    path = Path(path)
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(file.relative_to(path).as_posix().encode() + b"\0")
        h.update(bytes.fromhex(digest(file)))
    return h.hexdigest()


def sample_poses(group, frames):
    import numpy as np

    indices = sorted(
        set(np.rint(np.linspace(0, frames - 1, min(25, frames))).astype(int))
    )
    return {
        key: np.asarray(group[key].oindex[indices], dtype=np.float64)
        for key in POSE_KEYS
    }


def classify(group, frames):
    # Use the converter's established right-hand control and 10-degree median
    # criterion. Sample across the real episode, excluding padded tail frames.
    from egomimic.scripts.mecka_process.mecka_to_zarr import left_wrist_convention

    sample = sample_poses(group, frames)
    return left_wrist_convention(sample, frames=len(sample[LEFT]), tol_deg=10.0)


def canonicalize_episode(raw, canonical_root, frames, convention=None):
    import numpy as np
    import zarr

    from egomimic.rldb.zarr.action_chunk_transforms import RotateLocalFrame

    raw, canonical_root = Path(raw).resolve(), Path(canonical_root).resolve()
    require(raw.parent != canonical_root, "Raw and canonical roots must differ")
    group = zarr.open_group(str(raw), mode="r")
    observed = classify(group, frames)
    require(observed in ("pre_fix", "post_fix"), "Unknown raw wrist convention")
    require(
        convention is None or convention == observed, "Raw wrist classification changed"
    )
    root_meta = read_json(raw / "zarr.json")
    require(
        "moving_opscale_canonicalization" not in root_meta["attributes"],
        "Raw source is already a canonical view",
    )
    source = {
        "version": VERSION,
        "source_path": str(raw),
        "source_convention": observed,
        "target_convention": "post_fix",
        "frames": frames,
        "raw_root_sha256": digest(raw / "zarr.json"),
        "raw_left_sha256": tree_digest(raw / LEFT),
        "operation": "RotateLocalFrame(right_multiply_Rz180)"
        if observed == "pre_fix"
        else "unchanged",
    }
    canonical_root.mkdir(parents=True, exist_ok=True)
    target = canonical_root / raw.name
    receipt_path = canonical_root.parent / "canonical_provenance" / (raw.stem + ".json")

    def verify(path):
        meta = read_json(path / "zarr.json")
        record = meta["attributes"].get("moving_opscale_canonicalization", {})
        require(
            all(record.get(k) == v for k, v in source.items()),
            "Canonical source provenance differs",
        )
        require(
            record.get("output_left_sha256") == tree_digest(path / LEFT),
            "Canonical left wrist hash differs",
        )
        expected_meta = read_json(raw / "zarr.json")
        expected_meta["attributes"]["moving_opscale_canonicalization"] = record
        require(
            meta == expected_meta,
            "Canonical root metadata differs beyond compatibility provenance",
        )
        for child in raw.iterdir():
            if child.name == "zarr.json" or (
                child.name == LEFT and observed == "pre_fix"
            ):
                continue
            link = path / child.name
            require(
                link.is_symlink() and link.resolve() == child.resolve(),
                "Canonical unchanged-array link differs",
            )
        require(
            not (path / LEFT).is_symlink()
            if observed == "pre_fix"
            else (path / LEFT).is_symlink(),
            "Canonical left wrist storage differs from its convention",
        )
        require(
            classify(zarr.open_group(str(path), mode="r"), frames) == "post_fix",
            "Canonical wrist convention is not post_fix",
        )
        return dict(record, canonical_root_sha256=digest(path / "zarr.json"))

    if target.exists():
        record = verify(target)
        if receipt_path.exists():
            require(
                read_json(receipt_path) == record,
                "Canonical provenance receipt differs",
            )
        else:
            # Safe recovery of a crash between atomic directory rename and receipt.
            write_json(receipt_path, record)
        return record

    temporary_root = canonical_root.parent / "canonical_tmp"
    temporary_root.mkdir(exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=raw.stem + ".", suffix=".zarr", dir=temporary_root)
    )
    try:
        for child in raw.iterdir():
            if child.name == "zarr.json":
                continue
            if child.name == LEFT and observed == "pre_fix":
                shutil.copytree(child, temporary / child.name)
            else:
                (temporary / child.name).symlink_to(
                    child, target_is_directory=child.is_dir()
                )
        if observed == "pre_fix":
            array = zarr.open_array(str(temporary / LEFT), mode="r+")
            poses = np.asarray(array[:])
            corrected = RotateLocalFrame(keys=[LEFT]).transform({LEFT: poses})[LEFT]
            require(np.isfinite(corrected).all(), "Nonfinite canonical wrist poses")
            require(
                np.array_equal(poses[:, :3], corrected[:, :3]),
                "Wrist correction moved positions",
            )
            array[:] = corrected.astype(poses.dtype)
        record = dict(source, output_left_sha256=tree_digest(temporary / LEFT))
        root_meta["attributes"]["moving_opscale_canonicalization"] = record
        write_json(temporary / "zarr.json", root_meta)
        record = verify(temporary)
        # The destination is an immutable per-episode view; never replace it.
        require(not target.exists(), "Canonical episode appeared concurrently")
        temporary.rename(target)
        write_json(receipt_path, record)
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
