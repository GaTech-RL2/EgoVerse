"""Episode format version, attrs schema tables, and the episode validator.

Single source of truth for what ZarrWriter writes and ZarrEpisode reads.
`egoverse schema` renders the tables; `egoverse validate` runs validate_episode.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

FORMAT_VERSION = "1.0"
SUPPORTED_MAJOR = frozenset({0, 1})  # 0 = legacy episodes with no format_version

# (T, 7) xyz + wxyz pose arrays whose quaternions must be unit-norm.
POSE_KEYS = (
    "left.obs_ee_pose",
    "right.obs_ee_pose",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "obs_head_pose",
    "left.cmd_ee_pose",
    "right.cmd_ee_pose",
)
QUAT_NORM_ATOL = 1e-4
# Missing pose frames skip the norm check: all 7 values == POSE_MISSING_ZERO (Mecka's
# undetected-hand rows) or all |v| >= POSE_SENTINEL_THRESHOLD (the repo's 1e9
# missing-data sentinel, see aria_utils / pose_utils).
POSE_MISSING_ZERO = 0.0
POSE_SENTINEL_THRESHOLD = 1e8

# (name, type, description) -- rendered by schema_markdown(), checked by validate_episode().
REQUIRED_ATTRS: tuple[tuple[str, str, str], ...] = (
    (
        "format_version",
        "str",
        'Episode format version, "MAJOR.MINOR". Missing = legacy v0.',
    ),
    (
        "embodiment",
        "str",
        'Embodiment identifier, e.g. "human_bimanual" (must match the DB row).',
    ),
    ("total_frames", "int", "Number of valid frames (not padded)."),
    ("fps", "int", "Capture frame rate; 30 or 60."),
    ("task_name", "str", "Task name (must match the DB row)."),
    ("task_description", "str", "Free-text description of the trial."),
    (
        "intrinsics",
        "dict",
        "{camera_key: 3x4 K matrix}; non-empty; projection uses the 'front' entry.",
    ),
    (
        "features",
        "dict",
        "One entry per array key: {dtype, shape, names}; images dtype 'jpeg', annotations 'json'.",
    ),
)
OPTIONAL_ATTRS: tuple[tuple[str, str, str], ...] = (
    (
        "extrinsics",
        "dict | None",
        "None, or a non-empty dict of 4x4 world<-cam transforms keyed per arm.",
    ),
    (
        "provenance",
        "dict",
        "writer, egomimic_version, git_sha, created_at, converter, source_uri.",
    ),
)
FEATURE_DTYPES: tuple[tuple[str, str, str], ...] = (
    (
        "numeric",
        "numpy dtype string",
        'shape = per-frame shape, names = dimension labels (e.g. ["dim_0"]).',
    ),
    (
        "jpeg",
        '"jpeg"',
        'Images: shape [H, W, 3], names ["height", "width", "channel"].',
    ),
    (
        "json",
        '"json"',
        'Annotations: shape [N], names ["json"], format "annotation_v1".',
    ),
)


def schema_markdown() -> str:
    def esc(cell: str) -> str:
        # A raw pipe (e.g. the type "dict | None") would split the table cell.
        return cell.replace("|", "\\|")

    def table(title, rows):
        out = [f"**{title}**", "", "| Key | Type | Meaning |", "|---|---|---|"]
        out += [f"| `{esc(n)}` | {esc(t)} | {esc(d)} |" for n, t, d in rows]
        return "\n".join(out)

    return (
        "\n\n".join(
            [
                f"Format version: `{FORMAT_VERSION}` (readers accept majors {sorted(SUPPORTED_MAJOR)}).",
                table("Required attrs", REQUIRED_ATTRS),
                table("Optional attrs", OPTIONAL_ATTRS),
                table("`features[<key>].dtype`", FEATURE_DTYPES),
                f"**Pose arrays** ({', '.join(f'`{k}`' for k in POSE_KEYS)}): shape "
                f"`(T, 7)` = xyz + wxyz quaternion, unit-norm within {QUAT_NORM_ATOL:g}. "
                "A missing frame is exempt from the norm check; mark it with all 7 "
                f"values `{POSE_MISSING_ZERO:g}`, or all 7 with magnitude >= "
                f"{POSE_SENTINEL_THRESHOLD:g} (write `1e9`). Any other frame with a "
                "non-unit quaternion, NaN included, is an error.",
            ]
        )
        + "\n"
    )


class SchemaVersionError(ValueError):
    """The episode's format_version is one this egomimic cannot read."""


def check_format_version(attrs: Mapping) -> str:
    """Return the episode's format_version ("0.0" when absent) or raise
    SchemaVersionError when its major version is unsupported."""
    raw = attrs.get("format_version")
    if raw is None:
        return "0.0"
    version = str(raw)
    try:
        major = int(version.split(".")[0])
    except ValueError as exc:
        raise SchemaVersionError(f"malformed format_version {version!r}") from exc
    if major not in SUPPORTED_MAJOR:
        lo, hi = min(SUPPORTED_MAJOR), max(SUPPORTED_MAJOR)
        raise SchemaVersionError(
            f"episode is v{version}, this egomimic reads v{lo}-v{hi}; "
            "upgrade egomimic or re-export the episode"
        )
    return version


@dataclass
class ValidationReport:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _unwrap(raw):
    """Unwrap nested 0-d / 1-element object ndarrays down to the stored scalar."""
    while isinstance(raw, np.ndarray):
        raw = raw.item() if raw.shape == () else raw.flat[0]
    return raw


def validate_episode(path: str | Path) -> ValidationReport:
    """Check one episode directory against the format spec. Reports malformed
    attrs and arrays as lines in `errors` (or `warnings`) instead of raising."""
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    rep = ValidationReport(path=str(path))
    err, warn = rep.errors.append, rep.warnings.append
    try:
        store = zarr.open_group(str(path), mode="r")
    except Exception as e:  # not a zarr group at all
        err(f"cannot open zarr group: {e}")
        return rep
    meta = dict(store.attrs)

    version = meta.get("format_version")
    if version is None:
        warn("no format_version attr (legacy v0 episode); re-export to stamp it")
    elif not (isinstance(version, str) and re.fullmatch(r"\d+\.\d+", version)):
        err(f'format_version={version!r}, expected a "MAJOR.MINOR" string')
    else:
        try:
            check_format_version(meta)
        except SchemaVersionError as e:
            err(str(e))

    for name, _, _ in REQUIRED_ATTRS:
        if name != "format_version" and name not in meta:
            err(f"missing attr: {name}")
    if "fps" in meta and meta["fps"] not in (30, 60):
        err(f"fps={meta['fps']!r}, expected 30 or 60")
    try:
        get_embodiment_id(str(meta.get("embodiment", "")))
    except (KeyError, AttributeError):
        err(f"embodiment={meta.get('embodiment')!r} is not a valid identifier")

    intr = meta.get("intrinsics")
    if not isinstance(intr, dict) or not intr:
        err("intrinsics: missing or not a non-empty {camera_key: 3x4} dict")
    else:
        if not any("front" in str(k).lower() for k in intr):
            err(f"intrinsics: no front-camera entry (keys: {list(intr)})")
        for cam, K in intr.items():
            try:
                shape = np.asarray(K, dtype=float).shape
            except (TypeError, ValueError):
                shape = "a non-numeric or ragged value"
            if shape != (3, 4):
                err(f"intrinsics[{cam!r}]: expected 3x4, got {shape}")
    extr = meta.get("extrinsics")
    if extr is not None and (not isinstance(extr, dict) or not extr):
        err("extrinsics: present but not a non-empty dict (must be None or a dict)")

    T = meta.get("total_frames")
    if not isinstance(T, int):
        err(f"total_frames={T!r} is not an int")
        T = None
    features = meta.get("features") or {}
    if not isinstance(features, dict):
        err(f"features: expected a dict, got {type(features).__name__}")
        features = {}
    dtypes = {}  # array key -> features[key]["dtype"]
    for key, f in features.items():
        if isinstance(f, dict):
            dtypes[key] = f.get("dtype")
        else:
            err(f"features[{key!r}]: expected a dict, got {type(f).__name__}")

    if T is not None:
        for key in store.keys():
            node = store[key]
            if not isinstance(node, zarr.Array) or dtypes.get(key) == "json":
                continue
            if node.shape[0] < T:
                err(f"{key}: array length {node.shape[0]} < total_frames {T}")
    for key in ("images.front_1", "left.obs_ee_pose", "right.obs_ee_pose"):
        if key not in store:
            err(f"missing required key: {key}")
    for key in POSE_KEYS:
        if key not in store:
            continue
        shape = store[key].shape
        if len(shape) != 2 or shape[-1] != 7:
            err(f"{key}: expected shape (T, 7), got {shape}")
            continue
        pose = np.asarray(store[key][:T])
        missing = np.all(pose == POSE_MISSING_ZERO, axis=1) | np.all(
            np.abs(pose) >= POSE_SENTINEL_THRESHOLD, axis=1
        )
        frames = np.flatnonzero(~missing)
        norms = np.linalg.norm(pose[frames, 3:7], axis=1)
        bad = frames[~np.isclose(norms, 1.0, rtol=0.0, atol=QUAT_NORM_ATOL)]
        if bad.size:
            err(
                f"{key}: {bad.size} frames with non-unit quaternions (first: frame {bad[0]})"
            )
    for key in (
        "left.obs_gripper",
        "right.obs_gripper",
        "left.gripper",
        "right.gripper",
    ):
        if key in store and (len(store[key].shape) != 2 or store[key].shape[-1] != 1):
            err(f"{key}: expected shape (T, 1), got {store[key].shape}")
    for key in ("left.obs_keypoints", "right.obs_keypoints"):
        if key in store and store[key].shape[-1:] != (63,):
            err(f"{key}: expected last dim 63 (21x3)")
    for key, dtype in dtypes.items():
        if dtype != "json" or key not in store:
            continue
        node = store[key]
        bad, first = 0, None
        for i in range(node.shape[0]):
            raw = _unwrap(node[i])
            if isinstance(raw, np.bytes_):
                raw = bytes(raw)
            try:
                rec = json.loads(
                    raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                )
                for fld, typ in (("text", str), ("start_idx", int), ("end_idx", int)):
                    if not isinstance(rec.get(fld), typ):
                        raise ValueError(f"field {fld!r} missing or not {typ.__name__}")
                if not 0 <= rec["start_idx"] <= rec["end_idx"] or (
                    T is not None and rec["end_idx"] > T
                ):
                    raise ValueError("index range invalid")
            except Exception as e:
                bad += 1
                first = first or (i, str(e))
        if bad:
            err(
                f"{key}: {bad}/{node.shape[0]} annotations malformed (e.g. index {first[0]}: {first[1]})"
            )
    for key, dtype in dtypes.items():
        if dtype != "jpeg" or key not in store:
            continue
        try:
            import simplejpeg

            raw = _unwrap(store[key][0:1][0])
            frame = simplejpeg.decode_jpeg(bytes(raw), colorspace="RGB")
            if frame.ndim != 3 or frame.shape[2] != 3:
                err(f"{key}: decoded frame 0 has shape {frame.shape}")
        except Exception as e:
            err(f"{key}: failed to decode frame 0: {e}")
    return rep
