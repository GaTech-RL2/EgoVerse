"""Convert one zarr episode into a delivery: per-frame parquet plus MP4s re-encoded from its JPEGs.

For the vendor and robot zarrs that have no raw converter (FORMATS); Aria and EVA migrate
from raw (aria_to_lance, eva_to_lance), so their zarrs are refused. Steps rows are exactly
the zarr frames: the zarr converters already deleted the frames they could not use
(untracked hands, missing poses), so nothing is filled back in. With per-frame times
(obs_rgb_timestamps_ns) a gap in `timestamp_ns` is where zarr dropped frames; without them
frame i is at start + i / fps, so dropped stretches leave no gap, and provenance.clock says
"fps". Validity flags say which arms the embodiment has and which
zarr values are real (not the 1e8+ sentinels, not all-zero keypoints or robot poses). The
recording id is `recording_id("zarr", episode_hash)`.

Zarr attrs carry the task but usually not lab, operator or scene; pass those as
`metadata` (e.g. the SQL episodes row). lab, task_name and operator are required; scene may
be missing and objects default to []. With --stage-to the delivery is staged under
`<staging>/zarr/<vendor id>/` for `lance_commit --source zarr`.

    python -m egomimic.scripts.zarr_to_lance /path/2025-11-14-17-16-59-204000 \\
        --out-dir /tmp/deliveries --manifest manifest.csv --stage-to /tmp/staging
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import simplejpeg
import zarr

from egomimic.rldb.embodiment.embodiment import canonical_embodiment_name
from egomimic.rldb.embodiment.end_effector import END_EFFECTOR
from egomimic.rldb.lance import commit
from egomimic.rldb.lance.metadata import (
    objects_list,
    placeholder,
    read_manifest,
    text_or_none,
)
from egomimic.rldb.lance.schema import (
    HUMAN_STEPS,
    recording_id,
    robot_steps_table,
    steps_schema,
    zarr_columns,
)
from egomimic.rldb.zarr.episode_rules import (
    arm_sides,
    embodiment_error,
    episode_hash_start_ns,
)
from egomimic.rldb.zarr.required_metadata import UNWAIVABLE_FIELDS
from egomimic.utils.video_utils import open_av1_encoder

FFMPEG = "ffmpeg"
STAGING_SOURCE = "zarr"
SENTINEL = 1e8
SHORT_SIDE = 360
IDENTITY_POSE = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
READ_BATCH = 256
# Per zarr format: what acts at the arm; the default rig; the cameras that also get a
# full-resolution stream named `<cam>_<short side>` (as the raw aria delivery has
# front_1_480); non-`images.` JPEG keys to encode, by video name; whether head_pose is the
# front_1 camera pose (identity extrinsics when the zarr has none); whether an all-zero
# robot pose is a dropout to flag and forward-fill.
_HUMAN_VENDOR = dict(
    end_effector=END_EFFECTOR.HUMAN_HAND,
    full_res={"front_1"},
    head_camera=True,
)
_ROBOT = dict(full_res=set(), head_camera=False, zero_dropout=True)
FORMATS = {
    "mecka": dict(_HUMAN_VENDOR, rig_name="mecka"),
    "scale": dict(_HUMAN_VENDOR, rig_name="scale"),
    "microagi_v2": dict(_HUMAN_VENDOR, rig_name="microagi_mono"),
    "yam": dict(_ROBOT, end_effector=END_EFFECTOR.PARALLEL_JAW_GRIPPER, rig_name="yam"),
    "dexmate": dict(
        _ROBOT,
        end_effector=END_EFFECTOR.DEX_HAND,
        rig_name="dexmate",
        extra_videos={
            "left.obs_tactile_image": "left_tactile",
            "right.obs_tactile_image": "right_tactile",
        },
    ),
}
# Attrs the delivery stores in its own fields; the rest go to provenance.
CORE_ATTRS = {
    "embodiment",
    "total_frames",
    "fps",
    "features",
    "task_name",
    "task_description",
    "intrinsics",
    "extrinsics",
}
HUMAN_FIELDS = {
    "wrist_pose": ("obs_wrist_pose", IDENTITY_POSE),
    "ee_pose": ("obs_ee_pose", IDENTITY_POSE),
    "keypoints": ("obs_keypoints", 0.0),
    "aria_keypoints": ("obs_aria_keypoints", 0.0),
}


def open_episode(src, storage_options: dict | None = None) -> zarr.Group:
    """A zarr group from a local path, an fsspec URL (s3://...) or a zarr Store/Group."""
    if isinstance(src, zarr.Group):
        return src
    if isinstance(src, (str, Path)):
        src = str(src)
        if "://" in src:
            return zarr.open_group(src, mode="r", storage_options=storage_options)
    return zarr.open_group(src, mode="r")


def episode_hash_of(src) -> str:
    name = str(getattr(src, "path", None) or src).rstrip("/").rsplit("/", 1)[-1]
    return name[: -len(".zarr")] if name.endswith(".zarr") else name


def embodiment_of(attrs: dict) -> str:
    """The zarr's embodiment, lowercased, with vendor prefixes (MECKA_BIMANUAL) mapped
    onto human_*."""
    return canonical_embodiment_name(str(attrs.get("embodiment", ""))).lower()


def detect_format(attrs: dict, keys: set[str]) -> str:
    """The FORMATS entry of the converter that wrote the episode; fails on unknown ones."""
    emb = embodiment_of(attrs)
    if emb == "yam_bimanual":
        return "yam"
    if emb == "dexmate_bimanual":
        return "dexmate"
    if emb.startswith("human_"):
        if "scale_task_id" in attrs:
            return "scale"
        if "user_id" in attrs and "episode_id" in attrs:
            return "mecka"
        if "source_mcap" in attrs and "obs_rgb_timestamps_ns" in keys:
            return "microagi_v2"
    raise NotImplementedError(
        f"no Lance mapping for this zarr (embodiment {emb!r}, attrs {sorted(attrs)}, keys "
        f"{sorted(keys)}); known formats: {sorted(FORMATS)} (microagi v1 and trace are "
        "not; aria and eva migrate from raw)"
    )


def episode_metadata(attrs: dict, metadata: dict | None) -> dict:
    """Zarr attrs overlaid with `metadata`, whose blank or placeholder values (empty
    manifest cells) are ignored; rejects missing or placeholder lab/task_name/operator."""
    m = {k: attrs.get(k) for k in ("task_name", "task_description", "scene", "objects")}
    over = {k: v for k, v in (metadata or {}).items() if not placeholder(v)}
    # The SQL row names the task `task`; it wins over the zarr attr (mecka's is "debug").
    if "task" in over:
        over.setdefault("task_name", over.pop("task"))
    m.update(over)
    bad = [k for k in UNWAIVABLE_FIELDS if placeholder(m.get(k))]
    if bad:
        raise ValueError(f"required metadata missing or placeholder: {bad}")
    return {
        "lab": str(m["lab"]),
        "operator": str(m["operator"]),
        "task_name": str(m["task_name"]),
        "task_description": text_or_none(m.get("task_description")),
        "scene": None if placeholder(m.get("scene")) else str(m["scene"]),
        "objects": objects_list(m.get("objects")),
        "license": text_or_none(m.get("license")),
        "rig_name": text_or_none(m.get("rig_name")),
    }


def _read(group: zarr.Group, key: str, n: int) -> np.ndarray:
    x = np.asarray(group[key][:n])
    if len(x) < n:
        raise ValueError(f"{key}: {len(x)} rows, total_frames {n}")
    return x


def _ok(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(1) & (np.abs(x) < SENTINEL).all(1)


def _vec(x: np.ndarray, valid: np.ndarray, fill) -> pa.Array:
    x = np.asarray(x, dtype=np.float32).copy()
    x[~valid] = fill
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def human_steps(group, rid: str, ts: np.ndarray, arms) -> tuple[pa.Table, dict]:
    """HUMAN_STEPS rows; a hand is valid where its poses and keypoints are real and tracked."""
    n = len(ts)
    keys = set(group.keys())
    head = _read(group, "obs_head_pose", n)
    head_valid = _ok(head)
    cols = {
        "recording_id": pa.array([rid] * n),
        "timestamp_ns": pa.array(ts),
        "head_pose": _vec(head, head_valid, IDENTITY_POSE),
    }
    valid = {"head": head_valid}
    for side in ("left", "right"):
        fields = {}
        if side in arms:
            for f, (zk, _) in HUMAN_FIELDS.items():
                if f"{side}.{zk}" in keys:
                    fields[f] = _read(group, f"{side}.{zk}", n)
            missing = {"wrist_pose", "ee_pose", "keypoints"} - set(fields)
            if missing:
                raise ValueError(f"{side} hand: zarr has no {sorted(missing)}")
            v = np.logical_and.reduce(
                [_ok(fields[f]) for f in ("wrist_pose", "ee_pose", "keypoints")]
            )
            # Vendor zarrs write an untracked hand as zero keypoints (and identity poses).
            v &= (fields["keypoints"] != 0).any(1)
        else:
            v = np.zeros(n, bool)
        width = {
            f: HUMAN_STEPS.field(side).type.field(f).type.list_size
            for f in HUMAN_FIELDS
        }
        arrays = [
            _vec(fields.get(f, np.zeros((n, width[f]))), v, fill)
            for f, (_, fill) in HUMAN_FIELDS.items()
        ]
        # Zarr keeps no MPS confidence: NaN where the hand is valid, -1 (untracked) elsewhere.
        arrays.append(pa.array(np.where(v, np.nan, -1.0).astype(np.float32)))
        cols[side] = pa.StructArray.from_arrays(
            arrays, fields=list(HUMAN_STEPS.field(side).type)
        )
        valid[side] = v
    cols["left_valid"] = pa.array(valid["left"])
    cols["right_valid"] = pa.array(valid["right"])
    cols["head_pose_valid"] = pa.array(head_valid)
    if "obs_eye_gaze" in keys:
        eye = _read(group, "obs_eye_gaze", n)
        cols["eye_gaze"] = _vec(eye, _ok(eye), 0.0)
    else:
        cols["eye_gaze"] = _vec(np.zeros((n, 3)), np.zeros(n, bool), 0.0)
    return pa.table(cols, schema=HUMAN_STEPS), valid


def robot_steps(
    group, rid: str, ts: np.ndarray, arms, schema: pa.Schema, zero_dropout: bool
) -> tuple[pa.Table, dict]:
    """Steps rows of a robot schema, whose columns are the zarr keys. A side the robot lacks
    is zeros; a sentinel pose is invalid; with `zero_dropout` an all-zero ee pose is too,
    and the side is forward-filled there (as eva_to_zarr does)."""
    n = len(ts)
    keys = set(group.keys())
    arrays = {}
    for key in zarr_columns(schema):
        side = key.split(".")[0]
        typ = schema.field(side).type
        typ = typ.field(key.split(".", 1)[1]).type if pa.types.is_struct(typ) else typ
        if side in ("left", "right") and side not in arms:
            arrays[key] = np.zeros((n, getattr(typ, "list_size", 1)))
        elif key not in keys:
            raise ValueError(f"zarr has no {key}")
        else:
            arrays[key] = _read(group, key, n)
    valid = {}
    for side in ("left", "right"):
        if side not in arms:
            valid[side] = np.zeros(n, bool)
            continue
        poses = [arrays[f"{side}.{p}_ee_pose"] for p in ("obs", "cmd")]
        ok = _ok(poses[0]) & _ok(poses[1])
        if zero_dropout:
            dropped = np.logical_or.reduce([(p == 0).all(1) for p in poses])
            ok &= ~dropped
            src = np.maximum.accumulate(np.where(dropped, 0, np.arange(n)))
            for k in arrays:
                if k.startswith(f"{side}."):
                    arrays[k] = arrays[k][src]
        valid[side] = ok
    return robot_steps_table(schema, rid, ts, arrays, valid), valid


def hash_start_ns(episode_hash: str) -> int | None:
    """Epoch ns an episode hash encodes: a %Y-%m-%d-%H-%M-%S-%f time, or a Mongo ObjectId's
    creation second (mecka); None for others (abc's UUIDs)."""
    ns = episode_hash_start_ns(episode_hash)
    if ns is None and re.fullmatch(r"[0-9a-f]{24}", episode_hash):
        return int(episode_hash[:8], 16) * 1_000_000_000
    return ns


def frame_times(group, fmt_name: str, start_ns: int | None, n: int, fps: float):
    if "obs_rgb_timestamps_ns" in group:
        ts = _read(group, "obs_rgb_timestamps_ns", n).astype(np.int64)
    else:
        # As eva_to_lance: no per-frame times, frame i at start + i / fps; an episode
        # whose hash encodes no time starts at 0.
        ts = (start_ns or 0) + np.round(np.arange(n) * 1e9 / fps).astype(np.int64)
    if not (np.diff(ts) > 0).all():
        raise ValueError(f"{fmt_name}: frame timestamps are not strictly increasing")
    return ts


def annotations(group, ts: np.ndarray, end_ns: int) -> tuple[list[dict], list[str]]:
    """The zarr's `annotations` spans on the recording clock, and why any were dropped.

    A span's start_idx/end_idx count zarr frames, which are exactly the steps rows, so frame
    i is at ts[i] whether or not the zarr has obs_rgb_timestamps_ns; a span reaching the
    last frame ends at `end_ns`. An end past the last frame is clamped to it: vendors (yam)
    write spans against the pre-trim frame count.
    """
    if "annotations" not in group:
        return [], []
    n = len(ts)
    out, dropped = [], []
    for raw in group["annotations"][:]:
        d = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        if not isinstance(d, dict):
            continue
        s, e, text = d.get("start_idx"), d.get("end_idx"), d.get("text")
        if placeholder(text):
            dropped.append(f"no text: {d}")
        elif not (isinstance(s, int) and isinstance(e, int) and 0 <= s < min(e, n)):
            dropped.append(f"range [{s}, {e}) is outside the {n} frames: {text!r}")
        else:
            t1 = int(ts[e]) if e < n else end_ns
            out.append({"t_start_ns": int(ts[s]), "t_end_ns": t1, "text": str(text)})
    return out, dropped


def _intrinsics_3x3(k) -> np.ndarray:
    return np.asarray(k, dtype=np.float64)[:3, :3]


def encode_videos(
    group,
    out: Path,
    n: int,
    fps: float,
    attrs: dict,
    full_res: set,
    ffmpeg: str,
    extra: dict | None = None,
) -> tuple[list[str], dict]:
    """One MP4 per `images.<cam>` and per `extra` {zarr key: video name} at short side 360
    (plus `<cam>_<short>` at full size for `full_res` cameras); returns the video names and
    their 3x3 intrinsics."""
    features = attrs.get("features", {})
    jpeg_keys = {k for k, f in features.items() if f.get("dtype") == "jpeg"}
    cams = {k: k[len("images.") :] for k in jpeg_keys if k.startswith("images.")}
    if not cams:
        raise ValueError("zarr has no JPEG image keys")
    unknown = jpeg_keys - set(cams) - set(extra or {})
    if unknown:
        raise ValueError(f"JPEG keys with no video name: {sorted(unknown)}")
    for k, name in (extra or {}).items():
        if k in jpeg_keys:
            cams[k] = name
    intr = attrs.get("intrinsics") or {}
    names, intrinsics = [], {}
    for key, cam in sorted(cams.items()):
        h, w = features[key]["shape"][:2]
        scale = min(1.0, SHORT_SIDE / min(h, w))
        sizes = {cam: (round(h * scale) // 2 * 2, round(w * scale) // 2 * 2)}
        if cam in full_res and scale < 1:
            sizes[f"{cam}_{min(h, w)}"] = (h, w)
        enc = {
            name: open_av1_encoder(out / f"{name}.mp4", vh, vw, fps, ffmpeg)
            for name, (vh, vw) in sizes.items()
        }
        try:
            arr = group[key]
            for a in range(0, n, READ_BATCH):
                for i, jpeg in enumerate(arr[a : min(a + READ_BATCH, n)], a):
                    if not len(jpeg):
                        # A missing chunk reads as the fill value b"".
                        raise ValueError(f"{key}: frame {i} is empty (chunk missing?)")
                    frame = simplejpeg.decode_jpeg(bytes(jpeg), colorspace="RGB")
                    if frame.shape[:2] != (h, w):
                        raise ValueError(
                            f"{key}: frame {frame.shape}, metadata {h}x{w}"
                        )
                    for name, (vh, vw) in sizes.items():
                        img = (
                            frame
                            if (vh, vw) == (h, w)
                            else cv2.resize(
                                frame, (vw, vh), interpolation=cv2.INTER_AREA
                            )
                        )
                        enc[name].stdin.write(np.ascontiguousarray(img).tobytes())
        finally:
            for p in enc.values():
                p.stdin.close()
        if any(p.wait() != 0 for p in enc.values()):
            raise RuntimeError(f"ffmpeg failed on {key}")
        for name, (vh, _) in sizes.items():
            names.append(name)
            if cam in intr:
                k = _intrinsics_3x3(intr[cam]).copy()
                k[:2] *= vh / h
                intrinsics[name] = k.reshape(-1).tolist()
    return names, intrinsics


def convert_zarr_episode(
    zarr_path_or_store,
    out_dir,
    metadata: dict | None = None,
    *,
    episode_hash: str | None = None,
    storage_options: dict | None = None,
    ffmpeg: str = FFMPEG,
) -> Path:
    """Write the delivery of one zarr episode to `out_dir/<vendor id>/` and return that dir.

    `zarr_path_or_store` is a local path, an fsspec URL or a zarr Store/Group; `episode_hash`
    defaults to its basename. `metadata` overlays the zarr attrs (lab, operator, scene, ...).
    """
    t0 = time.time()
    group = open_episode(zarr_path_or_store, storage_options)
    attrs = dict(group.attrs)
    episode_hash = episode_hash or episode_hash_of(zarr_path_or_store)
    keys = set(group.keys())
    fmt_name = detect_format(attrs, keys)
    fmt = FORMATS[fmt_name]
    meta_in = episode_metadata(attrs, metadata)
    embodiment = embodiment_of(attrs)
    if err := embodiment_error(embodiment):
        raise ValueError(err)
    arms = arm_sides(embodiment)
    end_effector = fmt["end_effector"]
    schema = steps_schema(embodiment, end_effector.value)
    start_ns = hash_start_ns(episode_hash)
    rid = recording_id(STAGING_SOURCE, episode_hash)
    n = int(attrs["total_frames"])
    fps = float(attrs["fps"])

    out = Path(out_dir) / episode_hash
    out.mkdir(parents=True, exist_ok=True)
    ts = frame_times(group, fmt_name, start_ns, n, fps)
    per_frame_times = "obs_rgb_timestamps_ns" in keys
    if end_effector == END_EFFECTOR.HUMAN_HAND:
        steps, valid = human_steps(group, rid, ts, arms)
    else:
        steps, valid = robot_steps(group, rid, ts, arms, schema, fmt["zero_dropout"])
    pq.write_table(steps, out / "steps.parquet", compression="zstd")
    t_numeric = time.time() - t0

    videos, intrinsics = encode_videos(
        group,
        out,
        n,
        fps,
        attrs,
        fmt["full_res"],
        ffmpeg,
        fmt.get("extra_videos"),
    )
    if attrs.get("extrinsics"):
        extrinsics = {
            k: np.asarray(v, np.float64).reshape(-1).tolist()
            for k, v in attrs["extrinsics"].items()
        }
    elif fmt["head_camera"]:
        # As aria_to_lance: head_pose is the front_1 camera pose.
        extrinsics = {"front_1": np.eye(4).reshape(-1).tolist()}
    else:
        extrinsics = {}
    gaps = int((np.diff(ts) > 1.5e9 / fps).sum())
    # Half-open like every Lance interval: the last frame is inside its task and alias.
    t_end_ns = int(ts[-1]) + round(1e9 / fps)
    spans, dropped = annotations(group, ts, t_end_ns)
    meta = {
        "vendor_recording_id": episode_hash,
        "recording_id": rid,
        "episode_hash": episode_hash,
        "source": STAGING_SOURCE,
        "format": fmt_name,
        "embodiment": embodiment,
        "end_effector": end_effector.value,
        **meta_in,
        "rig_name": meta_in["rig_name"] or attrs.get("rig_name") or fmt["rig_name"],
        "license": meta_in["license"] or text_or_none(attrs.get("license")),
        "captured_at_ms": None if start_ns is None else start_ns // 1_000_000,
        "fps": fps,
        "num_frames": n,
        "videos": videos,
        "t_start_ns": int(ts[0]),
        "t_end_ns": t_end_ns,
        "annotations": spans,
        "annotations_dropped": dropped,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "valid_frac": {k: float(v.mean()) for k, v in valid.items()},
        "gaps": gaps,
        "provenance": {
            "from": "zarr",
            "format": fmt_name,
            "zarr": str(getattr(zarr_path_or_store, "path", zarr_path_or_store)),
            "clock": "obs_rgb_timestamps_ns" if per_frame_times else "fps",
            "attrs": {k: v for k, v in attrs.items() if k not in CORE_ATTRS},
        },
        "seconds_numeric": round(t_numeric, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("zarr", nargs="+", help="zarr episode paths or URLs")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--manifest", type=Path, help="parquet or CSV of metadata by episode_hash"
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="metadata for every episode, over the manifest",
    )
    ap.add_argument("--ffmpeg", default=FFMPEG)
    ap.add_argument(
        "--stage-to", help="staging root (s3:// or local) for the committer"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest = read_manifest(args.manifest) if args.manifest else {}
    extra = dict(s.split("=", 1) for s in args.set)
    for z in args.zarr:
        h = episode_hash_of(z)
        out = convert_zarr_episode(
            z,
            args.out_dir,
            {**manifest.get(h, {}), **extra},
            ffmpeg=args.ffmpeg,
        )
        print(out.joinpath("meta.json").read_text())
        if args.stage_to:
            commit.stage(out, commit.join(args.stage_to, STAGING_SOURCE, out.name))


if __name__ == "__main__":
    main()
