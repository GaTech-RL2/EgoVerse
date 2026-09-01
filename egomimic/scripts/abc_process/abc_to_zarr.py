"""
Convert ABC-130k (XDOF/ABC-130k) YAM episodes from MCAP to EgoVerse Zarr v3.

Written against docs/YAM_DATA_FORMAT.md in the dataset repo.

    episode_<uuid>/episode.mcap      -> <uuid>.zarr
    episode_<uuid>/annotation.mcap   -> the `annotations` array (annotated eps only)

Two station variants exist and differ only in cameras, so the top camera topic is
auto-detected rather than assumed:
  * RealSense -- mono `/top-camera`, all streams H.264, 640x480
  * ZED-X     -- stereo `/top-left-camera` + `/top-right-camera`, H.265 top /
                 H.264 wrists, 1920x1200. The left eye is `images.front_1`; a ZED-X
                 episode ALSO carries a `/top-camera` preview with both eyes
                 composited in, which must not be picked.

Video frames define the episode clock; joint/gripper streams poll at a few hundred
Hz on independent clocks and are matched onto the front-camera timestamps by
nearest neighbour (the association the format doc prescribes). That is what makes
`total_frames` consistent across every array, per CONTRIBUTING_DATA.md §6.2.

    python -m egomimic.scripts.abc_process.abc_to_zarr \
        --input abc_raw/data/train/fold_and_stack_the_towels \
        --output /path/to/processed_v3/yam_bimanual

Requires: pip install mcap mcap-protobuf-support
"""

import argparse
import json
import logging
import traceback
from pathlib import Path

import av
import cv2
import numpy as np
import simplejpeg

from egomimic.rldb.embodiment.yam import Yam
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

logger = logging.getLogger(__name__)

# Top-camera candidates in priority order. The stereo eyes come FIRST: on a
# ZED-X episode `/top-camera` also exists but is a 640x400 preview with the
# left/right insets baked into it, so preferring it would train on a composited
# thumbnail. Mono RealSense episodes have only the clean `/top-camera`.
# (Matches the rule in the dataset's own export_mcap.py.)
TOP_CAMERA_TOPICS = ("/top-left-camera", "/top-right-camera", "/top-camera")
VIDEO_TOPICS = {
    "/left-wrist-camera": "images.left_wrist",
    "/right-wrist-camera": "images.right_wrist",
}

# RobotState carries BOTH the joint vector and the 4x4 EE pose, so one topic
# fans out into two zarr keys.
ARM_TOPICS = {
    "/left-arm-state": ("left.obs_joints", "left.obs_ee_pose"),
    "/right-arm-state": ("right.obs_joints", "right.obs_ee_pose"),
    "/left-arm-action": ("left.cmd_joints", "left.cmd_ee_pose"),
    "/right-arm-action": ("right.cmd_joints", "right.cmd_ee_pose"),
}
# GripperState.position[0], normalised aperture (0 = closed, 1 = open).
GRIPPER_TOPICS = {
    "/left-ee-state": "left.obs_gripper",
    "/right-ee-state": "right.obs_gripper",
    "/left-ee-action": "left.cmd_gripper",
    "/right-ee-action": "right.cmd_gripper",
}
INSTRUCTION_TOPIC = "/instruction"
ANNOTATION_TOPIC = "/subtask-annotation"

# ZarrWriter's image path resizes to 640x480; intrinsics are rescaled to match.
TARGET_W, TARGET_H = 640, 480


def _read_mcap(path: Path) -> dict[str, list]:
    """{topic: [(log_time_ns, decoded_msg), ...]}, each list time-sorted."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    out: dict[str, list] = {}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages():
            out.setdefault(channel.topic, []).append((message.log_time, decoded))
    for v in out.values():
        v.sort(key=lambda kv: kv[0])
    return out


def _episode_metadata(path: Path) -> dict[str, str]:
    """Flatten the MCAP metadata records (camera types, operator, duration...)."""
    from mcap.reader import make_reader

    out: dict[str, str] = {}
    with open(path, "rb") as f:
        for record in make_reader(f).iter_metadata():
            out.update(record.metadata)
    return out


def _decode_video(entries: list) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """CompressedVideo messages -> (timestamps_ns, JPEG object array, [H, W, 3]).

    Frames are resized and JPEG-encoded as they are decoded rather than being
    accumulated raw: a 5k-frame episode at 3 cameras is ~14 GB of uint8 held at
    once, which OOMs. Encoding inline keeps it around 40 KB/frame.

    `data` is an Annex B byte stream for one frame and parameter sets are
    prepended to every keyframe, so a packet-by-packet decode is sufficient. The
    codec is per-stream, so it is read off `format` rather than assumed.
    """
    empty = (np.zeros(0, dtype=np.int64), np.empty(0, dtype=object), [0, 0, 3])
    if not entries:
        return empty

    fmt = str(getattr(entries[0][1], "format", "h264")).lower()
    ctx = av.CodecContext.create("hevc" if fmt in ("h265", "hevc") else "h264", "r")

    encoded, stamps = [], []

    def _emit(frame, log_time):
        img = frame.to_ndarray(format="rgb24")
        if img.shape[1] != TARGET_W or img.shape[0] != TARGET_H:
            img = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        encoded.append(
            simplejpeg.encode_jpeg(
                np.ascontiguousarray(img),
                quality=ZarrWriter.JPEG_QUALITY,
                colorspace="RGB",
            )
        )
        stamps.append(log_time)

    for log_time, msg in entries:
        for frame in ctx.decode(av.Packet(bytes(msg.data))):
            _emit(frame, log_time)
    for frame in ctx.decode(None):  # flush
        _emit(frame, entries[-1][0])

    if not encoded:
        return empty
    arr = np.empty(len(encoded), dtype=object)
    arr[:] = encoded
    return np.asarray(stamps, dtype=np.int64), arr, [TARGET_H, TARGET_W, 3]


def _nearest(src_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Indices into src_ts of the nearest sample to each target_ts."""
    pos = np.searchsorted(src_ts, target_ts)
    lo = np.clip(pos - 1, 0, len(src_ts) - 1)
    hi = np.clip(pos, 0, len(src_ts) - 1)
    take_hi = np.abs(src_ts[hi] - target_ts) < np.abs(src_ts[lo] - target_ts)
    return np.where(take_hi, hi, lo)


def _camera_K(topics: dict, video_topic: str) -> np.ndarray | None:
    """3x4 K from the sibling `<topic>-info` CameraCalibration, rescaled to output size."""
    entries = topics.get(f"{video_topic}-info")
    if not entries:
        return None
    msg = entries[0][1]
    K = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
    w, h = int(msg.width), int(msg.height)
    if w and h:
        K[0] *= TARGET_W / w
        K[1] *= TARGET_H / h
    return np.hstack([K, np.zeros((3, 1))])  # §6.4 wants 3x4


def _annotations(
    path: Path, frame_ts: np.ndarray
) -> list[tuple[str, int, int]]:
    """annotation.mcap -> [(label, start_frame, end_frame)].

    Each message marks where a subtask BEGINS; it runs until the next message,
    and the last one runs to the end of the episode.
    """
    if not path.exists():
        return []
    entries = _read_mcap(path).get(ANNOTATION_TOPIC, [])
    if not entries:
        return []

    starts = np.searchsorted(frame_ts, [t for t, _ in entries], side="left")
    n = len(frame_ts)
    out = []
    for i, (_t, msg) in enumerate(entries):
        start = int(np.clip(starts[i], 0, n - 1))
        end = int(np.clip(starts[i + 1], 0, n)) if i + 1 < len(entries) else n
        if end > start:
            out.append((str(msg.data), start, end))
    return out


def convert_episode(
    episode_dir: Path,
    output_dir: Path,
    embodiment: str,
    task_name: str | None = None,
    chunk_timesteps: int = 100,
) -> Path:
    topics = _read_mcap(episode_dir / "episode.mcap")
    meta = _episode_metadata(episode_dir / "episode.mcap")

    top_topic = next((t for t in TOP_CAMERA_TOPICS if t in topics), None)
    if top_topic is None:
        raise ValueError(
            f"{episode_dir.name}: no top camera among {TOP_CAMERA_TOPICS}; "
            f"topics present: {sorted(topics)}"
        )

    frame_ts, front, front_shape = _decode_video(topics[top_topic])
    if len(front) == 0:
        raise ValueError(f"{episode_dir.name}: decoded 0 frames from {top_topic}")
    T = len(front)
    del topics[top_topic]  # free the compressed packets

    pre_encoded = {"images.front_1": (front, front_shape)}
    for topic, key in VIDEO_TOPICS.items():
        if topic not in topics:
            logger.warning("%s: no %s, skipping %s", episode_dir.name, topic, key)
            continue
        ts, frames, shape = _decode_video(topics.pop(topic))
        if len(frames) == 0:
            continue
        pre_encoded[key] = (frames[_nearest(ts, frame_ts)], shape)

    numeric_data: dict[str, np.ndarray] = {
        "obs_rgb_timestamps_ns": frame_ts.astype(np.int64)
    }

    for topic, (joint_key, pose_key) in ARM_TOPICS.items():
        entries = topics.get(topic)
        if not entries:
            logger.warning("%s: no %s, skipping arm keys", episode_dir.name, topic)
            continue
        ts = np.asarray([t for t, _ in entries], dtype=np.int64)
        idx = _nearest(ts, frame_ts)
        joints = np.stack([np.asarray(m.position, np.float64) for _, m in entries])
        numeric_data[joint_key] = joints[idx]

        # RobotState.pose is documented as always present, but a real slice of
        # ABC ships it empty on every message of an episode (joints only). Write
        # the key only when the whole stream has it, rather than emitting zeros
        # or a half-populated pose track.
        if all(len(m.pose) == 16 for _, m in entries):
            poses = np.stack(
                [np.asarray(m.pose, np.float64).reshape(4, 4) for _, m in entries]
            )
            numeric_data[pose_key] = _matrix_to_xyzwxyz(poses[idx])
        else:
            n_empty = sum(1 for _, m in entries if len(m.pose) != 16)
            logger.warning(
                "%s: %s has no EE pose on %d/%d messages, skipping %s",
                episode_dir.name, topic, n_empty, len(entries), pose_key,
            )

    for topic, key in GRIPPER_TOPICS.items():
        entries = topics.get(topic)
        if not entries:
            logger.warning("%s: no %s, skipping %s", episode_dir.name, topic, key)
            continue
        ts = np.asarray([t for t, _ in entries], dtype=np.int64)
        vals = np.asarray(
            [[float(m.position[0])] for _, m in entries], dtype=np.float64
        )
        numeric_data[key] = vals[_nearest(ts, frame_ts)]

    pose_keys = [k for k in numeric_data if k.endswith("ee_pose")]
    if not pose_keys:
        raise ValueError(
            f"{episode_dir.name}: no EE pose on any arm stream (joints only). "
            "The Yam cartesian pipeline needs left/right obs+cmd ee_pose; this "
            "episode would need FK from the YAM MJCF to be usable."
        )

    if task_name is None:
        instr = topics.get(INSTRUCTION_TOPIC)
        task_name = str(instr[0][1].data) if instr else episode_dir.parent.name

    # ABC records no extrinsics, but the rig is published (i2rt-robotics/i2rt,
    # robot_models/station/...). Attach the top-camera transform only for the
    # mono RealSense D405 station that model describes -- a ZED-X episode uses a
    # different camera and rig with no published model.
    top_cam_type = meta.get("top_camera_type", "")
    extrinsics = None
    if top_topic == "/top-camera" and "D405" in top_cam_type:
        extrinsics = {"front_1": Yam.TOP_CAMERA_D405}
    else:
        logger.warning(
            "%s: top camera is %r on %s, no published extrinsics -- omitting",
            episode_dir.name, top_cam_type or "unknown", top_topic,
        )

    K = _camera_K(topics, top_topic)
    if K is None:
        raise ValueError(
            f"{episode_dir.name}: {top_topic}-info absent, so no intrinsics; "
            "intrinsics are mandatory (CONTRIBUTING_DATA.md §6.3)"
        )

    fps = 30
    span_s = (frame_ts[-1] - frame_ts[0]) / 1e9 if T > 1 else 0
    if span_s > 0:
        fps = int(round((T - 1) / span_s))

    return ZarrWriter.create_and_write(
        episode_path=output_dir / f"{episode_dir.name.removeprefix('episode_')}.zarr",
        numeric_data=numeric_data,
        pre_encoded_image_data=pre_encoded,
        embodiment=embodiment,
        fps=fps,
        task_name=task_name,
        task_description=task_name,
        annotations=_annotations(episode_dir / "annotation.mcap", frame_ts),
        chunk_timesteps=chunk_timesteps,
        intrinsics={"front_1": K},
        extrinsics=extrinsics,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="a task dir, or one episode dir")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--embodiment",
        default="yam_bimanual",
        help="member of EMBODIMENT (egomimic/rldb/embodiment/embodiment.py)",
    )
    p.add_argument(
        "--task-name", default=None, help="default: the /instruction message"
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk-timesteps", type=int, default=100)
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (in_path / "episode.mcap").exists():
        episodes = [in_path]
    else:
        episodes = sorted(d for d in in_path.iterdir() if (d / "episode.mcap").exists())
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit(f"no episodes with an episode.mcap under {in_path}")

    failures = []
    for i, ep in enumerate(episodes, 1):
        try:
            path = convert_episode(
                ep, out_dir, args.embodiment, args.task_name, args.chunk_timesteps
            )
            logger.info("[%d/%d] wrote %s", i, len(episodes), path)
        except Exception:
            failures.append(ep.name)
            logger.error(
                "[%d/%d] %s failed:\n%s", i, len(episodes), ep.name, traceback.format_exc()
            )

    logger.info("done: %d ok, %d failed", len(episodes) - len(failures), len(failures))
    if failures:
        print(json.dumps({"failed": failures}, indent=2))


if __name__ == "__main__":
    main()
