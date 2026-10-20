"""Check a delivered episode against the vendor MCAP spec (spec.py, SPEC.md).

    python -m egomimic.rldb.mcap.validate episode_<id>/ [episode_<id>/ ...]

Prints PASS, or every problem found, per episode; exits 1 if any episode fails. The
checks are the plan's six groups, plus the task / label spans the Lance tables need:

  1. everything there   required topics, types, non-empty fields, episode info
  2. video usable       codec, starts on a keyframe, decodes, size matches calibration, fps
  3. clocks             Unix ns, in order, no big gaps, streams cover the episode, rates
  4. calibration        every camera has intrinsics and a path to the world frame
  5. imu                raw (gravity still in it) and fast enough, noise specs present
  6. poses              no NaNs, unit rotations, gripper in [0, 1], lost hands flagged not zeroed
  7. tasks & labels     /task and /subtask-annotation spans are ordered, inside the data,
                        each label inside one task (the rules Lance's annotation review
                        applies, egomimic/rldb/lance/annotations.py in PR #717)

The file is read once, in file order. Video is only decoded for its first
VIDEO_SAMPLE_FRAMES messages, each of which must decode to exactly one frame, so that
every frame has its own capture time (Lance needs one steps row per frame). annotation.mcap, when present,
is read too and may only hold the label topics.

MCAP reading and packet-by-packet video decoding follow PR #606's
egomimic/scripts/abc_process/abc_to_zarr.py.
"""

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from egomimic.rldb.mcap import spec

GROUPS = {
    "present": "everything there",
    "video": "video usable",
    "clocks": "clocks",
    "calibration": "calibration",
    "imu": "imu",
    "poses": "poses",
    "labels": "tasks & labels",
}
# Topics whose messages form a time series; the rest are published once or rarely.
STREAM_KINDS = {
    "video",
    "imu",
    "pose",
    "tracking_quality",
    "hand_keypoints",
    "arm",
    "gripper",
    "eye_gaze",
}


def frames_in_message(data: bytes, codec: str) -> int:
    """Pictures (access units) in one video message; the spec wants exactly 1, so every
    frame has its own capture time. Counted with the parser: the decoder silently drops
    the extra frames of a packed message. Shared with the MCAP -> Lance converter."""
    import av

    parser = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
    with av.logging.Capture():  # lone P-frames log "non-existing PPS"
        return len(parser.parse(data)) + len(parser.parse(None))


# Topics allowed in annotation.mcap.
LABEL_TOPICS = {"/task", "/subtask-annotation"}
# Annex B NAL unit types that start a decodable picture (IDR / IRAP).
KEYFRAME_NALS = {"h264": {5}, "h265": set(range(16, 22))}


@dataclass(frozen=True)
class Problem:
    group: str
    topic: str | None
    message: str

    def __str__(self) -> str:
        where = f"{self.topic}: " if self.topic else ""
        return f"[{GROUPS[self.group]}] {where}{self.message}"


@dataclass
class _Stream:
    """Everything one pass keeps about one topic."""

    schema: str
    encoding: str
    times: list[int] = field(default_factory=list)
    msgs: list = field(default_factory=list)  # decoded; video keeps only the sample
    empty: Counter = field(
        default_factory=Counter
    )  # field -> messages where it is empty
    missing_fields: set = field(default_factory=set)  # fields the message type lacks
    undecodable: bool = False

    @property
    def n(self) -> int:
        return len(self.times)

    def span_s(self) -> float:
        return (max(self.times) - min(self.times)) / 1e9 if self.times else 0.0

    def rate_hz(self) -> float:
        span = self.span_s()
        return (self.n - 1) / span if span > 0 else 0.0


@dataclass
class _Episode:
    path: Path
    metadata: dict[str, dict[str, str]]  # record name -> its key/values
    streams: dict[str, _Stream]

    @property
    def info(self) -> dict[str, str]:
        return self.metadata.get(spec.METADATA_RECORD, {})


# ---------------------------------------------------------------- reading


def _is_empty(msg, name: str) -> bool:
    value = getattr(msg, name)
    if isinstance(value, (str, bytes)) or hasattr(value, "__len__"):
        return len(value) == 0
    try:
        return not msg.HasField(name)  # sub-message
    except ValueError:
        return False  # proto3 scalar: zero is a legitimate value


def _read(path: Path) -> _Episode:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    fields_of = {t.name: t.fields for p in (spec.HUMAN, spec.ROBOT) for t in p.topics}
    video_topics = {
        t.name for p in (spec.HUMAN, spec.ROBOT) for t in p.topics if t.kind == "video"
    }
    factory = DecoderFactory()
    decoders: dict[int, object] = {}
    streams: dict[str, _Stream] = {}

    with open(path, "rb") as f:
        reader = make_reader(f)
        metadata = {r.name: dict(r.metadata) for r in reader.iter_metadata()}
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            topic = channel.topic
            s = streams.get(topic)
            if s is None:
                s = streams[topic] = _Stream(
                    schema.name if schema else "", channel.message_encoding
                )
            s.times.append(message.log_time)
            if channel.id not in decoders:
                decoders[channel.id] = factory.decoder_for(
                    channel.message_encoding, schema
                )
            decode = decoders[channel.id]
            if decode is None:
                s.undecodable = True
                continue
            is_video = topic in video_topics or topic.endswith(spec.CAMERA_SUFFIX)
            if is_video and len(s.msgs) >= spec.VIDEO_SAMPLE_FRAMES:
                continue  # video is checked on its sample; decoding all of it is costly
            msg = decode(message.data)
            for name in fields_of.get(topic, ()):
                if not hasattr(msg, name):
                    s.missing_fields.add(name)
                elif _is_empty(msg, name):
                    s.empty[name] += 1
            s.msgs.append(msg)
    return _Episode(path, metadata, streams)


# ---------------------------------------------------------------- 1. everything there


def _embodiment(ep: _Episode) -> str:
    """Declared embodiment, or a guess from the topics when it is missing/invalid."""
    declared = ep.info.get("embodiment", "")
    if declared in spec.EMBODIMENTS:
        return declared
    return "human_bimanual" if "/main-camera" in ep.streams else "eva_bimanual"


def _int(value: str) -> int | None:
    """The integer an episode-info value spells, or None (int() alone takes "²", " 5")."""
    return int(value) if re.fullmatch(r"-?[0-9]+", value) else None


def _hand_joints_problem(value: str) -> str:
    names = value.split(",")
    unknown = [n for n in names if n not in spec.HAND_JOINTS]
    if unknown:
        return f"unknown joints {unknown}; use names from {list(spec.HAND_JOINTS)}"
    if len(set(names)) != len(names):
        return "lists a joint twice"
    if "wrist" not in names:
        return "must include wrist"
    return ""


def _hand_joint_count(ep: _Episode) -> int:
    """Joints per hand the episode declares; MANO's 21 when the key is unusable."""
    value = ep.info.get("hand-joints", "")
    if value and not _hand_joints_problem(value):
        return len(value.split(","))
    return len(spec.MANO_JOINTS)


def check_present(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    expected = spec.topics_for(embodiment)
    for t in expected:
        s = ep.streams.get(t.name)
        if s is None:
            if t.required:
                out.append(Problem("present", t.name, "missing required topic"))
            continue
        if t.schema and s.schema != t.schema:
            out.append(
                Problem(
                    "present", t.name, f"type is {s.schema!r}, expected {t.schema!r}"
                )
            )
        if s.encoding != "protobuf" or s.undecodable:
            out.append(
                Problem(
                    "present",
                    t.name,
                    f"cannot decode {s.encoding!r} messages; the spec uses protobuf",
                )
            )
        for name in sorted(s.missing_fields):
            out.append(
                Problem("present", t.name, f"message type has no field {name!r}")
            )
        for name, count in sorted(s.empty.items()):
            out.append(
                Problem("present", t.name, f"{name!r} empty on {count}/{s.n} messages")
            )

    # The embodiment must match the topics: no topics of the other profile or an unused arm.
    allowed = {t.name for t in expected}
    for p in (spec.HUMAN, spec.ROBOT):
        for t in p.topics:
            if t.name in ep.streams and t.name not in allowed:
                out.append(
                    Problem(
                        "present",
                        t.name,
                        f"topic does not fit embodiment {embodiment!r}",
                    )
                )

    human = spec.profile_for(embodiment) is spec.HUMAN
    if spec.METADATA_RECORD not in ep.metadata:
        found = ", ".join(sorted(ep.metadata)) or "none"
        out.append(
            Problem(
                "present",
                None,
                f"no {spec.METADATA_RECORD!r} metadata record (found: {found})",
            )
        )
        return out
    for key in spec.METADATA_KEYS:
        if key.human_only and not human:
            continue
        value = ep.info.get(key.name)
        if value is None or value == "":
            out.append(Problem("present", None, f"episode info {key.name!r} missing"))
        elif key.kind == "sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
            out.append(
                Problem(
                    "present",
                    None,
                    f"episode info {key.name!r} is not a SHA-256 hex digest "
                    "(hash names and emails)",
                )
            )
        elif key.kind == "shutter" and not spec.SHUTTER.fullmatch(value):
            out.append(
                Problem(
                    "present",
                    None,
                    f"episode info {key.name!r} = {value!r}, expected 'global' or "
                    "'rolling:<readout>ms'",
                )
            )
        elif key.kind == "slug" and not spec.SLUG.fullmatch(value):
            out.append(
                Problem(
                    "present",
                    None,
                    f"episode info {key.name!r} = {value!r} is not a lowercase slug "
                    "(a-z, 0-9, words joined by _ or -)",
                )
            )
        elif key.kind == "hand_joints" and (bad := _hand_joints_problem(value)):
            out.append(Problem("present", None, f"episode info {key.name!r}: {bad}"))
        elif key.kind == "int" and _int(value) is None:
            out.append(
                Problem(
                    "present",
                    None,
                    f"episode info {key.name!r} = {value!r} is not an integer",
                )
            )
        elif key.kind == "enum" and value not in key.values:
            out.append(
                Problem(
                    "present",
                    None,
                    f"episode info {key.name!r} = {value!r}, expected one of {list(key.values)}",
                )
            )
    return out


# ---------------------------------------------------------------- 2. video usable


def _codec(fmt: str) -> str | None:
    fmt = fmt.lower()
    return fmt if fmt in spec.VIDEO_CODECS else None


def _nal_types(data: bytes, codec: str) -> set[int]:
    types, i = set(), 0
    while (j := data.find(b"\x00\x00\x01", i)) >= 0 and j + 3 < len(data):
        b = data[j + 3]
        types.add(b & 0x1F if codec == "h264" else (b >> 1) & 0x3F)
        i = j + 3
    return types


def _calibration(ep: _Episode, camera: str):
    s = ep.streams.get(camera + spec.CALIBRATION_SUFFIX)
    return s.msgs[0] if s and s.msgs else None


def check_video(ep: _Episode) -> list[Problem]:
    import av

    out = []
    for topic, s in sorted(ep.streams.items()):
        if not topic.endswith(spec.CAMERA_SUFFIX) or not s.msgs:
            continue
        fmt = str(getattr(s.msgs[0], "format", ""))
        codec = _codec(fmt)
        if codec is None:
            out.append(
                Problem(
                    "video",
                    topic,
                    f"format {fmt!r}, expected one of {list(spec.VIDEO_CODECS)}",
                )
            )
            continue
        keyframes = [
            i
            for i, msg in enumerate(s.msgs)
            if _nal_types(bytes(msg.data), codec) & KEYFRAME_NALS[codec]
        ]
        if not keyframes:
            out.append(
                Problem(
                    "video",
                    topic,
                    f"no keyframe (Annex B IDR/IRAP) in the first {len(s.msgs)} messages",
                )
            )
            continue
        if keyframes[0] != 0:
            out.append(
                Problem(
                    "video", topic, "first message is not a keyframe (Annex B IDR/IRAP)"
                )
            )

        # Decode from the first keyframe, so a bad start is reported once, above.
        sample = s.msgs[keyframes[0] :]

        packed = sum(frames_in_message(bytes(m.data), codec) != 1 for m in sample)
        if packed:
            out.append(
                Problem(
                    "video",
                    topic,
                    f"{packed}/{len(sample)} sampled messages do not hold exactly one "
                    "frame; each message must be one frame at its capture time",
                )
            )
        ctx = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
        frames, size = 0, None
        try:
            for msg in sample:
                for frame in ctx.decode(av.Packet(bytes(msg.data))):
                    frames += 1
                    size = size or (frame.width, frame.height)
            for frame in ctx.decode(None):
                frames += 1
                size = size or (frame.width, frame.height)
        except av.error.FFmpegError as e:
            out.append(
                Problem(
                    "video",
                    topic,
                    f"decode error in the first {len(sample)} messages: {e}",
                )
            )
            continue
        if frames == 0:
            out.append(
                Problem(
                    "video",
                    topic,
                    f"no frames decoded from the first {len(sample)} messages",
                )
            )
            continue

        calib = _calibration(ep, topic)
        if calib is not None and size != (int(calib.width), int(calib.height)):
            out.append(
                Problem(
                    "video",
                    topic,
                    f"frames are {size[0]}x{size[1]}, calibration says {calib.width}x{calib.height}",
                )
            )

        fps = s.rate_hz()  # meaningless for packed messages, reported above
        if not packed and fps < spec.MIN_VIDEO_FPS * (1 - spec.RATE_TOL):
            out.append(
                Problem(
                    "video", topic, f"{fps:.1f} fps, need >= {spec.MIN_VIDEO_FPS:g}"
                )
            )
    return out


# ---------------------------------------------------------------- 3. clocks


def _time_series(ep: _Episode, embodiment: str) -> dict[str, _Stream]:
    """Cameras and the spec's time-series topics: what defines the recording's span."""
    expected = {t.name: t for t in spec.topics_for(embodiment)}
    return {
        topic: s
        for topic, s in ep.streams.items()
        if s.times
        and (
            topic.endswith(spec.CAMERA_SUFFIX)
            or (topic in expected and expected[topic].kind in STREAM_KINDS)
        )
    }


def _span(streams: dict[str, _Stream]) -> tuple[int, int]:
    return (
        min(min(s.times) for s in streams.values()),
        max(max(s.times) for s in streams.values()),
    )


def check_clocks(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    expected = {t.name: t for t in spec.topics_for(embodiment)}
    streams = _time_series(ep, embodiment)
    for topic, s in sorted(ep.streams.items()):
        if s.times and min(s.times) < spec.MIN_UNIX_NS:
            out.append(
                Problem(
                    "clocks", topic, "timestamps are before 2015: not absolute Unix ns"
                )
            )
    for topic, s in sorted(streams.items()):
        t = np.asarray(s.times, dtype=np.int64)
        back = int((np.diff(t) < 0).sum())
        if back:
            out.append(
                Problem("clocks", topic, f"timestamps go backwards {back} times")
            )
        same = int((np.diff(t) == 0).sum())
        if same and topic.endswith(spec.CAMERA_SUFFIX):
            # Every frame needs its own time: Lance steps timestamps strictly increase.
            out.append(Problem("clocks", topic, f"{same} frames repeat a timestamp"))
        if len(t) > 1:
            gap = float(np.diff(np.sort(t)).max()) / 1e9
            if gap > spec.MAX_GAP_S:
                out.append(
                    Problem(
                        "clocks", topic, f"{gap:.2f} s gap, max {spec.MAX_GAP_S:g} s"
                    )
                )
        # Video rates are checked from decoded frames; IMU rate in its own group.
        t_spec = expected.get(topic)
        if t_spec and t_spec.min_hz and t_spec.kind not in ("video", "imu"):
            if s.rate_hz() < t_spec.min_hz * (1 - spec.RATE_TOL):
                out.append(
                    Problem(
                        "clocks",
                        topic,
                        f"{s.rate_hz():.1f} Hz, need >= {t_spec.min_hz:g}",
                    )
                )

    if not streams:
        return out
    start, end = _span(streams)
    for topic, s in sorted(streams.items()):
        late, early = (min(s.times) - start) / 1e9, (end - max(s.times)) / 1e9
        if late > spec.SPAN_TOL_S or early > spec.SPAN_TOL_S:
            out.append(
                Problem(
                    "clocks",
                    topic,
                    f"covers only part of the episode (starts {late:.1f} s late, ends {early:.1f} s early)",
                )
            )

    for key, data_ns in (("start-time-unix", start), ("end-time-unix", end)):
        value = ep.info.get(key, "")
        if _int(value) is not None:
            off = abs(_int(value) * 1e6 - data_ns) / 1e9
            if off > spec.METADATA_TIME_TOL_S:
                out.append(
                    Problem(
                        "clocks",
                        None,
                        f"episode info {key!r} is {off:.1f} s from the data",
                    )
                )
    return out


# ---------------------------------------------------------------- 4. calibration


def _quat_ok(q) -> bool:
    v = [q.x, q.y, q.z, q.w]
    return (
        all(math.isfinite(c) for c in v)
        and abs(math.sqrt(sum(c * c for c in v)) - 1) <= spec.UNIT_TOL
    )


def check_calibration(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    cameras = sorted(t for t in ep.streams if t.endswith(spec.CAMERA_SUFFIX))
    for cam in cameras:
        calib = _calibration(ep, cam)
        if calib is None:
            out.append(
                Problem(
                    "calibration",
                    cam,
                    f"no {cam + spec.CALIBRATION_SUFFIX} calibration",
                )
            )
            continue
        info = ep.streams[cam + spec.CALIBRATION_SUFFIX]
        if info.schema != "foxglove.CameraCalibration" or info.n != 1:
            out.append(
                Problem(
                    "calibration",
                    cam + spec.CALIBRATION_SUFFIX,
                    f"expected one foxglove.CameraCalibration, got {info.n} "
                    f"{info.schema or 'untyped'} message(s)",
                )
            )
        K = list(getattr(calib, "K", []))
        D = list(getattr(calib, "D", []))
        if not all(math.isfinite(v) for v in K + D):
            out.append(Problem("calibration", cam, "K or D has NaN / Inf"))
        elif len(K) == 9 and not (
            0 <= K[2] <= calib.width and 0 <= K[5] <= calib.height
        ):
            out.append(
                Problem(
                    "calibration",
                    cam,
                    f"principal point ({K[2]:g}, {K[5]:g}) is outside the "
                    f"{calib.width}x{calib.height} image",
                )
            )
        if len(K) != 9 or K[0] <= 0 or K[4] <= 0 or not calib.width or not calib.height:
            out.append(
                Problem(
                    "calibration",
                    cam,
                    "calibration has no valid K (fx, fy > 0) or size",
                )
            )
        model, n_d = str(calib.distortion_model), len(calib.D)
        if model == "" and n_d == 0:
            pass  # pinhole
        elif model not in spec.DISTORTION_MODELS:
            out.append(
                Problem(
                    "calibration",
                    cam,
                    f"distortion_model {model!r}, expected one of "
                    f"{list(spec.DISTORTION_MODELS)}",
                )
            )
        elif n_d != spec.DISTORTION_MODELS[model]:
            out.append(
                Problem(
                    "calibration",
                    cam,
                    f"{model} needs {spec.DISTORTION_MODELS[model]} D coefficients, "
                    f"got {n_d}",
                )
            )

        extra = ep.streams.get(cam + spec.CAMERA_EXTRA_SUFFIX)
        if extra and extra.msgs:
            m = extra.msgs[0]
            if m.upright_rotation_cw_deg not in spec.UPRIGHT_ROTATIONS:
                out.append(
                    Problem(
                        "calibration",
                        cam,
                        f"upright_rotation_cw_deg {m.upright_rotation_cw_deg}, expected "
                        f"one of {list(spec.UPRIGHT_ROTATIONS)}",
                    )
                )
            if not (math.isfinite(m.valid_radius) and m.valid_radius >= 0):
                out.append(
                    Problem(
                        "calibration", cam, f"valid_radius {m.valid_radius} is not >= 0"
                    )
                )
            if m.frame_id != str(getattr(calib, "frame_id", m.frame_id)):
                out.append(
                    Problem(
                        "calibration",
                        cam + spec.CAMERA_EXTRA_SUFFIX,
                        f"frame_id {m.frame_id!r} differs from the camera's -info",
                    )
                )

    # Frame graph: /tf_static edges plus the moving links whose topic is present.
    edges = defaultdict(set)
    tf = ep.streams.get("/tf_static")
    bad_rot = 0
    for msg in tf.msgs if tf else ():
        for x in msg.transforms:
            edges[x.parent_frame_id].add(x.child_frame_id)
            edges[x.child_frame_id].add(x.parent_frame_id)
            bad_rot += not _quat_ok(x.rotation)
    if bad_rot:
        out.append(
            Problem(
                "calibration",
                "/tf_static",
                f"{bad_rot} transforms have a non-unit rotation",
            )
        )
    sides = spec.arm_sides(embodiment)
    for link in spec.profile_for(embodiment).moving_links:
        if link.topic in ep.streams and (
            not link.parent.endswith("_arm_base") or link.parent.split("_")[0] in sides
        ):
            edges[link.parent].add(link.child)
            edges[link.child].add(link.parent)
    reached, todo = {spec.WORLD_FRAME}, [spec.WORLD_FRAME]
    while todo:
        for nxt in edges[todo.pop()] - reached:
            reached.add(nxt)
            todo.append(nxt)

    # Cameras and the IMU must name their frame; every frame any topic names (hands,
    # wrist poses, gaze, ...) must be placed in the world.
    must_name = set(cameras) | {"/imu"}
    for topic, s in sorted(ep.streams.items()):
        skip = topic == "/tf_static" or topic.endswith(
            (spec.CALIBRATION_SUFFIX, spec.CAMERA_EXTRA_SUFFIX)
        )
        if skip or not s.msgs or not hasattr(s.msgs[0], "frame_id"):
            continue  # a camera's -info / -info-extra share its frame, checked above
        named = {str(m.frame_id) for m in s.msgs}
        if "" in named and topic in must_name:
            out.append(
                Problem("calibration", topic, "no frame_id, so it cannot be placed")
            )
        for frame in sorted(named - reached - {""}):
            out.append(
                Problem(
                    "calibration",
                    topic,
                    f"frame {frame!r} has no path to {spec.WORLD_FRAME!r} via /tf_static",
                )
            )
    head = ep.streams.get("/head-pose")
    if (
        head
        and head.msgs
        and {str(m.frame_id) for m in head.msgs} != {spec.WORLD_FRAME}
    ):
        out.append(
            Problem(
                "calibration",
                "/head-pose",
                f"poses must be in frame {spec.WORLD_FRAME!r} (T_world_head)",
            )
        )
    return out


# ---------------------------------------------------------------- 5. imu


def check_imu(ep: _Episode) -> list[Problem]:
    out = []
    s = ep.streams.get("/imu")
    if s and s.msgs:
        if s.rate_hz() < spec.MIN_IMU_HZ * (1 - spec.RATE_TOL):
            out.append(
                Problem(
                    "imu", "/imu", f"{s.rate_hz():.0f} Hz, need >= {spec.MIN_IMU_HZ:g}"
                )
            )
        acc = np.array(
            [
                [
                    m.linear_acceleration.x,
                    m.linear_acceleration.y,
                    m.linear_acceleration.z,
                ]
                for m in s.msgs
            ]
        )
        gyr = np.array(
            [
                [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]
                for m in s.msgs
            ]
        )
        if not (np.isfinite(acc).all() and np.isfinite(gyr).all()):
            out.append(Problem("imu", "/imu", "NaN/Inf samples"))
        else:
            # Stillest window = lowest mean rotation rate; a raw IMU there reads ~9.81 m/s^2.
            w = max(1, min(len(acc), round(s.rate_hz() * spec.STILL_WINDOW_S)))
            gyro_mag = np.convolve(
                np.linalg.norm(gyr, axis=1), np.ones(w) / w, mode="valid"
            )
            i = int(np.argmin(gyro_mag))
            still = float(np.linalg.norm(acc[i : i + w], axis=1).mean())
            if abs(still - spec.GRAVITY) > spec.GRAVITY_TOL:
                out.append(
                    Problem(
                        "imu",
                        "/imu",
                        f"mean |accel| at rest = {still:.2f} m/s^2, expected ~{spec.GRAVITY} (gravity removed, or wrong units?)",
                    )
                )
    info = ep.streams.get("/imu-info")
    if info and info.msgs:
        names = (
            "accel_noise_density",
            "accel_random_walk",
            "gyro_noise_density",
            "gyro_random_walk",
        )
        bad = [n for n in names if not getattr(info.msgs[0], n, 0) > 0]
        if bad:
            out.append(
                Problem(
                    "imu", "/imu-info", f"noise values must be > 0: {', '.join(bad)}"
                )
            )
    return out


# ---------------------------------------------------------------- 6. poses


def _pose_ok(pose) -> bool:
    p = pose.position
    return _quat_ok(pose.orientation) and all(math.isfinite(v) for v in (p.x, p.y, p.z))


def check_poses(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    for t in spec.topics_for(embodiment):
        s = ep.streams.get(t.name)
        if not s or not s.msgs:
            continue
        if t.kind == "pose":
            bad = sum(not _pose_ok(m.pose) for m in s.msgs)
            if bad:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{bad}/{s.n} poses have NaNs or a non-unit rotation",
                    )
                )
        elif t.kind == "hand_keypoints":
            joints = _hand_joint_count(ep)
            shape = sum(len(m.positions) not in (0, 3 * joints) for m in s.msgs)
            if shape:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{shape}/{s.n} messages have neither {3 * joints} positions "
                        f"({joints} hand-joints x xyz) nor none",
                    )
                )
            empty = sum(
                m.valid and not m.positions and not m.HasField("wrist_pose")
                for m in s.msgs
            )
            if empty:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{empty} valid messages have neither positions nor wrist_pose",
                    )
                )
            conf = sum(
                not 0 <= m.confidence <= 1
                or len(m.joint_confidence) not in (0, joints)
                or not all(0 <= c <= 1 for c in m.joint_confidence)
                for m in s.msgs
            )
            if conf:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{conf}/{s.n} messages have a confidence outside [0, 1] or "
                        f"joint_confidence not {joints} values",
                    )
                )
            wrist = sum(
                m.valid and m.HasField("wrist_pose") and not _pose_ok(m.wrist_pose)
                for m in s.msgs
            )
            if wrist:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{wrist} valid wrist poses have NaNs or a non-unit rotation",
                    )
                )
            valid = [
                np.asarray(m.positions)
                for m in s.msgs
                if m.valid and len(m.positions) == 3 * joints
            ]
            nan = sum(not np.isfinite(p).all() for p in valid)
            zero = sum(bool(np.isfinite(p).all() and not p.any()) for p in valid)
            if nan:
                out.append(
                    Problem(
                        "poses", t.name, f"{nan} valid messages have NaN/Inf positions"
                    )
                )
            if zero:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{zero} messages are all-zero with valid=true; lost tracking must be valid=false",
                    )
                )
        elif t.kind == "eye_gaze":
            bad = sum(
                m.valid and not all(math.isfinite(v) for v in (m.yaw, m.pitch, m.depth))
                for m in s.msgs
            )
            if bad:
                out.append(
                    Problem("poses", t.name, f"{bad} valid messages have NaN/Inf")
                )
        elif t.kind == "arm":
            names = sum(
                len(m.joint_names) != len(m.position)
                or (hasattr(m, "velocity") and len(m.velocity) != len(m.position))
                or (hasattr(m, "torque") and len(m.torque) != len(m.position))
                for m in s.msgs
            )
            if names:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{names}/{s.n} messages' joint arrays do not match joint_names",
                    )
                )
            bad = 0
            for m in s.msgs:
                pose = np.asarray(m.ee_pose, dtype=np.float64)
                if (
                    len(pose) != 16
                    or not np.isfinite(pose).all()
                    or not np.isfinite(np.asarray(m.position)).all()
                ):
                    bad += 1
                    continue
                T = pose.reshape(4, 4)
                R = T[:3, :3]
                if (
                    np.abs(R.T @ R - np.eye(3)).max() > spec.UNIT_TOL
                    or abs(np.linalg.det(R) - 1) > spec.UNIT_TOL
                    or np.abs(T[3] - [0, 0, 0, 1]).max() > spec.UNIT_TOL
                ):
                    bad += 1
            if bad:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{bad}/{s.n} poses are not a finite rigid 4x4 transform",
                    )
                )
        elif t.kind == "gripper":
            lo, hi = spec.GRIPPER_RANGE
            vals = np.array([m.position for m in s.msgs], dtype=np.float64)
            bad = int((~np.isfinite(vals) | (vals < lo) | (vals > hi)).sum())
            if bad:
                out.append(
                    Problem(
                        "poses",
                        t.name,
                        f"{bad}/{s.n} gripper values outside [{lo:g}, {hi:g}]",
                    )
                )
    return out


# ---------------------------------------------------------------- 7. tasks & labels


def _spans(s: _Stream | None) -> list[tuple[int, int]]:
    """(start, end) per message; end 0 (unset) gives end == start: a point event for a
    label (one frame wide in Lance), an error for a task."""
    if not s or not s.msgs:
        return []
    return [
        (t, int(getattr(m, "end_time_unix_ns", 0)) or t)
        for t, m in zip(s.times, s.msgs)
    ]


def check_labels(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    streams = _time_series(ep, embodiment)
    if not streams:
        return out
    start, end = _span(streams)
    tol = spec.SPAN_TOL_S * 1e9
    tasks = _spans(ep.streams.get("/task"))
    labels = _spans(ep.streams.get("/subtask-annotation"))
    # A point label becomes one steps-camera frame wide in Lance (as the converter writes
    # it), and Lance's annotation review wants the whole span inside a task.
    cam = ep.streams.get(spec.profile_for(embodiment).step_clock)
    cam_t = np.sort(np.asarray(cam.times if cam else [], dtype=np.int64))
    frame = float(np.median(np.diff(cam_t))) if len(cam_t) > 1 else 0.0
    wide = [(b, e if e > b else b + frame) for b, e in labels]
    for topic, spans, point_ok in (
        ("/task", tasks, False),
        ("/subtask-annotation", labels, True),
    ):
        backwards = sum(e < b or (e == b and not point_ok) for b, e in spans)
        outside = sum(b < start - tol or e > end + tol for b, e in spans)
        if backwards:
            out.append(
                Problem(
                    "labels", topic, f"{backwards} spans do not end after they start"
                )
            )
        if outside:
            out.append(
                Problem(
                    "labels",
                    topic,
                    f"{outside}/{len(spans)} spans fall outside the recorded data",
                )
            )
    if tasks:
        orphans = sum(not any(tb <= b and e <= te for tb, te in tasks) for b, e in wide)
        if orphans:
            out.append(
                Problem(
                    "labels",
                    "/subtask-annotation",
                    f"{orphans}/{len(labels)} labels do not lie inside one /task span",
                )
            )
    else:
        # The task made from the episode info: [first frame, last frame + one frame).
        if len(cam_t) and labels:
            first, end = int(cam_t[0]), int(cam_t[-1]) + frame
            beyond = sum(b < first or e > end for b, e in wide)
            if beyond:
                out.append(
                    Problem(
                        "labels",
                        "/subtask-annotation",
                        f"{beyond}/{len(labels)} labels fall outside "
                        f"{spec.profile_for(embodiment).step_clock}'s frames (no /task)",
                    )
                )
    s = ep.streams.get("/subtask-annotation")
    types = (
        {m.annotation_type for m in s.msgs if hasattr(m, "annotation_type")}
        if s
        else set()
    )
    bad = sorted(types - {"", *spec.ANNOTATION_TYPES})
    if bad:
        out.append(
            Problem(
                "labels",
                "/subtask-annotation",
                f"annotation_type {bad}, expected one of {list(spec.ANNOTATION_TYPES)}",
            )
        )
    return out


# ---------------------------------------------------------------- entry points


@dataclass
class Result:
    embodiment: str  # declared, or guessed from the topics; "" if unreadable
    recording_id: str  # the Lance recording_id it will get; "" without lab / vendor id
    problems: list[Problem]

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_episode(episode: Path) -> Result:
    """Validate one episode directory (or its episode.mcap)."""
    episode = Path(episode)
    path = episode / spec.EPISODE_FILE if episode.is_dir() else episode
    if not path.exists():
        return Result(
            "", "", [Problem("present", None, f"no {spec.EPISODE_FILE} in {episode}")]
        )
    try:
        ep = _read(path)
    except Exception as e:  # truncated or not an MCAP at all
        return Result(
            "",
            "",
            [
                Problem(
                    "present", None, f"cannot read {path.name}: {type(e).__name__}: {e}"
                )
            ],
        )
    problems = []
    annotation = path.with_name(spec.ANNOTATION_FILE)
    if path.name == spec.EPISODE_FILE and annotation.exists():
        try:
            extra = _read(annotation)
        except Exception as e:
            extra = None
            problems.append(
                Problem(
                    "present",
                    None,
                    f"cannot read {annotation.name}: {type(e).__name__}: {e}",
                )
            )
        if extra and extra.metadata:
            problems.append(
                Problem(
                    "present",
                    None,
                    f"{annotation.name} carries metadata records {sorted(extra.metadata)}; "
                    f"episode info belongs in {path.name} only",
                )
            )
        for topic, s in sorted(extra.streams.items()) if extra else ():
            if topic not in LABEL_TOPICS:
                problems.append(
                    Problem(
                        "present",
                        topic,
                        f"only {sorted(LABEL_TOPICS)} belong in {annotation.name}",
                    )
                )
            elif topic in ep.streams:
                problems.append(
                    Problem(
                        "present", topic, f"in both {path.name} and {annotation.name}"
                    )
                )
            else:
                ep.streams[topic] = s
    embodiment = _embodiment(ep)
    for group, check, args in (
        ("present", check_present, (ep, embodiment)),
        ("video", check_video, (ep,)),
        ("clocks", check_clocks, (ep, embodiment)),
        ("calibration", check_calibration, (ep, embodiment)),
        ("imu", check_imu, (ep,)),
        ("poses", check_poses, (ep, embodiment)),
        ("labels", check_labels, (ep, embodiment)),
    ):
        try:
            problems += check(*args)
        except Exception as e:  # a topic of the wrong type the group's checks assume
            problems.append(
                Problem(
                    group,
                    None,
                    f"could not check: {type(e).__name__}: {e} (see 'type is' problems)",
                )
            )
    lab, vid = ep.info.get("lab", ""), ep.info.get("vendor-recording-id", "")
    rid = spec.recording_id(lab, vid) if lab and vid else ""
    return Result(embodiment, rid, problems)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "episodes",
        nargs="+",
        type=Path,
        help="episode_<id>/ dirs or episode.mcap files",
    )
    args = p.parse_args(argv)
    failed = 0
    for episode in args.episodes:
        r = validate_episode(episode)
        label = (
            f"{episode} (spec {spec.SPEC_VERSION}, embodiment {r.embodiment or '?'}, "
            f"recording_id {r.recording_id or '?'})"
        )
        if r.problems:
            failed += 1
            print(f"FAIL {label}: {len(r.problems)} problem(s)")
            for problem in r.problems:
                print(f"  {problem}")
        else:
            print(f"PASS {label}")
    if len(args.episodes) > 1:
        print(f"\n{len(args.episodes) - failed}/{len(args.episodes)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
