"""Check a raw MCAP episode against the raw-data spec (spec.py, SPEC.md).

    python -m egomimic.rldb.mcap.validate episode_<id>/ [episode_<id>/ ...]

Prints PASS, or every problem found, per episode; exits 1 if any episode fails. The
checks fall in seven groups:

  1. everything there   required topics, types, non-empty fields, episode info
  2. video usable       codec, starts on a keyframe, no B-frames, decodes, size matches
                        calibration, fps
  3. clocks             Unix ns, in order, no big gaps, streams cover the episode, rates
  4. calibration        every camera has intrinsics and a path to the world frame
  5. imu                raw (gravity still in it) and fast enough, noise specs present
  6. poses              no NaNs, unit rotations, gripper in [0, 1], lost hands flagged not zeroed
  7. tasks & labels     /task and /subtask-annotation spans are ordered, inside the data,
                        each label inside one task (the rules Lance's annotation review
                        applies, egomimic/rldb/lance/annotations.py in PR #717)

The file is read once, in file order, with chunk and data-section CRCs checked; topics
the spec does not define are counted but never decoded. Every video message is
checked structurally (non-empty, one frame_id and format per stream, holds an H.264 /
H.265 picture); only the first VIDEO_SAMPLE_FRAMES are fully decoded, each of which
must decode to exactly one frame, so that every frame has its own capture time (Lance
needs one steps row per frame). annotation.mcap, when present, is read too and may only
hold the label topics.

MCAP reading and packet-by-packet video decoding follow PR #606's
egomimic/scripts/abc_process/abc_to_zarr.py.
"""

import argparse
import math
import re
import sys
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from egomimic.rldb.mcap import spec
from egomimic.rldb.zarr.required_metadata import (
    REQUIRED_METADATA_FIELDS,
    check_stored_metadata,
    is_placeholder,
    waiver_errors,
)

GROUPS = {
    "present": "everything there",
    "video": "video usable",
    "clocks": "clocks",
    "calibration": "calibration",
    "imu": "imu",
    "poses": "poses",
    "labels": "tasks & labels",
}
# Annex B NAL unit types that start a decodable picture (IDR / IRAP).
KEYFRAME_NALS = {"h264": {5}, "h265": set(range(16, 22))}
# NAL unit types that carry picture data (coded slices, VCL).
PICTURE_NALS = {"h264": set(range(1, 6)), "h265": set(range(0, 22))}


@dataclass(frozen=True)
class Problem:
    group: str
    topic: str | None
    message: str

    def __str__(self) -> str:
        where = f"{self.topic}: " if self.topic else ""
        return f"[{GROUPS[self.group]}] {where}{self.message}"


Add = Callable[[str | None, str], None]


def _adder(out: list[Problem], group: str) -> Add:
    """add(topic, message) appends a Problem of ``group`` to ``out``."""

    def add(topic: str | None, message: str) -> None:
        out.append(Problem(group, topic, message))

    return add


@dataclass
class _Stream:
    """Everything one pass keeps about one topic."""

    schema: str
    encoding: str
    times: array = field(default_factory=lambda: array("q"))
    # Decoded messages of the rare topics, and the video sample; time series keep only
    # `values` instead.
    msgs: list = field(default_factory=list)
    # Time series: _ROWS[kind] of every message, flattened, `width` per message.
    values: array = field(default_factory=lambda: array("d"))
    width: int = 0
    row_error: Exception | None = None  # a message _ROWS could not read (wrong type)
    # field -> messages where it is empty
    empty: Counter = field(default_factory=Counter)
    missing_fields: set = field(default_factory=set)  # fields the message type lacks
    undecodable: bool = False
    channels: set = field(default_factory=set)
    frame_ids: Counter = field(default_factory=Counter)  # of messages that have one
    # Video only, over every message (msgs holds just the sample).
    formats: Counter = field(default_factory=Counter)
    no_picture: int = 0

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
    record_counts: Counter = field(default_factory=Counter)  # record name -> count

    @property
    def info(self) -> dict[str, str]:
        return self.metadata.get(spec.METADATA_RECORD, {})


# ---------------------------------------------------------------- reading


def frames_in_message(data: bytes, codec: str) -> int:
    """Pictures (access units) in one video message; the spec wants exactly 1, so every
    frame has its own capture time. Counted with the parser: the decoder silently drops
    the extra frames of a packed message. Shared with the MCAP -> Lance converter."""
    import av

    parser = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
    with av.logging.Capture():  # lone P-frames log "non-existing PPS"
        return len(parser.parse(data)) + len(parser.parse(None))


def _nal_types(data: bytes, codec: str) -> set[int]:
    types, i = set(), 0
    while (j := data.find(b"\x00\x00\x01", i)) >= 0 and j + 3 < len(data):
        b = data[j + 3]
        types.add(b & 0x1F if codec == "h264" else (b >> 1) & 0x3F)
        i = j + 3
    return types


def _is_empty(msg, name: str) -> bool:
    value = getattr(msg, name)
    if isinstance(value, (str, bytes)) or hasattr(value, "__len__"):
        return len(value) == 0
    try:
        return not msg.HasField(name)  # sub-message
    except ValueError:
        return False  # proto3 scalar: zero is a legitimate value


def _note_video(s: _Stream, msg) -> None:
    fmt = str(getattr(msg, "format", ""))
    s.formats[fmt] += 1
    data = bytes(getattr(msg, "data", b""))
    if data and fmt in PICTURE_NALS and not _nal_types(data, fmt) & PICTURE_NALS[fmt]:
        s.no_picture += 1


def _imu_row(m) -> tuple:
    a, g = m.linear_acceleration, m.angular_velocity
    return (a.x, a.y, a.z, g.x, g.y, g.z)


def _hand_row(m) -> tuple:
    pos = np.asarray(m.positions, dtype=np.float64)
    finite = bool(np.isfinite(pos).all())
    wrist = m.HasField("wrist_pose")
    conf_ok = 0 <= m.confidence <= 1 and all(0 <= c <= 1 for c in m.joint_confidence)
    return (
        m.valid,
        len(pos),
        wrist,
        wrist and _pose_ok(m.wrist_pose),
        conf_ok,
        len(m.joint_confidence),
        finite,
        finite and not pos.any(),
    )


def _arm_row(m) -> tuple:
    names = len(m.joint_names) != len(m.position) or any(
        hasattr(m, a) and len(getattr(m, a)) != len(m.position)
        for a in ("velocity", "torque")
    )
    return (names, _rigid_4x4(m))


# Topic.kind -> what its checks need of one message, as a fixed-width row of floats.
_ROWS = {
    "imu": (6, _imu_row),
    "pose": (1, lambda m: (_pose_ok(m.pose),)),
    "hand_keypoints": (8, _hand_row),
    "eye_gaze": (
        1,
        lambda m: (m.valid and not all(map(math.isfinite, (m.yaw, m.pitch, m.depth))),),
    ),
    "arm": (2, _arm_row),
    "gripper": (1, lambda m: (m.position,)),
    "tracking_quality": (0, lambda m: ()),  # only its fields are checked
}


def _rows(s: _Stream) -> np.ndarray:
    """The stream's rows as an (n, width) array; raises if a message was unreadable."""
    if s.row_error is not None:
        raise s.row_error
    return np.array(s.values, dtype=np.float64).reshape(-1, s.width)


def _note(s: _Stream, topic: str, kind: str | None, fields: tuple, msg) -> None:
    for name in fields:
        if not hasattr(msg, name):
            s.missing_fields.add(name)
        elif _is_empty(msg, name):
            s.empty[name] += 1
    if hasattr(msg, "frame_id"):  # a type without it is reported in group 1
        s.frame_ids[str(msg.frame_id)] += 1
    if topic.endswith(spec.CAMERA_SUFFIX):
        _note_video(s, msg)
        if len(s.msgs) < spec.VIDEO_SAMPLE_FRAMES:
            s.msgs.append(msg)  # only the sample is kept and fully decoded
    elif kind in _ROWS:
        s.width, row = _ROWS[kind]
        if s.row_error is None:
            try:
                s.values.extend(map(float, row(msg)))
            except Exception as e:
                s.row_error = e.with_traceback(None)
    else:
        s.msgs.append(msg)


def _read(path: Path) -> _Episode:
    from mcap.records import Channel, Message, Metadata, Schema
    from mcap.stream_reader import StreamReader
    from mcap_protobuf.decoder import DecoderFactory

    topics = {t.name: t for p in (spec.HUMAN, spec.ROBOT) for t in p.topics}
    camera_topics = tuple(
        spec.CAMERA_SUFFIX + x
        for x in ("", spec.CALIBRATION_SUFFIX, spec.CAMERA_EXTRA_SUFFIX)
    )
    factory = DecoderFactory()
    decoders: dict[int, object] = {}
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    streams: dict[str, _Stream] = {}
    metadata: dict[str, dict[str, str]] = {}
    record_counts: Counter = Counter()

    # One linear pass: it checks the data-section CRC of unchunked files as well as
    # chunk CRCs (SeekingReader skips both without chunk indexes), and finds metadata
    # records whether or not the summary indexes them.
    with open(path, "rb") as f:
        for record in StreamReader(f, validate_crcs=True).records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Metadata):
                metadata[record.name] = dict(record.metadata)
                record_counts[record.name] += 1
            elif isinstance(record, Message):
                channel = channels[record.channel_id]
                topic = channel.topic
                s = streams.get(topic)
                if s is None:
                    schema = schemas.get(channel.schema_id)
                    s = streams[topic] = _Stream(
                        schema.name if schema else "", channel.message_encoding
                    )
                s.times.append(record.log_time)
                s.channels.add(channel.id)
                t = topics.get(topic)
                if t is None and not topic.endswith(camera_topics):
                    continue  # not in the spec: counted, never decoded
                if channel.id not in decoders:
                    decoders[channel.id] = factory.decoder_for(
                        channel.message_encoding, schemas.get(channel.schema_id)
                    )
                decode = decoders[channel.id]
                if decode is None:
                    s.undecodable = True
                    continue
                msg = decode(record.data)
                _note(s, topic, t and t.kind, t.fields if t else (), msg)
    return _Episode(path, metadata, streams, record_counts)


def _merge_annotation_file(ep: _Episode, path: Path) -> list[Problem]:
    """Add annotation.mcap's label topics to ``ep``; problems with the file."""
    out = []
    add = _adder(out, "present")
    annotation = path.with_name(spec.ANNOTATION_FILE)
    if path.name != spec.EPISODE_FILE or not annotation.exists():
        return out
    try:
        extra = _read(annotation)
    except Exception as e:
        add(None, f"cannot read {annotation.name}: {type(e).__name__}: {e}")
        return out
    if extra.metadata:
        add(
            None,
            f"{annotation.name} carries metadata records {sorted(extra.metadata)}; "
            f"episode info belongs in {path.name} only",
        )
    for topic, s in sorted(extra.streams.items()):
        if topic not in spec.LABEL_TOPICS:
            add(topic, f"only {sorted(spec.LABEL_TOPICS)} belong in {annotation.name}")
        elif topic in ep.streams:
            add(topic, f"in both {path.name} and {annotation.name}")
        else:
            ep.streams[topic] = s
    return out


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
    value = ep.info.get("hand_joints", "")
    if value and not _hand_joints_problem(value):
        return len(value.split(","))
    return len(spec.MANO_JOINTS)


def _required_metadata_problems(
    info: dict[str, str], allow_missing: bool
) -> tuple[list[str], list[str]]:
    """(problems, waiver warnings) from the zarr required-metadata rules on the episode
    info (CONTRIBUTING_DATA.md §6.3). Fields named in `missing_metadata` are waived
    (MissingMetadata), except spec.UNWAIVABLE_FIELDS."""
    raw = info.get(spec.MISSING_METADATA_KEY, "")
    waived = {f.strip() for f in raw.split(",") if f.strip()}
    out = waiver_errors(info, waived)
    meta = {}
    for name in REQUIRED_METADATA_FIELDS:
        value = info.get(name)
        if name in waived and not value:
            meta[name] = None
        elif value is not None:
            meta[name] = value
    errors, warnings, _passed = check_stored_metadata(meta, allow_missing=allow_missing)
    return out + errors, warnings


def _str_value_problem(key: spec.MetaKey, value: str) -> str | None:
    return (
        f" = {value!r} is a placeholder, not a value" if is_placeholder(value) else None
    )


def _int_value_problem(key: spec.MetaKey, value: str) -> str | None:
    return f" = {value!r} is not an integer" if _int(value) is None else None


def _enum_value_problem(key: spec.MetaKey, value: str) -> str | None:
    if value in key.values:
        return None
    return f" = {value!r}, expected one of {list(key.values)}"


def _shutter_value_problem(key: spec.MetaKey, value: str) -> str | None:
    if spec.SHUTTER.fullmatch(value):
        return None
    return f" = {value!r}, expected 'global' or 'rolling:<readout>ms'"


def _hand_joints_value_problem(key: spec.MetaKey, value: str) -> str | None:
    bad = _hand_joints_problem(value)
    return f": {bad}" if bad else None


# MetaKey.kind -> what follows "episode info 'key'" in its problem, or None if fine.
# required_metadata and waiver keys are checked by _required_metadata_problems.
_META_VALUE_CHECKS = {
    "str": _str_value_problem,
    "int": _int_value_problem,
    "enum": _enum_value_problem,
    "shutter": _shutter_value_problem,
    "hand_joints": _hand_joints_value_problem,
}


def check_present(
    ep: _Episode, embodiment: str, allow_missing: bool = False
) -> tuple[list[Problem], list[str]]:
    """(problems, waiver warnings) for group 1."""
    out = []
    add = _adder(out, "present")
    expected = spec.topics_for(embodiment)
    for t in expected:
        s = ep.streams.get(t.name)
        if s is None:
            if t.required:
                add(t.name, "missing required topic")
            continue
        if t.schema and s.schema != t.schema:
            add(t.name, f"type is {s.schema!r}, expected {t.schema!r}")
        if s.encoding != "protobuf" or s.undecodable:
            add(
                t.name, f"cannot decode {s.encoding!r} messages; the spec uses protobuf"
            )
        for name in sorted(s.missing_fields):
            add(t.name, f"message type has no field {name!r}")
        for name, count in sorted(s.empty.items()):
            add(t.name, f"{name!r} empty on {count}/{s.n} messages")
    for topic, s in sorted(ep.streams.items()):
        if len(s.channels) > 1:
            add(topic, f"published on {len(s.channels)} channels; use one per topic")
        once = topic in spec.ONCE_TOPICS or topic.endswith(spec.CAMERA_EXTRA_SUFFIX)
        if once and s.n > 1:
            add(topic, f"published {s.n} times, expected once")

    # The embodiment must match the topics: no topics of the other profile or an unused arm.
    allowed = {t.name for t in expected}
    for p in (spec.HUMAN, spec.ROBOT):
        for t in p.topics:
            if t.name in ep.streams and t.name not in allowed:
                add(t.name, f"topic does not fit embodiment {embodiment!r}")

    if spec.METADATA_RECORD not in ep.metadata:
        found = ", ".join(sorted(ep.metadata)) or "none"
        add(None, f"no {spec.METADATA_RECORD!r} metadata record (found: {found})")
        return out, []
    if ep.record_counts[spec.METADATA_RECORD] > 1:
        add(
            None,
            f"{ep.record_counts[spec.METADATA_RECORD]} {spec.METADATA_RECORD!r} "
            "records; write exactly one",
        )
    known = {k.name for k in spec.METADATA_KEYS}
    for name in sorted(set(ep.info) - known):
        hint = name.replace("-", "_")
        hint = f"; did you mean {hint!r}?" if hint in known else ""
        add(None, f"unknown episode info key {name!r}{hint}")
    messages, warnings = _required_metadata_problems(ep.info, allow_missing)
    for message in messages:
        add(None, message)
    human = spec.profile_for(embodiment) is spec.HUMAN
    for key in spec.METADATA_KEYS:
        check = _META_VALUE_CHECKS.get(key.kind)
        if check is None or (key.human_only and not human):
            continue
        value = ep.info.get(key.name)
        if value is None or value == "":
            if not key.optional:
                add(None, f"episode info {key.name!r} missing")
        elif (bad := check(key, value)) is not None:
            add(None, f"episode info {key.name!r}{bad}")
    return out, warnings


# ---------------------------------------------------------------- 2. video usable


def _codec(fmt: str) -> str | None:
    return fmt if fmt in spec.VIDEO_CODECS else None


def _calibration(ep: _Episode, camera: str):
    s = ep.streams.get(camera + spec.CALIBRATION_SUFFIX)
    if not s or not s.msgs or s.schema != "foxglove.CameraCalibration":
        return None
    return s.msgs[0]


def check_video(ep: _Episode) -> list[Problem]:
    import av

    out = []
    add = _adder(out, "video")
    for topic, s in sorted(ep.streams.items()):
        if not topic.endswith(spec.CAMERA_SUFFIX) or not s.msgs:
            continue
        fmt = str(getattr(s.msgs[0], "format", ""))
        codec = _codec(fmt)
        if codec is None:
            add(topic, f"format {fmt!r}, expected one of {list(spec.VIDEO_CODECS)}")
            continue
        if len(s.formats) > 1:
            add(
                topic,
                f"format changes within the stream: {dict(sorted(s.formats.items()))}",
            )
        if s.no_picture:
            add(
                topic,
                f"{s.no_picture}/{s.n} messages hold no {codec} picture (coded slice)",
            )
        keyframes = [
            i
            for i, msg in enumerate(s.msgs)
            if _nal_types(bytes(msg.data), codec) & KEYFRAME_NALS[codec]
        ]
        if not keyframes:
            add(
                topic,
                f"no keyframe (Annex B IDR/IRAP) in the first {len(s.msgs)} messages",
            )
            continue
        if keyframes[0] != 0:
            add(topic, "first message is not a keyframe (Annex B IDR/IRAP)")

        # Decode from the first keyframe, so a bad start is reported once, above.
        sample = s.msgs[keyframes[0] :]

        packed = sum(frames_in_message(bytes(m.data), codec) != 1 for m in sample)
        if packed:
            add(
                topic,
                f"{packed}/{len(sample)} sampled messages do not hold exactly one "
                "frame; each message must be one frame at its capture time",
            )
        ctx = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
        # Fail on bitstream errors instead of concealing them.
        ctx.options = {"err_detect": "explode"}
        # pts = message index: a decoder that hands frames back out of that order is
        # reordering B-frames (has_b_frames alone is set by streams that have none).
        out_order, b_frames, size = [], False, None
        try:
            for i, msg in enumerate(sample):
                packet = av.Packet(bytes(msg.data))
                packet.pts = i
                for frame in ctx.decode(packet):
                    out_order.append(frame.pts)
                    b_frames |= frame.pict_type == av.video.frame.PictureType.B
                    size = size or (frame.width, frame.height)
            for frame in ctx.decode(None):
                out_order.append(frame.pts)
                b_frames |= frame.pict_type == av.video.frame.PictureType.B
                size = size or (frame.width, frame.height)
        except av.error.FFmpegError as e:
            add(topic, f"decode error in the first {len(sample)} messages: {e}")
            continue
        frames = len(out_order)
        if frames == 0:
            add(topic, f"no frames decoded from the first {len(sample)} messages")
            continue
        if b_frames or (not packed and out_order != sorted(out_order)):
            add(
                topic,
                "uses B-frames; re-encode without them (x264: -bf 0), so each "
                "message decodes to its own frame in timestamp order",
            )
        if not packed and frames != len(sample):
            add(
                topic,
                f"{frames} frames decoded from {len(sample)} sampled messages; "
                "each message must decode to one frame",
            )

        calib = _calibration(ep, topic)
        if calib is not None and size != (int(calib.width), int(calib.height)):
            add(
                topic,
                f"frames are {size[0]}x{size[1]}, calibration says "
                f"{calib.width}x{calib.height}",
            )

        fps = s.rate_hz()  # meaningless for packed messages, reported above
        if not packed and fps < spec.MIN_VIDEO_FPS * (1 - spec.RATE_TOL):
            add(topic, f"{fps:.1f} fps, need >= {spec.MIN_VIDEO_FPS:g}")
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
            or (topic in expected and expected[topic].kind in spec.STREAM_KINDS)
        )
    }


def _span(streams: dict[str, _Stream]) -> tuple[int, int]:
    return (
        min(min(s.times) for s in streams.values()),
        max(max(s.times) for s in streams.values()),
    )


def check_clocks(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    add = _adder(out, "clocks")
    expected = {t.name: t for t in spec.topics_for(embodiment)}
    streams = _time_series(ep, embodiment)
    for topic, s in sorted(ep.streams.items()):
        if s.times and min(s.times) < spec.MIN_UNIX_NS:
            add(topic, "timestamps are before 2015: not absolute Unix ns")
    for topic, s in sorted(streams.items()):
        t = np.asarray(s.times, dtype=np.int64)
        back = int((np.diff(t) < 0).sum())
        if back:
            add(topic, f"timestamps go backwards {back} times")
        same = int((np.diff(t) == 0).sum())
        if same and topic.endswith(spec.CAMERA_SUFFIX):
            # Every frame needs its own time: Lance steps timestamps strictly increase.
            add(topic, f"{same} frames repeat a timestamp")
        if len(t) > 1:
            gap = float(np.diff(np.sort(t)).max()) / 1e9
            if gap > spec.MAX_GAP_S:
                add(topic, f"{gap:.2f} s gap, max {spec.MAX_GAP_S:g} s")
        # Video rates are checked from decoded frames; IMU rate in its own group.
        t_spec = expected.get(topic)
        if t_spec and t_spec.min_hz and t_spec.kind not in ("video", "imu"):
            if s.rate_hz() < t_spec.min_hz * (1 - spec.RATE_TOL):
                add(topic, f"{s.rate_hz():.1f} Hz, need >= {t_spec.min_hz:g}")

    if not streams:
        return out
    start, end = _span(streams)
    for topic, s in sorted(streams.items()):
        late, early = (min(s.times) - start) / 1e9, (end - max(s.times)) / 1e9
        if late > spec.SPAN_TOL_S or early > spec.SPAN_TOL_S:
            add(
                topic,
                f"covers only part of the episode (starts {late:.1f} s late, "
                f"ends {early:.1f} s early)",
            )

    for key, data_ns in (("start_time_unix", start), ("end_time_unix", end)):
        value = ep.info.get(key, "")
        if _int(value) is not None:
            off = abs(_int(value) * 1e6 - data_ns) / 1e9
            if off > spec.METADATA_TIME_TOL_S:
                add(None, f"episode info {key!r} is {off:.1f} s from the data")
    return out


# ---------------------------------------------------------------- 4. calibration


def _quat_ok(q) -> bool:
    v = [q.x, q.y, q.z, q.w]
    return (
        all(math.isfinite(c) for c in v)
        and abs(math.sqrt(sum(c * c for c in v)) - 1) <= spec.UNIT_TOL
    )


def _reach(edges: dict, start: str) -> set[str]:
    reached, todo = {start}, [start]
    while todo:
        for nxt in edges[todo.pop()] - reached:
            reached.add(nxt)
            todo.append(nxt)
    return reached


def check_calibration(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    cameras = sorted(t for t in ep.streams if t.endswith(spec.CAMERA_SUFFIX))
    for cam in cameras:
        try:
            out += _check_camera_calibration(ep, cam)
        except Exception as e:  # a wrong-typed -info / -info-extra
            out.append(
                Problem("calibration", cam, f"could not check: {type(e).__name__}: {e}")
            )
    return out + _check_frames(ep, embodiment, cameras)


def _check_camera_calibration(ep: _Episode, cam: str) -> list[Problem]:
    out = []
    add = _adder(out, "calibration")
    info = ep.streams.get(cam + spec.CALIBRATION_SUFFIX)
    calib = _calibration(ep, cam)
    if calib is None:
        name = cam + spec.CALIBRATION_SUFFIX
        if not info:
            add(cam, f"no {name} calibration")
        elif info.schema != "foxglove.CameraCalibration":
            add(
                cam,
                f"{name} is {info.schema or 'untyped'}, not foxglove.CameraCalibration",
            )
        else:
            add(cam, f"{name} has no decodable message")
        return out
    video = ep.streams.get(cam)
    if video and video.frame_ids:
        named = set(video.frame_ids)
        if named != {str(calib.frame_id)}:
            add(
                cam,
                f"video frames name {sorted(named)} but its calibration names "
                f"{calib.frame_id!r}: they must be the camera's one frame",
            )
    if info.schema != "foxglove.CameraCalibration" or info.n != 1:
        add(
            cam + spec.CALIBRATION_SUFFIX,
            f"expected one foxglove.CameraCalibration, got {info.n} "
            f"{info.schema or 'untyped'} message(s)",
        )
    K = list(getattr(calib, "K", []))
    D = list(getattr(calib, "D", []))
    if not all(math.isfinite(v) for v in K + D):
        add(cam, "K or D has NaN / Inf")
    elif len(K) == 9 and not (0 <= K[2] <= calib.width and 0 <= K[5] <= calib.height):
        add(
            cam,
            f"principal point ({K[2]:g}, {K[5]:g}) is outside the "
            f"{calib.width}x{calib.height} image",
        )
    if len(K) != 9 or K[0] <= 0 or K[4] <= 0 or not calib.width or not calib.height:
        add(cam, "calibration has no valid K (fx, fy > 0) or size")
    model, n_d = str(calib.distortion_model), len(calib.D)
    if model == "" and n_d == 0:
        pass  # pinhole
    elif model not in spec.DISTORTION_MODELS:
        add(
            cam,
            f"distortion_model {model!r}, expected one of "
            f"{list(spec.DISTORTION_MODELS)}",
        )
    elif n_d != spec.DISTORTION_MODELS[model]:
        add(
            cam,
            f"{model} needs {spec.DISTORTION_MODELS[model]} D coefficients, got {n_d}",
        )

    extra = ep.streams.get(cam + spec.CAMERA_EXTRA_SUFFIX)
    if extra and extra.msgs:
        m = extra.msgs[0]
        if m.upright_rotation_cw_deg not in spec.UPRIGHT_ROTATIONS:
            add(
                cam,
                f"upright_rotation_cw_deg {m.upright_rotation_cw_deg}, expected "
                f"one of {list(spec.UPRIGHT_ROTATIONS)}",
            )
        if not (math.isfinite(m.valid_radius) and m.valid_radius >= 0):
            add(cam, f"valid_radius {m.valid_radius} is not >= 0")
        if m.frame_id != str(getattr(calib, "frame_id", m.frame_id)):
            add(
                cam + spec.CAMERA_EXTRA_SUFFIX,
                f"frame_id {m.frame_id!r} differs from the camera's -info",
            )
    return out


def _check_frames(ep: _Episode, embodiment: str, cameras: list[str]) -> list[Problem]:
    out = []
    add = _adder(out, "calibration")
    # Frame graph: /tf_static edges plus the moving links whose topic is present.
    edges = defaultdict(set)
    static = defaultdict(set)  # /tf_static alone: what is rigidly mounted to what
    tf = ep.streams.get("/tf_static")
    bad_rot = bad_tr = 0
    for msg in tf.msgs if tf else ():
        for x in msg.transforms:
            for g in (edges, static):
                g[x.parent_frame_id].add(x.child_frame_id)
                g[x.child_frame_id].add(x.parent_frame_id)
            bad_rot += not _quat_ok(x.rotation)
            t = x.translation
            bad_tr += not all(math.isfinite(v) for v in (t.x, t.y, t.z))
    if bad_rot:
        add("/tf_static", f"{bad_rot} transforms have a non-unit rotation")
    if bad_tr:
        add("/tf_static", f"{bad_tr} transforms have a NaN / Inf translation")
    sides = spec.arm_sides(embodiment)
    profile = spec.profile_for(embodiment)
    for link in profile.moving_links:
        if link.topic in ep.streams and (
            not link.parent.endswith("_arm_base") or link.parent.split("_")[0] in sides
        ):
            edges[link.parent].add(link.child)
            edges[link.child].add(link.parent)
    reached = _reach(edges, spec.WORLD_FRAME)

    # A head-worn rig's steps camera rides on the head: its frame must be rigidly
    # mounted to the head (a /tf_static path), which is how the steps are placed.
    if any(link.child == spec.HEAD_FRAME for link in profile.moving_links):
        clock = _calibration(ep, profile.step_clock)
        fid = str(clock.frame_id) if clock is not None else ""
        # (An unplaced frame is already reported below as having no path to world.)
        if fid in reached and fid not in _reach(static, spec.HEAD_FRAME):
            add(
                profile.step_clock,
                f"frame {clock.frame_id!r} has no fixed path to "
                f"{spec.HEAD_FRAME!r} via /tf_static: the steps camera must be "
                "mounted on the head",
            )

    # Cameras and the IMU must name their frame; every frame any topic names (hands,
    # wrist poses, gaze, ...) must be placed in the world.
    must_name = set(cameras) | {"/imu"}
    for topic, s in sorted(ep.streams.items()):
        skip = topic == "/tf_static" or topic.endswith(
            (spec.CALIBRATION_SUFFIX, spec.CAMERA_EXTRA_SUFFIX)
        )
        if skip or not s.frame_ids:
            continue  # a camera's -info / -info-extra share its frame, checked above
        named = set(s.frame_ids)
        if "" in named and topic in must_name:
            add(topic, "no frame_id, so it cannot be placed")
        for frame in sorted(named - reached - {""}):
            add(
                topic,
                f"frame {frame!r} has no path to {spec.WORLD_FRAME!r} via /tf_static",
            )
    head = ep.streams.get("/head-pose")
    if head and head.frame_ids and set(head.frame_ids) != {spec.WORLD_FRAME}:
        add("/head-pose", f"poses must be in frame {spec.WORLD_FRAME!r} (T_world_head)")
    return out


# ---------------------------------------------------------------- 5. imu


def check_imu(ep: _Episode) -> list[Problem]:
    out = []
    add = _adder(out, "imu")
    s = ep.streams.get("/imu")
    rows = _rows(s) if s and s.width else ()
    if len(rows):
        if s.rate_hz() < spec.MIN_IMU_HZ * (1 - spec.RATE_TOL):
            add("/imu", f"{s.rate_hz():.0f} Hz, need >= {spec.MIN_IMU_HZ:g}")
        acc, gyr = rows[:, :3], rows[:, 3:]
        if not (np.isfinite(acc).all() and np.isfinite(gyr).all()):
            add("/imu", "NaN/Inf samples")
        else:
            # Stillest window = lowest mean rotation rate; a raw IMU there reads ~9.81 m/s^2.
            w = max(1, min(len(acc), round(s.rate_hz() * spec.STILL_WINDOW_S)))
            gyro_mag = np.convolve(
                np.linalg.norm(gyr, axis=1), np.ones(w) / w, mode="valid"
            )
            i = int(np.argmin(gyro_mag))
            still = float(np.linalg.norm(acc[i : i + w], axis=1).mean())
            if abs(still - spec.GRAVITY) > spec.GRAVITY_TOL:
                add(
                    "/imu",
                    f"mean |accel| at rest = {still:.2f} m/s^2, expected "
                    f"~{spec.GRAVITY} (gravity removed, or wrong units?)",
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
            add("/imu-info", f"noise values must be > 0: {', '.join(bad)}")
    return out


# ---------------------------------------------------------------- 6. poses


def _pose_ok(pose) -> bool:
    p = pose.position
    return _quat_ok(pose.orientation) and all(math.isfinite(v) for v in (p.x, p.y, p.z))


def _check_pose(ep: _Episode, name: str, rows: np.ndarray, add: Add) -> None:
    bad = int((rows[:, 0] == 0).sum())
    if bad:
        add(name, f"{bad}/{len(rows)} poses have NaNs or a non-unit rotation")


def _check_hand_keypoints(ep: _Episode, name: str, rows: np.ndarray, add: Add) -> None:
    joints = _hand_joint_count(ep)
    valid, n_pos, wrist, wrist_ok, conf_ok, n_conf, finite, zeros = rows.T.astype(bool)
    n_pos, n_conf = rows[:, 1], rows[:, 5]
    n = len(rows)
    shape = int((~np.isin(n_pos, (0, 3 * joints))).sum())
    if shape:
        add(
            name,
            f"{shape}/{n} messages have neither {3 * joints} positions "
            f"({joints} hand-joints x xyz) nor none",
        )
    empty = int((valid & (n_pos == 0) & ~wrist).sum())
    if empty:
        add(name, f"{empty} valid messages have neither positions nor wrist_pose")
    conf = int((~conf_ok | ~np.isin(n_conf, (0, joints))).sum())
    if conf:
        add(
            name,
            f"{conf}/{n} messages have a confidence outside [0, 1] or "
            f"joint_confidence not {joints} values",
        )
    bad_wrist = int((valid & wrist & ~wrist_ok).sum())
    if bad_wrist:
        add(name, f"{bad_wrist} valid wrist poses have NaNs or a non-unit rotation")
    full = valid & (n_pos == 3 * joints)
    nan, zero = int((full & ~finite).sum()), int((full & zeros).sum())
    if nan:
        add(name, f"{nan} valid messages have NaN/Inf positions")
    if zero:
        add(
            name,
            f"{zero} messages are all-zero with valid=true; lost tracking must be "
            "valid=false",
        )


def _check_eye_gaze(ep: _Episode, name: str, rows: np.ndarray, add: Add) -> None:
    bad = int(rows[:, 0].sum())
    if bad:
        add(name, f"{bad} valid messages have NaN/Inf")


def _rigid_4x4(m) -> bool:
    pose = np.asarray(m.ee_pose, dtype=np.float64)
    if (
        len(pose) != 16
        or not np.isfinite(pose).all()
        or not np.isfinite(np.asarray(m.position)).all()
    ):
        return False
    T = pose.reshape(4, 4)
    R = T[:3, :3]
    return (
        np.abs(R.T @ R - np.eye(3)).max() <= spec.UNIT_TOL
        and abs(np.linalg.det(R) - 1) <= spec.UNIT_TOL
        and np.abs(T[3] - [0, 0, 0, 1]).max() <= spec.UNIT_TOL
    )


def _check_arm(ep: _Episode, name: str, rows: np.ndarray, add: Add) -> None:
    names, bad = int(rows[:, 0].sum()), int((rows[:, 1] == 0).sum())
    if names:
        add(
            name, f"{names}/{len(rows)} messages' joint arrays do not match joint_names"
        )
    if bad:
        add(name, f"{bad}/{len(rows)} poses are not a finite rigid 4x4 transform")


def _check_gripper(ep: _Episode, name: str, rows: np.ndarray, add: Add) -> None:
    lo, hi = spec.GRIPPER_RANGE
    vals = rows[:, 0]
    bad = int((~np.isfinite(vals) | (vals < lo) | (vals > hi)).sum())
    if bad:
        add(name, f"{bad}/{len(rows)} gripper values outside [{lo:g}, {hi:g}]")


# Topic.kind -> its group-6 check.
_POSE_CHECKS = {
    "pose": _check_pose,
    "hand_keypoints": _check_hand_keypoints,
    "eye_gaze": _check_eye_gaze,
    "arm": _check_arm,
    "gripper": _check_gripper,
}


def check_poses(ep: _Episode, embodiment: str) -> list[Problem]:
    out = []
    add = _adder(out, "poses")
    for t in spec.topics_for(embodiment):
        s = ep.streams.get(t.name)
        check = _POSE_CHECKS.get(t.kind)
        rows = _rows(s) if check and s and s.width else ()
        if len(rows):
            check(ep, t.name, rows, add)
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
    add = _adder(out, "labels")
    streams = _time_series(ep, embodiment)
    if not streams:
        return out
    start, end = _span(streams)
    tol = spec.SPAN_TOL_S * 1e9
    tasks = _spans(ep.streams.get("/task"))
    labels = _spans(ep.streams.get("/subtask-annotation"))
    # A point label becomes one steps-camera frame wide in Lance (as the converter writes
    # it), and Lance's annotation review wants the whole span inside a task.
    step_clock = spec.profile_for(embodiment).step_clock
    cam = ep.streams.get(step_clock)
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
            add(topic, f"{backwards} spans do not end after they start")
        if outside:
            add(topic, f"{outside}/{len(spans)} spans fall outside the recorded data")
    if tasks:
        orphans = sum(not any(tb <= b and e <= te for tb, te in tasks) for b, e in wide)
        if orphans:
            add(
                "/subtask-annotation",
                f"{orphans}/{len(labels)} labels do not lie inside one /task span",
            )
    elif len(cam_t) and labels:
        # The task made from the episode info: [first frame, last frame + one frame).
        first, last = int(cam_t[0]), int(cam_t[-1]) + frame
        beyond = sum(b < first or e > last for b, e in wide)
        if beyond:
            add(
                "/subtask-annotation",
                f"{beyond}/{len(labels)} labels fall outside {step_clock}'s frames "
                "(no /task)",
            )
    s = ep.streams.get("/subtask-annotation")
    types = (
        {m.annotation_type for m in s.msgs if hasattr(m, "annotation_type")}
        if s
        else set()
    )
    bad = sorted(types - {"", *spec.ANNOTATION_TYPES})
    if bad:
        add(
            "/subtask-annotation",
            f"annotation_type {bad}, expected one of {list(spec.ANNOTATION_TYPES)}",
        )
    return out


# ---------------------------------------------------------------- entry points


@dataclass
class Result:
    embodiment: str  # declared, or guessed from the topics; "" if unreadable
    recording_id: str  # the Lance recording_id it will get; "" without lab / source id
    problems: list[Problem]
    # Required metadata waived via missing_metadata, accepted with allow_missing.
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_episode(episode: Path, *, allow_missing: bool = False) -> Result:
    """Validate one episode directory (or its episode.mcap).

    Required metadata waived in `missing_metadata` is a problem unless
    `allow_missing`, then a warning.
    """
    episode = Path(episode)
    path = episode / spec.EPISODE_FILE if episode.is_dir() else episode
    if not path.exists():
        return Result(
            "", "", [Problem("present", None, f"no {spec.EPISODE_FILE} in {episode}")]
        )
    try:
        ep = _read(path)
    except Exception as e:  # truncated or not an MCAP at all
        message = f"cannot read {path.name}: {type(e).__name__}: {e}"
        return Result("", "", [Problem("present", None, message)])
    problems = _merge_annotation_file(ep, path)
    embodiment = _embodiment(ep)
    warnings: list[str] = []

    def present() -> list[Problem]:
        found, warns = check_present(ep, embodiment, allow_missing)
        warnings.extend(warns)
        return found

    for group, check in (
        ("present", present),
        ("video", lambda: check_video(ep)),
        ("clocks", lambda: check_clocks(ep, embodiment)),
        ("calibration", lambda: check_calibration(ep, embodiment)),
        ("imu", lambda: check_imu(ep)),
        ("poses", lambda: check_poses(ep, embodiment)),
        ("labels", lambda: check_labels(ep, embodiment)),
    ):
        try:
            problems += check()
        except Exception as e:  # a topic of the wrong type the group's checks assume
            problems.append(
                Problem(
                    group,
                    None,
                    f"could not check: {type(e).__name__}: {e} (see 'type is' problems)",
                )
            )
    lab = ep.info.get("lab", "")
    source_id = ep.info.get("vendor_recording_id", "")
    rid = spec.recording_id(lab, source_id) if lab and source_id else ""
    return Result(embodiment, rid, problems, warnings)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "episodes",
        nargs="+",
        type=Path,
        help="episode_<id>/ dirs or episode.mcap files",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Accept required metadata listed in missing_metadata (warn instead of fail)",
    )
    args = p.parse_args(argv)
    import av

    av.logging.set_level(av.logging.PANIC)  # lone P-frames log "non-existing PPS"
    failed = 0
    for episode in args.episodes:
        r = validate_episode(episode, allow_missing=args.allow_missing)
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
        for warning in r.warnings:
            print(f"  warning: {warning}")
    if len(args.episodes) > 1:
        print(f"\n{len(args.episodes) - failed}/{len(args.episodes)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
