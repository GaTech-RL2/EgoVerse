"""Vendor MCAP validator (egomimic/rldb/mcap) on tiny synthetic episodes.

Valid episodes of every profile pass; each breakage in fixtures.mcap_episodes produces
exactly its expected problems and nothing else.
"""

from pathlib import Path

import pytest
from fixtures.mcap_episodes import T0, write_mcap_episode

from egomimic.rldb.mcap import schemas, spec
from egomimic.rldb.mcap.validate import main, validate_episode

PROTO = Path(schemas.__file__).parent / "proto" / "egoverse.proto"


def problems(tmp_path, embodiment="human_bimanual", **kw) -> list[tuple]:
    r = validate_episode(write_mcap_episode(tmp_path, embodiment, **kw))
    return [(p.group, p.topic, p.message) for p in r.problems]


@pytest.mark.parametrize(
    "embodiment, kw",
    [
        ("human_bimanual", {}),
        ("human_right_arm", {}),
        ("human_left_arm", {}),
        ("eva_bimanual", {}),
        ("eva_left_arm", {}),
        ("human_bimanual", {"labels_in": "episode"}),
        ("eva_bimanual", {"labels_in": None}),
        ("human_bimanual", {"hands_wrist_only": True}),
        ("human_bimanual", {"drop_topic": "/main-camera-info-extra"}),
        ("human_bimanual", {"labels_without_tasks": True}),
        ("human_bimanual", {"labels_without_tasks": True, "label_at_last_frame": True}),
        (
            "human_bimanual",
            {"distortion_model": "fisheye624", "distortion_coeffs": (0.1,) * 12},
        ),
    ],
)
def test_valid_episode_passes(tmp_path, embodiment, kw):
    r = validate_episode(write_mcap_episode(tmp_path, embodiment, **kw))
    assert r.problems == []
    assert r.embodiment == embodiment
    assert r.recording_id == spec.recording_id("rl2", "acme-rec-0001")


# (embodiment, breakage, [(group, topic, substring of the message), ...])
BREAKAGES = [
    # 1. everything there
    ("human_bimanual", {"drop_topic": "/imu-info"}, [("present", "/imu-info", "missing required topic")]),
    ("human_bimanual", {"drop_metadata_key": "operator"}, [("present", None, "Missing metadata field: operator")]),
    ("human_bimanual", {"metadata_override": {"scene": "unknown"}}, [("present", None, "scene='unknown' is a placeholder")]),
    ("human_bimanual", {"metadata_override": {"objects": "shirt,,basket"}}, [("present", None, "objects entries must be real names")]),
    ("human_bimanual", {"metadata_override": {"objects": " shirt, basket"}}, [("present", None, "objects=' shirt, basket' has leading/trailing whitespace")]),
    ("human_bimanual", {"metadata_override": {"missing_metadata": "objects"}}, [("present", None, "objects has both a value")]),
    ("human_bimanual", {"metadata_override": {"missing_metadata": "scen"}}, [("present", None, "names unknown fields ['scen']")]),
    ("human_bimanual", {"drop_metadata_key": "license"}, [("present", None, "'license' missing")]),
    ("human_bimanual", {"drop_metadata_key": "vendor_recording_id"}, [("present", None, "'vendor_recording_id' missing")]),
    ("human_bimanual", {"drop_metadata_key": "hand_joints"}, [("present", None, "'hand_joints' missing")]),
    ("human_bimanual", {"metadata_override": {"missing_metadata": "lab"}}, [("present", None, "['lab'] cannot be waived"), ("present", None, "lab has both a value")]),
    ("human_bimanual", {"metadata_override": {"missing_metadata": "operator"}}, [("present", None, "['operator'] cannot be waived"), ("present", None, "operator has both a value")]),
    ("human_bimanual", {"metadata_override": {"missing_metadata": "task_name"}}, [("present", None, "['task_name'] cannot be waived"), ("present", None, "task_name has both a value")]),
    ("human_bimanual", {"metadata_override": {"site": "rl2"}}, [("present", None, "unknown episode info key 'site'")]),
    ("human_bimanual", {"metadata_override": {"rig-name": "test_rig"}}, [("present", None, "unknown episode info key 'rig-name'; did you mean 'rig_name'?")]),
    ("human_bimanual", {"metadata_override": {"license": "tbd"}}, [("present", None, "'license' = 'tbd' is a placeholder")]),
    ("human_bimanual", {"metadata_override": {"lab": "RL2 Lab"}}, [("present", None, "lab='RL2 Lab' must be a short, stable, lowercase identifier")]),
    ("human_bimanual", {"metadata_override": {"shutter": "rolling"}}, [("present", None, "expected 'global' or 'rolling:<readout>ms'")]),
    ("human_bimanual", {"metadata_override": {"camera_timestamp_ref": "middle"}}, [("present", None, "'camera_timestamp_ref' = 'middle'")]),
    ("human_bimanual", {"annotation_metadata": True}, [("present", None, "annotation.mcap carries metadata records")]),
    ("human_bimanual", {"metadata_override": {"start_time_unix": "--1760000000000"}}, [("present", None, "is not an integer")]),
    ("human_bimanual", {"metadata_override": {"end_time_unix": "1760000002000²"}}, [("present", None, "is not an integer")]),
    ("human_bimanual", {"unknown_hand_joint": True}, [("present", None, "unknown joints ['thumb_base']")]),
    ("human_bimanual", {"declared_embodiment": "robot_dog"}, [("present", None, "'embodiment' = 'robot_dog'")]),
    ("human_bimanual", {"declared_embodiment": "human_right_arm"}, [("present", "/left-hand-keypoints", "does not fit embodiment")]),
    ("human_bimanual", {"stray_annotation_topic": True}, [("present", "/imu", "belong in annotation.mcap")]),
    ("eva_bimanual", {"empty_arm_pose": True}, [
        ("present", "/right-arm-state", "'ee_pose' empty on 61/61"),
        ("poses", "/right-arm-state", "not a finite rigid 4x4"),
    ]),
    # 2. video
    ("human_bimanual", {"video_format": "vp9"}, [("video", "/main-camera", "format 'vp9'")]),
    ("human_bimanual", {"skip_first_keyframe": True}, [("video", "/main-camera", "not a keyframe")]),
    ("human_bimanual", {"b_frames": True}, [("video", "/main-camera", "uses B-frames; re-encode without them (x264: -bf 0)")]),
    ("eva_bimanual", {"b_frames": True}, [("video", c, "uses B-frames") for c in ("/left-wrist-camera", "/right-wrist-camera", "/top-camera")]),
    ("human_bimanual", {"calibration_width": 128}, [("video", "/main-camera", "calibration says 128x48")]),
    ("human_bimanual", {"video_fps": 15}, [("video", "/main-camera", "15.0 fps")]),
    ("human_bimanual", {"multi_frame_video": True}, [("video", "/main-camera", "exactly one frame")]),
    ("eva_bimanual", {"multi_frame_video": True}, [("video", "/top-camera", "exactly one frame")]),
    # 3. clocks
    ("human_bimanual", {"backwards_head_pose": True}, [("clocks", "/head-pose", "backwards 1 times")]),
    ("human_bimanual", {"hand_gap_s": 0.8}, [("clocks", "/left-hand-keypoints", "0.80 s gap")]),
    ("human_bimanual", {"device_clock_imu_info": True}, [("clocks", "/imu-info", "before 2015")]),
    ("human_bimanual", {"repeat_frame_time": True}, [("clocks", "/main-camera", "1 frames repeat a timestamp")]),
    # 4. calibration
    ("human_bimanual", {"drop_topic": "/main-camera-info"}, [("calibration", "/main-camera", "no /main-camera-info")]),
    ("human_bimanual", {"unlink_main_camera": True}, [("calibration", "/main-camera", "no path to 'world'")]),
    ("human_bimanual", {"bad_tf_rotation": True}, [("calibration", "/tf_static", "non-unit rotation")]),
    ("human_bimanual", {"bad_upright_rotation": True}, [("calibration", "/main-camera", "upright_rotation_cw_deg 45")]),
    ("human_bimanual", {"calibration_twice": True}, [("calibration", "/main-camera-info", "got 2 foxglove.CameraCalibration")]),
    ("human_bimanual", {"principal_x": 500.0}, [("calibration", "/main-camera", "principal point (500, 24) is outside")]),
    ("human_bimanual", {"distortion_model": "equidistant", "distortion_coeffs": (0.1,) * 4}, [("calibration", "/main-camera", "distortion_model 'equidistant'")]),
    ("human_bimanual", {"distortion_model": "fisheye624", "distortion_coeffs": (0.1,) * 5}, [("calibration", "/main-camera", "needs 12 D coefficients, got 5")]),
    # 5. imu
    ("human_bimanual", {"imu_hz": 100}, [("imu", "/imu", "100 Hz")]),
    ("human_bimanual", {"gravity_removed": True}, [("imu", "/imu", "gravity removed")]),
    # 6. poses
    ("human_bimanual", {"nan_head_pose": True}, [("poses", "/head-pose", "1/61 poses")]),
    ("human_bimanual", {"zeroed_left_hand": True}, [("poses", "/left-hand-keypoints", "all-zero with valid=true")]),
    ("human_bimanual", {"bad_hand_confidence": True}, [("poses", "/left-hand-keypoints", "confidence outside [0, 1]")]),
    ("human_bimanual", {"bad_wrist_pose": True}, [("poses", "/left-hand-keypoints", "wrist poses")]),
    ("human_bimanual", {"nan_eye_gaze": True}, [("poses", "/eye-gaze", "NaN/Inf")]),
    ("eva_bimanual", {"gripper_out_of_range": True}, [("poses", "/right-ee-state", "outside [0, 1]")]),
    ("eva_bimanual", {"bad_arm_pose": True}, [("poses", "/right-arm-state", "1/61 poses")]),
    ("eva_bimanual", {"joint_names_mismatch": True}, [("poses", "/right-arm-state", "do not match joint_names")]),
    # 7. tasks & labels
    ("human_bimanual", {"task_past_end": True}, [("labels", "/task", "outside the recorded data")]),
    ("eva_bimanual", {"label_outside_tasks": True}, [("labels", "/subtask-annotation", "inside one /task")]),
    ("human_bimanual", {"label_crosses_tasks": True}, [("labels", "/subtask-annotation", "inside one /task")]),
    ("human_bimanual", {"bad_annotation_type": True}, [("labels", "/subtask-annotation", "['summary']")]),
    ("human_bimanual", {"labels_without_tasks": True, "label_past_video": True}, [("labels", "/subtask-annotation", "outside /main-camera's frames")]),
]  # fmt: skip


@pytest.mark.parametrize(
    "embodiment, kw, expected",
    BREAKAGES,
    ids=[f"{e}-{next(iter(kw))}={next(iter(kw.values()))}" for e, kw, _ in BREAKAGES],
)
def test_breakage_reports_exactly_its_problem(tmp_path, embodiment, kw, expected):
    found = problems(tmp_path, embodiment, **kw)
    assert len(found) == len(expected), found
    for (group, topic, message), (g, t, sub) in zip(sorted(found), sorted(expected)):
        assert (group, topic) == (g, t), found
        assert sub in message, found


def test_cli_pass_fail_and_summary(tmp_path, capsys):
    good = write_mcap_episode(tmp_path, episode_id="good")
    bad = write_mcap_episode(tmp_path, episode_id="bad", imu_hz=100)
    assert main([str(good)]) == 0
    assert main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert f"PASS {good}" in out
    assert f"FAIL {bad}" in out and "[imu] /imu: 100 Hz" in out
    assert "1/2 passed" in out
    assert spec.recording_id("rl2", "acme-rec-0001") in out


def test_waived_metadata_needs_allow_missing(tmp_path, capsys):
    ep = write_mcap_episode(
        tmp_path,
        drop_metadata_key="scene",
        metadata_override={"missing_metadata": "scene"},
    )
    strict = validate_episode(ep)
    assert [p.message for p in strict.problems] == [
        "scene: marked as not available (MissingMetadata). We heavily recommend "
        "providing it; pass --allow-missing to accept the episode anyway."
    ]
    lenient = validate_episode(ep, allow_missing=True)
    assert lenient.ok and lenient.warnings == [
        "scene: marked as not available (MissingMetadata)"
    ]
    assert main([str(ep)]) == 1
    assert main([str(ep), "--allow-missing"]) == 0
    assert "warning: scene: marked as not available" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["lab", "task_name", "operator"])
def test_unwaivable_metadata_fails_even_with_allow_missing(tmp_path, name):
    ep = write_mcap_episode(
        tmp_path, drop_metadata_key=name, metadata_override={"missing_metadata": name}
    )
    r = validate_episode(ep, allow_missing=True)
    assert not r.ok
    assert any(f"['{name}'] cannot be waived" in p.message for p in r.problems)


def test_missing_and_unreadable_episodes(tmp_path):
    r = validate_episode(tmp_path / "episode_nope")
    assert [p.message for p in r.problems] == [
        f"no episode.mcap in {tmp_path / 'episode_nope'}"
    ]
    junk = tmp_path / "episode_junk"
    junk.mkdir()
    (junk / "episode.mcap").write_bytes(b"not an mcap")
    (p,) = validate_episode(junk).problems
    assert p.message.startswith("cannot read episode.mcap")


def test_unreadable_annotation_file(tmp_path):
    ep = write_mcap_episode(tmp_path, labels_in=None)
    (ep / spec.ANNOTATION_FILE).write_bytes(b"garbage")
    (p,) = validate_episode(ep).problems
    assert p.message.startswith("cannot read annotation.mcap")


def test_proto_file_matches_schemas():
    assert PROTO.read_text() == schemas.render_proto(), (
        "regenerate: python -m egomimic.rldb.mcap.schemas > " + str(PROTO)
    )


def test_proto_field_numbers_are_unique():
    for m in schemas.MESSAGES:
        numbers = [f.number for f in m.fields]
        assert len(set(numbers)) == len(numbers), f"{m.name} reuses a field number"
        assert all(n >= 1 for n in numbers), m.name


def test_only_scene_and_objects_can_be_waived():
    from egomimic.rldb.zarr.required_metadata import REQUIRED_METADATA_FIELDS

    assert set(spec.UNWAIVABLE_FIELDS) == {"task_name", "operator", "lab"}
    waivable = set(REQUIRED_METADATA_FIELDS) - set(spec.UNWAIVABLE_FIELDS)
    assert waivable == {"scene", "objects"}


def test_embodiments_match_enum():
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT
    from egomimic.rldb.zarr import episode_rules

    assert set(episode_rules.EMBODIMENTS) == {e.name.lower() for e in EMBODIMENT}


def test_mano_joints_match_human_finger_edges():
    # Canonical MANO: wrist, then 4 joints per finger, thumb to pinky.
    assert len(spec.MANO_JOINTS) == 21 and spec.MANO_JOINTS[0] == "wrist"
    assert spec.MANO_JOINTS[1:5] == ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip")
    assert spec.MANO_JOINTS[17:] == ("pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip")


def test_lance_ids_and_annotation_types_match_lance_schema():
    lance_schema = pytest.importorskip("egomimic.rldb.lance.schema")
    assert spec.recording_id("aria", "x") == lance_schema.recording_id("aria", "x")
    if hasattr(lance_schema, "ANNOTATION_TYPES"):
        assert spec.ANNOTATION_TYPES == lance_schema.ANNOTATION_TYPES


def _late_hand_frame(topic, msg, t):
    if topic == "/left-hand-keypoints" and t > T0 + 10**9:
        msg.frame_id = "tracker"  # message 0 is fine; later ones are not linked
        return msg, t


def _head_in_odom(topic, msg, t):
    if topic == "/head-pose":
        msg.frame_id = "odom"
        return msg, t
    if (
        topic == "/tf_static"
    ):  # odom does reach world, but /head-pose must be T_world_head
        x = msg.transforms.add(parent_frame_id="world", child_frame_id="odom")
        x.rotation.w = 1.0
        return msg, t


def _nan_static_translation(topic, msg, t):
    if topic == "/tf_static":
        msg.transforms[0].translation.x = float("nan")
        return msg, t


def _main_camera_fixed_in_world(topic, msg, t):
    """A consistent camera frame, rigidly mounted to the world instead of the head."""
    if topic.startswith("/main-camera"):
        msg.frame_id = "room_cam"
        return msg, t
    if topic == "/tf_static":
        x = msg.transforms.add(parent_frame_id="world", child_frame_id="room_cam")
        x.rotation.w = 1.0
        return msg, t


def _point_label_at_task_end(topic, msg, t):
    if topic == "/subtask-annotation" and not msg.end_time_unix_ns:
        return msg, T0 + 10**9 - 10_000_000  # 10 ms before task 1 ends: not a frame


@pytest.mark.parametrize(
    "tamper, expected",
    [
        (
            _late_hand_frame,
            ("calibration", "/left-hand-keypoints", "'tracker' has no path"),
        ),
        (_head_in_odom, ("calibration", "/head-pose", "must be in frame 'world'")),
        (
            _nan_static_translation,
            ("calibration", "/tf_static", "1 transforms have a NaN / Inf translation"),
        ),
        (
            _main_camera_fixed_in_world,
            ("calibration", "/main-camera", "no fixed path to 'head'"),
        ),
        (
            _point_label_at_task_end,
            ("labels", "/subtask-annotation", "inside one /task"),
        ),
    ],
)
def test_tampered_episode_reports_exactly_its_problem(tmp_path, tamper, expected):
    found = problems(tmp_path, tamper=tamper)
    assert len(found) == 1, found
    group, topic, message = found[0]
    assert (group, topic) == expected[:2] and expected[2] in message, found


def test_calibration_naming_another_frame_than_the_video_is_caught(tmp_path):
    """The calibration (and -info-extra) moved to `world`, the video frames not: the
    camera has two frames, and the steps camera is no longer on the head."""

    def calib_in_world(topic, msg, t):
        if topic in ("/main-camera-info", "/main-camera-info-extra"):
            msg.frame_id = "world"
            return msg, t

    found = problems(tmp_path, tamper=calib_in_world)
    assert ("calibration", "/main-camera") == found[0][:2]
    assert any("video frames name ['main-camera']" in m for *_, m in found), found
    assert any("no fixed path to 'head'" in m for *_, m in found), found


def test_wrong_message_type_is_reported_not_raised(tmp_path):
    from egomimic.rldb.mcap import schemas

    def imu_as_task(topic, msg, t):
        if topic == "/imu":
            return schemas.Task(name="not an imu"), t

    found = problems(tmp_path, tamper=imu_as_task)
    assert (
        "present",
        "/imu",
        "type is 'egoverse.Task', expected 'egoverse.Imu'",
    ) in found
    assert any(g == "imu" and "could not check" in m for g, _, m in found), found


def _rewrite(path: Path, extra=None, **writer_options) -> None:
    """Rewrite an MCAP record by record, uncompressed (so a chunk's CRC is its only
    guard). extra(writer, messages, channel_ids, schema_ids) adds records at the end;
    writer_options go to mcap's Writer."""
    from mcap.reader import make_reader
    from mcap.writer import CompressionType, Writer

    with open(path, "rb") as f:
        reader = make_reader(f)
        records = [(r.name, dict(r.metadata)) for r in reader.iter_metadata()]
        messages = list(reader.iter_messages(log_time_order=False))
    schema_ids, channel_ids = {}, {}
    with open(path, "wb") as f:
        w = Writer(f, compression=CompressionType.NONE, **writer_options)
        w.start()
        for name, data in records:
            w.add_metadata(name, data)
        for schema, channel, message in messages:
            if schema.id not in schema_ids:
                schema_ids[schema.id] = w.register_schema(
                    schema.name, schema.encoding, schema.data
                )
            if channel.id not in channel_ids:
                channel_ids[channel.id] = w.register_channel(
                    channel.topic, channel.message_encoding, schema_ids[schema.id]
                )
            w.add_message(
                channel_ids[channel.id],
                message.log_time,
                message.data,
                message.publish_time,
            )
        if extra:
            extra(w, messages, channel_ids, schema_ids)
        w.finish()


def test_corrupt_chunk_fails_crc(tmp_path):
    from fixtures.mcap_episodes import h264_packets

    ep = write_mcap_episode(tmp_path)
    path = ep / spec.EPISODE_FILE
    _rewrite(path)
    data = bytearray(path.read_bytes())
    i = data.find(h264_packets(60)[5][-16:])
    assert i > 0
    data[i] ^= 0x01
    path.write_bytes(bytes(data))
    (p,) = validate_episode(ep).problems
    assert p.message.startswith("cannot read episode.mcap") and "CRC" in p.message


def test_corrupt_unchunked_file_fails_crc(tmp_path):
    from fixtures.mcap_episodes import h264_packets

    ep = write_mcap_episode(tmp_path)
    path = ep / spec.EPISODE_FILE
    _rewrite(path, use_chunking=False, enable_data_crcs=True)
    data = bytearray(path.read_bytes())
    i = data.find(h264_packets(60)[5][-16:])
    assert i > 0
    data[i] ^= 0x01
    path.write_bytes(bytes(data))
    (p,) = validate_episode(ep).problems
    assert p.message.startswith("cannot read episode.mcap"), p.message
    assert "crc" in p.message.lower(), p.message


def test_file_without_metadata_index_passes(tmp_path):
    from mcap.writer import IndexType

    ep = write_mcap_episode(tmp_path)
    no_metadata = IndexType.ATTACHMENT | IndexType.CHUNK | IndexType.MESSAGE
    _rewrite(ep / spec.EPISODE_FILE, index_types=no_metadata)
    assert validate_episode(ep).problems == []


def test_unknown_topics_are_not_decoded_and_series_keep_no_messages(tmp_path):
    from egomimic.rldb.mcap.validate import _read

    ep = write_mcap_episode(tmp_path)

    def garbage_topic(w, messages, channel_ids, schema_ids):
        imu_schema = next(s.id for s, c, _ in messages if c.topic == "/imu")
        ch = w.register_channel("/vendor-debug", "protobuf", schema_ids[imu_schema])
        for i in range(3):
            w.add_message(ch, T0 + i, b"\xff not protobuf", T0 + i)

    _rewrite(ep / spec.EPISODE_FILE, extra=garbage_topic)
    assert validate_episode(ep).problems == []
    streams = _read(ep / spec.EPISODE_FILE).streams
    assert streams["/vendor-debug"].n == 3 and not streams["/vendor-debug"].msgs
    for topic in ("/imu", "/head-pose", "/left-hand-keypoints", "/head-pose-quality"):
        assert streams[topic].n > 0 and not streams[topic].msgs, topic
    assert len(streams["/main-camera"].msgs) == spec.VIDEO_SAMPLE_FRAMES


def _after_sample(change):
    """Tamper every /main-camera message past the decoded sample."""
    seen = {"n": 0}

    def tamper(topic, msg, t):
        if topic == "/main-camera":
            seen["n"] += 1
            if seen["n"] > spec.VIDEO_SAMPLE_FRAMES:
                change(msg)
                return msg, t

    return tamper


def _empty_data(msg):
    msg.data = b""


def _no_picture(msg):
    msg.data = b"\x00\x00\x01\x06garbage"  # an SEI NAL only


def _other_frame(msg):
    msg.frame_id = "other"


def _other_format(msg):
    msg.format = "h265"


@pytest.mark.parametrize(
    "change, expected",
    [
        (_empty_data, ("present", "/main-camera", "'data' empty on 60/120 messages")),
        (_no_picture, ("video", "/main-camera", "60/120 messages hold no h264 picture")),
        (_other_frame, ("calibration", "/main-camera", "video frames name ['main-camera', 'other']")),
        (_other_format, ("video", "/main-camera", "format changes within the stream")),
    ],
)  # fmt: skip
def test_video_is_checked_past_the_decoded_sample(tmp_path, change, expected):
    found = problems(tmp_path, video_fps=60, tamper=_after_sample(change))
    assert any(f[:2] == expected[:2] and expected[2] in f[2] for f in found), found


def test_video_format_is_case_sensitive(tmp_path):
    found = problems(tmp_path, video_format="H264")
    assert ("video", "/main-camera") in [f[:2] for f in found], found
    assert any("format 'H264'" in m for *_, m in found), found


def test_duplicate_records_channels_and_once_topics(tmp_path):
    ep = write_mcap_episode(tmp_path)

    def duplicates(w, messages, channel_ids, schema_ids):
        info = {"lab": "rl2"}
        w.add_metadata(spec.METADATA_RECORD, info)
        for schema, channel, m in messages:
            if channel.topic == "/tf_static":
                w.add_message(
                    channel_ids[channel.id], m.log_time + 1, m.data, m.log_time
                )
            if channel.topic == "/imu-info":
                second = w.register_channel(
                    "/imu-info", "protobuf", schema_ids[schema.id]
                )
                w.add_message(second, m.log_time, m.data, m.log_time)

    _rewrite(ep / spec.EPISODE_FILE, extra=duplicates)
    found = [(p.group, p.topic, p.message) for p in validate_episode(ep).problems]
    assert any("2 'session-metadata' records" in m for *_, m in found), found
    assert ("present", "/tf_static", "published 2 times, expected once") in found
    assert any(
        t == "/imu-info" and "published on 2 channels" in m for _, t, m in found
    ), found


def test_wrong_typed_optional_calibration_does_not_hide_other_faults(tmp_path):
    def slam_info_as_task(topic, msg, t):
        if topic == "/main-camera-info":
            return schemas.Task(name="not a calibration"), t

    found = problems(tmp_path, unlink_main_camera=True, tamper=slam_info_as_task)
    assert any(
        g == "calibration"
        and t == "/main-camera"
        and "not foxglove.CameraCalibration" in m
        for g, t, m in found
    ), found
    assert not any("could not check" in m for *_, m in found), found


def test_corrupt_slice_inside_the_sample_fails_decoding(tmp_path):
    seen = {"n": 0}

    def corrupt_frame_10(topic, msg, t):
        if topic == "/main-camera":
            seen["n"] += 1
            if seen["n"] == 11:
                msg.data = b"\x00\x00\x01\x41\x9a\x00" + b"\x5a" * 200
                return msg, t

    found = problems(tmp_path, tamper=corrupt_frame_10)
    assert [f[:2] for f in found] == [("video", "/main-camera")], found
    assert "decode error" in found[0][2]
