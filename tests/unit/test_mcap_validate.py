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
    assert r.recording_id == spec.recording_id("aria", "acme-rec-0001")


# (embodiment, breakage, [(group, topic, substring of the message), ...])
BREAKAGES = [
    # 1. everything there
    ("human_bimanual", {"drop_topic": "/imu-info"}, [("present", "/imu-info", "missing required topic")]),
    ("human_bimanual", {"drop_metadata_key": "vendor"}, [("present", None, "'vendor' missing")]),
    ("human_bimanual", {"drop_metadata_key": "site"}, [("present", None, "'site' missing")]),
    ("human_bimanual", {"drop_metadata_key": "license"}, [("present", None, "'license' missing")]),
    ("human_bimanual", {"drop_metadata_key": "vendor-recording-id"}, [("present", None, "'vendor-recording-id' missing")]),
    ("human_bimanual", {"drop_metadata_key": "hand-joints"}, [("present", None, "'hand-joints' missing")]),
    ("human_bimanual", {"bad_site_slug": True}, [("present", None, "not a lowercase slug")]),
    ("human_bimanual", {"metadata_override": {"operator-id": "jane.doe@gatech.edu"}}, [("present", None, "not a SHA-256 hex digest")]),
    ("human_bimanual", {"metadata_override": {"shutter": "rolling"}}, [("present", None, "expected 'global' or 'rolling:<readout>ms'")]),
    ("human_bimanual", {"metadata_override": {"camera-timestamp-ref": "middle"}}, [("present", None, "'camera-timestamp-ref' = 'middle'")]),
    ("human_bimanual", {"annotation_metadata": True}, [("present", None, "annotation.mcap carries metadata records")]),
    ("human_bimanual", {"metadata_override": {"start-time-unix": "--1760000000000"}}, [("present", None, "is not an integer")]),
    ("human_bimanual", {"metadata_override": {"end-time-unix": "1760000002000²"}}, [("present", None, "is not an integer")]),
    ("human_bimanual", {"unknown_hand_joint": True}, [("present", None, "unknown joints ['thumb_base']")]),
    ("human_bimanual", {"declared_embodiment": "robot_dog"}, [("present", None, "'embodiment' = 'robot_dog'")]),
    ("human_bimanual", {"declared_embodiment": "human_right_arm"}, [("present", "/left-hand-keypoints", "does not fit embodiment")]),
    ("human_bimanual", {"stray_annotation_topic": True}, [("present", "/instruction", "belong in annotation.mcap")]),
    ("eva_bimanual", {"empty_arm_pose": True}, [
        ("present", "/right-arm-state", "'ee_pose' empty on 61/61"),
        ("poses", "/right-arm-state", "not a finite rigid 4x4"),
    ]),
    # 2. video
    ("human_bimanual", {"video_format": "vp9"}, [("video", "/main-camera", "format 'vp9'")]),
    ("human_bimanual", {"skip_first_keyframe": True}, [("video", "/main-camera", "not a keyframe")]),
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
    assert spec.recording_id("aria", "acme-rec-0001") in out


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


def test_embodiments_match_enum():
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT

    assert set(spec.EMBODIMENTS) == {e.name.lower() for e in EMBODIMENT}


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


def test_wrong_message_type_is_reported_not_raised(tmp_path):
    from egomimic.rldb.mcap import schemas

    def imu_as_instruction(topic, msg, t):
        if topic == "/imu":
            return schemas.Instruction(data="not an imu"), t

    found = problems(tmp_path, tamper=imu_as_instruction)
    assert (
        "present",
        "/imu",
        "type is 'egoverse.Instruction', expected 'egoverse.Imu'",
    ) in found
    assert any(g == "imu" and "could not check" in m for g, _, m in found), found
