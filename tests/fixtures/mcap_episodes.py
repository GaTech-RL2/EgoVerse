"""Tiny synthetic vendor MCAP episodes (spec 0.1), valid unless a breakage is asked for.

`write_mcap_episode` writes `episode_<id>/episode.mcap` with every topic the spec
requires for the embodiment: ~2 s of 64x48 H.264 video, 200 Hz IMU, 30 Hz poses, etc.
It also writes the optional label topics (two /task spans with a /subtask-annotation
each, in annotation.mcap by default) and, for humans, /eye-gaze.
Each keyword-only switch breaks exactly one thing, so a test can assert the validator
reports that problem and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from mcap_protobuf.writer import Writer

from egomimic.rldb.mcap import schemas, spec

T0 = 1_760_000_000 * 10**9  # 2025-10-09, absolute Unix ns
DURATION_S = 2
WIDTH, HEIGHT = 64, 48
IMU_HZ = 200
POSE_HZ = 30
VIDEO_FPS = 30


@lru_cache(maxsize=None)
def _h264_packets(n_frames: int) -> tuple[bytes, ...]:
    """Annex B H.264, one frame per packet, keyframe every 2 frames, SPS/PPS repeated."""
    import av

    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    enc.time_base = Fraction(1, VIDEO_FPS)
    enc.options = {
        "bframes": "0",
        "g": "2",
        "repeat-headers": "1",
        "tune": "zerolatency",
    }
    packets = []
    for i in range(n_frames):
        img = np.full((HEIGHT, WIDTH, 3), i * 4 % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(
            format="yuv420p"
        )
        frame.pts = i
        packets += [bytes(p) for p in enc.encode(frame)]
    packets += [bytes(p) for p in enc.encode(None)]
    return tuple(packets)


def _times(hz: float) -> list[int]:
    return [T0 + round(i * 1e9 / hz) for i in range(int(DURATION_S * hz) + 1)]


def _sides(embodiment: str) -> tuple[str, ...]:
    return spec.arm_sides(embodiment)


def _metadata(embodiment: str, declared: str) -> dict[str, str]:
    info = {
        "lab": "aria",
        "site": "rl2",
        "operator-id": "ab" * 32,
        "task": "fold_clothes",
        "instruction": "fold the shirt",
        "embodiment": declared,
        "rig-name": "test_rig",
        "device-serial": "SN0001",
        "scene": "kitchen_A",
        "objects": "shirt,basket",
        "start-time-unix": str(T0 // 10**6),
        "end-time-unix": str((T0 + DURATION_S * 10**9) // 10**6),
        "vendor-recording-id": "acme-rec-0001",
        "vendor": "acme",
        "spec-version": spec.SPEC_VERSION,
        "license": "CC-BY-4.0",
    }
    if spec.profile_for(embodiment) is spec.HUMAN:
        info["camera-timestamp-ref"] = "mid"
        info["hand-joints"] = ",".join(spec.MANO_JOINTS)
        info["shutter"] = "rolling:16ms"
    return info


def _tf(links: list[tuple[str, str]], bad_rotation: bool) -> FrameTransforms:
    tf = FrameTransforms()
    for i, (parent, child) in enumerate(links):
        x = tf.transforms.add(parent_frame_id=parent, child_frame_id=child)
        x.translation.z = 0.1
        x.rotation.w = 2.0 if bad_rotation and i == 0 else 1.0
    return tf


def _rigid(z: float) -> list[float]:
    T = np.eye(4)
    T[:3, 3] = (0.3, 0.0, z)
    return T.ravel().tolist()


def write_mcap_episode(
    root: Path,
    embodiment: str = "human_bimanual",
    episode_id: str = "0",
    *,
    # 1. everything there
    drop_topic: str | None = None,
    drop_metadata_key: str | None = None,
    declared_embodiment: str | None = None,
    bad_site_slug: bool = False,
    metadata_override: dict[str, str] | None = None,
    unknown_hand_joint: bool = False,
    empty_arm_pose: bool = False,
    # 2. video
    video_format: str = "h264",
    skip_first_keyframe: bool = False,
    calibration_width: int = WIDTH,
    video_fps: int = VIDEO_FPS,
    # 3. clocks
    backwards_head_pose: bool = False,
    hand_gap_s: float = 0.0,
    device_clock_imu_info: bool = False,
    # 4. calibration
    unlink_main_camera: bool = False,
    bad_tf_rotation: bool = False,
    bad_upright_rotation: bool = False,
    # 5. imu
    imu_hz: int = IMU_HZ,
    gravity_removed: bool = False,
    # 6. poses
    nan_head_pose: bool = False,
    zeroed_left_hand: bool = False,
    gripper_out_of_range: bool = False,
    bad_arm_pose: bool = False,
    bad_hand_confidence: bool = False,
    hands_wrist_only: bool = False,
    bad_wrist_pose: bool = False,
    nan_eye_gaze: bool = False,
    joint_names_mismatch: bool = False,
    multi_frame_video: bool = False,
    # 7. tasks & labels
    labels_in: str | None = "annotation",
    task_past_end: bool = False,
    label_outside_tasks: bool = False,
    stray_annotation_topic: bool = False,
    label_crosses_tasks: bool = False,
    labels_without_tasks: bool = False,
    label_past_video: bool = False,
    label_at_last_frame: bool = False,
    annotation_metadata: bool = False,
    calibration_twice: bool = False,
    principal_x: float = WIDTH / 2,
    repeat_frame_time: bool = False,
    tamper: Callable | None = None,
    distortion_model: str = "",
    distortion_coeffs: tuple[float, ...] = (),
    bad_annotation_type: bool = False,
) -> Path:
    """Write `root/episode_<episode_id>/episode.mcap` and return the episode dir.

    With no switches set, the episode passes the validator. Each switch breaks one
    thing:

      drop_topic            leave this topic out (e.g. "/imu-info", "/main-camera-info")
      drop_metadata_key     leave this episode-info key out
      declared_embodiment   write this as the "embodiment" key, while the topics
                            follow `embodiment`
      bad_site_slug         "site" is "RL2 Lab" (not a lowercase slug)
      metadata_override     episode-info values to set as given
      unknown_hand_joint    "hand-joints" names "thumb_base", not in the vocabulary
      empty_arm_pose        robot: one arm's state `pose` left empty (as in ABC's
                            release)
      video_format          CompressedVideo.format on every camera
      skip_first_keyframe   cameras start on a P-frame
      calibration_width     width in /main-camera-info (frames stay 64 wide)
      video_fps             frame rate of every camera
      backwards_head_pose   two /head-pose messages swap timestamps
      hand_gap_s            hole of this length in /left-hand-keypoints
      device_clock_imu_info /imu-info stamped with device time (1 s), not Unix time
      unlink_main_camera    /tf_static has no transform for the main camera
      bad_tf_rotation       first /tf_static transform has a non-unit quaternion
      bad_upright_rotation  every camera's -info-extra says to turn it 45 degrees
      imu_hz                /imu sample rate
      gravity_removed       /imu reads ~0 m/s^2 at rest
      nan_head_pose         one /head-pose position is NaN
      zeroed_left_hand      /left-hand-keypoints all zero with valid=true
      gripper_out_of_range  robot: one /<side>-ee-state value is 1.5
      bad_arm_pose          robot: one /<side>-arm-state pose has a scaled rotation
      bad_hand_confidence   one /left-hand-keypoints confidence is 1.5
      hands_wrist_only      valid: hands carry only wrist_pose, no keypoints
      bad_wrist_pose        one /left-hand-keypoints wrist_pose has a non-unit rotation
      joint_names_mismatch  robot: one arm's state lists 5 joint names for 6 joints
      multi_frame_video     /main-camera (or /top-camera) packs 2 frames per message
      nan_eye_gaze          one valid /eye-gaze message has a NaN yaw
      labels_in             where the label topics go: "annotation" (annotation.mcap),
                            "episode" (inline, as ABC does) or None (left out)
      task_past_end         the second /task ends 3 s after the recorded data
      label_outside_tasks   only the first /task is written, so the second label is
                            in no task
      stray_annotation_topic annotation.mcap also carries /instruction
      label_crosses_tasks   the first label runs on into the second task
      labels_without_tasks  valid: labels but no /task (the episode info is the task)
      label_past_video      the point label sits after the last video frame
      label_at_last_frame   valid: the point label sits on the last video frame
      annotation_metadata   annotation.mcap carries its own session-metadata record
      calibration_twice     every camera's -info is published twice
      principal_x           cx in every camera's K (default: the image centre)
      repeat_frame_time     frame 21 of every camera repeats frame 20's timestamp
      tamper                tamper(topic, msg, t) -> None | (msg, t), applied to every
                            message before it is written (both files): any breakage
      distortion_model, distortion_coeffs  every camera's -info lens model and D
      bad_annotation_type   the labels are typed "summary" (only "language" is known)
    """
    human = spec.profile_for(embodiment) is spec.HUMAN
    sides = _sides(embodiment)
    out = Path(root) / f"episode_{episode_id}"
    out.mkdir(parents=True, exist_ok=True)

    info = _metadata(embodiment, declared_embodiment or embodiment)
    if drop_metadata_key:
        info.pop(drop_metadata_key)
    if bad_site_slug:
        info["site"] = "RL2 Lab"
    info.update(metadata_override or {})
    if unknown_hand_joint:
        info["hand-joints"] = info["hand-joints"].replace("thumb_cmc", "thumb_base")

    def keep(topic: str) -> bool:
        return topic != drop_topic

    n_frames = DURATION_S * video_fps
    packets = _h264_packets(n_frames)
    if skip_first_keyframe:
        packets = packets[1:]
    video_times = [
        T0 + round((i + skip_first_keyframe) * 1e9 / video_fps)
        for i in range(len(packets))
    ]
    if repeat_frame_time:
        video_times[21] = video_times[20]

    if human:
        cameras = {"/main-camera": "main-camera"}
        links = [("head", "main-camera"), ("head", "imu"), ("head", "et")]
        if unlink_main_camera:
            links.remove(("head", "main-camera"))
    else:
        cameras = {"/top-camera": "top-camera"}
        links = [("world", "top-camera")]
        for s in sides:
            cameras[f"/{s}-wrist-camera"] = f"{s}-wrist-camera"
            links += [("world", f"{s}_arm_base"), (f"{s}_ee", f"{s}-wrist-camera")]

    with open(out / spec.EPISODE_FILE, "wb") as f, Writer(f) as w:

        def write(topic: str, msg, t: int) -> None:
            if tamper and (changed := tamper(topic, msg, t)):
                msg, t = changed
            if keep(topic):
                w.write_message(topic, msg, log_time=t, publish_time=t)

        # mcap_protobuf's Writer has no metadata call; its underlying mcap writer does.
        w._writer.add_metadata(spec.METADATA_RECORD, info)

        for topic, frame_id in cameras.items():
            width = calibration_width if topic == "/main-camera" else WIDTH
            write(
                topic + spec.CALIBRATION_SUFFIX,
                CameraCalibration(
                    frame_id=frame_id,
                    width=width,
                    height=HEIGHT,
                    K=[50, 0, principal_x, 0, 50, HEIGHT / 2, 0, 0, 1],
                    distortion_model=distortion_model,
                    D=list(distortion_coeffs),
                ),
                T0,
            )
            if calibration_twice:
                write(
                    topic + spec.CALIBRATION_SUFFIX,
                    CameraCalibration(frame_id=frame_id, width=width, height=HEIGHT),
                    T0 + 1,
                )
            write(
                topic + spec.CAMERA_EXTRA_SUFFIX,
                schemas.CameraInfoExtra(
                    frame_id=frame_id,
                    valid_radius=0.0,
                    upright_rotation_cw_deg=45 if bad_upright_rotation else 0,
                ),
                T0,
            )
            pairs = list(zip(packets, video_times))
            if multi_frame_video and topic in ("/main-camera", "/top-camera"):
                pairs = [
                    (b"".join(d for d, _ in pairs[i : i + 2]), pairs[i][1])
                    for i in range(0, len(pairs), 2)
                ]
            for data, t in pairs:
                write(
                    topic,
                    CompressedVideo(frame_id=frame_id, data=data, format=video_format),
                    t,
                )

        write("/tf_static", _tf(links, bad_tf_rotation), T0)
        write("/instruction", schemas.Instruction(data=info["instruction"]), T0)

        if human:
            _write_human(
                write,
                sides,
                backwards_head_pose=backwards_head_pose,
                hand_gap_s=hand_gap_s,
                device_clock_imu_info=device_clock_imu_info,
                imu_hz=imu_hz,
                gravity_removed=gravity_removed,
                nan_head_pose=nan_head_pose,
                zeroed_left_hand=zeroed_left_hand,
                bad_hand_confidence=bad_hand_confidence,
                hands_wrist_only=hands_wrist_only,
                bad_wrist_pose=bad_wrist_pose,
                nan_eye_gaze=nan_eye_gaze,
            )
        else:
            _write_robot(
                write,
                sides,
                empty_arm_pose=empty_arm_pose,
                gripper_out_of_range=gripper_out_of_range,
                bad_arm_pose=bad_arm_pose,
                joint_names_mismatch=joint_names_mismatch,
            )
        if labels_in == "episode":
            _write_labels(
                write,
                task_past_end,
                label_outside_tasks,
                label_crosses_tasks,
                labels_without_tasks,
                label_past_video,
                label_at_last_frame,
                bad_annotation_type,
            )

    if labels_in == "annotation":
        with open(out / spec.ANNOTATION_FILE, "wb") as f, Writer(f) as w:

            def write_label(topic: str, msg, t: int) -> None:
                if tamper and (changed := tamper(topic, msg, t)):
                    msg, t = changed
                w.write_message(topic, msg, log_time=t, publish_time=t)

            _write_labels(
                write_label,
                task_past_end,
                label_outside_tasks,
                label_crosses_tasks,
                labels_without_tasks,
                label_past_video,
                label_at_last_frame,
                bad_annotation_type,
            )
            if annotation_metadata:
                w._writer.add_metadata(spec.METADATA_RECORD, {"lab": "other"})
            if stray_annotation_topic:
                write_label("/instruction", schemas.Instruction(data="fold"), T0)
    return out


def _write_labels(
    write,
    task_past_end: bool,
    label_outside_tasks: bool,
    label_crosses_tasks: bool,
    labels_without_tasks: bool,
    label_past_video: bool,
    label_at_last_frame: bool,
    bad_annotation_type: bool,
) -> None:
    """Two back-to-back tasks, one ranged label inside the first and a point label
    inside the second."""
    half = DURATION_S * 10**9 // 2
    end = T0 + 2 * half + (3 * 10**9 if task_past_end else 0)
    tasks = [
        (T0, T0 + half, "fold_clothes", ["shirt"]),
        (T0 + half, end, "pack_clothes", ["shirt", "basket"]),
    ]
    for start, stop, name, objects in tasks[
        : 0 if labels_without_tasks else 1 if label_outside_tasks else 2
    ]:
        write(
            "/task",
            schemas.Task(
                name=name,
                description=name.replace("_", " "),
                objects=objects,
                end_time_unix_ns=stop,
            ),
            start,
        )
    kind = "summary" if bad_annotation_type else "language"
    first_end = T0 + half + half // 2 if label_crosses_tasks else T0 + half // 2
    write(
        "/subtask-annotation",
        schemas.Annotation(
            data="pick up the shirt",
            end_time_unix_ns=first_end,
            annotation_type=kind,
            annotation_vendor="acme",
        ),
        T0 + half // 5,
    )
    write(
        "/subtask-annotation",
        schemas.Annotation(data="drop it in the basket", annotation_type=kind),
        T0 + 2 * half - 10**6
        if label_past_video
        else T0 + round((DURATION_S * VIDEO_FPS - 1) * 1e9 / VIDEO_FPS)
        if label_at_last_frame
        else T0 + half + half // 3,
    )


def _write_human(
    write,
    sides,
    *,
    backwards_head_pose,
    hand_gap_s,
    device_clock_imu_info,
    imu_hz,
    gravity_removed,
    nan_head_pose,
    zeroed_left_hand,
    bad_hand_confidence,
    hands_wrist_only,
    bad_wrist_pose,
    nan_eye_gaze,
) -> None:
    write(
        "/imu-info",
        schemas.ImuInfo(
            frame_id="imu",
            accel_noise_density=1e-3,
            accel_random_walk=1e-4,
            gyro_noise_density=1e-4,
            gyro_random_walk=1e-5,
        ),
        10**9 if device_clock_imu_info else T0,
    )
    for i, t in enumerate(_times(imu_hz)):
        m = schemas.Imu(frame_id="imu")
        m.linear_acceleration.z = 0.02 if gravity_removed else spec.GRAVITY
        m.angular_velocity.x = 0.01 * np.sin(i / 20)
        write("/imu", m, t)

    pose_times = _times(POSE_HZ)
    if backwards_head_pose:
        pose_times[10], pose_times[11] = pose_times[11], pose_times[10]
    for i, t in enumerate(pose_times):
        p = PoseInFrame(frame_id=spec.WORLD_FRAME)
        p.pose.position.z = 1.5
        p.pose.orientation.w = 1.0
        if nan_head_pose and i == 5:
            p.pose.position.x = float("nan")
        write("/head-pose", p, t)

    for i, t in enumerate(_times(POSE_HZ)):
        g = schemas.EyeGaze(frame_id="et", yaw=0.1, pitch=-0.3, depth=0.8, valid=True)
        if nan_eye_gaze and i == 5:
            g.yaw = float("nan")
        write("/eye-gaze", g, t)

    ok = schemas.TrackingQuality.OK
    hand = np.linspace(0.1, 0.5, 63).tolist()
    for t in _times(POSE_HZ):
        write("/head-pose-quality", schemas.TrackingQuality(state=ok), t)
        for s in sides:
            in_gap = s == "left" and T0 + 0.5e9 < t < T0 + (0.5 + hand_gap_s) * 1e9
            if in_gap:
                continue
            zero = zeroed_left_hand and s == "left"
            k = schemas.HandKeypoints(
                frame_id=spec.WORLD_FRAME,
                positions=[] if hands_wrist_only else [0.0] * 63 if zero else hand,
                confidence=1.5
                if bad_hand_confidence and s == "left" and t == T0
                else 0.9,
                joint_confidence=[] if hands_wrist_only else [0.9] * 21,
                valid=True,
            )
            k.wrist_pose.position.z = 1.2
            k.wrist_pose.orientation.w = (
                2.0 if bad_wrist_pose and s == "left" and t == T0 else 1.0
            )
            write(f"/{s}-hand-keypoints", k, t)


def _write_robot(
    write,
    sides,
    *,
    empty_arm_pose,
    gripper_out_of_range,
    bad_arm_pose,
    joint_names_mismatch,
):
    joints = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]
    names = [f"joint{j}" for j in range(1, 7)]
    for i, t in enumerate(_times(POSE_HZ)):
        for s in sides:
            pose = _rigid(0.2 + 0.001 * i)
            if bad_arm_pose and s == sides[-1] and i == 5:
                pose[0] = 2.0  # R[0, 0]: no longer a rotation
            last = s == sides[-1]
            write(
                f"/{s}-arm-state",
                schemas.ArmState(
                    frame_id=f"{s}_arm_base",
                    joint_names=names[:5] if joint_names_mismatch and last else names,
                    position=joints,
                    velocity=[0.0] * 6,
                    torque=[0.5] * 6,
                    ee_pose=[] if empty_arm_pose and last else pose,
                ),
                t,
            )
            write(
                f"/{s}-arm-action",
                schemas.ArmAction(
                    frame_id=f"{s}_arm_base",
                    joint_names=names,
                    position=joints,
                    ee_pose=_rigid(0.2 + 0.001 * (i + 1)),
                ),
                t,
            )
            g = 1.5 if gripper_out_of_range and last and i == 5 else 0.4
            write(f"/{s}-ee-state", schemas.GripperState(position=g), t)
            write(f"/{s}-ee-action", schemas.GripperState(position=0.4), t)
