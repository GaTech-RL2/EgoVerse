"""Raw-data MCAP spec 0.1 in machine-readable form: what `validate.py` checks against.

SPEC.md is the human-readable version of this file; change both together. Numbers the
plan states are used as-is; the ones it does not state are marked PROVISIONAL, to be tuned
on real deliveries.

An episode is one whole recording session, never a pre-cut clip: it becomes one row of
the Lance `recordings` table, and the tasks inside it become `tasks` rows.

    episode_<id>/
    ├── episode.mcap      # everything: video, sensors, poses, calibration, episode info
    └── annotation.mcap   # optional: /task spans and /subtask-annotation labels

Every message is stamped with absolute Unix time in nanoseconds (its MCAP log time), in
both files. Every camera is a topic ending in `-camera`; its calibration is on the sibling
`<topic>-info`. Each video message holds exactly one frame, stamped with its capture time.

The profile's `step_clock` camera defines the Lance steps rows: one row per decoded frame
of it, every other stream aligned to those times during conversion. Other cameras stay
separate videos and add no rows.

Who, what and where: the episode info carries the same required fields as a zarr
episode (task_name, operator, lab, scene, objects; CONTRIBUTING_DATA.md §6.3), named and
checked exactly as in egomimic.rldb.zarr.required_metadata. `lab` is who the data came
from (rl2, mecka, microagi). Hardware goes in rig_name / device_serial. Task text is
task_name, task_description and the annotations; there is no separate instruction.

IDs: Lance recording_id = recording_id(lab, vendor_recording_id) below, so
vendor_recording_id must stay the same across every re-export or correction of the
recording. A zarr episode trimmed from it names it in its vendor_recording_id attr. Task
and annotation IDs are computed downstream from their content.
"""

import re
import uuid
from dataclasses import dataclass

from egomimic.rldb.zarr.episode_rules import EMBODIMENTS, QUAT_UNIT_TOL, arm_sides
from egomimic.rldb.zarr.required_metadata import (
    REQUIRED_METADATA_FIELDS,
    WAIVABLE_FIELDS,
)

SPEC_VERSION = "0.1"
SUPPORTED_SPEC_VERSIONS = ("0.1",)

EPISODE_FILE = "episode.mcap"
# Its topics may instead be inline in episode.mcap, as ABC-130k does.
ANNOTATION_FILE = "annotation.mcap"
# ABC's delivery format names its episode-info record this; keys are snake_case.
METADATA_RECORD = "session-metadata"
MISSING_METADATA_KEY = "missing_metadata"

CAMERA_SUFFIX = "-camera"
CALIBRATION_SUFFIX = "-info"
# Optional egoverse.CameraInfoExtra (valid radius, upright rotation) next to the -info.
CAMERA_EXTRA_SUFFIX = "-info-extra"
UPRIGHT_ROTATIONS = (0, 90, 180, 270)
# CameraCalibration.distortion_model -> number of D coefficients. An empty model with no
# coefficients is a pinhole camera.
DISTORTION_MODELS = {
    "plumb_bob": 5,
    "rational_polynomial": 8,
    "kannala_brandt": 4,
    "fisheye624": 12,
}

# Hand joint vocabulary. Vendors list the joints their device tracks, in the order they
# write them, in the "hand_joints" key. Names that are MANO joints are what the MANO fit
# (egomimic/scripts/aria_process/aria_utils.py) matches; extras are kept but not fitted.
# Canonical MANO order, as Human.FINGER_EDGES uses it: a vendor sending this exactly
# needs no fit.
MANO_JOINTS = (
    "wrist",
    *(
        f"{finger}_{joint}"
        for finger, joints in (
            ("thumb", ("cmc", "mcp", "ip", "tip")),
            ("index", ("mcp", "pip", "dip", "tip")),
            ("middle", ("mcp", "pip", "dip", "tip")),
            ("ring", ("mcp", "pip", "dip", "tip")),
            ("pinky", ("mcp", "pip", "dip", "tip")),
        )
        for joint in joints
    ),
)
EXTRA_HAND_JOINTS = ("palm_center",)
HAND_JOINTS = MANO_JOINTS + EXTRA_HAND_JOINTS
# Shutter: global, or rolling with its readout time.
SHUTTER = re.compile(r"global|rolling:[0-9]+(\.[0-9]+)?ms")

# Copied from egomimic/rldb/lance/schema.py (PR #716) so the validator needs no Lance
# install; a unit test checks the two agree once that module is on main.
RECORDING_NAMESPACE = uuid.UUID("6f1d4c3e-2b1a-5e7f-9c8d-0a1b2c3d4e5f")


def recording_id(lab: str, vendor_recording_id: str) -> str:
    """The Lance recordings.recording_id this episode will get."""
    return str(uuid.uuid5(RECORDING_NAMESPACE, f"{lab}/{vendor_recording_id}"))


# Frames every vendor uses by these names in /tf_static and in poses.
WORLD_FRAME = "world"
HEAD_FRAME = "head"

# ---- Thresholds stated in the plan ----
MIN_VIDEO_FPS = 30.0
MIN_IMU_HZ = 200.0
MIN_HEAD_POSE_HZ = 30.0
GRAVITY = 9.81  # m/s^2: mean |accel| of a raw IMU at rest
VIDEO_CODECS = ("h264", "h265")
GRIPPER_RANGE = (0.0, 1.0)  # 0 = closed, 1 = open

# ---- PROVISIONAL thresholds (not in the plan) ----
# A measured rate may fall this fraction below its minimum (29.97 fps passes).
RATE_TOL = 0.05
GRAVITY_TOL = 0.5  # m/s^2 allowed around GRAVITY in the stillest window
STILL_WINDOW_S = 1.0  # window used to find the IMU at rest
MAX_GAP_S = 0.5  # largest allowed gap between consecutive messages of one stream
SPAN_TOL_S = 1.0  # how far a stream may start late / end early vs. the episode
METADATA_TIME_TOL_S = 1.0  # start/end_time_unix vs. the data's actual span
UNIT_TOL = QUAT_UNIT_TOL  # |quaternion| - 1, and R^T R - I for rotation matrices
# Frames fully decoded per camera; structural checks run on every message.
VIDEO_SAMPLE_FRAMES = 60
MIN_UNIX_NS = 1_420_070_400 * 10**9  # 2015-01-01: anything earlier is not Unix time


@dataclass(frozen=True)
class Topic:
    """One channel an episode may carry.

    `kind` picks the checks the validator runs on it. `schema` is the exact schema name
    required, or None for any. `fields` must be non-empty on every message. `side` marks a
    per-arm topic, dropped for single-arm embodiments.
    """

    name: str
    kind: str
    required: bool = True
    schema: str | None = None
    fields: tuple[str, ...] = ()
    min_hz: float | None = None
    side: str | None = None


@dataclass(frozen=True)
class MovingLink:
    """A parent -> child transform that changes over time, published on `topic`
    instead of /tf_static (e.g. the head moving in the world)."""

    parent: str
    child: str
    topic: str


@dataclass(frozen=True)
class MetaKey:
    """One episode-info key. MCAP metadata values are strings; `kind` is what the
    string must parse as: "str", "int", "enum" (one of `values`), "shutter"
    (SHUTTER), "hand_joints" (comma-separated HAND_JOINTS names, each once,
    including wrist), "required_metadata" (one of REQUIRED_METADATA_FIELDS,
    checked with the zarr rules; `objects` is comma-separated) or "waiver"
    (comma-separated REQUIRED_METADATA_FIELDS names the source does not have)."""

    name: str
    kind: str = "str"
    values: tuple[str, ...] = ()
    human_only: bool = False
    optional: bool = False


@dataclass(frozen=True)
class Profile:
    name: str
    step_clock: str  # camera whose frames define the Lance steps rows
    topics: tuple[Topic, ...]
    moving_links: tuple[MovingLink, ...]


_VIDEO = dict(
    kind="video",
    schema="foxglove.CompressedVideo",
    min_hz=MIN_VIDEO_FPS,
    fields=("data", "format", "frame_id"),
)
_TF = Topic(
    "/tf_static", "tf_static", schema="foxglove.FrameTransforms", fields=("transforms",)
)
# Optional in either profile. Without /task, the episode-info task_name / task_description / objects
# keys describe one task spanning the whole recording; with /task, the spans are
# authoritative and those keys only summarize the session (no task row is made from them).
_LABELS = (
    Topic("/task", "task", required=False, schema="egoverse.Task", fields=("name",)),
    Topic(
        "/subtask-annotation",
        "annotation",
        required=False,
        schema="egoverse.Annotation",
        fields=("data",),
    ),
)
# Must match egomimic.rldb.lance.schema.ANNOTATION_TYPES (PR #717); empty means "language".
ANNOTATION_TYPES = ("language",)


def _per_side(make) -> tuple[Topic, ...]:
    return tuple(make(side) for side in ("left", "right"))


HUMAN = Profile(
    name="human",
    step_clock="/main-camera",
    topics=(
        Topic("/main-camera", **_VIDEO),
        Topic("/slam-left-camera", **_VIDEO, required=False),
        Topic("/slam-right-camera", **_VIDEO, required=False),
        *_per_side(lambda s: Topic(f"/{s}-wrist-camera", **_VIDEO, required=False)),
        Topic(
            "/imu",
            "imu",
            schema="egoverse.Imu",
            min_hz=MIN_IMU_HZ,
            fields=("frame_id", "linear_acceleration", "angular_velocity"),
        ),
        Topic("/imu-info", "imu_info", schema="egoverse.ImuInfo", fields=("frame_id",)),
        Topic(
            "/head-pose",
            "pose",
            schema="foxglove.PoseInFrame",
            min_hz=MIN_HEAD_POSE_HZ,
            fields=("frame_id", "pose"),
        ),
        Topic(
            "/head-pose-quality",
            "tracking_quality",
            schema="egoverse.TrackingQuality",
            min_hz=MIN_HEAD_POSE_HZ,
        ),
        *_per_side(
            lambda s: Topic(
                f"/{s}-hand-keypoints",
                "hand_keypoints",
                schema="egoverse.HandKeypoints",
                fields=("frame_id",),
                side=s,
            )
        ),
        Topic(
            "/eye-gaze",
            "eye_gaze",
            required=False,
            schema="egoverse.EyeGaze",
            fields=("frame_id",),
        ),
        _TF,
        *_LABELS,
    ),
    moving_links=(MovingLink(WORLD_FRAME, HEAD_FRAME, "/head-pose"),),
)

ROBOT = Profile(
    name="robot",
    step_clock="/top-camera",
    topics=(
        Topic("/top-camera", **_VIDEO),
        *_per_side(lambda s: Topic(f"/{s}-wrist-camera", **_VIDEO, side=s)),
        *_per_side(
            lambda s: Topic(
                f"/{s}-arm-state",
                "arm",
                schema="egoverse.ArmState",
                side=s,
                fields=(
                    "frame_id",
                    "joint_names",
                    "position",
                    "velocity",
                    "torque",
                    "ee_pose",
                ),
            )
        ),
        *_per_side(
            lambda s: Topic(
                f"/{s}-arm-action",
                "arm",
                schema="egoverse.ArmAction",
                side=s,
                fields=("frame_id", "joint_names", "position", "ee_pose"),
            )
        ),
        *_per_side(
            lambda s: Topic(
                f"/{s}-ee-state", "gripper", schema="egoverse.GripperState", side=s
            )
        ),
        *_per_side(
            lambda s: Topic(
                f"/{s}-ee-action", "gripper", schema="egoverse.GripperState", side=s
            )
        ),
        _TF,
        *_LABELS,
    ),
    # The end effector moves relative to its arm base; the wrist camera hangs off it.
    moving_links=tuple(
        MovingLink(f"{s}_arm_base", f"{s}_ee", f"/{s}-arm-state")
        for s in ("left", "right")
    ),
)

METADATA_KEYS = (
    # The zarr required metadata (CONTRIBUTING_DATA.md §6.3), same names and rules.
    *(MetaKey(f, "required_metadata") for f in REQUIRED_METADATA_FIELDS),
    MetaKey("task_description", optional=True),  # full task text
    # Comma-separated required fields the source truly does not have (MissingMetadata).
    # Only if absolutely necessary; the episode then fails without --allow-missing.
    MetaKey(MISSING_METADATA_KEY, "waiver", optional=True),
    MetaKey("embodiment", "enum", EMBODIMENTS),
    MetaKey("rig_name"),
    MetaKey("device_serial"),
    MetaKey("start_time_unix", "int"),  # ms
    MetaKey("end_time_unix", "int"),  # ms
    # The source's own ID for this recording; stable across re-exports and corrections.
    MetaKey("vendor_recording_id"),
    MetaKey("spec_version", "enum", SUPPORTED_SPEC_VERSIONS),
    MetaKey("license"),  # e.g. "CC-BY-4.0"; the Lance recordings.license column
    # Where in the exposure frames are stamped.
    MetaKey("camera_timestamp_ref", "enum", ("start", "mid", "end"), human_only=True),
    MetaKey("shutter", "shutter", human_only=True),  # "global" or "rolling:16ms"
    # HandKeypoints.positions order, chosen by the vendor, e.g. ",".join(MANO_JOINTS)
    MetaKey("hand_joints", "hand_joints", human_only=True),
)
# Every converter must give these (task_name, operator, lab); only scene and objects
# may be listed in missing_metadata. The shared waiver_errors rejects the rest.
UNWAIVABLE_FIELDS = tuple(
    f for f in REQUIRED_METADATA_FIELDS if f not in WAIVABLE_FIELDS
)
# Topics published exactly once per episode.
ONCE_TOPICS = ("/tf_static", "/imu-info")
# Topic kinds whose messages form a time series; the rest are published once or rarely.
STREAM_KINDS = frozenset(
    {
        "video",
        "imu",
        "pose",
        "tracking_quality",
        "hand_keypoints",
        "arm",
        "gripper",
        "eye_gaze",
    }
)
# Topics allowed in annotation.mcap.
LABEL_TOPICS = frozenset({"/task", "/subtask-annotation"})


def profile_for(embodiment: str) -> Profile:
    return HUMAN if embodiment.startswith("human_") else ROBOT


def topics_for(embodiment: str) -> tuple[Topic, ...]:
    """The profile's topics, minus per-side topics of an arm the embodiment lacks."""
    sides = arm_sides(embodiment)
    return tuple(
        t for t in profile_for(embodiment).topics if t.side is None or t.side in sides
    )
