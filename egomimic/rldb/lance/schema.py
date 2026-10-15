"""Arrow schemas for the Lance episode tables (see docs/episode_representation_implementation.md)."""

import uuid

import pyarrow as pa

RECORDING_NAMESPACE = uuid.UUID("6f1d4c3e-2b1a-5e7f-9c8d-0a1b2c3d4e5f")


def recording_id(lab: str, vendor_recording_id: str) -> str:
    return str(uuid.uuid5(RECORDING_NAMESPACE, f"{lab}/{vendor_recording_id}"))


def annotation_id(
    recording_id: str,
    annotation_vendor: str,
    annotation_type: str,
    t_start_ns: int,
    t_end_ns: int,
    text: str,
) -> str:
    """Same content, same id: vendors can compute the id of a row they sent, to delete it."""
    key = "\0".join(
        map(
            str,
            (
                recording_id,
                annotation_vendor,
                annotation_type,
                t_start_ns,
                t_end_ns,
                text,
            ),
        )
    )
    return str(uuid.uuid5(RECORDING_NAMESPACE, f"annotation/{key}"))


def task_instance_id(recording_id: str, task_name: str | None, t_start_ns: int) -> str:
    """Stable across rebuilds and computable by vendors; old episode_hashes resolve through aliases."""
    return str(
        uuid.uuid5(
            RECORDING_NAMESPACE, f"task/{recording_id}/{task_name or ''}/{t_start_ns}"
        )
    )


def _vec(n: int) -> pa.DataType:
    return pa.list_(pa.float32(), n)


HAND = pa.struct(
    [
        ("wrist_pose", _vec(7)),
        ("ee_pose", _vec(7)),
        ("keypoints", _vec(63)),
        # Pilot extras: raw Aria landmarks (pre-MANO) and MPS confidence (-1 = untracked).
        ("aria_keypoints", _vec(63)),
        ("confidence", pa.float32()),
    ]
)

HUMAN_STEPS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("timestamp_ns", pa.int64()),
        ("head_pose", _vec(7)),
        ("left", HAND),
        ("right", HAND),
        ("left_valid", pa.bool_()),
        ("right_valid", pa.bool_()),
        ("head_pose_valid", pa.bool_()),
        ("eye_gaze", _vec(3)),
    ]
)

RECORDINGS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("vendor_recording_id", pa.string()),
        ("lab", pa.string()),
        # Pilot extra: SQL `lab` is the collecting site for Aria, not the vendor.
        ("site", pa.string()),
        ("embodiment", pa.string()),
        ("rig_name", pa.string()),
        ("operator", pa.string()),
        ("scene", pa.string()),
        ("captured_at", pa.timestamp("ms", tz="UTC")),
        ("fps", pa.float32()),
        ("num_frames", pa.int64()),
        ("video_path", pa.map_(pa.string(), pa.string())),
        ("intrinsics", pa.map_(pa.string(), pa.list_(pa.float32()))),
        ("extrinsics", pa.map_(pa.string(), pa.list_(pa.float32()))),
        ("license", pa.string()),
        # sha256 over the recording's steps rows and videos; cache keys use it instead of the table version.
        ("content_digest", pa.string()),
    ]
)

TASKS = pa.schema(
    [
        ("task_instance_id", pa.string()),
        ("recording_id", pa.string()),
        ("t_start_ns", pa.int64()),
        ("t_end_ns", pa.int64()),
        ("task_name", pa.string()),
        ("task_description", pa.string()),
        ("objects", pa.list_(pa.string())),
    ]
)

ALIASES = pa.schema(
    [
        ("episode_hash", pa.string()),
        ("recording_id", pa.string()),
        ("t_start_ns", pa.int64()),
        ("t_end_ns", pa.int64()),
    ]
)

ANNOTATION_TYPES = ("language",)

# Intervals are half-open [t_start_ns, t_end_ns) on the recording clock; a point event is one frame wide.
ANNOTATIONS = pa.schema(
    [
        ("annotation_id", pa.string()),
        ("recording_id", pa.string()),
        ("t_start_ns", pa.int64()),
        ("t_end_ns", pa.int64()),
        ("annotation_type", pa.string()),
        ("text", pa.string()),
        ("annotation_vendor", pa.string()),
        # Rubric items passed (0-5); null until graded.
        ("quality_score", pa.int8()),
        ("graded_by", pa.string()),
        ("rubric_version", pa.string()),
    ]
)
