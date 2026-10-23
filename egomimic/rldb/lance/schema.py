"""Arrow schemas for the Lance episode tables (see docs/episode_representation_implementation.md)."""

import uuid

import pyarrow as pa

RECORDING_NAMESPACE = uuid.UUID("6f1d4c3e-2b1a-5e7f-9c8d-0a1b2c3d4e5f")


def recording_id(source: str, vendor_recording_id: str) -> str:
    """`source` is a fixed per-converter namespace ("aria", "eva"), unrelated to the `lab` column."""
    return str(uuid.uuid5(RECORDING_NAMESPACE, f"{source}/{vendor_recording_id}"))


# One steps table per (lab, embodiment, end_effector); recordings rows name theirs by these columns.
STEPS_KEY = ("lab", "embodiment", "end_effector")


def steps_path(lab: str, embodiment: str, end_effector: str) -> str:
    """Steps table of one (lab, embodiment, end_effector), relative to the Lance root."""
    return f"steps/{lab}/{embodiment}/{end_effector}.lance"


def steps_paths(recordings) -> list[str]:
    """Sorted steps tables referenced by a recordings pyarrow Table or DataFrame."""
    cols = [recordings[c] for c in STEPS_KEY]
    cols = [c.to_pylist() if hasattr(c, "to_pylist") else list(c) for c in cols]
    return sorted({steps_path(*k) for k in zip(*cols)})


def only_steps_path(recordings) -> str:
    """The one steps table `recordings` references, for tools that read a single table."""
    paths = steps_paths(recordings)
    if len(paths) != 1:
        raise ValueError(f"expected one steps table, recordings reference {paths}")
    return paths[0]


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
        # The collecting lab, as in SQL and zarr (e.g. "rl2").
        ("lab", pa.string()),
        ("embodiment", pa.string()),
        # An END_EFFECTOR value.
        ("end_effector", pa.string()),
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

ANNOTATION = pa.struct(
    [
        ("t_start_ns", pa.int64()),
        ("t_end_ns", pa.int64()),
        ("text", pa.string()),
        ("annotator", pa.string()),
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
        ("annotations", pa.list_(ANNOTATION)),
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
