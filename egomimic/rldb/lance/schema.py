"""Arrow schemas for the Lance episode tables."""

import uuid

import pyarrow as pa

RECORDING_NAMESPACE = uuid.UUID("6f1d4c3e-2b1a-5e7f-9c8d-0a1b2c3d4e5f")


def recording_id(source: str, vendor_recording_id: str) -> str:
    """UUIDv5 of `source/vendor_recording_id`. `source` names whoever assigns
    vendor_recording_id: the converter for in-house data ("aria", "eva"), the lab for
    MCAP deliveries (egomimic.rldb.mcap.spec.recording_id). A recording must always
    reach Lance through the same source, or it gets a second id."""
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

# Robot arm, robot base frame: poses xyz (m) + quaternion wxyz, joints in rad, gripper in [0, 1].
ARM = pa.struct(
    [
        ("obs_ee_pose", _vec(7)),
        ("obs_joints", _vec(6)),
        ("obs_gripper", pa.float32()),
        ("cmd_ee_pose", _vec(7)),
        ("cmd_joints", _vec(6)),
        ("cmd_gripper", pa.float32()),
    ]
)

# An arm the robot lacks is zeros with its flag False (Arrow's parquet reader rejects
# null fixed-size lists); an all-zero pose frame is forward-filled and flagged False.
EVA_STEPS = pa.schema(
    [
        ("recording_id", pa.string()),
        ("timestamp_ns", pa.int64()),
        ("left", ARM),
        ("right", ARM),
        ("left_valid", pa.bool_()),
        ("right_valid", pa.bool_()),
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

# One row per committer batch; `tag` names the version of every table the batch produced.
# `source` is "annotations" for an annotation commit.
CHANGES = pa.schema(
    [
        ("tag", pa.string()),
        # The converter that staged the batch (the tag prefix), e.g. "aria"; not a lab.
        ("source", pa.string()),
        # Steps tables the batch wrote, as steps_path gives them.
        ("steps_tables", pa.list_(pa.string())),
        ("committed_at", pa.timestamp("ms", tz="UTC")),
        ("operator", pa.string()),
        # A delivery batch's recordings appended and replaced; for source "annotations",
        # the recordings gaining and losing annotation rows.
        ("added", pa.list_(pa.string())),
        ("replaced", pa.list_(pa.string())),
    ]
)
