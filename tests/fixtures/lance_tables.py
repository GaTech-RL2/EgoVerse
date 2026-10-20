"""Synthetic Lance steps rows for unit tests."""

import lance
import numpy as np
import pyarrow as pa

from egomimic.rldb.lance.schema import (
    ALIASES,
    ANNOTATIONS,
    HUMAN_STEPS,
    RECORDINGS,
    TASKS,
    annotation_id,
    recording_id,
    task_instance_id,
)

PERIOD_NS = 33_333_333


def annotation(rid, f0, f1, text, vendor="scale", quality=None):
    """An ANNOTATIONS row over frames [f0, f1) of a `steps_table` recording."""
    t0, t1 = f0 * PERIOD_NS, f1 * PERIOD_NS
    return dict(
        annotation_id=annotation_id(rid, vendor, "language", t0, t1, text),
        recording_id=rid,
        t_start_ns=t0,
        t_end_ns=t1,
        annotation_type="language",
        text=text,
        annotation_vendor=vendor,
        quality_score=quality,
        graded_by=None,
        rubric_version=None,
    )


def vec(x, n):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(x, np.float32).reshape(-1)), n
    )


def steps_table(rid, n, offset, valid=None):
    t = np.arange(n, dtype=np.float32)
    hand = lambda s: pa.StructArray.from_arrays(  # noqa: E731
        [
            vec(np.c_[t + offset + s, np.zeros((n, 6))], 7),
            vec(np.c_[t + offset + s, np.zeros((n, 6))], 7),
            vec(np.zeros((n, 63)), 63),
            vec(np.zeros((n, 63)), 63),
            pa.array(np.ones(n, np.float32)),
        ],
        fields=list(HUMAN_STEPS.field("left").type),
    )
    left_valid = np.ones(n, bool)
    left_valid[4] = False
    flags = {
        "left_valid": left_valid,
        "right_valid": np.ones(n, bool),
        "head_pose_valid": np.ones(n, bool),
        **(valid or {}),
    }
    return pa.table(
        {
            "recording_id": pa.array([rid] * n),
            "timestamp_ns": pa.array(np.arange(n, dtype=np.int64) * PERIOD_NS),
            "head_pose": vec(np.c_[t + offset, np.zeros((n, 6))], 7),
            "left": hand(0.5),
            "right": hand(0.25),
            **{k: pa.array(v) for k, v in flags.items()},
            "eye_gaze": vec(np.zeros((n, 3)), 3),
        },
        schema=HUMAN_STEPS,
    )


def task(rid, f0, f1, name="fold_clothes", description="folding clothes"):
    """A TASKS row over frames [f0, f1) of a `steps_table` recording."""
    t0 = f0 * PERIOD_NS
    return dict(
        task_instance_id=task_instance_id(rid, name, t0),
        recording_id=rid,
        t_start_ns=t0,
        t_end_ns=f1 * PERIOD_NS,
        task_name=name,
        task_description=description,
        objects=[],
    )


def alias(rid, f0, f1, episode_hash):
    return dict(
        episode_hash=episode_hash,
        recording_id=rid,
        t_start_ns=f0 * PERIOD_NS,
        t_end_ns=f1 * PERIOD_NS,
    )


def write_catalog(
    tmp_path,
    annotations=lambda recs: [],
    tasks=lambda recs: [task(rid, 0, n) for rid, _, n in recs],
    aliases=lambda recs: [alias(rid, 0, n, h) for rid, h, n in recs],
    valid=lambda rid, n: {},
):
    """Steps, recordings, tasks, aliases and annotations tables for two aria recordings
    under ``tmp_path``; returns their sorted ``(recording_id, episode_hash, num_frames)``.
    ``annotations(recs)``, ``tasks(recs)`` and ``aliases(recs)`` give those rows (by
    default one task and one alias per whole recording); ``valid(rid, n)`` overrides
    validity columns of a recording's steps."""
    recs = [
        (recording_id("aria", "1"), "2025-01-01-00-00-00-000000", 12),
        (recording_id("aria", "2"), "2025-01-01-00-00-01-000000", 8),
    ]
    recs.sort()
    steps = pa.concat_tables(
        [
            steps_table(rid, n, 100 * i, valid(rid, n))
            for i, (rid, _, n) in enumerate(recs)
        ]
    )
    lance.write_dataset(steps, tmp_path / "steps" / "aria.lance")
    k = [200.0, 0, 240, 0, 200.0, 180, 0, 0, 1]
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                dict(
                    recording_id=rid,
                    vendor_recording_id=h,
                    lab="aria",
                    site="rl2",
                    embodiment="human_bimanual",
                    rig_name="aria_gen1",
                    operator="op",
                    scene="1",
                    captured_at=None,
                    fps=30.0,
                    num_frames=n,
                    video_path=[("front_1", f"videos/{rid}/front_1.mp4")],
                    intrinsics=[("front_1", k)],
                    extrinsics=[],
                    license="x",
                )
                for rid, h, n in recs
            ],
            schema=RECORDINGS,
        ),
        tmp_path / "recordings.lance",
    )
    lance.write_dataset(
        pa.Table.from_pylist(tasks(recs), schema=TASKS), tmp_path / "tasks.lance"
    )
    lance.write_dataset(
        pa.Table.from_pylist(aliases(recs), schema=ALIASES),
        tmp_path / "aliases.lance",
    )
    lance.write_dataset(
        pa.Table.from_pylist(annotations(recs), schema=ANNOTATIONS),
        tmp_path / "annotations.lance",
    )
    return recs
