"""Synthetic Lance steps rows for unit tests."""

import json

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egomimic.rldb.lance.schema import (
    ALIASES,
    ANNOTATIONS,
    ARM,
    EVA_STEPS,
    HUMAN_STEPS,
    RECORDINGS,
    TASKS,
    annotation_id,
    recording_id,
    steps_path,
    task_instance_id,
)

PERIOD_NS = 33_333_333
EVA_DELIVERY_FRAMES = 6


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


def eva_steps_table(rid, n, offset, arms=("left", "right"), valid=None):
    """Robot rows: obs_ee_pose x = frame + offset (+0.5 left, +0.25 right), cmd x one
    frame ahead; an arm not in `arms` is zeros and flagged False."""
    t = np.arange(n, dtype=np.float32)

    def arm(side, s):
        on = float(side in arms)
        pose = lambda x: vec(  # noqa: E731
            on * np.c_[x, np.zeros((n, 3)), np.ones(n), np.zeros((n, 2))], 7
        )
        return pa.StructArray.from_arrays(
            [
                pose(t + offset + s),
                vec(np.zeros((n, 6)), 6),
                pa.array(np.full(n, 0.5 * on, np.float32)),
                pose(t + offset + s + 1),
                vec(np.zeros((n, 6)), 6),
                pa.array((on * t / n).astype(np.float32)),
            ],
            fields=list(ARM),
        )

    flags = {f"{a}_valid": np.full(n, a in arms) for a in ("left", "right")}
    flags.update(valid or {})
    return pa.table(
        {
            "recording_id": pa.array([rid] * n),
            "timestamp_ns": pa.array(np.arange(n, dtype=np.int64) * PERIOD_NS),
            "left": arm("left", 0.5),
            "right": arm("right", 0.25),
            **{k: pa.array(v) for k, v in flags.items()},
        },
        schema=EVA_STEPS,
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
    source="aria",
):
    """Steps, recordings, tasks, aliases and annotations tables for two recordings of
    lab rl2 from `source` under ``tmp_path``; returns their sorted ``(recording_id,
    episode_hash, num_frames)``. ``annotations(recs)``, ``tasks(recs)`` and
    ``aliases(recs)`` give those rows (by default one task and one alias per whole
    recording); ``valid(rid, n)`` overrides validity columns of a recording's steps.
    For "eva" (grippers) the first recording is bimanual and the second left-arm only,
    each in its own steps table; aria ones (hands) are bimanual."""
    recs = [
        (recording_id(source, "1"), "2025-01-01-00-00-00-000000", 12),
        (recording_id(source, "2"), "2025-01-01-00-00-01-000000", 8),
    ]
    embodiment = {recs[0][0]: "human_bimanual", recs[1][0]: "human_bimanual"}
    end_effector = "hand"
    if source == "eva":
        end_effector = "gripper"
        embodiment = {recs[0][0]: "eva_bimanual", recs[1][0]: "eva_left_arm"}
    recs.sort()

    def rows(i, rid, n):
        if source != "eva":
            return steps_table(rid, n, 100 * i, valid(rid, n))
        arms = ("left", "right") if embodiment[rid] == "eva_bimanual" else ("left",)
        return eva_steps_table(rid, n, 100 * i, arms, valid(rid, n))

    for emb in sorted(set(embodiment.values())):
        steps = pa.concat_tables(
            [
                rows(i, rid, n)
                for i, (rid, _, n) in enumerate(recs)
                if embodiment[rid] == emb
            ]
        )
        lance.write_dataset(steps, tmp_path / steps_path("rl2", emb, end_effector))
    k = [200.0, 0, 240, 0, 200.0, 180, 0, 0, 1]
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                dict(
                    recording_id=rid,
                    vendor_recording_id=h,
                    lab="rl2",
                    embodiment=embodiment[rid],
                    end_effector=end_effector,
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


def eva_delivery(out, vid, frames_in_video=EVA_DELIVERY_FRAMES, offset=0):
    """An eva_to_lance-shaped delivery directory for a bimanual demo of EVA_DELIVERY_FRAMES frames."""
    d = out / vid
    d.mkdir(parents=True)
    rid = recording_id("eva", vid)
    pq.write_table(
        eva_steps_table(rid, EVA_DELIVERY_FRAMES, offset), d / "steps.parquet"
    )
    from fixtures.videos import write_av1

    write_av1(d / "front_1.mp4", frames_in_video)
    meta = {
        "vendor_recording_id": vid,
        "recording_id": rid,
        "episode_hash": f"clip-{vid}",
        "embodiment": "eva_bimanual",
        "site": "rl2",
        "operator": "op",
        "scene": "1",
        "task_name": "cup_on_saucer",
        "task_description": "cup on saucer",
        "objects": ["cup"],
        "fps": 30.0,
        "num_frames": EVA_DELIVERY_FRAMES,
        "videos": ["front_1"],
        "t_start_ns": 0,
        "t_end_ns": EVA_DELIVERY_FRAMES * PERIOD_NS,
        "intrinsics": {"front_1": [1.0] * 9},
        "extrinsics": {},
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return d
