"""Catalog rows from episode metadata (egomimic/rldb/lance/catalog.py)."""

import datetime

from egomimic.rldb.lance.catalog import episode_task, objects_list, rows_from_metadata
from egomimic.rldb.lance.schema import task_instance_id
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash

META = {
    "task_name": "fold_clothes",
    "task_description": "fold the shirt",
    "operator": "op_7",
    "lab": "rl2",
    "scene": "kitchen_A",
    "objects": "shirt, basket",
    "embodiment": "human_bimanual",
    "rig_name": "test_rig",
    "license": "CC-BY-4.0",
    "vendor_recording_id": "acme-rec-0",
    "start_time_unix": "1760000000000",
}


def rows(meta=META, tasks=None):
    t0, t1 = 1_760_000_000_000_000_000, 1_760_000_002_000_000_000
    return rows_from_metadata(
        meta,
        recording_id="rid",
        end_effector="hand",
        fps=30.0,
        num_frames=60,
        t_start_ns=t0,
        t_end_ns=t1,
        intrinsics={"front_1": [1.0]},
        extrinsics={},
        tasks=tasks if tasks is not None else [episode_task(meta, t0, t1)],
    )


def test_objects_list_takes_text_lists_and_waivers():
    assert objects_list("shirt, basket,") == ["shirt", "basket"]
    assert objects_list(["a", "b"]) == ["a", "b"]
    assert objects_list(None) == []


def test_one_episode_task_over_the_whole_recording():
    rec, (task,), alias = rows()
    assert (rec["lab"], rec["operator"], rec["rig_name"]) == ("rl2", "op_7", "test_rig")
    assert rec["captured_at"] == datetime.datetime(
        2025, 10, 9, 8, 53, 20, tzinfo=datetime.timezone.utc
    )
    assert rec["intrinsics"] == [("front_1", [1.0])]
    assert task["task_instance_id"] == task_instance_id(
        "rid", "fold_clothes", 1_760_000_000_000_000_000
    )
    assert task["objects"] == ["shirt", "basket"]
    assert alias["episode_hash"] == timestamp_ms_to_episode_hash(1760000000000)
    assert (alias["t_start_ns"], alias["t_end_ns"]) == (
        task["t_start_ns"],
        task["t_end_ns"],
    )


def test_waived_scene_and_objects_stay_empty():
    meta = {k: v for k, v in META.items() if k not in ("scene", "objects")}
    rec, (task,), _ = rows(meta)
    assert rec["scene"] is None and task["objects"] == []


def test_segment_tasks_are_kept_as_given():
    segs = [(0, 5, "fold", None, []), (5, 9, "pack", "pack it", ["box"])]
    _, tasks, _ = rows(tasks=segs)
    assert [(t["task_name"], t["t_start_ns"]) for t in tasks] == [
        ("fold", 0),
        ("pack", 5),
    ]
