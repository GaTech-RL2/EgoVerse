import dataclasses
import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import PERIOD_NS, eva_steps_table, steps_table  # noqa: E402

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import (  # noqa: E402
    TASKS,
    recording_id,
    steps_path,
    task_instance_id,
)

STEPS = steps_path("rl2", "human_bimanual", "human_hand")

KEY_MAP = {
    "left.action_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.obs_ee_pose",
        "horizon": 3,
    },
    "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    "annotations": {"key_type": "annotation_keys", "zarr_key": "annotations"},
}


def _delivery(tmp, i, n, offset=None, video=b"mp4", **rec):
    rid = recording_id("aria", str(i))
    h = f"2025-01-01-00-00-{i:02d}-000000"
    src = tmp / "src" / f"{i}-{offset}"
    src.mkdir(parents=True, exist_ok=True)
    (src / "front_1.mp4").write_bytes(video)
    return commit.Delivery(
        recording=dict(
            recording_id=rid,
            vendor_recording_id=h,
            lab="rl2",
            embodiment="human_bimanual",
            end_effector="human_hand",
            operator="op",
            fps=30.0,
            num_frames=n,
            intrinsics=[],
            extrinsics=[],
        )
        | rec,
        task=dict(
            task_instance_id=f"t{i}",
            recording_id=rid,
            t_start_ns=0,
            t_end_ns=n * PERIOD_NS,
            task_name="fold",
            task_description=f"task {i}",
            objects=[],
        ),
        alias=dict(
            episode_hash=h, recording_id=rid, t_start_ns=0, t_end_ns=n * PERIOD_NS
        ),
        steps=steps_table(rid, n, 100 * i if offset is None else offset),
        videos={"front_1": src / "front_1.mp4"},
    )


def _resolve(root, **kw):
    return LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}, **kw).resolve()


def _head(leaf):
    return float(leaf[0]["obs_head_pose"][0])


def test_append_skips_committed(tmp_path):
    root = tmp_path / "lance"
    a, b = _delivery(tmp_path, 0, 10), _delivery(tmp_path, 1, 8)
    assert commit.append(root, [a]) == [a.recording_id]
    assert commit.append(root, [a, b]) == [b.recording_id]
    assert commit.append(root, [a, b]) == []
    assert lance.dataset(root / STEPS).count_rows() == 18
    assert len(lance.dataset(root / STEPS).get_fragments()) == 2


def test_steps_table_per_lab_embodiment_end_effector(tmp_path):
    root = tmp_path / "lance"
    a = _delivery(tmp_path, 0, 10)
    b = _delivery(tmp_path, 1, 8, lab="mecka")
    commit.append(root, [a, b])
    assert lance.dataset(root / STEPS).count_rows() == 10
    other = steps_path("mecka", "human_bimanual", "human_hand")
    assert lance.dataset(root / other).count_rows() == 8
    assert {str(Path(u).relative_to(root)) for u in commit.table_uris(root)} >= {
        STEPS,
        other,
    }
    leaves = _resolve(root)
    assert {leaf.steps_uri for leaf in leaves.values()} == {
        f"{root}/{STEPS}",
        f"{root}/{other}",
    }
    with pytest.raises(ValueError, match="changed"):
        commit.replace(root, _delivery(tmp_path, 1, 8, lab="rl2"))
    for bad in (
        {"end_effector": "claw"},
        {"end_effector": None},
        {"lab": None},
        {"operator": None},
    ):
        with pytest.raises(ValueError):
            commit.append(root, [_delivery(tmp_path, 2, 8, **bad)])
    no_task = _delivery(tmp_path, 2, 8)
    no_task.task["task_name"] = None
    with pytest.raises(ValueError, match="task_name"):
        commit.append(root, [no_task])
    assert commit.committed_ids(root) == {a.recording_id, b.recording_id}


def test_replace_is_one_commit_and_keeps_old_versions_readable(tmp_path):
    root = tmp_path / "lance"
    ds = [_delivery(tmp_path, i, 10) for i in range(3)]
    commit.append(root, ds)
    commit.tag(root, "t1")
    before = _resolve(root)
    steps_v = lance.dataset(root / STEPS).version

    new = _delivery(tmp_path, 0, 12, offset=1000, video=b"mp4-v2")
    commit.replace(root, new)
    assert lance.dataset(root / STEPS).version == steps_v + 1

    after = _resolve(root)
    h0 = new.alias["recording_id"]
    assert _head(after[h0]) == 1000 and len(after[h0]) == 12 - 3
    assert (
        after[h0].row_start == 20
    )  # moved to the end, contiguous and in timestamp order
    for h in before:
        if h != h0:
            assert _head(after[h]) == _head(before[h])
            assert after[h].cache_fingerprint() == before[h].cache_fingerprint()
    assert after[h0].cache_fingerprint() != before[h0].cache_fingerprint()

    old = _resolve(root, tag="t1")
    assert _head(old[h0]) == 0 and old[h0].steps_version == steps_v
    assert old[h0].content_digest == before[h0].content_digest


def test_edit_add_columns_tag_cleanup(tmp_path):
    root = tmp_path / "lance"
    ds = [_delivery(tmp_path, i, 10) for i in range(2)]
    commit.append(root, ds)
    commit.tag(root, "t1")
    commit.replace(root, _delivery(tmp_path, 1, 10, offset=500, video=b"mp4-v2"))

    commit.edit(
        root,
        "tasks.lance",
        pa.table({"task_instance_id": ["t0"], "task_description": ["fixed"]}),
        on="task_instance_id",
    )
    with pytest.raises(KeyError):
        commit.edit(
            root,
            "tasks.lance",
            pa.table({"task_instance_id": ["nope"], "task_description": ["x"]}),
            on="task_instance_id",
        )
    with pytest.raises(ValueError):
        commit.edit(
            root,
            STEPS,
            pa.table({"recording_id": ["x"]}),
            on="recording_id",
        )
    rid0 = ds[0].recording_id
    with pytest.raises(ValueError, match="changed"):
        commit.edit(
            root,
            "recordings.lance",
            pa.table({"recording_id": [rid0], "lab": ["mecka"]}),
            on="recording_id",
        )
    commit.edit(
        root,
        "recordings.lance",
        pa.table({"recording_id": [rid0], "lab": ["rl2"], "scene": ["kitchen"]}),
        on="recording_id",
    )
    leaves = _resolve(root)
    h0, h1 = (d.alias["recording_id"] for d in ds)
    assert leaves[h0].task_text == "fixed" and leaves[h1].task_text == "task 1"

    fp = {h: leaf.cache_fingerprint() for h, leaf in leaves.items()}
    commit.add_columns(root, STEPS, {"both_valid": "left_valid AND right_valid"})
    commit.index(root)
    commit.compact(root)
    steps = lance.dataset(root / STEPS)
    assert len(steps.get_fragments()) == 1
    assert np.asarray(steps.to_table(columns=["both_valid"]).column(0)).sum() == 18
    assert {h: leaf.cache_fingerprint() for h, leaf in _resolve(root).items()} == fp

    videos = sorted(p.relative_to(root) for p in (root / "videos").rglob("*.mp4"))
    assert len(videos) == 3
    commit.cleanup(root, older_than=datetime.timedelta(0))
    assert (
        sorted(p.relative_to(root) for p in (root / "videos").rglob("*.mp4")) == videos
    )  # t1 still needs the old one
    assert _head(_resolve(root, tag="t1")[h1]) == 100
    assert _head(_resolve(root)[h1]) == 500

    lance.dataset(root / "recordings.lance").tags.delete("t1")
    commit.cleanup(root, older_than=datetime.timedelta(0))
    assert len(list((root / "videos").rglob("*.mp4"))) == 2


def test_append_after_interrupted_append_does_not_duplicate_steps(
    tmp_path, monkeypatch
):
    root = tmp_path / "lance"
    commit.append(root, [_delivery(tmp_path, 0, 10)])
    a, b = _delivery(tmp_path, 1, 8), _delivery(tmp_path, 2, 9)
    real_append = commit._append

    def crash_on_recordings(uri, table):
        if str(uri).endswith("recordings.lance"):
            raise OSError("preempted")
        real_append(uri, table)

    monkeypatch.setattr(commit, "_append", crash_on_recordings)
    with pytest.raises(OSError):
        commit.append(root, [a, b])
    monkeypatch.setattr(commit, "_append", real_append)
    assert lance.dataset(root / STEPS).count_rows() == 27

    assert commit.append(root, [a, b, a]) == [a.recording_id, b.recording_id]
    assert lance.dataset(root / STEPS).count_rows() == 27
    assert lance.dataset(root / "aliases.lance").count_rows() == 3
    leaves = _resolve(root)
    assert sorted(len(leaf) for leaf in leaves.values()) == [8 - 3, 9 - 3, 10 - 3]


def test_task_instance_id_is_stable_and_keyed_on_name_and_start():
    rid = recording_id("aria", "1")
    a = task_instance_id(rid, "fold_clothes", 100)
    assert a == task_instance_id(rid, "fold_clothes", 100)
    assert a != task_instance_id(rid, "fold_clothes", 101)
    assert a != task_instance_id(rid, "iron_clothes", 100)
    assert a != task_instance_id(recording_id("aria", "2"), "fold_clothes", 100)


def test_release_tag_covers_annotations(tmp_path):
    root = tmp_path / "lance"
    commit.append(root, [_delivery(tmp_path, 0, 10)])
    tagged = commit.tag(root, "release-1")
    assert "annotations.lance" in tagged
    resolver = LanceEpisodeResolver(
        root, key_map=KEY_MAP, cameras={}, tag="release-1", annotations={}
    )
    (leaf,) = resolver.resolve().values()
    assert leaf[0]["annotations"] == []


def test_drop_task_annotations_brings_old_tables_to_the_schema(tmp_path):
    root = tmp_path / "lance"
    first, second = _delivery(tmp_path, 0, 10), _delivery(tmp_path, 1, 8)
    commit.append(root, [first])
    tasks = root / "tasks.lance"
    old = lance.dataset(tasks).to_table()
    nested = pa.list_(pa.struct([("text", pa.string())]))
    lance.write_dataset(
        old.append_column("annotations", pa.array([[]] * old.num_rows, nested)),
        tasks,
        mode="overwrite",
    )
    commit.drop_task_annotations(root)
    assert lance.dataset(tasks).schema.names == TASKS.names
    assert commit.append(root, [second]) == [second.recording_id]
    commit.drop_task_annotations(root)


def test_tag_covers_every_steps_table(tmp_path):
    root = tmp_path / "lance"
    eva = _delivery(
        tmp_path, 1, 5, embodiment="eva_bimanual", end_effector="parallel_jaw_gripper"
    )
    eva = dataclasses.replace(eva, steps=eva_steps_table(eva.recording_id, 5, 0))
    commit.append(root, [_delivery(tmp_path, 0, 10), eva])
    eva_steps = steps_path("rl2", "eva_bimanual", "parallel_jaw_gripper")
    tagged = commit.tag(root, "t1")
    assert {STEPS, eva_steps} <= set(tagged)
    assert "t1" in lance.dataset(root / eva_steps).tags.list()
