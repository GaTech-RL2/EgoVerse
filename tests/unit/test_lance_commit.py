import datetime

import numpy as np
import pyarrow as pa
import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import steps_table  # noqa: E402

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import (  # noqa: E402
    TASKS,
    recording_id,
    task_instance_id,
)

KEY_MAP = {
    "left.action_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.obs_ee_pose",
        "horizon": 3,
    },
    "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    "annotations": {"key_type": "annotation_keys", "zarr_key": "annotations"},
}


def _delivery(tmp, i, n, offset=None, video=b"mp4"):
    rid = recording_id("aria", str(i))
    h = f"2025-01-01-00-00-{i:02d}-000000"
    src = tmp / "src" / f"{i}-{offset}"
    src.mkdir(parents=True, exist_ok=True)
    (src / "front_1.mp4").write_bytes(video)
    return commit.Delivery(
        recording=dict(
            recording_id=rid,
            vendor_recording_id=h,
            lab="aria",
            site="rl2",
            embodiment="human_bimanual",
            fps=30.0,
            num_frames=n,
            intrinsics=[],
            extrinsics=[],
        ),
        task=dict(
            task_instance_id=f"t{i}",
            recording_id=rid,
            t_start_ns=0,
            t_end_ns=n * 33_333_333,
            task_name="fold",
            task_description=f"task {i}",
            objects=[],
        ),
        alias=dict(
            episode_hash=h, recording_id=rid, t_start_ns=0, t_end_ns=n * 33_333_333
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
    assert lance.dataset(root / "steps" / "aria.lance").count_rows() == 18
    assert len(lance.dataset(root / "steps" / "aria.lance").get_fragments()) == 2


def test_replace_is_one_commit_and_keeps_old_versions_readable(tmp_path):
    root = tmp_path / "lance"
    ds = [_delivery(tmp_path, i, 10) for i in range(3)]
    commit.append(root, ds)
    commit.tag(root, "t1")
    before = _resolve(root)
    steps_v = lance.dataset(root / "steps" / "aria.lance").version

    new = _delivery(tmp_path, 0, 12, offset=1000, video=b"mp4-v2")
    commit.replace(root, new)
    assert lance.dataset(root / "steps" / "aria.lance").version == steps_v + 1

    after = _resolve(root)
    h0 = new.alias["episode_hash"]
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
            "steps/aria.lance",
            pa.table({"recording_id": ["x"]}),
            on="recording_id",
        )
    leaves = _resolve(root)
    h0, h1 = (d.alias["episode_hash"] for d in ds)
    assert leaves[h0].task_text == "fixed" and leaves[h1].task_text == "task 1"

    fp = {h: leaf.cache_fingerprint() for h, leaf in leaves.items()}
    commit.add_columns(
        root, "steps/aria.lance", {"both_valid": "left_valid AND right_valid"}
    )
    commit.index(root)
    commit.compact(root)
    steps = lance.dataset(root / "steps" / "aria.lance")
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
