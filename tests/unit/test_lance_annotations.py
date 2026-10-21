import argparse
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import PERIOD_NS, annotation, write_catalog  # noqa: E402

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.annotations import diff, review  # noqa: E402
from egomimic.rldb.lance.dataset import (  # noqa: E402
    LanceEpisodeResolver,
    LanceStreamingResolver,
)
from egomimic.rldb.lance.schema import ANNOTATIONS, annotation_id  # noqa: E402

KEY_MAP = {
    "left.action_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.obs_ee_pose",
        "horizon": 3,
    },
    "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    "annotations": {"key_type": "annotation_keys", "zarr_key": "annotations"},
}


def _rows(recs):
    rid = recs[0][0]
    return [
        annotation(rid, 0, 6, "reach"),
        annotation(rid, 5, 9, "grasp", quality=4),
        annotation(rid, 1, 3, "look", vendor="rl2"),
    ]


@pytest.fixture
def annotated(tmp_path):
    return tmp_path, write_catalog(tmp_path, annotations=_rows)


def _leaves(root, resolver=LanceEpisodeResolver, **ann):
    return resolver(
        root, key_map=KEY_MAP, cameras={}, annotations=ann or None
    ).resolve()


def _at_frame(leaf, frame):
    i = next(i for i in range(len(leaf)) if leaf.anchor(i) == frame)
    return leaf[i]


def _change(op, **row):
    return {"op": op, **row}


def _changes(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows])


def test_annotation_id_is_content_addressed():
    a = annotation_id("r", "scale", "language", 0, 10, "reach")
    assert a == annotation_id("r", "scale", "language", 0, 10, "reach")
    assert a != annotation_id("r", "scale", "language", 0, 10, "Reach")
    assert a != annotation_id("r", "rl2", "language", 0, 10, "reach")


def test_without_annotations_config_the_task_text_stands_in(annotated):
    root, recs = annotated
    leaf = _leaves(root)[recs[0][0]]
    assert leaf[0]["annotations"] == ["folding clothes"]


def test_covering_annotations_in_start_order(annotated):
    root, recs = annotated
    leaf = _leaves(root, types=["language"])[recs[0][0]]
    assert _at_frame(leaf, 1)["annotations"] == ["reach", "look"]
    assert _at_frame(leaf, 5)["annotations"] == ["reach", "grasp"]
    assert _at_frame(leaf, 9)["annotations"] == []
    other = _leaves(root, types=["language"])[recs[1][0]]
    assert other[0]["annotations"] == []


def test_vendor_and_quality_filters(annotated):
    root, recs = annotated
    h = recs[0][0]
    assert _at_frame(_leaves(root, vendors=["scale"])[h], 1)["annotations"] == ["reach"]
    graded = _leaves(root, min_quality=4)[h]
    assert _at_frame(graded, 5)["annotations"] == ["grasp"]
    assert _at_frame(graded, 0)["annotations"] == []


def test_clamp_actions_stops_at_the_covering_annotation_end(annotated):
    root, recs = annotated
    h = recs[0][0]
    plain = _leaves(root)[h]
    clamped = _leaves(root, clamp_actions=True)[h]
    # Frame 5 is covered by "reach" [0, 6) and "grasp" [5, 9): the earlier end wins.
    assert _at_frame(plain, 5)["action_pad_mask"].tolist() == [1, 1, 1]
    assert _at_frame(clamped, 5)["action_pad_mask"].tolist() == [1, 0, 0]
    assert _at_frame(clamped, 0)["action_pad_mask"].tolist() == [1, 1, 1]
    assert plain.cache_fingerprint() != clamped.cache_fingerprint()


def test_require_keeps_only_annotated_anchors(annotated):
    root, recs = annotated
    leaves = _leaves(root, require=True)
    assert list(leaves) == [recs[0][0]]
    # Eligible anchors are 0, 1 and 5.. (frame 4 is invalid); annotations cover [0, 9).
    assert [leaves[recs[0][0]].anchor(i) for i in range(6)] == [0, 1, 5, 6, 7, 8]
    assert len(leaves[recs[0][0]]) == 6


def test_streaming_batch_gives_the_same_annotations(annotated):
    from egomimic.rldb.multi import MultiDataset

    root, _ = annotated
    local = MultiDataset(datasets=_leaves(root, clamp_actions=True), mode="total")
    stream = MultiDataset(
        datasets=_leaves(root, LanceStreamingResolver, clamp_actions=True),
        mode="total",
    )
    idxs = list(range(len(local)))
    for a, b in zip(stream.__getitems__(idxs), [local[i] for i in idxs], strict=True):
        assert a["annotations"] == b["annotations"]
        assert a["action_pad_mask"].tolist() == b["action_pad_mask"].tolist()


def test_unknown_annotation_option_is_rejected(annotated):
    root, _ = annotated
    with pytest.raises(ValueError, match="unknown annotations options"):
        _leaves(root, typo=True)


def _add(rid, t0, t1, text="x", **kw):
    row = dict(
        recording_id=rid,
        annotation_vendor="scale",
        annotation_type="language",
        t_start_ns=t0,
        t_end_ns=t1,
        text=text,
    )
    return _change("add", **{**row, **kw})


def test_review_rejects_each_bad_row(annotated):
    root, recs = annotated
    rid, n = recs[0][0], recs[0][2]
    ok = _add(rid, 0, PERIOD_NS)
    bad = [
        _add(rid, 0, PERIOD_NS, annotation_type="subtask"),
        _add(rid, PERIOD_NS, PERIOD_NS),
        _add(rid, 0, (n + 1) * PERIOD_NS),
        _add("no-such-recording", 0, PERIOD_NS),
        _add(rid, 0, PERIOD_NS, text="y", quality_score=9),
        _add(rid, 0, PERIOD_NS, text="z", annotation_id="wrong"),
        ok,
        _change("delete", annotation_id="never-committed"),
        _change("upsert", annotation_id="x"),
    ]
    rv = review(root, _changes([ok, *bad]))
    assert rv.adds.num_rows == 1
    joined = "\n".join(rv.errors)
    for needle in (
        "row 1: annotation_type 'subtask'",
        "row 2: t_start_ns",
        "row 3: [0, ",
        "row 4: recording no-such-recording has no tasks",
        "row 5: quality_score 9",
        "row 6: annotation_id wrong does not match",
        "row 7: ",
        "row 8: delete of never-committed",
        "row 9: op 'upsert'",
    ):
        assert needle in joined, needle
    assert len(rv.errors) == 9


def test_apply_adds_and_deletes_in_one_version_and_resubmit_is_a_noop(annotated):
    root, recs = annotated
    rid = recs[0][0]
    uri = str(root / "annotations.lance")
    old = _rows(recs)[0]
    edited = _add(rid, 0, 6 * PERIOD_NS, text="reach for the shirt")
    changes = _changes([_change("delete", annotation_id=old["annotation_id"]), edited])
    rv = review(root, changes)
    assert not rv.errors
    assert "-1" in diff(root, rv) and "reach for the shirt" in diff(root, rv)
    v = lance.dataset(uri).version
    assert commit.apply_annotations(root, rv.adds, rv.deletes, "approved by me") == {
        "added": 1,
        "deleted": 1,
    }
    ds = lance.dataset(uri)
    assert ds.version == v + 1
    props = ds.read_transaction(ds.version).transaction_properties
    assert "approved by me" in props.values()
    texts = set(ds.to_table(columns=["text"]).column(0).to_pylist())
    assert texts == {"reach for the shirt", "grasp", "look"}

    again = review(root, _changes([edited]))
    assert again.adds.num_rows == 0 and len(again.unchanged) == 1
    assert commit.apply_annotations(root, again.adds, again.deletes) == {
        "added": 0,
        "deleted": 0,
    }
    assert lance.dataset(uri).version == v + 1


def test_grading_through_edit_keeps_the_id(annotated):
    root, recs = annotated
    aid = _rows(recs)[0]["annotation_id"]
    commit.edit(
        root,
        "annotations.lance",
        pa.table(
            {
                "annotation_id": [aid],
                "quality_score": pa.array([3], pa.int8()),
                "graded_by": ["grader"],
                "rubric_version": ["v1"],
            }
        ),
        on="annotation_id",
    )
    rows = lance.dataset(str(root / "annotations.lance")).to_table().to_pylist()
    row = next(r for r in rows if r["annotation_id"] == aid)
    assert (row["quality_score"], row["graded_by"], row["text"]) == (
        3,
        "grader",
        "reach",
    )
    assert lance.dataset(str(root / "annotations.lance")).schema == ANNOTATIONS


def test_edit_refuses_annotation_content_columns(annotated):
    root, recs = annotated
    aid = _rows(recs)[0]["annotation_id"]
    with pytest.raises(ValueError, match="quality_score"):
        commit.edit(
            root,
            "annotations.lance",
            pa.table({"annotation_id": [aid], "text": ["grasp"]}),
            on="annotation_id",
        )


def test_from_zarr_maps_clip_frames_onto_the_recording_clock(tmp_path):
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    from egomimic.scripts.lance_annotations import cmd_from_zarr

    root = tmp_path / "lance"
    recs = write_catalog(root)
    rid, h, _ = recs[0]
    # A clip of recording frames 2..9 with frame 5 dropped, as zarr cut and filtered it.
    frames = np.array([2, 3, 4, 6, 7, 8, 9])
    ZarrWriter.create_and_write(
        tmp_path / "zarr" / f"{h}.zarr",
        numeric_data={
            "obs_rgb_timestamps_ns": (frames * PERIOD_NS).astype(np.int64),
            "obs_head_pose": np.zeros((len(frames), 7), np.float32),
        },
        embodiment="human_bimanual",
        annotations=[("reach", 0, 3), ("fold", 4, 7), ("bad", 3, 2)],
        intrinsics={"front_1": np.eye(3, 4)},
        task_name="synthetic",
        operator="op_00",
        lab="synthetic",
        scene="test_scene",
        objects=["cube"],
        vendor_recording_id="synthetic_session",
    )
    out = tmp_path / "changes.parquet"
    cmd_from_zarr(
        argparse.Namespace(
            lance=str(root), zarr=tmp_path / "zarr", vendor="scale", out=out
        )
    )
    rows = pq.read_table(out).to_pylist()
    spans = {r["text"]: (r["t_start_ns"], r["t_end_ns"]) for r in rows}
    # [0, 3) ends at the clip's next kept frame (recording frame 6), so frame 5 is covered.
    assert spans["reach"] == (2 * PERIOD_NS, 6 * PERIOD_NS)
    # Running to the clip's end, it ends at the recording frame after the clip's last.
    assert spans["fold"] == (7 * PERIOD_NS, 10 * PERIOD_NS)
    assert "bad" not in spans
    rv = review(root, pq.read_table(out))
    assert not rv.errors and rv.adds.num_rows == 2


def test_from_bucket_counts_frames_from_the_alias_start(tmp_path):
    from fixtures.lance_tables import alias

    from egomimic.scripts.lance_annotations import cmd_from_bucket

    root = tmp_path / "lance"
    recs = write_catalog(
        root, aliases=lambda recs: [alias(rid, 3, n, h) for rid, h, n in recs]
    )
    (rid, h, n), (_, h2, _) = recs
    bucket = tmp_path / "bucket"
    bucket.mkdir()
    spans = [
        {"text": "reach", "start_idx": 0, "end_idx": 2},
        {"text": "fold", "start_idx": 2, "end_idx": n - 3},
        {"text": "too long", "start_idx": 0, "end_idx": n},
    ]
    (bucket / f"{h}_annotations.json").write_text(json.dumps(spans))
    (bucket / "unknown_annotations.json").write_text(json.dumps(spans))
    (bucket / f"{h2}_other.json").write_text(json.dumps(spans))
    out = tmp_path / "changes.parquet"
    cmd_from_bucket(
        argparse.Namespace(
            lance=str(root),
            bucket=str(bucket),
            key="annotations",
            vendor="scale",
            out=out,
        )
    )
    rows = pq.read_table(out).to_pylist()
    assert {r["text"]: (r["t_start_ns"], r["t_end_ns"]) for r in rows} == {
        "reach": (3 * PERIOD_NS, 5 * PERIOD_NS),
        "fold": (5 * PERIOD_NS, n * PERIOD_NS),
    }
    assert {r["recording_id"] for r in rows} == {rid}
    assert not review(root, pq.read_table(out)).errors


def test_commit_annotations_records_and_tags_the_change(annotated):
    root, recs = annotated
    rid = recs[0][0]
    old = _rows(recs)[0]
    rv = review(
        root,
        _changes(
            [
                _change("delete", annotation_id=old["annotation_id"]),
                _add(recs[1][0], 0, PERIOD_NS, text="new"),
            ]
        ),
    )
    out = commit.commit_annotations(root, rv.adds, rv.deletes, operator="me")
    assert out["tag"].startswith("annotations-")
    assert out["rows"] == {"added": 1, "deleted": 1}
    (row,) = lance.dataset(str(root / "changes.lance")).to_table().to_pylist()
    assert (row["lab"], row["added"], row["replaced"], row["operator"]) == (
        "annotations",
        [recs[1][0]],
        [rid],
        "me",
    )
    ds = lance.dataset(str(root / "annotations.lance"))
    assert ds.tags.list()[out["tag"]]["version"] == ds.version
    assert commit.commit_annotations(root, rv.adds.slice(0, 0), []) is None
