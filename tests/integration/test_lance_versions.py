"""Versioned commits on a copy of the Lance pilot tables.

v1 commit 24 recordings, tag t1; v2 append 10 one commit each; v3 replace one with
shifted poses; v4 edit a task's text; v5 add a column; v6 compact, tag t6, clean up.
Needs EGOVERSE_TEST_LANCE_ROOT (the pilot tables written by commit.py). CPU only.
"""

import datetime
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch

lance = pytest.importorskip("lance")

from egomimic.rldb.lance import commit  # noqa: E402
from egomimic.rldb.lance.dataset import LanceEpisodeResolver  # noqa: E402
from egomimic.rldb.lance.schema import STEPS_KEY, steps_path, steps_paths  # noqa: E402

CAMERA = "front_1"
SAMPLES_PER_LEAF = 3


@pytest.fixture(scope="module")
def pilot():
    root = os.environ.get("EGOVERSE_TEST_LANCE_ROOT")
    if not root or not Path(root).exists():
        pytest.skip("set EGOVERSE_TEST_LANCE_ROOT to the pilot Lance tables")
    return Path(root)


def _deliveries(src: Path) -> list[commit.Delivery]:
    """Rebuild a delivery per bimanual pilot recording from the committed tables."""
    recs = lance.dataset(src / "recordings.lance").to_table().to_pylist()
    tasks = {
        r["recording_id"]: r
        for r in lance.dataset(src / "tasks.lance").to_table().to_pylist()
    }
    aliases = {
        r["recording_id"]: r
        for r in lance.dataset(src / "aliases.lance").to_table().to_pylist()
    }
    out = []
    for r in sorted(recs, key=lambda r: r["recording_id"]):
        if r["embodiment"] != "human_bimanual":
            continue
        rid = r["recording_id"]
        videos = {cam: src / p for cam, p in r["video_path"] if cam == CAMERA}
        rec = {k: v for k, v in r.items() if k not in ("video_path", "content_digest")}
        steps = lance.dataset(src / steps_path(*(r[c] for c in STEPS_KEY)))
        st = steps.to_table(filter=f"recording_id = '{rid}'")
        out.append(commit.Delivery(rec, tasks[rid], aliases[rid], st, videos))
    return out


def _shift_head(d: commit.Delivery, dx: float) -> commit.Delivery:
    head = d.steps.column("head_pose").combine_chunks()
    vals = np.asarray(head.flatten()).reshape(-1, 7).copy()
    vals[:, 0] += dx
    new = pa.FixedSizeListArray.from_arrays(pa.array(vals.reshape(-1)), 7)
    steps = d.steps.set_column(
        d.steps.schema.get_field_index("head_pose"), "head_pose", new
    )
    return commit.Delivery(d.recording, d.task, d.alias, steps, d.videos)


def _key_map():
    from egomimic.rldb.embodiment.human import Human

    return Human.get_keymap(keymap_mode="keypoints"), Human.get_transform_list(
        mode="keypoints_headframe_6d", stride=3
    )


def _resolver(root, **kw):
    key_map, transforms = _key_map()
    return LanceEpisodeResolver(
        root,
        key_map=key_map,
        transform_list=transforms,
        cameras={"images.front_1": CAMERA},
        **kw,
    )


def _samples(leaves, only=None):
    out = {}
    for h, leaf in leaves.items():
        if only is not None and h not in only:
            continue
        idx = np.linspace(0, len(leaf) - 1, SAMPLES_PER_LEAF).astype(int)
        out[h] = [leaf[int(i)] for i in idx]
    return out


def _same(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
                return False
        elif x != y:
            return False
    return True


def _all_same(s1, s2, hashes) -> list[str]:
    return [
        h
        for h in hashes
        if not all(_same(a, b) for a, b in zip(s1[h], s2[h], strict=True))
    ]


def _row_count_ok(root, **kw) -> bool:
    r = _resolver(root, **kw)
    _, _, recs = r._open("recordings.lance")
    (name,) = steps_paths(recs.to_table(columns=list(STEPS_KEY)))
    _, _, steps = r._open(name)
    return (
        steps.count_rows()
        == pc.sum(recs.to_table(columns=["num_frames"]).column(0)).as_py()
    )


def test_versioned_commits(pilot, tmp_path):
    root = tmp_path / "lance"
    ds = _deliveries(pilot)
    assert len(ds) >= 25
    first, rest = ds[:24], ds[24:]

    # v1
    commit.append(root, first)
    commit.tag(root, "t1")
    leaves1 = _resolver(root).resolve()
    s1 = _samples(leaves1)
    fp1 = {h: leaf.cache_fingerprint() for h, leaf in leaves1.items()}
    assert _row_count_ok(root)

    # v2: one commit per recording, so many fragments
    for d in rest:
        commit.append(root, [d])
    steps_uri = commit.steps_uri(root, *rest[0].steps_key)
    assert len(lance.dataset(steps_uri).get_fragments()) == 1 + len(rest)
    r2 = _resolver(root)
    leaves2 = r2.resolve()
    pinned2 = dict(r2.versions)
    s2 = _samples(leaves2)
    assert _all_same(s1, s2, s1) == []
    assert {h: leaves2[h].cache_fingerprint() for h in fp1} == fp1
    assert _row_count_ok(root)

    # v3: replace one recording with shifted head poses
    victim = first[3]
    vh = victim.alias["recording_id"]
    commit.replace(root, _shift_head(victim, 0.05))
    leaves3 = _resolver(root).resolve()
    s3 = _samples(leaves3)
    untouched = [h for h in s1 if h != vh]
    assert _all_same(s2, s3, untouched) == []
    assert _all_same(s2, s3, [vh]) == [vh]
    assert leaves3[vh].row_start > leaves2[vh].row_start
    assert {h: leaves3[h].cache_fingerprint() for h in untouched} == {
        h: fp1[h] for h in untouched
    }
    assert leaves3[vh].cache_fingerprint() != fp1[vh]
    assert _row_count_ok(root)

    # v4: edit one task's text
    edited = first[5]
    commit.edit(
        root,
        "tasks.lance",
        pa.table(
            {
                "task_instance_id": [edited.task["task_instance_id"]],
                "task_description": ["edited text"],
            }
        ),
        on="task_instance_id",
    )
    leaves4 = _resolver(root).resolve()
    eh = edited.alias["recording_id"]
    assert leaves4[eh].task_text == "edited text"
    assert leaves1[eh].task_text != "edited text"
    assert _resolver(root, tag="t1").resolve()[eh].task_text == leaves1[eh].task_text

    # v5: add a column without rewriting existing data files
    files_before = {
        f.path
        for frag in lance.dataset(steps_uri).get_fragments()
        for f in frag.data_files()
    }
    commit.add_columns(
        root,
        str(Path(steps_uri).relative_to(root)),
        {"both_valid": "left_valid AND right_valid"},
    )
    steps5 = lance.dataset(steps_uri)
    files_after = {f.path for frag in steps5.get_fragments() for f in frag.data_files()}
    assert files_before < files_after
    assert len(files_after - files_before) == len(steps5.get_fragments())
    leaves5 = _resolver(root).resolve()
    assert {h: leaves5[h].cache_fingerprint() for h in untouched} == {
        h: fp1[h] for h in untouched
    }

    # A reader pinned at v2 still serves v2 data after v3-v5.
    assert _all_same(s2, _samples(leaves2), s2) == []
    assert (
        _all_same(s2, _samples(_resolver(root, versions=pinned2).resolve()), s2) == []
    )

    # v6: compact, tag, clean up
    commit.index(root)
    commit.compact(root)
    assert len(lance.dataset(steps_uri).get_fragments()) == 1
    commit.tag(root, "t6")
    leaves6 = _resolver(root).resolve()
    assert _all_same(s3, _samples(leaves6), s3) == []
    assert {h: leaves6[h].cache_fingerprint() for h in leaves6} == {
        h: leaves3[h].cache_fingerprint() for h in leaves3
    }
    assert (
        _all_same(s2, _samples(leaves2), s2) == []
    )  # compaction leaves old files in place

    commit.cleanup(root, older_than=datetime.timedelta(0))
    t1 = _resolver(root, tag="t1").resolve()
    assert _all_same(s1, _samples(t1), s1) == []
    assert {h: leaf.cache_fingerprint() for h, leaf in t1.items()} == fp1
    assert _row_count_ok(root, tag="t1") and _row_count_ok(root, tag="t6")
    # Lance's ValueError for a missing manifest, or ArrowInvalid (a ValueError) for a
    # data file cleanup removed under an already-open version.
    with pytest.raises(ValueError, match="Not found"):
        _resolver(root, versions=pinned2).resolve()
