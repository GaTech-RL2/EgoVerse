import numpy as np
import pyarrow as pa
import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import steps_table  # noqa: E402

from egomimic.rldb.filters import DatasetFilter  # noqa: E402
from egomimic.rldb.lance.dataset import (  # noqa: E402
    LanceEpisodeResolver,
    LanceRecordingDataset,
    LanceStreamingDataset,
    LanceStreamingResolver,
    _runs,
    eligible_intervals,
)
from egomimic.rldb.lance.schema import (  # noqa: E402
    ALIASES,
    RECORDINGS,
    TASKS,
    recording_id,
    steps_path,
)


def test_eligible_intervals_requires_whole_chunk_valid():
    n = 10
    valid = {
        "head": np.ones(n, bool),
        "left": np.ones(n, bool),
        "right": np.ones(n, bool),
    }
    valid["left"][5] = False
    iv = eligible_intervals(valid, ("left", "right"), horizon=3)
    # anchors 3, 4, 5 have frame 5 in [t, t+3); the tail chunk is clamped at n.
    np.testing.assert_array_equal(iv, [[0, 3], [6, 10]])


def test_eligible_intervals_ignores_unused_hand_and_needs_head():
    n = 6
    valid = {
        "head": np.ones(n, bool),
        "left": np.zeros(n, bool),
        "right": np.ones(n, bool),
    }
    valid["head"][2] = False
    iv = eligible_intervals(valid, ("right",), horizon=2)
    np.testing.assert_array_equal(iv, [[0, 2], [3, 6]])


def test_runs_merge_across_chunks_and_keep_repeats():
    ids = pa.chunked_array([["a", "a", "b"], ["b", "c"], [], ["c", "a"]])
    values, starts = _runs(ids)
    assert values == ["a", "b", "c", "a"]
    np.testing.assert_array_equal(starts, [0, 2, 4, 6])
    values, starts = _runs(pa.chunked_array([], pa.string()))
    assert values == [] and len(starts) == 0


@pytest.fixture
def catalog(tmp_path):
    recs = [
        (recording_id("aria", "1"), "2025-01-01-00-00-00-000000", 12),
        (recording_id("aria", "2"), "2025-01-01-00-00-01-000000", 8),
    ]
    recs.sort()
    steps = pa.concat_tables(
        [steps_table(rid, n, 100 * i) for i, (rid, _, n) in enumerate(recs)]
    )
    lance.write_dataset(steps, tmp_path / steps_path("rl2", "human_bimanual", "hand"))
    k = [200.0, 0, 240, 0, 200.0, 180, 0, 0, 1]
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                dict(
                    recording_id=rid,
                    vendor_recording_id=h,
                    lab="rl2",
                    embodiment="human_bimanual",
                    end_effector="hand",
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
        pa.Table.from_pylist(
            [
                dict(
                    task_instance_id=f"t{i}",
                    recording_id=rid,
                    t_start_ns=0,
                    t_end_ns=n,
                    task_name="fold_clothes",
                    task_description="folding clothes",
                    objects=[],
                    annotations=[],
                )
                for i, (rid, _, n) in enumerate(recs)
            ],
            schema=TASKS,
        ),
        tmp_path / "tasks.lance",
    )
    lance.write_dataset(
        pa.Table.from_pylist(
            [
                dict(episode_hash=h, recording_id=rid, t_start_ns=0, t_end_ns=n)
                for rid, h, n in recs
            ],
            schema=ALIASES,
        ),
        tmp_path / "aliases.lance",
    )
    return tmp_path, recs


KEY_MAP = {
    "left.action_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.obs_ee_pose",
        "horizon": 3,
    },
    "right.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_ee_pose"},
    "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    "annotations": {"key_type": "annotation_keys", "zarr_key": "annotations"},
}


def test_resolver_reads_windows_from_the_right_rows(catalog):
    root, recs = catalog
    leaves = LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}).resolve()
    assert set(leaves) == {h for _, h, _ in recs}
    for i, (_, h, n) in enumerate(recs):
        leaf = leaves[h]
        # left_valid is false at frame 4, so anchors 2..4 are dropped.
        assert len(leaf) == n - 3
        assert [leaf.anchor(j) for j in range(3)] == [0, 1, 5]
        s = leaf[2]
        offset = 100 * i
        np.testing.assert_allclose(
            s["left.action_ee_pose"][:, 0].numpy(), offset + 0.5 + np.arange(5, 8)
        )
        assert float(s["right.obs_ee_pose"][0]) == offset + 5.25
        assert float(s["obs_head_pose"][0]) == offset + 5
        assert s["annotations"] == ["folding clothes"]
        assert s["action_pad_mask"].tolist() == [1, 1, 1]
        tail = leaf[len(leaf) - 1]
        assert tail["action_pad_mask"].tolist() == [1, 0, 0]


def test_resolver_applies_filters(catalog):
    root, recs = catalog
    pin = recs[0][1]
    leaves = LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}).resolve(
        filters=DatasetFilter(
            filter_lambdas=["lambda row: row['lab'] == 'rl2'"], episode_hashes=[pin]
        )
    )
    assert list(leaves) == [pin]


def _same_sample(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if hasattr(a[k], "shape"):
            assert a[k].shape == b[k].shape and bool((a[k] == b[k]).all()), k
        else:
            assert a[k] == b[k], k


def test_streaming_getitems_batches_reads_and_local_does_not(catalog, monkeypatch):
    from egomimic.rldb.lance import dataset as lance_dataset
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    root, _ = catalog
    local = MultiDataset(
        datasets=LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}).resolve(),
        mode="total",
    )
    leaves = LanceStreamingResolver(
        root, video_chunks={}, key_map=KEY_MAP, cameras={}
    ).resolve()
    ds = MultiDataset(datasets=leaves, mode="total")
    idxs = [0, len(ds) - 1, 3, 3, len(ds) // 2, 1]
    want = [local[i] for i in idxs]

    takes = []
    real_fetch = lance_dataset._fetch_numeric
    monkeypatch.setattr(
        lance_dataset,
        "_fetch_numeric",
        lambda items: takes.append(len(items)) or real_fetch(items),
    )
    for a, b in zip(ds.__getitems__(idxs), want, strict=True):
        _same_sample(a, b)
    assert takes == [len(idxs)]  # one fetch for the whole batch, both recordings

    takes.clear()
    for a, b in zip(local.__getitems__(idxs), want, strict=True):
        _same_sample(a, b)
    assert takes == [1] * len(idxs)

    def broken(items):
        raise OSError("object store unavailable")

    monkeypatch.setattr(LanceStreamingDataset, "fetch_planned", staticmethod(broken))
    # Only the batched read fails; the per-sample fallback still reads the store.
    monkeypatch.setattr(
        LanceStreamingDataset, "__getitem__", LanceRecordingDataset.__getitem__
    )
    for a, b in zip(ds.__getitems__(idxs), want, strict=True):
        _same_sample(a, b)


def test_streaming_resolver_needs_a_chunk_table_per_camera(catalog):
    root, _ = catalog
    key_map = {
        **KEY_MAP,
        "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1"},
    }
    with pytest.raises(ValueError, match="images.front_1"):
        LanceStreamingResolver(root, video_chunks={}, key_map=key_map)


@pytest.mark.parametrize("overlap_reads", [True, False])
def test_streaming_getitems_split_into_parts_matches(
    catalog, monkeypatch, overlap_reads
):
    from egomimic.rldb.lance import dataset as lance_dataset
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    root, _ = catalog
    local = MultiDataset(
        datasets=LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}).resolve(),
        mode="total",
    )
    ds = MultiDataset(
        datasets=LanceStreamingResolver(
            root,
            video_chunks={},
            fetch_split=2,
            overlap_reads=overlap_reads,
            key_map=KEY_MAP,
            cameras={},
        ).resolve(),
        mode="total",
    )
    idxs = [0, len(ds) - 1, 3, 3, len(ds) // 2]
    want = [local[i] for i in idxs]
    takes = []
    real_fetch = lance_dataset._fetch_numeric
    monkeypatch.setattr(
        lance_dataset,
        "_fetch_numeric",
        lambda items: takes.append(len(items)) or real_fetch(items),
    )
    for a, b in zip(ds.__getitems__(idxs), want, strict=True):
        _same_sample(a, b)
    assert sorted(takes) == [1, 2, 2]
