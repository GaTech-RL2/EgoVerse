import numpy as np
import pyarrow as pa
import pytest

lance = pytest.importorskip("lance")

from fixtures.lance_tables import alias, task, write_catalog  # noqa: E402

from egomimic.rldb.filters import DatasetFilter, PinError  # noqa: E402
from egomimic.rldb.lance.dataset import (  # noqa: E402
    LanceEpisodeResolver,
    LanceRecordingDataset,
    LanceStreamingDataset,
    LanceStreamingResolver,
    _runs,
    eligible_intervals,
)
from egomimic.rldb.lance.schema import task_instance_id  # noqa: E402


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


def test_eligible_intervals_requires_history_frames_valid():
    n = 10
    valid = {
        "head": np.ones(n, bool),
        "left": np.ones(n, bool),
        "right": np.ones(n, bool),
    }
    valid["right"][4] = False
    iv = eligible_intervals(valid, ("right",), horizon=1, history=[0, 2])
    # anchor 4 itself, and anchor 6 whose history step 6 - 2 is frame 4.
    np.testing.assert_array_equal(iv, [[0, 4], [5, 6], [7, 10]])


def test_history_offsets_follow_zarr_history_indices():
    key_map = {
        "obs": {"key_type": "proprio_keys", "zarr_key": "x", "history": 3},
        "lagged": {
            "key_type": "proprio_keys",
            "zarr_key": "y",
            "history": 2,
            "history_stride_s": 0.1,
            "lag_s": 0.05,
        },
        "cam": {"key_type": "camera_keys", "zarr_key": "z", "history": 4},
    }
    r = LanceEpisodeResolver("unused", key_map=key_map)
    # obs: 0, 1, 2; lagged at 30 fps: lag 2 frames, stride 3 -> 2, 5.
    assert r._history_offsets(30.0) == [0, 1, 2, 5]


def test_runs_merge_across_chunks_and_keep_repeats():
    ids = pa.chunked_array([["a", "a", "b"], ["b", "c"], [], ["c", "a"]])
    values, starts = _runs(ids)
    assert values == ["a", "b", "c", "a"]
    np.testing.assert_array_equal(starts, [0, 2, 4, 6])
    values, starts = _runs(pa.chunked_array([], pa.string()))
    assert values == [] and len(starts) == 0


@pytest.fixture
def catalog(tmp_path):
    return tmp_path, write_catalog(tmp_path)


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
    assert set(leaves) == {rid for rid, _, _ in recs}
    for i, (rid, _, n) in enumerate(recs):
        leaf = leaves[rid]
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
    assert list(leaves) == [recs[0][0]]


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
    leaves = LanceStreamingResolver(root, key_map=KEY_MAP, cameras={}).resolve()
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


def test_even_stride_getitems_reads_strided_samples(catalog):
    from egomimic.rldb.zarr.zarr_dataset_multi import EvenStrideDataset, MultiDataset

    root, _ = catalog
    local = MultiDataset(
        datasets=LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}).resolve(),
        mode="total",
    )
    ds = MultiDataset(
        datasets=LanceStreamingResolver(root, key_map=KEY_MAP, cameras={}).resolve(),
        mode="total",
    )
    strided = EvenStrideDataset(ds, stride=3)
    idxs = list(range(len(strided)))
    want = [local[strided.indices[i]] for i in idxs]
    for a, b in zip(strided.__getitems__(idxs), want, strict=True):
        _same_sample(a, b)


@pytest.mark.parametrize("n, shape", [(9, (3, 3)), (12, (3, 4))])
def test_intrinsics_reshapes_flat_k(n, shape):
    leaf = LanceRecordingDataset.__new__(LanceRecordingDataset)
    leaf._intrinsics = np.arange(n, dtype=np.float32)
    assert leaf.intrinsics.shape == shape


IMAGE_KEY_MAP = {
    **KEY_MAP,
    "front_img_1": {"key_type": "camera_keys", "zarr_key": "images.front_1"},
}


def _videos(root, recs, short=None):
    av = pytest.importorskip("av")
    from fixtures.videos import write_av1

    for rid, _, n in recs:
        path = root / "videos" / rid / "front_1.mp4"
        path.parent.mkdir(parents=True)
        try:
            write_av1(path, n - 1 if rid == short else n)
        except (av.FFmpegError, ValueError) as e:
            pytest.skip(f"no AV1 encoder: {e}")


class _PyAVDecoder:
    """Stands in for torchcodec: a full sequential decode, independent of mp4_ranges."""

    def __init__(self, path):
        import av

        with av.open(path) as c:
            self._frames = np.stack(
                [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
            )

    def get_frames_at(self, indices):
        import types

        return types.SimpleNamespace(
            data=types.SimpleNamespace(numpy=lambda: self._frames[indices])
        )


def test_streaming_frames_from_byte_ranges_match_a_full_decode(catalog, monkeypatch):
    from egomimic.rldb.lance import dataset as lance_dataset
    from egomimic.rldb.lance import mp4_ranges
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    root, recs = catalog
    _videos(root, recs)
    monkeypatch.setattr(lance_dataset, "_decoder", _PyAVDecoder)
    monkeypatch.setattr(lance_dataset, "_MP4S", lance_dataset.OrderedDict())
    kw = dict(key_map=IMAGE_KEY_MAP, cameras={"images.front_1": "front_1"})
    local = MultiDataset(
        datasets=LanceEpisodeResolver(root, **kw).resolve(), mode="total"
    )
    ds = MultiDataset(
        datasets=LanceStreamingResolver(root, **kw).resolve(), mode="total"
    )
    indexed = []
    real_index = mp4_ranges.read_index
    monkeypatch.setattr(
        mp4_ranges,
        "read_index",
        lambda read, *a: indexed.append(read) or real_index(read, *a),
    )
    # Two samples per recording, so a batch shares each MP4 between samples.
    idxs = [0, 1, len(ds) - 1, len(ds) - 2, 3]
    got = ds.__getitems__(idxs)
    for a, i in zip(got, idxs, strict=True):
        b = local[i]
        assert a["front_img_1"].shape == (3, 48, 64)
        _same_sample(a, b)
    assert len(indexed) == len(recs)  # each MP4's index is read once per process


def test_streaming_frames_reject_an_mp4_with_the_wrong_frame_count(catalog):
    from egomimic.rldb.lance import dataset as lance_dataset

    root, recs = catalog
    _videos(root, recs, short=recs[0][0])
    leaf = LanceStreamingResolver(
        root, key_map=IMAGE_KEY_MAP, cameras={"images.front_1": "front_1"}
    ).resolve()[recs[0][0]]
    with pytest.raises(ValueError, match="frames, num_frames"):
        lance_dataset._fetch_range_frames([(leaf, leaf.plan_sample(0))])


def _all_valid(rid, n):
    return {"left_valid": np.ones(n, bool)}


def _two_tasks(recs):
    """The 12-frame recording: reach [0, 4), a gap, fold [6, 12); the 8-frame one: one
    task, and two old clips aliased to its halves."""
    (a, _, _), (b, _, _) = _by_length(recs)
    return [
        task(a, 0, 4, "reach", "reach"),
        task(a, 6, 12, "fold", "fold"),
        task(b, 0, 8),
    ]


def _clips(recs):
    (a, ha, _), (b, _, _) = _by_length(recs)
    return [alias(a, 0, 12, ha), alias(b, 0, 4, "clip_b1"), alias(b, 4, 8, "clip_b2")]


def _by_length(recs):
    return sorted(recs, key=lambda r: -r[2])


@pytest.fixture
def segmented(tmp_path):
    recs = write_catalog(tmp_path, tasks=_two_tasks, aliases=_clips, valid=_all_valid)
    (a, _, _), (b, _, _) = _by_length(recs)
    return tmp_path, a, b


def _frames(leaf):
    return [leaf.anchor(i) for i in range(len(leaf))]


def _resolve(root, filters=None, **kw):
    return LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={}, **kw).resolve(
        filters=filters
    )


def test_one_leaf_per_recording_sampling_only_its_tasks(segmented):
    root, a, b = segmented
    leaves = _resolve(root)
    assert set(leaves) == {a, b}
    assert _frames(leaves[a]) == [0, 1, 2, 3, *range(6, 12)]
    assert _frames(leaves[b]) == list(range(8))
    assert leaves[a][1]["annotations"] == ["reach"]
    assert leaves[a][4]["annotations"] == ["fold"]


def test_taskless_frames_only_when_asked(segmented):
    root, a, _ = segmented
    leaf = _resolve(root, include_taskless_frames=True)[a]
    assert _frames(leaf) == list(range(12))
    assert leaf[4]["annotations"] == []
    gaps_only = _resolve(
        root,
        DatasetFilter(filter_lambdas=["lambda row: row['task'] is None"]),
        include_taskless_frames=True,
    )
    assert list(gaps_only) == [a] and _frames(gaps_only[a]) == [4, 5]


def test_filter_lambdas_run_per_task(segmented):
    root, a, _ = segmented
    leaves = _resolve(
        root, DatasetFilter(filter_lambdas=["lambda row: row['task'] == 'fold'"])
    )
    assert list(leaves) == [a] and _frames(leaves[a]) == list(range(6, 12))
    tid = task_instance_id(a, "reach", 0)
    leaves = _resolve(
        root,
        DatasetFilter(
            filter_lambdas=[f"lambda row: row['task_instance_id'] != '{tid}'"]
        ),
    )
    assert _frames(leaves[a]) == list(range(6, 12))


def test_clip_hash_pins_and_excludes_its_time_range(segmented):
    root, a, b = segmented
    pinned = _resolve(root, DatasetFilter(episode_hashes=["clip_b2"]))
    assert list(pinned) == [b] and _frames(pinned[b]) == [4, 5, 6, 7]
    both = _resolve(root, DatasetFilter(episode_hashes=["clip_b2", a]))
    assert _frames(both[a]) == [0, 1, 2, 3, *range(6, 12)]
    excluded = _resolve(root, DatasetFilter(exclude_hashes=["clip_b1", "zarr-only"]))
    assert _frames(excluded[b]) == [4, 5, 6, 7] and a in excluded
    with pytest.raises(PinError, match="nope"):
        _resolve(root, DatasetFilter(episode_hashes=["nope"]))


def test_clip_pins_through_multidataset(segmented):
    from egomimic.rldb.multi import MultiDataset, pinned_episode_subset

    root, _, b = segmented
    resolver = LanceEpisodeResolver(root, key_map=KEY_MAP, cameras={})
    ds = MultiDataset._from_resolver(
        resolver, filters=DatasetFilter(episode_hashes=["clip_b1"]), mode="total"
    )
    assert list(ds.datasets) == [b]
    whole = MultiDataset(datasets=resolver.resolve(), mode="total")
    video = pinned_episode_subset(whole, ["clip_b2"])
    assert _frames(video.datasets["clip_b2"]) == [4, 5, 6, 7]
    with pytest.raises(PinError):
        pinned_episode_subset(whole, ["nope"])


HISTORY_KEY_MAP = {
    **KEY_MAP,
    "obs_head_pose": {
        "key_type": "proprio_keys",
        "zarr_key": "obs_head_pose",
        "history": 3,
    },
}


def _at(leaf, frame):
    return leaf[_frames(leaf).index(frame)]


def test_action_chunks_stop_at_the_task_end_unless_asked(segmented):
    root, a, _ = segmented
    leaf = _resolve(root)[a]
    assert _at(leaf, 2)["action_pad_mask"].tolist() == [1, 1, 0]
    head = _at(leaf, 2)["left.action_ee_pose"][:, 0].tolist()
    assert head == [2.5, 3.5, 3.5]
    through = _resolve(root, action_clamp="recording")[a]
    assert _at(through, 2)["action_pad_mask"].tolist() == [1, 1, 1]
    gaps = _resolve(root, include_taskless_frames=True)[a]
    assert _at(gaps, 4)["action_pad_mask"].tolist() == [1, 1, 0]


def test_history_floor_at_the_task_start_when_asked(segmented):
    root, a, _ = segmented
    reach_back = LanceEpisodeResolver(
        root, key_map=HISTORY_KEY_MAP, cameras={}
    ).resolve()[a]
    s = _at(reach_back, 7)
    assert s["obs_head_pose"][:, 0].tolist() == [5, 6, 7]
    floored = LanceEpisodeResolver(
        root, key_map=HISTORY_KEY_MAP, cameras={}, history_floor="task"
    ).resolve()[a]
    s = _at(floored, 7)
    assert s["obs_head_pose"][:, 0].tolist() == [6, 6, 7]
    with pytest.raises(ValueError, match="history_floor"):
        LanceEpisodeResolver(root, key_map=KEY_MAP, history_floor="episode")


def _hands(n=8):
    valid = {h: np.ones(n, bool) for h in ("head", "left", "right")}
    valid["left"][2] = False
    valid["right"][5] = False
    return valid


@pytest.mark.parametrize(
    "require, want",
    [
        ("both", [[0, 1], [3, 4], [6, 8]]),
        ("either", [[0, 8]]),
        ("left", [[0, 1], [3, 8]]),
        ("right", [[0, 4], [6, 8]]),
        ("none", [[0, 8]]),
    ],
)
def test_require_hands_in_frame_rules(require, want):
    iv = eligible_intervals(_hands(), ("left", "right"), 2, require)
    np.testing.assert_array_equal(iv, want)


def test_require_hands_always_needs_the_head():
    valid = _hands()
    valid["head"][7] = False
    for require in ("either", "none"):
        assert eligible_intervals(valid, ("left", "right"), 2, require)[-1, 1] == 7


def test_require_a_hand_the_embodiment_lacks_is_an_error():
    with pytest.raises(ValueError, match="left"):
        eligible_intervals(_hands(), ("right",), 2, "left")


def test_samples_carry_per_hand_masks(catalog):
    root, recs = catalog
    rid = recs[0][0]
    both = _resolve(root)[rid]
    s = both[0]
    assert s["action_hand_mask"].tolist() == [1, 1]
    assert s["proprio_hand_valid"].tolist() == [1, 1]
    either = _resolve(root, require_hands_in_frame="either")[rid]
    assert _frames(either) == list(range(recs[0][2]))
    # left is untracked at frame 4: in the chunk of anchor 3, at the anchor 4.
    assert _at(either, 3)["action_hand_mask"].tolist() == [0, 1]
    assert _at(either, 3)["proprio_hand_valid"].tolist() == [1, 1]
    assert _at(either, 4)["proprio_hand_valid"].tolist() == [0, 1]
    assert _at(either, 5)["action_hand_mask"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="require_hands_in_frame"):
        _resolve(root, require_hands_in_frame="at_least_one")


@pytest.mark.parametrize("episode_cache", [False, True])
def test_norm_stats_leave_out_untracked_hand_chunks(catalog, tmp_path, episode_cache):
    from egomimic.rldb.multi import MultiDataset

    root, recs = catalog
    leaves = _resolve(root, require_hands_in_frame="either")
    ds = MultiDataset(datasets=leaves, mode="total")
    stats = MultiDataset(state={}, norm_mode="zscore")
    stats.populate_from_datasets({"human_bimanual": ds})
    stats.infer_shapes_from_batch(ds[0])
    stats.infer_norm_from_dataset(
        ds,
        "human_bimanual",
        sample_frac=1.0,
        num_workers=0,
        episode_cache=tmp_path / "norm" if episode_cache else None,
    )
    emb = int(ds[0]["embodiment"])
    tracked = [
        s["left.action_ee_pose"].numpy()
        for s in (ds[i] for i in range(len(ds)))
        if s["action_hand_mask"][0] == 1
    ]
    assert len(tracked) == len(ds) - 3 * len(recs)
    np.testing.assert_allclose(
        stats.norm_stats[emb]["left.action_ee_pose"]["mean"],
        np.mean(tracked, axis=0),
        rtol=1e-5,
    )


EVA_KEY_MAP = {
    "left.cmd_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.cmd_ee_pose",
        "horizon": 3,
    },
    "left.cmd_gripper": {
        "key_type": "action_keys",
        "zarr_key": "left.cmd_gripper",
        "horizon": 3,
    },
    "right.obs_ee_pose": {"key_type": "proprio_keys", "zarr_key": "right.obs_ee_pose"},
    "left.obs_gripper": {"key_type": "proprio_keys", "zarr_key": "left.obs_gripper"},
}


def _eva(tmp_path, valid=lambda rid, n: {}):
    recs = write_catalog(tmp_path, source="eva", valid=valid)
    (a, _, _), (b, _, _) = _by_length(recs)
    return tmp_path, a, b


def _resolve_eva(root, key_map=EVA_KEY_MAP, **kw):
    return LanceEpisodeResolver(root, key_map=key_map, cameras={}, **kw).resolve()


def test_eva_reads_robot_columns_by_their_zarr_keys(tmp_path):
    root, a, b = _eva(tmp_path)
    leaf = _resolve_eva(root)[a]
    assert _frames(leaf) == list(range(12))
    s = _at(leaf, 2)
    offset = 100 * sorted([a, b]).index(a)
    np.testing.assert_allclose(
        s["left.cmd_ee_pose"][:, 0].numpy(), offset + 0.5 + np.arange(3, 6)
    )
    assert s["left.cmd_gripper"].shape == (3, 1)
    np.testing.assert_allclose(
        s["left.cmd_gripper"][:, 0].numpy(), np.arange(2, 5) / 12
    )
    assert float(s["right.obs_ee_pose"][0]) == offset + 2.25
    assert float(s["left.obs_gripper"][0]) == 0.5
    assert s["action_hand_mask"].tolist() == [1, 1]


def test_eva_arm_dropouts_gate_anchors_and_mask_the_arm(tmp_path):
    def dropout(rid, n):
        v = np.ones(n, bool)
        v[4] = False
        return {"right_valid": v} if n == 12 else {}

    root, a, _ = _eva(tmp_path, valid=dropout)
    assert _frames(_resolve_eva(root)[a]) == [0, 1, *range(5, 12)]
    either = _resolve_eva(root, require_hands_in_frame="either")[a]
    assert _at(either, 3)["action_hand_mask"].tolist() == [1, 0]


def test_eva_single_arm_robot_never_masks_the_missing_arm(tmp_path):
    root, _, b = _eva(tmp_path)
    key_map = {k: v for k, v in EVA_KEY_MAP.items() if not k.startswith("right.")}
    leaf = _resolve_eva(root, key_map=key_map)[b]
    assert _frames(leaf) == list(range(8))
    assert leaf[0]["action_hand_mask"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="require_hands_in_frame"):
        _resolve_eva(root, key_map=key_map, require_hands_in_frame="right")
