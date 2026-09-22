"""``ZarrDataset.__getitem__`` reads each zarr key once, not once per keymap entry.

The mecka keypoint keymap with ``proprio_history=4`` names ``left/right.obs_keypoints``
and ``left/right.obs_wrist_pose`` twice each -- once as a forward action chunk
``(i, i+H)``, once as a backward proprio window ``(i-K+1, i+1)``. Those were ten
separate ``ZarrEpisode.read`` calls over six distinct keys; they are now one zarr call
per key (``ZarrEpisode.read_intervals``), sliced twice. Nothing the dataset emits may change.

The episode here is RAMPED -- every cell is ``frame * 1000 + column`` -- so a slice
taken at the wrong offset into the coalesced span produces different numbers rather
than the same ones. ``tests/fixtures/synthetic_episodes.py`` deliberately holds each
numeric key constant in time (for the norm-stat quantile bounds), which would make an
off-by-one here invisible: the single most likely bug in this change.
"""

from __future__ import annotations

import numpy as np
import pytest
import simplejpeg
import zarr

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import (
    ZarrAnnotationCutoffDataset,
    ZarrDataset,
    ZarrEpisode,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

T = 160
HORIZON = Human.ACTION_HORIZON
# half-open [start, end); ZarrAnnotationCutoffDataset clamps action chunks at the
# end of the span holding the start frame, so idx 60 and 130 both hit a cutoff
ANN_SPANS = ((0, 80), (80, T - 1))

# (label, keymap kwargs). Covers both action spaces, history on and off, and the
# ee_pose variant that gives four entries over two shared keys.
KEYMAPS = {
    "keypoints_k4": {"keymap_mode": "keypoints", "proprio_history": 4},
    "keypoints_k1": {"keymap_mode": "keypoints", "proprio_history": 1},
    "keypoints_k4_ee": {
        "keymap_mode": "keypoints",
        "proprio_history": 4,
        "include_ee_pose": True,
    },
    "cartesian_k4": {"keymap_mode": "cartesian", "proprio_history": 4},
    "cartesian_k1": {"keymap_mode": "cartesian", "proprio_history": 1},
}


def _keymap(label):
    return Human.get_keymap(
        has_head_pose=True, annotation_key="annotations", **KEYMAPS[label]
    )


def _ramp(width: int) -> np.ndarray:
    """``[t, d] = t * 1000 + d`` -- every cell identifies its own frame."""
    return (
        np.arange(T, dtype=np.float64)[:, None] * 1000.0
        + np.arange(width, dtype=np.float64)[None, :]
    )


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    numeric = {"obs_head_pose": _ramp(7)}
    for side in ("left", "right"):
        # distinct per side so a left/right mix-up is caught too
        off = 0.0 if side == "left" else 0.5
        numeric[f"{side}.obs_wrist_pose"] = _ramp(7) + off
        numeric[f"{side}.obs_keypoints"] = _ramp(63) + off
        numeric[f"{side}.obs_ee_pose"] = _ramp(7) + off
    rng = np.random.default_rng(0)
    return ZarrWriter.create_and_write(
        tmp_path_factory.mktemp("ramped") / "ramped.zarr",
        numeric_data=numeric,
        image_data={"images.front_1": rng.integers(0, 255, (T, 32, 32, 3), np.uint8)},
        embodiment="human_bimanual",
        task_name="ramp",
        annotations=[(t, a, b) for t, (a, b) in zip(("fold", "put down"), ANN_SPANS)],
        intrinsics={"front_1": np.eye(3, 4)},
    )


def _ann_end(idx: int) -> int | None:
    return next((e for s, e in ANN_SPANS if s <= idx < e), None)


def _expected_window(spec, idx, total, cutoff: bool):
    """The read window for one keymap entry, written out independently of the
    dataset: the spec this change must not move."""
    horizon, history = spec.get("horizon"), spec.get("history")
    if horizon is not None:
        end = min(idx + horizon, total)
        if cutoff and spec.get("key_type") == "action_keys":
            end = min(end, _ann_end(idx) or end)
        return (idx, end)
    if history is not None and int(history) > 1:
        return (max(0, idx - int(history) + 1), idx + 1)
    return (idx, None)


def _oracle(episode, key_map, idx, cutoff=False):
    """One zarr slice per keymap entry -- the pre-coalescing read path."""
    store = zarr.open_group(str(episode), mode="r")
    out = {}
    for k, spec in key_map.items():
        if spec.get("key_type") == "annotation_keys":
            continue
        start, end = _expected_window(spec, idx, T, cutoff)
        arr = store[spec["zarr_key"]]
        chunk = arr[start:end] if end is not None else arr[start : start + 1][0]
        horizon, history = spec.get("horizon"), spec.get("history")
        if horizon is not None and chunk.shape[0] < horizon:
            pad = np.repeat(chunk[-1:], horizon - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        if horizon is None and history is not None and int(history) > 1:
            if chunk.shape[0] < int(history):
                pad = np.repeat(chunk[:1], int(history) - chunk.shape[0], axis=0)
                chunk = np.concatenate([pad, chunk], axis=0)
        out[k] = chunk
    return out


# 0..3 clamp the backward window, 60 is interior, the rest truncate the forward chunk
@pytest.mark.parametrize("idx", [0, 1, 3, 60, 79, 130, T - HORIZON, T - 3, T - 1])
@pytest.mark.parametrize("label", sorted(KEYMAPS))
@pytest.mark.parametrize("cls", [ZarrDataset, ZarrAnnotationCutoffDataset])
def test_getitem_matches_the_per_entry_read_path(episode, idx, label, cls):
    key_map = _keymap(label)
    sample = cls(episode, key_map=key_map, transform_list=None)[idx]
    expected = _oracle(episode, key_map, idx, cutoff=cls is ZarrAnnotationCutoffDataset)

    # __getitem__ emits float32; compare bit-exactly in the dtype it emits
    image_keys = {k for k, s in key_map.items() if s.get("key_type") == "camera_keys"}
    for k, want in expected.items():
        if k in image_keys:
            want = simplejpeg.decode_jpeg(want, colorspace="RGB")
            want = np.transpose(want, (2, 0, 1)) / 255.0
        got = np.asarray(sample[k])
        assert got.shape == want.shape, k
        np.testing.assert_array_equal(got, want.astype(np.float32), err_msg=k)


def test_shapes_are_the_keymap_contract(episode):
    sample = ZarrDataset(episode, key_map=_keymap("keypoints_k4"))[60]
    assert sample["left.action_keypoints"].shape == (HORIZON, 63)
    assert sample["left.action_wrist_pose"].shape == (HORIZON, 7)
    assert sample["left.obs_keypoints"].shape == (4, 63)
    assert sample["left.obs_wrist_pose"].shape == (4, 7)
    assert sample["obs_head_pose"].shape == (7,)


def _count_reads(monkeypatch) -> list[dict]:
    """Per ``read_intervals`` call, ``{key: [(group_start, rows read), ...]}``."""
    calls: list[dict] = []
    inner = ZarrEpisode.read_intervals

    def spy(self, intervals):
        out = inner(self, intervals)
        calls.append({k: [(lo, len(rows)) for lo, rows in g] for k, g in out.items()})
        return out

    # ZarrEpisode has __slots__, so the counter goes on the class
    monkeypatch.setattr(ZarrEpisode, "read_intervals", spy)
    return calls


def test_one_read_per_distinct_zarr_key(episode, monkeypatch):
    key_map = _keymap("keypoints_k4")
    calls = _count_reads(monkeypatch)

    ZarrDataset(episode, key_map=key_map, transform_list=None)[60]

    assert len(calls) == 1, "one read call per __getitem__"
    read_keys = set(calls[0])
    entries = [s for s in key_map.values() if s.get("key_type") != "annotation_keys"]
    assert read_keys == {s["zarr_key"] for s in entries}
    assert len(read_keys) == 6 < len(entries) == 10, "the four duplicates collapsed"


def test_rows_read_are_the_union_of_the_windows(episode, monkeypatch):
    """Asserts rows READ, which the value oracle cannot see: a loader that read
    the whole episode and sliced it would still match every value."""
    calls = _count_reads(monkeypatch)
    ZarrDataset(episode, key_map=_keymap("keypoints_k4"), transform_list=None)[60]
    rows = calls[0]
    assert rows["left.obs_keypoints"] == [
        (57, 3 + HORIZON)
    ], "history + chunk, one group"
    assert rows["obs_head_pose"] == [(60, 1)], "a key read once is not widened"
    assert rows["images.front_1"] == [(60, 1)]


@pytest.mark.parametrize(
    "intervals, chunk, groups",
    [
        # numeric keys: 100-row chunks
        ([(57, 61), (60, 160)], 100, [(57, 160)]),
        ([(0, 1), (150, 151)], 100, [(0, 151)]),  # adjacent chunks: no new fetch
        ([(0, 1), (250, 251)], 100, [(0, 1), (250, 251)]),  # chunk 1 untouched
        ([(99, 100), (100, 101)], 100, [(99, 101)]),  # straddles a boundary
        # JPEG keys: one frame per chunk
        ([(60, 61), (57, 58)], 1, [(57, 58), (60, 61)]),
        ([(59, 60), (60, 61)], 1, [(59, 61)]),
        ([(10, 20), (15, 25)], 1, [(10, 25)]),
        ([(0, 1), (0, 1)], 1, [(0, 1)]),
    ],
)
def test_plan_reads_bridges_only_chunks_already_fetched(intervals, chunk, groups):
    assert ZarrEpisode.plan_reads(intervals, chunk) == groups


def test_far_apart_jpeg_frames_are_two_rows_not_the_gap(episode):
    reader = ZarrEpisode(episode)
    store = zarr.open_group(str(episode), mode="r")
    got = reader.read_intervals({"images.front_1": [(100, 101), (10, 11)]})
    assert [(lo, len(rows)) for lo, rows in got["images.front_1"]] == [
        (10, 1),
        (100, 1),
    ]
    for lo, rows in got["images.front_1"]:
        assert rows[0] == store["images.front_1"][lo : lo + 1][0]


def test_union_does_not_depend_on_keymap_order(episode):
    """The real keymaps happen to list the forward chunk before the backward
    window, which would hide a union that just took the last entry's bounds."""
    pose = "left.obs_wrist_pose"
    order = {
        "proprio": {"key_type": "proprio_keys", "zarr_key": pose, "history": 4},
        "action": {"key_type": "action_keys", "zarr_key": pose, "horizon": 10},
    }
    for key_map in (order, dict(reversed(order.items()))):
        ds = ZarrDataset(episode, key_map=key_map, transform_list=None)
        starts = [w[1][0] for w in ds._read_windows(60).values()]
        assert min(starts) == 57 and max(starts) == 60
        sample = ds[60]
        np.testing.assert_array_equal(
            np.asarray(sample["proprio"]), _ramp(7)[57:61].astype(np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(sample["action"]), _ramp(7)[60:70].astype(np.float32)
        )


def test_shared_span_slices_do_not_alias(episode):
    """Two entries sliced out of one span must own their arrays: a transform is
    handed the raw ndarrays and is free to write into them. Asserted at that
    point, because __getitem__'s closing float64 -> float32 cast would sever the
    aliasing before anything downstream could see it."""
    seen = {}

    class _Spy:
        def transform(self, data):
            seen["shares"] = np.shares_memory(
                data["left.action_keypoints"], data["left.obs_keypoints"]
            )
            data["left.action_keypoints"][0] += 1.0  # a transform writing in place
            seen["proprio_last"] = data["left.obs_keypoints"][-1].copy()
            return data

    ds = ZarrDataset(episode, key_map=_keymap("keypoints_k4"), transform_list=[_Spy()])
    ds[60]
    assert seen["shares"] is False, "the two windows must not share a buffer"
    # frame 60 of the ramp, unperturbed by the write to the action chunk
    np.testing.assert_array_equal(seen["proprio_last"], _ramp(63)[60])


def test_single_frame_entry_is_sliced_at_the_span_offset(episode):
    """A key read as a single frame AND as a window starting earlier: the
    single-frame slice needs the span offset like any other. No shipped keymap
    mixes the two on one key (the history keys are exactly the obs entries), so
    nothing else would catch a `block[0]` here."""
    pose = "left.obs_wrist_pose"
    key_map = {
        "history": {"key_type": "proprio_keys", "zarr_key": pose, "history": 4},
        "current": {"key_type": "proprio_keys", "zarr_key": pose},
    }
    sample = ZarrDataset(episode, key_map=key_map, transform_list=None)[60]
    np.testing.assert_array_equal(
        np.asarray(sample["current"]), _ramp(7)[60].astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(sample["history"]), _ramp(7)[57:61].astype(np.float32)
    )


def test_a_jpeg_failure_rereads_at_the_new_index(episode, monkeypatch):
    """The decode retry picks a different frame and restarts the loop, so the
    windows, spans and one read all have to be recomputed against the new idx --
    which the coalescing moved."""
    real = simplejpeg.decode_jpeg
    calls: list[int] = []

    def _fail_once(data, **kw):
        calls.append(len(data))
        if len(calls) == 1:
            raise ValueError("corrupt jpeg")
        return real(data, **kw)

    monkeypatch.setattr(
        "egomimic.rldb.zarr.zarr_dataset_multi.simplejpeg.decode_jpeg", _fail_once
    )
    ds = ZarrDataset(episode, key_map=_keymap("keypoints_k4"), transform_list=None)
    sample = ds[60]

    assert len(calls) == 2, "one failure, then the retry's decode"
    # every numeric key comes from the RETRY's frame, not frame 60
    frame = int(round(float(np.asarray(sample["obs_head_pose"])[0]) / 1000.0))
    assert frame != 60, "the retry moved to another frame"
    np.testing.assert_array_equal(
        np.asarray(sample["obs_head_pose"]), _ramp(7)[frame].astype(np.float32)
    )
    np.testing.assert_array_equal(
        np.asarray(sample["left.obs_keypoints"]),
        _ramp(63)[max(0, frame - 3) : frame + 1].astype(np.float32),
    )


def test_an_array_shorter_than_its_window_fails_loudly(tmp_path):
    """A StopIteration out of __getitem__ would end the epoch silently."""
    numeric = {"obs_head_pose": _ramp(7)}
    for side in ("left", "right"):
        numeric[f"{side}.obs_wrist_pose"] = _ramp(7)
        numeric[f"{side}.obs_keypoints"] = _ramp(63)
        numeric[f"{side}.obs_ee_pose"] = _ramp(7)
    path = ZarrWriter.create_and_write(
        tmp_path / "short.zarr",
        numeric_data=numeric,
        image_data={"images.front_1": np.zeros((T, 32, 32, 3), np.uint8)},
        embodiment="human_bimanual",
        task_name="ramp",
        annotations=[("fold", 0, T - 1)],
        intrinsics={"front_1": np.eye(3, 4)},
    )
    zarr.open_group(str(path), mode="r+")["left.obs_keypoints"].resize((130, 63))
    ds = ZarrDataset(path, key_map=_keymap("keypoints_k4"), transform_list=None)
    with pytest.raises(ValueError, match="left.obs_keypoints"):
        ds[115]
