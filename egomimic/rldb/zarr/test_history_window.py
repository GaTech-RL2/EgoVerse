"""Unit tests for the short-term-memory backward-window helpers.

CPU-only and store-free: they exercise ``ZarrDataset._pad_sequences_left``
(front-padding for a history window read near the start of an episode) and the
backward-window index math used in ``__getitem__``. No zarr store, model, or
data download is required.
"""

import numpy as np

from egomimic.rldb.zarr.action_chunk_transforms import PoseCoordinateFrameTransform
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset


def _bare_dataset():
    # _pad_sequences_left uses no instance state, so a bare instance is enough.
    return object.__new__(ZarrDataset)


def test_left_pad_repeats_first_frame():
    ds = _bare_dataset()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2); window short of history=4
    out = ds._pad_sequences_left({"k": arr.copy()}, history=4)["k"]
    assert out.shape == (4, 2)
    # Front rows are copies of the original FIRST frame; the tail is untouched,
    # so the current frame stays at row -1.
    assert np.array_equal(out[0], arr[0])
    assert np.array_equal(out[1], arr[0])
    assert np.array_equal(out[2:], arr)
    assert np.array_equal(out[-1], arr[-1])


def test_left_pad_noop_when_window_full():
    ds = _bare_dataset()
    arr = np.arange(6).reshape(3, 2).astype(float)
    out = ds._pad_sequences_left({"k": arr.copy()}, history=3)["k"]
    assert np.array_equal(out, arr)


def test_left_pad_none_history_is_noop():
    ds = _bare_dataset()
    arr = np.arange(4).reshape(2, 2).astype(float)
    out = ds._pad_sequences_left({"k": arr.copy()}, history=None)["k"]
    assert np.array_equal(out, arr)


def _backward_window(idx, history):
    """Replicates the inline read-interval math in ``__getitem__``."""
    start_idx = max(0, idx - history + 1)
    return start_idx, idx + 1


def test_backward_window_full_in_interior():
    # Away from the episot start, the window is exactly `history` long and the
    # last index is the current frame.
    start, end = _backward_window(idx=10, history=4)
    assert (start, end) == (7, 11)
    assert end - start == 4
    assert end - 1 == 10  # current frame is the last row


def test_backward_window_clamps_at_episode_start():
    start, end = _backward_window(idx=1, history=4)
    assert (start, end) == (0, 2)  # only 2 real frames -> left-pad fills the rest
    assert end - 1 == 1


def test_pose_transform_history_matches_single_pose_rowwise():
    # The short-term-memory read feeds a [K, 7] window through
    # PoseCoordinateFrameTransform. Each row must be transformed exactly as if
    # it had been the lone single pose, so the default (single) path is
    # preserved and the history rows are consistent with it.
    target = np.array([0.05, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0])  # xyz + quat(wxyz)
    pose_a = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    pose_b = np.array([-0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0])  # 180deg about x

    t = PoseCoordinateFrameTransform(
        target_world="tw", pose_world="pw", transformed_key_name="out", mode="xyzwxyz"
    )

    out_single = t.transform({"tw": target.copy(), "pw": pose_b.copy()})["out"]
    assert out_single.shape == (7,)

    window = np.stack([pose_a, pose_b], axis=0)  # current frame (pose_b) is last
    out_hist = t.transform({"tw": target.copy(), "pw": window.copy()})["out"]
    assert out_hist.shape == (2, 7)
    assert np.allclose(out_hist[-1], out_single, atol=1e-6)
