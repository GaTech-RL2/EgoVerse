"""GT / prediction replay video (``egomimic/eval/replay_viz.py``)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.eval.replay_viz import ReplayClip, render_replay, rigid_fit
from egomimic.rldb.embodiment.human import Human


def _clip(n, horizon, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.normal(size=(n, horizon, 126)) * 0.05 + np.tile([0.0, 0.0, 0.5], 42)
    gt = gt.astype(np.float32)
    return ReplayClip(
        images=np.zeros((n, 90, 160, 3), np.uint8),
        gt=gt,
        pred=gt + 0.01,
        intrinsics=[None] * n,
    )


def test_rigid_fit_recovers_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(42, 3))
    R0 = Rotation.random(random_state=1).as_matrix()
    t0 = np.array([0.1, -0.2, 0.3])
    R, t = rigid_fit(src, src @ R0.T + t0)
    np.testing.assert_allclose(R, R0, atol=1e-9)
    np.testing.assert_allclose(t, t0, atol=1e-9)


def test_rigid_fit_ignores_invalid_points_and_degenerates_to_identity():
    src = np.full((42, 3), np.nan)
    R, t = rigid_fit(src, src)
    np.testing.assert_array_equal(R, np.eye(3))
    np.testing.assert_array_equal(t, np.zeros(3))


def test_render_replays_each_complete_chunk_twice():
    horizon = 10
    frames, used = render_replay(_clip(25, horizon), Human, mode="keypoints")
    # Two complete chunks (anchors 0 and 10); frames 20-24 wait for the next clip.
    assert used == 2 * horizon
    assert frames.shape == (2 * 2 * horizon, 90, 160, 3)
    assert frames.dtype == np.uint8


def test_render_with_stride_spans_stride_frames_per_step():
    horizon, stride = 5, 3
    frames, used = render_replay(
        _clip(16, horizon), Human, mode="keypoints", stride=stride
    )
    assert used == horizon * stride
    assert len(frames) == 2 * horizon * stride


def test_short_clip_renders_nothing_and_consumes_nothing():
    frames, used = render_replay(_clip(9, 10), Human, mode="keypoints")
    assert used == 0
    assert len(frames) == 0


def test_tail_and_concat_round_trip():
    clip = _clip(12, 4)
    joined = ReplayClip.concat([clip.tail(8), clip.tail(10)])
    assert len(joined) == 6
    np.testing.assert_array_equal(
        joined.gt, np.concatenate([clip.gt[8:], clip.gt[10:]])
    )
    assert joined.intrinsics == clip.intrinsics[8:] + clip.intrinsics[10:]
