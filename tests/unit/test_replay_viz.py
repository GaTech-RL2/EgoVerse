"""GT / prediction replay video (``egomimic/eval/replay_viz.py``)."""

from __future__ import annotations

import types

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.eval.eval_video import EvalVideo
from egomimic.eval.replay_viz import ReplayClip, infer_stride, render_replay, rigid_fit
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
    frames, used = render_replay(_clip(25, horizon), Human, mode="keypoints", stride=1)
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
    frames, used = render_replay(_clip(9, 10), Human, mode="keypoints", stride=1)
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


def _moving_camera_clip(stride, n=60, horizon=12, seed=0):
    """GT chunks of a hand moving in the world, seen by a rotating camera."""
    rng = np.random.default_rng(seed)
    steps = n + horizon * stride
    world = np.cumsum(rng.normal(scale=0.01, size=(steps, 42, 3)), axis=0)
    world += rng.normal(scale=0.05, size=(42, 3)) + [0.0, 0.0, 0.5]
    cams = Rotation.from_rotvec(np.cumsum(rng.normal(scale=0.02, size=(steps, 3)), 0))
    gt = np.stack(
        [
            np.stack(
                [
                    cams[t].inv().apply(world[t + k * stride]).reshape(-1)
                    for k in range(horizon)
                ]
            )
            for t in range(n)
        ]
    ).astype(np.float32)
    return ReplayClip(np.zeros((n, 90, 160, 3), np.uint8), gt, gt, [None] * n)


def test_infer_stride_recovers_the_frames_per_step():
    for stride in (1, 2, 3):
        assert infer_stride(_moving_camera_clip(stride)) == stride


def _resampled_clip(stride, n=60, horizon=100, seed=0):
    """As _moving_camera_clip, but with a hand and camera that move
    continuously, so a chunk step can fall between frames."""
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.05, size=(42, 3)) + [0.0, 0.0, 0.5]
    amp, freq, phase = rng.normal(scale=0.05, size=(3, 42, 3))
    rot_amp = rng.normal(scale=0.2, size=3)

    def world(t):
        return base + amp * np.sin(0.2 * t * (1 + freq) + phase)

    def cam(t):
        return Rotation.from_rotvec(rot_amp * np.sin(0.05 * t))

    gt = np.stack(
        [
            np.stack(
                [
                    cam(t).inv().apply(world(t + k * stride)).reshape(-1)
                    for k in range(horizon)
                ]
            )
            for t in range(n)
        ]
    ).astype(np.float32)
    return ReplayClip(np.zeros((n, 90, 160, 3), np.uint8), gt, gt, [None] * n)


def test_infer_stride_recovers_a_resampled_chunk():
    # 30 raw frames resampled to 100 steps (mecka, action stride 1).
    assert infer_stride(_resampled_clip(29 / 99)) == 29 / 99


def test_render_with_fractional_stride_plays_in_real_time():
    stride, horizon = 29 / 99, 100
    frames, used = render_replay(
        _resampled_clip(stride, n=70, horizon=horizon),
        Human,
        mode="keypoints",
        stride=stride,
    )
    # Anchors 0 and 30: each chunk covers frames a..a+29, once as GT, once as pred.
    assert used == 60
    assert len(frames) == 2 * 2 * 30


def test_infer_stride_needs_keypoints():
    clip = _clip(20, 10)
    clip.gt = clip.gt[..., :12]
    assert infer_stride(clip) is None


class _Buffering(EvalVideo):
    def __init__(self, images_fn):
        super().__init__(viz_max_batches=100)
        self._images_fn = images_fn

    def compute_metrics_and_viz(self, batch, do_viz=True):
        return {}, ({3: self._images_fn()} if do_viz else {})


def test_overlay_buffer_counts_frames_not_pixel_rows(monkeypatch):
    written = []
    monkeypatch.setattr(
        "egomimic.eval.eval_video.tvio.write_video",
        lambda path, frames, **kw: written.append(len(frames)),
    )
    ev = _Buffering(lambda: np.zeros((8, 360, 640, 3), np.uint8))
    ev.trainer = types.SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        max_epochs=1,
        global_step=0,
        loggers=[],
        default_root_dir="/tmp/unused",
        lightning_module=types.SimpleNamespace(
            device="cpu", log_dict=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(ev, "_upload_video", lambda key, path: None)
    monkeypatch.setattr("os.makedirs", lambda *a, **k: None)
    for i in range(10):
        ev.on_validation_step({}, i, mode="both")
    ev.on_validation_end()
    assert written == [80]
