"""GT / prediction replay video (``egomimic/eval/replay_viz.py``)."""

from __future__ import annotations

import types

import numpy as np
from scipy.spatial.transform import Rotation

from egomimic.eval.eval_video import EvalVideo
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


def _resampled_clip(stride, n=60, horizon=100, seed=0):
    """GT chunks of a hand moving continuously in the world, seen by a
    rotating camera, so a chunk step can fall between frames."""
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


def test_replay_stops_after_its_chunk_budget():
    """A pinned video loader replays REPLAY_CHUNKS chunks of horizon * stride
    frames each, then skips the forward pass."""
    forwards = []
    ev = _Buffering(lambda: forwards.append(1) or _clip(4, horizon=10))
    ev.action_stride = {"human_bimanual": 0.5}  # 5 frames a chunk, 25 in all
    ev.trainer = types.SimpleNamespace(
        current_epoch=0,
        max_epochs=1,
        lightning_module=types.SimpleNamespace(device="cpu"),
    )
    for i in range(10):
        ev.on_validation_step({}, i, mode="video")
    assert len(forwards) == 7  # ceil(25 / 4 frames a batch)


def test_action_stride_is_derived_from_the_data_pipeline():
    """Raw horizon frames, every stride-th kept, resampled to the chunk."""
    from egomimic.rldb.embodiment.eva import Eva
    from egomimic.trainHydra import _action_strides

    def leaf(cls, keymap_mode, mode, **kw):
        return types.SimpleNamespace(
            key_map=cls.get_keymap(keymap_mode=keymap_mode),
            transform=cls.get_transform_list(mode=mode, **kw),
        )

    strides = _action_strides(
        {
            "mecka": leaf(Human, "cartesian", "cartesian_wristframe_6d", stride=1),
            "mecka_kp": leaf(Human, "keypoints", "keypoints_wristframe_6d", stride=1),
            "aria": leaf(Human, "cartesian", "cartesian_wristframe_6d", stride=3),
            "eva": leaf(Eva, "cartesian", "cartesian_wristframe_6d"),
            "raw": types.SimpleNamespace(key_map={}, transform=None),
        }
    )
    # 30 frames -> 100 steps; aria keeps frames 0, 3, .., 27; eva reads 45.
    assert strides == {
        "mecka": 29 / 99,
        "mecka_kp": 29 / 99,
        "aria": 27 / 99,
        "eva": 44 / 99,
    }
