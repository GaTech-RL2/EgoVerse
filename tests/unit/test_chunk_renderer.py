"""The horizon trace is the only keypoint renderer, and it draws every frame.

``Human.viz(mode="keypoints")`` has exactly one path: ``_viz_keypoints_horizon_trace``.
The static-skeleton renderer it used to fall back to is gone on purpose -- a
chunk that reached it rendered as a motionless overlay, which is what a
whole 50k-step campaign's validation videos turned out to be (2026-09-18).
"""

import numpy as np
import pytest

from egomimic.utils.viz_utils import _viz_keypoints_horizon_trace

# px = 100 * x / z + 100, py = 100 * y / z + 100 on a 200x200 canvas.
K = np.array([[100, 0, 100, 0], [0, 100, 100, 0], [0, 0, 1, 0]])

BEHIND_CAMERA = [0.0, 0.0, -1.0]

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _trace(data, **kwargs):
    return _viz_keypoints_horizon_trace(
        np.zeros((200, 200, 3), np.uint8),
        data,
        K,
        colors={name: (20, 255, 20) for name in FINGERS},
        **kwargs,
    )


def _sweep(length):
    """A left hand whose slot 0 sweeps across the image, one x per frame.

    Slot 1 sits 0.4 above it; every other slot and the whole right hand stay
    behind the camera. Returns a (T, 126) chunk.
    """
    points = np.tile(np.array(BEHIND_CAMERA), (length, 2, 21, 1))
    xs = np.linspace(-0.7, 0.7, length)
    points[:, 0, 0] = np.column_stack((xs, np.zeros(length), np.ones(length)))
    points[:, 0, 1] = points[:, 0, 0] + [0.0, 0.4, 0.0]
    return points.reshape(length, -1)


def test_horizon_trace_connects_every_step_of_the_chunk():
    """The whole sweep is a polyline, not just its endpoints."""
    image = _trace(_sweep(30))
    for x in np.linspace(-0.7, 0.7, 30):
        assert image[100, round(100 + 100 * x)].any()
    # the line fills the gaps between sampled steps too
    assert image[100, 100].any()


def test_horizon_trace_accepts_the_126d_revert_output_and_144d_rot6d():
    """The evaluator hands it 126-D; raw wrist-frame actions are 144-D."""
    assert _trace(_sweep(8)).any()  # 126-D, wrist falls back to MANO slot 0

    hand = np.concatenate([np.zeros(9), np.tile(BEHIND_CAMERA, 21)])
    chunk = np.tile(np.concatenate([hand, hand]), (8, 1))
    chunk[:, 0:3] = np.column_stack(
        (np.linspace(-0.5, 0.5, 8), np.zeros(8), np.ones(8))
    )
    assert chunk.shape[-1] == 144
    assert _trace(chunk).any()  # wrist xyz block drives the trace


def test_a_lone_timestep_matches_its_own_flattened_form():
    """``(1, D)`` and ``(D,)`` are the same single-step trace, not two paths."""
    chunk = _sweep(1)
    np.testing.assert_array_equal(_trace(chunk), _trace(chunk.reshape(-1)))


def test_non_finite_slots_are_skipped():
    """A NaN or absurd slot drops out of the trace; the finite ones still draw."""
    chunk = _sweep(3)
    chunk[1, 0:3] = np.nan  # frame 1's slot 0
    chunk[2, 0:3] = 1e12
    image = _trace(chunk)
    # _sweep puts frame t's slot 0 at px = 100 * x + 100, i.e. 30 / 100 / 170.
    assert image[100, 30].any()  # frame 0 still drawn
    assert not image[100, 100].any()  # frame 1 was NaN
    assert not image[100, 170].any()  # frame 2 was 1e12


def test_horizon_trace_rejects_a_bad_shape_and_an_unknown_width():
    with pytest.raises(ValueError, match="keypoint trace expects"):
        _trace(np.zeros((2, 30, 126)))
    with pytest.raises(ValueError, match="keypoint trace expects"):
        _trace(np.zeros((4, 127)))


def test_human_viz_has_no_renderer_but_the_trace(monkeypatch):
    """Every keypoint input shape routes to the trace -- there is no fallback."""
    import egomimic.rldb.embodiment.human as human_mod
    from egomimic.rldb.embodiment.human import Human

    assert not hasattr(human_mod, "_viz_keypoints"), (
        "the static skeleton renderer is back; mode='keypoints' must have "
        "exactly one path"
    )

    seen = []

    def _record(**kw):
        seen.append(kw["actions"].shape)
        return np.zeros((2, 2, 3), np.uint8)

    monkeypatch.setattr(human_mod, "_viz_keypoints_horizon_trace", _record)
    image = np.zeros((200, 200, 3), np.uint8)
    Human.viz(image, _sweep(8), mode="keypoints", intrinsics=K)
    Human.viz(image, _sweep(1), mode="keypoints", intrinsics=K)
    Human.viz(image, _sweep(1).reshape(-1), mode="keypoints", intrinsics=K)

    assert seen == [(8, 126), (1, 126), (126,)]
