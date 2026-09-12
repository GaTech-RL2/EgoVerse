"""Multi-frame projections must draw every frame and respect slot ownership."""

import numpy as np
import pytest

from egomimic.utils.viz_utils import _viz_keypoints

K = np.array([[100, 0, 100, 0], [0, 100, 100, 0], [0, 0, 1, 0]])


def _draw(data, **kwargs):
    return _viz_keypoints(
        np.zeros((200, 200, 3), np.uint8),
        data,
        K,
        edges=[(0, 1)],
        colors={"finger": (20, 255, 20)},
        edge_ranges=[("finger", 0, 1)],
        n_kp=3,
        **kwargs,
    )


@pytest.mark.parametrize("length", [1, 2, 30])
def test_all_chunk_frames_draw_owned_slots_only(length):
    points = np.full((length, 2, 3, 3), 1e9)
    points[:, 0, 0] = np.column_stack(
        (np.linspace(-0.7, 0.7, length), np.zeros(length), np.ones(length))
    )
    points[:, 0, 1] = points[:, 0, 0] + [0, 0.4, 0]
    points[:, 1, 0] = [np.nan, np.nan, np.nan]
    image = _draw(points.reshape(length, -1), valid_slots=[0])
    for x in np.linspace(-0.7, 0.7, length):
        assert image[100, round(100 + 100 * x)].any()
    assert not image[125:150].any()  # masked dots and incident skeleton edges


def test_chunk_edges_never_connect_different_frames():
    points = np.full((2, 2, 3, 3), 1e9)
    points[:, 0, :2] = [[[-0.6, 0, 1], [-0.6, 0.3, 1]], [[0.6, 0, 1], [0.6, 0.3, 1]]]
    image = _draw(points.reshape(2, -1), valid_slots=[0, 1])
    assert image[115, 40].any() and image[115, 160].any()
    assert not image[110:120, 60:140].any()
    np.testing.assert_array_equal(
        _draw(points[0].reshape(-1)), _draw(points[:1].reshape(1, -1))
    )


def test_ambiguous_batch_and_wrong_width_are_rejected():
    with pytest.raises(ValueError, match="expects"):
        _draw(np.zeros((2, 30, 18)))
    with pytest.raises(ValueError, match="width"):
        _draw(np.zeros((1, 19)))
