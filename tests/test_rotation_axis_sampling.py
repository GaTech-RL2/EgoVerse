import numpy as np

from egomimic.utils.viz_utils import _rotation_axis_sample_indices


def test_five_rotation_axes_cover_a_100_action_chunk():
    np.testing.assert_array_equal(
        _rotation_axis_sample_indices(100, 5),
        np.array([0, 25, 50, 74, 99]),
    )


def test_rotation_axis_sampling_handles_short_and_empty_chunks():
    np.testing.assert_array_equal(
        _rotation_axis_sample_indices(3, 5), np.array([0, 1, 2])
    )
    assert _rotation_axis_sample_indices(0, 5).size == 0
    assert _rotation_axis_sample_indices(100, 0).size == 0
