import numpy as np

from scripts.dataset.resample_pushshapes_speed import (
    output_frame_count,
    recolor_red_pusher_blue,
    resample_actions,
)


def test_output_frame_count_preserves_endpoints() -> None:
    assert output_frame_count(10, 1.5) == 7
    assert output_frame_count(10, 0.5) == 19
    assert output_frame_count(1, 2.0) == 1


def test_speed_and_duration_scale_inversely() -> None:
    source_frames = 301
    for speed in (0.5, 1.5, 2.0):
        output_intervals = output_frame_count(source_frames, speed) - 1
        assert output_intervals * speed == source_frames - 1


def test_resample_actions_preserves_first_and_last() -> None:
    source = np.arange(20, dtype=np.float64).reshape(10, 2)
    for speed in (0.5, 1.0, 1.5, 2.0):
        result = resample_actions(source, speed)
        np.testing.assert_array_equal(result[0], source[0])
        np.testing.assert_array_equal(result[-1], source[-1])


def test_blue_recolor_only_changes_red_pixels() -> None:
    image = np.array([[[210, 60, 60], [60, 100, 200], [240, 240, 240]]], dtype=np.uint8)
    result = recolor_red_pusher_blue(image)
    np.testing.assert_array_equal(result[0, 0], [60, 60, 210])
    np.testing.assert_array_equal(result[0, 1:], image[0, 1:])
    np.testing.assert_array_equal(image[0, 0], [210, 60, 60])
