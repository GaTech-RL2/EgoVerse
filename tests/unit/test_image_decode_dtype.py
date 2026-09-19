"""Decoded frames leave the dataset as float32.

`uint8 / 255.0` is float64 in numpy, which doubles what the collate, the
pinned-memory staging and the H2D copy move for every image in every batch --
and the model casts to float32 on arrival regardless.
"""

import numpy as np

from egomimic.rldb.zarr.utils import jpeg_pixels_to_float


def test_float32_division_is_bit_identical_to_float64():
    """The cheap dtype is also the same number: there are only 256 inputs, so
    this is exhaustive, not a sample."""
    u = np.arange(256, dtype=np.uint8)
    wide = (u / 255.0).astype(np.float32)
    narrow = u.astype(np.float32) / np.float32(255.0)
    assert np.array_equal(wide, narrow)


def test_jpeg_pixels_to_float():
    decoded = np.random.randint(0, 256, size=(8, 6, 3), dtype=np.uint8)
    out = jpeg_pixels_to_float(decoded)
    assert out.dtype == np.float32
    assert out.shape == (3, 8, 6)
    assert np.array_equal(
        out, (np.transpose(decoded, (2, 0, 1)) / 255.0).astype(np.float32)
    )
    window = np.stack([decoded, decoded])
    assert jpeg_pixels_to_float(window).shape == (2, 3, 8, 6)
