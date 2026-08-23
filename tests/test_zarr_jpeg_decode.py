import numpy as np
import simplejpeg

from egomimic.rldb.zarr._common import decode_jpeg_single, decode_jpeg_window


def _jpeg_bytes() -> bytes:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[..., 0] = 64
    image[..., 1] = 128
    image[..., 2] = 192
    return simplejpeg.encode_jpeg(image, colorspace="RGB", quality=95)


def test_decode_jpeg_single_returns_chw_float32():
    decoded = decode_jpeg_single(_jpeg_bytes())

    assert decoded.shape == (3, 8, 10)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()
    assert 0.0 <= decoded.min() <= decoded.max() <= 1.0


def test_decode_jpeg_window_handles_zarr_object_arrays():
    buffer = _jpeg_bytes()
    buffers = np.empty(2, dtype=object)
    buffers[:] = [buffer, buffer]

    decoded = decode_jpeg_window(buffers)

    assert decoded.shape == (2, 3, 8, 10)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded[0], decoded[1])
