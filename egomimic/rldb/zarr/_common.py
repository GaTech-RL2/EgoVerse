"""Shared decoding helpers for EgoVerse Zarr datasets."""

from __future__ import annotations

import numpy as np
import simplejpeg

__all__ = ["decode_jpeg_single", "decode_jpeg_window"]


def _scale_image_float32(image: np.ndarray) -> np.ndarray:
    """Convert an HWC uint8 image to CHW float32 in [0, 1]."""
    chw = np.transpose(image, (2, 0, 1))
    return np.divide(chw, np.float32(255.0), dtype=np.float32)


def decode_jpeg_single(buffer) -> np.ndarray:
    """Decode one JPEG byte buffer."""
    decoded = simplejpeg.decode_jpeg(buffer, colorspace="RGB")
    return _scale_image_float32(decoded)


def decode_jpeg_window(buffers) -> np.ndarray:
    """Decode a Zarr object array of JPEG buffers to (T, C, H, W)."""
    return np.stack([decode_jpeg_single(buffer) for buffer in buffers], axis=0)
