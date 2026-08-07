"""Shared read/decode helpers for the zarr loader paths.

Both zarr loader paths — the padded/windowed reader
(``ZarrDataset.__getitem__``) and the packed/span reader
(``ZarrDataset._read_span``, consumed by ``ZarrEpisodePackedDataset``) — used
to inline byte-for-byte copies of the same JPEG/JSON decode, float32
tensorization, and embodiment-tagging logic. Collapse 3 factors that shared
logic here so there is ONE source of truth; each loader keeps only its genuine
differences (windowing+padding+resample loop vs. exact-span read + ``seq_len``
metadata).

Every function here is a pure transformation extracted verbatim from the
pre-collapse loader bodies — see ``tests/test_loader_equality.py`` for the
behavioral-equality proof (frozen reference hashes + cross-loader
``torch.equal``).
"""

from __future__ import annotations

import numpy as np
import simplejpeg
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id

__all__ = [
    "decode_jpeg_single",
    "decode_jpeg_window",
    "decode_json_array",
    "tensorize_float32",
    "tag_embodiment",
]


def decode_jpeg_single(buf) -> np.ndarray:
    """Decode one JPEG buffer to a CHW float image in ``[0, 1]``.

    Verbatim extraction of the single-frame decode shared by
    ``ZarrDataset.__getitem__`` (no-horizon image read) and
    ``ZarrActionExpertDataset._load_obs_at``.
    """
    decoded = simplejpeg.decode_jpeg(buf, colorspace="RGB")
    return np.transpose(decoded, (2, 0, 1)) / 255.0


def decode_jpeg_window(buffers) -> np.ndarray:
    """Decode an array of per-frame JPEG buffers to a stacked ``(T, C, H, W)``.

    simplejpeg can't vectorize across the buffer-array dtype, so each frame is
    decoded individually then stacked. Verbatim extraction of the windowed
    image decode shared by ``ZarrDataset.__getitem__`` (horizon > 1) and
    ``ZarrDataset._read_span``.
    """
    frames = []
    for buf in buffers:
        decoded = simplejpeg.decode_jpeg(buf, colorspace="RGB")
        frames.append(np.transpose(decoded, (2, 0, 1)) / 255.0)
    return np.stack(frames, axis=0)


def decode_json_array(arr, decode_entry) -> list:
    """Decode an array of JSON-encoded entries via ``decode_entry``.

    ``decode_entry`` is ``ZarrDataset._decode_json_entry`` (a staticmethod);
    passed in to avoid a circular import. Verbatim extraction of the
    list-comprehension shared by both loader paths.
    """
    return [decode_entry(v) for v in arr]


def tensorize_float32(data: dict, *, skip_object_dtype: bool) -> dict:
    """In-place convert every ndarray value in ``data`` to a float32 tensor.

    The two loaders differ by exactly one predicate:
      - ``_read_span`` skips object-dtype arrays (``skip_object_dtype=True``)
        because annotation lists can leave object arrays in the dict.
      - ``__getitem__`` converts every ndarray (``skip_object_dtype=False``).

    Both originals iterated and replaced in place; this preserves that.
    """
    if skip_object_dtype:
        for k, v in list(data.items()):
            if isinstance(v, np.ndarray) and v.dtype != object:
                data[k] = torch.from_numpy(v).to(torch.float32)
    else:
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)
    return data


def tag_embodiment(data: dict, embodiment) -> dict:
    """Stamp ``embodiment`` + ``metadata.robot_name`` with the embodiment id.

    Verbatim extraction of the two-line tag both loaders append at the end.
    """
    emb_id = get_embodiment_id(embodiment)
    data["embodiment"] = emb_id
    data["metadata.robot_name"] = emb_id
    return data
