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
    "decode_jpeg_window_strided",
    "decode_video_span",
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


def decode_jpeg_window_strided(buffers, stride: int) -> np.ndarray:
    """Decode ONLY every ``stride``-th frame of a JPEG-buffer array.

    FREE decode-waste elimination for STRIDED packed training: the model only
    consumes every ``stride``-th frame per span (``TargetBuilder`` keeps
    ``kept = arange(0, seq_len, stride)`` per episode via ``strided_view`` and
    decimates every ``obs/*`` key with ``v[kept]``). The other frames are
    decoded and then thrown away. This decodes exactly the kept frames and
    leaves the discarded positions as cheap zero placeholders, so:

      * the returned tensor keeps the SAME ``(T, C, H, W)`` full-rate shape and
        dtype as :func:`decode_jpeg_window` (so ``pack_collate`` /
        ``TargetBuilder`` need no change and per-frame alignment is untouched),
      * the KEPT frames are byte-identical to :func:`decode_jpeg_window`
        (same simplejpeg decode + transpose + ``/255``), and
      * decode cost drops ~``stride``x.

    ``stride <= 1`` routes to :func:`decode_jpeg_window` verbatim (OFF = no-op).

    NOTE: ``stride`` MUST equal the pipeline ``TargetBuilder.stride`` — the
    kept-frame set here (``range(0, T, stride)`` per span) is exactly what
    ``strided_view`` selects at each ``cu_seqlens`` boundary. The equivalence
    gate (tests/scratch) proves this for the fold ``stride=4`` configs.
    """
    if int(stride) <= 1:
        return decode_jpeg_window(buffers)
    stride = int(stride)
    T = len(buffers)
    out = None
    for i in range(0, T, stride):
        decoded = simplejpeg.decode_jpeg(buffers[i], colorspace="RGB")
        frame = np.transpose(decoded, (2, 0, 1)) / 255.0
        if out is None:
            out = np.zeros((T,) + frame.shape, dtype=frame.dtype)
        out[i] = frame
    return out


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


def decode_video_span(store_array, video_meta: dict, start: int, end: int,
                      stride: int = 1):
    """Decode frames ``[start, end)`` from a CHUNKED h264 image array.

    The array holds one mp4 per ``frames_per_chunk`` frames, so frame f lives in
    chunk ``f // fpc`` at offset ``f % fpc``. Decode only the covering chunks,
    concatenate, then slice -- reading the whole episode per span would pull
    ~10x more than needed.

    Returns ``(end-start, 3, H, W)`` float in [0,1] -- byte-compatible with
    :func:`decode_jpeg_window`, so pack_collate / TargetBuilder need no change.

    ``stride>1`` mirrors decode_jpeg_window_strided: full-rate shape, with the
    discarded positions left as zeros.
    """
    import numpy as np
    from egomimic.rldb.zarr.video_codec import decode_chunk

    fpc = int(video_meta.get("frames_per_chunk") or 0)
    if fpc <= 0:
        raise ValueError(f"video_meta missing frames_per_chunk: {video_meta!r}")
    c0, c1 = start // fpc, (end - 1) // fpc + 1
    frames = []
    for ci in range(c0, c1):
        frames.append(decode_chunk(store_array[ci]))
    block = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), np.uint8)
    lo = start - c0 * fpc
    win = block[lo : lo + (end - start)]
    if len(win) != end - start:
        raise ValueError(
            f"video span [{start},{end}) resolved {len(win)} frames from chunks "
            f"[{c0},{c1}) (fpc={fpc}) -- chunk/frame arithmetic mismatch"
        )
    out = np.transpose(win, (0, 3, 1, 2)).astype(np.float64) / 255.0
    if stride and stride > 1:
        keep = np.zeros(len(out), dtype=bool)
        keep[::stride] = True
        out[~keep] = 0.0
    return out
