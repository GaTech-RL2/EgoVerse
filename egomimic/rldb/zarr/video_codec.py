"""H.264 image storage for EgoVerse zarr episodes.

WHY: per-frame JPEG codes every frame independently, so it cannot exploit the
temporal redundancy of 30fps video. Measured on real fold episodes:

    JPEG (q75, as shipped)   44.8 KB/frame   8.33 ms/frame decode
    h264 crf15               20.2 KB/frame   1.68 ms/frame decode
                             2.2x smaller    5.0x faster

At EQUAL bytes h264 also beats JPEG on PSNR (+2.4 to +2.9 dB in the 4-8 KB
range), so this is not a quality-for-size trade -- it is strictly better coding.

LAYOUT: one mp4 per GROUP of ``frames_per_chunk`` frames, stored as the elements
of a VariableLengthBytes array. NOT one blob per episode: a span read would then
have to pull the whole episode (~68 MB) to decode any window, and we read ~13
spans per episode. Frame f lives in chunk ``f // frames_per_chunk`` at offset
``f % frames_per_chunk``.

The reader learns everything it needs from ``_features[key]``::

    {"dtype": "h264", "shape": [H, W, 3], "video": {
        "codec", "crf", "gop", "pix_fmt", "fps", "frames_per_chunk",
        "total_frames", "n_chunks"}}

``dtype`` stays "jpeg" for JPEG episodes, so existing data and readers are
untouched. Readers should not trust it blindly though: :func:`resolve_encoding`
detects the real encoding from the stored bytes, so an episode with stale or
absent metadata still loads and the two encodings can coexist in one dataset.

Sections below, in order: format constants, write-side configuration, container
encode/decode, chunk grouping, read-side access, encoding detection, metadata.
"""
from __future__ import annotations

import io
import os

import numpy as np

# --------------------------------------------------------------------------
# Format constants
# --------------------------------------------------------------------------

DEFAULT_CODEC = "h264"
DEFAULT_CRF = 15          # decode speed is flat across CRF, so don't over-compress
DEFAULT_GOP = 30          # 1 s at 30 fps; bounds worst-case seek within a chunk
DEFAULT_PIX_FMT = "yuv420p"
DEFAULT_FRAMES_PER_CHUNK = 300   # 10 s; a multiple of GOP so chunks start on a keyframe

JPEG_DTYPE = "jpeg"

#: dtype strings the READER accepts as "this key is chunked video". Wider than
#: what the writer emits (always "h264") because episodes written before the
#: codec string was normalised can carry any of these on disk. NOT the set of
#: values EGOVERSE_IMAGE_CODEC accepts -- see :func:`image_codec_enabled`.
VIDEO_DTYPES = ("h264", "avc", "video")

_JPEG_SOI = b"\xff\xd8\xff"


# --------------------------------------------------------------------------
# Write-side configuration
# --------------------------------------------------------------------------

def codec_settings_from_env() -> dict:
    """Read encode settings from env so a run is reproducible from its launch.

    These names must be forwarded to Ray workers via runtime_env env_vars
    (run_conversion.py does); the encoding decision is made inside the worker,
    so an unforwarded var means the worker silently uses these defaults instead
    of what the driver asked for.
    """
    fpc = int(os.environ.get("EGOVERSE_VIDEO_FRAMES_PER_CHUNK", DEFAULT_FRAMES_PER_CHUNK))
    gop = int(os.environ.get("EGOVERSE_VIDEO_GOP", DEFAULT_GOP))
    if fpc % gop:
        raise ValueError(
            f"frames_per_chunk ({fpc}) must be a multiple of gop ({gop}) so every "
            f"chunk starts on a keyframe -- otherwise decoding a chunk requires "
            f"the previous one."
        )
    return {
        # NORMALISED, not the raw env value: recording the spelling the caller
        # happened to use would put different dtype strings on byte-identical
        # data and make any dtype comparison spelling-dependent.
        "codec": DEFAULT_CODEC,
        "crf": int(os.environ.get("EGOVERSE_VIDEO_CRF", DEFAULT_CRF)),
        "gop": gop,
        "pix_fmt": os.environ.get("EGOVERSE_VIDEO_PIXFMT", DEFAULT_PIX_FMT),
        "frames_per_chunk": fpc,
    }


def image_codec_enabled() -> bool:
    """True when episodes should be written as video instead of per-frame JPEG.

    DEFAULT IS NOW h264: it is 2.2x smaller and 5x faster to decode than the
    per-frame JPEG it replaces, and wins on PSNR at equal bytes, so there is no
    quality-for-size trade to opt into.

    ``EGOVERSE_IMAGE_CODEC=jpeg`` opts a run back out to per-frame JPEG. This
    affects WRITING only -- readers detect whichever encoding was actually
    written, so flipping it never changes how existing episodes are read, and
    JPEG and video episodes can coexist in one dataset.

    Exactly two values are accepted, ``h264`` and ``jpeg`` -- one spelling per
    outcome, deliberately narrower than :data:`VIDEO_DTYPES`. An unrecognised
    value RAISES rather than falling back: silently treating ``h265`` or a typo
    as "not video" would opt a run out of the default without saying so, and the
    mistake would only surface as a dataset quietly 2.2x larger than intended.
    """
    raw = os.environ.get("EGOVERSE_IMAGE_CODEC", DEFAULT_CODEC).lower()
    if raw == DEFAULT_CODEC:
        return True
    if raw == JPEG_DTYPE:
        return False
    raise ValueError(
        f"EGOVERSE_IMAGE_CODEC={raw!r} is not a supported image codec; "
        f"use '{DEFAULT_CODEC}' or '{JPEG_DTYPE}'"
    )


# --------------------------------------------------------------------------
# Container encode / decode
# --------------------------------------------------------------------------

def encode_chunk(frames_hwc_rgb: np.ndarray, settings: dict, fps: int = 30) -> bytes:
    """Encode a (T, H, W, 3) uint8 RGB block to a self-contained mp4.

    Self-contained matters: the first frame of every chunk must be a keyframe so
    a reader can decode the chunk without touching its neighbours.
    """
    import av

    if frames_hwc_rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames, got {frames_hwc_rgb.dtype}")
    if frames_hwc_rgb.ndim != 4 or frames_hwc_rgb.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3), got {frames_hwc_rgb.shape}")

    T, H, W, _ = frames_hwc_rgb.shape
    buf = io.BytesIO()
    enc = "libx264rgb" if settings["pix_fmt"] == "rgb24" else "libx264"
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream(enc, rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = settings["pix_fmt"]
        stream.options = {
            "crf": str(settings["crf"]),
            "g": str(settings["gop"]),
            # keyint_min=g stops x264 inserting EXTRA keyframes on scene cuts,
            # which would desync the frame->chunk arithmetic the reader relies on.
            "keyint_min": str(settings["gop"]),
            "sc_threshold": "0",
        }
        for i in range(T):
            container.mux(stream.encode(
                av.VideoFrame.from_ndarray(np.ascontiguousarray(frames_hwc_rgb[i]),
                                           format="rgb24")))
        container.mux(stream.encode())
    return buf.getvalue()


def decode_chunk(blob: bytes, n_expected: int | None = None) -> np.ndarray:
    """Decode an mp4 chunk back to (T, H, W, 3) uint8 RGB. Reader-side helper.

    Kept here so writer and reader share one definition of the container format.
    """
    import av

    # zarr hands back VLenBytes elements wrapped in (possibly nested) 0-d
    # object arrays; bytes() on those yields garbage, not the payload.
    while isinstance(blob, np.ndarray):
        blob = blob.item() if blob.shape == () else blob[0]
    if isinstance(blob, memoryview):
        blob = blob.tobytes()

    out = []
    with av.open(io.BytesIO(bytes(blob))) as container:
        for frame in container.decode(video=0):
            out.append(frame.to_ndarray(format="rgb24"))
    arr = np.stack(out, axis=0) if out else np.zeros((0, 0, 0, 3), np.uint8)
    if n_expected is not None and len(arr) != n_expected:
        raise ValueError(f"chunk decoded {len(arr)} frames, expected {n_expected}")
    return arr


# --------------------------------------------------------------------------
# Chunk grouping -- the single definition of how frames map to chunks
# --------------------------------------------------------------------------

def encode_chunks(frames_hwc_rgb, settings: dict, fps: int = 30):
    """Yield one mp4 per ``frames_per_chunk`` frames of a (T, H, W, 3) block.

    Both writer paths go through this so the frame -> chunk grouping has exactly
    one definition: the bulk writer hands it a whole episode, the incremental
    writer hands it one buffered group at a time. A change to chunk semantics
    then has one place to be correct rather than two.
    """
    fpc = settings["frames_per_chunk"]
    for i in range(0, len(frames_hwc_rgb), fpc):
        yield encode_chunk(np.ascontiguousarray(frames_hwc_rgb[i : i + fpc]),
                           settings, fps=fps)


# --------------------------------------------------------------------------
# Read-side access
# --------------------------------------------------------------------------

def decode_video_span(store_array, video_meta: dict, start: int, end: int):
    """Decode frames ``[start, end)`` from a CHUNKED h264 image array.

    The array holds one mp4 per ``frames_per_chunk`` frames, so frame f lives in
    chunk ``f // fpc``. Decode only the covering chunks, concatenate, then
    slice -- reading the whole episode per span would pull ~10x more than needed.

    Returns ``(end-start, 3, H, W)`` float in [0,1], matching the layout the
    JPEG path produces, so downstream collate/normalisation needs no change.
    """
    fpc = int(video_meta.get("frames_per_chunk") or 0)
    if fpc <= 0:
        raise ValueError(f"video_meta missing frames_per_chunk: {video_meta!r}")
    c0, c1 = start // fpc, (end - 1) // fpc + 1
    frames = [decode_chunk(store_array[ci]) for ci in range(c0, c1)]
    block = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), np.uint8)
    lo = start - c0 * fpc
    win = block[lo : lo + (end - start)]
    if len(win) != end - start:
        raise ValueError(
            f"video span [{start},{end}) resolved {len(win)} frames from chunks "
            f"[{c0},{c1}) (fpc={fpc}) -- chunk/frame arithmetic mismatch"
        )
    return np.transpose(win, (0, 3, 1, 2)).astype(np.float64) / 255.0


def frames_per_chunk_from_data(store_array) -> int:
    """Recover ``frames_per_chunk`` by decoding chunk 0 and counting frames.

    Used when an array is detected as video but carries no ``video`` metadata
    block. It cannot be derived arithmetically from the element count: with
    ``total_frames=1000`` and 4 chunks, every fpc in (250, 333] yields 4 chunks,
    so the count alone is ambiguous. Decoding chunk 0 is exact, since all chunks
    but the last are full.
    """
    return int(len(decode_chunk(store_array[0])))


# --------------------------------------------------------------------------
# Encoding detection
# --------------------------------------------------------------------------

def normalize_dtype(declared) -> str | None:
    """Map a declared dtype string to ``"jpeg"``/``"h264"``, or None if neither."""
    if declared in VIDEO_DTYPES:
        return DEFAULT_CODEC
    if declared == JPEG_DTYPE:
        return JPEG_DTYPE
    return None


def _as_bytes(blob):
    """Unwrap a zarr VLenBytes element to real ``bytes``, or None if it isn't."""
    while isinstance(blob, np.ndarray):
        if blob.shape == ():
            blob = blob.item()
        elif len(blob):
            blob = blob[0]
        else:
            return None
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if isinstance(blob, bytearray):
        blob = bytes(blob)
    return blob if isinstance(blob, bytes) else None


def sniff_encoding(blob) -> str | None:
    """Identify a stored image element from its leading magic bytes.

    Returns ``"jpeg"``, ``"h264"``, or None when the payload is neither (e.g. a
    plain numeric array). This is the ground truth: ``dtype`` in the metadata is
    a claim, and a claim can be stale -- episodes written before the codec
    default flipped, or whose metadata was copied from a sibling, declare the
    wrong encoding. Handing an mp4 to simplejpeg raises deep in the decoder, far
    from the actual cause.
    """
    b = _as_bytes(blob)
    if not b or len(b) < 4:
        return None
    if b[:3] == _JPEG_SOI:
        return JPEG_DTYPE
    # ISO-BMFF (what encode_chunk writes): a 4-byte size then the 'ftyp' box.
    if len(b) >= 8 and b[4:8] == b"ftyp":
        return DEFAULT_CODEC
    # Bare Annex-B elementary stream, in case a writer ever skips the container.
    if b[:4] == b"\x00\x00\x00\x01" or b[:3] == b"\x00\x00\x01":
        return DEFAULT_CODEC
    return None


def resolve_encoding(declared, n_elements, total_frames, read_first=None):
    """Decide how an image key is really encoded. Returns ``(verdict, sniffed)``.

    ``verdict`` is ``"jpeg"`` or ``"h264"``; ``sniffed`` is the byte-level answer
    if the payload had to be read, else None -- callers use it to report a
    metadata/payload disagreement, which is why it is returned rather than
    logged here (this function has no episode context to name).

    Three signals, cheapest first, escalating only on disagreement:

    1. ``declared`` -- the dtype claimed in metadata;
    2. ``n_elements`` vs ``total_frames``. Per-frame JPEG stores one element per
       frame, chunked video one mp4 per frames_per_chunk frames, so an array at
       least as long as the episode is JPEG and a much shorter one is video. It
       is ">=" and not "==": the writer pads past total_frames (a 290-frame
       episode occupies 300 slots), so equality would never hold and every
       episode would fall through to a byte read;
    3. ``read_first()`` -- a zero-arg callable returning element 0, invoked ONLY
       to break a tie, since for video that element is a whole mp4 chunk. Its
       answer wins: dtype is a claim, the payload is the fact.

    Pure apart from ``read_first``, so the decision is unit-testable without
    building a zarr episode.
    """
    verdict = normalize_dtype(declared)
    if verdict is None:
        return None, None
    if not n_elements:
        return verdict, None

    # Only a 1-frame episode is genuinely ambiguous, since both layouts then
    # hold exactly one element.
    by_count = None
    if total_frames and total_frames > 1:
        by_count = JPEG_DTYPE if n_elements >= total_frames else DEFAULT_CODEC

    if by_count is not None and by_count == verdict:
        return verdict, None
    if read_first is None:
        return (by_count or verdict), None

    sniffed = sniff_encoding(read_first())
    if sniffed and sniffed != verdict:
        return sniffed, sniffed
    if not sniffed and by_count is not None and by_count != verdict:
        return by_count, None
    return verdict, sniffed


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def build_feature_entry(img_shape, settings: dict, total_frames: int, fps: int) -> dict:
    fpc = settings["frames_per_chunk"]
    return {
        "dtype": settings["codec"],          # "h264" -- readers dispatch on this
        "shape": list(img_shape),
        "names": ["height", "width", "channel"],
        "video": {
            "codec": settings["codec"],
            "crf": settings["crf"],
            "gop": settings["gop"],
            "pix_fmt": settings["pix_fmt"],
            "fps": fps,
            "frames_per_chunk": fpc,
            "total_frames": int(total_frames),
            "n_chunks": int((total_frames + fpc - 1) // fpc),
        },
    }
