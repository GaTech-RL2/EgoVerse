"""Zarr-backed image / thumbnail loading for the latent inspector."""

from __future__ import annotations

import base64
import io
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def open_zarr_for_hash(zarr_root: str, video_hash: str):
    """Open zarr group for this episode (cached so successive clicks on the
    same episode are fast)."""
    import zarr

    path = os.path.join(zarr_root, video_hash)
    if not os.path.isdir(path):
        return None
    try:
        return zarr.open(path, mode="r")
    except Exception as e:
        logger.warning("Failed to open zarr %s: %s", path, e)
        return None


_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")


def _bytes_from_zarr_element(elem) -> bytes:
    """Turn one element of a zarr array into raw bytes.
    Zarr v3 with VLenBytesCodec returns a nested 0-d numpy.object_ array
    instead of bytes directly (e.g. `array(array(array(b'...')))`); we
    repeatedly call `.item()` to drill through the wrappers.
    For uint8 arrays, falls back to `.tobytes()`."""
    for _ in range(8):  # bounded — never deeper than a few wrappers in practice
        if isinstance(elem, (bytes, bytearray, memoryview)):
            return bytes(elem)
        # uint8/int8 numpy array → raw bytes.
        if (
            hasattr(elem, "dtype")
            and getattr(elem.dtype, "itemsize", 0) == 1
            and getattr(elem.dtype, "kind", "") in ("u", "i", "b")
        ):
            try:
                return elem.tobytes()
            except Exception:
                pass
        # 0-d object array → unwrap with .item().
        if hasattr(elem, "item"):
            try:
                nxt = elem.item()
            except Exception:
                break
            if nxt is elem:
                break
            elem = nxt
            continue
        break
    if isinstance(elem, (bytes, bytearray, memoryview)):
        return bytes(elem)
    return bytes(elem)


def _looks_like_image_array(arr) -> bool:
    """Probe one frame and check if its bytes start with a known image
    magic header (JPEG / PNG / GIF). Works regardless of the zarr key name."""
    try:
        if not hasattr(arr, "shape") or len(arr.shape) < 1 or arr.shape[0] == 0:
            return False
        b = _bytes_from_zarr_element(arr[0])
        if len(b) < 8:
            return False
        return any(b.startswith(m) for m in _IMAGE_MAGIC)
    except Exception:
        return False


def _candidate_image_keys(grp, primary: str):
    """Return a deduped list of plausible image-array keys to try, primary
    first. Strategy:
      1. user-supplied --image-key
      2. common name patterns (images.front_1, etc.)
      3. content-based scan: any zarr array whose first byte is JPEG/PNG
         magic — works regardless of naming convention.
    """
    seen = set()
    out: list[str] = []

    def add(k):
        if k and k not in seen:
            seen.add(k)
            out.append(k)

    add(primary)
    common = [
        # eva-style
        "images.front_1",
        "images.front",
        "images.left_1",
        "images.right_1",
        "images.wrist_1",
        "images.top_1",
        "image",
        # aria-style guesses
        "images.aria_main",
        "images.video_aria_main",
        "obs_video",
        "obs_rgb",
        "video",
        "rgb",
        # observation-prefixed
        "observations.images.front_1",
        "observations.images.front",
        "observations.images.aria",
    ]
    for k in common:
        add(k)

    # Content-based: walk the group and probe every array's first byte for
    # a JPEG/PNG magic header. Bounded to depth 3 to stay quick.
    def _walk(g, prefix="", depth=0):
        if depth > 3:
            return
        try:
            keys = list(g.keys())
        except Exception:
            return
        for k in keys:
            path = f"{prefix}/{k}" if prefix else k
            try:
                sub = g[k]
            except Exception:
                continue
            if hasattr(sub, "shape") and len(sub.shape) >= 1:
                if _looks_like_image_array(sub):
                    add(path)
            else:
                _walk(sub, path, depth + 1)

    _walk(grp)
    return out


def _resolve_zarr_path(grp, key: str):
    """Look up a zarr key. Tries the literal name FIRST (handles cases like
    'images.front_1' where the dot is part of the key, not a path
    delimiter). Then falls back to slash-split nested lookup so paths like
    'annotations/lang' still work."""
    if not key:
        return None
    # 1) Literal key (covers 'images.front_1' and any other dotted leaf names).
    try:
        return grp[key]
    except Exception:
        pass
    # 2) Slash-separated nested traversal: 'annotations/lang/text'
    cur = grp
    for part in key.split("/"):
        if part == "":
            continue
        try:
            cur = cur[part]
        except Exception:
            return None
    return cur


def _walk_zarr_str_arrays(grp, prefix=""):
    """Yield (path, array) for every leaf array whose dtype looks like text
    (string / object / bytes). Stops at depth 4 to bound work."""
    try:
        keys = list(grp.keys())
    except Exception:
        return
    for k in keys:
        try:
            sub = grp[k]
        except Exception:
            continue
        path = f"{prefix}/{k}" if prefix else k
        if hasattr(sub, "dtype") and hasattr(sub, "shape"):
            kind = getattr(sub.dtype, "kind", "")
            if kind in ("U", "S", "O"):
                yield path, sub
        elif prefix.count("/") < 4:
            yield from _walk_zarr_str_arrays(sub, path)


def load_image_b64(zarr_root: str, video_hash: str, frame_idx: int, image_key: str):
    """Read JPEG bytes from zarr at (hash, frame_idx) and return
    (data_uri or None, debug_info_list)."""
    tried: list[str] = []
    grp = open_zarr_for_hash(zarr_root, video_hash)
    if grp is None:
        return None, [f"zarr group not found at {os.path.join(zarr_root, video_hash)}"]

    for key in _candidate_image_keys(grp, image_key):
        arr = _resolve_zarr_path(grp, key)
        if arr is None or not hasattr(arr, "shape"):
            tried.append(f"{key} ✗ (missing)")
            continue
        if not arr.shape:
            tried.append(f"{key} ✗ (scalar, no frames)")
            continue
        n = arr.shape[0]
        if n == 0:
            tried.append(f"{key} ✗ (empty)")
            continue
        # See load_thumbnail_jpeg for the rationale: -1 is eval_latent's
        # "no source frame recorded" sentinel; fall back to frame 0 so the
        # UI shows the recording's first frame rather than nothing.
        if frame_idx == -1:
            tried.append(f"{key} ⚠ (frame_idx=-1 sentinel; falling back to frame 0)")
            frame_idx = 0
        if not (0 <= frame_idx < n):
            tried.append(
                f"{key} ✗ (frame_idx={frame_idx} out of range; episode has {n} frames — "
                f"CSV likely written by an old eval_latent build that stored a per-run "
                f"sample index instead of the source frame index)"
            )
            continue
        idx = frame_idx
        try:
            raw = _bytes_from_zarr_element(arr[idx])
        except Exception as e:
            tried.append(f"{key} ✗ ({type(e).__name__}: {e})")
            continue
        if len(raw) < 4:
            tried.append(f"{key} ✗ (only {len(raw)} bytes — corrupt frame)")
            continue
        # Try PIL first (more permissive).
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            img.load()
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            tried.append(f"{key} ✓ ({len(raw)} bytes, decoded)")
            return f"data:image/jpeg;base64,{b64}", tried
        except Exception as e_pil:
            # Bytes start with JPEG magic? Hand raw to the browser.
            if raw[:3] == b"\xff\xd8\xff":
                b64 = base64.b64encode(raw).decode("ascii")
                tried.append(f"{key} ✓ ({len(raw)} bytes, raw — PIL decode failed)")
                return f"data:image/jpeg;base64,{b64}", tried
            tried.append(f"{key} ✗ (decode failed: {type(e_pil).__name__})")
            continue
    return None, tried


def load_thumbnail_jpeg(
    zarr_root: str, video_hash: str, frame_idx: int, image_key: str, size: int = 360
) -> bytes | None:
    """Read a frame, decode, downsize to fit `size` px, return JPEG bytes.
    Returns None if the frame can't be located/decoded. Used by the
    /thumbnail route so each grid card gets a usable image without
    embedding base64 data URIs in the HTML.

    `size` defaults to 360 because the grid cards are ~280px wide; smaller
    source images get upscaled by the browser and look blurry/color-banded.
    """
    grp = open_zarr_for_hash(zarr_root, video_hash)
    if grp is None:
        return None
    for key in _candidate_image_keys(grp, image_key):
        arr = _resolve_zarr_path(grp, key)
        if arr is None or not hasattr(arr, "shape") or len(arr.shape) < 1:
            continue
        if arr.shape[0] == 0:
            continue
        # `frame_idx == -1` is the eval_latent sentinel used when the
        # source dataset didn't emit `frame_index` (safe_collate then drops
        # the key from the batch). Rather than rendering nothing — which
        # leaves every grid card blank — fall back to frame 0 so the user
        # at least sees the recording's first frame as a placeholder.
        if frame_idx == -1:
            frame_idx = 0
        if not (0 <= frame_idx < arr.shape[0]):
            # Out-of-range source frame: skip rather than silently render
            # frame 0 (which made stale CSVs look superficially valid).
            continue
        idx = frame_idx
        try:
            raw = _bytes_from_zarr_element(arr[idx])
        except Exception:
            continue
        if len(raw) < 4:
            continue
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            img.load()
            img = img.convert("RGB")
            img.thumbnail((size, size), resample=Image.LANCZOS)
            buf = io.BytesIO()
            # subsampling=0 → 4:4:4 chroma (no color aliasing on small images);
            # quality=85 keeps file size sub-15KB at 360px and removes the
            # banding/washed-out look of quality=72.
            img.save(buf, format="JPEG", quality=85, subsampling=0, optimize=True)
            return buf.getvalue()
        except Exception:
            continue
    return None
