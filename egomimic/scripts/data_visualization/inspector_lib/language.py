"""Annotation / language prompt lookup for the latent inspector."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from .images import (
    _resolve_zarr_path,
    _walk_zarr_str_arrays,
    open_zarr_for_hash,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=512)
def annotation_intervals(
    zarr_root: str, video_hash: str
) -> tuple[tuple[int, int, str, str], ...]:
    """Return (start_idx, end_idx, lower_text, original_text) tuples parsed
    from this recording's `annotations` zarr array. Sorted by start_idx.
    Cached because we call this once per (video_hash) for filtering and
    once per click for the language prompt display."""
    grp = open_zarr_for_hash(zarr_root, video_hash)
    if grp is None:
        return tuple()
    try:
        ann = grp["annotations"]
    except Exception:
        return tuple()
    out: list[tuple[int, int, str, str]] = []
    for i in range(getattr(ann, "shape", (0,))[0]):
        raw = ann[i]
        # Same unwrap dance as test_zarr.validate_episode.
        while isinstance(raw, np.ndarray):
            raw = raw.item() if raw.shape == () else raw.flat[0]
        if isinstance(raw, np.bytes_):
            raw = bytes(raw)
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
        try:
            import json as _json

            rec = _json.loads(raw)
            s = int(rec["start_idx"])
            e = int(rec["end_idx"])
            t_orig = str(rec.get("text", "")).strip()
            if e > s:
                out.append((s, e, t_orig.lower(), t_orig))
        except Exception:
            continue
    return tuple(sorted(out))


def lang_for_frame(intervals: tuple, frame: int, *, original: bool = False) -> str:
    """Find the most-recent annotation interval containing `frame`. Returns
    lower-cased text by default (for substring filtering); pass
    `original=True` to get the un-cased text for UI display."""
    idx = 3 if original else 2
    last = ""
    for entry in intervals:
        s, e = entry[0], entry[1]
        if s <= frame < e:
            last = entry[idx] if len(entry) > idx else ""
        elif s > frame:
            break
    return last


def interval_for_frame(intervals: tuple, frame: int):
    """Return the (s, e, original_text) tuple of the interval covering
    `frame`, or None if no interval matches."""
    last = None
    for entry in intervals:
        s, e = entry[0], entry[1]
        if s <= frame < e:
            last = (s, e, entry[3] if len(entry) >= 4 else entry[2])
        elif s > frame:
            break
    return last


def all_lang_concat_lower(intervals: tuple, frame: int) -> str:
    """Lowercase concatenation of EVERY annotation interval covering
    `frame`. Many recordings have multiple paraphrases per interval (e.g.
    5 different ways to say 'pick up the beige bowl' all spanning frames
    0–162); when filtering by substring we want a hit on ANY paraphrase,
    not just whichever one happens to sort last."""
    parts: list[str] = []
    for entry in intervals:
        s, e = entry[0], entry[1]
        if s <= frame < e:
            lo = entry[2] if len(entry) > 2 else ""
            if lo:
                parts.append(lo)
        elif s > frame:
            break
    return " | ".join(parts)


def format_value(val) -> str:
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace")
        except Exception:
            return str(val)
    return str(val)


def load_language_prompt(
    zarr_root: str, video_hash: str, frame_idx: int, lang_key: str | None = None
):
    """Find a language/annotation string for this frame. If `lang_key` is
    given, use it directly. Otherwise: 1) try a list of common paths, then
    2) auto-walk the zarr for any string-typed array.

    Returns a tuple (text_or_none, debug_info). `text_or_none` is the prompt
    string or None; `debug_info` lists the paths attempted so missing
    prompts can be diagnosed in the UI."""
    tried: list[str] = []
    grp = open_zarr_for_hash(zarr_root, video_hash)
    if grp is None:
        return None, ["zarr group could not be opened"]

    # Preferred: parse the `annotations` array as a list of
    # (start_idx, end_idx, text) records and find the one that COVERS
    # `frame_idx`. The legacy code below treated annotations as if it
    # were one entry per frame and indexed `annotations[frame_idx]`,
    # which returns an arbitrary record unrelated to the actual frame.
    if not lang_key:
        intervals = annotation_intervals(zarr_root, video_hash)
        if intervals:
            tried.append(f"annotations intervals: {len(intervals)} records")
            match = interval_for_frame(intervals, int(frame_idx))
            if match is not None:
                s, e, t = match
                return f"{t}  [start_idx={s}, end_idx={e}]", tried
            tried.append(f"frame {frame_idx} not covered by any annotation")

    paths_to_try: list[str] = []
    if lang_key:
        paths_to_try.append(lang_key)
    paths_to_try += [
        "annotations",
        "annotation",
        "language",
        "lang",
        "task",
        "prompt",
        "annotations/lang",
        "annotations/text",
        "annotations/instruction",
        "annotations/0/text",
        "annotations/0/lang",
        "obs_lang",
        "obs/lang",
        "language_prompt",
    ]

    # Pass 1: explicit candidates.
    for key in paths_to_try:
        sub = _resolve_zarr_path(grp, key)
        tried.append(key + (" ✓ exists" if sub is not None else " ✗"))
        if sub is None:
            continue
        try:
            if hasattr(sub, "shape") and len(sub.shape) >= 1 and sub.shape[0] > 0:
                idx = min(frame_idx, sub.shape[0] - 1) if frame_idx >= 0 else 0
                val = sub[idx]
            else:
                val = sub[()]  # scalar zarr
            text = format_value(val).strip()
            if text:
                return f"{key}: {text}", tried
        except Exception as e:
            tried.append(f"  read error on {key}: {e}")
            continue

    # Pass 2: auto-walk for any string-typed array we missed.
    auto = []
    for path, arr in _walk_zarr_str_arrays(grp):
        try:
            if len(arr.shape) >= 1 and arr.shape[0] > 0:
                idx = min(frame_idx, arr.shape[0] - 1) if frame_idx >= 0 else 0
                val = arr[idx]
            else:
                val = arr[()]
            text = format_value(val).strip()
            if text:
                auto.append((path, text))
        except Exception:
            continue
    if auto:
        path, text = auto[0]
        tried.append(f"auto-walk found {path}")
        return f"{path} (auto): {text}", tried

    return None, tried
