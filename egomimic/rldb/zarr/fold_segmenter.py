"""SE(3) fold segmenter — derives fold spans from the head pose track.

A fold-clothes demo is a repeating motion: the operator folds, then resets. The
reset shows up as a jump in head translation OR rotation, so boundaries are the
union of peaks on both channels, spaced by a period estimated from the strong
peaks. Nothing is invented -- every boundary is an observed peak.

Two entry points, one implementation:
  * `segment_head_pose(h, T)`   -- conversion time, when the (T,7) array is in hand
  * `segment_group(group)`      -- post hoc, against a written zarr group

`apply_gate=True` (default) drops length-outlier folds, which is what the
shipped `fold_segments` contain: on the 571 already-segmented PACE episodes the
raw segmenter emits 9109 spans while the shipped set holds 8560 -- the 549
dropped are the LONG ones (median 249 frames vs 152 kept). Gating reproduces
71.6% of episodes exactly; boundary recall vs the shipped spans is 1.000.

Thresholds live in DEFAULTS so a variant is a dict override, not an edit.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

# translation in metres, rotation in radians; (candidate, strong) per channel.
DEFAULTS = {
    "dt_c": 0.012, "dt_s": 0.08,     # translation candidate / strong
    "dr_c": 0.05,  "dr_s": 0.20,     # rotation    candidate / strong
    "dist": 25,    "sdist": 40,      # min peak separation (frames)
    "gap": 45,                       # strong peak counts as accounted within this
    "min_seg": 45,                   # shortest admissible fold (1.5 s @ 30 fps)
    "len_hi": 1.6, "len_lo": 0.55,   # length gate, relative to episode median
}

FOLD_SPAN_TEXT = "fold the clothes"


def _cfg(overrides=None):
    c = dict(DEFAULTS)
    if overrides:
        unknown = set(overrides) - set(c)
        if unknown:
            raise ValueError("unknown segmenter knobs: %s" % sorted(unknown))
        c.update(overrides)
    return c


def _geodesic(q):
    """Frame-to-frame quaternion geodesic distance, sign-invariant."""
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    d = np.abs((q[1:] * q[:-1]).sum(1)).clip(-1, 1)
    return 2 * np.arccos(d)


def head_jumps(head_pose):
    """(T,7) pose [xyz, quat] -> (translation delta, rotation delta) per frame."""
    h = np.asarray(head_pose)
    if h.ndim != 2 or h.shape[1] < 7:
        raise ValueError("head_pose must be (T,7+), got %s" % (h.shape,))
    dt = np.linalg.norm(np.diff(h[:, :3], axis=0), axis=1)
    dr = _geodesic(h[:, 3:7])
    return dt, dr


def _merge(peaks, gap):
    out = []
    for p in sorted(int(x) for x in peaks):
        if out and p - out[-1] < gap:
            continue
        out.append(p)
    return np.array(out, dtype=int)


def _segment(dt, dr, T, c):
    score = np.maximum(dt / c["dt_c"], dr / c["dr_c"])
    pt, _ = find_peaks(dt, height=c["dt_c"], distance=c["dist"], prominence=c["dt_c"] * 0.5)
    pr, _ = find_peaks(dr, height=c["dr_c"], distance=c["dist"], prominence=c["dr_c"] * 0.5)
    cand = _merge(list(pt) + list(pr), c["dist"])
    st, _ = find_peaks(dt, height=c["dt_s"], distance=c["sdist"], prominence=c["dt_s"] * 0.5)
    sr, _ = find_peaks(dr, height=c["dr_s"], distance=c["sdist"], prominence=c["dr_s"] * 0.5)
    strong = _merge(list(st) + list(sr), c["sdist"])

    ms = c["min_seg"]
    if len(strong) >= 4 and np.median(np.diff(strong)) >= ms:
        P = float(np.median(np.diff(strong)))
    elif len(cand) >= 4:
        P = float(np.median(np.diff(cand)))
    else:
        P = None

    if P is None:
        bnds = [0] + [int(x) + 1 for x in cand] + [T]
    else:
        P = max(P, ms * 1.2)
        bnds, last, guard = [0], 0, 0
        while guard < 4000:
            guard += 1
            lo, hi = last + c["len_lo"] * P, last + c["len_hi"] * P
            inwin = cand[(cand >= lo) & (cand <= hi)]
            if len(inwin):
                nb = int(inwin[np.argmax(score[inwin])]) + 1
            else:
                nxt = cand[cand >= max(lo, last + ms)]
                if not len(nxt):
                    break
                nb = int(nxt[0]) + 1
            if nb >= T - 0.5 * ms or nb <= last:
                break
            if nb - last >= ms:
                bnds.append(nb); last = nb
            else:
                nxt = cand[cand >= last + ms]
                if not len(nxt):
                    break
                last = int(nxt[0])
        bnds.append(T)

    segs = [(int(a), int(b)) for a, b in zip(bnds[:-1], bnds[1:]) if b - a >= ms]
    return segs, strong, np.array(bnds, dtype=int)


def _gate(segs, strong, bnds, c):
    """Flag outliers; returns (kept_segs, info)."""
    L = np.array([b - a for a, b in segs], dtype=float)
    m = float(np.median(L)) if len(L) else 0.0
    out_idx = [i for i, l in enumerate(L)
               if m > 0 and (l > c["len_hi"] * m or l < c["len_lo"] * m)]
    unacc = [int(s) for s in strong
             if len(bnds) and np.abs(bnds - (s + 1)).min() > c["gap"]]
    kept = [s for i, s in enumerate(segs) if i not in set(out_idx)]
    return kept, {"flagged": bool(out_idx or unacc), "n_len_outliers": len(out_idx),
                  "n_unaccounted": len(unacc), "n_raw": len(segs), "median_len": m}


def segment_head_pose(head_pose, total_frames=None, apply_gate=True, knobs=None):
    """Segment one episode. Returns {'segs': [(s,e)...], 'info': {...}}."""
    c = _cfg(knobs)
    h = np.asarray(head_pose)
    T = int(h.shape[0]) if total_frames is None else int(total_frames)
    h = h[:T]
    if T < c["min_seg"] * 2:
        return {"segs": [], "info": {"flagged": True, "reason": "too_short", "n_raw": 0}}
    dt, dr = head_jumps(h)
    segs, strong, bnds = _segment(dt, dr, T, c)
    info = {"flagged": False, "n_raw": len(segs)}
    if apply_gate:
        segs, info = _gate(segs, strong, bnds, c)
    return {"segs": segs, "info": info}


def segment_group(group, total_frames=None, apply_gate=True, knobs=None):
    """Same, reading `obs_head_pose` + `attrs['total_frames']` off a zarr group."""
    T = int(group.attrs.get("total_frames")) if total_frames is None else int(total_frames)
    return segment_head_pose(group["obs_head_pose"][:T], T, apply_gate, knobs)


def spans_to_annotations(segs, text=FOLD_SPAN_TEXT):
    """-> [(text, start, end)] as `ZarrWriter.append_annotations` expects."""
    return [(text, int(a), int(b)) for a, b in segs]
