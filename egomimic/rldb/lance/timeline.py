"""Recording timeline and per-frame validity helpers shared by converters and the reader."""

import numpy as np

# 30 cm in one frame is 9 m/s at 30 fps: a tracking jump, not motion (as scan_egomotion.py).
JUMP_M = 0.3
FROZEN_RUN = 15


def build_timeline(ts: np.ndarray, fps: float = 30.0):
    """Constant-rate timeline over the RGB frames: dropped frames become filler rows.

    Returns (timestamps, source frame index per row, filler mask, first-row-after-gap mask).
    """
    period = 1e9 / fps
    dt = np.diff(ts)
    missing = np.maximum(np.rint(dt / period).astype(np.int64) - 1, 0)
    n = len(ts) + int(missing.sum())
    out_ts = np.empty(n, dtype=np.int64)
    src = np.empty(n, dtype=np.int64)
    filler = np.zeros(n, dtype=bool)
    after_gap = np.zeros(n, dtype=bool)
    r = 0
    for i in range(len(ts)):
        out_ts[r], src[r] = ts[i], i
        if i > 0 and missing[i - 1] > 0:
            after_gap[r] = True
        r += 1
        if i < len(missing) and missing[i] > 0:
            k = missing[i]
            out_ts[r : r + k] = ts[i] + np.rint(np.arange(1, k + 1) * period).astype(
                np.int64
            )
            src[r : r + k] = i
            filler[r : r + k] = True
            r += k
    return out_ts, src, filler, after_gap


def nearest_dt(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|query - nearest(ref)| in ns; ref sorted."""
    j = np.clip(np.searchsorted(ref, query), 1, len(ref) - 1)
    return np.minimum(np.abs(query - ref[j - 1]), np.abs(ref[j] - query))


def pose_outliers(xyz: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Rows that jump more than JUMP_M from the previous valid row, or sit in a frozen run."""
    bad = np.zeros(len(xyz), dtype=bool)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return bad
    step = np.linalg.norm(np.diff(xyz[idx], axis=0), axis=1)
    bad[idx[1:]] = step > JUMP_M
    # Hand results can repeat for a frame or two when MPS runs below the RGB rate, so
    # only a long run of identical positions counts as frozen.
    run_start = 0
    for j in range(1, len(idx) + 1):
        if j == len(idx) or step[j - 1] != 0:
            if j - run_start >= FROZEN_RUN:
                bad[idx[run_start:j]] = True
            run_start = j
    return bad
