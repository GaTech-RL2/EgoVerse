"""Client-side Real-Time Chunking controller (RTC — Physical Intelligence).

Pure numpy, no robot deps: unit-testable standalone (`python rtc_client.py`).
The rollout loop owns the threads and the queue; this class owns the RTC
bookkeeping so the hooks in rollout_rby1_policy_real.py stay tiny:

    rtc = RTCController(freeze_steps=3, period=8, ...)
    # first (plain, un-constrained) chunk arrives:
    rows = rtc.on_chunk(chunk_raw)                  # -> rows to enqueue (all 64)
    # every executed action:
    rtc.on_execute()
    # each tick, may we start a new inference?
    if rtc.should_submit(step): rtc.attach(payload, step)   # adds rtc_* keys
    # when the worker's chunk lands:
    rows = rtc.on_chunk(chunk_raw)                  # -> chunk[delta:]; queue.clear() first
    print(rtc.last_report)

Definitions (10 Hz control, dt=0.1 s):
  d (freeze_steps)  rows of the new chunk frozen to the executing chunk; must
                    cover the rows consumed while inference runs (delta).
                    d < delta  =>  the boundary jerk RTC removes comes back.
  K (period)        control steps between inference snapshots (replan cadence).
  E (soft_horizon)  row where soft consistency ends (weights 1 -> 0 over [d,E)).
  gamma             soft-decay exponent (1 = linear).
"""
from __future__ import annotations

import numpy as np

BLOCKS = {"base": (0, 3), "torso": (3, 9), "head": (9, 11), "l_arm": (11, 18),
          "r_arm": (18, 25), "l_hand": (25, 37), "r_hand": (37, 49)}


def smooth_chunk(chunk: np.ndarray, passes: int = 1) -> np.ndarray:
    """Zero-lag within-chunk smoothing: [0.25, 0.5, 0.25] along time, per dim.

    Measured 2026-09-21 (V0, basket train frame): one pass takes the chunk's
    arm |d2a| from 0.0044 to 0.0015 — demo level (GT 0.0010) — with velocity
    and GT-MAE unchanged. The policy's raw chunks are intrinsically ~4x
    jerkier than the demos (proprio domain gap; denoising steps 8..100 make
    no difference), so this is THE within-motion smoothness lever. Because the
    whole chunk is available before execution, the filter is symmetric — no
    group delay, unlike an online EMA. Endpoints are kept unfiltered.
    """
    x = np.asarray(chunk, dtype=np.float64)
    if len(x) < 3 or passes < 1:
        return x
    for _ in range(int(passes)):
        y = x.copy()
        y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        x = y
    return x


class RTCController:
    def __init__(self, freeze_steps: int = 3, period: int = 8,
                 soft_horizon: int | None = None, soft_power: float = 1.0,
                 horizon: int = 64):
        if period < 1:
            raise ValueError("period must be >= 1")
        if freeze_steps < 1:
            raise ValueError("freeze_steps must be >= 1 (cover the inference latency)")
        self.d = int(freeze_steps)
        self.K = int(period)
        self.E = int(soft_horizon) if soft_horizon else self.d + 16
        self.gamma = float(soft_power)
        self.H = int(horizon)
        # executing-chunk state
        self.prev_raw: np.ndarray | None = None   # (H, D) server action space
        self.base_idx = 0                         # rows of prev_raw already consumed
        # in-flight inference state
        self.pending = False
        self._submit_step = -10**9
        self._submit_base = 0
        self._last_snapshot_step = -10**9
        self.last_report = ""
        self.n_late = 0
        self.n_chunks = 0

    # ---- executing-chunk lifecycle -------------------------------------- #
    def on_execute(self) -> None:
        """Call once per action actually popped for execution."""
        self.base_idx += 1

    def should_submit(self, step: int) -> bool:
        """Start a new inference? (the loop also requires no pending future)."""
        if self.prev_raw is None or self.pending:
            return False
        if step - self._last_snapshot_step < self.K:
            return False
        # need enough remaining rows to survive the freeze window
        return (len(self.prev_raw) - self.base_idx) > self.d + 1

    def attach(self, payload: dict, step: int) -> None:
        """Add the rtc_* keys to an obs payload snapshotted at this instant."""
        assert self.prev_raw is not None and not self.pending
        tail = np.ascontiguousarray(self.prev_raw[self.base_idx:], dtype=np.float32)
        payload["rtc_prev_actions"] = tail
        payload["rtc_freeze_steps"] = int(self.d)
        payload["rtc_soft_horizon"] = int(self.E)
        payload["rtc_soft_power"] = float(self.gamma)
        self.pending = True
        self._submit_step = int(step)
        self._submit_base = int(self.base_idx)
        self._last_snapshot_step = int(step)

    def on_chunk(self, chunk: np.ndarray, step: int | None = None) -> np.ndarray:
        """A chunk arrived (constrained if self.pending, plain otherwise).

        Returns the rows to enqueue; caller clears the queue first (splice).
        ``step`` anchors the replan clock for a fresh (unconstrained) chunk.
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        self.n_chunks += 1
        if not self.pending or self.prev_raw is None:
            # first chunk / fallback resync: execute from row 0
            self.prev_raw = chunk
            self.base_idx = 0
            self.pending = False
            if step is not None:
                self._last_snapshot_step = int(step)
            self.last_report = f"[rtc] fresh chunk ({len(chunk)} rows, no constraint)"
            return chunk

        delta = self.base_idx - self._submit_base    # rows consumed during inference
        if delta < 0 or delta >= len(chunk):
            # cannot happen in a healthy loop (delta ~ latency ticks << 64);
            # a degenerate arrival must not latch bad indices — resync fresh.
            self.prev_raw = chunk
            self.base_idx = 0
            self.pending = False
            self.last_report = f"[rtc] RESYNC: absurd delta={delta}, treating as fresh chunk"
            return chunk
        late = delta > self.d
        if late:
            self.n_late += 1
        j = min(self.base_idx, len(self.prev_raw) - 1)
        dnew = min(delta, len(chunk) - 1)
        gap = np.abs(chunk[dnew] - self.prev_raw[j])
        per_block = "  ".join(
            f"{k} {gap[a:b].max():.4f}" for k, (a, b) in BLOCKS.items()
            if b <= len(gap))
        self.last_report = (
            f"[rtc] splice delta={delta} (d={self.d}"
            f"{', LATE — raise --rtc-freeze-steps' if late else ''}) "
            f"boundary max|new-prev|: {per_block}")
        self.prev_raw = chunk
        self.base_idx = delta
        self.pending = False
        return chunk[delta:]

    def note_first_snapshot(self, step: int) -> None:
        """Anchor the replan clock at the first chunk's arrival."""
        self._last_snapshot_step = int(step)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    H, D = 64, 49
    A = rng.normal(size=(H, D))
    rtc = RTCController(freeze_steps=3, period=8, soft_horizon=19)

    rows = rtc.on_chunk(A, step=100)
    assert rows.shape == (H, D) and np.allclose(rows, A)
    assert rtc.n_chunks == 1
    assert not rtc.should_submit(105)          # K=8 not elapsed
    for _ in range(8):
        rtc.on_execute()                       # consumed rows 0..7
    assert rtc.should_submit(108)
    payload = {}
    rtc.attach(payload, step=108)
    assert payload["rtc_prev_actions"].shape == (H - 8, D)
    assert np.allclose(payload["rtc_prev_actions"][0], A[8])   # time-aligned
    assert payload["rtc_freeze_steps"] == 3
    assert not rtc.should_submit(120)          # pending blocks re-submit

    # inference took 2 ticks: rows 8,9 consumed meanwhile
    rtc.on_execute(); rtc.on_execute()
    B = rng.normal(size=(H, D))
    B[:3] = A[8:11]                            # server froze d=3 rows
    rows = rtc.on_chunk(B)
    assert rows.shape == (H - 2, D)            # delta=2 rows skipped
    assert np.allclose(rows[0], B[2])
    assert "LATE" not in rtc.last_report and "delta=2" in rtc.last_report
    assert rtc.base_idx == 2 and not rtc.pending

    # boundary metric ~0 because B[2] == A[10] (frozen)
    assert max(float(s.split()[-1]) for s in [rtc.last_report.split("r_hand")[-1]]) < 1e-9

    # late case: delta=5 > d=3 flags LATE
    for _ in range(6):
        rtc.on_execute()
    rtc.attach({}, step=118)
    for _ in range(5):
        rtc.on_execute()
    rtc.on_chunk(rng.normal(size=(H, D)))
    assert "LATE" in rtc.last_report and rtc.n_late == 1

    # fallback resync when a chunk arrives without a pending constraint
    rows = rtc.on_chunk(rng.normal(size=(H, D)))
    assert len(rows) == H and rtc.base_idx == 0
    print("rtc_client selftest OK")


if __name__ == "__main__":
    _selftest()
