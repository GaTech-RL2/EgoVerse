"""Aria Gen 2 frame filter: opening calibration, resets and demo segmentation.

Gen 2 tracks hands well enough to see them during the opening calibration
(look-around head sweeps) and during resets (glasses lifted onto the
forehead), so the Gen 1 "drop frames without a hand" rule is not enough.

A recording is split into demos at the *boundaries*: resets (long eye-gaze
dropouts, i.e. glasses off the face) and the opening calibration. Everything
else that goes wrong inside a demo (tracking gaps, short gaze dropouts, fast
head turns, a hand at the headset, an untracked hand) is an *error*: it never
splits a demo and is dropped or kept according to the caller.

Everything here works on per-RGB-frame numpy arrays; reading them from a VRS
or MPS outputs lives in aria_utils (gen2_frame_signals).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Gen2FilterConfig:
    """Thresholds for the Gen 2 filter.

    Defaults were set from the 2026-11-15/16 recordings: resets are eye-gaze
    dropouts of 5-20 s, calibration head sweeps run at 115-140 deg/s against
    a task p99 of ~45 deg/s, and demos are ~15-45 s long.
    """

    # Max |sample - RGB frame| gap before a frame counts as not tracked.
    max_pose_dt_s: float = 0.005  # MPS closed loop is 1 kHz, on-device VIO ~800 Hz
    max_hand_dt_s: float = 0.020  # hand tracking is 30-60 Hz
    max_gaze_dt_s: float = 0.050  # on-device eye gaze is 30 Hz
    # Eye gaze invalid for at least reset_min_s: glasses off the face (a
    # reset, a demo boundary), padded for reaching for / settling the glasses.
    reset_min_s: float = 2.0
    reset_pad_s: float = 1.0
    # Shorter dropouts down to gaze_dropout_min_s are in-demo errors; shorter
    # still are blinks and are kept.
    gaze_dropout_min_s: float = 0.3
    # Fast head turn: mean head angular speed over head_sweep_window_s.
    head_sweep_deg_s: float = 80.0
    head_sweep_window_s: float = 0.5
    head_sweep_pad_s: float = 0.5
    # Opening calibration: a chain of fast head sweeps (measured before
    # padding) that starts within calib_start_s, with gaps of at most
    # calib_max_gap_s between sweeps, at least calib_min_sweep_s of sweeping
    # in total including one sustained sweep of calib_min_run_s, and ending
    # within calib_window_s. Calibration then covers everything from the start
    # of the recording to the end of that chain. If the first chain is only a
    # stray glance (<= calib_glance_max_s of sweeping), one later chain that
    # starts within calib_restart_start_s may be the calibration instead (seen
    # on s_4: a glance at 1.6 s, calibration from 11 s). Real calibrations hold
    # 1.1-7.2 s sweeps; task glances are <= 0.7 s, so a glance (or a run of
    # glances) is an error, and a late head turn is never calibration.
    calib_start_s: float = 10.0
    calib_restart_start_s: float = 15.0
    calib_glance_max_s: float = 0.7
    calib_max_gap_s: float = 5.0
    calib_min_sweep_s: float = 1.5
    calib_min_run_s: float = 1.0
    calib_window_s: float = 30.0
    # Any hand keypoint this close to the device: grabbing the glasses.
    hand_at_headset_m: float = 0.12
    hand_at_headset_pad_s: float = 0.5
    # Demos with less usable (non-error) time than this are not episodes.
    min_demo_s: float = 5.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> Gen2FilterConfig:
        """Load overrides from a YAML mapping of field name -> value."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown Gen2FilterConfig fields: {sorted(unknown)}")
        bad = [
            k
            for k, v in data.items()
            if isinstance(v, bool) or not isinstance(v, (int, float))
        ]
        if bad:
            raise ValueError(f"Gen2FilterConfig fields must be numbers: {sorted(bad)}")
        return cls(**{k: float(v) for k, v in data.items()})


@dataclass
class Gen2FrameSignals:
    """Per-RGB-frame inputs to the filter (all arrays have length T)."""

    t_ns: np.ndarray  # RGB device timestamps
    tracking_ok: np.ndarray  # bool: pose and hand samples close enough in time
    gaze_valid: np.ndarray  # bool: on-device combined eye gaze valid
    device_forward_world: np.ndarray  # (T, 3) device z axis in the world frame
    hand_min_dist_m: np.ndarray  # closest hand keypoint to the device, nan if none
    left_tracked: np.ndarray  # bool
    right_tracked: np.ndarray  # bool
    # False when the recording has no on-device eye gaze: resets (glasses off
    # the face) cannot be detected, so demos are not split at resets.
    gaze_available: bool = True


# Per-frame drop reasons, most specific first (used for reporting).
REASONS = (
    "not_tracking",
    "reset",
    "calibration",
    "gaze_dropout",
    "head_sweep",
    "hand_at_headset",
    "no_hands",
    "left_hand_not_tracked",
    "right_hand_not_tracked",
)
BOUNDARY_REASONS = ("reset", "calibration")


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) index pairs of True runs."""
    edges = np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _dilate(mask: np.ndarray, t_s: np.ndarray, pad_s: float) -> np.ndarray:
    out = mask.copy()
    for s, e in runs(mask):
        out |= (t_s >= t_s[s] - pad_s) & (t_s <= t_s[e - 1] + pad_s)
    return out


def _seconds(t_ns: np.ndarray) -> np.ndarray:
    t_ns = np.asarray(t_ns)
    return (t_ns - t_ns[0]) / 1e9


def _frame_dt_s(t_s: np.ndarray) -> float:
    return float(np.median(np.diff(t_s))) if len(t_s) > 1 else 1 / 30


def _run_s(s: int, e: int, dt: float) -> float:
    """Duration of frames [s, e): every frame covers one frame period."""
    return (e - s) * dt


def head_angular_speed(
    t_ns: np.ndarray, device_forward_world: np.ndarray, window_s: float
) -> np.ndarray:
    """Head angular speed (deg/s) of the device forward axis, window-averaged."""
    t_s = _seconds(t_ns)
    T = len(t_s)
    if T < 2:
        return np.zeros(T)
    fwd = np.asarray(device_forward_world, float)
    norm = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd = fwd / np.where(norm > 0, norm, 1)
    cos = np.clip(np.sum(fwd[1:] * fwd[:-1], axis=1), -1, 1)
    dt = np.maximum(np.diff(t_s), 1e-6)
    speed = np.r_[0.0, np.degrees(np.arccos(cos)) / dt]
    k = min(T, max(1, int(round(window_s / np.median(dt)))))
    return np.convolve(speed, np.ones(k) / k, mode="same")


def _opening_calibration(
    raw_sweep: np.ndarray, t_s: np.ndarray, cfg: Gen2FilterConfig
) -> np.ndarray:
    """Frames up to the end of the opening chain of head sweeps (see config).

    raw_sweep is the unpadded fast-turn mask; the calibration ends
    head_sweep_pad_s after the chain's last sweep.
    """
    dt = _frame_dt_s(t_s)

    def qualifies(chain):
        return (
            chain is not None
            and chain["swept"] >= cfg.calib_min_sweep_s
            and chain["longest"] >= cfg.calib_min_run_s
        )

    chain, restarted = None, False
    for s, e in runs(raw_sweep):
        if t_s[s] > cfg.calib_window_s:
            break
        if chain is not None and t_s[s] - t_s[chain["end"]] > cfg.calib_max_gap_s:
            if qualifies(chain):
                break
            if restarted or chain["swept"] > cfg.calib_glance_max_s:
                return np.zeros(len(t_s), bool)  # not a calibration
            chain, restarted = None, True  # a stray early glance: try once more
        if chain is None:
            start_limit = cfg.calib_restart_start_s if restarted else cfg.calib_start_s
            if t_s[s] > start_limit:
                break
            chain = {"end": e - 1, "swept": 0.0, "longest": 0.0}
        chain["end"] = e - 1
        chain["swept"] += _run_s(s, e, dt)
        chain["longest"] = max(chain["longest"], _run_s(s, e, dt))
    if not qualifies(chain):
        return np.zeros(len(t_s), bool)
    return t_s <= t_s[chain["end"]] + cfg.head_sweep_pad_s


def classify_frames(
    signals: Gen2FrameSignals, arm: str, cfg: Gen2FilterConfig
) -> dict[str, np.ndarray]:
    """Per-frame boolean masks for every rule in REASONS, plus:

    boundary : reset or opening calibration; splits demos, always dropped
    error    : any other rule (for the hands the arm needs), outside boundaries
    """
    t_s = _seconds(signals.t_ns)
    T = len(t_s)

    dt = _frame_dt_s(t_s)
    reset = np.zeros(T, bool)
    gaze_dropout = np.zeros(T, bool)
    for s, e in runs(~np.asarray(signals.gaze_valid, bool)):
        length = _run_s(s, e, dt)
        if length >= cfg.reset_min_s:
            reset[s:e] = True
        elif length >= cfg.gaze_dropout_min_s:
            gaze_dropout[s:e] = True
    reset = _dilate(reset, t_s, cfg.reset_pad_s)

    speed = head_angular_speed(
        signals.t_ns, signals.device_forward_world, cfg.head_sweep_window_s
    )
    raw_sweep = speed > cfg.head_sweep_deg_s
    sweep = _dilate(raw_sweep, t_s, cfg.head_sweep_pad_s)
    calibration = _opening_calibration(raw_sweep, t_s, cfg)

    near = np.nan_to_num(signals.hand_min_dist_m, nan=np.inf) < cfg.hand_at_headset_m
    hand_at_headset = _dilate(near, t_s, cfg.hand_at_headset_pad_s)

    need_left = arm in ("left", "both")
    need_right = arm in ("right", "both")
    no_left = ~np.asarray(signals.left_tracked, bool)
    no_right = ~np.asarray(signals.right_tracked, bool)
    masks = {
        "not_tracking": ~np.asarray(signals.tracking_ok, bool),
        "reset": reset,
        "calibration": calibration,
        "gaze_dropout": gaze_dropout,
        "head_sweep": sweep,
        "hand_at_headset": hand_at_headset,
        "no_hands": no_left & no_right & (need_left or need_right),
        "left_hand_not_tracked": no_left & need_left,
        "right_hand_not_tracked": no_right & need_right,
    }
    boundary = reset | calibration
    error = np.zeros(T, bool)
    for name in REASONS:
        if name not in BOUNDARY_REASONS:
            error |= masks[name]
    masks["boundary"] = boundary
    masks["error"] = error & ~boundary
    return masks


def first_reason(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Index into REASONS of the first rule that drops each frame, -1 if kept."""
    label = np.full(len(masks["boundary"]), -1)
    for k in reversed(range(len(REASONS))):
        label[masks[REASONS[k]]] = k
    return label


def demo_segments(
    t_ns: np.ndarray,
    boundary: np.ndarray,
    min_demo_s: float,
    usable: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Split a recording into demos at the boundaries.

    A between-boundary piece becomes a demo only if its usable frames (all of
    them if usable is None) add up to at least min_demo_s.

    Returns
    -------
    demo_index : (T,) int, demo number per frame, -1 outside any demo
    demos : (start, end_exclusive) frame ranges
    removed : between-boundary pieces with too little usable time
    """
    t_s = _seconds(t_ns)
    dt = _frame_dt_s(t_s)
    usable = np.ones(len(t_s), bool) if usable is None else np.asarray(usable, bool)
    demo_index = np.full(len(t_s), -1)
    demos, removed = [], []
    for s, e in runs(~np.asarray(boundary, bool)):
        if usable[s:e].sum() * dt < min_demo_s:
            removed.append((s, e))
            continue
        demo_index[s:e] = len(demos)
        demos.append((s, e))
    return demo_index, demos, removed
