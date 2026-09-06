"""E1 (mixed-speed protocol) variants of the bimanual arc-length tokenizer.

Additive subclass of ``TokenizeBimanualArcLengthCartesian`` — nothing upstream
changes. Three things over the parent:

* **Vectorized tokenize / detokenize with the parent's exact semantics.** The
  parent resamples rotation with one scipy ``Slerp`` object per waypoint
  (≈ 200 Rotation/Slerp constructions per sample), which measured 1 s per
  sample in loader workers — 25 min per 100-batch epoch. Here the bracketing
  is one ``searchsorted``, positions / grippers are one linear blend, and
  rotation is one batched ``rot_i * exp(alpha · log(rot_i⁻¹ rot_{i+1}))`` —
  the same geodesic scipy's ``Slerp`` walks. ``e1_data_smoke.py --check-parent``
  asserts agreement with the parent to float precision.

* ``velocity_norm="path"`` — the trailing velocity row keeps the chord
  direction but its magnitude becomes the token's PATH speed (arc length the
  token covers / time it took) instead of chord / time. ``detokenize`` walks the
  waypoint polyline at ``||vel||``, so the upstream chord-norm token runs slow on
  any curved motion (Step 0 on mecka fold chunks: token speed / measured path
  speed = 0.23, d_clock 2.8 s median). "path" is the protocol's Arc-mean.

* ``velocity_mode="profile"`` — Arc+Vel. No velocity row; every waypoint row
  carries the per-arm speed at that arc-length position as two extra columns:
  ``(M, 16) = [14 canonical | v_L(u_m), v_R(u_m)]``. ``detokenize`` integrates
  the clock, t(u) = ∫ du / v(u) with v piecewise linear between waypoints.
  The waypoint speeds are the raw 30 Hz chunk's path speed (7-frame moving
  average — see ``chunk_speed`` for why not a Butterworth here).

* ``velocity_mode="logdur"`` — the tempo-invariance ablation's channel
  (Ideas note *Arc Tokenizer Tempo Invariance Changes*, #1 + #2). Same
  ``(M, 16)`` layout, but per arm the extra column holds, in row 0, the log of
  the token's mean slowness ``log(T_span / span)`` (s/m — the one scalar that
  carries tempo) and, in rows 1..M-1, the log of each waypoint segment's
  duration relative to the mean segment duration (a tempo-normalized profile
  that sums to the token's time by construction). ``detokenize`` rebuilds the
  clock as a cumulative sum of durations — no division by a near-zero speed,
  and a timing error is a *relative* error at any tempo.

* ``progress_smooth_hz`` — (#4) low-pass the chunk's positions (zero-phase
  4th-order Butterworth, the Step 0 filter) before cumulative arc length is
  accumulated, so ``D`` metres of *measured* progress is the same true path at
  every tempo (Thm 3.7: the raw polyline length of a jittery track grows with
  samples per metre, i.e. with slowness). Waypoint 0 stays the raw anchor.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    cumulative_arc_length,
)

try:  # the parent fills chunks with out-of-range poses; mirror that rule
    from egomimic.rldb.zarr.arc_length_tokenizer import INVALID_POSE_FILL, INVALID_POSE_THRESHOLD
except ImportError:  # pragma: no cover
    INVALID_POSE_THRESHOLD, INVALID_POSE_FILL = np.inf, 0.0

# (xyz offset, ypr offset, grip offset, velocity-row xyz slice) per arm in the
# canonical 14-dim layout [L xyz ypr grip | R xyz ypr grip].
ARM_LAYOUT = ((0, 3, 6, slice(0, 3)), (7, 10, 13, slice(7, 10)))
E1_ARCVEL_DIM = 16


# ---------------------------------------------------------------------------
# vectorized arc-length resampling (parent semantics)
# ---------------------------------------------------------------------------
def _bracket_vec(cum: np.ndarray, targets: np.ndarray):
    """Vectorized ``_bracket_segment``: segment index i and alpha per target."""
    n = len(cum)
    i = np.searchsorted(cum, targets, side="left") - 1
    i = np.clip(i, 0, max(n - 2, 0))
    s0, s1 = cum[i], cum[np.minimum(i + 1, n - 1)]
    span = s1 - s0
    alpha = np.where(span > 1e-12, (targets - s0) / np.where(span > 1e-12, span, 1.0), 0.0)
    lo, hi = targets <= cum[0], targets >= cum[-1]
    i = np.where(lo, 0, np.where(hi, max(n - 2, 0), i))
    alpha = np.where(lo, 0.0, np.where(hi, 1.0, alpha))
    return i, np.clip(alpha, 0.0, 1.0)


def _slerp_vec(rot: R, i: np.ndarray, alpha: np.ndarray) -> R:
    """Geodesic interpolation between rot[i] and rot[i+1] at alpha (what scipy's Slerp does)."""
    n = len(rot)
    j = np.minimum(i + 1, n - 1)
    r0, r1 = rot[i], rot[j]
    rel = r0.inv() * r1
    return r0 * R.from_rotvec(rel.as_rotvec() * alpha[:, None])


def resample_at_s(pos, ypr, grip, cum, targets, rot: R | None = None):
    """Interpolate (pos, ypr, grip) at arc lengths ``targets`` against ``cum``.

    Same edge rules as the parent's ``_interp_*_at_s``: below cum[0] → first
    row, at/above cum[-1] → last row, alpha == 0 → the bracketing row itself.
    """
    i, alpha = _bracket_vec(cum, targets)
    j = np.minimum(i + 1, len(cum) - 1)
    a = alpha[:, None]
    pos_out = (1.0 - a) * pos[i] + a * pos[j]
    grip_out = (1.0 - a) * grip[i] + a * grip[j]
    rot = R.from_euler("ZYX", ypr) if rot is None else rot
    ypr_out = _slerp_vec(rot, i, alpha).as_euler("ZYX", degrees=False)
    exact0, exact1 = alpha <= 0.0, alpha >= 1.0
    ypr_out[exact0] = ypr[i[exact0]]
    ypr_out[exact1] = ypr[j[exact1]]
    return pos_out, ypr_out, grip_out


def chunk_speed(pos: np.ndarray, dt: float, smooth_frames: int = 7) -> np.ndarray:
    """Per-frame PATH speed of a (T, 3) chunk: d(arc length)/dt, smoothed with a
    centred ``smooth_frames`` moving average (edge-padded).

    Deliberately not an IIR low-pass: a zero-phase Butterworth on a 200-frame
    chunk has a ~10-frame transient at the anchor, exactly where the token's
    first waypoints (and E_time) live. Path speed rather than chord speed so the
    integral clock traverses the token's own arc length — jitter included — at
    the rate it was actually traversed.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if len(pos) < 2:
        return np.zeros(len(pos))
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    v = np.concatenate([step, step[-1:]])
    w = max(1, int(smooth_frames))
    if w > 1 and len(v) >= w:
        vp = np.pad(v, (w // 2, w - 1 - w // 2), mode="edge")
        v = np.convolve(vp, np.ones(w) / w, mode="valid")
    return v


def integral_clock(cum: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """Time-of-progress at the waypoints: t(u_m) = ∫_0^{u_m} du / v(u)."""
    v = np.asarray(speed, dtype=np.float64)
    seg = np.diff(cum)
    dtime = seg * 0.5 * (1.0 / v[:-1] + 1.0 / v[1:])
    return np.concatenate(([0.0], np.cumsum(dtime)))


def lowpass_positions(pos: np.ndarray, fc_hz: float, fs_hz: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass of a (T, 3) track (Step 0's filter).
    Chunks too short for the filter's padding are returned unchanged."""
    pos = np.asarray(pos, dtype=np.float64)
    if fc_hz is None or fc_hz <= 0 or len(pos) < 20:
        return pos
    sos = butter(4, float(fc_hz), btype="low", fs=float(fs_hz), output="sos")
    return sosfiltfilt(sos, pos, axis=0)


LOGDUR_CLIP = np.log(20.0)  # |log(segment duration / mean segment duration)| cap
LOGDUR_MIN_DT = 1e-4        # s, floor on a segment duration at tokenize time


def durations_to_clock(col: np.ndarray, span: float, min_speed: float = 0.01, max_speed: float = 5.0) -> np.ndarray:
    """logdur column (M,) + polyline span (m) -> time-of-progress at the M waypoints (s).

    The decoded mean speed is bounded to [min_speed, max_speed] — the same
    floor the profile mode's integral clock uses — so a wild row-0 prediction
    cannot make one chunk's clock dominate d_clock (an untrained head gave
    10^5 s in the launcher smoke). Tokenized values sit well inside the range
    (fold data: 0.10–0.42 m/s), so training targets are never clipped.
    """
    col = np.asarray(col, dtype=np.float64)
    M = len(col)
    t_span = max(float(span), 1e-9) * np.exp(np.clip(col[0], -np.log(max_speed), -np.log(min_speed)))
    seg = np.exp(np.clip(col[1:], -LOGDUR_CLIP, LOGDUR_CLIP)) * (t_span / max(M - 1, 1))
    return np.concatenate(([0.0], np.cumsum(seg)))


class CopyKeyRows(Transform):
    """``batch[dst] = batch[src][:n_rows]`` — carries the un-tokenized time chunk
    (``actions_time``) alongside the model target so the E1 evaluator can score
    every variant against the same 30 Hz ground truth."""

    def __init__(self, src: str, dst: str, n_rows: int):
        self.src, self.dst, self.n_rows = src, dst, int(n_rows)

    def transform(self, batch: dict) -> dict:
        x = np.asarray(batch[self.src], dtype=np.float64)
        batch[self.dst] = x[: self.n_rows].copy()
        return batch


class TokenizeBimanualArcLengthE1(TokenizeBimanualArcLengthCartesian):
    def __init__(
        self,
        *,
        velocity_norm: str = "chord",
        velocity_mode: str = "mean",
        speed_smooth_frames: int = 7,
        min_speed: float = 0.01,
        progress_smooth_hz: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if velocity_norm not in ("chord", "path"):
            raise ValueError(f"velocity_norm must be 'chord' or 'path', got {velocity_norm!r}")
        if velocity_mode not in ("mean", "profile", "logdur"):
            raise ValueError(f"velocity_mode must be 'mean', 'profile' or 'logdur', got {velocity_mode!r}")
        self.velocity_norm = velocity_norm
        self.velocity_mode = velocity_mode
        self.speed_smooth_frames = int(speed_smooth_frames)
        self.min_speed = float(min_speed)
        self.progress_smooth_hz = None if progress_smooth_hz in (None, 0, 0.0) else float(progress_smooth_hz)

    @property
    def wide(self) -> bool:
        """(M, 16) layouts: a per-waypoint timing column per arm."""
        return self.velocity_mode in ("profile", "logdur")

    def _progress_positions(self, pos: np.ndarray) -> np.ndarray:
        """Positions used for arc length / resampling / timing (#4 when smoothing is on)."""
        if self.progress_smooth_hz is None:
            return np.asarray(pos, dtype=np.float64)
        return lowpass_positions(pos, self.progress_smooth_hz, 1.0 / self.tokenizer.config.dt)

    # -- one arm, parent semantics of ArcLengthTokenizer.tokenize_at(t=0) -----
    def _tokenize_arm(self, arm: np.ndarray):
        """arm: (T, 7) [xyz ypr grip] → (waypoints (M, 7), mean velocity (3,), num_steps, span)."""
        cfg = self.tokenizer.config
        M, D, dt = self.M, cfg.min_distance_unit, cfg.dt
        pos_raw, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
        pos = self._progress_positions(pos_raw)
        n = len(pos)
        cum = cumulative_arc_length(pos)
        end_s = min(D, float(cum[-1]))
        end_idx = int(np.searchsorted(cum, end_s, side="right") - 1)
        end_idx = max(0, min(end_idx, n - 1))
        num_steps = max(1, end_idx)
        alphas = np.linspace(0.0, 1.0, M)
        if end_s < cfg.zero_dist_epsilon or num_steps > cfg.max_steps_per_chunk:
            # zero token: hold the pose, slerp rotation to the end frame, ramp the gripper
            end = min(end_idx, n - 1)
            rot = R.from_euler("ZYX", np.stack([ypr[0], ypr[end]]))
            ypr_rs = _slerp_vec(rot, np.zeros(M, dtype=int), alphas).as_euler("ZYX", degrees=False)
            ypr_rs[0] = ypr[0]
            ypr_rs[-1] = ypr[end]
            wp = np.concatenate([np.repeat(pos_raw[:1], M, 0), ypr_rs, (1 - alphas[:, None]) * grip[0] + alphas[:, None] * grip[end]], axis=1)
            return wp, np.zeros(3), num_steps, end_s
        targets = np.linspace(0.0, end_s, M)
        pos_rs, ypr_rs, grip_rs = resample_at_s(pos, ypr, grip, cum, targets)
        pos_rs[0], ypr_rs[0], grip_rs[0] = pos_raw[0], ypr[0], grip[0]  # start_idx=0 anchoring (raw)
        vel = (pos_rs[-1] - pos_rs[0]) / max(num_steps * dt, 1e-8)  # MEAN_PER_DIM
        return np.concatenate([pos_rs, ypr_rs, grip_rs], axis=1), vel, num_steps, end_s

    # -- tokenize ----------------------------------------------------------
    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.action_key], dtype=np.float64)
        M = self.M
        dt = self.tokenizer.config.dt
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise ValueError(f"expected (T, 14) chunk, got {chunk.shape}")
        if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
            rows, cols = (M, E1_ARCVEL_DIM) if self.wide else (M + 1, 14)
            batch[self.output_action_key] = np.full((rows, cols), INVALID_POSE_FILL, dtype=np.float64)
            return batch

        wps, vels, spans = [], [], []
        for xyz_off, _, _, _ in ARM_LAYOUT:
            wp, vel, _, span = self._tokenize_arm(chunk[:, xyz_off : xyz_off + 7])
            wps.append(wp)
            vels.append(vel)
            spans.append(span)
        waypoints = np.concatenate(wps, axis=1)  # (M, 14)

        if self.wide:
            prof = np.zeros((M, 2), dtype=np.float64)
            for k, (xyz_off, _, _, _) in enumerate(ARM_LAYOUT):
                pos = self._progress_positions(chunk[:, xyz_off : xyz_off + 3])
                cum = cumulative_arc_length(pos)
                span = spans[k]
                if span <= 1e-8 or len(pos) < 2:
                    if self.velocity_mode == "logdur":
                        prof[0, k] = -np.log(self.min_speed)  # stationary arm: slowness at the floor speed
                    continue
                u = np.linspace(0.0, span, M)
                fidx = np.interp(u, cum, np.arange(len(cum)))
                if self.velocity_mode == "profile":
                    v = chunk_speed(pos, dt, self.speed_smooth_frames)
                    prof[:, k] = np.maximum(np.interp(fidx, np.arange(len(v)), v), 0.0)
                else:  # logdur: row 0 = log mean slowness, rows 1.. = log relative segment durations
                    seg = np.maximum(np.diff(fidx) * dt, LOGDUR_MIN_DT)
                    t_span = float(seg.sum())
                    prof[0, k] = np.log(t_span / span)
                    prof[1:, k] = np.clip(np.log(seg / (t_span / max(M - 1, 1))), -LOGDUR_CLIP, LOGDUR_CLIP)
            batch[self.output_action_key] = np.concatenate([waypoints, prof], axis=1)
            return batch

        # mean mode: the parent's (M+1, 14) layout with its velocity row
        vel_token = np.zeros(14, dtype=np.float64)
        default_dur = max(M - 1, 1) * dt
        for k, (xyz_off, ypr_off, grip_off, vsl) in enumerate(ARM_LAYOUT):
            wp = wps[k]
            vel = vels[k]
            speed = float(np.linalg.norm(vel))
            chord = float(np.linalg.norm(wp[-1, 0:3] - wp[0, 0:3]))
            dur = (chord / speed) if speed > 1e-8 else default_dur
            if self.velocity_norm == "path" and speed > 1e-8 and chord > 1e-8:
                span_wp = float(cumulative_arc_length(wp[:, 0:3])[-1])
                if span_wp > 1e-8:
                    vel = vel * ((span_wp / dur) / speed)
            vel_token[vsl] = vel
            vel_token[ypr_off : ypr_off + 3] = (wp[-1, 3:6] - wp[0, 3:6]) / max(dur, 1e-8)
            vel_token[grip_off] = float(wp[-1, 6] - wp[0, 6]) / max(dur, 1e-8)
        batch[self.output_action_key] = np.concatenate([waypoints, vel_token[None]], axis=0)
        return batch

    # -- detokenize --------------------------------------------------------
    def detokenize(self, arc_actions: np.ndarray, action_horizon: int) -> np.ndarray:
        arc = np.asarray(arc_actions, dtype=np.float64)
        h = int(action_horizon)
        dt = self.tokenizer.config.dt
        t = dt * np.arange(h, dtype=np.float64)
        profile = self.wide
        if profile:
            if arc.ndim != 2 or arc.shape[1] != E1_ARCVEL_DIM:
                raise ValueError(f"{self.velocity_mode} detokenize expects (M, {E1_ARCVEL_DIM}), got {arc.shape}")
            M = arc.shape[0]
        else:
            if arc.ndim != 2 or arc.shape[1] != 14:
                raise ValueError(f"detokenize expects (M+1, 14), got {arc.shape}")
            M = arc.shape[0] - 1
        arms = []
        for k, (xyz_off, ypr_off, grip_off, vsl) in enumerate(ARM_LAYOUT):
            xyz_wp = arc[:M, xyz_off : xyz_off + 3]
            ypr_wp = arc[:M, ypr_off : ypr_off + 3]
            grip_wp = arc[:M, grip_off : grip_off + 1]
            cum = cumulative_arc_length(xyz_wp)
            total = float(cum[-1])
            if profile:
                degenerate = total < 1e-9
                s = None if degenerate else np.interp(t, self._wide_clock(arc[:, 14 + k], cum), cum)
            else:
                speed = float(np.linalg.norm(arc[M, vsl]))
                degenerate = total < 1e-9 or speed < 1e-8
                s = None if degenerate else np.minimum(speed * t, total)
            if degenerate:
                arms.append(np.concatenate([np.repeat(xyz_wp[:1], h, 0), np.repeat(ypr_wp[:1], h, 0), np.repeat(grip_wp[:1], h, 0)], axis=-1))
                continue
            pos_t, ypr_t, grip_t = resample_at_s(xyz_wp, ypr_wp, grip_wp, cum, s)
            arms.append(np.concatenate([pos_t, ypr_t, grip_t], axis=-1))
        return np.concatenate(arms, axis=-1)  # (H, 14)

    def _wide_clock(self, col: np.ndarray, cum: np.ndarray) -> np.ndarray:
        """Time-of-progress at the M waypoints from one arm's timing column."""
        if self.velocity_mode == "logdur":
            return durations_to_clock(col, float(cum[-1]), min_speed=self.min_speed)
        return integral_clock(cum, np.maximum(col, self.min_speed))

    def clock_at_waypoints(self, arc_actions: np.ndarray) -> list[np.ndarray]:
        """Per arm, the token's implied time-of-progress at its M waypoints (s)."""
        arc = np.asarray(arc_actions, dtype=np.float64)
        M = arc.shape[0] if self.wide else arc.shape[0] - 1
        out = []
        for k, (xyz_off, _, _, vsl) in enumerate(ARM_LAYOUT):
            cum = cumulative_arc_length(arc[:M, xyz_off : xyz_off + 3])
            if self.wide:
                out.append(self._wide_clock(arc[:, 14 + k], cum))
            else:
                speed = float(np.linalg.norm(arc[M, vsl]))
                out.append(cum / max(speed, self.min_speed))
        return out
