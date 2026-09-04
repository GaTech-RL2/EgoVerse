"""E1 (mixed-speed protocol) variants of the bimanual arc-length tokenizer.

Additive subclass of ``TokenizeBimanualArcLengthCartesian`` — nothing upstream
changes. Two opt-in knobs:

* ``velocity_norm="path"`` — the trailing velocity row keeps the chord
  direction but its magnitude becomes the token's PATH speed (arc length the
  token covers / time it took) instead of chord / time. ``detokenize`` walks the
  waypoint polyline at ``||vel||``, so the upstream chord-norm token runs slow on
  any curved motion (Step 0 on mecka fold chunks: token speed / measured path
  speed = 0.23, d_clock 2.7 s median). "path" is the protocol's Arc-mean.

* ``velocity_mode="profile"`` — Arc+Vel. No velocity row; every waypoint row
  carries the per-arm speed at that arc-length position as two extra columns:
  ``(M, 16) = [14 canonical | v_L(u_m), v_R(u_m)]``. ``detokenize`` integrates
  the clock, t(u) = ∫ du / v(u) with v piecewise linear between waypoints, so a
  chunk that slows down mid-token is reconstructed slowing down.

The waypoint speeds are the raw 30 Hz chunk's path speed (7-frame moving
average — see ``chunk_speed`` for why not a Butterworth here), read at the
waypoints' fractional frame indices.
"""

from __future__ import annotations

import numpy as np

from egomimic.rldb.zarr.action_chunk_transforms import Transform
from egomimic.rldb.zarr.arc_length_tokenizer import (
    TokenizeBimanualArcLengthCartesian,
    _interp_linear_at_s,
    _interp_pos_at_s,
    _interp_ypr_at_s,
    cumulative_arc_length,
)

# (xyz offset, ypr offset, grip offset, velocity-row xyz slice) per arm in the
# canonical 14-dim layout [L xyz ypr grip | R xyz ypr grip].
ARM_LAYOUT = ((0, 3, 6, slice(0, 3)), (7, 10, 13, slice(7, 10)))
E1_ARCVEL_DIM = 16


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        if velocity_norm not in ("chord", "path"):
            raise ValueError(f"velocity_norm must be 'chord' or 'path', got {velocity_norm!r}")
        if velocity_mode not in ("mean", "profile"):
            raise ValueError(f"velocity_mode must be 'mean' or 'profile', got {velocity_mode!r}")
        self.velocity_norm = velocity_norm
        self.velocity_mode = velocity_mode
        self.speed_smooth_frames = int(speed_smooth_frames)
        self.min_speed = float(min_speed)

    # -- tokenize ----------------------------------------------------------
    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.action_key], dtype=np.float64)
        tok = super().transform({self.action_key: chunk.copy()})[self.output_action_key]  # (M+1, 14)
        M = self.M
        dt = self.tokenizer.config.dt
        D = self.tokenizer.config.min_distance_unit

        if self.velocity_mode == "mean":
            if self.velocity_norm == "path":
                for xyz_off, _, _, vsl in ARM_LAYOUT:
                    wp = tok[:M, xyz_off : xyz_off + 3]
                    vel = tok[M, vsl]
                    speed = float(np.linalg.norm(vel))
                    chord = float(np.linalg.norm(wp[-1] - wp[0]))
                    span = float(cumulative_arc_length(wp)[-1])
                    if speed > 1e-8 and chord > 1e-8 and span > 1e-8:
                        duration = chord / speed  # what the upstream row encodes
                        tok[M, vsl] = vel * ((span / duration) / speed)
            batch[self.output_action_key] = tok
            return batch

        # profile mode: (M, 16) — waypoints + per-arm speed at each waypoint
        prof = np.zeros((M, 2), dtype=np.float64)
        for k, (xyz_off, _, _, _) in enumerate(ARM_LAYOUT):
            pos = chunk[:, xyz_off : xyz_off + 3]
            cum = cumulative_arc_length(pos)
            span = min(D, float(cum[-1]))
            if span <= 1e-8 or len(pos) < 2:
                continue  # stationary arm: zero speed, waypoints hold the pose
            u = np.linspace(0.0, span, M)
            fidx = np.interp(u, cum, np.arange(len(cum)))
            v = chunk_speed(pos, dt, self.speed_smooth_frames)
            prof[:, k] = np.maximum(np.interp(fidx, np.arange(len(v)), v), 0.0)
        batch[self.output_action_key] = np.concatenate([tok[:M], prof], axis=1)
        return batch

    # -- detokenize --------------------------------------------------------
    def detokenize(self, arc_actions: np.ndarray, action_horizon: int) -> np.ndarray:
        arc = np.asarray(arc_actions, dtype=np.float64)
        if self.velocity_mode == "mean":
            return super().detokenize(arc, action_horizon)
        if arc.ndim != 2 or arc.shape[1] != E1_ARCVEL_DIM:
            raise ValueError(f"profile detokenize expects (M, {E1_ARCVEL_DIM}), got {arc.shape}")
        dt = self.tokenizer.config.dt
        h = int(action_horizon)
        t = dt * np.arange(h, dtype=np.float64)
        arms = []
        for k, (xyz_off, ypr_off, grip_off, _) in enumerate(ARM_LAYOUT):
            xyz_wp = arc[:, xyz_off : xyz_off + 3]
            ypr_wp = arc[:, ypr_off : ypr_off + 3]
            grip_wp = arc[:, grip_off : grip_off + 1]
            cum = cumulative_arc_length(xyz_wp)
            if float(cum[-1]) < 1e-9:
                arms.append(np.concatenate([np.repeat(xyz_wp[:1], h, 0), np.repeat(ypr_wp[:1], h, 0), np.repeat(grip_wp[:1], h, 0)], axis=-1))
                continue
            t_of_u = integral_clock(cum, np.maximum(arc[:, 14 + k], self.min_speed))
            s = np.interp(t, t_of_u, cum)  # clamps at the last waypoint once the token is exhausted
            pos_t = np.stack([_interp_pos_at_s(xyz_wp, cum, float(sk)) for sk in s])
            ypr_t = np.stack([_interp_ypr_at_s(ypr_wp, cum, float(sk)) for sk in s])
            grip_t = np.stack([_interp_linear_at_s(grip_wp, cum, float(sk)) for sk in s])
            arms.append(np.concatenate([pos_t, ypr_t, grip_t], axis=-1))
        return np.concatenate(arms, axis=-1)  # (H, 14)

    def clock_at_waypoints(self, arc_actions: np.ndarray) -> list[np.ndarray]:
        """Per arm, the token's implied time-of-progress at its M waypoints (s)."""
        arc = np.asarray(arc_actions, dtype=np.float64)
        M = self.M
        out = []
        for k, (xyz_off, _, _, vsl) in enumerate(ARM_LAYOUT):
            cum = cumulative_arc_length(arc[:M, xyz_off : xyz_off + 3])
            if self.velocity_mode == "profile":
                out.append(integral_clock(cum, np.maximum(arc[:, 14 + k], self.min_speed)))
            else:
                speed = float(np.linalg.norm(arc[M, vsl]))
                out.append(cum / max(speed, self.min_speed))
        return out
