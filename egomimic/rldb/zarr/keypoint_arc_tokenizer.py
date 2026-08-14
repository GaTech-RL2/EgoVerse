"""Arc-length tokenizer for 21-keypoint hand pose + wrist pose.

Generalizes ``arc_length_tokenizer.TokenizeBimanualArcLengthCartesian`` from a
single wrist point to the full hand. The cartesian version parameterizes a
chunk by the wrist's translational arc length; this one parameterizes by a
*joint* distance over all 21 MANO keypoints, so hand-shape change (grasp
closure, finger articulation) advances progress even when the wrist barely
moves.

Distance modes
--------------
Per frame-pair the per-keypoint displacement vector is
``d = (||Δp_1||, ..., ||Δp_21||)``; the scalar step is a norm of it:

- ``linf``  : ``max_i d_i``   — token advances once ANY joint has moved D, which
  bounds per-joint resampling error by D. Measured best on the tail (worst-case
  reconstruction error roughly halved vs wrist-only at equal token budget).
- ``l2``    : ``||d||_2``
- ``l1mean``: ``mean_i d_i`` — the ``sum`` form of the E_pose metric, divided by
  N. The raw sum inflates path length ~22x on real data (joints move
  coherently), which would need D~6.8m to emit a comparable token count and
  makes D uninterpretable; the mean keeps D in metres.

An optional rotation term ``lambda * mu(R_t, R_{t+1})`` on the wrist is added
when ``lambda_rot > 0``, where

    mu(R, Rh) = 0.5 * || (R^T Rh)^{1/2} - I ||_F  =  sqrt(2) * sin(theta/4)

theta being the geodesic angle. That is a true metric on SO(3) (verified: 0
triangle-inequality violations over 3000 random triples), so summing it with a
translational metric still yields a valid arc length. It saturates at theta=pi
but frame-to-frame angles at 30Hz are small, so it is used in its linear
regime. ``lambda_rot = 2*sqrt(2)*r`` makes the term equal the arc traced by a
point at radius ``r`` from the wrist, i.e. lambda is "how far from the pivot do
I care" rather than a free hyperparameter. Defaults to 0 because on folding
data the rotation term measured as a wash.

Choosing D and M
----------------
Reconstruction error depends ONLY on waypoint spacing ``h = D / (M - 1)`` --
verified to three decimals across D in {0.13, 0.26, 0.52, 1.04} at fixed h. So
D sets the horizon, h sets fidelity, and M follows. Measured mean per-keypoint
error vs h: 30mm -> 2.33mm, 20mm -> 1.85mm, 15mm -> 1.53mm, 10mm -> 1.15mm.
Below ~1.2mm you are fitting tracking jitter rather than signal (low-pass
filtering at 167ms removes 3.2x of the error), so h ~ 15mm is the knee.
"""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 21
KP_DIM = NUM_KEYPOINTS * 3  # 63
# aria has no gripper and nothing pads one on this path, so the wrist is
# xyz + ypr only. Confirmed against the live pipeline: actions_keypoints is
# (100, 138) = 2 * (63 + 6).
WRIST_DIM = 6  # xyz + ypr
PER_HAND_DIM = WRIST_DIM + KP_DIM  # 69
BIMANUAL_DIM = 2 * PER_HAND_DIM  # 138

DISTANCE_MODES = ("linf", "l2", "l1mean")


def _rot_metric_from_ypr(ypr: np.ndarray) -> np.ndarray:
    """Per-step mu(R_t, R_{t+1}) for a (T, 3) ypr sequence -> (T-1,).

    Uses the closed form sqrt(2)*sin(theta/4) rather than a matrix square root:
    identical to 5 decimals and far cheaper. theta is recovered from the
    relative rotation's trace.
    """
    from scipy.spatial.transform import Rotation as Rot

    R = Rot.from_euler("ZYX", np.asarray(ypr, dtype=np.float64)).as_matrix()
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])  # R_t^T R_{t+1}
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos)
    return np.sqrt(2.0) * np.sin(theta / 4.0)


def keypoint_step_distance(
    kp: np.ndarray,
    mode: str = "linf",
    weights: np.ndarray | None = None,
    ypr: np.ndarray | None = None,
    lambda_rot: float = 0.0,
) -> np.ndarray:
    """(T, 21, 3) keypoints -> (T-1,) non-negative per-step distance."""
    kp = np.asarray(kp, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected (T, {NUM_KEYPOINTS}, 3), got {kp.shape}")
    d = np.linalg.norm(np.diff(kp, axis=0), axis=2)  # (T-1, 21)
    if weights is not None:
        d = d * np.asarray(weights, dtype=np.float64)[None, :]
    if mode == "linf":
        step = d.max(axis=1)
    elif mode == "l2":
        step = np.linalg.norm(d, axis=1)
    elif mode == "l1mean":
        step = d.mean(axis=1)
    else:
        raise ValueError(f"mode must be one of {DISTANCE_MODES}, got {mode!r}")
    if lambda_rot > 0.0 and ypr is not None:
        step = step + float(lambda_rot) * _rot_metric_from_ypr(ypr)
    return step


def cumulative_arc_length_3d(pos: np.ndarray) -> np.ndarray:
    """(T, 3) -> (T,) cumulative translational arc length for one 3D point."""
    pos = np.asarray(pos, dtype=np.float64)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    return np.concatenate([np.array([0.0]), np.cumsum(step)])


def cumulative_keypoint_distance(kp: np.ndarray, **kwargs) -> np.ndarray:
    """(T, 21, 3) -> (T,) cumulative joint distance, starting at 0."""
    step = keypoint_step_distance(kp, **kwargs)
    return np.concatenate([np.array([0.0]), np.cumsum(step)])


def _interp_rows(x: np.ndarray, cum: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Linearly interpolate rows of x (T, ...) at arc-length positions."""
    idx = np.searchsorted(cum, targets, side="left").clip(1, len(cum) - 1)
    lo, hi = idx - 1, idx
    span = cum[hi] - cum[lo]
    span = np.where(span <= 0, 1e-12, span)
    w = ((targets - cum[lo]) / span).clip(0.0, 1.0)
    wshape = (len(targets),) + (1,) * (x.ndim - 1)
    return x[lo] * (1.0 - w.reshape(wshape)) + x[hi] * w.reshape(wshape)


class TokenizeBimanualArcLengthKeypoints:
    """(T, 140) bimanual keypoint+wrist chunk -> (M+1, 140) arc token.

    Layout per row, matching what the keypoints pipeline emits:
        [L wrist xyz ypr(6) | L kp(63) | R wrist xyz ypr(6) | R kp(63)]

    Verified by bone-length test on the live pipeline: this slicing yields
    39.7mm bones with 0.38mm variation across a 100-frame chunk, whereas
    keypoints-first slicing yields 330mm "bones" with 3.7m outliers.

    Row M is the velocity token: per hand, the mean progress rate ds/dt over the
    token span, broadcast into that hand's first slot and zero elsewhere. One
    scalar rate per hand is sufficient because the parameterization is a single
    scalar -- detokenize recovers duration as D / rate, exactly as the cartesian
    tokenizer does.

    Each hand keeps its OWN cumulative distance, matching the cartesian
    tokenizer's per-arm independence. Going further (per-joint independent
    parameterization) would decouple waypoint index from time and desynchronize
    finger coordination, which is the signal in hand data.
    """

    def __init__(
        self,
        action_key: str = "actions_keypoints",
        output_action_key: str = "actions_keypoints",
        min_distance_unit: float = 0.30,
        resampled_vector_length: int = 21,
        dt: float = 1.0 / 30.0,
        distance_mode: str = "linf",
        lambda_rot: float = 0.0,
        fingertip_weight: float = 1.0,
    ):
        if distance_mode not in DISTANCE_MODES:
            raise ValueError(f"distance_mode must be one of {DISTANCE_MODES}")
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.D = float(min_distance_unit)
        self.M = int(resampled_vector_length)
        self.dt = float(dt)
        self.distance_mode = distance_mode
        self.lambda_rot = float(lambda_rot)
        # MANO fingertip indices; upweighting them biases waypoint placement
        # toward the joints that carry contact information.
        self.weights = np.ones(NUM_KEYPOINTS)
        if fingertip_weight != 1.0:
            self.weights[[4, 8, 12, 16, 20]] = float(fingertip_weight)

    def _hand_slice(self, hand: int) -> tuple[slice, slice]:
        """-> (keypoint slice, wrist slice) for this hand. Wrist comes FIRST."""
        off = hand * PER_HAND_DIM
        return (
            slice(off + WRIST_DIM, off + PER_HAND_DIM),  # 63 keypoint dims
            slice(off, off + WRIST_DIM),  # 6 wrist dims
        )

    def tokenize(self, chunk: np.ndarray) -> np.ndarray:
        """(T, 140) -> (M+1, 140)."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != BIMANUAL_DIM:
            raise ValueError(f"expected (T, {BIMANUAL_DIM}), got {chunk.shape}")
        T = len(chunk)
        out = np.zeros((self.M + 1, BIMANUAL_DIM), dtype=np.float64)
        for hand in range(2):
            kp_sl, wr_sl = self._hand_slice(hand)
            kp = chunk[:, kp_sl].reshape(T, NUM_KEYPOINTS, 3)
            wrist = chunk[:, wr_sl]
            cum = cumulative_keypoint_distance(
                kp,
                mode=self.distance_mode,
                weights=self.weights,
                ypr=wrist[:, 3:6],
                lambda_rot=self.lambda_rot,
            )
            end_s = float(min(self.D, cum[-1]))
            targets = np.linspace(0.0, end_s, self.M)
            out[: self.M, kp_sl] = _interp_rows(
                kp.reshape(T, -1), cum, targets
            )
            out[: self.M, wr_sl] = _interp_rows(wrist, cum, targets)
            # Velocity token: mean progress rate over the span the token covers.
            # Degenerate (stationary) hands get 0, and detokenize falls back to
            # holding the first waypoint, mirroring the cartesian tokenizer.
            n_steps = max(
                int(np.searchsorted(cum, end_s, side="right")) - 1, 1
            )
            duration = n_steps * self.dt
            out[self.M, wr_sl.start] = end_s / duration if duration > 0 else 0.0
        return out

    def detokenize(self, token: np.ndarray, action_horizon: int) -> np.ndarray:
        """(M+1, 140) -> (H, 140) time-parameterized chunk at the control rate."""
        token = np.asarray(token, dtype=np.float64)
        H = int(action_horizon)
        out = np.zeros((H, BIMANUAL_DIM), dtype=np.float64)
        for hand in range(2):
            kp_sl, wr_sl = self._hand_slice(hand)
            wps = token[: self.M]
            rate = float(token[self.M, wr_sl.start])
            if rate <= 1e-9:
                out[:, kp_sl] = wps[0, kp_sl]
                out[:, wr_sl] = wps[0, wr_sl]
                continue
            duration = self.D / rate
            # Uniform in arc length over the span the horizon actually covers.
            span = min(duration, H * self.dt)
            s_grid = np.linspace(0.0, self.D * (span / duration), H)
            wp_s = np.linspace(0.0, self.D, self.M)
            out[:, kp_sl] = _interp_rows(wps[:, kp_sl], wp_s, s_grid)
            out[:, wr_sl] = _interp_rows(wps[:, wr_sl], wp_s, s_grid)
        return out

    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.action_key])
        if chunk.ndim == 3:  # batched
            batch[self.output_action_key] = np.stack(
                [self.tokenize(c) for c in chunk]
            )
        else:
            batch[self.output_action_key] = self.tokenize(chunk)
        return batch


# ---------------------------------------------------------------------------
# Hybrid two-stream arc tokenizer
# ---------------------------------------------------------------------------

# Per hand the 138-dim layout splits into two SEMANTIC streams:
#   eef       cols o..o+6   wrist xyz + ypr, in head frame -> global motion
#   keypoints cols o+6..o+69  21 MANO joints, in WRIST frame -> hand shape only
# Because the keypoints are wrist-relative, the wrist's motion is already
# removed from them, so the two streams measure genuinely independent things
# and each deserves its own D and its own distance function.
STREAM_EEF, STREAM_KP = "eef", "keypoints"


class TokenizeBimanualHybridArcKeypoints:
    """Two-stream arc tokenizer: eef and keypoints parameterised independently.

    Each stream accumulates its own arc length with its own distance function
    and its own ``D``, is resampled to the SAME M waypoints, and carries its own
    velocity. The token stays (M+1, 138); row M holds four velocities (2 hands x
    2 streams) written into each stream's first column.

    Independent parameterisation would normally desynchronise the streams --
    waypoint index k means a different time for eef than for keypoints. The
    rollout assumption fixes it: only the window over which EVERY stream is
    still inside its token is emitted,

        duration_s = D_s / v_s        per stream
        H_valid    = min_s duration_s

    so within [0, H_valid] every stream interpolates at s(t) = v_s * t <= D_s,
    i.e. inside its own waypoints, and the reconstruction stays time-consistent.

    Choosing the two D values: they should be set so the streams emit
    comparable token counts, otherwise one stream's tokens are chronically
    truncated by the other's. Measured on a real 94s episode, total path is
    48.71m for eef and 44.51m for wrist-frame keypoint L-inf, so
    ``D_kp ~ 0.91 * D_eef``.

    M is free and can be denser than the scalar variant: reconstruction error
    depends on spacing D_s/(M-1) per stream, so raising M refines both.
    """

    def __init__(
        self,
        action_key: str = "actions_keypoints",
        output_action_key: str = "actions_keypoints",
        min_distance_unit_eef: float = 0.45,
        min_distance_unit_kp: float = 0.41,
        resampled_vector_length: int = 30,
        dt: float = 1.0 / 30.0,
        kp_distance_mode: str = "linf",
        rollout_fraction: float = 1.0,
    ):
        if kp_distance_mode not in DISTANCE_MODES:
            raise ValueError(f"kp_distance_mode must be one of {DISTANCE_MODES}")
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.D_eef = float(min_distance_unit_eef)
        self.D_kp = float(min_distance_unit_kp)
        self.M = int(resampled_vector_length)
        self.dt = float(dt)
        self.kp_distance_mode = kp_distance_mode
        # < 1.0 shrinks the common window below min_s duration_s, so no stream
        # is ever extrapolated past its last waypoint.
        self.rollout_fraction = float(rollout_fraction)

    def _streams(self, hand: int):
        """-> (eef_slice, kp_slice, eef_vel_col, kp_vel_col) for this hand."""
        o = hand * PER_HAND_DIM
        return (
            slice(o, o + WRIST_DIM),
            slice(o + WRIST_DIM, o + PER_HAND_DIM),
            o,
            o + WRIST_DIM,
        )

    def tokenize(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != BIMANUAL_DIM:
            raise ValueError(f"expected (T, {BIMANUAL_DIM}), got {chunk.shape}")
        T = len(chunk)
        out = np.zeros((self.M + 1, BIMANUAL_DIM), dtype=np.float64)
        for hand in range(2):
            eef_sl, kp_sl, eef_v, kp_v = self._streams(hand)
            # --- eef stream: translational arc length of the wrist point
            wrist = chunk[:, eef_sl]
            cum_e = cumulative_arc_length_3d(wrist[:, :3])
            end_e = float(min(self.D_eef, cum_e[-1]))
            out[: self.M, eef_sl] = _interp_rows(
                wrist, cum_e, np.linspace(0.0, end_e, self.M)
            )
            n_e = max(int(np.searchsorted(cum_e, end_e, side="right")) - 1, 1)
            out[self.M, eef_v] = end_e / (n_e * self.dt)
            # --- keypoint stream: joint distance over the 21 wrist-frame points
            kp = chunk[:, kp_sl].reshape(T, NUM_KEYPOINTS, 3)
            cum_k = cumulative_keypoint_distance(kp, mode=self.kp_distance_mode)
            end_k = float(min(self.D_kp, cum_k[-1]))
            out[: self.M, kp_sl] = _interp_rows(
                kp.reshape(T, -1), cum_k, np.linspace(0.0, end_k, self.M)
            )
            n_k = max(int(np.searchsorted(cum_k, end_k, side="right")) - 1, 1)
            out[self.M, kp_v] = end_k / (n_k * self.dt)
        return out

    def common_horizon(self, token: np.ndarray) -> float:
        """Seconds over which BOTH streams of BOTH hands remain in-token."""
        durs = []
        for hand in range(2):
            _, _, eef_v, kp_v = self._streams(hand)
            for v, D in ((token[self.M, eef_v], self.D_eef), (token[self.M, kp_v], self.D_kp)):
                if float(v) > 1e-9:
                    durs.append(D / float(v))
        if not durs:
            return 0.0
        return self.rollout_fraction * min(durs)

    def detokenize(self, token: np.ndarray, action_horizon: int) -> np.ndarray:
        token = np.asarray(token, dtype=np.float64)
        H = int(action_horizon)
        out = np.zeros((H, BIMANUAL_DIM), dtype=np.float64)
        span = self.common_horizon(token)
        if span <= 0.0:
            out[:] = token[0]
            return out
        t_grid = np.linspace(0.0, min(span, H * self.dt), H)
        for hand in range(2):
            eef_sl, kp_sl, eef_v, kp_v = self._streams(hand)
            for sl, vcol, D in ((eef_sl, eef_v, self.D_eef), (kp_sl, kp_v, self.D_kp)):
                v = float(token[self.M, vcol])
                s = np.clip(v * t_grid, 0.0, D)          # each stream at its OWN rate
                wp_s = np.linspace(0.0, D, self.M)
                out[:, sl] = _interp_rows(token[: self.M, sl], wp_s, s)
        return out

    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.action_key])
        if chunk.ndim == 3:
            batch[self.output_action_key] = np.stack([self.tokenize(c) for c in chunk])
        else:
            batch[self.output_action_key] = self.tokenize(chunk)
        return batch
