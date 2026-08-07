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
WRIST_DIM = 7  # xyz + ypr + gripper
PER_HAND_DIM = KP_DIM + WRIST_DIM  # 70
BIMANUAL_DIM = 2 * PER_HAND_DIM  # 140

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

    Layout per row, matching the input:
        [L kp(63) | L wrist xyz ypr grip(7) | R kp(63) | R wrist xyz ypr grip(7)]

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
        off = hand * PER_HAND_DIM
        return slice(off, off + KP_DIM), slice(off + KP_DIM, off + PER_HAND_DIM)

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
            out[self.M, kp_sl.start] = end_s / duration if duration > 0 else 0.0
        return out

    def detokenize(self, token: np.ndarray, action_horizon: int) -> np.ndarray:
        """(M+1, 140) -> (H, 140) time-parameterized chunk at the control rate."""
        token = np.asarray(token, dtype=np.float64)
        H = int(action_horizon)
        out = np.zeros((H, BIMANUAL_DIM), dtype=np.float64)
        for hand in range(2):
            kp_sl, wr_sl = self._hand_slice(hand)
            wps = token[: self.M]
            rate = float(token[self.M, kp_sl.start])
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
