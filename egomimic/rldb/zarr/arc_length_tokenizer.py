"""Bimanual arc-length tokenizer for EEF pose action chunks.

Re-parameterizes a time-indexed bimanual cartesian action chunk by translational
arc length instead of time: each arm's next ``min_distance_unit`` meters of EEF
travel is resampled to ``resampled_vector_length`` waypoints spaced uniformly in
arc length, and a translational-velocity channel is appended so the original
timing can be reconstructed at deploy time.

Ported from the GR00T arc-length tokenizer
(gr00t branch ``rpunamiya/arc-length-tokenizer``,
``groot/core/utils/state_action/arc_length_tokenizer.py`` +
``groot/core/data/state_action/arc_length_action_transform.py``), adapted to
this repo's conventions:

- Rotations are xyz + ypr (ZYX euler, matching ``egomimic.utils.pose_utils``)
  instead of 6D rotation. Interpolation still happens in rotation space
  (scipy SLERP), never per-euler-angle.
- Operates on the canonical (T, 14) bimanual cartesian layout
  ``[L xyz ypr grip | R xyz ypr grip]`` produced by the embodiment transform
  pipelines, rather than per-group dicts. Optional per-arm joint chunks
  (T, J) resample along that arm's arc length (``tokenize_with_joints``).
- Each arm has its own independent cumulative arc length; a stationary arm
  yields a zero token (pose held, zero velocity) while the other arm tokenizes
  normally.
- Inputs containing the >=1e8 invalid-pose sentinel produce a 1e9-filled
  output, matching the interpolation helpers in ``pose_utils``.

Layout of the tokenized chunk (per waypoint): per arm
``[xyz(3) ypr(3) grip(1) trans_vel(velocity_dim)]``, arms concatenated —
16 dims for the scalar velocity modes (velocity_dim=1), 20 for the per-dim
modes (velocity_dim=3). Velocity stats for normalization live in
``arc_length_stats.py``.

Only numpy + scipy are required, so this module stays importable in
environments without torch / projectaria. ``TokenizeBimanualArcLength``
implements the same ``transform(batch: dict) -> dict`` interface as
``action_chunk_transforms.Transform`` and composes into those pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Canonical bimanual cartesian layout: [L xyz ypr grip | R xyz ypr grip].
ARM_DIM = 7  # xyz(3) + ypr(3) + gripper(1)
BIMANUAL_CARTESIAN_DIM = 2 * ARM_DIM  # 14
# Tokenized layout appends velocity_dim(mode) trans_vel columns per arm.
# These constants give the scalar-mode (velocity_dim=1) layout; use
# BimanualArcLengthTokenizer.arc_arm_dim / .arc_dim for the mode-aware dims.
ARC_ARM_DIM = ARM_DIM + 1  # 8
ARC_BIMANUAL_DIM = 2 * ARC_ARM_DIM  # 16

# Invalid-pose convention shared with pose_utils._interpolate_euler.
INVALID_POSE_THRESHOLD = 1e8
INVALID_POSE_FILL = 1e9


class VelocityMode(str, Enum):
    """How the translational-velocity payload of a chunk is summarized.

    MEAN_SCALAR: one scalar per chunk — chord distance / chunk duration.
    MEAN_PER_DIM: one 3-vector per chunk — displacement / chunk duration.
    PER_STEP_SCALAR: one scalar per waypoint segment (last value repeated).
    PER_STEP_PER_DIM: one 3-vector per waypoint segment (last value repeated).
    """

    MEAN_SCALAR = "mean_scalar"
    MEAN_PER_DIM = "mean_per_dim"
    PER_STEP_SCALAR = "per_step_scalar"
    PER_STEP_PER_DIM = "per_step_per_dim"


def velocity_dim(mode: VelocityMode | str) -> int:
    """Columns the velocity channel occupies per arm in the tokenized layout."""
    mode = VelocityMode(mode)
    if mode in (VelocityMode.MEAN_SCALAR, VelocityMode.PER_STEP_SCALAR):
        return 1
    return 3


# ---------------------------------------------------------------------------
# Rotation helpers (ypr <-> scipy Rotation; interpolation in rotation space)
# ---------------------------------------------------------------------------


def _ypr_to_rotation(ypr: np.ndarray) -> R:
    """Convert (N, 3) or (3,) yaw-pitch-roll (ZYX euler) to a scipy Rotation."""
    return R.from_euler("ZYX", np.asarray(ypr, dtype=np.float64).reshape(-1, 3))


def _rotation_to_ypr(rotation: R) -> np.ndarray:
    """Convert a scipy Rotation of length N to (N, 3) yaw-pitch-roll."""
    ypr = rotation.as_euler("ZYX", degrees=False)
    return np.atleast_2d(ypr).astype(np.float64)


def slerp_pair_ypr(
    ypr_a: np.ndarray, ypr_b: np.ndarray, alphas: np.ndarray
) -> np.ndarray:
    """SLERP between two ypr rotations at the given alphas in [0, 1].

    returns: (len(alphas), 3) ypr.
    """
    alphas = np.clip(np.asarray(alphas, dtype=np.float64), 0.0, 1.0)
    if len(alphas) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    keyframes = _ypr_to_rotation(
        np.stack([np.asarray(ypr_a), np.asarray(ypr_b)], axis=0)
    )
    slerp = Slerp([0.0, 1.0], keyframes)
    return _rotation_to_ypr(slerp(alphas))


def slerp_through_ypr(ypr_seq: np.ndarray, num: int) -> np.ndarray:
    """SLERP through N ypr waypoints, sampling `num` points uniformly.

    returns: (num, 3) ypr.
    """
    ypr_seq = np.asarray(ypr_seq, dtype=np.float64).reshape(-1, 3)
    n = len(ypr_seq)
    if n == 0:
        return np.zeros((num, 3), dtype=np.float64)
    if n == 1:
        return np.repeat(ypr_seq, num, axis=0)
    slerp = Slerp(np.linspace(0.0, 1.0, n), _ypr_to_rotation(ypr_seq))
    return _rotation_to_ypr(slerp(np.linspace(0.0, 1.0, num)))


# ---------------------------------------------------------------------------
# Arc-length helpers (3D position)
# ---------------------------------------------------------------------------


def cumulative_arc_length(pos: np.ndarray) -> np.ndarray:
    """pos: (N, 3) -> cumulative translational arc length (N,), starting at 0."""
    pos = np.asarray(pos, dtype=np.float64)
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    return np.concatenate([np.array([0.0]), np.cumsum(step)])


def _bracket_segment(cumdist: np.ndarray, target_s: float) -> tuple[int, float]:
    """Locate the segment containing arc length target_s.

    returns: (segment start index i, alpha in [0, 1]), clamped to the first /
    last segment when target_s falls outside [cumdist[0], cumdist[-1]].
    """
    if target_s <= cumdist[0]:
        return 0, 0.0
    if target_s >= cumdist[-1]:
        return len(cumdist) - 2, 1.0
    i = int(np.searchsorted(cumdist, target_s, side="left") - 1)
    i = max(0, min(i, len(cumdist) - 2))
    s0, s1 = cumdist[i], cumdist[i + 1]
    if s1 <= s0 + 1e-12:
        return i, 0.0
    return i, float((target_s - s0) / (s1 - s0))


def _interp_pos_at_s(
    pos: np.ndarray, cumdist: np.ndarray, target_s: float
) -> np.ndarray:
    """Linear-interp position at arc length target_s. returns: (3,)."""
    if target_s <= cumdist[0]:
        return pos[0].copy()
    if target_s >= cumdist[-1]:
        return pos[-1].copy()
    i, alpha = _bracket_segment(cumdist, target_s)
    return (1.0 - alpha) * pos[i] + alpha * pos[i + 1]


def _interp_ypr_at_s(
    ypr: np.ndarray, cumdist: np.ndarray, target_s: float
) -> np.ndarray:
    """SLERP rotation at arc length target_s. returns: (3,) ypr."""
    if target_s <= cumdist[0]:
        return ypr[0].copy()
    if target_s >= cumdist[-1]:
        return ypr[-1].copy()
    i, alpha = _bracket_segment(cumdist, target_s)
    if alpha == 0.0:
        return ypr[i].copy()
    return slerp_pair_ypr(ypr[i], ypr[i + 1], np.array([alpha]))[0]


def _interp_linear_at_s(
    values: np.ndarray, cumdist: np.ndarray, target_s: float
) -> np.ndarray:
    """Linear-interp (N, D) values at arc length target_s. returns: (D,)."""
    if target_s <= cumdist[0]:
        return values[0].copy()
    if target_s >= cumdist[-1]:
        return values[-1].copy()
    i, alpha = _bracket_segment(cumdist, target_s)
    return (1.0 - alpha) * values[i] + alpha * values[i + 1]


def resample_by_distance(
    pos: np.ndarray,
    ypr: np.ndarray,
    gripper: np.ndarray,
    cumdist: np.ndarray,
    start_s: float,
    end_s: float,
    num_samples: int,
    *,
    start_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample pos/ypr/gripper uniformly in arc length over [start_s, end_s].

    Position and gripper interpolate linearly along the original segments;
    rotation SLERPs per-segment. If start_idx is given, the first sample is
    taken explicitly from that index — this breaks arc-length-lookup ambiguity
    in stationary clusters where multiple timesteps share one cumulative arc
    length but differ in rotation/gripper.

    returns: (pos (M, 3), ypr (M, 3), gripper (M, G)).
    """
    targets = np.linspace(start_s, end_s, num_samples)
    pos_out = np.stack(
        [_interp_pos_at_s(pos, cumdist, float(s)) for s in targets], axis=0
    )
    ypr_out = np.stack(
        [_interp_ypr_at_s(ypr, cumdist, float(s)) for s in targets], axis=0
    )
    grip_out = np.stack(
        [_interp_linear_at_s(gripper, cumdist, float(s)) for s in targets], axis=0
    )
    if start_idx is not None:
        pos_out[0] = pos[start_idx]
        ypr_out[0] = ypr[start_idx]
        grip_out[0] = gripper[start_idx]
    return pos_out, ypr_out, grip_out


def _dist_interval_indices(
    cumdist: np.ndarray, start_s: float, end_s: float
) -> tuple[int, int]:
    """Original-timestep index range [start_idx, end_idx] covering [start_s, end_s]."""
    start_idx = int(np.searchsorted(cumdist, start_s, side="left"))
    end_idx = int(np.searchsorted(cumdist, end_s, side="right") - 1)
    start_idx = max(0, min(start_idx, len(cumdist) - 1))
    end_idx = max(start_idx, min(end_idx, len(cumdist) - 1))
    return start_idx, end_idx


# ---------------------------------------------------------------------------
# Velocity payloads
# ---------------------------------------------------------------------------


def make_velocity_payload(
    pos_rs: np.ndarray, chunk_duration_steps: int, dt: float, mode: VelocityMode
) -> np.ndarray:
    """Summarize a resampled chunk's translational speed.

    args:
        pos_rs: (M, 3) arc-length-resampled positions.
        chunk_duration_steps: number of original time steps the chunk covers.
        dt: seconds per original time step.
    returns:
        MEAN_SCALAR: (1,) chord distance / total time.
        MEAN_PER_DIM: (3,) displacement / total time.
        PER_STEP_SCALAR: (M,) per-segment speed, last value repeated.
        PER_STEP_PER_DIM: (M, 3) per-segment velocity, last value repeated.
    """
    m = pos_rs.shape[0]
    total_time = max(chunk_duration_steps * dt, 1e-8)
    if mode == VelocityMode.MEAN_SCALAR:
        dist = float(np.linalg.norm(pos_rs[-1] - pos_rs[0]))
        return np.array([dist / total_time], dtype=np.float64)
    if mode == VelocityMode.MEAN_PER_DIM:
        return ((pos_rs[-1] - pos_rs[0]) / total_time).astype(np.float64)
    seg = pos_rs[1:] - pos_rs[:-1]
    per_seg_time = max(total_time / max(m - 1, 1), 1e-8)
    if mode == VelocityMode.PER_STEP_SCALAR:
        v = np.linalg.norm(seg, axis=-1) / per_seg_time
        return np.concatenate([v, v[-1:]]).astype(np.float64)
    if mode == VelocityMode.PER_STEP_PER_DIM:
        v = seg / per_seg_time
        return np.concatenate([v, v[-1:]], axis=0).astype(np.float64)
    raise ValueError(f"Unsupported velocity mode: {mode}")


def _zero_velocity(mode: VelocityMode, m: int) -> np.ndarray:
    if mode == VelocityMode.MEAN_SCALAR:
        return np.zeros((1,), dtype=np.float64)
    if mode == VelocityMode.MEAN_PER_DIM:
        return np.zeros((3,), dtype=np.float64)
    if mode == VelocityMode.PER_STEP_SCALAR:
        return np.zeros((m,), dtype=np.float64)
    if mode == VelocityMode.PER_STEP_PER_DIM:
        return np.zeros((m, 3), dtype=np.float64)
    raise ValueError(f"Unsupported velocity mode: {mode}")


def _broadcast_velocity(
    trans_vel: np.ndarray, mode: VelocityMode, m: int
) -> np.ndarray:
    """Reshape a velocity payload into a uniform (M, velocity_dim(mode)) block."""
    v = np.asarray(trans_vel, dtype=np.float64)
    if mode == VelocityMode.MEAN_SCALAR:
        return np.broadcast_to(v.reshape(1, 1), (m, 1)).copy()
    if mode == VelocityMode.MEAN_PER_DIM:
        return np.broadcast_to(v.reshape(1, 3), (m, 3)).copy()
    if mode == VelocityMode.PER_STEP_SCALAR:
        return v.reshape(m, 1)
    if mode == VelocityMode.PER_STEP_PER_DIM:
        return v.reshape(m, 3)
    raise ValueError(f"Unsupported velocity mode: {mode}")


# ---------------------------------------------------------------------------
# Single-arm tokenizer core
# ---------------------------------------------------------------------------


@dataclass
class ArcChunkToken:
    """One arc-length chunk of a single arm's trajectory.

    kind is "motion" for a normal chunk or "zero" for a degenerate one
    (stationary, or covering too many time steps for the distance unit).
    """

    kind: str
    mode: str
    pos: np.ndarray  # (M, 3) arc-length-resampled positions
    ypr: np.ndarray  # (M, 3) resampled rotations (ZYX euler)
    gripper: np.ndarray  # (M, G) resampled gripper
    trans_vel: np.ndarray  # (1,), (3,), (M,), or (M, 3) depending on mode
    start_idx: int
    end_idx: int
    num_original_steps: int
    chunk_distance: float
    dt: float
    zero_reason: str | None = None


class ArcLengthTokenizer:
    """Single-arm arc-length tokenizer for xyz + ypr (+ gripper) trajectories.

    Chunks a trajectory into fixed translational arc-length units, resamples
    each chunk to ``resampled_vector_length`` waypoints uniform in arc length,
    and stores a velocity payload for timing reconstruction.
    """

    def __init__(
        self,
        min_distance_unit: float = 0.05,
        resampled_vector_length: int = 20,
        mode: str = "mean_scalar",
        dt: float = 1.0 / 30.0,
        zero_dist_epsilon: float = 1e-6,
        max_steps_per_chunk: int = 200,
    ):
        if resampled_vector_length < 2:
            raise ValueError(
                f"resampled_vector_length must be >= 2, got {resampled_vector_length}"
            )
        self.min_distance_unit = float(min_distance_unit)
        self.M = int(resampled_vector_length)
        self.mode = VelocityMode(mode)
        self.dt = float(dt)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        self.max_steps_per_chunk = int(max_steps_per_chunk)

    @staticmethod
    def _validate_inputs(
        pos: np.ndarray, ypr: np.ndarray, gripper: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.asarray(pos, dtype=np.float64)
        ypr = np.asarray(ypr, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
        if ypr.ndim != 2 or ypr.shape[1] != 3:
            raise ValueError(f"ypr must have shape (N, 3), got {ypr.shape}")
        if len(pos) != len(ypr):
            raise ValueError(
                f"pos and ypr must have the same length, got {len(pos)} vs {len(ypr)}"
            )
        if len(pos) < 2:
            raise ValueError(f"Need at least 2 timesteps, got {len(pos)}")
        if gripper is None:
            gripper = np.zeros((len(pos), 1), dtype=np.float64)
        else:
            gripper = np.asarray(gripper, dtype=np.float64)
            if gripper.ndim == 1:
                gripper = gripper[:, None]
            if len(gripper) != len(pos):
                raise ValueError(
                    f"gripper length {len(gripper)} does not match pos length {len(pos)}"
                )
        return pos, ypr, gripper

    # -- tokenize ----------------------------------------------------------

    def tokenize(
        self,
        pos: np.ndarray,
        ypr: np.ndarray,
        gripper: np.ndarray | None = None,
    ) -> list[ArcChunkToken]:
        """Chunk the full trajectory into consecutive non-overlapping tokens."""
        pos, ypr, gripper = self._validate_inputs(pos, ypr, gripper)
        cumdist = cumulative_arc_length(pos)
        total_s = float(cumdist[-1])

        tokens: list[ArcChunkToken] = []
        cur_s = 0.0
        while cur_s < total_s - 1e-12:
            next_s = min(cur_s + self.min_distance_unit, total_s)
            start_idx, end_idx = _dist_interval_indices(cumdist, cur_s, next_s)
            num_steps = max(1, end_idx - start_idx)
            chunk_dist = float(next_s - cur_s)

            if chunk_dist < self.zero_dist_epsilon:
                reason = "distance_below_epsilon"
            elif num_steps > self.max_steps_per_chunk:
                reason = "too_many_steps_for_distance_unit"
            else:
                reason = None

            if reason is not None:
                tokens.append(
                    self._make_zero_token(
                        pos,
                        ypr,
                        gripper,
                        cumdist,
                        cur_s,
                        next_s,
                        start_idx,
                        end_idx,
                        num_steps,
                        chunk_dist,
                        reason,
                    )
                )
            else:
                pos_rs, ypr_rs, grip_rs = resample_by_distance(
                    pos, ypr, gripper, cumdist, cur_s, next_s, self.M
                )
                tokens.append(
                    ArcChunkToken(
                        kind="motion",
                        mode=self.mode.value,
                        pos=pos_rs,
                        ypr=ypr_rs,
                        gripper=grip_rs,
                        trans_vel=make_velocity_payload(
                            pos_rs, num_steps, self.dt, self.mode
                        ),
                        start_idx=start_idx,
                        end_idx=end_idx,
                        num_original_steps=num_steps,
                        chunk_distance=chunk_dist,
                        dt=self.dt,
                    )
                )
            cur_s = next_s
        return tokens

    def tokenize_at(
        self,
        pos: np.ndarray,
        ypr: np.ndarray,
        gripper: np.ndarray | None = None,
        t: int = 0,
    ) -> ArcChunkToken:
        """Tokenize the next min_distance_unit of travel starting at timestep t.

        This is the time-synced form used for prediction targets: the token
        covers arc lengths [cumdist[t], cumdist[t] + min_distance_unit]
        (clipped to the trajectory end), with the first waypoint anchored
        exactly at timestep t.
        """
        pos, ypr, gripper = self._validate_inputs(pos, ypr, gripper)
        n = len(pos)
        if not 0 <= t < n:
            raise ValueError(f"t must be in [0, {n - 1}], got {t}")
        cumdist = cumulative_arc_length(pos)
        total_s = float(cumdist[-1])

        start_s = float(cumdist[t])
        end_s = min(start_s + self.min_distance_unit, total_s)
        chunk_dist = end_s - start_s
        _, end_idx = _dist_interval_indices(cumdist, start_s, end_s)
        num_steps = max(1, end_idx - t)

        if chunk_dist < self.zero_dist_epsilon:
            reason = "distance_below_epsilon"
        elif num_steps > self.max_steps_per_chunk:
            reason = "too_many_steps_for_distance_unit"
        else:
            reason = None

        if reason is not None:
            end = min(end_idx, n - 1)
            alphas = np.linspace(0.0, 1.0, self.M)
            return ArcChunkToken(
                kind="zero",
                mode=self.mode.value,
                pos=np.repeat(pos[t : t + 1], self.M, axis=0),
                ypr=slerp_pair_ypr(ypr[t], ypr[end], alphas),
                gripper=(1.0 - alphas[:, None]) * gripper[t]
                + alphas[:, None] * gripper[end],
                trans_vel=_zero_velocity(self.mode, self.M),
                start_idx=t,
                end_idx=end_idx,
                num_original_steps=num_steps,
                chunk_distance=chunk_dist,
                dt=self.dt,
                zero_reason=reason,
            )

        pos_rs, ypr_rs, grip_rs = resample_by_distance(
            pos, ypr, gripper, cumdist, start_s, end_s, self.M, start_idx=t
        )
        return ArcChunkToken(
            kind="motion",
            mode=self.mode.value,
            pos=pos_rs,
            ypr=ypr_rs,
            gripper=grip_rs,
            trans_vel=make_velocity_payload(pos_rs, num_steps, self.dt, self.mode),
            start_idx=t,
            end_idx=end_idx,
            num_original_steps=num_steps,
            chunk_distance=chunk_dist,
            dt=self.dt,
        )

    def tokenize_per_timestep(
        self,
        pos: np.ndarray,
        ypr: np.ndarray,
        gripper: np.ndarray | None = None,
    ) -> list[ArcChunkToken]:
        """One (overlapping) token per timestep, time-synced with observations."""
        pos, ypr, gripper = self._validate_inputs(pos, ypr, gripper)
        return [self.tokenize_at(pos, ypr, gripper, t=t) for t in range(len(pos))]

    # -- detokenize ---------------------------------------------------------

    def detokenize(
        self, tokens: list[ArcChunkToken]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct an approximate time-indexed trajectory from consecutive tokens.

        returns: (pos (N, 3), ypr (N, 3), gripper (N, G)). Consecutive chunks
        share their boundary waypoint, so each chunk after the first drops its
        first reconstructed step.
        """
        pos_parts: list[np.ndarray] = []
        ypr_parts: list[np.ndarray] = []
        grip_parts: list[np.ndarray] = []
        for token in tokens:
            p, r, g = self._detokenize_one(token)
            if pos_parts:
                p, r, g = p[1:], r[1:], g[1:]
            pos_parts.append(p)
            ypr_parts.append(r)
            grip_parts.append(g)
        if not pos_parts:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 1), dtype=np.float64),
            )
        return (
            np.concatenate(pos_parts, axis=0),
            np.concatenate(ypr_parts, axis=0),
            np.concatenate(grip_parts, axis=0),
        )

    def _detokenize_one(
        self, token: ArcChunkToken
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos_rs = np.asarray(token.pos, dtype=np.float64)
        ypr_rs = np.asarray(token.ypr, dtype=np.float64)
        grip_rs = np.asarray(token.gripper, dtype=np.float64)

        if token.kind == "zero":
            n = max(2, int(token.num_original_steps))
            alphas = np.linspace(0.0, 1.0, n)
            return (
                np.repeat(pos_rs[0:1], n, axis=0),
                slerp_pair_ypr(ypr_rs[0], ypr_rs[-1], alphas),
                (1.0 - alphas[:, None]) * grip_rs[0] + alphas[:, None] * grip_rs[-1],
            )

        n_steps = max(2, self._infer_steps(token))
        src = np.linspace(0.0, 1.0, len(pos_rs))
        tgt = np.linspace(0.0, 1.0, n_steps)

        pos_out = np.stack(
            [np.interp(tgt, src, pos_rs[:, d]) for d in range(3)], axis=-1
        )
        ypr_out = slerp_through_ypr(ypr_rs, n_steps)
        grip_out = np.stack(
            [np.interp(tgt, src, grip_rs[:, d]) for d in range(grip_rs.shape[1])],
            axis=-1,
        )
        return pos_out, ypr_out, grip_out

    def _infer_steps(self, token: ArcChunkToken) -> int:
        """Number of time-indexed waypoints to emit for a motion chunk.

        The chunk covers num_original_steps step-intervals, i.e.
        num_original_steps + 1 positions; the velocity payload tells us how
        many control steps the chunk's distance corresponds to.
        """
        pos_rs = np.asarray(token.pos, dtype=np.float64)
        vel = np.asarray(token.trans_vel, dtype=np.float64)
        dt = float(token.dt)
        seg_d = np.linalg.norm(pos_rs[1:] - pos_rs[:-1], axis=-1)
        total_dist = float(seg_d.sum())

        if total_dist < self.zero_dist_epsilon:
            return max(2, int(token.num_original_steps) + 1)

        mode = VelocityMode(token.mode)
        if mode == VelocityMode.MEAN_SCALAR:
            v = max(float(vel[0]), 1e-8)
            return max(2, int(np.ceil(total_dist / (v * dt))) + 1)
        if mode == VelocityMode.MEAN_PER_DIM:
            v = max(float(np.linalg.norm(vel)), 1e-8)
            return max(2, int(np.ceil(total_dist / (v * dt))) + 1)
        if mode == VelocityMode.PER_STEP_SCALAR:
            v = np.maximum(vel[:-1], 1e-8)
            seg_steps = np.ceil(seg_d / (v * dt)).astype(int)
            return max(2, int(np.sum(np.maximum(seg_steps, 1))) + 1)
        if mode == VelocityMode.PER_STEP_PER_DIM:
            v = np.maximum(np.linalg.norm(vel[:-1], axis=-1), 1e-8)
            seg_steps = np.ceil(seg_d / (v * dt)).astype(int)
            return max(2, int(np.sum(np.maximum(seg_steps, 1))) + 1)
        raise ValueError(f"Unsupported velocity mode: {mode}")


# ---------------------------------------------------------------------------
# Bimanual layer: canonical (T, 14) chunk <-> (M, 14 + 2 * vel_dim) arc chunk
# ---------------------------------------------------------------------------


@dataclass
class BimanualArcLengthConfig:
    """Config for the bimanual arc-length tokenizer.

    min_distance_unit: meters of per-arm EEF travel covered by one token.
    resampled_vector_length: M, waypoints per token (the post-tokenization
        chunk length).
    mode: velocity payload mode; detokenize supports the mean modes only.
    dt: seconds per source time step (control period at deploy time).
    """

    min_distance_unit: float = 0.05
    resampled_vector_length: int = 20
    mode: str = "mean_scalar"
    dt: float = 1.0 / 30.0
    zero_dist_epsilon: float = 1e-6
    max_steps_per_chunk: int = 200


class BimanualArcLengthTokenizer:
    """Arc-length tokenizer over the canonical bimanual cartesian layout.

    tokenize: (T, 14) time-indexed chunk -> (M, arc_dim) arc-indexed chunk,
    where each arm's block is [xyz(3) ypr(3) grip(1) vel(velocity_dim)] —
    arc_dim is 16 for the scalar velocity modes and 20 for the per-dim modes.
    Each arm is tokenized against its own cumulative arc length, so a
    stationary arm holds pose (zero velocity) while the other arm covers its
    distance unit. The token starts at t=0 of the input chunk (time-synced
    with the observation) and spans the next min_distance_unit of that arm's
    travel.

    tokenize_with_joints additionally resamples per-arm joint chunks (T, J)
    at the same M waypoints along that arm's arc-length parameterization
    (zero tokens hold the initial joints), mirroring the GR00T action
    transform's joint handling.

    detokenize: (M, arc_dim) -> (H, 14) time-indexed chunk sampled at the
    control period ``dt`` under the mean-velocity constant-speed assumption
    (modes "mean_scalar" and "mean_per_dim"; per-step timing is not wired):
    control step k sits at arc length v * k * dt along the token (clamped to
    the token span), and every channel is interpolated there against the
    waypoints' own cumulative arc length. The trans_vel columns are consumed
    to set the timing, not returned.
    """

    def __init__(self, config: BimanualArcLengthConfig | None = None):
        self.config = config or BimanualArcLengthConfig()
        self._core = ArcLengthTokenizer(
            min_distance_unit=self.config.min_distance_unit,
            resampled_vector_length=self.config.resampled_vector_length,
            mode=self.config.mode,
            dt=self.config.dt,
            zero_dist_epsilon=self.config.zero_dist_epsilon,
            max_steps_per_chunk=self.config.max_steps_per_chunk,
        )

    @property
    def M(self) -> int:
        return self._core.M

    @property
    def velocity_dim(self) -> int:
        return velocity_dim(self._core.mode)

    @property
    def arc_arm_dim(self) -> int:
        return ARM_DIM + self.velocity_dim

    @property
    def arc_dim(self) -> int:
        return 2 * self.arc_arm_dim

    def tokenize(self, chunk: np.ndarray) -> np.ndarray:
        """args: chunk (T, 14) [L xyz ypr grip | R xyz ypr grip], T >= 2.

        returns: (M, arc_dim) [L xyz ypr grip vel | R xyz ypr grip vel].
        """
        actions, _, _ = self.tokenize_with_joints(chunk)
        return actions

    def tokenize_with_joints(
        self,
        chunk: np.ndarray,
        joints_left: np.ndarray | None = None,
        joints_right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Tokenize the bimanual chunk plus optional per-arm joint chunks.

        args:
            chunk: (T, 14) bimanual cartesian chunk.
            joints_left / joints_right: optional (T, J) absolute joint targets,
                resampled along that arm's arc length at the same M waypoints.
        returns:
            (actions (M, arc_dim), joints_left (M, J) or None,
             joints_right (M, J) or None)
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != BIMANUAL_CARTESIAN_DIM:
            raise ValueError(
                f"BimanualArcLengthTokenizer.tokenize expects (T, "
                f"{BIMANUAL_CARTESIAN_DIM}), got {chunk.shape}"
            )
        joints = []
        for name, j in (("joints_left", joints_left), ("joints_right", joints_right)):
            if j is None:
                joints.append(None)
                continue
            j = np.asarray(j, dtype=np.float64)
            if j.ndim == 1:
                j = j[:, None]
            if j.ndim != 2 or len(j) != len(chunk):
                raise ValueError(
                    f"{name} must have shape (T, J) with T={len(chunk)}, got {j.shape}"
                )
            joints.append(j)

        if np.any(np.abs(chunk) >= INVALID_POSE_THRESHOLD):
            actions = np.full((self.M, self.arc_dim), INVALID_POSE_FILL)
            filled = [
                None if j is None else np.full((self.M, j.shape[1]), INVALID_POSE_FILL)
                for j in joints
            ]
            return actions, filled[0], filled[1]

        arms = []
        joints_out: list[np.ndarray | None] = []
        for arm_start, j in ((0, joints[0]), (ARM_DIM, joints[1])):
            arm = chunk[:, arm_start : arm_start + ARM_DIM]
            pos = arm[:, 0:3]
            token = self._core.tokenize_at(
                pos=pos, ypr=arm[:, 3:6], gripper=arm[:, 6:7], t=0
            )
            vel = _broadcast_velocity(token.trans_vel, self._core.mode, self.M)
            arms.append(
                np.concatenate([token.pos, token.ypr, token.gripper, vel], axis=-1)
            )
            joints_out.append(
                None if j is None else self._resample_arm_joints(j, pos, token)
            )
        return np.concatenate(arms, axis=-1), joints_out[0], joints_out[1]

    def _resample_arm_joints(
        self, joints: np.ndarray, pos: np.ndarray, token: ArcChunkToken
    ) -> np.ndarray:
        """Resample (T, J) joints at the token's M waypoints along the arm's
        arc length. Zero tokens hold the initial joints (matching the token's
        held pose); motion tokens interpolate against the arm's cumdist with
        the first waypoint anchored at the token's start timestep."""
        if token.kind == "zero":
            return np.repeat(
                joints[token.start_idx : token.start_idx + 1], self.M, axis=0
            )
        cumdist = cumulative_arc_length(pos)
        start_s = float(cumdist[token.start_idx])
        targets = np.linspace(start_s, start_s + token.chunk_distance, self.M)
        out = np.stack(
            [_interp_linear_at_s(joints, cumdist, float(s)) for s in targets], axis=0
        )
        out[0] = joints[token.start_idx]
        return out

    def detokenize(self, arc_chunk: np.ndarray, action_horizon: int) -> np.ndarray:
        """args: arc_chunk (M, arc_dim), action_horizon H.

        returns: (H, 14) time-indexed chunk at the control period.
        """
        actions, _, _ = self.detokenize_with_joints(arc_chunk, action_horizon)
        return actions

    def detokenize_with_joints(
        self,
        arc_chunk: np.ndarray,
        action_horizon: int,
        joints_left: np.ndarray | None = None,
        joints_right: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Detokenize the arc chunk plus optional arc-indexed joint chunks.

        args:
            arc_chunk: (M, arc_dim) tokenized bimanual chunk.
            joints_left / joints_right: optional (M, J) arc-indexed joints (as
                produced by tokenize_with_joints), interpolated at the same
                control timestamps against that arm's waypoint arc length.
        returns:
            (actions (H, 14), joints_left (H, J) or None,
             joints_right (H, J) or None)
        """
        mode = self._core.mode
        if mode not in (VelocityMode.MEAN_SCALAR, VelocityMode.MEAN_PER_DIM):
            raise NotImplementedError(
                "Arc-length detokenize supports the mean modes ('mean_scalar', "
                f"'mean_per_dim') only, got {mode.value!r}"
            )
        arc_chunk = np.asarray(arc_chunk, dtype=np.float64)
        if arc_chunk.ndim != 2 or arc_chunk.shape[1] != self.arc_dim:
            raise ValueError(
                f"BimanualArcLengthTokenizer.detokenize expects (M, "
                f"{self.arc_dim}), got {arc_chunk.shape}"
            )
        m = arc_chunk.shape[0]
        joints = []
        for name, j in (("joints_left", joints_left), ("joints_right", joints_right)):
            if j is None:
                joints.append(None)
                continue
            j = np.asarray(j, dtype=np.float64)
            if j.ndim == 1:
                j = j[:, None]
            if j.ndim != 2 or len(j) != m:
                raise ValueError(
                    f"{name} must have shape (M, J) with M={m}, got {j.shape}"
                )
            joints.append(j)

        h = int(action_horizon)
        dt = self._core.dt
        vd = self.velocity_dim

        arms = []
        joints_out: list[np.ndarray | None] = []
        for arm_start, j in ((0, joints[0]), (self.arc_arm_dim, joints[1])):
            arm = arc_chunk[:, arm_start : arm_start + self.arc_arm_dim]
            pos, ypr, grip = arm[:, 0:3], arm[:, 3:6], arm[:, 6:7]
            vel = arm[0, ARM_DIM : ARM_DIM + vd]
            v = max(float(vel[0]) if vd == 1 else float(np.linalg.norm(vel)), 0.0)

            cumdist = cumulative_arc_length(pos)
            total = float(cumdist[-1])
            if total < 1e-9 or v <= 1e-8:
                pos_t = np.repeat(pos[:1], h, axis=0)
                ypr_t = np.repeat(ypr[:1], h, axis=0)
                grip_t = np.repeat(grip[:1], h, axis=0)
                j_t = None if j is None else np.repeat(j[:1], h, axis=0)
            else:
                s = np.minimum(v * np.arange(h, dtype=np.float64) * dt, total)
                pos_t = np.stack(
                    [_interp_pos_at_s(pos, cumdist, float(sk)) for sk in s]
                )
                ypr_t = np.stack(
                    [_interp_ypr_at_s(ypr, cumdist, float(sk)) for sk in s]
                )
                grip_t = np.stack(
                    [_interp_linear_at_s(grip, cumdist, float(sk)) for sk in s]
                )
                j_t = (
                    None
                    if j is None
                    else np.stack(
                        [_interp_linear_at_s(j, cumdist, float(sk)) for sk in s]
                    )
                )
            arms.append(np.concatenate([pos_t, ypr_t, grip_t], axis=-1))
            joints_out.append(j_t)
        return np.concatenate(arms, axis=-1), joints_out[0], joints_out[1]


class TokenizeBimanualArcLength:
    """Transform-style adapter: arc-length-tokenize a bimanual cartesian chunk.

    Implements the same ``transform(batch: dict) -> dict`` interface as
    ``action_chunk_transforms.Transform`` (duck-typed so this module stays
    importable without torch/projectaria). Reads ``action_key`` (T, 14) from
    the batch and writes the (M, arc_dim) tokenized chunk to
    ``output_action_key``. If ``left_joint_key`` / ``right_joint_key`` are
    given, those (T, J) chunks are resampled along the matching arm's arc
    length and written to ``output_*_joint_key`` (default: ``<key>_arc``).
    """

    def __init__(
        self,
        action_key: str = "actions_cartesian",
        output_action_key: str = "actions_arc",
        config: BimanualArcLengthConfig | None = None,
        left_joint_key: str | None = None,
        right_joint_key: str | None = None,
        output_left_joint_key: str | None = None,
        output_right_joint_key: str | None = None,
    ):
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.left_joint_key = left_joint_key
        self.right_joint_key = right_joint_key
        self.output_left_joint_key = output_left_joint_key or (
            f"{left_joint_key}_arc" if left_joint_key else None
        )
        self.output_right_joint_key = output_right_joint_key or (
            f"{right_joint_key}_arc" if right_joint_key else None
        )
        self.tokenizer = BimanualArcLengthTokenizer(config)

    def transform(self, batch: dict) -> dict:
        joints_left = (
            np.asarray(batch[self.left_joint_key]) if self.left_joint_key else None
        )
        joints_right = (
            np.asarray(batch[self.right_joint_key]) if self.right_joint_key else None
        )
        actions, joints_left_rs, joints_right_rs = self.tokenizer.tokenize_with_joints(
            np.asarray(batch[self.action_key]), joints_left, joints_right
        )
        batch[self.output_action_key] = actions
        if self.left_joint_key:
            batch[self.output_left_joint_key] = joints_left_rs
        if self.right_joint_key:
            batch[self.output_right_joint_key] = joints_right_rs
        return batch


# Canonical layout for the (M+1, 8) arc-token variant used by the FM policy
# head (see hpt_cotrain_mecka_flow_shared_head_arc.yaml). Per-slot dims:
#   [Lx, Ly, Lz, L_grip, Rx, Ry, Rz, R_grip]
# Rows: 0..M-1 = waypoints, row M = velocity token (per-action-dim mean rate).
ARC_TOK_PER_ARM_DIM = 4  # xyz(3) + gripper(1)
ARC_TOK_BIMANUAL_DIM = 2 * ARC_TOK_PER_ARM_DIM  # 8


class TokenizeBimanualArcLengthCartesian:
    """Transform: (T, 14) actions_cartesian -> (M+1, 8) arc-tokenized layout.

    Layout per row:
        [Lx, Ly, Lz, L_grip, Rx, Ry, Rz, R_grip]
    Rows 0..M-1 are the M waypoints uniform in each arm's arc length over
    the first ``min_distance_unit`` meters of that arm's travel; row M is a
    per-action-dim mean-velocity token computed as
        vel[d] = (waypoints[M-1, d] - waypoints[0, d]) / duration_arm[d]
    where duration_arm is derived per-arm from the underlying tokenizer's
    MEAN_PER_DIM xyz velocity (duration = ||chord_xyz|| / ||vel_xyz||).
    Gripper columns of the velocity token use the same per-arm duration.

    Assumes the input chunk is already in the model's target cam frame
    (post InterpolatePose + ActionChunkCoordinateFrameTransform + XYZWXYZ_
    to_XYZYPR + ConcatKeys). Rotation (dims 3-5, 10-12) is intentionally
    dropped from the output — this variant carries xyz+gripper only.

    Args:
        action_key: input batch key holding the (T, 14) chunk.
        output_action_key: where to write the (M+1, 8) tokenized chunk.
        min_distance_unit: per-arm arc length span of the token, in meters
            (i.e. the D parameter in the sweep script; the tokenizer covers
            the first ``min_distance_unit`` meters of each arm's travel).
        resampled_vector_length: number of waypoints M (the sequence has
            M+1 rows once the velocity token is appended).
        dt: seconds between consecutive rows of the preprocessed input chunk.
        zero_dist_epsilon: below-this-arc-length chunks are treated as
            stationary; the vel token comes out as zeros in that case so
            downstream (loss, detokenize) doesn't NaN.
    """

    def __init__(
        self,
        action_key: str = "actions_cartesian",
        output_action_key: str = "actions_cartesian",
        min_distance_unit: float = 0.60,
        resampled_vector_length: int = 20,
        dt: float = 1.0 / 30.0,
        zero_dist_epsilon: float = 1e-6,
    ):
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.min_distance_unit = float(min_distance_unit)
        # MEAN_PER_DIM velocity mode: we consume the per-axis xyz velocity
        # directly. Gripper velocity is computed here (the base tokenizer
        # doesn't track gripper velocity).
        cfg = BimanualArcLengthConfig(
            min_distance_unit=self.min_distance_unit,
            resampled_vector_length=int(resampled_vector_length),
            mode=VelocityMode.MEAN_PER_DIM.value,
            dt=float(dt),
            zero_dist_epsilon=float(zero_dist_epsilon),
        )
        self.tokenizer = BimanualArcLengthTokenizer(cfg)
        self.zero_dist_epsilon = float(zero_dist_epsilon)

    @property
    def M(self) -> int:
        return self.tokenizer.M

    def _physical_mean_velocity(
        self, input_pos: np.ndarray, waypoint_pos: np.ndarray
    ) -> np.ndarray:
        """Recover chord velocity using the preprocessed chunk's true time axis."""
        cumdist = cumulative_arc_length(input_pos)
        total = float(cumdist[-1])
        if total < self.zero_dist_epsilon:
            return np.zeros(3, dtype=np.float64)

        if total <= self.min_distance_unit + self.zero_dist_epsilon:
            duration_steps = float(max(len(input_pos) - 1, 1))
        else:
            end_s = self.min_distance_unit
            right = int(np.searchsorted(cumdist, end_s, side="left"))
            right = max(1, min(right, len(cumdist) - 1))
            left = right - 1
            segment_distance = float(cumdist[right] - cumdist[left])
            fraction = (
                0.0
                if segment_distance <= 1e-12
                else float(end_s - cumdist[left]) / segment_distance
            )
            duration_steps = float(left) + fraction

        duration = max(duration_steps * self.tokenizer.config.dt, 1e-8)
        return ((waypoint_pos[-1] - waypoint_pos[0]) / duration).astype(
            np.float64
        )

    def transform(self, batch: dict) -> dict:
        input_actions = np.asarray(batch[self.action_key], dtype=np.float64)
        if np.any(np.abs(input_actions) >= INVALID_POSE_THRESHOLD):
            raise ValueError(
                f"{self.action_key} contains the invalid-pose sentinel; "
                "rejecting the sample before normalization"
            )
        arc = self.tokenizer.tokenize(input_actions)
        # Per-arm block: [xyz(3), ypr(3), grip(1), vel_xyz(3)] = 10 dims.
        # Bimanual concat: 20 dims total.
        L_xyz = arc[:, 0:3]
        L_grip = arc[:, 6:7]
        R_xyz = arc[:, 10:13]
        R_grip = arc[:, 16:17]
        # The source core quantizes duration to an integer end index. Recompute
        # the model-facing mean velocity at the fractional D crossing so the
        # payload remains a physical velocity after wide-window interpolation.
        L_vel = self._physical_mean_velocity(input_actions[:, 0:3], L_xyz)
        R_vel = self._physical_mean_velocity(input_actions[:, 7:10], R_xyz)

        waypoints = np.concatenate([L_xyz, L_grip, R_xyz, R_grip], axis=-1)

        # Per-arm duration from mean_per_dim xyz vel: duration = chord / speed.
        # Guard for degenerate arms (no motion or below the tokenizer's
        # zero_dist_epsilon) by falling back to the full input duration so
        # gripper velocity remains physical even when xyz is stationary.
        L_chord = float(np.linalg.norm(L_xyz[-1] - L_xyz[0]))
        R_chord = float(np.linalg.norm(R_xyz[-1] - R_xyz[0]))
        L_speed = float(np.linalg.norm(L_vel))
        R_speed = float(np.linalg.norm(R_vel))
        default_dur = max(len(input_actions) - 1, 1) * self.tokenizer.config.dt
        L_dur = (L_chord / L_speed) if L_speed > 1e-8 else default_dur
        R_dur = (R_chord / R_speed) if R_speed > 1e-8 else default_dur
        # Gripper velocity per arm — same "mean rate over the token" formula
        # as xyz's mean_per_dim, using the arm's derived duration.
        L_grip_vel = float(L_grip[-1, 0] - L_grip[0, 0]) / max(L_dur, 1e-8)
        R_grip_vel = float(R_grip[-1, 0] - R_grip[0, 0]) / max(R_dur, 1e-8)

        # Assemble (1, 8) velocity token: [Lx_v, Ly_v, Lz_v, L_grip_v,
        # Rx_v, Ry_v, Rz_v, R_grip_v]. A stationary wrist has zero xyz
        # velocity but can retain a nonzero open/close gripper rate.
        vel_token = np.array(
            [
                [
                    L_vel[0],
                    L_vel[1],
                    L_vel[2],
                    L_grip_vel,
                    R_vel[0],
                    R_vel[1],
                    R_vel[2],
                    R_grip_vel,
                ]
            ],
            dtype=np.float64,
        )
        out = np.concatenate([waypoints, vel_token], axis=0)  # (M+1, 8)
        batch[self.output_action_key] = out
        return batch

    def detokenize(
        self,
        arc_actions: np.ndarray,
        action_horizon: int,
    ) -> np.ndarray:
        """Inverse of ``transform`` — take a (M+1, 8) arc token back to a
        time-parameterized (H, 8) chunk at the control period.

        Semantics:
          - Read the vel token (row M) to derive per-arm duration from the
            predicted chord displacement and mean xyz velocity.
          - Uniformly sample H timesteps and interpolate xyz and gripper
            against the arm's cumulative arc length, preserving the source
            transform's co-located schedule for moving wrists.
          - A degenerate xyz velocity holds position. In that case only,
            gripper timing is reconstructed from its own mean-rate entry so a
            stationary wrist can still open or close.

        Output layout (H, 8): same as the waypoint layout, one row per
        control step. Used by the val-video eval to project a stream of
        setpoints at 30 Hz through the front camera.
        """
        arc_actions = np.asarray(arc_actions, dtype=np.float64)
        if arc_actions.ndim != 2 or arc_actions.shape[1] != ARC_TOK_BIMANUAL_DIM:
            raise ValueError(
                f"detokenize expects (M+1, {ARC_TOK_BIMANUAL_DIM}), got "
                f"{arc_actions.shape}"
            )
        M_plus_1 = arc_actions.shape[0]
        M = M_plus_1 - 1
        if M < 2:
            raise ValueError(f"Need M >= 2 waypoints, got M+1={M_plus_1}")

        waypoints = arc_actions[:M]  # (M, 8)
        vel_token = arc_actions[M]  # (8,)
        dt = self.tokenizer.config.dt
        h = int(action_horizon)

        arms_out = []
        for xyz_off, grip_off, vel_slice in (
            (0, 3, slice(0, 3)),  # left
            (4, 7, slice(4, 7)),  # right
        ):
            xyz_wp = waypoints[:, xyz_off : xyz_off + 3]  # (M, 3)
            grip_wp = waypoints[:, grip_off : grip_off + 1]  # (M, 1)
            vel_xyz = vel_token[vel_slice]  # (3,)
            speed = float(np.linalg.norm(vel_xyz))

            cumdist = cumulative_arc_length(xyz_wp)
            total = float(cumdist[-1])
            if total < 1e-9 or speed < 1e-8:
                # Degenerate translation: hold the first xyz waypoint.
                pos_t = np.repeat(xyz_wp[:1], h, axis=0)
                grip_delta = float(grip_wp[-1, 0] - grip_wp[0, 0])
                grip_velocity = float(vel_token[grip_off])
                if abs(grip_delta) > 1e-9 and abs(grip_velocity) > 1e-8:
                    grip_duration = abs(grip_delta / grip_velocity)
                    grip_progress = np.minimum(
                        np.arange(h, dtype=np.float64) * dt / grip_duration, 1.0
                    )
                    grip_t = np.interp(
                        grip_progress,
                        np.linspace(0.0, 1.0, M),
                        grip_wp[:, 0],
                    )[:, None]
                else:
                    grip_t = np.repeat(grip_wp[:1], h, axis=0)
            else:
                # Same reconstruction as BimanualArcLengthTokenizer.detoken-
                # ize, but convert chord velocity into path speed so curved
                # waypoint paths retain the encoded chord/time duration.
                chord = float(np.linalg.norm(xyz_wp[-1] - xyz_wp[0]))
                if chord < 1e-9:
                    path_speed = speed
                else:
                    duration = chord / speed
                    path_speed = total / max(duration, 1e-8)
                s = np.minimum(
                    path_speed * np.arange(h, dtype=np.float64) * dt, total
                )
                pos_t = np.stack(
                    [_interp_pos_at_s(xyz_wp, cumdist, float(sk)) for sk in s]
                )
                grip_t = np.stack(
                    [_interp_linear_at_s(grip_wp, cumdist, float(sk)) for sk in s]
                )
            arms_out.append(np.concatenate([pos_t, grip_t], axis=-1))
        return np.concatenate(arms_out, axis=-1)  # (H, 8)


# Planar U-socket layout.  Waypoint rows are [x, y, cos(theta), sin(theta)]
# and the final kinematics row is [vx, vy, omega, arc_speed].  The final row is
# intentionally normalized slotwise rather than as another pose waypoint.
USOCKET_ARC_DIM = 4


class TokenizeUSocketArcLength:
    """Tokenize a planar U-socket trajectory by SE(2) arc length.

    The input is a time-indexed ``(T, 3)`` action chunk ``[x, y, theta]``.
    Rows ``0..M-1`` of the output are uniformly spaced along a combined planar
    arc whose segment length is

    ``sqrt(dx**2 + dy**2 + (rotation_radius * dtheta)**2)``.

    Including rotation in the metric is essential for U-socket: a physically
    meaningful rotate-in-place segment must not collapse to a stationary
    translation token.  The final row stores ``[vx, vy, omega, arc_speed]`` so
    the fixed-rate rollout adapter can restore timing.
    """

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 25,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 40.0,
        zero_dist_epsilon: float = 1e-6,
    ):
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.min_distance_unit = float(min_distance_unit)
        self.resampled_vector_length = int(resampled_vector_length)
        self.dt = float(dt)
        self.rotation_radius = float(rotation_radius)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        if self.min_distance_unit <= 0:
            raise ValueError("min_distance_unit must be positive")
        if self.resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least 2")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.rotation_radius < 0:
            raise ValueError("rotation_radius must be non-negative")

    @property
    def M(self) -> int:
        return self.resampled_vector_length

    def _arc_parameter(
        self, xy: np.ndarray, theta_unwrapped: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dxy = np.diff(xy, axis=0)
        dtheta = np.diff(theta_unwrapped)
        step = np.sqrt(
            np.square(dxy).sum(axis=-1)
            + np.square(self.rotation_radius * dtheta)
        )
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(step)]
        )
        return cumulative, step

    @staticmethod
    def _interp(values: np.ndarray, cumulative: np.ndarray, target: float):
        return _interp_linear_at_s(values, cumulative, float(target))

    def transform(self, batch: dict) -> dict:
        input_actions = np.asarray(batch[self.action_key])
        if input_actions.ndim != 2 or input_actions.shape[1] != 3:
            raise ValueError(
                "TokenizeUSocketArcLength expects (T, 3) [x, y, theta], "
                f"got {input_actions.shape}"
            )
        if len(input_actions) < 2:
            raise ValueError("TokenizeUSocketArcLength needs at least two steps")
        if not np.isfinite(input_actions).all():
            raise ValueError(f"{self.action_key} contains non-finite values")

        actions = input_actions.astype(np.float64, copy=False)
        xy = actions[:, :2]
        theta = np.unwrap(actions[:, 2])
        cumulative, step = self._arc_parameter(xy, theta)
        total = float(cumulative[-1])
        covered = min(total, self.min_distance_unit)

        if total <= self.zero_dist_epsilon:
            xy_waypoints = np.repeat(xy[:1], self.M, axis=0)
            theta_waypoints = np.repeat(theta[:1], self.M)
            velocity = np.zeros(USOCKET_ARC_DIM, dtype=np.float64)
        else:
            targets = np.linspace(0.0, covered, self.M)
            xy_waypoints = np.stack(
                [self._interp(xy, cumulative, target) for target in targets]
            )
            theta_waypoints = np.array(
                [
                    self._interp(theta[:, None], cumulative, target)[0]
                    for target in targets
                ],
                dtype=np.float64,
            )

            if covered < total - self.zero_dist_epsilon:
                segment, alpha = _bracket_segment(cumulative, covered)
                duration_steps = float(segment) + float(alpha)
            else:
                moving = np.flatnonzero(step > self.zero_dist_epsilon)
                duration_steps = float(moving[-1] + 1) if moving.size else 0.0
            duration = max(duration_steps * self.dt, self.dt)
            delta_xy = xy_waypoints[-1] - xy_waypoints[0]
            delta_theta = theta_waypoints[-1] - theta_waypoints[0]
            velocity = np.array(
                [
                    delta_xy[0] / duration,
                    delta_xy[1] / duration,
                    delta_theta / duration,
                    covered / duration,
                ],
                dtype=np.float64,
            )

        waypoints = np.column_stack(
            [
                xy_waypoints,
                np.cos(theta_waypoints),
                np.sin(theta_waypoints),
            ]
        )
        output = np.concatenate([waypoints, velocity[None]], axis=0)
        output_dtype = (
            input_actions.dtype
            if np.issubdtype(input_actions.dtype, np.floating)
            else np.float32
        )
        batch[self.output_action_key] = output.astype(output_dtype, copy=False)
        return batch

    def detokenize(self, arc_actions: np.ndarray, action_horizon: int) -> np.ndarray:
        """Decode ``(M+1, 4)`` tokens to fixed-rate ``(H, 3)`` actions."""
        value = np.asarray(arc_actions, dtype=np.float64)
        expected = (self.M + 1, USOCKET_ARC_DIM)
        if value.shape != expected:
            raise ValueError(
                f"TokenizeUSocketArcLength.detokenize expects {expected}, "
                f"got {value.shape}"
            )
        horizon = int(action_horizon)
        if horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not np.isfinite(value).all():
            raise ValueError("arc_actions contains non-finite values")

        waypoints = value[: self.M]
        kinematics = value[self.M]
        xy = waypoints[:, :2]
        rotvec = waypoints[:, 2:4]
        rot_norm = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        safe_rotvec = np.divide(
            rotvec,
            rot_norm,
            out=np.tile(np.array([[1.0, 0.0]]), (self.M, 1)),
            where=rot_norm > self.zero_dist_epsilon,
        )
        theta = np.unwrap(np.arctan2(safe_rotvec[:, 1], safe_rotvec[:, 0]))
        cumulative, _ = self._arc_parameter(xy, theta)
        total = float(cumulative[-1])

        path_speed = abs(float(kinematics[3]))
        if path_speed <= self.zero_dist_epsilon and total > self.zero_dist_epsilon:
            durations = []
            xy_chord = float(np.linalg.norm(xy[-1] - xy[0]))
            xy_speed = float(np.linalg.norm(kinematics[:2]))
            if xy_chord > self.zero_dist_epsilon and xy_speed > self.zero_dist_epsilon:
                durations.append(xy_chord / xy_speed)
            theta_span = abs(float(theta[-1] - theta[0]))
            omega = abs(float(kinematics[2]))
            if theta_span > self.zero_dist_epsilon and omega > self.zero_dist_epsilon:
                durations.append(theta_span / omega)
            if durations:
                path_speed = total / max(float(np.median(durations)), self.dt)

        if total <= self.zero_dist_epsilon or path_speed <= self.zero_dist_epsilon:
            xy_out = np.repeat(xy[:1], horizon, axis=0)
            theta_out = np.repeat(theta[:1], horizon)
        else:
            targets = np.minimum(
                path_speed * np.arange(horizon, dtype=np.float64) * self.dt,
                total,
            )
            xy_out = np.stack(
                [self._interp(xy, cumulative, target) for target in targets]
            )
            theta_out = np.array(
                [
                    self._interp(theta[:, None], cumulative, target)[0]
                    for target in targets
                ]
            )
        theta_out = np.angle(np.exp(1j * theta_out))
        return np.column_stack([xy_out, theta_out])


# ChainGripper point rows and their mean-velocity payload use the invertible
# middle-joint-anchored embedding defined below.
CHAIN_GRIPPER_POINT_ARC_DIM = 6
_CHAIN_GRIPPER_RELATIVE_SCALE = np.sqrt(2.0)


def chain_gripper_points_to_arc_embedding(points: np.ndarray) -> np.ndarray:
    """Map ordered points to orthogonal translation and shape coordinates.

    For ``P = [L, C, R]`` this returns
    ``Phi(P) = [C, (L-C)/sqrt(2), (R-C)/sqrt(2)]``.  Center translation is
    represented only by the first two coordinates, while the other four
    coordinates describe shape relative to the middle joint.
    """
    value = np.asarray(points)
    if value.ndim == 0 or value.shape[-1] != CHAIN_GRIPPER_POINT_ARC_DIM:
        raise ValueError(
            "chain_gripper_points_to_arc_embedding expects last dimension 6, "
            f"got {value.shape}"
        )
    left, center, right = value[..., 0:2], value[..., 2:4], value[..., 4:6]
    return np.concatenate(
        (
            center,
            (left - center) / _CHAIN_GRIPPER_RELATIVE_SCALE,
            (right - center) / _CHAIN_GRIPPER_RELATIVE_SCALE,
        ),
        axis=-1,
    )


def chain_gripper_arc_embedding_to_points(embedding: np.ndarray) -> np.ndarray:
    """Invert :func:`chain_gripper_points_to_arc_embedding` exactly."""
    value = np.asarray(embedding)
    if value.ndim == 0 or value.shape[-1] != CHAIN_GRIPPER_POINT_ARC_DIM:
        raise ValueError(
            "chain_gripper_arc_embedding_to_points expects last dimension 6, "
            f"got {value.shape}"
        )
    center = value[..., 0:2]
    left = center + _CHAIN_GRIPPER_RELATIVE_SCALE * value[..., 2:4]
    right = center + _CHAIN_GRIPPER_RELATIVE_SCALE * value[..., 4:6]
    return np.concatenate((left, center, right), axis=-1)


def chain_gripper_point_step_norm(delta: np.ndarray) -> np.ndarray:
    """Euclidean step length in the middle-joint-anchored embedding.

    Since ``Phi`` is linear, applying it to a point displacement gives
    ``[dC, (dL-dC)/sqrt(2), (dR-dC)/sqrt(2)]``. Translation and relative-shape
    motion therefore contribute in separate coordinates. A rigid translation
    by ``d`` has length ``||d||``, while tip articulation remains visible.
    """
    value = np.asarray(delta, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != CHAIN_GRIPPER_POINT_ARC_DIM:
        raise ValueError(
            f"chain_gripper_point_step_norm expects last dimension 6, got {value.shape}"
        )
    embedding_delta = chain_gripper_points_to_arc_embedding(value)
    return np.linalg.norm(embedding_delta, axis=-1)


class TokenizeChainGripperPointArcLength:
    """Tokenize ordered ChainGripper points in anchored arc coordinates.

    Input rows are ``[left_tip_xy, middle_joint_xy, right_tip_xy]`` and are
    first mapped through ``Phi(P) = [C, (L-C)/sqrt(2), (R-C)/sqrt(2)]``.
    Segment length, interpolation, waypoint rows ``0..M-1``, and the final
    six-dimensional mean-velocity row all use this same ``Phi`` basis.
    Detokenization reconstructs fixed-rate ``Phi`` commands and inverts them
    to ordered points before the rollout adapter performs kinematic projection.
    """

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 25,
        dt: float = 1.0 / 30.0,
        zero_dist_epsilon: float = 1e-6,
    ):
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.min_distance_unit = float(min_distance_unit)
        self.resampled_vector_length = int(resampled_vector_length)
        self.dt = float(dt)
        self.zero_dist_epsilon = float(zero_dist_epsilon)
        if self.min_distance_unit <= 0:
            raise ValueError("min_distance_unit must be positive")
        if self.resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least 2")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def M(self) -> int:
        return self.resampled_vector_length

    @staticmethod
    def _interp(values: np.ndarray, cumulative: np.ndarray, target: float):
        return _interp_linear_at_s(values, cumulative, float(target))

    @staticmethod
    def _arc_parameter(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step = np.linalg.norm(np.diff(embedding, axis=0), axis=-1)
        cumulative = np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(step)])
        return cumulative, step

    def transform(self, batch: dict) -> dict:
        input_actions = np.asarray(batch[self.action_key])
        if input_actions.ndim != 2 or input_actions.shape[1] != 6:
            raise ValueError(
                "TokenizeChainGripperPointArcLength expects (T, 6) ordered "
                f"points, got {input_actions.shape}"
            )
        if len(input_actions) < 2:
            raise ValueError(
                "TokenizeChainGripperPointArcLength needs at least two steps"
            )
        if not np.isfinite(input_actions).all():
            raise ValueError(f"{self.action_key} contains non-finite values")

        points = input_actions.astype(np.float64, copy=False)
        embedding = chain_gripper_points_to_arc_embedding(points)
        cumulative, step = self._arc_parameter(embedding)
        total = float(cumulative[-1])
        covered = min(total, self.min_distance_unit)

        if total <= self.zero_dist_epsilon:
            waypoints = np.repeat(embedding[:1], self.M, axis=0)
            velocity = np.zeros(CHAIN_GRIPPER_POINT_ARC_DIM, dtype=np.float64)
        else:
            targets = np.linspace(0.0, covered, self.M)
            waypoints = np.stack(
                [self._interp(embedding, cumulative, target) for target in targets]
            )
            if covered < total - self.zero_dist_epsilon:
                segment, alpha = _bracket_segment(cumulative, covered)
                duration_steps = float(segment) + float(alpha)
            else:
                moving = np.flatnonzero(step > self.zero_dist_epsilon)
                duration_steps = float(moving[-1] + 1) if moving.size else 0.0
            duration = max(duration_steps * self.dt, self.dt)
            velocity = (waypoints[-1] - waypoints[0]) / duration

        output = np.concatenate([waypoints, velocity[None]], axis=0)
        output_dtype = (
            input_actions.dtype
            if np.issubdtype(input_actions.dtype, np.floating)
            else np.float32
        )
        batch[self.output_action_key] = output.astype(output_dtype, copy=False)
        return batch

    def detokenize(self, arc_actions: np.ndarray, action_horizon: int) -> np.ndarray:
        """Decode ``(M+1, 6)`` Phi tokens to fixed-rate ``(H, 6)`` points."""
        value = np.asarray(arc_actions, dtype=np.float64)
        expected = (self.M + 1, CHAIN_GRIPPER_POINT_ARC_DIM)
        if value.shape != expected:
            raise ValueError(
                "TokenizeChainGripperPointArcLength.detokenize expects "
                f"{expected}, got {value.shape}"
            )
        horizon = int(action_horizon)
        if horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not np.isfinite(value).all():
            raise ValueError("arc_actions contains non-finite values")

        waypoints = value[: self.M]
        mean_velocity = value[self.M]
        cumulative, _ = self._arc_parameter(waypoints)
        total = float(cumulative[-1])
        chord = float(np.linalg.norm(waypoints[-1] - waypoints[0]))
        chord_speed = float(np.linalg.norm(mean_velocity))

        if total <= self.zero_dist_epsilon:
            embedding_out = np.repeat(waypoints[:1], horizon, axis=0)
            return chain_gripper_arc_embedding_to_points(embedding_out)
        if chord > self.zero_dist_epsilon and chord_speed > self.zero_dist_epsilon:
            duration = chord / chord_speed
        else:
            # A closed loop has zero mean coordinate velocity, so its exact
            # duration is not recoverable from one mean-velocity row. Spread it
            # over the requested horizon instead of freezing a nonzero path.
            duration = max((horizon - 1) * self.dt, self.dt)
        path_speed = total / max(duration, self.dt)
        targets = np.minimum(
            path_speed * np.arange(horizon, dtype=np.float64) * self.dt,
            total,
        )
        embedding_out = np.stack(
            [self._interp(waypoints, cumulative, target) for target in targets]
        )
        return chain_gripper_arc_embedding_to_points(embedding_out)


# --------------------------------------------------------------------------
# Planar (PushShapes) arc tokenization
#
# Mirrors groot/core/utils/state_action/arc_length_distance.py and
# groot/core/data/state_action/arc_length_action_transform.py on
# rpunamiya/arc-length-tokenizer, reduced to SE(2):
#   * a token is emitted for EVERY timestep (tokenize_at(t)), not for a
#     partition of the episode;
#   * the step distance is additive, ||dp|| + lambda * mu(dtheta), so it is a
#     metric on SE(2) and its cumulative sum is a genuine arc length;
#   * lambda is expressed through a radius, lambda = 2*sqrt(2)*r, so the knob
#     is "how far from the pivot do I care about" rather than a bare weight;
#   * gripper rides the SAME waypoints as the pose and is repeated when the
#     token carries no motion.
# --------------------------------------------------------------------------

PLANAR_POSE_DIM = 4   # [x, y, cos, sin]
PLANAR_ARC_DIM = 5    # [x, y, cos, sin, grip]


def lambda_for_radius(radius: float) -> float:
    """Weight making the rotation term equal the arc swept at ``radius``.

    ``mu`` is dimensionless. For small angles mu ~= theta / (2*sqrt(2)), and a
    point at distance r from the pivot sweeps r*theta, so equating the two
    gives lambda = 2*sqrt(2)*r. The parameter to reason about is therefore the
    radius at which rotation should "cost" as much as translation -- for a
    PushShapes effector whose contact face sits ~30 units from its axis, that
    is ``lambda_for_radius(30)``.
    """
    return 2.0 * math.sqrt(2.0) * float(radius)


def rotation_step_metric_planar(theta: np.ndarray) -> np.ndarray:
    """Per-step mu for a planar angle sequence -> (N-1,).

    mu(R_a, R_b) = sqrt(2) * sin(dtheta / 4), the scaled chordal metric on
    SO(3) restricted to rotations about one axis. Using it rather than raw
    |dtheta| keeps the planar tokenizer on the same metric as the 3D one, so a
    radius tuned in one transfers to the other.
    """
    d = np.abs(np.diff(np.asarray(theta, dtype=np.float64)))
    return math.sqrt(2.0) * np.sin(d / 4.0)


def planar_step_distance(xy: np.ndarray, theta: np.ndarray | None = None,
                         lambda_rot: float = 0.0) -> np.ndarray:
    """Per-step SE(2) distance ||dp|| + lambda * mu(dtheta) -> (N-1,).

    ``lambda_rot = 0`` gives pure translational arc length, which is the
    behaviour to use when rotation should ride along the path rather than
    consume its budget.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (N, 2), got {xy.shape}")
    if len(xy) < 2:
        return np.zeros((0,), dtype=np.float64)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    if lambda_rot > 0.0:
        if theta is None:
            raise ValueError("lambda_rot > 0 requires theta")
        if len(theta) != len(xy):
            raise ValueError(f"length mismatch: xy {len(xy)} vs theta {len(theta)}")
        step = step + float(lambda_rot) * rotation_step_metric_planar(theta)
    return step


def planar_common_horizon(distances, velocities, velocity_epsilon: float = 1e-6) -> float:
    """H_valid = min_s (D_s / v_s) over streams that actually move.

    A stream with v ~= 0 imposes no limit and simply holds its first waypoint;
    including it would contribute D/0. If every stream is degenerate this
    returns 0.0, which callers treat as "hold".
    """
    if len(distances) != len(velocities):
        raise ValueError("distances and velocities must have equal length")
    live = [float(d) / float(v) for d, v in zip(distances, velocities)
            if float(v) > velocity_epsilon]
    return float(min(live)) if live else 0.0


class PadPlanarAction:
    """Widen any PushShapes action to a common ``[x, y, cos, sin, grip]``.

    The 13 effectors do not agree on action width -- 4 emit ``[x, y]``, 3 emit
    ``[x, y, theta]``, 6 emit ``[x, y, theta, grip]`` -- so a co-trained policy
    has no single action head until they are padded to one layout. Absent
    channels get the identity value for that channel: theta 0 (cos=1, sin=0)
    and grip 0 (open), which is what those effectors physically do.
    """

    def __init__(self, keys: list[str] | None = None):
        self.keys = list(keys or ["actions"])

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if key not in batch:
                continue
            v = np.asarray(batch[key])
            if v.ndim != 2 or v.shape[-1] not in (2, 3, 4):
                raise ValueError(
                    f"PadPlanarAction expects (T, 2|3|4) for '{key}', got {v.shape}"
                )
            f = v.astype(np.float64, copy=False)
            T, C = f.shape
            theta = f[:, 2] if C >= 3 else np.zeros(T)
            grip = f[:, 3] if C >= 4 else np.zeros(T)
            out = np.column_stack([f[:, 0], f[:, 1], np.cos(theta), np.sin(theta), grip])
            dtype = v.dtype if np.issubdtype(v.dtype, np.floating) else np.float32
            batch[key] = out.astype(dtype, copy=False)
        return batch


class TokenizePlanarArcLength:
    """Per-timestep planar arc tokenizer for PushShapes actions.

    Accepts ``(T, 2|3|4)`` = ``[x, y[, theta[, grip]]]`` and emits one token
    anchored at timestep ``t``: M waypoints uniform in arc length over
    ``[s(t), s(t) + D]`` plus a velocity payload.

    rotation_radius
        0 (default) measures arc by translation alone and lets theta ride the
        path. >0 adds ``lambda_for_radius(r) * mu(dtheta)`` to the step, so a
        rotate-in-place manoeuvre advances the clock instead of measuring zero.

    hybrid_rotation_unit
        If set, theta gets its OWN budget and the token spans the common
        horizon min(D/v_trans, D_rot/v_rot); waypoints are still placed by xy
        distance, so the rotation stream only truncates the window.

    velocity_mode
        ``mean_scalar`` stores one mean speed; ``per_step_scalar`` stores a
        speed per waypoint.
    velocity_layout
        ``append`` adds the velocity as an extra row -> (M+1, 5).
        ``concat`` appends it to every waypoint -> (M, 5+V).

    Waypoint 0 is taken EXPLICITLY from index ``t`` rather than by arc lookup.
    In a stationary cluster many timesteps share one cumulative arc value, so
    the lookup returns the first of them and silently drops a grip transition
    that happens while the effector is not moving -- measured as exactly one
    row per episode, and it was the grasp trigger, which took a stride-1
    replay from 100% to 0%.
    """

    def __init__(
        self,
        action_key: str = "actions",
        output_action_key: str = "actions",
        min_distance_unit: float = 50.0,
        resampled_vector_length: int = 50,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 0.0,
        hybrid_rotation_unit: float | None = None,
        velocity_mode: str = "mean_scalar",
        velocity_layout: str = "append",
        zero_dist_epsilon: float = 1e-9,
    ):
        if min_distance_unit <= 0:
            raise ValueError("min_distance_unit must be positive")
        if resampled_vector_length < 2:
            raise ValueError("resampled_vector_length must be at least 2")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if rotation_radius < 0:
            raise ValueError("rotation_radius must be non-negative")
        if velocity_mode not in ("mean_scalar", "per_step_scalar"):
            raise ValueError(f"unknown velocity_mode {velocity_mode!r}")
        if velocity_layout not in ("append", "concat"):
            raise ValueError(f"unknown velocity_layout {velocity_layout!r}")
        self.action_key = str(action_key)
        self.output_action_key = str(output_action_key)
        self.min_distance_unit = float(min_distance_unit)
        self.M = int(resampled_vector_length)
        self.dt = float(dt)
        self.rotation_radius = float(rotation_radius)
        self.hybrid_rotation_unit = (None if hybrid_rotation_unit is None
                                     else float(hybrid_rotation_unit))
        self.velocity_mode = str(velocity_mode)
        self.velocity_layout = str(velocity_layout)
        self.zero_dist_epsilon = float(zero_dist_epsilon)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _split(actions: np.ndarray):
        n, C = actions.shape
        xy = actions[:, :2]
        theta = np.unwrap(actions[:, 2]) if C >= 3 else np.zeros(n)
        grip = actions[:, 3] if C >= 4 else np.zeros(n)
        return xy, theta, grip, C

    def _cumulative(self, xy, theta):
        lam = lambda_for_radius(self.rotation_radius) if self.rotation_radius > 0 else 0.0
        step = planar_step_distance(xy, theta, lam)
        return np.concatenate([np.zeros(1), np.cumsum(step)])

    def _window_end(self, cum, xy, theta, t):
        """Arc coordinate where this token stops."""
        end = cum[t] + self.min_distance_unit
        if self.hybrid_rotation_unit is not None:
            # Rotation gets its own budget; the token spans the common horizon.
            rot = np.concatenate([np.zeros(1),
                                  np.cumsum(rotation_step_metric_planar(theta))])
            span_t = float(cum[-1] - cum[t])
            span_r = float(rot[-1] - rot[t])
            if span_r > self.zero_dist_epsilon and span_t > self.zero_dist_epsilon:
                frac = min(1.0, self.hybrid_rotation_unit / span_r)
                end = min(end, cum[t] + frac * span_t)
        return min(end, float(cum[-1]))

    def tokenize_at(self, actions: np.ndarray, t: int) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        n, C = actions.shape
        xy, theta, grip, C = self._split(actions)
        cum = self._cumulative(xy, theta)
        end = self._window_end(cum, xy, theta, t)
        covered = max(0.0, end - cum[t])

        if covered <= self.zero_dist_epsilon:
            # No motion in this token: hold the pose and REPEAT the gripper,
            # matching the reference's zero-motion collapse.
            xy_w = np.repeat(xy[t:t + 1], self.M, axis=0)
            th_w = np.repeat(theta[t], self.M)
            gr_w = np.repeat(grip[t], self.M)
            speed = np.zeros(self.M if self.velocity_mode == "per_step_scalar" else 1)
        else:
            targets = np.linspace(cum[t], end, self.M)
            xy_w = np.stack([_interp_linear_at_s(xy, cum, s) for s in targets])
            th_w = np.array([_interp_linear_at_s(theta[:, None], cum, s)[0] for s in targets])
            gr_w = np.array([_interp_linear_at_s(grip[:, None], cum, s)[0] for s in targets])
            seg = int(np.searchsorted(cum, end, side="left"))
            steps = max(1, seg - t)
            duration = max(steps * self.dt, self.dt)
            if self.velocity_mode == "mean_scalar":
                speed = np.array([covered / duration])
            else:
                per = np.diff(targets) / max(duration / max(self.M - 1, 1), self.dt)
                speed = np.concatenate([per[:1], per])

        # Anchor: waypoint 0 is the action AT t, never an arc lookup.
        xy_w[0] = xy[t]; th_w[0] = theta[t]; gr_w[0] = grip[t]

        way = np.column_stack([xy_w, np.cos(th_w), np.sin(th_w), gr_w])
        if self.velocity_layout == "append":
            vrow = np.zeros((1, PLANAR_ARC_DIM))
            vrow[0, 0] = float(np.mean(speed))
            return np.concatenate([way, vrow], axis=0)
        return np.concatenate(
            [way, np.repeat(speed[:, None], 1, axis=1) if len(speed) == self.M
             else np.repeat(np.asarray(speed).reshape(1, 1), self.M, axis=0)], axis=1)

    def transform(self, batch: dict) -> dict:
        raw = np.asarray(batch[self.action_key])
        if raw.ndim != 2 or raw.shape[1] not in (2, 3, 4):
            raise ValueError(
                f"TokenizePlanarArcLength expects (T, 2|3|4), got {raw.shape}")
        if len(raw) < 2:
            raise ValueError("needs at least two steps")
        if not np.isfinite(raw).all():
            raise ValueError(f"{self.action_key} contains non-finite values")
        out = self.tokenize_at(raw.astype(np.float64, copy=False), 0)
        dtype = raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.float32
        batch[self.output_action_key] = out.astype(dtype, copy=False)
        return batch

    def decode_first_action(self, token: np.ndarray, width: int) -> np.ndarray:
        """The action to execute now: waypoint 0, narrowed to native width."""
        w = np.asarray(token, dtype=np.float64)[0]
        theta = math.atan2(float(w[3]), float(w[2]))
        full = np.array([w[0], w[1], theta, w[4]], dtype=np.float64)
        return full[:width]


class ResidualizeArcTokenXY:
    """Re-express an arc token's xy waypoints RELATIVE to the pusher's position.

    PushShapes actions are ABSOLUTE pusher targets, and the pusher's own xy is
    channels 0-1 of ``state_agent_obj``. The planner emits
    ``tx = px + clip(gx-ox, -3, 3)`` -- position plus a <=3px delta -- so a
    persistence baseline ("command where I already am") already scores FVE
    0.85-0.95, with a median error of 0.00px on the ideal control gap.

    Training MSE on the absolute target therefore spends almost all of its
    gradient on a quantity the policy can copy straight off its own input, and
    almost none on the small delta that actually drives the task. Measured
    consequence: every arm over-commands by 3-12x, leaves the data manifold
    within ~50 rollout steps, and settles into an absorbing fixed point where
    the commanded target equals the current position -- the pusher stops, the
    object is never engaged, and coverage sits at exactly 0.0000.

    Subtracting the anchor here makes the delta the learning target. Because
    norm stats are computed downstream from whatever this emits, the deltas also
    get normalized on THEIR own scale (~10px) rather than the absolute position
    range (~500px). That rescaling is the point: adding the anchor back inside
    the model without it leaves a 3px delta at ~0.012 in normalized units and
    the loss just as badly allocated.

    Only xy is residualized. cos/sin/grip are already local quantities, and the
    trailing velocity row is a rate, not a position -- shifting either would
    corrupt them.
    """

    def __init__(
        self,
        action_key: str = "actions",
        state_key: str = "state_agent_obj",
        velocity_layout: str = "append",
    ):
        self.action_key = str(action_key)
        self.state_key = str(state_key)
        self.velocity_layout = str(velocity_layout)
        if self.velocity_layout not in {"append", "concat"}:
            raise ValueError(
                f"velocity_layout must be append|concat, got {velocity_layout!r}"
            )

    def transform(self, batch: dict) -> dict:
        if self.state_key not in batch:
            raise KeyError(
                f"ResidualizeArcTokenXY needs {self.state_key!r} in the batch to "
                f"anchor the token; present keys: {sorted(batch)}"
            )
        token = np.asarray(batch[self.action_key])
        if token.ndim != 2 or token.shape[1] < 2:
            raise ValueError(
                f"expected a tokenized (rows, channels) action, got {token.shape}"
            )
        state = np.asarray(batch[self.state_key]).reshape(-1)
        if state.shape[0] < 2:
            raise ValueError(
                f"{self.state_key} must carry pusher xy in channels 0-1, "
                f"got shape {state.shape}"
            )
        anchor = state[:2].astype(np.float64)

        out = token.astype(np.float64, copy=True)
        # 'append' puts a trailing velocity row after the M waypoints; it is a
        # rate and must not be shifted. 'concat' keeps velocity in extra
        # CHANNELS, so every row is a waypoint.
        n_way = out.shape[0] - 1 if self.velocity_layout == "append" else out.shape[0]
        out[:n_way, 0] -= anchor[0]
        out[:n_way, 1] -= anchor[1]

        dtype = token.dtype if np.issubdtype(token.dtype, np.floating) else np.float32
        batch[self.action_key] = out.astype(dtype, copy=False)
        return batch
