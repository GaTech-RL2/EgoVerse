"""2-D arc-length tokenizer for the PushShapes push-T sim.

Adapts the bimanual arc tokenizer (arc_length_tokenizer.py, 14-D) to 2-D
pusher-target trajectories. Given a raw window of (T, 2) pusher targets in
world PIXELS, produces a (M+1, 2) chunk:

  slots 0..M-1 : M waypoints uniformly spaced in arc length over the first
                 min_distance_unit pixels of the trajectory.
  slot M       : velocity token = [x_vel, y_vel] mean per-dim rate across
                 the chunk (pixels/second).

Unit convention: PushShapes actions live in world pixels (WORLD_SIZE = 512).
min_distance_unit is in the SAME units as the action stream (pixels).

The velocity token is used at inference to reconstruct the time schedule:
    duration_s = D / ||mean_xy_vel||
Then the M waypoints are linear-resampled uniformly over [0, duration_s]
at 30 Hz. See ``expand_arc_chunk_to_time`` below and
``egomimic.algo.hpt.algo.HPT.inference_step`` (arc_tok branch).
"""

from __future__ import annotations

import numpy as np

from egomimic.rldb.zarr.action_chunk_transforms import Transform


def _cumulative_arc_length_2d(pos: np.ndarray) -> np.ndarray:
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    return np.concatenate([[0.0], np.cumsum(step)])


def _interp_at_arc(
    pos: np.ndarray, cumdist: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Linear-interp (N, 2) positions at arc-length targets (M,) -> (M, 2)."""
    total = float(cumdist[-1])
    out = np.zeros((len(targets), pos.shape[1]), dtype=np.float64)
    for k, s in enumerate(targets):
        s = float(s)
        if s <= cumdist[0]:
            out[k] = pos[0]
        elif s >= total:
            out[k] = pos[-1]
        else:
            i = int(np.searchsorted(cumdist, s, side="left") - 1)
            i = max(0, min(i, len(cumdist) - 2))
            s0, s1 = cumdist[i], cumdist[i + 1]
            alpha = 0.0 if (s1 - s0) <= 1e-12 else (s - s0) / (s1 - s0)
            out[k] = (1.0 - alpha) * pos[i] + alpha * pos[i + 1]
    return out


class TokenizePushShapesArcLength(Transform):
    """Rewrite the ``actions`` key from (T, 2) time-indexed to (M+1, 2) arc-tok.

    Applied per sample by the dataloader (MultiDataset transform_list). The
    raw window ``T`` should be wide enough that ``min_distance_unit`` of arc
    length usually fits inside it; trajectories shorter than that produce a
    zero-velocity token (first waypoint held for all M slots, vel = 0) so
    training stays well-defined near episode boundaries.
    """

    def __init__(
        self,
        action_key: str = "actions",
        min_distance_unit: float = 100.0,
        resampled_vector_length: int = 40,
        dt: float = 1.0 / 30.0,
    ):
        if resampled_vector_length < 2:
            raise ValueError(
                f"resampled_vector_length must be >= 2, got {resampled_vector_length}"
            )
        self.action_key = action_key
        self.D = float(min_distance_unit)
        self.M = int(resampled_vector_length)
        self.dt = float(dt)

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(
                f"TokenizePushShapesArcLength expects (T, 2), got {actions.shape} "
                f"for key '{self.action_key}'"
            )
        if actions.shape[0] < 2:
            waypoints = np.tile(actions[0], (self.M, 1))
            vel = np.zeros(2, dtype=np.float64)
        else:
            cumdist = _cumulative_arc_length_2d(actions)
            total = float(cumdist[-1])
            end_s = min(self.D, total)
            if end_s < 1e-6:
                waypoints = np.tile(actions[0], (self.M, 1))
                vel = np.zeros(2, dtype=np.float64)
            else:
                targets = np.linspace(0.0, end_s, self.M)
                waypoints = _interp_at_arc(actions, cumdist, targets)
                end_idx = int(np.searchsorted(cumdist, end_s, side="right") - 1)
                num_steps = max(1, end_idx)
                duration = num_steps * self.dt
                vel = (waypoints[-1] - waypoints[0]) / max(duration, 1e-8)

        out = np.concatenate([waypoints, vel[None, :]], axis=0).astype(np.float32)
        batch = dict(batch)
        batch[self.action_key] = out
        return batch


def expand_arc_chunk_to_time(
    chunk: np.ndarray,
    min_distance_unit: float,
    dt: float = 1.0 / 30.0,
    max_frames: int = 200,
) -> np.ndarray:
    """Detokenize one (M+1, 2) arc-tok chunk into a time-uniform (T_frames, 2)
    action buffer at ``1/dt`` Hz.

    Reads the trailing velocity token to compute chunk duration:
        duration_s = D / ||mean_xy_vel||
    Linear-resamples the M waypoints uniformly over [0, duration_s] at 30 Hz.

    Args:
        chunk: (M+1, 2) float array. Slots 0..M-1 = waypoints, slot M = vel.
        min_distance_unit: D used at tokenization (pixels for pushshapes).
        dt: control period in seconds (default 1/30 = 30 Hz).
        max_frames: hard cap on emitted frames — guards against pathological
                    expansions when predicted velocity approaches zero.

    Returns:
        (T_frames, 2) float32 array of pusher targets, T_frames >= 1.
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 2:
        raise ValueError(
            f"expand_arc_chunk_to_time expects (M+1, 2), got {chunk.shape}"
        )
    if chunk.shape[0] < 3:
        return chunk[:1].astype(np.float32)

    waypoints = chunk[:-1]  # (M, 2)
    vel = chunk[-1]  # (2,)

    speed = float(np.linalg.norm(vel))
    if speed < 1e-6:
        return waypoints[:1].astype(np.float32)

    duration_s = float(min_distance_unit) / speed
    n_frames = int(np.clip(round(duration_s / float(dt)), 1, int(max_frames)))

    alphas = np.linspace(0.0, 1.0, waypoints.shape[0])
    time_alphas = np.linspace(0.0, 1.0, n_frames)
    out = np.zeros((n_frames, 2), dtype=np.float64)
    out[:, 0] = np.interp(time_alphas, alphas, waypoints[:, 0])
    out[:, 1] = np.interp(time_alphas, alphas, waypoints[:, 1])
    return out.astype(np.float32)
