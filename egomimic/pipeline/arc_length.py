"""Inference adapters for arc-length action tokens."""

from __future__ import annotations

import numpy as np
import torch

from egomimic.rldb.zarr.arc_length_tokenizer import (
    ARC_TOK_BIMANUAL_DIM,
    TokenizeBimanualArcLengthCartesian,
)


class ArcLengthRolloutAdapter:
    """Decode ``M`` arc waypoints plus one velocity token for control.

    The tokenizer predicts eight values per slot:
    ``[L xyz, L grip, R xyz, R grip]``.  Rollout reconstructs a fixed-rate
    trajectory and expands it to the repository's 14-D bimanual cartesian
    layout.  Arc tokens do not contain rotation, so each arm holds the current
    orientation from ``state_ee_pose`` throughout the decoded chunk.
    """

    requires_bimanual = True
    requires_cartesian = True
    preserves_decoded_timing = True

    def __init__(
        self,
        min_distance_unit: float,
        resampled_vector_length: int,
        action_horizon: int = 100,
        dt: float = 1.0 / 30.0,
        state_key: str = "state_ee_pose",
    ):
        self.min_distance_unit = float(min_distance_unit)
        self.resampled_vector_length = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.dt = float(dt)
        self.state_key = str(state_key)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.detokenizer = TokenizeBimanualArcLengthCartesian(
            action_key="actions_cartesian",
            output_action_key="actions_cartesian",
            min_distance_unit=self.min_distance_unit,
            resampled_vector_length=self.resampled_vector_length,
            dt=self.dt,
        )

    def _current_ypr(
        self, context: dict | None, batch_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(3, dtype=np.float64)
        if not context or context.get(self.state_key) is None:
            return zeros, zeros
        state = context[self.state_key]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state)
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim == 1:
            state = state[None]
        row = state[batch_index]
        if row.shape[-1] == 14:
            return row[3:6].astype(np.float64), row[10:13].astype(np.float64)
        if row.shape[-1] == 12:
            return row[3:6].astype(np.float64), row[9:12].astype(np.float64)
        raise ValueError(
            "state_ee_pose must be 12-D [two xyz+ypr poses] or 14-D "
            "[two xyz+ypr+gripper poses] for arc-token rollout; "
            f"got {row.shape[-1]}"
        )

    def decode(self, arc_tokens, context: dict | None = None):
        """Return a ``(B, H, 14)`` fixed-rate bimanual cartesian chunk."""
        input_is_tensor = torch.is_tensor(arc_tokens)
        if input_is_tensor:
            device, dtype = arc_tokens.device, arc_tokens.dtype
            tokens_np = arc_tokens.detach().cpu().numpy().astype(np.float64)
        else:
            tokens_np = np.asarray(arc_tokens, dtype=np.float64)
            device = dtype = None
        if tokens_np.ndim == 2:
            tokens_np = tokens_np[None]
        expected = (self.resampled_vector_length + 1, ARC_TOK_BIMANUAL_DIM)
        if tokens_np.ndim != 3 or tuple(tokens_np.shape[1:]) != expected:
            raise ValueError(
                f"ArcLengthRolloutAdapter expects (B, {expected[0]}, "
                f"{expected[1]}), got {tokens_np.shape}"
            )

        output = np.zeros(
            (tokens_np.shape[0], self.action_horizon, 14), dtype=np.float64
        )
        for batch_index, token in enumerate(tokens_np):
            decoded = self.detokenizer.detokenize(
                token, action_horizon=self.action_horizon
            )
            left_ypr, right_ypr = self._current_ypr(context, batch_index)
            output[batch_index, :, 0:3] = decoded[:, 0:3]
            output[batch_index, :, 3:6] = left_ypr
            output[batch_index, :, 6:7] = decoded[:, 3:4]
            output[batch_index, :, 7:10] = decoded[:, 4:7]
            output[batch_index, :, 10:13] = right_ypr
            output[batch_index, :, 13:14] = decoded[:, 7:8]

        if input_is_tensor:
            return torch.from_numpy(output).to(device=device, dtype=dtype)
        return output.astype(np.float32)

    __call__ = decode
