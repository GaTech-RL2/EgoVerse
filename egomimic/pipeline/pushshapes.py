"""PushShapes-specific adapters for Pipeline policy rollout."""

from __future__ import annotations

import numpy as np
import torch

from egomimic.rldb.zarr.arc_length_tokenizer import (
    USOCKET_ARC_DIM,
    TokenizeUSocketArcLength,
)


class USocketRotVecRolloutAdapter:
    """Decode ``[x, y, cos(theta), sin(theta)]`` into simulator actions."""

    preserves_decoded_timing = True

    def decode(self, actions, context: dict | None = None):
        del context
        if torch.is_tensor(actions):
            if actions.ndim < 2 or actions.shape[-1] != 4:
                raise ValueError(
                    "USocketRotVecRolloutAdapter expects (..., 4) actions, "
                    f"got {tuple(actions.shape)}"
                )
            theta = torch.atan2(actions[..., 3], actions[..., 2])
            return torch.cat((actions[..., :2], theta.unsqueeze(-1)), dim=-1)

        value = np.asarray(actions)
        if value.ndim < 2 or value.shape[-1] != 4:
            raise ValueError(
                "USocketRotVecRolloutAdapter expects (..., 4) actions, "
                f"got {value.shape}"
            )
        theta = np.arctan2(value[..., 3], value[..., 2])
        return np.concatenate((value[..., :2], theta[..., None]), axis=-1)

    __call__ = decode


class USocketArcLengthRolloutAdapter:
    """Decode planar U-socket arc tokens into fixed-rate simulator actions."""

    preserves_decoded_timing = True

    def __init__(
        self,
        min_distance_unit: float = 200.0,
        resampled_vector_length: int = 25,
        action_horizon: int = 100,
        dt: float = 1.0 / 30.0,
        rotation_radius: float = 40.0,
    ):
        self.resampled_vector_length = int(resampled_vector_length)
        self.action_horizon = int(action_horizon)
        self.detokenizer = TokenizeUSocketArcLength(
            min_distance_unit=min_distance_unit,
            resampled_vector_length=self.resampled_vector_length,
            dt=dt,
            rotation_radius=rotation_radius,
        )

    def decode(self, actions, context: dict | None = None):
        del context
        input_is_tensor = torch.is_tensor(actions)
        if input_is_tensor:
            device, dtype = actions.device, actions.dtype
            value = actions.detach().cpu().numpy().astype(np.float64)
        else:
            value = np.asarray(actions, dtype=np.float64)
            device = dtype = None
        if value.ndim == 2:
            value = value[None]
        expected = (self.resampled_vector_length + 1, USOCKET_ARC_DIM)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected:
            raise ValueError(
                "USocketArcLengthRolloutAdapter expects "
                f"(B, {expected[0]}, {expected[1]}), got {value.shape}"
            )
        decoded = np.stack(
            [
                self.detokenizer.detokenize(token, self.action_horizon)
                for token in value
            ]
        )
        if input_is_tensor:
            return torch.from_numpy(decoded).to(device=device, dtype=dtype)
        return decoded.astype(np.float32)

    __call__ = decode
