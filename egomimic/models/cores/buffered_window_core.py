"""Shared rolling-window lifecycle for replay-based causal cores.

Subclasses provide only ``_encode``.  This base keeps the training/evaluation
windowing and rollout-buffer contract identical across the Transformer and
H-Net BC cores without owning parameters or changing their ``state_dict``
layout.
"""

from typing import Optional

import torch
import torch.nn as nn


class BufferedWindowCore(nn.Module):
    """Base for causal cores whose rollout re-encodes a growing obs buffer."""

    max_window: int

    def _encode(self, x_in: torch.Tensor) -> torch.Tensor:
        """Encode one window.  Concrete cores must implement this hook."""
        raise NotImplementedError

    def forward(self, obs_emb: torch.Tensor, hidden: Optional[dict] = None):
        """Encode a train window or an eval episode split into fresh windows.

        ``hidden`` is accepted for interface parity with recurrent cores.  A
        call always starts a fresh window, matching the previous concrete-core
        implementations and rollout's explicit ``init_hidden`` boundaries.
        """
        del hidden
        T = obs_emb.shape[1]
        if T <= self.max_window:
            out = self._encode(obs_emb)
            return out, {"obs": obs_emb}

        W = self.max_window
        outs = []
        last = obs_emb[:, :W]
        for s in range(0, T, W):
            chunk = obs_emb[:, s : s + W]
            outs.append(self._encode(chunk))
            last = chunk
        out = torch.cat(outs, dim=1)
        return out, {"obs": last}

    def init_hidden(self, batch_size: int, device, dtype=None) -> dict:
        """Return the empty rolling buffer used at a rollout-window boundary."""
        del batch_size, device, dtype
        return {"obs": None}

    def step(self, obs_emb_t: torch.Tensor, hidden: dict):
        """Append one observation, replay the buffer, and return its last row."""
        x_t = obs_emb_t.unsqueeze(1)
        prev = hidden.get("obs", None) if hidden is not None else None
        if prev is None:
            buf = x_t
        else:
            buf = torch.cat([prev, x_t], dim=1)
        out = self._encode(buf)
        return out[:, -1], {"obs": buf}
