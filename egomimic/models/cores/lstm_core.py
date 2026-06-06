"""LSTM core for BC-RNN.

This IS the history mechanism of the robomimic BC-RNN recipe: an LSTM unrolled
over the sequence of per-frame OBSERVATION embeddings. The recurrence carries
observation history forward; there is NO past-action input (the action is the
OUTPUT decoded from the hidden state by the GMM head, not fed back in).

Two entry points:
  * ``forward(obs_emb)`` -- teacher-forced training: unroll over a full length-T
    embedding sequence in one shot, return the per-step hidden states.
  * ``init_hidden`` / ``step`` -- closed-loop rollout: carry ``(h, c)`` across
    env steps, feeding one freshly-observed frame embedding at a time.

The two paths share the same ``nn.LSTM`` weights, so train == eval.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class LSTMCore(nn.Module):
    """Multi-layer LSTM over per-frame obs embeddings.

    Args:
        input_dim:  per-frame obs embedding width (ObsEncoder.embed_dim).
        hidden_dim: LSTM hidden size (robomimic BC-RNN default 1000; we expose
                    it as a config knob).
        num_layers: stacked LSTM layers (robomimic default 2).
        dropout:    dropout between stacked LSTM layers (0 if num_layers==1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1000,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self,
        obs_emb: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Unroll over a ``(B, T, input_dim)`` embedding sequence.

        Returns:
            out:    ``(B, T, hidden_dim)`` per-step hidden states.
            hidden: final ``(h, c)`` (each ``(num_layers, B, hidden_dim)``),
                    so callers can continue the rollout if they wish.
        """
        out, hidden = self.lstm(obs_emb, hidden)
        return out, hidden

    def init_hidden(
        self, batch_size: int, device, dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero ``(h, c)`` for a fresh rollout. Each ``(num_layers, B, hidden)``."""
        dtype = dtype or next(self.parameters()).dtype
        shape = (self.num_layers, batch_size, self.hidden_dim)
        h = torch.zeros(*shape, device=device, dtype=dtype)
        c = torch.zeros(*shape, device=device, dtype=dtype)
        return h, c

    def step(
        self,
        obs_emb_t: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One rollout step. ``obs_emb_t`` ``(B, input_dim)`` (a single frame).

        Returns ``(h_t (B, hidden_dim), new_hidden)``. The caller carries
        ``new_hidden`` to the next step so the history accumulates exactly as in
        the teacher-forced unroll.
        """
        out, hidden = self.lstm(obs_emb_t.unsqueeze(1), hidden)  # (B,1,hidden)
        return out[:, 0], hidden
