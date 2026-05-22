"""BC-RNN OuterStage. Faithful to robomimic's ``RNNActorNetwork``.

Flow per training forward (padded mode, ``(B, T, *)``):

    obs dict ──[CondEncoderModule]──► rnn_inputs (B, T, d_input)
                                       │
                                  nn.LSTM / nn.GRU
                                       │
                                       ▼
                              hidden (B, T, H)
                                       │
                              per-step action head
                                       │
                                       ▼
                              pred_action (B, T, action_dim)

Packed mode (``(T_total, *)`` + ``cu_seqlens``) is supported by splitting
into per-episode tensors, padding to a single ``(B_eps, T_max, *)`` batch,
and running pack_padded_sequence through the RNN so the LSTM state is
correctly bounded per episode.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from egomimic.algo.outer_stage import OuterStage
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule


class _LSTMTrunk(nn.Module):
    """Wraps ``nn.LSTM`` / ``nn.GRU`` so it fits OuterStage's
    ``inner_stage(x, ctx) -> x`` contract. Reads / writes ``ctx.rnn_state``
    so callers can thread hidden state across calls (used in single-step
    rollout).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        rnn_type: str = "LSTM",
        bidirectional: bool = False,
    ):
        super().__init__()
        if rnn_type not in {"LSTM", "GRU"}:
            raise ValueError(f"rnn_type must be 'LSTM' or 'GRU', got {rnn_type!r}")
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * self.num_directions

    def init_state(self, batch_size: int, device, dtype=None):
        h = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype or torch.float32,
        )
        if self.rnn_type == "LSTM":
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def forward(self, x: torch.Tensor, ctx) -> torch.Tensor:
        state = getattr(ctx, "rnn_state", None)
        out, new_state = self.rnn(x, state)
        ctx.rnn_state = new_state
        return out


class BCRNNOuterStage(OuterStage):
    """Per-step recurrent policy.

    Args:
        action_dim: action feature width.
        cond_encoder: ``CondEncoderModule`` for obs -> (B, T, d_cond) per-step
            features. ``d_cond`` here is the LSTM input width.
        rnn_hidden_dim: LSTM/GRU hidden width (robomimic default 400).
        rnn_num_layers: number of stacked layers (robomimic default 2).
        rnn_type: ``"LSTM"`` (default) or ``"GRU"``.
        rnn_bidirectional: if True, doubles the effective output width.
        action_head_type: ``"linear"`` (default) or ``"mlp"``. Robomimic's
            BC_RNN uses a single Linear (the per_step_net is just
            ``ObservationDecoder`` = a Linear).
    """

    def __init__(
        self,
        action_dim: int,
        cond_encoder: CondEncoderModule,
        rnn_hidden_dim: int = 400,
        rnn_num_layers: int = 2,
        rnn_type: str = "LSTM",
        rnn_bidirectional: bool = False,
        action_head_type: str = "linear",
    ):
        trunk = _LSTMTrunk(
            input_dim=cond_encoder.d_cond,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            bidirectional=rnn_bidirectional,
        )
        super().__init__(inner_stage=trunk)
        self.cond_encoder = cond_encoder
        self.cond_output_key = cond_encoder.output_key
        self.action_dim = int(action_dim)

        head_in = trunk.output_dim
        if action_head_type == "linear":
            self.action_head = nn.Linear(head_in, action_dim)
        elif action_head_type == "mlp":
            self.action_head = nn.Sequential(
                nn.Linear(head_in, head_in),
                nn.ReLU(),
                nn.Linear(head_in, action_dim),
            )
        else:
            raise ValueError(
                f"action_head_type must be 'linear' or 'mlp', got {action_head_type!r}"
            )
        self.action_head_type = action_head_type

    # ------------------------------------------------------------------
    # OuterStage API. ``encode`` returns ``(B, T, d_cond)`` ready for the
    # LSTM; ``decode`` projects per-step to action_dim.
    # ------------------------------------------------------------------

    def encode(self, batch: dict, ctx) -> torch.Tensor:
        is_packed = getattr(ctx, "cu_seqlens", None) is not None
        if is_packed:
            return self._encode_packed(batch, ctx)
        return self._encode_padded(batch, ctx)

    def _encode_padded(self, batch: dict, ctx) -> torch.Tensor:
        obs = batch["__obs"]
        actions = batch["actions"]  # (B, T, A)
        _, T, _ = actions.shape
        cond = self.cond_encoder.encode(obs, T_action=T)
        x = cond[self.cond_output_key]  # (B, T, d_cond)
        return x

    def _encode_packed(self, batch: dict, ctx) -> torch.Tensor:
        """Pack -> split -> pad to a (B_eps, T_max, d_cond) tensor. Stores
        ``ctx._bcrnn_pack_meta`` (cu_seqlens, T_max, B_eps) so ``decode``
        can scatter predictions back to ``(T_total, action_dim)``.
        """
        obs_packed = batch["__obs"]  # each value: (T_total, ...)
        T_total = batch["actions"].shape[0]
        device = batch["actions"].device
        cu = ctx.cu_seqlens.to(device=device, dtype=torch.long)
        seq_lens = (cu[1:] - cu[:-1]).tolist()
        B_eps = len(seq_lens)
        T_max = int(max(seq_lens))

        # Encode per-step on the flat (T_total,...) stream by faking B=1.
        obs_3d = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond = self.cond_encoder.encode(obs_3d, T_action=T_total)
        feat = cond[self.cond_output_key].squeeze(0)  # (T_total, d_cond)
        d_cond = feat.shape[-1]

        # Scatter into a padded (B_eps, T_max, d_cond) buffer.
        padded = feat.new_zeros(B_eps, T_max, d_cond)
        for i, L in enumerate(seq_lens):
            start = int(cu[i].item())
            padded[i, :L] = feat[start : start + L]

        # Stash meta for decode + LSTM packing.
        ctx._bcrnn_pack_meta = {
            "cu": cu,
            "seq_lens": torch.tensor(seq_lens, device=device, dtype=torch.long),
            "T_total": T_total,
            "T_max": T_max,
            "B_eps": B_eps,
        }
        return padded

    def decode(self, h: torch.Tensor, batch: dict, ctx) -> None:
        meta = getattr(ctx, "_bcrnn_pack_meta", None)
        if meta is None:
            # Padded mode: h is (B, T, H). Project per-step.
            batch["pred_action"] = self.action_head(h)
            return
        # Packed mode: h is (B_eps, T_max, H). Project then scatter back to
        # (T_total, action_dim) using cu_seqlens.
        pred_padded = self.action_head(h)  # (B_eps, T_max, A)
        T_total = meta["T_total"]
        cu = meta["cu"]
        seq_lens = meta["seq_lens"].tolist()
        out = pred_padded.new_zeros(T_total, self.action_dim)
        for i, L in enumerate(seq_lens):
            start = int(cu[i].item())
            out[start : start + L] = pred_padded[i, :L]
        batch["pred_action"] = out

    # ------------------------------------------------------------------
    # Forward override: packed mode needs pack_padded_sequence around the
    # LSTM so the last hidden state isn't polluted by zero-padding. Padded
    # mode falls through to the standard ``encode -> inner_stage -> decode``.
    # ------------------------------------------------------------------

    def forward(self, batch: dict, ctx) -> torch.Tensor:
        x = self.encode(batch, ctx)
        meta = getattr(ctx, "_bcrnn_pack_meta", None)
        if meta is None:
            h = self.inner_stage(x, ctx)
        else:
            seq_lens = meta["seq_lens"].to(device="cpu", dtype=torch.long)
            packed = pack_padded_sequence(
                x, seq_lens, batch_first=True, enforce_sorted=False
            )
            packed_out, new_state = self.inner_stage.rnn(packed)
            ctx.rnn_state = new_state
            h, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=meta["T_max"]
            )
        self.decode(h, batch, ctx)
        return h

    # ------------------------------------------------------------------
    # Inference: single-step rollout with persisted hidden state.
    # ``init_step_state`` allocates; ``step`` consumes one obs and produces
    # one action. Hidden state lives in the returned dict.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_step_state(self, batch_size: int, device, dtype=None) -> dict:
        return {
            "rnn_state": self.inner_stage.init_state(batch_size, device, dtype),
            "batch_size": int(batch_size),
            "device": device,
            "dtype": dtype,
        }

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        """Returns (B, 1, action_dim). ``state['rnn_state']`` is updated
        in-place."""
        cond = self.cond_encoder.encode(obs_norm, T_action=1)
        x = cond[self.cond_output_key]  # (B, 1, d_cond)
        out, new_state = self.inner_stage.rnn(x, state["rnn_state"])
        state["rnn_state"] = new_state
        return self.action_head(out)
