"""``FlatFusedOuterStage`` — OuterStage subclass for the flat-transformer
(no chunker) H-Net variant. Interleaved [c_t, a_t] tokens in a 2T-length
sequence, single Isotropic stack, causal attention.

Structurally a rename of the prior ``FlatFusedPolicy`` with the addition
of the ``(batch, ctx)`` dispatcher so it plugs into the HNet algo class
via the same outer_stage field as HNetOuterStage. Legacy ``forward``,
``forward_packed``, ``generate``, ``step``, ``init_step_state`` methods
are preserved verbatim for the algo's inference paths.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from egomimic.algo.outer_stage import OuterStage
from egomimic.models.hnet_nets.cond_encoders import CondEncoderModule
from egomimic.models.hnet_nets.context import HNetContext


class FlatFusedOuterStage(OuterStage):
    """Flat transformer with interleaved [c_t, a_t] tokens.

    Input layout (length 2T):
      x[:, 0]  = cond_in(c_0)         x[:, 1]  = BOS
      x[:, 2]  = cond_in(c_1)         x[:, 3]  = action_in(a_0)
      ...

    Output extraction:
      pred[:, t] = action_out(out[:, 2t+1])
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        d_model: int,
        d_cond: int,
        cond_encoder: CondEncoderModule,
        arch_layout: str = "T8",
        num_heads: int = 4,
        d_intermediate: int = 512,
        dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__(inner_stage=None)
        from egomimic.models.hnet_nets.isotropic_builder import build_isotropic

        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.d_cond = d_cond
        self.cond_encoder = cond_encoder

        self.action_in = nn.Linear(action_dim, d_model)
        self.action_out = nn.Linear(d_model, action_dim)
        self.cond_in = nn.Linear(d_cond, d_model)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.bos, std=0.02)
        # pos_emb covers the 2T-token interleaved sequence.
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * action_horizon, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Single Isotropic stack, causal, no per-block cond.
        self.backbone = build_isotropic(
            {
                "arch_layout": arch_layout,
                "d_model": d_model,
                "d_intermediate": d_intermediate,
                "num_heads": num_heads,
                "cond": False,
                "dropout": dropout,
                "resid_dropout": resid_dropout,
            },
            d_cond=0,
            causal=True,
        )

    # ------------------------------------------------------------------
    # OuterStage API — (batch, ctx) dispatcher over the legacy forwards.
    # ------------------------------------------------------------------

    def encode(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        raise NotImplementedError(
            "FlatFusedOuterStage does not split encode/decode — the trunk runs "
            "inside the unified forward path. Use forward(batch, ctx) instead."
        )

    def decode(self, x, batch: dict, ctx: HNetContext) -> None:
        raise NotImplementedError(
            "FlatFusedOuterStage does not split encode/decode. See forward."
        )

    def forward(self, batch: dict, ctx: HNetContext) -> torch.Tensor:
        """Dispatch padded vs packed based on ctx.cu_seqlens. Writes
        ``batch['pred_action']``; ctx.aux stays empty (no chunker)."""
        actions = batch["actions"]
        obs = batch["__obs"]
        if ctx.cu_seqlens is not None:
            pred, _ = self.forward_packed(
                actions, obs, ctx.cu_seqlens, ctx.max_seqlen
            )
        else:
            pred, _ = self.forward_padded(actions, obs)
        batch["pred_action"] = pred
        # ctx.aux stays [] — no chunker contributions to the loss.
        return pred

    # ------------------------------------------------------------------
    # Legacy bridges + the actual implementation (verbatim from the
    # prior FlatFusedPolicy class — only the class name and base changed).
    # ------------------------------------------------------------------

    def _encode_cond(self, obs: dict, T: int) -> torch.Tensor:
        cond_dict = self.cond_encoder.encode(obs, T)
        c = cond_dict.get("fused_cond")
        if c is None:
            raise KeyError(
                "FlatFusedOuterStage requires 'fused_cond' in cond_encoder output."
            )
        return c

    def forward_padded(self, actions: torch.Tensor, obs: dict):
        """Padded teacher-forced forward.

        actions: (B, T, action_dim)
        obs:     dict of per-frame (B, T, ...) obs tensors.
        Returns: (pred (B, T, action_dim), aux=[]).
        """
        B, T, _ = actions.shape
        c = self._encode_cond(obs, T)
        c_tok = self.cond_in(c)
        a_tok = self.action_in(actions)
        a_shifted = torch.cat([self.bos.expand(B, -1, -1), a_tok[:, :-1]], dim=1)

        x = torch.empty(B, 2 * T, self.d_model, device=actions.device, dtype=a_tok.dtype)
        x[:, 0::2] = c_tok
        x[:, 1::2] = a_shifted
        x = x + self.pos_emb[:, : 2 * T].to(x.dtype)

        x = self.backbone(x)
        pred = self.action_out(x[:, 1::2])
        return pred, []

    def forward_packed(
        self,
        actions_packed: torch.Tensor,
        obs_packed: dict,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ):
        """Packed-mode teacher-forced forward."""
        device = actions_packed.device
        T_total = actions_packed.shape[0]
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        if max_seqlen > self.action_horizon:
            raise ValueError(
                f"max_seqlen={max_seqlen} exceeds action_horizon={self.action_horizon}"
            )

        obs_for_encode = {k: v.unsqueeze(0) for k, v in obs_packed.items()}
        cond_seq = self._encode_cond(obs_for_encode, T_total).squeeze(0)
        c_tok = self.cond_in(cond_seq)
        a_tok = self.action_in(actions_packed)

        a_shifted = torch.empty_like(a_tok)
        bos = self.bos.squeeze(0).squeeze(0).to(a_tok.dtype)
        a_shifted[cu_seqlens[:-1]] = bos
        non_start = torch.ones(T_total, dtype=torch.bool, device=device)
        non_start[cu_seqlens[:-1]] = False
        idx_non_start = torch.nonzero(non_start, as_tuple=False).squeeze(-1)
        a_shifted[idx_non_start] = a_tok[idx_non_start - 1]

        pos = torch.arange(T_total, device=device)
        seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
        local_pos = pos - cu_seqlens[seq_idx]

        sub_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        new_lens = 2 * sub_lens
        new_cu = torch.zeros(len(cu_seqlens), dtype=torch.long, device=device)
        new_cu[1:] = torch.cumsum(new_lens, dim=0)
        new_T_total = int(new_cu[-1].item())

        x = torch.empty(new_T_total, self.d_model, device=device, dtype=a_tok.dtype)
        target_c = new_cu[seq_idx] + 2 * local_pos
        target_a = target_c + 1
        x[target_c] = c_tok
        x[target_a] = a_shifted
        pos_c = (2 * local_pos).clamp(max=2 * self.action_horizon - 1)
        pos_a = (2 * local_pos + 1).clamp(max=2 * self.action_horizon - 1)
        x[target_c] = x[target_c] + self.pos_emb[0, pos_c].to(x.dtype)
        x[target_a] = x[target_a] + self.pos_emb[0, pos_a].to(x.dtype)

        out = self.backbone(
            x,
            cu_seqlens=new_cu,
            max_seqlen=2 * int(max_seqlen),
        )

        pred = self.action_out(out[target_a])
        return pred, []

    @torch.no_grad()
    def generate(
        self,
        obs: dict,
        batch_size: int,
        device,
        T: Optional[int] = None,
    ) -> torch.Tensor:
        if T is None:
            T = self.action_horizon
        if T > self.action_horizon:
            raise ValueError(f"generate T={T} exceeds action_horizon")

        cond_seq = self._encode_cond(obs, T)
        c_tok = self.cond_in(cond_seq)
        dtype = c_tok.dtype

        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T,
            device=device,
            dtype=dtype,
        )

        actions = torch.zeros(batch_size, T, self.action_dim, device=device)
        a_prev = self.bos.expand(batch_size, -1, -1).to(dtype)
        for t in range(T):
            x_c = c_tok[:, t : t + 1] + self.pos_emb[:, 2 * t : 2 * t + 1].to(dtype)
            _ = self.backbone.step(x_c, params)
            x_a = a_prev + self.pos_emb[:, 2 * t + 1 : 2 * t + 2].to(dtype)
            h = self.backbone.step(x_a, params)
            a_t = self.action_out(h)
            actions[:, t : t + 1] = a_t
            a_prev = self.action_in(a_t)

        return actions

    @torch.no_grad()
    def init_step_state(self, batch_size: int, T_max: int, device, dtype=None) -> dict:
        T_max = int(min(T_max, self.action_horizon))
        dtype = dtype or next(self.parameters()).dtype
        params = self.backbone.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=2 * T_max,
            device=device,
            dtype=dtype,
        )
        a_prev_emb = self.bos.expand(batch_size, 1, self.d_model).to(dtype)
        return {
            "params": params,
            "a_prev_emb": a_prev_emb,
            "dtype": dtype,
            "T_max": T_max,
        }

    @torch.no_grad()
    def step(self, state: dict, obs_norm: dict, t: int) -> torch.Tensor:
        dtype = state["dtype"]
        cond_seq = self._encode_cond(obs_norm, T=1)
        fused = cond_seq.squeeze(1)
        c_tok = (
            self.cond_in(fused).unsqueeze(1) + self.pos_emb[:, 2 * t : 2 * t + 1]
        ).to(dtype)
        _ = self.backbone.step(c_tok, state["params"])

        a_tok = (state["a_prev_emb"] + self.pos_emb[:, 2 * t + 1 : 2 * t + 2]).to(dtype)
        h = self.backbone.step(a_tok, state["params"])
        a_t_norm = self.action_out(h)
        state["a_prev_emb"] = self.action_in(a_t_norm).to(dtype)
        return a_t_norm
