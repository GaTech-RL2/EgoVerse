"""Layer-wise cross-attention DiT flow-matching head.

Port of the GR00T N1.5 DiT as starVLA's QwenPI / QwenPI_v3 use it
(``starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py``
and ``LayerwiseFM_ActionHeader.py``, Apache-2.0), written with torch
primitives instead of ``diffusers``:

* DiT block ``i`` cross-attends to VLM hidden state ``i`` (one context per
  block); with ``interleave_self_attention`` the odd blocks self-attend over
  the action tokens instead and their context is unused.
* Time enters every block through AdaLN (scale / shift from a sinusoidal
  then MLP embedding of the CONTINUOUS flow time ``t`` in [0, 1]); a final
  time-modulated norm precedes the action decoder.
* ``adaln_zero`` adds zero-initialised gates so every block is the identity
  at init (user decision 3 for the deferred HPT DiT head); off for QwenVLA.

Flow-matching convention (``egomimic/models/fm_policy.py``):
``x_t = t * noise + (1 - t) * a``, target ``noise - a``, ``t`` from 1 to 0.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.denoising_nets import posemb_sincos

# openpi's posemb_sincos period range for a flow time in [0, 1].
TIME_MIN_PERIOD = 4e-3
TIME_MAX_PERIOD = 4.0


def _time_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) flow time in [0, 1] -> (B, dim) sinusoidal features."""
    return posemb_sincos(t.float(), dim, TIME_MIN_PERIOD, TIME_MAX_PERIOD)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        feats = _time_features(t, self.freq_dim).to(self.mlp[0].weight.dtype)
        return self.mlp(feats)


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale / shift come from the time embedding."""

    def __init__(self, dim: int, zero_init: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    """Multi-head attention; self-attention when ``context`` is None."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"hidden {dim} is not divisible by {num_heads} heads")
        kv_dim = dim if kv_dim is None else kv_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(kv_dim, dim)
        self.to_v = nn.Linear(kv_dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ctx = x if context is None else context
        B, T, _ = x.shape
        S = ctx.shape[1]
        h = self.num_heads
        q = self.to_q(x).view(B, T, h, -1).transpose(1, 2)
        k = self.to_k(ctx).view(B, S, h, -1).transpose(1, 2)
        v = self.to_v(ctx).view(B, S, h, -1).transpose(1, 2)
        attn_mask = None
        if context is not None and context_mask is not None:
            # (B, S) bool, True = attend -> (B, 1, 1, S), what SDPA expects
            attn_mask = context_mask.to(torch.bool)[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.to_out(out.transpose(1, 2).reshape(B, T, -1))


class DiTBlock(nn.Module):
    """AdaLN -> (cross | self) attention -> LN -> feed-forward, with residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_dim: Optional[int] = None,
        dropout: float = 0.0,
        final_dropout: bool = True,
        ff_mult: int = 4,
        adaln_zero: bool = False,
    ) -> None:
        super().__init__()
        self.is_cross = cross_dim is not None
        self.norm1 = AdaLayerNorm(dim, zero_init=adaln_zero)
        self.attn = Attention(dim, num_heads, kv_dim=cross_dim, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout) if final_dropout else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout) if final_dropout else nn.Identity(),
        )
        self.gate = None
        if adaln_zero:
            self.gate = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x, temb)
        if self.is_cross:
            attn = self.attn(h, context, context_mask)
        else:
            attn = self.attn(h)
        attn = self.attn_dropout(attn)
        if self.gate is None:
            x = x + attn
            return x + self.ff(self.norm2(x))
        gate_attn, gate_ff = self.gate(F.silu(temb)).chunk(2, dim=-1)
        x = x + gate_attn[:, None] * attn
        return x + gate_ff[:, None] * self.ff(self.norm2(x))


class LayerwiseDiT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden: int,
        num_heads: int,
        cross_dim: Optional[int] = None,
        dropout: float = 0.2,
        final_dropout: bool = True,
        interleave_self_attention: bool = True,
        adaln_zero: bool = False,
    ) -> None:
        super().__init__()
        cross_dim = hidden if cross_dim is None else cross_dim
        self.time_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden,
                    num_heads,
                    cross_dim=None
                    if (interleave_self_attention and i % 2 == 1)
                    else cross_dim,
                    dropout=dropout,
                    final_dropout=final_dropout,
                    adaln_zero=adaln_zero,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(hidden, zero_init=adaln_zero)

    def forward(
        self,
        tokens: torch.Tensor,
        contexts: list,
        context_mask: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        if len(contexts) != len(self.blocks):
            raise ValueError(
                f"LayerwiseDiT has {len(self.blocks)} blocks but got {len(contexts)} contexts"
            )
        temb = self.time_embed(t)
        x = tokens
        for i, block in enumerate(self.blocks):
            if block.is_cross:
                x = block(x, temb, contexts[i], context_mask)
            else:
                x = block(x, temb)
        return self.norm_out(x, temb)


class ActionEncoder(nn.Module):
    """starVLA's ActionEncoder: Linear(a) ++ sinusoid(t) -> swish(Linear) -> Linear."""

    def __init__(self, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.layer1 = nn.Linear(action_dim, hidden)
        self.layer2 = nn.Linear(2 * hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)

    def forward(self, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.layer1(actions)
        tau = (
            _time_features(t, self.hidden)
            .to(a.dtype)[:, None]
            .expand(-1, a.shape[1], -1)
        )
        x = F.silu(self.layer2(torch.cat([a, tau], dim=-1)))
        return self.layer3(x)


class StateEncoder(nn.Module):
    """One state MLP per embodiment (starVLA's ``MLP``), one token per proprio
    step, plus the HPT proprio-history extras: a learned zero-init per-step
    embedding and whole-context history dropout (train only)."""

    def __init__(
        self,
        state_dims: dict,
        hidden: int,
        history_len: int = 1,
        history_dropout: float = 0.0,
        mlp_hidden: int = 1024,
    ) -> None:
        super().__init__()
        self.state_dims = {str(k): int(v) for k, v in state_dims.items()}
        self.history_len = int(history_len)
        self.history_dropout = float(history_dropout)
        self.mlps = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, hidden)
                )
                for name, dim in self.state_dims.items()
            }
        )
        self.time_embed = (
            nn.Parameter(torch.zeros(self.history_len, hidden))
            if self.history_len > 1
            else None
        )

    def forward(self, state: torch.Tensor, embodiment: str) -> torch.Tensor:
        if embodiment not in self.mlps:
            raise KeyError(
                f"StateEncoder has no MLP for embodiment {embodiment!r}; "
                f"known: {sorted(self.mlps)}"
            )
        if state.ndim != 3:
            raise ValueError(f"state must be (B, K, D), got {tuple(state.shape)}")
        if state.shape[1] != self.history_len:
            raise ValueError(
                f"state carries {state.shape[1]} history steps but history_len="
                f"{self.history_len}; set the keymap's proprio_history and the "
                "model's history_len to the same K"
            )
        if state.shape[-1] != self.state_dims[embodiment]:
            raise ValueError(
                f"{embodiment} state is {state.shape[-1]}-wide, expected "
                f"{self.state_dims[embodiment]}"
            )
        if self.training and self.history_dropout > 0 and self.history_len > 1:
            drop = (
                torch.rand(state.shape[0], device=state.device) < self.history_dropout
            )
            current = state[:, -1:].expand_as(state)
            state = torch.where(drop[:, None, None], current, state)
        y = self.mlps[embodiment](state)
        if self.time_embed is not None:
            y = y + self.time_embed[None]
        return y


class LayerwiseFMHead(nn.Module):
    """Flow-matching action head over ``LayerwiseDiT``.

    DiT sequence: ``[state x K] + [registers] + [action_horizon action tokens]``;
    only the action positions are decoded. ``projectors`` map every VLM layer
    to ``dit_hidden`` (identity when the widths match: QwenPI; LayerNorm +
    Linear otherwise: QwenPI_v3's compressed head).
    """

    def __init__(
        self,
        action_width: int,
        action_horizon: int,
        vlm_hidden: int,
        num_layers: int,
        state_dims: Optional[dict] = None,
        dit_hidden: Optional[int] = None,
        head_dim: int = 64,
        num_register_tokens: int = 32,
        history_len: int = 1,
        history_dropout: float = 0.0,
        dropout: float = 0.2,
        final_dropout: bool = True,
        interleave_self_attention: bool = True,
        adaln_zero: bool = False,
        num_inference_steps: int = 10,
        time_dist: str = "beta",
        repeated_diffusion_steps: int = 1,
    ) -> None:
        super().__init__()
        self.action_width = int(action_width)
        self.action_horizon = int(action_horizon)
        self.vlm_hidden = int(vlm_hidden)
        self.dit_hidden = int(vlm_hidden if dit_hidden is None else dit_hidden)
        self.num_inference_steps = int(num_inference_steps)
        self.time_dist = time_dist
        self.repeated_diffusion_steps = int(repeated_diffusion_steps)
        if self.dit_hidden % head_dim:
            raise ValueError(
                f"dit_hidden {self.dit_hidden} not divisible by head_dim {head_dim}"
            )
        if time_dist not in ("beta", "uniform"):
            raise ValueError(f"time_dist must be beta or uniform, got {time_dist!r}")

        self.projectors = nn.ModuleList(
            [
                nn.Identity()
                if self.dit_hidden == self.vlm_hidden
                else nn.Sequential(
                    nn.LayerNorm(self.vlm_hidden),
                    nn.Linear(self.vlm_hidden, self.dit_hidden),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.action_encoder = ActionEncoder(self.action_width, self.dit_hidden)
        self.pos_embed = nn.Parameter(torch.zeros(self.action_horizon, self.dit_hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.registers = nn.Parameter(
            torch.zeros(int(num_register_tokens), self.dit_hidden)
        )
        nn.init.normal_(self.registers, std=0.02)
        self.state_encoder = (
            StateEncoder(state_dims, self.dit_hidden, history_len, history_dropout)
            if state_dims
            else None
        )
        self.dit = LayerwiseDiT(
            int(num_layers),
            self.dit_hidden,
            self.dit_hidden // head_dim,
            cross_dim=self.dit_hidden,
            dropout=dropout,
            final_dropout=final_dropout,
            interleave_self_attention=interleave_self_attention,
            adaln_zero=adaln_zero,
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(self.dit_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_width),
        )

    # -- pieces ------------------------------------------------------------

    def project(self, contexts: list) -> list:
        if len(contexts) != len(self.projectors):
            raise ValueError(
                f"got {len(contexts)} VLM layers but the head has {len(self.projectors)} projectors"
            )
        return [proj(ctx) for proj, ctx in zip(self.projectors, contexts)]

    def velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        contexts: list,
        context_mask: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        embodiment: Optional[str] = None,
    ) -> torch.Tensor:
        B, T, _ = x_t.shape
        if T != self.action_horizon:
            raise ValueError(
                f"action chunk has {T} steps, head expects {self.action_horizon}"
            )
        parts = []
        if state is not None:
            if self.state_encoder is None:
                raise ValueError(
                    "state given but the head was built without state_dims"
                )
            parts.append(self.state_encoder(state, embodiment))
        parts.append(self.registers[None].expand(B, -1, -1))
        parts.append(self.action_encoder(x_t, t) + self.pos_embed[None])
        tokens = torch.cat(parts, dim=1)
        out = self.dit(tokens, contexts, context_mask, t)
        return self.action_decoder(out[:, -T:])

    def sample_time(self, n: int, device) -> torch.Tensor:
        if self.time_dist == "beta":
            t = torch.distributions.Beta(1.5, 1.0).sample((n,)).to(device)
        else:
            t = torch.rand(n, device=device)
        return t * 0.999 + 0.001

    # -- training / inference ------------------------------------------------

    def compute_loss(
        self,
        contexts: list,
        context_mask: Optional[torch.Tensor],
        actions: torch.Tensor,
        loss_mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        embodiment: Optional[str] = None,
    ) -> torch.Tensor:
        contexts = self.project(contexts)
        R = self.repeated_diffusion_steps
        if R > 1:
            contexts = [c.repeat(R, 1, 1) for c in contexts]
            context_mask = None if context_mask is None else context_mask.repeat(R, 1)
            actions = actions.repeat(R, 1, 1)
            loss_mask = loss_mask.repeat(R, 1, 1)
            state = None if state is None else state.repeat(R, 1, 1)
        noise = torch.randn_like(actions)
        t = self.sample_time(actions.shape[0], actions.device).to(actions.dtype)
        te = t[:, None, None]
        x_t = te * noise + (1 - te) * actions
        u_t = noise - actions
        v_t = self.velocity(x_t, t, contexts, context_mask, state, embodiment)
        mask = loss_mask.to(v_t.dtype).expand_as(v_t)
        return ((v_t - u_t) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def sample(
        self,
        contexts: list,
        context_mask: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        embodiment: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        contexts = self.project(contexts)
        B = contexts[0].shape[0]
        device, dtype = contexts[0].device, self.pos_embed.dtype
        n = self.num_inference_steps if num_steps is None else int(num_steps)
        x = torch.randn(
            (B, self.action_horizon, self.action_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        dt = -1.0 / n
        t = torch.ones(B, device=device, dtype=dtype)
        for _ in range(n):
            x = x + dt * self.velocity(x, t, contexts, context_mask, state, embodiment)
            t = t + dt
        return x
