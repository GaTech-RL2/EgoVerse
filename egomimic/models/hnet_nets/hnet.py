"""
Vendored, simplified H-Net.

Recursive architecture mirroring upstream `hnet/models/hnet.py`:
    [encoder]  ->  routing -> chunk -> [main_network (HNet or Isotropic)] -> dechunk -> [decoder]

Differences from upstream:
    - No SSM ('m'/'M') support; only transformer ('t'/'T') blocks.
    - DeChunkLayer EMA is a pure-PyTorch recurrence (no mamba kernel).
    - Adds AdaLN conditioning *inside the main_network* (chunked / boundary tokens),
      so an external conditioning vector influences every boundary token.

Variable naming follows upstream as closely as possible:
    inference_params, routing_module_state, main_network_state, dechunk_state,
    encoder_state, decoder_state, bpred_output, prev_boundary_predictions.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn

from egomimic.models.hnet_nets.blocks import (
    Isotropic,
    IsotropicInferenceParams,
    KVCache,
    MambaCache,
)
from egomimic.models.hnet_nets.config import HNetConfig
from egomimic.models.hnet_nets.routing import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleState,
)


@dataclass
class HNetState:
    """Inference state for one stage of HNet (matches upstream field names)."""
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


def _chunk_cond(
    cond: Optional[torch.Tensor],
    boundary_mask: torch.Tensor,
    packed: bool,
    next_max_seqlen: Optional[int],
) -> Optional[torch.Tensor]:
    """Project per-token cond into the chunked-token space so deeper stages
    stay aligned with `next_hidden_states`.

    Padded mode:
        cond: (B, T, d_cond)  or  None
        boundary_mask: (B, T)
        returns: (B, M, d_cond) with M = next_max_seqlen, gathered by the
            same sort that ChunkLayer uses on hidden_states.
    Packed mode:
        cond: (T_total, d_cond)
        boundary_mask: (T_total,)
        returns: (M_total, d_cond) = cond[boundary_mask]
    Per-trajectory cond (cond.dim() == 2 in padded mode) is left untouched --
    AdaLN downstream will broadcast it across the chunked seq dim.
    """
    if cond is None:
        return None
    if packed:
        return cond[boundary_mask]
    # padded
    if cond.dim() == 2:
        return cond  # per-trajectory cond -- broadcasts inside AdaLN
    L = boundary_mask.shape[1]
    d = cond.shape[-1]
    device = boundary_mask.device
    M = int(next_max_seqlen) if next_max_seqlen is not None else int(boundary_mask.sum(dim=-1).max())
    token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
    sorted_idx = torch.argsort(token_idx, dim=1)[:, :M]
    return torch.gather(cond, dim=1, index=sorted_idx.unsqueeze(-1).expand(-1, -1, d))


class HNet(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int = 0,
        d_cond: int = 0,
        causal: bool = False,
        cond_encoder_stages: Optional[List[int]] = None,
        cond_decoder_stages: Optional[List[int]] = None,
    ):
        super().__init__()
        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        self.d_cond = d_cond
        self.causal = causal

        # Resolve cond-position lists. Take from config on first call; pass the
        # already-resolved lists down recursively so all stages agree.
        if cond_encoder_stages is None:
            cond_encoder_stages = list(getattr(config, "cond_encoder_stages", [0]))
        if cond_decoder_stages is None:
            cond_decoder_stages = list(getattr(config, "cond_decoder_stages", []))
        self.cond_encoder_stages = cond_encoder_stages
        self.cond_decoder_stages = cond_decoder_stages

        # AdaLN flags at this stage.
        self._enc_cond = (stage_idx in cond_encoder_stages) and (d_cond > 0)
        self._dec_cond = (stage_idx in cond_decoder_stages) and (d_cond > 0)

        # Whether any deeper stage will need cond -- if so, propagate d_cond
        # down so deeper HNets/Isotropics can build AdaLN modules.
        deeper_needs_cond = any(
            s > stage_idx for s in (cond_encoder_stages + cond_decoder_stages)
        )
        inner_d_cond = d_cond if (deeper_needs_cond and d_cond > 0) else 0

        layout = config.arch_layout
        for _ in range(stage_idx):
            layout = layout[1]
        assert isinstance(layout, list), f"Bad arch_layout at stage {stage_idx}: {layout}"

        if len(layout) == 3:
            self.is_innermost = False
            self.encoder = Isotropic(
                config, pos_idx=0, stage_idx=stage_idx,
                d_cond=d_cond, cond_here=self._enc_cond, causal=causal,
            )
            self.main_network = HNet(
                config, stage_idx=stage_idx + 1,
                d_cond=inner_d_cond, causal=causal,
                cond_encoder_stages=cond_encoder_stages,
                cond_decoder_stages=cond_decoder_stages,
            )
            self.decoder = Isotropic(
                config, pos_idx=2, stage_idx=stage_idx,
                d_cond=d_cond, cond_here=self._dec_cond, causal=causal,
            )
            self.routing_module = RoutingModule(self.d_model)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)
            self.residual_proj = nn.Linear(self.d_model, self.d_model)
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
            self.residual_proj.weight._no_reinit = True
        elif len(layout) == 1:
            # Innermost main_network is an Isotropic. It has no encoder/decoder
            # of its own and currently receives no AdaLN.
            self.is_innermost = True
            self.main_network = Isotropic(
                config, pos_idx=0, stage_idx=stage_idx,
                d_cond=0, cond_here=False, causal=causal,
            )
        else:
            raise NotImplementedError(f"arch_layout level must have 1 or 3 entries, got {len(layout)}")

        # Pad dimension to bridge stages with different d_model (matches upstream)
        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(self.d_model - config.d_model[stage_idx - 1])
            )
        else:
            self.pad_dimension = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        inference_params: Optional["HNetState"] = None,
    ):
        """
        Padded mode: hidden_states (B, L, D), mask (B, L).
        Packed mode: hidden_states (T_total, D), cu_seqlens (B+1,), max_seqlen int.

        AdaLN conditioning is applied **only at the outer encoder** (option B).
        main_network and decoder receive cond=None and have no AdaLN modules.

        In packed mode, ``cond`` may be either:
            - (B, d_cond):  one cond row per subseq → auto-expanded to (T_total, d_cond)
              via seq_idx so AdaLN modulates every position with its subseq's cond.
            - (T_total, d_cond): already per-position; passed through.

        If `inference_params` is provided (padded only), the routing module
        updates `last_hidden_state` / `has_seen_tokens` so a subsequent `step()`
        continues from the end of this prefill.

        Returns:
            output_states, list of RoutingModuleOutput per non-innermost stage,
            list of structure tokens per stage:
                padded: stage masks (B, L_stage)
                packed: stage cu_seqlens (B+1,)
        """
        assert (mask is not None) or (cu_seqlens is not None), \
            "Either mask (padded) or cu_seqlens (packed) must be provided"
        packed = cu_seqlens is not None

        # In packed mode, expand cond to per-token form so the outer encoder's
        # AdaLN modulates every position with its subseq's cond row.
        if packed and cond is not None and cond.dim() == 2 and cond.shape[0] != hidden_states.shape[0]:
            T_total = hidden_states.shape[0]
            pos = torch.arange(T_total, device=cu_seqlens.device)
            seq_idx = (pos[:, None] >= cu_seqlens[None, 1:]).sum(dim=-1)
            cond = cond[seq_idx]  # (T_total, d_cond)

        D_in = hidden_states.shape[-1]
        if self.pad_dimension is not None:
            pad = self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))
            hidden_states = torch.cat([hidden_states, pad], dim=-1)

        if self.is_innermost:
            # Innermost Isotropic -- never receives cond under the current policy.
            hidden_states = self.main_network(
                hidden_states, mask=mask,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            )
            hidden_states = hidden_states[..., :D_in]
            return hidden_states, [], []

        # 1) Encoder -- AdaLN fires here iff this stage is in cond_encoder_stages.
        hidden_states = self.encoder(
            hidden_states, mask=mask,
            cond=(cond if self._enc_cond else None),
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )

        # 2) Residual (fp32 like upstream)
        residual = self.residual_proj(hidden_states.float())

        # 3) Routing -> chunk
        bpred_output = self.routing_module(
            hidden_states, cu_seqlens=cu_seqlens, mask=mask,
            inference_params=(
                inference_params.routing_module_state if inference_params is not None else None
            ),
        )
        next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens=cu_seqlens, mask=mask,
        )

        # 4) Slice cond to the chunked token space and pass to main_network so
        # deeper-stage AdaLN (if any) stays per-token aligned.
        inner_cond = _chunk_cond(cond, bpred_output.boundary_mask, packed, next_max_seqlen)
        next_hidden_states, prev_boundary_predictions, prev_struct = self.main_network(
            next_hidden_states,
            mask=next_mask,
            cond=inner_cond,
            cu_seqlens=next_cu_seqlens,
            max_seqlen=next_max_seqlen,
        )

        # 5) DeChunk back to full length. Upstream API: pass the *chunked-space*
        # cu_seqlens (which the chunk_layer already returned as next_cu_seqlens).
        hidden_states = self.dechunk_layer(
            next_hidden_states, bpred_output.boundary_mask, bpred_output.boundary_prob,
            cu_seqlens=next_cu_seqlens,
            inference_params=(
                inference_params.dechunk_state if inference_params is not None else None
            ),
            mask=mask,
        )

        # 6) STE residual gating. selected_probs shape:
        #   padded (B, L, 1) — broadcasts over hidden_states (B, L, D)
        #   packed (T_total, 1) — broadcasts over hidden_states (T_total, D)
        hidden_states = (
            hidden_states.float() * ste_func(bpred_output.selected_probs) + residual
        ).to(hidden_states.dtype)

        # 7) Decoder -- AdaLN fires here iff this stage is in cond_decoder_stages.
        hidden_states = self.decoder(
            hidden_states, mask=mask,
            cond=(cond if self._dec_cond else None),
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )

        hidden_states = hidden_states[..., :D_in]
        stage_struct = cu_seqlens if packed else mask
        return (
            hidden_states,
            [bpred_output, *prev_boundary_predictions],
            [stage_struct, *prev_struct],
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=torch.float32):
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype
                ),
            )
        return HNetState(
            encoder_state=self.encoder.allocate_inference_cache(batch_size, max_seqlen, device, dtype),
            routing_module_state=self.routing_module.allocate_inference_cache(batch_size, max_seqlen, device, dtype),
            main_network_state=self.main_network.allocate_inference_cache(batch_size, max_seqlen, device, dtype),
            dechunk_state=self.dechunk_layer.allocate_inference_cache(batch_size, max_seqlen, device, dtype),
            decoder_state=self.decoder.allocate_inference_cache(batch_size, max_seqlen, device, dtype),
        )

    def step(
        self,
        hidden_states: torch.Tensor,
        inference_params: HNetState,
        cond: Optional[torch.Tensor] = None,
    ):
        """
        hidden_states: (B, 1, D_prev).
        Returns: (B, 1, D_prev), list of per-stage RoutingModuleOutput.
        """
        D_in = hidden_states.shape[-1]
        if self.pad_dimension is not None:
            pad = self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))
            hidden_states = torch.cat([hidden_states, pad], dim=-1)

        if self.is_innermost:
            # No AdaLN in main_network -- cond dropped.
            hidden_states = self.main_network.step(
                hidden_states, inference_params.main_network_state
            )
            return hidden_states[..., :D_in], []

        # Encoder -- AdaLN fires iff this stage is in cond_encoder_stages.
        hidden_states = self.encoder.step(
            hidden_states, inference_params.encoder_state,
            cond=(cond if self._enc_cond else None),
        )

        # Residual (fp32)
        residual = self.residual_proj(hidden_states.float())

        # Routing
        bpred_output = self.routing_module.step(
            hidden_states, inference_params.routing_module_state
        )

        # Chunk: only step inner network on batch elements where boundary fired.
        # Slice cond by boundary_mask (batch dim) so deeper-stage AdaLN sees
        # the cond rows for the elements that are advancing this tick.
        hidden_states_inner = self.chunk_layer.step(hidden_states, bpred_output.boundary_mask)
        if hidden_states_inner.shape[0] > 0:
            sub_inference_params = _slice_inference_params(
                inference_params.main_network_state, bpred_output.boundary_mask
            )
            sub_cond = cond[bpred_output.boundary_mask] if cond is not None else None
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner, sub_inference_params, cond=sub_cond,
            )
            _scatter_inference_params(
                inference_params.main_network_state, sub_inference_params, bpred_output.boundary_mask
            )
        else:
            prev_boundary_predictions = []

        # DeChunk back to full batch
        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        # STE residual
        hidden_states = (
            hidden_states.float() * ste_func(bpred_output.selected_probs.unsqueeze(1)) + residual
        ).to(hidden_states.dtype)

        # Decoder -- AdaLN fires iff this stage is in cond_decoder_stages.
        hidden_states = self.decoder.step(
            hidden_states, inference_params.decoder_state,
            cond=(cond if self._dec_cond else None),
        )

        hidden_states = hidden_states[..., :D_in]
        return hidden_states, [bpred_output, *prev_boundary_predictions]


def _slice_inference_params(inference_params, mask):
    """Recursively view-slice batch dim of an inference_params object by `mask`.
    Returns a new object whose tensors are cloned masked rows."""
    if inference_params is None:
        return None
    if isinstance(inference_params, IsotropicInferenceParams):
        new = IsotropicInferenceParams(
            layer_caches={},
            max_seqlen=inference_params.max_seqlen,
            batch_size=int(mask.sum().item()),
        )
        for k, c in inference_params.layer_caches.items():
            if isinstance(c, KVCache):
                new.layer_caches[k] = KVCache(
                    k=c.k[mask].clone(),
                    v=c.v[mask].clone(),
                    offsets=c.offsets[mask].clone(),
                )
            elif isinstance(c, MambaCache):
                new.layer_caches[k] = MambaCache(
                    conv_state=c.conv_state[mask].clone(),
                    ssm_state=c.ssm_state[mask].clone(),
                )
            else:
                raise TypeError(f"Unknown layer cache type: {type(c)}")
        return new
    if isinstance(inference_params, HNetState):
        return HNetState(
            encoder_state=_slice_inference_params(inference_params.encoder_state, mask),
            routing_module_state=(
                RoutingModuleState(
                    has_seen_tokens=inference_params.routing_module_state.has_seen_tokens[mask].clone(),
                    last_hidden_state=inference_params.routing_module_state.last_hidden_state[mask].clone(),
                ) if inference_params.routing_module_state is not None else None
            ),
            main_network_state=_slice_inference_params(inference_params.main_network_state, mask),
            dechunk_state=(
                DeChunkState(last_value=inference_params.dechunk_state.last_value[mask].clone())
                if inference_params.dechunk_state is not None else None
            ),
            decoder_state=_slice_inference_params(inference_params.decoder_state, mask),
        )
    raise TypeError(f"Unknown inference_params type: {type(inference_params)}")


def _scatter_inference_params(inference_params, sub_inference_params, mask):
    """Copy sub_inference_params' rows back into inference_params at `mask` positions."""
    if inference_params is None or sub_inference_params is None:
        return
    if isinstance(inference_params, IsotropicInferenceParams):
        for k, c in inference_params.layer_caches.items():
            sub = sub_inference_params.layer_caches[k]
            if isinstance(c, KVCache):
                c.k[mask] = sub.k.to(c.k.dtype)
                c.v[mask] = sub.v.to(c.v.dtype)
                c.offsets[mask] = sub.offsets
            elif isinstance(c, MambaCache):
                c.conv_state[mask] = sub.conv_state.to(c.conv_state.dtype)
                c.ssm_state[mask] = sub.ssm_state.to(c.ssm_state.dtype)
            else:
                raise TypeError(f"Unknown layer cache type: {type(c)}")
        return
    if isinstance(inference_params, HNetState):
        _scatter_inference_params(
            inference_params.encoder_state, sub_inference_params.encoder_state, mask
        )
        if (
            inference_params.routing_module_state is not None
            and sub_inference_params.routing_module_state is not None
        ):
            inference_params.routing_module_state.has_seen_tokens[mask] = (
                sub_inference_params.routing_module_state.has_seen_tokens
            )
            inference_params.routing_module_state.last_hidden_state[mask] = (
                sub_inference_params.routing_module_state.last_hidden_state.to(
                    inference_params.routing_module_state.last_hidden_state.dtype
                )
            )
        _scatter_inference_params(
            inference_params.main_network_state, sub_inference_params.main_network_state, mask
        )
        if (
            inference_params.dechunk_state is not None
            and sub_inference_params.dechunk_state is not None
        ):
            inference_params.dechunk_state.last_value[mask] = (
                sub_inference_params.dechunk_state.last_value.to(
                    inference_params.dechunk_state.last_value.dtype
                )
            )
        _scatter_inference_params(
            inference_params.decoder_state, sub_inference_params.decoder_state, mask
        )
        return
    raise TypeError(f"Unknown inference_params type: {type(inference_params)}")


def ratio_loss(
    bpred_outputs: List, structs: List, target_ratios: List[float]
) -> torch.Tensor:
    """
    H-Net auxiliary ratio loss (paper eq.):
        L = N/(N-1) * ((N-1) * F * G + (1 - F) * (1 - G))
      where N = target compression ratio,
            F = fraction of valid tokens selected as boundaries,
            G = mean boundary probability over valid tokens.
    `structs[i]` is the stage mask (padded) or stage cu_seqlens (packed).
    """
    if not bpred_outputs:
        return torch.zeros((), device="cpu")
    losses = []
    for i, bpred_output in enumerate(bpred_outputs):
        N = target_ratios[i] if i < len(target_ratios) else 2.0
        struct = structs[i]
        if struct is not None and struct.dtype == torch.bool:
            mask_f = struct.float()
            valid = mask_f.sum().clamp(min=1.0)
            F_ = (bpred_output.boundary_mask.float() * mask_f).sum() / valid
            G_ = (bpred_output.boundary_prob[..., -1] * mask_f).sum() / valid
        else:
            T = bpred_output.boundary_mask.shape[0]
            valid = torch.tensor(float(max(T, 1)), device=bpred_output.boundary_prob.device)
            F_ = bpred_output.boundary_mask.float().sum() / valid
            G_ = bpred_output.boundary_prob[..., -1].sum() / valid
        loss_i = (N / (N - 1)) * ((N - 1) * F_ * G_ + (1 - F_) * (1 - G_))
        losses.append(loss_i)
    return torch.stack(losses).sum()
