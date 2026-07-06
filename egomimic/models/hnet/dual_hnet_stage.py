"""Dual-*H-Net* compute stage for the cross-embodiment GMM H-Net.

This is a drop-in replacement for ``DualStreamComputeStage`` that swaps the two
transformer trunks (``TwoStreamTrunk``) for two genuinely-separate **H-Nets**:

  * AGNOSTIC stream ``A`` -> ONE shared, dynamic-chunking ``HNet`` (weight-shared
    across all embodiments, so it is forced to learn embodiment-agnostic
    structure).
  * SPECIFIC stream ``S`` -> a PER-EMBODIMENT ``nn.ModuleDict`` of dynamic-
    chunking ``HNet``s, dispatched by ``ctx.embodiment_id`` (separate weights per
    embodiment -> embodiment-private structure).

The two H-Nets are FULLY INDEPENDENT: there is no cross-stream attention (unlike
``TwoStreamTrunk``). Each runs its own router -> chunk -> inner-trunk -> dechunk
on its own stream, so A literally never sees S (and S never sees A) -- a stronger
form of the asym isolation. Both top representations meet only at the shared
partitioned GMM head (``A_top`` -> agnostic modes, ``S_top`` -> specific modes,
joint gate), exactly as before.

Layout / protocol (UNCHANGED from ``DualStreamComputeStage`` -- this is why the
stage is a drop-in under ``DualStreamOuterStage``):
  x = cat([A, S], dim=0)   with A = x[:T_total], S = x[T_total:]   (L = 2*T_total)
  ctx.extras['dual'] = {"T_total": int, "cu_seqlens": (B+1,), "time_pos": (T_total,)}
  returns ``A_top`` (T_total, d_model); stashes ``S_top`` in
  ctx.extras['specific_tokens'] for the GMM head.

Scope:
  * PACKED path only (asserts ``ctx.packed``); terminal stage (no inner_stage).
  * Each inner H-Net handles its own causality + episode isolation via its
    packed attention (``cu_seqlens`` + ``causal``); RoPE is per-subseq inside
    each Isotropic block. We do NOT build the 2T allow-mask here.
  * Ratio-loss / chunk-stats aux from BOTH inner H-Nets is merged into the outer
    ``ctx.aux`` (agnostic chunkers first, then specific) so the algo's
    ``ratio_loss_from_aux`` SUMS them and ``chunk_stats_from_aux`` indexes them
    as ``boundary_rate_0`` / ``boundary_rate_1`` / ... with no collision.
  * NO incremental KV-cache: closed-loop AR is recompute-over-prefix, driven by
    ``DualStreamOuterStage.step`` calling ``forward`` over the growing prefix.
"""

from typing import Dict

import torch
import torch.nn as nn

from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet
from egomimic.models.hnet.stages import _BaseStage


class DualHNetComputeStage(_BaseStage):
    """Terminal dual-stream stage backed by two separate H-Nets.

    Args:
        input_hidden_dim / output_hidden_dim: must be equal (dim-preserving) and
            equal the obs-token width ``d_model`` fed by ``DualStreamOuterStage``
            -- so ``A_top`` / ``S_top`` come back at ``d_model`` for the head.
        hnet_agnostic: an instantiated ``HNet`` (the shared, embodiment-agnostic
            stream). Must be dim-preserving at ``input_hidden_dim``.
        hnet_specific: a ``dict[domain -> HNet]`` (one per embodiment). Each must
            be dim-preserving at ``input_hidden_dim``. Wrapped in an
            ``nn.ModuleDict`` and dispatched by ``ctx.embodiment_id``.
        cond_key: kept for config symmetry; cond is OFF inside the inner H-Nets
            (their stages use ``cond: false``). The shared ``ctx.cond_dict`` is
            still threaded into the sub-contexts so a config that DOES enable
            cond stays correct.
    """

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        hnet_agnostic: HNet,
        hnet_specific: Dict[str, HNet],
        cond_key=None,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if int(input_hidden_dim) != int(output_hidden_dim):
            raise ValueError(
                f"DualHNetComputeStage requires input_hidden_dim==output_hidden_dim "
                f"(got {input_hidden_dim} vs {output_hidden_dim}); the stage is "
                f"dim-preserving so A_top/S_top return at d_model for the head."
            )
        d_model = int(input_hidden_dim)

        # hnet_specific may arrive as a plain dict or an OmegaConf DictConfig
        # (Hydra keeps containers as DictConfig under the default _convert_; its
        # values are already-instantiated HNet objects). Coerce to a plain dict
        # robustly so nn.ModuleDict accepts it.
        try:
            spec_dict = dict(hnet_specific)
        except (TypeError, ValueError):
            spec_dict = None
        if not spec_dict:
            raise ValueError(
                "hnet_specific must be a non-empty mapping {embodiment: HNet}."
            )
        self.hnet_agnostic = hnet_agnostic
        self.hnet_specific = nn.ModuleDict(spec_dict)

        # Every inner H-Net must be dim-preserving at d_model so the value
        # returned for stream A (and the stashed S) feeds head(a_top, s) at the
        # head's d_model. Fail loud at construction rather than mid-forward.
        for name, h in [("agnostic", self.hnet_agnostic), *self.hnet_specific.items()]:
            hi, ho = int(h.input_hidden_dim), int(h.output_hidden_dim)
            if hi != d_model or ho != d_model:
                raise ValueError(
                    f"inner H-Net '{name}' must be dim-preserving at d_model="
                    f"{d_model} (got input={hi}, output={ho}). The agnostic and "
                    f"specific streams must return d_model tokens for the GMM head."
                )

    # ------------------------------------------------------------------ #
    # forward (packed, training + teacher-forced eval + AR recompute)
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if not ctx.packed:
            raise NotImplementedError(
                "DualHNetComputeStage supports the PACKED path only "
                "(ctx.cu_seqlens must be set)."
            )
        if self.inner_stage is not None:
            raise NotImplementedError(
                "DualHNetComputeStage is terminal (no inner_stage chaining)."
            )
        dual = ctx.extras.get("dual")
        if dual is None:
            raise RuntimeError(
                "DualHNetComputeStage requires ctx.extras['dual'] "
                "(set by DualStreamOuterStage.encode)."
            )
        T_total = int(dual["T_total"])
        if x.shape[0] != 2 * T_total:
            raise RuntimeError(
                f"DualHNetComputeStage expected x of length 2*T_total="
                f"{2 * T_total}, got {x.shape[0]}."
            )

        emb = ctx.embodiment_id
        if emb not in self.hnet_specific:
            raise KeyError(
                f"DualHNetComputeStage has no specific H-Net for embodiment "
                f"{emb!r}; available: {sorted(self.hnet_specific.keys())}."
            )
        hnet_s = self.hnet_specific[emb]

        A = x[:T_total]
        S = x[T_total:]

        # Run each H-Net in its OWN sub-context so their internal cu_seqlens /
        # max_seqlen mutations (ChunkerStage swaps in the chunked-space cu and
        # restores it via try/finally) and their aux lists stay isolated. Both
        # streams share the SAME per-stream packing (ctx.cu_seqlens describes the
        # T_total token axis) and the shared cond_dict. inference_params is None:
        # the inner H-Nets always run their packed forward here (AR is recompute-
        # over-prefix, never a single-token step on these H-Nets).
        ctx_a = HNetContext(
            cond_dict=ctx.cond_dict,
            aux=[],
            inference_params=None,
            cu_seqlens=ctx.cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            embodiment_id=emb,
        )
        ctx_s = HNetContext(
            cond_dict=ctx.cond_dict,
            aux=[],
            inference_params=None,
            cu_seqlens=ctx.cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            embodiment_id=emb,
        )

        A_top = self.hnet_agnostic(A, ctx_a)  # (T_total, d_model)
        S_top = hnet_s(S, ctx_s)  # (T_total, d_model)

        # Merge both H-Nets' chunker aux into the OUTER ctx.aux so the algo's
        # GMMLoss/ratio_loss_from_aux SUMS every chunker's ratio loss and
        # chunk_stats_from_aux indexes them positionally (agnostic chunkers ->
        # _0.., specific chunkers -> _k..). Deterministic order: agnostic first.
        if ctx_a.aux:
            ctx.aux.extend(ctx_a.aux)
        if ctx_s.aux:
            ctx.aux.extend(ctx_s.aux)

        ctx.extras["specific_tokens"] = S_top
        return A_top

    # ------------------------------------------------------------------ #
    # AR / inference surface. Closed-loop AR for the dual stream is
    # recompute-over-prefix: DualStreamOuterStage.step rebuilds the packed
    # prefix and calls THIS stage's ``forward`` (never ``step``). These exist
    # only so the HNet wrapper contract holds (allocate_inference_cache ->
    # root._allocate); there is no per-token state to cache because each inner
    # H-Net is re-run over the whole prefix.
    # ------------------------------------------------------------------ #
    def step(self, x: torch.Tensor, ctx: HNetContext, state):
        raise NotImplementedError(
            "DualHNetComputeStage has no single-token step; closed-loop AR is "
            "recompute-over-prefix (DualStreamOuterStage.step calls forward)."
        )

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return None  # recompute-over-prefix: nothing to cache.

    def allocate_inference_cache(
        self, batch_size, max_seqlen=None, device=None, dtype=None
    ):
        return None

    # ------------------------------------------------------------------ #
    # Training-recipe init hook (opt-in; mirrors DualStreamComputeStage). Each
    # inner H-Net does its own residual-stream-aware init internally, so we
    # delegate and leave the parent residual count unchanged (terminal stage).
    # ------------------------------------------------------------------ #
    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        self.hnet_agnostic.init_weights(initializer_range)
        for h in self.hnet_specific.values():
            h.init_weights(initializer_range)
        return parent_residuals
