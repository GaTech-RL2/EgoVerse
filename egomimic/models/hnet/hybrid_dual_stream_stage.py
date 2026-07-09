"""Hybrid dual-stream + CORAL compute stage for the cross-embodiment GMM H-Net.

A drop-in replacement for ``DualHNetComputeStage`` / ``DualStreamComputeStage``
under ``DualStreamOuterStage``. It is a HYBRID of the two prior variants:

  * SPECIFIC stream ``S`` -> a PER-EMBODIMENT ``nn.ModuleDict`` of genuine
    dynamic-chunking ``HNet``s (EncDec -> Chunker -> Compute), exactly as in
    ``DualHNetComputeStage``. Each runs its own router/chunk/inner-trunk/dechunk
    on its own (obs) stream. Output ``S_top`` (T_total, d) -> GMM specific modes.
  * AGNOSTIC stream ``A`` -> ONE shared, CHUNKER-FREE transformer ``HNet`` (the
    embodiment-agnostic encoder/trunk/decoder, full token resolution -- NO
    dynamic chunking). It produces ``A_full`` (T_total, d).

The two trunks are coupled by a SINGLE one-way cross-attention: the agnostic
trunk attends to the SPECIFIC trunk's compressed tokens ``S_M`` (and NOT the
other way around). Because the specific H-Net's inner trunk lives in chunked
space (M < T_total tokens) while the agnostic trunk is full-res, we bridge the
mismatch by COMPRESSING THE AGNOSTIC at the SAME chunk boundaries the specific
H-Net's chunker produced (first-token gather -> ``A_M`` (M, d)), running an
equal-length cross-attention ``A_M`` (queries) over ``detach(S_M)`` (keys /
values), then duplicate-dechunking the cross-attention residual back to T_total
and adding it onto ``A_full``. So:

    A_full = agnostic_hnet(A)                    # (T_total, d) full-res
    A_M    = A_full[boundary_mask]               # (M, d)  first-token gather
    cross  = cross_attn(A_M, detach(S_M))        # (M, d)  chunk-causal x-attn
    A_top  = A_full + dechunk_duplicate(cross)   # (T_total, d) residual inject
    -> A_top -> GMM agnostic modes + CORAL only.

Borrowing the specific's boundaries makes ``A_M`` and ``S_M`` the same length M
with identical chunk structure, so the EXISTING ``CrossMultiHeadAttention``
(equal-length packed) applies directly; compress/dechunk are parameter-free
index ops that exactly invert the chunker's first-token/duplicate interfaces.

Gradient contract:
  * ``S_M`` is DETACHED before the agnostic reads it, so the specific H-Net is
    shaped ONLY by its own GMM specific modes (+ ratio loss), never by the
    agnostic's loss.
  * ``A_top`` is shaped by the agnostic GMM modes + the auxiliary CORAL loss
    (computed in the algo across embodiments). It is NOT fed anywhere except the
    GMM head + CORAL.
  * The cross-attention ``out_proj`` is zero-initialised so at init
    ``A_top == A_full`` (the cross-attention ramps in during training).

Layout / protocol (UNCHANGED -- this is why it is a drop-in under
``DualStreamOuterStage``):
  x = cat([A, S], dim=0)   with A = x[:T_total], S = x[T_total:]   (L = 2*T_total)
  ctx.extras['dual'] = {"T_total": int, "cu_seqlens": (B+1,), "time_pos": (T,)}
  returns ``A_top`` (T_total, d); stashes ``S_top`` in
  ctx.extras['specific_tokens'] and ``A_top`` in ctx.extras['agnostic_repr']
  (for the head + CORAL).

Scope: PACKED path only; terminal stage (no inner_stage). Closed-loop AR is
recompute-over-prefix (``DualStreamOuterStage.step`` re-runs ``forward``).
"""

from typing import Dict

import torch
import torch.nn as nn

from egomimic.models.hnet.blocks import CrossMultiHeadAttention, RMSNorm
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet
from egomimic.models.hnet.stages import _BaseStage


class HybridDualStreamCoralStage(_BaseStage):
    """Terminal hybrid stage: shared chunker-free agnostic trunk that attends
    (one-way, detached) to per-embodiment chunking specific H-Nets.

    Args:
        input_hidden_dim / output_hidden_dim: must be equal (dim-preserving) and
            equal the obs-token width ``d_model`` fed by ``DualStreamOuterStage``.
        hnet_agnostic: an instantiated CHUNKER-FREE ``HNet`` (the shared
            embodiment-agnostic encoder/trunk/decoder). Must be dim-preserving at
            ``input_hidden_dim``. It MUST NOT contain a ChunkerStage -- the
            agnostic stream stays at full token resolution.
        hnet_specific: a ``dict[domain -> HNet]`` (one per embodiment), each a
            dynamic-chunking H-Net dim-preserving at ``input_hidden_dim``.
        num_heads: heads for the agnostic->specific cross-attention.
        cross_attn_causal: if True (default) the agnostic chunk-token at chunk
            position k attends only to specific chunk-tokens <= k (chunk-space
            causal), matching the streams' time-causal semantics.
        cond_key: kept for config symmetry; unused (sub-streams run cond-free).
    """

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        hnet_agnostic: HNet,
        hnet_specific: Dict[str, HNet],
        num_heads: int = 8,
        cross_attn_causal: bool = True,
        cond_key=None,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if int(input_hidden_dim) != int(output_hidden_dim):
            raise ValueError(
                f"HybridDualStreamCoralStage requires input_hidden_dim=="
                f"output_hidden_dim (got {input_hidden_dim} vs {output_hidden_dim})."
            )
        d_model = int(input_hidden_dim)

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

        # Dim-preserving check (fail loud at construction).
        for name, h in [
            ("agnostic", self.hnet_agnostic),
            *self.hnet_specific.items(),
        ]:
            hi, ho = int(h.input_hidden_dim), int(h.output_hidden_dim)
            if hi != d_model or ho != d_model:
                raise ValueError(
                    f"inner H-Net '{name}' must be dim-preserving at d_model="
                    f"{d_model} (got input={hi}, output={ho})."
                )

        # One-way agnostic->specific cross-attention at the (chunked) trunk level.
        # d_cond == d_model: S_M arrives at d_model. out_proj zero-init so the
        # cross-attention is a no-op at init (A_top == A_full -> agnostic GMM
        # modes train cleanly from step 0; the cross-attention ramps in).
        self.cross_norm = RMSNorm(d_model)
        self.cross_attn = CrossMultiHeadAttention(
            d_model=d_model,
            d_cond=d_model,
            num_heads=int(num_heads),
            causal=bool(cross_attn_causal),
        )
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)

    # ------------------------------------------------------------------ #
    # forward (packed; training + teacher-forced eval + AR recompute)
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if not ctx.packed:
            raise NotImplementedError(
                "HybridDualStreamCoralStage supports the PACKED path only."
            )
        if self.inner_stage is not None:
            raise NotImplementedError(
                "HybridDualStreamCoralStage is terminal (no inner_stage chaining)."
            )
        dual = ctx.extras.get("dual")
        if dual is None:
            raise RuntimeError(
                "HybridDualStreamCoralStage requires ctx.extras['dual'] "
                "(set by DualStreamOuterStage.encode)."
            )
        T_total = int(dual["T_total"])
        if x.shape[0] != 2 * T_total:
            raise RuntimeError(
                f"HybridDualStreamCoralStage expected x of length 2*T_total="
                f"{2 * T_total}, got {x.shape[0]}."
            )

        emb = ctx.embodiment_id
        if emb not in self.hnet_specific:
            raise KeyError(
                f"HybridDualStreamCoralStage has no specific H-Net for embodiment "
                f"{emb!r}; available: {sorted(self.hnet_specific.keys())}."
            )
        hnet_s = self.hnet_specific[emb]

        A = x[:T_total]
        S = x[T_total:]

        # 1) SPECIFIC stream first: run the per-embodiment chunking H-Net and have
        #    its chunker expose the compressed trunk tokens S_M + boundaries via
        #    the opt-in hook (ctx.extras['_expose_chunk_trunk']).
        expose: list = []
        ctx_s = HNetContext(
            cond_dict=ctx.cond_dict,
            aux=[],
            inference_params=None,
            extras={"_expose_chunk_trunk": expose},
            cu_seqlens=ctx.cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            embodiment_id=emb,
        )
        S_top = hnet_s(S, ctx_s)  # (T_total, d)

        if len(expose) == 0:
            raise RuntimeError(
                "HybridDualStreamCoralStage: the specific H-Net exposed no chunker "
                "trunk tokens. The specific H-Net must contain a ChunkerStage "
                "(first_token/duplicate) so the agnostic can borrow its boundaries."
            )
        # The OUTERMOST chunker (first appended) operates on the full-res T_total
        # stream -- that is the one whose boundaries align with A_full.
        sm = expose[0]
        S_M = sm["trunk"]  # (M, d) compressed specific trunk tokens
        boundary_mask = sm["boundary_mask"]  # (T_total,) bool
        next_cu = sm["next_cu"]  # (B'+1,) chunked-space cu_seqlens

        # 2) AGNOSTIC stream: shared chunker-free transformer at full resolution.
        ctx_a = HNetContext(
            cond_dict=ctx.cond_dict,
            aux=[],
            inference_params=None,
            cu_seqlens=ctx.cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            embodiment_id=emb,
        )
        A_full = self.hnet_agnostic(A, ctx_a)  # (T_total, d)

        # 3) Cross-attention residual: compress A_full at the SPECIFIC boundaries,
        #    attend detach(S_M), duplicate-dechunk back to T_total, add onto A_full.
        A_top = A_full + self._cross_residual(A_full, S_M, boundary_mask, next_cu)

        # Merge chunker aux (agnostic is chunker-free -> contributes none; only the
        # specific chunkers register ratio-loss aux).
        if ctx_a.aux:
            ctx.aux.extend(ctx_a.aux)
        if ctx_s.aux:
            ctx.aux.extend(ctx_s.aux)

        ctx.extras["specific_tokens"] = S_top
        ctx.extras["agnostic_repr"] = A_top
        return A_top

    # ------------------------------------------------------------------ #
    # cross-attention residual (compress @ boundaries -> x-attn -> dechunk)
    # ------------------------------------------------------------------ #
    def _cross_residual(
        self,
        A_full: torch.Tensor,
        S_M: torch.Tensor,
        boundary_mask: torch.Tensor,
        next_cu: torch.Tensor,
    ) -> torch.Tensor:
        """Return the (T_total, d) cross-attention residual added to A_full.

        ``boundary_mask`` (T_total,) bool marks each chunk's first frame (the
        chunk-start), exactly as the specific chunker's first_token interface
        used. ``S_M`` (M, d) are the specific trunk tokens in chunk order; M ==
        boundary_mask.sum(). We:
          * COMPRESS: A_M = A_full[boundary_mask]            (M, d)  first-token
          * X-ATTN:   cross = cross_attn(norm(A_M), detach(S_M), next_cu)  (M, d)
          * DECHUNK:  broadcast cross[chunk_id(t)] back to T_total  (duplicate)
        """
        bm = boundary_mask.to(device=A_full.device)
        if bm.dtype != torch.bool:
            bm = bm != 0
        M = int(bm.sum().item())
        if M != int(S_M.shape[0]):
            raise RuntimeError(
                f"HybridDualStreamCoralStage: boundary count ({M}) != specific "
                f"trunk token count ({int(S_M.shape[0])}); the agnostic compression "
                f"must align 1:1 with the specific chunks."
            )
        # COMPRESS (first-token gather, chunk order == increasing frame index).
        A_M = A_full[bm]  # (M, d)
        # X-ATTN at the chunked resolution; detach S_M so no grad to specific.
        h = self.cross_norm(A_M)
        cross = self.cross_attn(
            h, S_M.detach(), cu_seqlens=next_cu, max_seqlen=None
        )  # (M, d)
        # DECHUNK (duplicate-broadcast): frame t -> its chunk index.
        chunk_id = torch.cumsum(bm.long(), dim=0) - 1  # (T_total,), >=0
        return cross[chunk_id]  # (T_total, d)

    # ------------------------------------------------------------------ #
    # AR / inference surface (recompute-over-prefix; mirrors the dual stages).
    # ------------------------------------------------------------------ #
    def step(self, x: torch.Tensor, ctx: HNetContext, state):
        raise NotImplementedError(
            "HybridDualStreamCoralStage has no single-token step; closed-loop AR "
            "is recompute-over-prefix (DualStreamOuterStage.step calls forward)."
        )

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return None  # recompute-over-prefix: nothing to cache.

    def allocate_inference_cache(
        self, batch_size, max_seqlen=None, device=None, dtype=None
    ):
        return None

    # ------------------------------------------------------------------ #
    # Training-recipe init hook (opt-in). Each inner H-Net self-inits via its own
    # recursion; the cross-attention lives on THIS stage (not inside any inner
    # H-Net) so the per-stage init never touches it -- its zero-init out_proj
    # from __init__ stays put, keeping the residual-injection a no-op at init.
    # ------------------------------------------------------------------ #
    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        self.hnet_agnostic.init_weights(initializer_range)
        for h in self.hnet_specific.values():
            h.init_weights(initializer_range)
        return parent_residuals
