"""Dual-stream compute stage for the cross-embodiment GMM H-Net (v2: TWO trunks).

TWO transformer trunks with GENUINELY SEPARATE WEIGHTS — the weight-shared
*agnostic* stream ``A`` and the per-embodiment *specific* stream ``S`` each get
their own per-layer RMSNorm + QKV + out-proj + SwiGLU MLP (see
``TwoStreamTrunk``). Streams talk every layer through a channel-aware CROSS-
ATTENTION: per-stream Q/K/V are concatenated into one ``2T``-token sequence and
attended under a boolean ``(2T, 2T)`` allow-mask (asym: A attends A-only, S
attends both; sym: both attend both). This replaces v1's single shared
``Isotropic`` trunk + 2T-token mask (which shared ALL weights between A and S).

v2 scope (unchanged from v1 except the trunk):
  * PACKED path only (asserts ``ctx.packed``).
  * NO depth split (``depth_split_k=None``).
  * NO incremental KV-cache: closed-loop AR is recompute-over-prefix, driven by
    ``DualStreamOuterStage.step`` calling ``forward`` over the growing prefix.

Layout (set by ``DualStreamOuterStage.encode`` in ``ctx.extras['dual']``):
  x = cat([A, S], dim=0)   with A = x[:T_total], S = x[T_total:]   (L = 2*T_total)
  dual = {"T_total": int, "cu_seqlens": (B+1,), "time_pos": (T_total,)}

Attention allow-mask (query i, key j over the full 2T sequence):
    allow = same_episode & time_causal & channel_rule  (diagonal forced True)
  - same_episode: A_t and S_t of episode e share episode e; cross-episode blocked.
  - time_causal:  key time <= query time (A_t and S_t SHARE time t).
  - channel_rule (mask_mode):
      "asym": agnostic-query attends agnostic-keys ONLY; specific-query both.
      "sym":  every query attends both streams.

RoPE positions = the within-episode time index (A_t and S_t both at position t).
Returns the post-trunk + final-norm agnostic tokens ``A_top``; stashes the post-
trunk + final-norm specific tokens in ``ctx.extras['specific_tokens']`` for the
partitioned GMM head.
"""

from typing import Optional

import torch
import torch.nn as nn

from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.stages import _BaseStage
from egomimic.models.hnet.two_stream_trunk import TwoStreamTrunk


class DualStreamComputeStage(_BaseStage):
    """Stage with TWO separate-weight trunks (agnostic ‖ specific) + per-layer
    channel-aware cross-attention. Terminal (no ``inner_stage`` chaining).

    Args:
        input_hidden_dim / output_hidden_dim: must both equal ``d_model``.
        d_model / num_heads / d_intermediate / n_layers / rotary_emb_dim:
            per-stream trunk geometry (each stream is this size; the two
            streams together are the capacity-matched component).
        mask_mode:     "asym" (A sees A only; S sees both) | "sym" (all see all).
        cond_key / d_cond / causal: kept for config symmetry. cond is OFF (the
            trunk takes no AdaLN cond); ``causal`` is enforced by the allow-mask
            (``time_causal``), not by the attention module.
        depth_split_k: reserved for a future depth-split; v2 requires None.
        dropout:       attention dropout (training only).
    """

    def __init__(
        self,
        input_hidden_dim: int,
        output_hidden_dim: int,
        d_model: int,
        num_heads: int,
        d_intermediate: int,
        n_layers: int,
        rotary_emb_dim: int = 0,
        mask_mode: str = "asym",
        cond_key: Optional[str] = None,
        d_cond: int = 0,
        causal: bool = True,
        depth_split_k: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if input_hidden_dim != output_hidden_dim:
            raise ValueError(
                f"DualStreamComputeStage requires input_hidden_dim==output_hidden_dim "
                f"(got {input_hidden_dim} vs {output_hidden_dim})."
            )
        if int(d_model) != int(input_hidden_dim):
            raise ValueError(
                f"d_model ({d_model}) must equal stage input_hidden_dim "
                f"({input_hidden_dim})."
            )
        if mask_mode not in ("asym", "sym"):
            raise ValueError(f"mask_mode must be 'asym'|'sym', got {mask_mode!r}")
        if depth_split_k is not None:
            raise NotImplementedError(
                "DualStreamComputeStage v2 does not implement depth_split_k. "
                "Leave it None."
            )
        self.mask_mode = mask_mode
        self.causal = causal
        self.depth_split_k = depth_split_k
        self.trunk = TwoStreamTrunk(
            d_model=int(d_model),
            num_heads=int(num_heads),
            d_intermediate=int(d_intermediate),
            n_layers=int(n_layers),
            rotary_emb_dim=int(rotary_emb_dim),
            dropout=float(dropout),
        )

    # ------------------------------------------------------------------
    # Mask construction (identical channel/episode/time semantics to v1).
    # ------------------------------------------------------------------

    def _build_allow_mask(
        self,
        T_total: int,
        cu_seqlens: torch.Tensor,
        time_pos: torch.Tensor,
        device,
    ) -> torch.Tensor:
        """Return the (2T, 2T) bool allow-mask (True = query i may attend key j)."""
        L = 2 * T_total
        cu = cu_seqlens.to(device=device, dtype=torch.long)

        ep = torch.zeros(T_total, dtype=torch.long, device=device)
        for b in range(cu.shape[0] - 1):
            ep[cu[b] : cu[b + 1]] = b
        ep2 = torch.cat([ep, ep], dim=0)  # (2T,)

        tp = time_pos.to(device=device, dtype=torch.long)
        tp2 = torch.cat([tp, tp], dim=0)  # (2T,)

        same_episode = ep2[:, None] == ep2[None, :]      # (2T, 2T)
        time_causal = tp2[None, :] <= tp2[:, None]       # key time <= query time

        is_agnostic = torch.zeros(L, dtype=torch.bool, device=device)
        is_agnostic[:T_total] = True  # first half = agnostic stream
        if self.mask_mode == "sym":
            channel_rule = torch.ones(L, L, dtype=torch.bool, device=device)
        else:  # asym: agnostic-query -> agnostic-keys only; specific-query -> both
            agn_q = is_agnostic[:, None]
            agn_k = is_agnostic[None, :]
            channel_rule = (~agn_q) | agn_k

        allow = same_episode & time_causal & channel_rule
        # Force the diagonal True so no row is all-False (-> SDPA NaN).
        allow = allow | torch.eye(L, dtype=torch.bool, device=device)
        return allow

    # ------------------------------------------------------------------
    # forward.
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, ctx: HNetContext) -> torch.Tensor:
        if not ctx.packed:
            raise NotImplementedError(
                "DualStreamComputeStage supports the PACKED path only "
                "(ctx.cu_seqlens must be set)."
            )
        if self.inner_stage is not None:
            raise NotImplementedError(
                "DualStreamComputeStage is terminal (no inner_stage chaining)."
            )
        dual = ctx.extras.get("dual")
        if dual is None:
            raise RuntimeError(
                "DualStreamComputeStage requires ctx.extras['dual'] "
                "(set by DualStreamOuterStage.encode)."
            )
        T_total = int(dual["T_total"])
        if x.shape[0] != 2 * T_total:
            raise RuntimeError(
                f"DualStreamComputeStage expected x of length 2*T_total="
                f"{2 * T_total}, got {x.shape[0]}."
            )
        device = x.device

        allow = self._build_allow_mask(
            T_total=T_total,
            cu_seqlens=dual["cu_seqlens"],
            time_pos=dual["time_pos"],
            device=device,
        )
        tp = dual["time_pos"].to(device=device, dtype=torch.long)
        rope_positions = torch.cat([tp, tp], dim=0)  # (2T,)

        A = x[:T_total]
        S = x[T_total:]
        A_top, S_top = self.trunk(A, S, allow, rope_positions)
        ctx.extras["specific_tokens"] = S_top
        return A_top

    # ------------------------------------------------------------------
    # AR / sim path — RECOMPUTE-OVER-PREFIX (no KV-cache).
    #
    # ``DualStreamOuterStage.step`` drives the recompute by calling ``forward``
    # over a freshly-built packed prefix; it never routes through ``step`` /
    # ``allocate_inference_cache`` of this stage. These are provided only so the
    # HNet wrapper's contract holds (``allocate_inference_cache`` ->
    # ``root._allocate``); there is no per-token state to cache.
    # ------------------------------------------------------------------

    def step(self, x: torch.Tensor, ctx: HNetContext, state):
        raise NotImplementedError(
            "DualStreamComputeStage has no single-token step; closed-loop AR is "
            "recompute-over-prefix (DualStreamOuterStage.step calls forward)."
        )

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return None  # recompute-over-prefix: nothing to cache.

    def allocate_inference_cache(self, batch_size, max_seqlen=None, device=None, dtype=None):
        return None

    # ------------------------------------------------------------------
    # Residual-stream-aware Linear init (mirrors _init_isotropic_linears, but
    # for the two-stream naming: out_A/out_S attention out-proj + fc2 FFN).
    # ------------------------------------------------------------------

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        n_residuals = parent_residuals + self.trunk.height
        scaled_std = initializer_range / max(n_residuals, 1) ** 0.5
        for name, m in self.trunk.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if getattr(m.weight, "_no_reinit", False):
                continue
            if name.endswith("out_A") or name.endswith("out_S") or "fc2" in name:
                nn.init.normal_(m.weight, mean=0.0, std=scaled_std)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Terminal stage: no inner_stage to recurse into.
        return n_residuals
