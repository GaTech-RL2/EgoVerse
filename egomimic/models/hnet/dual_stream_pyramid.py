"""Non-terminal dual-stream compute stage for the tapered dual-stream pyramid.

Carries a :class:`StreamBundle`, runs ``MultiStreamTrunk`` (per-stream intra
self-attn + gated cross-stream comm), then passes the bundle to ``inner_stage``
(the next chunker). Used for the INTERMEDIATE (non-apex) compute levels of the
tapered pyramid; the apex stays the ordinary single-stream ``ComputeStage``
(agnostic-only). Dim-preserving (input==output per stream); the chunker tapers.

With ``embodiments`` set, the trunk's SPECIFIC stream(s) (idx>=1) are
per-embodiment (swapped per embodiment via ctx.embodiment_id); the AGNOSTIC
stream stays shared. See MultiStreamTrunk for the per-emb contract.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from egomimic.models.hnet.multi_stream_trunk import MultiStreamTrunk
from egomimic.models.hnet.stages import _BaseStage
from egomimic.models.hnet.stream_bundle import StreamBundle, build_same_episode_causal_mask


class DualBundleComputeStage(_BaseStage):
    def __init__(self, input_hidden_dim, output_hidden_dim, streams_cfg, adjacency=None,
                 n_layers=6, rotary_emb_dim=0, mask_mode="asym", causal=True,
                 cond_key=None, dropout=0.0, embodiments: Optional[List[str]] = None,
                 allow_agnostic_cross: bool = False, decoder_layout=None):
        super().__init__(input_hidden_dim, output_hidden_dim, cond_key=cond_key)
        if int(input_hidden_dim) != int(output_hidden_dim):
            raise ValueError("DualBundleComputeStage preserves dim (input==output).")
        streams_cfg = [dict(s) for s in streams_cfg]
        if int(streams_cfg[0]["d_model"]) != int(input_hidden_dim):
            raise ValueError(
                f"streams_cfg[0].d_model ({streams_cfg[0]['d_model']}) (agnostic) "
                f"must == input_hidden_dim ({input_hidden_dim})."
            )
        if mask_mode not in ("asym", "sym"):
            raise ValueError(f"mask_mode must be 'asym'|'sym', got {mask_mode!r}")
        N = len(streams_cfg)
        if adjacency is None:
            if mask_mode == "sym":
                adjacency = [[j for j in range(N) if j != i] for i in range(N)]
            else:  # asym: agnostic (0) reads no cross; each specific reads agnostic
                adjacency = [[] if i == 0 else [0] for i in range(N)]
        self.mask_mode = mask_mode
        self.causal = bool(causal)
        self.streams_cfg = streams_cfg
        self.embodiments = list(embodiments) if embodiments else None
        self.trunk = MultiStreamTrunk(streams_cfg, adjacency, n_layers, rotary_emb_dim, dropout,
                                      embodiments=embodiments, allow_agnostic_cross=allow_agnostic_cross)
        # --- DECODE-side trunk (mirrors stages.py EncoderDecoderStage: encoder ->
        #     inner_stage -> decoder, ALL in one stage). A SECOND MultiStreamTrunk
        #     with the SAME streams_cfg/adjacency/rope (dim-preserving, A + S), run
        #     AFTER the inner chunker returns its already-residual-mixed bundle. It
        #     just transforms that output — NO new residual/skip is added here (the
        #     single STE-gated residual lives inside the chunker). ``decoder_layout``
        #     (e.g. 'T4' == n_layers 4, or an int) selects depth. DEFAULT None =>
        #     no module built, forward byte-identical (BC). ---
        self.decoder_layout = decoder_layout
        dec_n_layers = self._n_layers_from_layout(decoder_layout)
        self.decoder_trunk = (
            MultiStreamTrunk(streams_cfg, adjacency, dec_n_layers, rotary_emb_dim, dropout,
                             embodiments=embodiments, allow_agnostic_cross=allow_agnostic_cross)
            if dec_n_layers is not None
            else None
        )
        self.inner_stage = None  # set by HNet (the next chunker)

    @staticmethod
    def _n_layers_from_layout(layout):
        """Map a decoder_layout spec to an int layer count. Accepts 'T4'/'t4'
        (Isotropic-style), a bare int/'4', or None (=> no decoder)."""
        if layout is None:
            return None
        if isinstance(layout, int):
            return int(layout)
        s = str(layout).strip()
        if s and s[0] in ("T", "t"):
            s = s[1:]
        return int(s)

    @property
    def inner_working_dim(self) -> int:
        return self.output_hidden_dim  # dim-preserving; the next chunker tapers

    def forward(self, x, ctx) -> StreamBundle:  # type: ignore[override]
        if not isinstance(x, StreamBundle):
            raise TypeError("DualBundleComputeStage requires a StreamBundle.")
        if not ctx.packed:
            raise NotImplementedError("DualBundleComputeStage supports the PACKED path only.")
        dev = x.A.device
        mask = build_same_episode_causal_mask(x.T_total, x.cu_seqlens, x.time_pos, self.causal, dev)
        rope = x.time_pos.to(device=dev, dtype=torch.long)
        tops = self.trunk([x.A, x.S], mask, rope, ctx.embodiment_id)
        bundle = x.replace(A=tops[0], S=tops[1])
        if self.inner_stage is not None:
            bundle = self.inner_stage(bundle, ctx)
        # DECODE-side trunk: transform the chunker's (already-residual-mixed)
        # output. bundle.A/S are back at this stage's dims + full outer length
        # (the chunker dechunks), so the SAME mask/rope apply. No extra residual.
        if self.decoder_trunk is not None:
            douts = self.decoder_trunk([bundle.A, bundle.S], mask, rope, ctx.embodiment_id)
            bundle = bundle.replace(A=douts[0], S=douts[1])
        return bundle

    def step(self, x, ctx, state):
        raise NotImplementedError(
            "DualBundleComputeStage has no incremental step; AR is recompute-over-prefix."
        )

    def _allocate(self, batch_size, max_seqlen, dtype, device):
        return None

    def allocate_inference_cache(self, batch_size, dtype, device):
        return None

    def _init_weights(self, initializer_range: float, parent_residuals: int) -> int:
        # Mirrors MultiStreamComputeStage._init_weights (scaled residual-out init).
        # Both the encoder trunk AND the optional decoder trunk contribute to this
        # stage's residual stream (mirrors EncoderDecoderStage counting
        # encoder.height + decoder.height), so scale by their combined depth.
        n_residuals = parent_residuals + self.trunk.height
        if self.decoder_trunk is not None:
            n_residuals += self.decoder_trunk.height
        scaled_std = initializer_range / max(n_residuals, 1) ** 0.5
        trunks = [self.trunk] + ([self.decoder_trunk] if self.decoder_trunk is not None else [])
        for trunk in trunks:
            for name, m in trunk.named_modules():
                if not isinstance(m, nn.Linear) or getattr(m.weight, "_no_reinit", False):
                    continue
                if name.endswith("out") or "w2" in name or "fc2" in name or "down" in name:
                    nn.init.normal_(m.weight, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Chain init into the inner stage (mirrors single-stream ChunkerStage/
        # ComputeStage). Without this, HNet.init_weights (root-only) would leave
        # every downstream stage at PyTorch default init.
        if self.inner_stage is not None:
            n_residuals = self.inner_stage._init_weights(initializer_range, n_residuals)
        return n_residuals
