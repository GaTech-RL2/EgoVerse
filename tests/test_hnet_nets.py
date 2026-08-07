"""
Unit tests for egomimic.models.hnet.

CPU-only: every test runs against the pure-PyTorch fallback paths so the
absence of flash_attn / mamba_ssm / causal_conv1d does not gate test
execution. Where CUDA kernels would change behavior (DeChunkLayer EMA via
mamba_chunk_scan, MultiHeadAttention via flash_attn_varlen), the pure-Python
recurrence / SDPA path is the reference — and the kernel path is documented
to match by construction (same EMA recurrence; same causal varlen mask).

Tests are grouped by module:
  - RoutingModule
  - ChunkLayer
  - DeChunkLayer
  - Isotropic
  - Stages (EncoderDecoder / Chunker / Compute)
  - HNet end-to-end
  - ratio_loss_from_aux
"""

from __future__ import annotations

import pytest
import torch

from egomimic.models.hnet.blocks import (
    AdaLNModulation,
    Isotropic,
    MultiHeadAttention,
    RMSNorm,
    _adaln,
)
from egomimic.models.hnet.config import AttnConfig, HNetConfig, SSMConfig
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import (
    HNet,
    chunk_stats_from_aux,
    ratio_loss_from_aux,
)
from egomimic.models.hnet.routing import (
    ChunkLayer,
    DeChunkLayer,
    RoutingModule,
    get_seq_idx,
    routing_output_from_probs,
)
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
    _scatter_state,
    _slice_state,
    _ste,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _seed(s: int = 0):
    torch.manual_seed(s)


def _make_iso(
    d_model=32,
    n_layers=2,
    num_heads=4,
    d_inter=64,
    causal=True,
    d_cond=0,
    cond_here=False,
) -> Isotropic:
    cfg = HNetConfig(
        arch_layout=[f"T{n_layers}"],
        d_model=[d_model],
        d_intermediate=[d_inter],
        attn_cfg=AttnConfig(num_heads=[num_heads]),
        ssm_cfg=SSMConfig(),
    )
    return Isotropic(
        cfg,
        pos_idx=0,
        stage_idx=0,
        d_cond=d_cond,
        cond_here=cond_here,
        causal=causal,
    )


# =========================================================================== #
# RoutingModule
# =========================================================================== #


class TestRoutingModule:
    def test_output_helper_preserves_deterministic_tie_break(self):
        p = torch.tensor([0.0, 0.5, 1.0])
        out = routing_output_from_probs(p)

        assert torch.equal(
            out.boundary_prob,
            torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        )
        assert torch.equal(out.boundary_mask, torch.tensor([False, False, True]))
        assert torch.equal(out.selected_probs, torch.tensor([[1.0], [0.5], [1.0]]))

    def test_padded_first_token_always_boundary(self):
        _seed()
        rm = RoutingModule(d_model=8)
        x = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.bool)
        out = rm(x, mask=mask)
        assert out.boundary_mask[
            :, 0
        ].all(), "PAD_PROB=1.0 on index 0 must force first token to be a boundary"

    def test_padded_boundary_mask_respects_input_mask(self):
        _seed()
        rm = RoutingModule(d_model=8)
        x = torch.randn(2, 6, 8)
        mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        # Force ALL boundary probs above threshold by replacing q/k weights with
        # a sign-flipper that produces cos_sim=-1 → boundary_prob=1.
        with torch.no_grad():
            rm.k_proj_layer.weight.copy_(-torch.eye(8))
        out = rm(x, mask=mask)
        # boundary_mask is selected_idx==1 AND mask, so no boundary outside mask.
        assert (out.boundary_mask & ~mask).sum() == 0

    def test_padded_identical_tokens_no_internal_boundaries(self):
        """If consecutive hidden states are identical, cos_sim=1 → boundary_prob=0.
        Only the forced first-token boundary should remain."""
        rm = RoutingModule(d_model=8)
        x = torch.ones(1, 5, 8)  # identical along T
        mask = torch.ones(1, 5, dtype=torch.bool)
        out = rm(x, mask=mask)
        assert out.boundary_mask[0, 0].item() is True
        assert out.boundary_mask[0, 1:].sum().item() == 0

    def test_packed_subseq_starts_are_boundaries(self):
        _seed()
        rm = RoutingModule(d_model=8)
        cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.long)
        T_total = int(cu_seqlens[-1])
        x = torch.randn(T_total, 8)
        out = rm(x, cu_seqlens=cu_seqlens)
        # Every subseq-start index must be a boundary.
        for s in cu_seqlens[:-1].tolist():
            assert out.boundary_mask[s].item() is True

    def test_packed_padded_equivalence_single_seq(self):
        """For a single packed subseq, packed-mode boundary_prob (index 1)
        must equal padded-mode boundary_prob (index 1) elementwise."""
        _seed()
        rm = RoutingModule(d_model=8)
        T = 12
        x = torch.randn(T, 8)

        out_p = rm(x, cu_seqlens=torch.tensor([0, T], dtype=torch.long))
        out_d = rm(x.unsqueeze(0), mask=torch.ones(1, T, dtype=torch.bool))

        assert torch.allclose(
            out_p.boundary_prob[..., 1], out_d.boundary_prob[0, :, 1], atol=1e-6, rtol=0
        )
        assert torch.equal(out_p.boundary_mask, out_d.boundary_mask[0])

    def test_step_matches_prefill_then_next_token(self):
        """Prefill T-1 tokens via padded forward (with inference_params), then
        step on token T-1. The boundary_prob at position T-1 in step output
        must match what a full padded forward would have produced."""
        _seed()
        rm = RoutingModule(d_model=8)
        T = 6
        x = torch.randn(1, T, 8)

        # Reference: full padded forward over all T tokens.
        full = rm(x, mask=torch.ones(1, T, dtype=torch.bool))
        ref_prob = full.boundary_prob[0, T - 1, 1]  # boundary prob at last position

        # Prefill T-1 tokens, then step the last.
        state = rm.allocate_inference_cache(
            batch_size=1, max_seqlen=T, device=x.device, dtype=torch.float32
        )
        rm(
            x[:, : T - 1],
            mask=torch.ones(1, T - 1, dtype=torch.bool),
            inference_params=state,
        )
        step_out = rm.step(x[:, T - 1 : T], state)
        step_prob = step_out.boundary_prob[0, 1]

        assert torch.allclose(
            step_prob, ref_prob, atol=1e-6, rtol=0
        ), f"step prob {step_prob.item():.6f} vs full {ref_prob.item():.6f}"

    def test_step_first_token_forced_boundary(self):
        """First step() call (has_seen_tokens=False) must report boundary."""
        rm = RoutingModule(d_model=8)
        state = rm.allocate_inference_cache(
            1, 4, device=torch.device("cpu"), dtype=torch.float32
        )
        x = torch.randn(1, 1, 8)
        out = rm.step(x, state)
        assert out.boundary_mask.item() is True
        assert torch.allclose(out.boundary_prob[..., 1], torch.tensor([1.0]))

    def test_get_seq_idx_correct(self):
        cu = torch.tensor([0, 3, 7, 10], dtype=torch.long)
        idx = get_seq_idx(cu)
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=torch.long)
        assert torch.equal(idx, expected)


# =========================================================================== #
# ChunkLayer
# =========================================================================== #


class TestChunkLayer:
    def test_padded_gathers_correct_tokens(self):
        cl = ChunkLayer()
        x = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
        bm = torch.tensor(
            [
                [1, 0, 1, 0, 1],
                [1, 1, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        out, _, _, next_mask = cl(x, bm, mask=torch.ones(2, 5, dtype=torch.bool))
        # next_max_seqlen = max(3, 2) = 3 — for row 0 first 3 are real, for
        # row 1 first 2 are real (positions 3..M-1 are garbage tails).
        # Boundary positions in row 0: indices [0, 2, 4]; row 1: [0, 1].
        assert torch.equal(out[0, :3], x[0, [0, 2, 4]])
        assert torch.equal(out[1, :2], x[1, [0, 1]])
        # next_mask reflects counts.
        assert torch.equal(next_mask[0], torch.tensor([1, 1, 1], dtype=torch.bool))
        assert torch.equal(next_mask[1], torch.tensor([1, 1, 0], dtype=torch.bool))

    def test_packed_gathers_correct_tokens(self):
        cl = ChunkLayer()
        T = 7
        x = torch.arange(T * 3).reshape(T, 3).float()
        bm = torch.tensor([1, 0, 1, 1, 0, 1, 0], dtype=torch.bool)
        cu = torch.tensor([0, 4, 7], dtype=torch.long)
        out, next_cu, next_max, _ = cl(x, bm, cu_seqlens=cu)
        # Gathered tokens: positions [0, 2, 3, 5]
        assert torch.equal(out, x[[0, 2, 3, 5]])
        # next_cu: cumsum-boundaries at sub-seq ends → after pos 3: 3 boundaries;
        # after pos 6: 4 boundaries.
        assert torch.equal(next_cu, torch.tensor([0, 3, 4], dtype=torch.long))
        assert next_max == 3

    def test_padded_all_boundaries_is_identity(self):
        cl = ChunkLayer()
        x = torch.randn(2, 5, 4)
        bm = torch.ones(2, 5, dtype=torch.bool)
        out, _, _, _ = cl(x, bm, mask=torch.ones(2, 5, dtype=torch.bool))
        assert torch.equal(out, x)


# =========================================================================== #
# DeChunkLayer
# =========================================================================== #


class TestDeChunkLayer:
    def _manual_padded_ema(self, x, p):
        """Reference: per-row EMA along axis 1."""
        B, M, D = x.shape
        out = torch.zeros_like(x)
        prev = torch.zeros(B, D, dtype=x.dtype)
        for t in range(M):
            pt = p[:, t : t + 1]
            prev = pt * x[:, t] + (1 - pt) * prev
            out[:, t] = prev
        return out

    def test_padded_ema_loop_matches_manual(self):
        _seed()
        dl = DeChunkLayer(d_model=8, headdim=4)
        B, L, D = 2, 5, 8
        x_full = torch.randn(B, L, D)
        # Boundary at positions [0, 2, 4] for both rows → 3 chunks each.
        bm = torch.zeros(B, L, dtype=torch.bool)
        bm[:, [0, 2, 4]] = True
        # Random boundary_prob (we only use index -1 = boundary prob)
        bp_yes = torch.rand(B, L).clamp(0.1, 0.9)
        bp = torch.stack([1 - bp_yes, bp_yes], dim=-1)
        # Build chunked hidden_states with M = max chunks = 3.
        x_chunked = x_full[:, [0, 2, 4]]
        out = dl(x_chunked, bm, bp, mask=torch.ones(B, L, dtype=torch.bool))
        # Reference: EMA over the three chunked tokens, then plug-back broadcast.
        # The dechunk gathers chunked p via sorted_idx (boundaries first).
        # In our setup boundaries are already at positions [0,2,4] → after
        # argsort, sorted_idx[:, :3] = [[0,2,4],[0,2,4]] → p_chunked is just bp_yes at those positions.
        p_chunked = torch.stack([bp_yes[i, [0, 2, 4]] for i in range(B)], dim=0)
        p_chunked = p_chunked.clamp(min=1e-4, max=1 - 1e-4)
        ema = self._manual_padded_ema(x_chunked, p_chunked)
        # plug_back_idx = cumsum(bm)-1 = [0,0,1,1,2,...] etc.
        plug = torch.clamp(torch.cumsum(bm, dim=1) - 1, min=0)
        expected = torch.gather(ema, dim=1, index=plug.unsqueeze(-1).expand(-1, -1, D))
        assert torch.allclose(out, expected, atol=1e-5, rtol=0)

    def test_step_matches_loop(self):
        """Loop the step() interface over one batch element and verify it
        matches a manual EMA recurrence."""
        _seed()
        dl = DeChunkLayer(d_model=8, headdim=4)
        B, D = 1, 8
        state = dl.allocate_inference_cache(
            B, max_seqlen=4, device=torch.device("cpu"), dtype=torch.float32
        )
        T = 5
        out_step = []
        prev = torch.zeros(B, D)
        for t in range(T):
            is_boundary = t % 2 == 0
            bm = torch.tensor([is_boundary], dtype=torch.bool)
            p_t = torch.rand(()) * 0.6 + 0.2
            bp = torch.stack([1 - p_t.expand(B), p_t.expand(B)], dim=-1)
            x_t = torch.randn(1, 1, D) if is_boundary else torch.zeros(0, 1, D)
            y = dl.step(x_t, bm, bp, state)
            # Manual: only update prev when boundary fires.
            if is_boundary:
                p_clip = p_t.clamp(min=1e-4, max=1 - 1e-4)
                prev = p_clip * x_t.squeeze(1) + (1 - p_clip) * prev
            else:
                # No boundary → p[bm]=0 → result = 0*0 + 1*prev = prev (DOES NOT update prev forward in time per the impl, since p=0).
                # Verify: p_zero = 0 → result = 0*0 + 1*last_value = last_value
                pass
            out_step.append(y.squeeze(1))
            assert torch.allclose(y.squeeze(1), prev, atol=1e-5, rtol=0)

    def test_padded_packed_equivalence_single_seq(self):
        """Single packed subseq vs padded with mask-all-True must agree."""
        _seed()
        dl = DeChunkLayer(d_model=8, headdim=4)
        L, D = 6, 8
        x_full = torch.randn(L, D)
        bm = torch.tensor([1, 0, 1, 0, 0, 1], dtype=torch.bool)
        bp_yes = torch.rand(L).clamp(0.2, 0.8)
        bp = torch.stack([1 - bp_yes, bp_yes], dim=-1)

        # Padded: B=1, M = #boundaries = 3.
        x_chunked = x_full[bm]  # (3, D)
        out_padded = dl(
            x_chunked.unsqueeze(0),
            bm.unsqueeze(0),
            bp.unsqueeze(0),
            mask=torch.ones(1, L, dtype=torch.bool),
        ).squeeze(0)

        # Packed: cu_seqlens (chunked-space) is [0, 3].
        out_packed = dl(
            x_chunked,
            bm,
            bp,
            cu_seqlens=torch.tensor([0, 3], dtype=torch.long),
        )

        assert torch.allclose(out_padded, out_packed, atol=1e-5, rtol=0)


# =========================================================================== #
# Isotropic
# =========================================================================== #


class TestIsotropic:
    def test_forward_shape(self):
        _seed()
        iso = _make_iso(d_model=32, n_layers=2)
        x = torch.randn(2, 7, 32)
        y = iso(x, mask=torch.ones(2, 7, dtype=torch.bool))
        assert y.shape == (2, 7, 32)

    def test_causal_no_future_leak(self):
        """Mutate a future token; outputs at earlier positions must not change."""
        _seed()
        iso = _make_iso(d_model=16, n_layers=2, num_heads=4, causal=True).eval()
        x = torch.randn(1, 8, 16)
        with torch.no_grad():
            y1 = iso(x, mask=torch.ones(1, 8, dtype=torch.bool))
            x2 = x.clone()
            x2[0, 5] = torch.randn(16) * 10  # perturb position 5
            y2 = iso(x2, mask=torch.ones(1, 8, dtype=torch.bool))
        # Positions 0..4 should be unchanged.
        for t in range(5):
            assert torch.allclose(
                y1[0, t], y2[0, t], atol=1e-5, rtol=0
            ), f"future-leak at t={t}"

    def test_step_matches_forward_bs1(self):
        _seed()
        iso = _make_iso(d_model=16, n_layers=2, num_heads=4, causal=True).eval()
        T = 5
        x = torch.randn(1, T, 16)
        with torch.no_grad():
            y_full = iso(x, mask=torch.ones(1, T, dtype=torch.bool))
            state = iso.allocate_inference_cache(
                1, T, device=x.device, dtype=torch.float32
            )
            y_step = torch.zeros_like(y_full)
            for t in range(T):
                y_step[:, t : t + 1] = iso.step(x[:, t : t + 1], state)
        diff = (y_full - y_step).abs().max().item()
        assert diff < 1e-4, f"step/forward diff {diff:.3e}"

    def test_adaln_noop_when_no_cond(self):
        """With cond_here=False the stack must ignore any cond tensor passed."""
        _seed()
        iso = _make_iso(d_model=16, n_layers=2, d_cond=0, cond_here=False).eval()
        x = torch.randn(2, 4, 16)
        with torch.no_grad():
            y0 = iso(x, mask=torch.ones(2, 4, dtype=torch.bool), cond=None)
            y1 = iso(
                x, mask=torch.ones(2, 4, dtype=torch.bool), cond=torch.randn(2, 4, 16)
            )
        # No cond wiring means cond is silently ignored by the blocks.
        assert torch.equal(y0, y1)


# =========================================================================== #
# Stages
# =========================================================================== #


class TestStages:
    def test_encoder_decoder_shape_no_inner(self):
        _seed()
        stage = EncoderDecoderStage(
            input_hidden_dim=16,
            output_hidden_dim=16,
            encoder=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            decoder=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        ).eval()
        x = torch.randn(2, 4, 16)
        with torch.no_grad():
            y = stage(x, HNetContext())
        assert y.shape == (2, 4, 16)

    def test_chunker_dim_mismatch_creates_projections(self):
        # DeChunkLayer requires d_model % headdim==0; default headdim=32 → use 32/64.
        stage = ChunkerStage(input_hidden_dim=32, output_hidden_dim=64)
        assert stage.proj_in is not None
        assert stage.proj_out is not None
        assert stage.proj_in.in_features == 32
        assert stage.proj_in.out_features == 64
        assert stage.proj_out.in_features == 64
        assert stage.proj_out.out_features == 32

    def test_chunker_no_dim_mismatch_no_projections(self):
        stage = ChunkerStage(input_hidden_dim=32, output_hidden_dim=32)
        assert stage.proj_in is None
        assert stage.proj_out is None

    def test_chunker_residual_proj_zero_init(self):
        stage = ChunkerStage(input_hidden_dim=32, output_hidden_dim=32)
        assert torch.equal(
            stage.residual_proj.weight, torch.zeros_like(stage.residual_proj.weight)
        )
        assert torch.equal(
            stage.residual_proj.bias, torch.zeros_like(stage.residual_proj.bias)
        )

    def test_chunker_registers_aux(self):
        _seed()
        stage = ChunkerStage(
            input_hidden_dim=32,
            output_hidden_dim=32,
            target_compression_ratio=2.5,
            ratio_loss_weight=0.07,
        )
        stage.eval()
        x = torch.randn(2, 6, 32)
        ctx = HNetContext()
        with torch.no_grad():
            _ = stage(x, ctx)
        assert len(ctx.aux) == 1
        entry = ctx.aux[0]
        assert entry["target_ratio"] == 2.5
        assert entry["weight"] == 0.07
        assert entry["bpred"].boundary_mask.shape == (2, 6)

    def test_compute_stage_shape(self):
        _seed()
        st = ComputeStage(
            input_hidden_dim=16,
            output_hidden_dim=16,
            main_network=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        ).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            y = st(x, HNetContext())
        assert y.shape == (2, 5, 16)

    def test_state_slice_scatter_round_trip(self):
        """Allocate state, slice by an all-True mask, scatter back — should equal."""
        _seed()
        stage = ComputeStage(
            input_hidden_dim=16,
            output_hidden_dim=16,
            main_network=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        )
        state = stage._allocate(
            4, max_seqlen=8, dtype=torch.float32, device=torch.device("cpu")
        )
        # Mutate a KV cache slot so we can detect scatter correctness.
        cache = state.main_state.layer_caches[0]
        cache.k.uniform_()
        cache.v.uniform_()
        cache.offsets.fill_(3)
        snapshot_k = cache.k.clone()
        snapshot_v = cache.v.clone()
        snapshot_off = cache.offsets.clone()
        mask = torch.ones(4, dtype=torch.bool)
        sub = _slice_state(state, mask)
        # Zero out original to verify scatter overwrites.
        cache.k.zero_()
        cache.v.zero_()
        cache.offsets.zero_()
        _scatter_state(state, sub, mask)
        assert torch.equal(cache.k, snapshot_k)
        assert torch.equal(cache.v, snapshot_v)
        assert torch.equal(cache.offsets, snapshot_off)


# =========================================================================== #
# HNet end-to-end
# =========================================================================== #


def _build_tiny_hnet(d_model=32, d_inner=32, d_cond=0, with_cond=False) -> HNet:
    return HNet(
        [
            EncoderDecoderStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_model,
                encoder=dict(
                    arch_layout="T1",
                    d_model=d_model,
                    d_intermediate=64,
                    num_heads=4,
                    cond=with_cond,
                ),
                decoder=dict(
                    arch_layout="T1",
                    d_model=d_model,
                    d_intermediate=64,
                    num_heads=4,
                    cond=with_cond,
                ),
                cond_key="fused_cond",
                d_cond=d_cond,
                causal=True,
            ),
            ChunkerStage(
                input_hidden_dim=d_model,
                output_hidden_dim=d_inner,
                target_compression_ratio=2.0,
                ratio_loss_weight=0.05,
            ),
            ComputeStage(
                input_hidden_dim=d_inner,
                output_hidden_dim=d_inner,
                main_network=dict(
                    arch_layout="T1",
                    d_model=d_inner,
                    d_intermediate=64,
                    num_heads=4,
                    cond=False,
                ),
                d_cond=0,
                causal=True,
            ),
        ]
    )


class TestHNetAssembly:
    def test_stage_chain_has_single_canonical_registration(self):
        net = _build_tiny_hnet()

        assert net.stages[0].inner_stage is net.stages[1]
        assert net.stages[1].inner_stage is net.stages[2]
        assert all("inner_stage" not in stage._modules for stage in net.stages)
        assert not any(".inner_stage." in key for key in net.state_dict())

    def test_strict_load_accepts_legacy_inner_stage_aliases(self):
        net = _build_tiny_hnet()
        legacy = net.state_dict()
        for key, value in list(legacy.items()):
            if key.startswith("stages.1."):
                suffix = key.removeprefix("stages.1.")
                legacy[f"stages.0.inner_stage.{suffix}"] = value.clone()
            elif key.startswith("stages.2."):
                suffix = key.removeprefix("stages.2.")
                legacy[f"stages.1.inner_stage.{suffix}"] = value.clone()
                legacy[f"stages.0.inner_stage.inner_stage.{suffix}"] = value.clone()

        restored = _build_tiny_hnet()
        restored.load_state_dict(legacy, strict=True)
        for key, value in net.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key])

    def test_dim_mismatch_raises(self):
        # ChunkerStage output_hidden_dim=24, but inner ComputeStage input_hidden_dim=32.
        with pytest.raises(ValueError, match="Hidden-dim mismatch"):
            HNet(
                [
                    ChunkerStage(input_hidden_dim=32, output_hidden_dim=24),
                    ComputeStage(
                        input_hidden_dim=32,
                        output_hidden_dim=32,
                        main_network=dict(
                            arch_layout="T1",
                            d_model=32,
                            d_intermediate=64,
                            num_heads=4,
                            cond=False,
                        ),
                        d_cond=0,
                        causal=True,
                    ),
                ]
            )

    def test_inner_stage_double_set_raises(self):
        st0 = EncoderDecoderStage(
            input_hidden_dim=16,
            output_hidden_dim=16,
            encoder=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            decoder=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        )
        st1 = ComputeStage(
            input_hidden_dim=16,
            output_hidden_dim=16,
            main_network=dict(
                arch_layout="T1", d_model=16, d_intermediate=32, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        )
        st0.inner_stage = st1  # pre-wire — user error
        with pytest.raises(ValueError, match="already has inner_stage"):
            HNet(
                [
                    st0,
                    ComputeStage(
                        input_hidden_dim=16,
                        output_hidden_dim=16,
                        main_network=dict(
                            arch_layout="T1",
                            d_model=16,
                            d_intermediate=32,
                            num_heads=4,
                            cond=False,
                        ),
                        d_cond=0,
                        causal=True,
                    ),
                ]
            )

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="at least one stage"):
            HNet([])


class TestHNetForward:
    def test_forward_shape(self):
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32).eval()
        x = torch.randn(2, 12, 32)
        with torch.no_grad():
            y = net(x, HNetContext())
        assert y.shape == (2, 12, 32)

    def test_step_matches_forward(self):
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32).eval()
        B, T = 1, 10
        x = torch.randn(B, T, 32)
        with torch.no_grad():
            ctx_full = HNetContext()
            y_full = net(x, ctx_full)
            state = net.allocate_inference_cache(
                B, T, device=x.device, dtype=torch.float32
            )
            y_step = torch.zeros_like(y_full)
            for t in range(T):
                ctx_step = HNetContext(inference_params=state)
                y_step[:, t : t + 1] = net.step(x[:, t : t + 1], ctx_step)
        diff = (y_full - y_step).abs().max().item()
        # The smoke test inherits a known small drift on step/forward at chunker
        # boundaries — we re-use its tolerance.
        assert diff < 5e-4, f"step/forward diff {diff:.3e}"

    def test_backward_runs_and_no_nan_grads(self):
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32)
        net.train()
        x = torch.randn(2, 8, 32, requires_grad=True)
        ctx = HNetContext()
        y = net(x, ctx)
        loss = y.pow(2).mean() + ratio_loss_from_aux(ctx.aux, device=x.device)
        loss.backward()
        # No NaN/Inf in any grad.
        for n, p in net.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_ste_gradient_flows_through_selected_probs(self):
        """The STE is the *only* path for the routing module's q_proj/k_proj to
        receive gradient from downstream loss. Verify those grads are non-zero."""
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32)
        net.train()
        x = torch.randn(2, 8, 32)
        ctx = HNetContext()
        y = net(x, ctx)
        y.sum().backward()  # do NOT include ratio_loss to isolate the STE path
        chunker = net.stages[1]
        gq = chunker.routing_module.q_proj_layer.weight.grad
        gk = chunker.routing_module.k_proj_layer.weight.grad
        assert gq is not None and gq.abs().sum().item() > 0
        assert gk is not None and gk.abs().sum().item() > 0


# =========================================================================== #
# ratio_loss_from_aux
# =========================================================================== #


class TestRatioLoss:
    def test_zero_aux_returns_zero(self):
        loss = ratio_loss_from_aux([], device=torch.device("cpu"))
        assert float(loss) == 0.0

    def test_chunk_stats_basic(self):
        """chunk_stats_from_aux: boundary_rate = #True / total,
        avg_chunk_len = total / max(#True, 1)."""

        class _Bpred:
            def __init__(self, bm):
                self.boundary_mask = bm
                self.boundary_prob = torch.stack([1 - bm.float(), bm.float()], dim=-1)

        # Chunker 0: 2 boundaries out of 8 → F=0.25, avg_len=4.
        bm0 = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
        # Chunker 1: 1 boundary out of 4 → F=0.25, avg_len=4 too.
        bm1 = torch.tensor([1, 0, 0, 0], dtype=torch.bool)
        aux = [
            {"bpred": _Bpred(bm0), "target_ratio": 4.0, "weight": 0.03},
            {"bpred": _Bpred(bm1), "target_ratio": 4.0, "weight": 0.03},
        ]
        stats = chunk_stats_from_aux(aux)
        assert stats["boundary_rate_0"] == 0.25
        assert stats["avg_chunk_len_0"] == 4.0
        assert stats["boundary_rate_1"] == 0.25
        assert stats["avg_chunk_len_1"] == 4.0
        assert stats["boundary_rate"] == 0.25
        assert stats["avg_chunk_len"] == 4.0

    def test_chunk_stats_zero_boundaries_clamps(self):
        """If no boundaries fire, avg_chunk_len falls back to T (one chunk
        covering everything) rather than division-by-zero."""

        class _Bpred:
            def __init__(self, bm):
                self.boundary_mask = bm
                self.boundary_prob = torch.zeros(*bm.shape, 2)

        bm = torch.zeros(10, dtype=torch.bool)
        stats = chunk_stats_from_aux(
            [
                {
                    "bpred": _Bpred(bm),
                    "target_ratio": 4.0,
                    "weight": 0.03,
                }
            ]
        )
        assert stats["boundary_rate_0"] == 0.0
        assert stats["avg_chunk_len_0"] == 10.0  # fallback

    def test_chunk_stats_empty_aux(self):
        assert chunk_stats_from_aux([]) == {}

    def test_optimum_at_F_and_G_equal_inverse_N(self):
        """Closed-form check: with F = G = 1/N, the per-chunker term is
            w * (N/(N-1)) * ((N-1)*F*G + (1-F)*(1-G))
          = w * (N/(N-1)) * ((N-1)*(1/N^2) + ((N-1)/N)^2)
          = w * (N/(N-1)) * ((N-1)/N^2 + (N-1)^2/N^2)
          = w * (N/(N-1)) * (N-1)*(1 + N - 1)/N^2
          = w * (N/(N-1)) * (N-1)*N/N^2
          = w * (N^2 - N) / (N^2 - N)
          = w * 1
        So the loss equals w. (The min of the ratio-loss surface is w; this is
        the documented optimum.)
        """
        N = 4.0
        w = 0.03
        T = 100
        boundary_mask = torch.zeros(T, dtype=torch.float32)
        boundary_mask[:: int(N)] = 1.0
        boundary_prob_yes = torch.full((T,), 1 / N)

        # Wrap into a minimal namedtuple-like.
        class _Bpred:
            def __init__(self, bm, bp):
                self.boundary_mask = bm
                self.boundary_prob = torch.stack([1 - bp, bp], dim=-1)

        aux = [
            {
                "bpred": _Bpred(boundary_mask, boundary_prob_yes),
                "target_ratio": N,
                "weight": w,
            }
        ]
        loss = ratio_loss_from_aux(aux).item()
        assert abs(loss - w) < 1e-4, f"expected {w}, got {loss}"


# =========================================================================== #
# STE behavior
# =========================================================================== #


class TestSTE:
    def test_forward_is_ones(self):
        x = torch.randn(4, 3, requires_grad=True)
        y = _ste(x)
        assert torch.equal(y, torch.ones_like(x))

    def test_backward_is_identity(self):
        x = torch.randn(4, 3, requires_grad=True)
        y = _ste(x)
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))


# =========================================================================== #
# RMSNorm + AdaLN
# =========================================================================== #


class TestNormAndAdaLN:
    def test_rmsnorm_unit_init(self):
        n = RMSNorm(8)
        # Default weight is ones → output preserves RMS scaling.
        x = torch.randn(2, 3, 8) * 5
        y = n(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=5e-3)

    def test_adaln_zero_init_is_identity(self):
        ada = AdaLNModulation(d_cond=4, d_model=8)
        # Weights are zero-initialized; scale=shift=0 → _adaln(x,0,0) = x*1+0 = x.
        x = torch.randn(2, 3, 8)
        s, b = ada(torch.randn(2, 4))
        y = _adaln(x, s, b)
        assert torch.allclose(y, x, atol=1e-6)


# =========================================================================== #
# Packed-mode (cu_seqlens) tests — SDPA fallback path the user will train
# with when flash_attn is not installed. These exercise the block-diagonal
# causal mask in MultiHeadAttention._forward_packed plus the full Isotropic
# stack in packed mode, plus DeChunkLayer cross-subseq EMA reset.
# =========================================================================== #


class TestPackedAttention:
    def _pack(self, x_list):
        """Concat (Li, D) tensors along dim 0, return (T_total, D) and cu_seqlens."""
        cu = torch.tensor([0] + [t.shape[0] for t in x_list], dtype=torch.long).cumsum(
            0
        )
        return torch.cat(x_list, dim=0), cu

    def test_packed_matches_padded_uniform_lengths(self):
        """For B identical-length sequences, packed SDPA must match padded SDPA."""
        _seed()
        d_model, num_heads, B, L = 16, 4, 3, 6
        mha = MultiHeadAttention(d_model, num_heads, causal=True).eval()

        x_padded = torch.randn(B, L, d_model)
        with torch.no_grad():
            y_padded = mha(x_padded, mask=torch.ones(B, L, dtype=torch.bool))

        x_packed = x_padded.reshape(B * L, d_model)
        cu = torch.arange(B + 1, dtype=torch.long) * L
        with torch.no_grad():
            y_packed = mha(x_packed, cu_seqlens=cu, max_seqlen=L)
        y_packed_reshaped = y_packed.reshape(B, L, d_model)

        # SDPA path on both sides — should be bitwise-close.
        diff = (y_padded - y_packed_reshaped).abs().max().item()
        assert diff < 1e-5, f"packed/padded mismatch: {diff:.3e}"

    def test_packed_cross_subseq_isolation(self):
        """Token in subseq B must NOT influence any token in subseq A.
        Perturb the second subseq; first subseq's outputs must be unchanged."""
        _seed()
        d_model, num_heads = 16, 4
        mha = MultiHeadAttention(d_model, num_heads, causal=True).eval()

        x1 = torch.randn(4, d_model)
        x2 = torch.randn(5, d_model)
        x_packed, cu = self._pack([x1, x2])

        with torch.no_grad():
            y = mha(x_packed, cu_seqlens=cu, max_seqlen=5)
            # Perturb only subseq #2.
            x2p = x2 + torch.randn_like(x2) * 10
            x_packed_p, _ = self._pack([x1, x2p])
            yp = mha(x_packed_p, cu_seqlens=cu, max_seqlen=5)

        # First subseq's outputs (rows 0..3) must be identical.
        assert torch.allclose(
            y[:4], yp[:4], atol=1e-6
        ), f"cross-subseq leak: {(y[:4] - yp[:4]).abs().max().item():.3e}"

    def test_packed_causal_within_subseq(self):
        """Within a single packed subseq, perturbing a future token must not
        change earlier outputs."""
        _seed()
        d_model, num_heads = 16, 4
        mha = MultiHeadAttention(d_model, num_heads, causal=True).eval()

        x = torch.randn(8, d_model)
        cu = torch.tensor([0, 8], dtype=torch.long)
        with torch.no_grad():
            y = mha(x, cu_seqlens=cu, max_seqlen=8)
            xp = x.clone()
            xp[5] = torch.randn(d_model) * 10
            yp = mha(xp, cu_seqlens=cu, max_seqlen=8)
        for t in range(5):
            assert torch.allclose(
                y[t], yp[t], atol=1e-6
            ), f"future-leak in packed mode at t={t}"


class TestIsotropicPacked:
    def test_packed_matches_padded_uniform(self):
        """Full Isotropic stack: B uniform-length sequences packed must equal
        padded (B, L, D) call."""
        _seed()
        iso = _make_iso(d_model=32, n_layers=2, num_heads=4, causal=True).eval()
        B, L, D = 2, 7, 32
        x_padded = torch.randn(B, L, D)

        with torch.no_grad():
            y_padded = iso(x_padded, mask=torch.ones(B, L, dtype=torch.bool))
            x_packed = x_padded.reshape(B * L, D)
            cu = torch.arange(B + 1, dtype=torch.long) * L
            y_packed = iso(x_packed, cu_seqlens=cu, max_seqlen=L)

        diff = (y_padded - y_packed.reshape(B, L, D)).abs().max().item()
        # Stacked SDPA accumulates float noise; loosen tolerance.
        assert diff < 5e-5, f"packed/padded Isotropic mismatch: {diff:.3e}"

    def test_packed_cross_subseq_isolation_full_stack(self):
        """End-to-end packed: perturb subseq #2; subseq #1 outputs must be
        unchanged across the whole transformer stack."""
        _seed()
        iso = _make_iso(d_model=32, n_layers=2, num_heads=4, causal=True).eval()
        L1, L2, D = 4, 6, 32

        x1 = torch.randn(L1, D)
        x2 = torch.randn(L2, D)
        x_packed = torch.cat([x1, x2], dim=0)
        cu = torch.tensor([0, L1, L1 + L2], dtype=torch.long)
        max_L = max(L1, L2)

        with torch.no_grad():
            y = iso(x_packed, cu_seqlens=cu, max_seqlen=max_L)
            x2p = x2 + torch.randn_like(x2) * 10
            x_packed_p = torch.cat([x1, x2p], dim=0)
            yp = iso(x_packed_p, cu_seqlens=cu, max_seqlen=max_L)

        assert torch.allclose(y[:L1], yp[:L1], atol=1e-5), (
            f"cross-subseq leak in full Isotropic: "
            f"{(y[:L1] - yp[:L1]).abs().max().item():.3e}"
        )


class TestDeChunkPacked:
    def test_packed_cross_subseq_ema_reset(self):
        """EMA must reset at each subseq boundary. Perturbing chunked values
        in subseq #2 must leave subseq #1's plug-back unchanged."""
        _seed()
        dl = DeChunkLayer(d_model=32, headdim=32)
        # Outer (token) layout: subseq1 has 4 tokens [B,_,B,_], subseq2 has 5 [B,_,B,_,_]
        bm = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.bool)
        bp_yes = torch.rand(9).clamp(0.2, 0.8)
        bp = torch.stack([1 - bp_yes, bp_yes], dim=-1)

        # M_total = 4 chunked tokens (2 per subseq).
        x_chunked = torch.randn(4, 32)
        cu_chunked = torch.tensor([0, 2, 4], dtype=torch.long)

        with torch.no_grad():
            y = dl(x_chunked, bm, bp, cu_seqlens=cu_chunked)
            # Perturb subseq #2's chunked tokens (indices 2..3).
            x_chunked_p = x_chunked.clone()
            x_chunked_p[2:] = x_chunked_p[2:] + torch.randn_like(x_chunked_p[2:]) * 5
            yp = dl(x_chunked_p, bm, bp, cu_seqlens=cu_chunked)

        # Outer rows 0..3 belong to subseq 1; must be unchanged.
        assert torch.allclose(y[:4], yp[:4], atol=1e-5), (
            f"DeChunk EMA leaked across subseq boundary: "
            f"{(y[:4] - yp[:4]).abs().max().item():.3e}"
        )


class TestRoutingPackedMultiSeq:
    def test_packed_padded_equivalence_multi_seq(self):
        """Compare packed boundary_prob[..., 1] against per-subseq padded calls."""
        _seed()
        rm = RoutingModule(d_model=8)
        x1 = torch.randn(5, 8)
        x2 = torch.randn(3, 8)
        x_packed = torch.cat([x1, x2], dim=0)
        cu = torch.tensor([0, 5, 8], dtype=torch.long)

        out_packed = rm(x_packed, cu_seqlens=cu)
        out_1 = rm(x1.unsqueeze(0), mask=torch.ones(1, 5, dtype=torch.bool))
        out_2 = rm(x2.unsqueeze(0), mask=torch.ones(1, 3, dtype=torch.bool))

        assert torch.allclose(
            out_packed.boundary_prob[:5, 1], out_1.boundary_prob[0, :, 1], atol=1e-6
        )
        assert torch.allclose(
            out_packed.boundary_prob[5:, 1], out_2.boundary_prob[0, :, 1], atol=1e-6
        )
        # And subseq starts are boundaries.
        assert out_packed.boundary_mask[0].item() is True
        assert out_packed.boundary_mask[5].item() is True


class TestChunkLayerPacked:
    def test_packed_next_cu_seqlens_multi_subseq(self):
        """Mixed-boundary multi-subseq case: verify next_cu_seqlens math."""
        cl = ChunkLayer()
        # 3 subseqs of lengths 4, 3, 5. Boundaries per subseq: 2, 1, 3.
        bm = torch.tensor(
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
            dtype=torch.bool,
        )
        cu = torch.tensor([0, 4, 7, 12], dtype=torch.long)
        x = torch.arange(12 * 3).reshape(12, 3).float()
        out, next_cu, next_max, _ = cl(x, bm, cu_seqlens=cu)

        # Expected gathered positions: 0,2,  4,  7,9,11.
        assert torch.equal(out, x[[0, 2, 4, 7, 9, 11]])
        # next_cu cumulative counts: 0, 2, 3, 6.
        assert torch.equal(next_cu, torch.tensor([0, 2, 3, 6], dtype=torch.long))
        assert next_max == 3


# =========================================================================== #
# Stage-level packed-mode (cu_seqlens-on-context) tests
#
# These exercise the new ctx.packed path through each stage: the algo sets
# cu_seqlens + max_seqlen on HNetContext and the stages route everything in
# packed (T_total, D) shape. ChunkerStage substitutes the chunked-space
# cu_seqlens / max_seqlen for the inner recursion and restores afterwards.
# =========================================================================== #


class TestStagesPacked:
    def _pack_uniform(self, x_padded):
        B, L, D = x_padded.shape
        x_packed = x_padded.reshape(B * L, D)
        cu = torch.arange(B + 1, dtype=torch.long) * L
        return x_packed, cu, L

    def test_encoder_decoder_packed_padded_equivalence(self):
        _seed()
        stage = EncoderDecoderStage(
            input_hidden_dim=32,
            output_hidden_dim=32,
            encoder=dict(
                arch_layout="T1", d_model=32, d_intermediate=64, num_heads=4, cond=False
            ),
            decoder=dict(
                arch_layout="T1", d_model=32, d_intermediate=64, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        ).eval()
        B, L, D = 2, 6, 32
        x_padded = torch.randn(B, L, D)
        with torch.no_grad():
            y_padded = stage(x_padded, HNetContext())
            x_packed, cu, ms = self._pack_uniform(x_padded)
            ctx_packed = HNetContext(cu_seqlens=cu, max_seqlen=ms)
            y_packed = stage(x_packed, ctx_packed).reshape(B, L, D)
        diff = (y_padded - y_packed).abs().max().item()
        assert diff < 5e-5, f"EncoderDecoderStage packed/padded diff {diff:.3e}"

    def test_compute_stage_packed_padded_equivalence(self):
        _seed()
        st = ComputeStage(
            input_hidden_dim=32,
            output_hidden_dim=32,
            main_network=dict(
                arch_layout="T1", d_model=32, d_intermediate=64, num_heads=4, cond=False
            ),
            d_cond=0,
            causal=True,
        ).eval()
        B, L, D = 2, 5, 32
        x_padded = torch.randn(B, L, D)
        with torch.no_grad():
            y_padded = st(x_padded, HNetContext())
            x_packed, cu, ms = self._pack_uniform(x_padded)
            y_packed = st(x_packed, HNetContext(cu_seqlens=cu, max_seqlen=ms)).reshape(
                B, L, D
            )
        diff = (y_padded - y_packed).abs().max().item()
        assert diff < 5e-5, f"ComputeStage packed/padded diff {diff:.3e}"

    def test_chunker_stage_packed_runs_and_registers_aux(self):
        """ChunkerStage in packed mode: output shape matches input, aux entry
        is registered with the right shape (T_total,) for boundary_mask."""
        _seed()
        stage = ChunkerStage(
            input_hidden_dim=32,
            output_hidden_dim=32,
            target_compression_ratio=2.0,
            ratio_loss_weight=0.05,
        ).eval()
        x1 = torch.randn(5, 32)
        x2 = torch.randn(7, 32)
        x_packed = torch.cat([x1, x2], dim=0)
        cu = torch.tensor([0, 5, 12], dtype=torch.long)
        ctx = HNetContext(cu_seqlens=cu, max_seqlen=7)
        with torch.no_grad():
            y = stage(x_packed, ctx)
        assert y.shape == (12, 32), y.shape
        assert len(ctx.aux) == 1
        bpred = ctx.aux[0]["bpred"]
        assert bpred.boundary_mask.shape == (12,)
        # Sub-seq starts must still be forced boundaries.
        assert bpred.boundary_mask[0].item() is True
        assert bpred.boundary_mask[5].item() is True

    def test_chunker_restores_ctx_after_inner_call(self):
        """After ChunkerStage runs, ctx.cu_seqlens / ctx.max_seqlen must be
        restored to their original outer values (the chunked-space cu_seqlens
        is only visible inside the inner_stage recursion)."""
        _seed()
        net = HNet(
            [
                ChunkerStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    target_compression_ratio=2.0,
                    ratio_loss_weight=0.05,
                ),
                ComputeStage(
                    input_hidden_dim=32,
                    output_hidden_dim=32,
                    main_network=dict(
                        arch_layout="T1",
                        d_model=32,
                        d_intermediate=64,
                        num_heads=4,
                        cond=False,
                    ),
                    d_cond=0,
                    causal=True,
                ),
            ]
        ).eval()
        x = torch.randn(10, 32)
        cu = torch.tensor([0, 4, 10], dtype=torch.long)
        ctx = HNetContext(cu_seqlens=cu, max_seqlen=6)
        with torch.no_grad():
            _ = net(x, ctx)
        assert torch.equal(ctx.cu_seqlens, cu), "outer cu_seqlens leaked"
        assert ctx.max_seqlen == 6, "outer max_seqlen leaked"

    def test_hnet_packed_padded_equivalence_full_tree(self):
        """Full 3-stage HNet (EncoderDecoder → Chunker → Compute) end-to-end:
        packed mode with uniform-length subseqs must match padded mode."""
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32).eval()
        B, L, D = 2, 8, 32
        x_padded = torch.randn(B, L, D)
        with torch.no_grad():
            y_padded = net(x_padded, HNetContext())
            x_packed, cu, ms = TestStagesPacked()._pack_uniform(x_padded)
            y_packed = net(x_packed, HNetContext(cu_seqlens=cu, max_seqlen=ms)).reshape(
                B, L, D
            )
        diff = (y_padded - y_packed).abs().max().item()
        # End-to-end accumulated float noise: chunker introduces a few extra
        # gathers and an EMA; keep a looser tolerance than single-module tests.
        assert diff < 5e-4, f"HNet packed/padded diff {diff:.3e}"

    def test_hnet_packed_cross_subseq_isolation(self):
        """Through the full HNet stage tree, perturbing tokens in subseq #2
        must NOT change subseq #1's outputs (no leakage across the pack)."""
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32).eval()
        L1, L2, D = 5, 7, 32
        x1 = torch.randn(L1, D)
        x2 = torch.randn(L2, D)
        x_packed = torch.cat([x1, x2], dim=0)
        cu = torch.tensor([0, L1, L1 + L2], dtype=torch.long)
        ms = max(L1, L2)
        with torch.no_grad():
            y = net(x_packed, HNetContext(cu_seqlens=cu, max_seqlen=ms))
            x2p = x2 + torch.randn_like(x2) * 10
            x_packed_p = torch.cat([x1, x2p], dim=0)
            yp = net(x_packed_p, HNetContext(cu_seqlens=cu, max_seqlen=ms))
        diff = (y[:L1] - yp[:L1]).abs().max().item()
        # Routing module's cosine output is bounded, dechunked-residual gating
        # via STE plus residual_proj=0 at init keeps the path linear; the EMA
        # is causal+per-subseq. Cross-subseq leakage should be exactly 0 up to
        # float jitter.
        assert diff < 5e-5, f"cross-subseq leak through HNet: {diff:.3e}"

    def test_hnet_packed_backward_runs(self):
        """Packed-mode backward through the full HNet stage tree must produce
        finite grads on every parameter that received signal."""
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32)
        net.train()
        x = torch.randn(15, 32, requires_grad=True)
        cu = torch.tensor([0, 6, 15], dtype=torch.long)
        ctx = HNetContext(cu_seqlens=cu, max_seqlen=9)
        y = net(x, ctx)
        loss = y.pow(2).mean() + ratio_loss_from_aux(ctx.aux, device=x.device)
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_hnet_packed_ste_gradient_flows_to_routing(self):
        """In packed mode the STE must still deliver gradient to the chunker's
        routing module q/k projections."""
        _seed()
        net = _build_tiny_hnet(d_model=32, d_inner=32)
        net.train()
        x = torch.randn(15, 32)
        cu = torch.tensor([0, 6, 15], dtype=torch.long)
        ctx = HNetContext(cu_seqlens=cu, max_seqlen=9)
        y = net(x, ctx)
        y.sum().backward()
        chunker = net.stages[1]
        gq = chunker.routing_module.q_proj_layer.weight.grad
        gk = chunker.routing_module.k_proj_layer.weight.grad
        assert gq is not None and gq.abs().sum().item() > 0
        assert gk is not None and gk.abs().sum().item() > 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
