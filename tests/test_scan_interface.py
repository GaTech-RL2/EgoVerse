"""Unit tests for the scan-based chunk interface (module-first gate before
any integration into ChunkerStage — see vault 'Expressive Chunk
Downsampler-Upsampler', Implementation plan)."""

import pytest
import torch

from egomimic.models.hnet.scan_interface import (
    ChunkScanEncoder,
    ScanRouter,
    SplineUpsampler,
)

torch.manual_seed(0)


def make_batch(T1=11, T2=7, D=16):
    cu = torch.tensor([0, T1, T1 + T2])
    x = torch.randn(cu[-1], D)
    # boundaries: subseq starts forced + a few interior ones
    bm = torch.zeros(cu[-1], dtype=torch.bool)
    bm[cu[:-1]] = True
    bm[3] = bm[8] = bm[T1 + 3] = True
    return x, bm, cu


class TestChunkScanEncoder:
    def setup_method(self):
        self.enc = ChunkScanEncoder(d_model=16, d_out=24, m_sublatents=4)

    def test_shapes(self):
        x, bm, cu = make_batch()
        h, sub, summ, ncu, nmax = self.enc(x, bm, cu)
        N = int(bm.sum())
        assert h.shape == (x.shape[0], 24)
        assert sub.shape == (N, 4, 24)
        assert summ.shape == (N, 24)
        assert ncu.tolist()[0] == 0 and ncu.tolist()[-1] == N

    def test_boundary_reset_isolation(self):
        """Perturbing a token in chunk j must NOT change other chunks'
        sub-latents (state reset at boundaries)."""
        x, bm, cu = make_batch()
        _, sub0, _, _, _ = self.enc(x, bm, cu)
        seg = bm.cumsum(0) - 1
        x2 = x.clone()
        x2[4] += 10.0  # token 4 lives in chunk seg[4]
        _, sub1, _, _, _ = self.enc(x2, bm, cu)
        j = int(seg[4])
        others = [k for k in range(sub0.shape[0]) if k != j]
        assert torch.allclose(sub0[others], sub1[others], atol=1e-6)
        assert not torch.allclose(sub0[j], sub1[j])

    def test_order_sensitivity(self):
        """Permuting tokens WITHIN a chunk must change its summary (the
        anti-pooling property)."""
        x, bm, cu = make_batch()
        _, _, summ0, _, _ = self.enc(x, bm, cu)
        x2 = x.clone()
        x2[4], x2[5] = x[5].clone(), x[4].clone()  # both in chunk of token 4
        seg = bm.cumsum(0) - 1
        assert seg[4] == seg[5], "test setup: tokens must share a chunk"
        _, _, summ1, _, _ = self.enc(x2, bm, cu)
        assert not torch.allclose(summ0[int(seg[4])], summ1[int(seg[4])])

    def test_gradient_reaches_every_token(self):
        x, bm, cu = make_batch()
        x = x.requires_grad_(True)
        _, sub, _, _, _ = self.enc(x, bm, cu)
        sub.sum().backward()
        assert (x.grad.abs().sum(dim=-1) > 0).all(), "some token got no grad"

    def test_len1_chunks(self):
        x = torch.randn(3, 16)
        bm = torch.ones(3, dtype=torch.bool)
        cu = torch.tensor([0, 3])
        h, sub, summ, ncu, nmax = self.enc(x, bm, cu)
        assert summ.shape == (3, 24) and nmax == 3


class TestScanRouter:
    def setup_method(self):
        self.router = ScanRouter(d_model=16, d_state=24)

    def test_contract(self):
        x, _, cu = make_batch()
        out = self.router(x, cu_seqlens=cu)
        T = x.shape[0]
        assert out.boundary_prob.shape == (T, 2)
        assert out.boundary_mask.shape == (T,)
        assert out.selected_probs.shape == (T, 1)
        assert out.boundary_mask[cu[:-1]].all(), "subseq starts must be boundaries"
        assert torch.allclose(out.boundary_prob.sum(-1), torch.ones(T))

    def test_prefix_causality(self):
        x, _, cu = make_batch()
        p0 = self.router(x, cu_seqlens=cu).boundary_prob[..., 1]
        x2 = x.clone()
        x2[6:11] += 5.0  # only future of position 5 within subseq 0
        p1 = self.router(x2, cu_seqlens=cu).boundary_prob[..., 1]
        assert torch.allclose(p0[:6], p1[:6], atol=1e-6)
        assert not torch.allclose(p0[6:11], p1[6:11])

    def test_grad_flows(self):
        x, _, cu = make_batch()
        x = x.requires_grad_(True)
        out = self.router(x, cu_seqlens=cu)
        out.boundary_prob[..., 1][1:].sum().backward()  # skip forced start
        assert x.grad is not None and x.grad.abs().sum() > 0


class TestSplineUpsampler:
    def setup_method(self):
        self.enc = ChunkScanEncoder(d_model=16, d_out=24, m_sublatents=4)
        self.up = SplineUpsampler(
            d_inner=24, d_state=24, d_model=16, m_sublatents=4
        )

    def _run(self, x, bm, cu):
        h, sub, _, _, _ = self.enc(x, bm, cu)
        p = torch.full((x.shape[0],), 0.7)
        bp = torch.stack([1 - p, p], dim=-1)
        return self.up(sub, h, x, bm, bp, cu)

    def test_anti_staircase(self):
        """Within-chunk outputs must VARY (the prior cross-attn upsampler
        failure: near-identical values across a chunk)."""
        x, bm, cu = make_batch()
        out = self._run(x, bm, cu)
        seg = bm.cumsum(0) - 1
        # chunk containing tokens 3..7 (boundary at 3, next at 8)
        chunk = out[3:8]
        assert seg[3] == seg[7]
        var = chunk.var(dim=0).mean()
        assert var > 1e-6, f"within-chunk variance collapsed: {var}"

    def test_causality_within_chunk(self):
        """Perturbing FUTURE tokens within a chunk must not change earlier
        positions' outputs (h_t prefix + prev-chunk basis only)."""
        x, bm, cu = make_batch()
        out0 = self._run(x, bm, cu)
        x2 = x.clone()
        x2[6] += 10.0  # token 6 is mid-chunk (chunk 3..7)
        out1 = self._run(x2, bm, cu)
        assert torch.allclose(out0[3:6], out1[3:6], atol=1e-5)
        assert not torch.allclose(out0[6:8], out1[6:8])

    def test_first_chunk_uses_no_basis(self):
        """First chunk of each subseq has no previous chunk: its outputs
        must be invariant to OTHER chunks' content entirely."""
        x, bm, cu = make_batch()
        out0 = self._run(x, bm, cu)
        x2 = x.clone()
        x2[9] += 10.0  # token in a LATER chunk of subseq 0
        out1 = self._run(x2, bm, cu)
        assert torch.allclose(out0[0:3], out1[0:3], atol=1e-5)

    def test_confidence_scales_output(self):
        x, bm, cu = make_batch()
        h, sub, _, _, _ = self.enc(x, bm, cu)
        lo = torch.full((x.shape[0],), 0.6)
        hi = torch.full((x.shape[0],), 0.9)
        out_lo = self.up(sub, h, x, bm, torch.stack([1 - lo, lo], -1), cu)
        out_hi = self.up(sub, h, x, bm, torch.stack([1 - hi, hi], -1), cu)
        # At boundary tokens c = p, so out_hi/out_lo = 0.9/0.6 elementwise.
        # Cross-multiply instead of dividing: outputs legitimately cross zero
        # (FiLM beta), so an elementwise ratio is numerically invalid there.
        assert torch.allclose(
            out_hi[bm] * 0.6, out_lo[bm] * 0.9, atol=1e-5
        )


class TestIntegrationMini:
    def test_end_to_end_forward_backward(self):
        x, _, cu = make_batch()
        x = x.requires_grad_(True)
        router = ScanRouter(16, 24)
        enc = ChunkScanEncoder(16, 24, 4)
        up = SplineUpsampler(24, 24, 16, 4)
        r = router(x, cu_seqlens=cu)
        h, sub, summ, ncu, nmax = enc(x, r.boundary_mask, cu)
        out = up(sub, h, x, r.boundary_mask, r.boundary_prob, cu)
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for mod in (router, enc, up):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in mod.parameters()
            ), f"{type(mod).__name__} got no gradient"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestScanRouterMambaPath:
    """Same contract + causality checks, on the Mamba backend (CUDA)."""

    def test_contract_and_causality_cuda(self):
        dev = torch.device("cuda")
        router = ScanRouter(d_model=64, backend="auto").to(dev)
        assert router._mamba is not None, "mamba backend unavailable"
        T1, T2 = 33, 21
        cu = torch.tensor([0, T1, T1 + T2], device=dev)
        x = torch.randn(cu[-1], 64, device=dev)
        out = router(x, cu_seqlens=cu)
        T = x.shape[0]
        assert out.boundary_prob.shape == (T, 2)
        assert out.boundary_mask[cu[:-1]].all()
        # prefix causality on the mamba scan
        x2 = x.clone()
        x2[20:T1] += 5.0
        p0 = router(x, cu_seqlens=cu).boundary_prob[..., 1]
        p1 = router(x2, cu_seqlens=cu).boundary_prob[..., 1]
        assert torch.allclose(p0[:20], p1[:20], atol=1e-4)
        # subseq isolation: subseq 1 untouched by subseq-0 perturbation
        assert torch.allclose(p0[T1:], p1[T1:], atol=1e-4)

    def test_grad_flows_cuda(self):
        dev = torch.device("cuda")
        router = ScanRouter(d_model=64, backend="mamba").to(dev)
        cu = torch.tensor([0, 17], device=dev)
        x = torch.randn(17, 64, device=dev, requires_grad=True)
        out = router(x, cu_seqlens=cu)
        out.boundary_prob[..., 1][1:].sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
