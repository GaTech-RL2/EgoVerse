"""Unit test for the GMM action head (+ regression/discrete unchanged check).

Verifies, CPU-only and tiny:
  * configure -> head width = M*(2D+1) (x K for chunk),
  * raw shapes for chunk_k==1 and chunk_k>1,
  * NLL loss is a finite scalar AND a few SGD steps drive it DOWN,
  * decode commits to the argmax-logit mode's mean (NOT the weighted mean),
    is in-range (finite, sensible) for both K==1 and K>1,
  * sample-mode decode is finite,
  * the analytic GMM log-prob matches a hand single-Gaussian special case,
  * continuous + discrete heads are byte-/value-identical to before (the GMM
    code path must not touch them).
"""
import math

import torch
import torch.nn as nn

from egomimic.models.hnet_nets.action_heads import (
    configure_action_head,
    action_head_raw,
    action_head_loss,
    action_head_decode,
    action_head_decode_chunk,
    _gmm_split,
    _gmm_log_prob,
)


class Dummy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        configure_action_head(self, d_model=16, action_dim=2, cfg=cfg)


def test_gmm_width_and_shapes():
    M, D, K = 5, 2, 1
    m = Dummy({"mode": "gmm", "gmm_num_modes": M})
    assert m.action_out.out_features == M * (2 * D + 1) * K
    h = torch.randn(4, 7, 16)
    raw = action_head_raw(m, h)
    assert raw.shape == (4, 7, M * (2 * D + 1)), raw.shape
    means, log_stds, logits = _gmm_split(m, raw)
    assert means.shape == (4, 7, M, D)
    assert log_stds.shape == (4, 7, M, D)
    assert logits.shape == (4, 7, M)
    # log-stds were clamped into the configured range.
    assert log_stds.min() >= m._act_gmm_logstd_min - 1e-6
    assert log_stds.max() <= m._act_gmm_logstd_max + 1e-6
    print("gmm width+shapes OK")


def test_gmm_chunk_shapes():
    M, D, K = 4, 2, 8
    m = Dummy({"mode": "gmm", "gmm_num_modes": M, "chunk_k": K})
    assert m.action_out.out_features == M * (2 * D + 1) * K
    h = torch.randn(3, 16)  # (B, d_model)  -> chunk head over flat batch
    raw = action_head_raw(m, h)
    assert raw.shape == (3, K, M * (2 * D + 1)), raw.shape
    # loss with windowed chunk targets (padded path: actions (B,T,D)).
    actions = torch.randn(3, 5, D)
    # raw needs a T axis to align with _build_chunk_targets (B,T,...).
    h2 = torch.randn(3, 5, 16)
    raw2 = action_head_raw(m, h2)  # (3,5,K,M*(2D+1))
    loss = action_head_loss(m, raw2, actions)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    # full-chunk decode -> (B, T, K, D), per-step committed.
    dec = action_head_decode_chunk(m, raw2)
    assert dec.shape == (3, 5, K, D), dec.shape
    assert torch.isfinite(dec).all()
    # k=0 single-step decode -> (B, T, D).
    dec0 = action_head_decode(m, raw2)
    assert dec0.shape == (3, 5, D), dec0.shape
    assert torch.allclose(dec0, dec[..., 0, :])  # k=0 == executed step
    print("gmm chunk shapes OK", float(loss))


def test_gmm_decode_commits_to_argmax_mode():
    # Hand-build raw params: 2 modes, mode 1 has the higher logit. Decode must
    # return mode-1's mean exactly (NOT a weighted average of the two means).
    M, D = 2, 2
    m = Dummy({"mode": "gmm", "gmm_num_modes": M})
    means = torch.tensor([[[0.0, 0.0], [3.0, -2.0]]])  # (1, M, D)
    log_stds = torch.zeros(1, M, D)
    logits = torch.tensor([[0.1, 5.0]])  # mode 1 dominates
    raw = torch.cat(
        [means.reshape(1, -1), log_stds.reshape(1, -1), logits], dim=-1
    )  # (1, M*(2D+1))
    dec = action_head_decode(m, raw)
    assert torch.allclose(dec, torch.tensor([[3.0, -2.0]])), dec
    # explicitly assert it is NOT the weight-weighted mean.
    w = torch.softmax(logits, dim=-1)  # (1,M)
    weighted = (w.unsqueeze(-1) * means).sum(-2)  # (1,D)
    assert not torch.allclose(dec, weighted)
    print("gmm decode commits to argmax mode OK", dec.tolist())


def test_gmm_logprob_matches_single_gaussian():
    # With M=1 the mixture log-prob must equal a plain diagonal Gaussian.
    M, D = 1, 2
    m = Dummy({"mode": "gmm", "gmm_num_modes": M})
    mean = torch.tensor([[[0.5, -0.3]]])  # (1,M,D)
    log_std = torch.tensor([[[0.0, math.log(2.0)]]])  # sigma = [1, 2]
    logits = torch.zeros(1, M)
    a = torch.tensor([[0.7, 0.1]])  # (1,D)
    lp = _gmm_log_prob(m, mean, log_std, logits, a)  # (1,)
    sigma = torch.exp(log_std).squeeze(1)  # (1,D)
    manual = (
        -0.5 * ((a - mean.squeeze(1)) / sigma) ** 2
        - log_std.squeeze(1)
        - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    assert torch.allclose(lp, manual, atol=1e-6), (lp, manual)
    print("gmm log-prob == single-gaussian OK", float(lp))


def test_gmm_nll_decreases():
    # Fit the head on a fixed (h, a) pair; NLL must drop with SGD.
    torch.manual_seed(0)
    m = Dummy({"mode": "gmm", "gmm_num_modes": 5})
    h = torch.randn(32, 16)
    a = torch.randn(32, 2) * 0.5
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = last = None
    for i in range(200):
        opt.zero_grad()
        raw = action_head_raw(m, h)
        loss = action_head_loss(m, raw, a)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), (i, loss)
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first, (first, last)
    print("gmm NLL decreased  %.4f -> %.4f" % (first, last))
    # decode is in-range / finite after fitting.
    dec = action_head_decode(m, action_head_raw(m, h))
    assert dec.shape == (32, 2) and torch.isfinite(dec).all()
    print("gmm decode finite after fit OK")


def test_gmm_sample_mode_finite():
    torch.manual_seed(1)
    m = Dummy({"mode": "gmm", "gmm_num_modes": 4, "gmm_sample": True})
    raw = action_head_raw(m, torch.randn(6, 3, 16))
    dec = action_head_decode(m, raw)
    assert dec.shape == (6, 3, 2) and torch.isfinite(dec).all()
    print("gmm sample-mode decode finite OK")


# ---- regression / discrete unchanged guards ----

def test_continuous_unchanged():
    m = Dummy({"mode": "continuous"})
    assert not getattr(m, "_act_gmm", False)
    h = torch.randn(4, 7, 16)
    raw = action_head_raw(m, h)
    assert raw.shape == (4, 7, 2), raw.shape
    tgt = torch.randn(4, 7, 2)
    loss = action_head_loss(m, raw, tgt)
    assert torch.allclose(loss, nn.functional.mse_loss(raw, tgt))
    dec = action_head_decode(m, raw)
    assert torch.allclose(dec, raw)  # identity
    print("continuous unchanged OK", float(loss))


def test_discrete_unchanged():
    m = Dummy({"mode": "discrete", "n_bins": 256, "bin_min": -4, "bin_max": 4})
    assert not getattr(m, "_act_gmm", False)
    h = torch.randn(4, 7, 16)
    raw = action_head_raw(m, h)
    assert raw.shape == (4, 7, 2, 256), raw.shape
    tgt = torch.randn(4, 7, 2).clamp(-3, 3)
    loss = action_head_loss(m, raw, tgt)
    assert torch.isfinite(loss)
    dec = action_head_decode(m, raw)
    assert dec.shape == (4, 7, 2) and dec.min() >= -4.05 and dec.max() <= 4.05
    print("discrete unchanged OK", float(loss))


def test_bad_mode_errors():
    try:
        Dummy({"mode": "banana"})
    except ValueError:
        print("bad-mode error OK")
        return
    raise AssertionError("expected ValueError for unknown mode")


if __name__ == "__main__":
    test_gmm_width_and_shapes()
    test_gmm_chunk_shapes()
    test_gmm_decode_commits_to_argmax_mode()
    test_gmm_logprob_matches_single_gaussian()
    test_gmm_nll_decreases()
    test_gmm_sample_mode_finite()
    test_continuous_unchanged()
    test_discrete_unchanged()
    test_bad_mode_errors()
    print("ALL GMM TESTS PASSED")
