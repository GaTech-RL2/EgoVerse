"""Unit test for the swappable action head (continuous + discrete).
Verifies: raw shapes, loss is finite scalar, decode stays in-range, and a
bin round-trip (argmax decode of a binned-then-onehot logit recovers the
nearest bin center). CPU-only, tiny."""
import torch
import torch.nn as nn
from egomimic.models.hnet_nets.action_heads import (
    configure_action_head, action_head_raw, action_head_loss,
    action_head_decode, _to_bins,
)


class Dummy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        configure_action_head(self, d_model=16, action_dim=2, cfg=cfg)


def test_continuous():
    m = Dummy({"mode": "continuous"})
    h = torch.randn(4, 7, 16)
    raw = action_head_raw(m, h)
    assert raw.shape == (4, 7, 2), raw.shape
    tgt = torch.randn(4, 7, 2)
    loss = action_head_loss(m, raw, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    dec = action_head_decode(m, raw)
    assert torch.allclose(dec, raw)  # identity
    print("continuous OK", float(loss))


def test_discrete_shapes():
    m = Dummy({"mode": "discrete", "n_bins": 256, "bin_min": -4, "bin_max": 4})
    h = torch.randn(4, 7, 16)
    raw = action_head_raw(m, h)
    assert raw.shape == (4, 7, 2, 256), raw.shape
    tgt = torch.randn(4, 7, 2).clamp(-3, 3)
    loss = action_head_loss(m, raw, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    dec = action_head_decode(m, raw)
    assert dec.shape == (4, 7, 2), dec.shape
    assert dec.min() >= -4.05 and dec.max() <= 4.05, (dec.min(), dec.max())
    print("discrete shapes OK", float(loss))


def test_bin_roundtrip():
    # Build logits that are one-hot at the bin of a known target; argmax
    # decode (sample_temp<=0) must return that bin's center, within one bin
    # width of the original value.
    m = Dummy({"mode": "discrete", "n_bins": 256, "bin_min": -4, "bin_max": 4,
               "sample_temp": 0.0})
    vals = torch.tensor([[-3.1, 0.0], [1.7, 3.9]])  # (2,2) in-range
    idx = _to_bins(m, vals)                          # (2,2) long
    logits = torch.full((2, 2, 256), -10.0)
    logits.scatter_(-1, idx.unsqueeze(-1), 10.0)     # one-hot at idx
    dec = action_head_decode(m, logits)
    bin_w = 8.0 / 256
    err = (dec - vals).abs().max().item()
    assert err <= bin_w, (err, bin_w, dec, vals)
    print("bin roundtrip OK  max_err=%.4f  bin_width=%.4f" % (err, bin_w))


if __name__ == "__main__":
    test_continuous()
    test_discrete_shapes()
    test_bin_roundtrip()
    print("ALL TESTS PASSED")
