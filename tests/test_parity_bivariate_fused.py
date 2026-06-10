"""
Unit tests for ``DiscreteDiffusion.compute_bivariate_fused_weights`` — the
dormant bivariate fused-SNR strategy for decoupled obs/action noise levels
(designed 2026-06-09; intended caller is the decoupled two-stream outer
stage once parity training lands).

Limit cases covered:

  1. All-noise history + all-noise cross terms reduce each stream to the
     plain per-stream min-SNR weights (S̄_h → 0 ⇒ fused = own S).
  2. Clean current obs, "product" mode, kappa=1: action weights become
     CONSTANT in k_act (the documented clean-obs degeneracy → uniform).
  3. Clean current obs, "blend" mode, mu<1: action weights still vary with
     k_act (mu budgets the total cross-signal — the saturation fix).
  4. lambda_blend=1 ignores the action history entirely.
  5. Direction follows ``use_causal_mask``: a clean FUTURE frame raises a
     noisy frame's weight only in the bidirectional (non-causal) core.
  6. Shapes/guards: (B,T) in → (B,T) out for both streams; 1-D k raises;
     scalar-path requests of 'bivariate_fused' raise with a pointer.

All tests are pure-torch on CPU, no data dependencies.
"""

from __future__ import annotations

import pytest
import torch

from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion

ACTION_DIM = 3
T = 6
K_MAX = 999  # timesteps=1000


def _make(objective="pred_v", causal=False, **kw) -> DiscreteDiffusion:
    return DiscreteDiffusion(
        action_dim=ACTION_DIM,
        objective=objective,
        loss_weighting_strategy="min_snr",
        use_causal_mask=causal,
        **kw,
    )


def _full(level: int) -> torch.Tensor:
    return torch.full((1, T), level, dtype=torch.long)


class TestLimitCases:
    def test_all_noise_history_reduces_to_per_stream_min_snr(self):
        diff = _make()
        # Context must be signal-free for BOTH streams: terminal SNR is
        # exactly 0 (cosine zero-terminal), so every frame except the probed
        # one sits at K_MAX, and the shifted EMA excludes the probe itself.
        j = 3
        k_obs = _full(K_MAX)
        for k_act_level in (0, 250, 900):
            k_act = _full(K_MAX)
            k_act[0, j] = k_act_level
            _, w_act = diff.compute_bivariate_fused_weights(
                k_obs, k_act, fusion_mode="product", kappa=1.0
            )
            ref = diff.compute_loss_weights(
                torch.tensor([[k_act_level]], dtype=torch.long), strategy="min_snr"
            )
            assert torch.allclose(w_act[:, j], ref[:, 0], atol=1e-5), (
                f"k_act={k_act_level}: bivariate with zero context should "
                "match plain min_snr at the probed frame"
            )

    def test_clean_obs_product_kappa1_action_weights_uniform(self):
        diff = _make()
        k_obs = _full(0)  # clean current obs at every frame
        w = []
        for k_act_level in (1, 250, 500, 990):
            _, w_act = diff.compute_bivariate_fused_weights(
                k_obs, _full(k_act_level), fusion_mode="product", kappa=1.0
            )
            w.append(w_act)
        # S_o = 1 (snr at k=0 far above clip) ⇒ (1 - κ·S_o) = 0 ⇒ fused = 1
        # regardless of k_act ⇒ weights constant across action levels.
        for other in w[1:]:
            assert torch.allclose(w[0], other, atol=1e-5)

    def test_clean_obs_blend_mu_half_action_weights_vary(self):
        diff = _make()
        k_obs = _full(0)
        _, w_lo = diff.compute_bivariate_fused_weights(
            k_obs, _full(1), fusion_mode="blend", mu=0.5
        )
        _, w_hi = diff.compute_bivariate_fused_weights(
            k_obs, _full(990), fusion_mode="blend", mu=0.5
        )
        assert not torch.allclose(w_lo, w_hi, atol=1e-4), (
            "blend mode with mu<1 must keep the action's own level "
            "modulating its weight under clean obs"
        )

    def test_lambda_one_ignores_action_history(self):
        diff = _make()
        k_obs = _full(500)
        k_act_clean_hist = torch.tensor([[0, 0, 0, 0, 0, 700]], dtype=torch.long)
        k_act_noisy_hist = torch.tensor(
            [[K_MAX, K_MAX, K_MAX, K_MAX, K_MAX, 700]], dtype=torch.long
        )
        out = [
            diff.compute_bivariate_fused_weights(
                k_obs, ka, fusion_mode="blend", lambda_blend=1.0, mu=0.0
            )
            for ka in (k_act_clean_hist, k_act_noisy_hist)
        ]
        # With lambda=1 the blended history uses only the obs stream, and
        # mu=0 removes the current-obs term: identical weights at the final
        # frame regardless of the action history.
        assert torch.allclose(out[0][1][:, -1], out[1][1][:, -1], atol=1e-6)

    def test_direction_follows_use_causal_mask(self):
        # Frame 2 fully noised; only FUTURE obs frames (3..5) are clean.
        k_obs = torch.tensor([[K_MAX, K_MAX, K_MAX, 0, 0, 0]], dtype=torch.long)
        k_act = _full(K_MAX)
        w_causal = _make(causal=True).compute_bivariate_fused_weights(
            k_obs, k_act, fusion_mode="blend", mu=0.0
        )[0][:, 2]
        w_bidir = _make(causal=False).compute_bivariate_fused_weights(
            k_obs, k_act, fusion_mode="blend", mu=0.0
        )[0][:, 2]
        # pred_v weight is increasing in fused SNR below the clip: the
        # bidirectional core must price in the clean future, the causal one
        # must not.
        assert (w_bidir > w_causal + 1e-6).all()


class TestShapesAndGuards:
    def test_output_shapes(self):
        diff = _make()
        k = torch.randint(0, K_MAX, (3, 7))
        for mode in ("blend", "product"):
            w_obs, w_act = diff.compute_bivariate_fused_weights(
                k, k.flip(1), fusion_mode=mode
            )
            assert w_obs.shape == (3, 7) and w_act.shape == (3, 7)
            assert torch.isfinite(w_obs).all() and torch.isfinite(w_act).all()

    def test_1d_k_raises(self):
        diff = _make()
        with pytest.raises(ValueError, match=r"\(B, T\)"):
            diff.compute_bivariate_fused_weights(
                torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
            )

    def test_unknown_fusion_mode_raises(self):
        diff = _make()
        with pytest.raises(ValueError, match="fusion_mode"):
            diff.compute_bivariate_fused_weights(
                _full(0), _full(0), fusion_mode="nope"
            )

    def test_scalar_path_request_raises_with_pointer(self):
        diff = _make()
        with pytest.raises(ValueError, match="compute_bivariate_fused_weights"):
            diff.compute_loss_weights(_full(0), strategy="bivariate_fused")
