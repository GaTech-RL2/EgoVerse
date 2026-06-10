"""
Unit tests for the DFoT parity-port loss-weighting changes in the two
diffusion cores (Port Parity Checklist D04/D15/D16 seams).

Covers:

  1. Per-call weighting override on ``compute_loss_weights`` /
     ``compute_loss`` for both ``DiscreteDiffusion`` and
     ``ContinuousDiffusion`` — every (ctor_strategy, override) pair matches
     manually computed weights, and ``override=None`` is bit-identical to
     the ctor-strategy path.
  2. New 'constant' strategy returns ``constant_weight`` at every noise
     level with NO objective-dependent factor (upstream
     ``diffusion_transition.py`` constant-weight parity).
  3. ``max_beta`` clamp (parity 0.999) changes only the terminal beta /
     derived terminal alphas; ``max_beta=None`` keeps the schedule
     untouched (including the ``sqrt_recip_alphas_cumprod[-1]=inf``
     landmine, D15).
  4. ``clipped_snr`` / ``logsnr`` buffers exist regardless of ctor strategy
     and stay out of ``state_dict`` (persistent=False).
  5. 'fused_min_snr' with 1-D ``k`` warns once per process about the silent
     min_snr fallback (the D04 config lie); requesting 'fused_min_snr' via
     the per-call override raises on both cores.

All tests are pure-torch on CPU, no data dependencies.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import egomimic.models.diffusion.diffusion.discrete_diffusion as discrete_diffusion_mod
from egomimic.models.diffusion.diffusion.continuous_diffusion import (
    ContinuousDiffusion,
)
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion

ACTION_DIM = 3
CONSTANT_WEIGHT = 2.5
DISCRETE_STRATEGIES = ["uniform", "min_snr", "sigmoid", "constant"]
CONTINUOUS_STRATEGIES = ["uniform", "sigmoid", "constant"]

# 2-D per-token k covering low/mid/high noise plus the terminal level.
K_2D = torch.tensor([[0, 1, 250, 500], [900, 990, 998, 999]], dtype=torch.long)


def _make_discrete(strategy: str, **kwargs) -> DiscreteDiffusion:
    return DiscreteDiffusion(
        action_dim=ACTION_DIM,
        loss_weighting_strategy=strategy,
        constant_weight=CONSTANT_WEIGHT,
        **kwargs,
    )


def _manual_discrete_weights(
    core: DiscreteDiffusion, k: torch.Tensor, strategy: str
) -> torch.Tensor:
    """Independent re-derivation of the per-token weights from the buffers."""
    if strategy == "uniform":
        return torch.ones_like(k, dtype=torch.float32)
    if strategy == "constant":
        return torch.full_like(k, core.constant_weight, dtype=torch.float32)
    snr = core.snr[k]
    if strategy == "min_snr":
        epsilon_weighting = snr.clamp(max=core.snr_clip) / snr.clamp(min=1e-8)
    elif strategy == "sigmoid":
        epsilon_weighting = torch.sigmoid(core.sigmoid_bias - torch.log(snr))
    else:
        raise ValueError(strategy)
    if core.objective == "pred_noise":
        return epsilon_weighting
    if core.objective == "pred_x0":
        return epsilon_weighting * snr
    return epsilon_weighting * snr / (snr + 1)


# --------------------------------------------------------------------------- #
# (1) Discrete: full (ctor_strategy x override) matrix vs manual weights
# --------------------------------------------------------------------------- #


class TestDiscreteWeightOverrideMatrix:
    @pytest.mark.parametrize("ctor_strategy", DISCRETE_STRATEGIES)
    @pytest.mark.parametrize("override", [None] + DISCRETE_STRATEGIES)
    def test_weights_match_manual(self, ctor_strategy, override):
        core = _make_discrete(ctor_strategy)
        effective = ctor_strategy if override is None else override
        w = core.compute_loss_weights(K_2D, strategy=override)
        expected = _manual_discrete_weights(core, K_2D, effective)
        torch.testing.assert_close(w, expected)

    @pytest.mark.parametrize("ctor_strategy", DISCRETE_STRATEGIES)
    def test_none_override_bit_identical_to_legacy_call(self, ctor_strategy):
        """``strategy=None`` and the pre-change positional call traverse the
        identical path (bitwise-equal weights)."""
        core = _make_discrete(ctor_strategy)
        assert torch.equal(
            core.compute_loss_weights(K_2D),
            core.compute_loss_weights(K_2D, strategy=None),
        )
        assert torch.equal(
            core.compute_loss_weights(K_2D, strategy=None),
            core.compute_loss_weights(K_2D, strategy=ctor_strategy),
        )

    def test_compute_loss_override_routes_weights(self):
        core = _make_discrete("min_snr")
        torch.manual_seed(0)
        B, T = 2, 4
        x = torch.randn(B, T, ACTION_DIM)
        k = K_2D.clone()
        noise = torch.randn(B, T, ACTION_DIM)
        x_t = core.q_sample(x, k, noise=noise)
        q_state = {"x_t": x_t, "k": k, "noise": noise, "x_start": x}
        v_pred = torch.randn(B, T, ACTION_DIM)

        # Default path is bit-identical with and without the new kwarg.
        assert torch.equal(
            core.compute_loss(v_pred, q_state),
            core.compute_loss(v_pred, q_state, weighting_strategy="min_snr"),
        )

        # Constant override == unweighted per-token MSE * constant_weight.
        target = core.predict_v(x, k, noise)
        per_token = ((v_pred - target) ** 2).mean(dim=-1)
        torch.testing.assert_close(
            core.compute_loss(v_pred, q_state, weighting_strategy="constant"),
            per_token * CONSTANT_WEIGHT,
        )


# --------------------------------------------------------------------------- #
# (2) 'constant' ignores the objective-dependent factor
# --------------------------------------------------------------------------- #


class TestConstantStrategy:
    @pytest.mark.parametrize("objective", ["pred_noise", "pred_x0", "pred_v"])
    def test_constant_ignores_objective_factor(self, objective):
        core = _make_discrete("constant", objective=objective)
        w = core.compute_loss_weights(K_2D)
        assert torch.equal(
            w, torch.full_like(K_2D, CONSTANT_WEIGHT, dtype=torch.float32)
        )

    def test_constant_weight_default_is_one(self):
        core = DiscreteDiffusion(
            action_dim=ACTION_DIM, loss_weighting_strategy="constant"
        )
        assert torch.equal(
            core.compute_loss_weights(K_2D), torch.ones_like(K_2D, dtype=torch.float32)
        )


# --------------------------------------------------------------------------- #
# (3) max_beta clamp touches only the terminal schedule entries
# --------------------------------------------------------------------------- #


class TestMaxBeta:
    def test_max_beta_changes_only_terminal(self):
        ref = DiscreteDiffusion(action_dim=ACTION_DIM)  # max_beta=None
        clamped = DiscreteDiffusion(action_dim=ACTION_DIM, max_beta=0.999)

        assert torch.equal(ref.betas[:-1], clamped.betas[:-1])
        assert torch.equal(ref.alphas_cumprod[:-1], clamped.alphas_cumprod[:-1])
        assert torch.equal(ref.snr[:-1], clamped.snr[:-1])

        assert ref.betas[-1].item() == pytest.approx(1.0)
        assert clamped.betas[-1].item() == pytest.approx(0.999)

        # D15 landmine: terminal SNR 0 => inf recip; the clamp defuses it.
        assert torch.isinf(ref.sqrt_recip_alphas_cumprod[-1])
        assert torch.isfinite(clamped.sqrt_recip_alphas_cumprod[-1])
        assert clamped.snr[-1] > 0


# --------------------------------------------------------------------------- #
# (4) Buffers registered unconditionally, persistent=False
# --------------------------------------------------------------------------- #


class TestUnconditionalBuffers:
    @pytest.mark.parametrize("ctor_strategy", DISCRETE_STRATEGIES + ["fused_min_snr"])
    def test_buffers_exist_for_every_strategy(self, ctor_strategy):
        core = _make_discrete(ctor_strategy)
        assert core.clipped_snr.shape == (core.timesteps,)
        assert core.logsnr.shape == (core.timesteps,)
        # persistent=False — checkpoints must not change.
        assert "clipped_snr" not in core.state_dict()
        assert "logsnr" not in core.state_dict()

    def test_override_works_on_uniform_configured_core(self):
        """The original motivation: a min_snr/sigmoid override must work even
        when the ctor strategy never needed those buffers."""
        core = _make_discrete("uniform")
        for override in ("min_snr", "sigmoid"):
            torch.testing.assert_close(
                core.compute_loss_weights(K_2D, strategy=override),
                _manual_discrete_weights(core, K_2D, override),
            )


# --------------------------------------------------------------------------- #
# (5) fused_min_snr: 1-D-k fallback warning + per-call override rejection
# --------------------------------------------------------------------------- #


class TestFusedMinSnr:
    def test_fused_1d_k_warns_once_and_falls_back_to_min_snr(self, monkeypatch):
        monkeypatch.setattr(
            discrete_diffusion_mod, "_FUSED_MIN_SNR_1D_K_WARNED", False
        )
        core = _make_discrete("fused_min_snr")
        k_1d = torch.tensor([0, 250, 900, 998], dtype=torch.long)

        with pytest.warns(UserWarning, match="fused_min_snr"):
            w = core.compute_loss_weights(k_1d)
        torch.testing.assert_close(
            w, _manual_discrete_weights(core, k_1d, "min_snr")
        )

        # Once per process: the second call must stay silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            core.compute_loss_weights(k_1d)

    def test_fused_2d_k_does_not_warn(self, monkeypatch):
        monkeypatch.setattr(
            discrete_diffusion_mod, "_FUSED_MIN_SNR_1D_K_WARNED", False
        )
        core = _make_discrete("fused_min_snr")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            core.compute_loss_weights(K_2D)

    def test_fused_override_raises_discrete(self):
        core = _make_discrete("min_snr")
        with pytest.raises(ValueError, match="fused_min_snr"):
            core.compute_loss_weights(K_2D, strategy="fused_min_snr")

        torch.manual_seed(0)
        x = torch.randn(2, 4, ACTION_DIM)
        noise = torch.randn_like(x)
        q_state = {
            "x_t": core.q_sample(x, K_2D, noise=noise),
            "k": K_2D,
            "noise": noise,
            "x_start": x,
        }
        with pytest.raises(ValueError, match="fused_min_snr"):
            core.compute_loss(
                torch.randn_like(x), q_state, weighting_strategy="fused_min_snr"
            )

    def test_fused_override_raises_continuous(self):
        core = ContinuousDiffusion(action_dim=ACTION_DIM)
        torch.manual_seed(0)
        x = torch.randn(2, 4, ACTION_DIM)
        t = torch.rand(2, 4)
        q_state = core.q_sample(x, t)
        with pytest.raises(ValueError, match="fused_min_snr"):
            core.compute_loss_weights(
                q_state["logsnr"], strategy="fused_min_snr"
            )
        with pytest.raises(ValueError, match="fused_min_snr"):
            core.compute_loss(
                torch.randn_like(x), q_state, weighting_strategy="fused_min_snr"
            )


# --------------------------------------------------------------------------- #
# (6) Continuous: (ctor x override) matrix on compute_loss + weights helper
# --------------------------------------------------------------------------- #


class TestContinuousWeightOverrideMatrix:
    def _setup(self, ctor_strategy: str):
        core = ContinuousDiffusion(
            action_dim=ACTION_DIM,
            loss_weighting=ctor_strategy,
            constant_weight=CONSTANT_WEIGHT,
        )
        torch.manual_seed(0)
        x = torch.randn(2, 4, ACTION_DIM)
        t = torch.rand(2, 4)
        q_state = core.q_sample(x, t)
        v_pred = torch.randn_like(x)
        return core, q_state, v_pred

    @pytest.mark.parametrize("ctor_strategy", CONTINUOUS_STRATEGIES)
    @pytest.mark.parametrize("override", [None] + CONTINUOUS_STRATEGIES)
    def test_loss_matches_manual(self, ctor_strategy, override):
        core, q_state, v_pred = self._setup(ctor_strategy)
        effective = ctor_strategy if override is None else override

        noise_pred = (
            q_state["alpha_t"] * v_pred + q_state["sigma_t"] * q_state["x_t"]
        )
        base = (noise_pred - q_state["noise"]) ** 2
        if effective == "uniform":
            expected = base
        elif effective == "constant":
            expected = base * CONSTANT_WEIGHT
        else:
            expected = base * torch.sigmoid(
                core.sigmoid_bias - q_state["logsnr"]
            ).unsqueeze(-1)

        out = core.compute_loss(v_pred, q_state, weighting_strategy=override)
        torch.testing.assert_close(out, expected)

    @pytest.mark.parametrize("ctor_strategy", CONTINUOUS_STRATEGIES)
    def test_none_override_bit_identical_to_legacy_call(self, ctor_strategy):
        core, q_state, v_pred = self._setup(ctor_strategy)
        assert torch.equal(
            core.compute_loss(v_pred, q_state),
            core.compute_loss(v_pred, q_state, weighting_strategy=None),
        )
        assert torch.equal(
            core.compute_loss(v_pred, q_state),
            core.compute_loss(v_pred, q_state, weighting_strategy=ctor_strategy),
        )

    @pytest.mark.parametrize("override", [None] + CONTINUOUS_STRATEGIES)
    def test_weights_helper_matches_manual(self, override):
        core, q_state, _ = self._setup("sigmoid")
        logsnr = q_state["logsnr"]
        effective = "sigmoid" if override is None else override
        if effective == "uniform":
            expected = torch.ones_like(logsnr)
        elif effective == "constant":
            expected = torch.full_like(logsnr, CONSTANT_WEIGHT)
        else:
            expected = torch.sigmoid(core.sigmoid_bias - logsnr)
        torch.testing.assert_close(
            core.compute_loss_weights(logsnr, strategy=override), expected
        )
