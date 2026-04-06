"""
Tests for the KSG mutual information estimator.

Theoretical ground truth for two jointly Gaussian variables with
correlation ρ:

    I(X; Y) = -0.5 * log(1 - ρ²)   (nats)

We verify that ksg_mi and ksg_mi_averaged recover this within 15%
for N ≥ 1000 samples and typical correlation values ρ ∈ {0.5, 0.9}.
"""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.curation.ksg import ksg_mi, ksg_mi_averaged

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _true_mi_gaussian(rho: float) -> float:
    """Analytical MI for bivariate Gaussian with correlation ρ (nats)."""
    return -0.5 * np.log(1.0 - rho**2)


def _sample_bivariate_gaussian(
    rho: float,
    n: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (X, Y) from a bivariate standard Gaussian with correlation ρ."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    samples = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
    x = samples[:, 0:1]  # (N, 1)
    y = samples[:, 1:2]  # (N, 1)
    return x, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKsgMiShape:
    """Output shape and type contracts."""

    def test_returns_ndarray(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 2))
        y = rng.standard_normal((100, 3))
        mi = ksg_mi(x, y, k=3)
        assert isinstance(mi, np.ndarray)

    def test_output_shape_matches_n(self):
        rng = np.random.default_rng(1)
        N = 200
        x = rng.standard_normal((N, 2))
        y = rng.standard_normal((N, 2))
        mi = ksg_mi(x, y, k=3)
        assert mi.shape == (N,)

    def test_raises_when_k_ge_n(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 2))
        with pytest.raises(ValueError, match="k="):
            ksg_mi(x, y, k=10)

    def test_1d_inputs_accepted(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        mi = ksg_mi(x, y, k=3)
        assert mi.shape == (100,)


class TestKsgMiAccuracy:
    """
    Accuracy against analytically known MI for bivariate Gaussians.

    Tolerance: 15% relative error on N=1000 samples.
    """

    @pytest.mark.parametrize("rho", [0.5, 0.9])
    def test_mean_mi_within_tolerance(self, rho: float):
        N = 1000
        x, y = _sample_bivariate_gaussian(rho, N, seed=42)
        mi_samples = ksg_mi(x, y, k=5)
        estimated = float(mi_samples.mean())
        true_val = _true_mi_gaussian(rho)

        rel_error = abs(estimated - true_val) / true_val
        assert rel_error < 0.15, (
            f"ρ={rho}: estimated MI={estimated:.4f}, true={true_val:.4f}, "
            f"relative error={rel_error:.2%} (> 15%)"
        )

    @pytest.mark.parametrize("rho", [0.5, 0.9])
    def test_averaged_mi_within_tolerance(self, rho: float):
        N = 1000
        x, y = _sample_bivariate_gaussian(rho, N, seed=99)
        mi_samples = ksg_mi_averaged(x, y, k_range=(3, 7))
        estimated = float(mi_samples.mean())
        true_val = _true_mi_gaussian(rho)

        rel_error = abs(estimated - true_val) / true_val
        assert rel_error < 0.15, (
            f"ρ={rho}: averaged MI={estimated:.4f}, true={true_val:.4f}, "
            f"relative error={rel_error:.2%} (> 15%)"
        )

    def test_high_correlation_higher_mi_than_low(self):
        """MI(X; Y) should be larger for ρ=0.9 than ρ=0.5."""
        N = 1000
        x_lo, y_lo = _sample_bivariate_gaussian(rho=0.5, n=N, seed=7)
        x_hi, y_hi = _sample_bivariate_gaussian(rho=0.9, n=N, seed=7)
        mi_lo = ksg_mi(x_lo, y_lo, k=5).mean()
        mi_hi = ksg_mi(x_hi, y_hi, k=5).mean()
        assert (
            mi_hi > mi_lo
        ), f"Expected MI(ρ=0.9) > MI(ρ=0.5) but got {mi_hi:.4f} ≤ {mi_lo:.4f}"

    def test_independent_variables_near_zero(self):
        """Independent X, Y should have MI ≈ 0."""
        rng = np.random.default_rng(123)
        N = 1000
        x = rng.standard_normal((N, 1))
        y = rng.standard_normal((N, 1))
        mi = ksg_mi(x, y, k=5).mean()
        # Allow generous tolerance since KSG can have small bias for independent vars
        assert abs(mi) < 0.3, f"Expected MI ≈ 0 for independent vars, got {mi:.4f}"


class TestKsgMiAveraged:
    """Tests specific to ksg_mi_averaged."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        N = 100
        x = rng.standard_normal((N, 2))
        y = rng.standard_normal((N, 2))
        mi = ksg_mi_averaged(x, y, k_range=(3, 5))
        assert mi.shape == (N,)

    def test_raises_on_invalid_k_range(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((100, 2))
        y = rng.standard_normal((100, 2))
        with pytest.raises(ValueError, match="k_range"):
            ksg_mi_averaged(x, y, k_range=(7, 3))
