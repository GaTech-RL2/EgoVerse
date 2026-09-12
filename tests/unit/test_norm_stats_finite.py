"""Norm-stat inference must not let a NaN/Inf sample poison the quantiles."""

import numpy as np
import pytest

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def test_drop_nonfinite_rows_keeps_finite_samples_only():
    X = np.ones((5, 100, 14))
    X[1, 3, 7] = np.nan  # one NaN cell in sample 1
    X[4, 0, 0] = np.inf  # one Inf cell in sample 4
    kept = MultiDataset._drop_nonfinite_rows(X, "actions_cartesian")
    assert kept.shape == (3, 100, 14)
    assert np.isfinite(kept).all()


def test_drop_nonfinite_rows_is_noop_on_clean_input():
    X = np.random.default_rng(0).normal(size=(6, 14))
    assert MultiDataset._drop_nonfinite_rows(X, "k") is X


def test_drop_nonfinite_rows_raises_when_nothing_is_finite():
    X = np.full((3, 2), np.nan)
    with pytest.raises(ValueError, match="every collected norm sample is non-finite"):
        MultiDataset._drop_nonfinite_rows(X, "k")


def test_stats_are_finite_after_dropping_bad_samples():
    X = np.random.default_rng(1).normal(size=(50, 100, 14))
    X[7] = np.nan  # a whole bad sample (e.g. an all-NaN gripper command)
    stats = MultiDataset._compute_stats_for_array(
        MultiDataset._drop_nonfinite_rows(X, "actions_cartesian")
    )
    for name, arr in stats.items():
        assert np.isfinite(arr).all(), name
