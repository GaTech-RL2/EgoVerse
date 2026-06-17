"""Tests for MultiDataset's per-sample retry on bounds/exception failures.

Verifies that retries draw from the *whole* MultiDataset, not just the
failing leaf, so a single bad episode can't trap the loader inside itself.
"""

from __future__ import annotations

import random

import pytest
import torch

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


class _DummyLeaf:
    """Minimal stand-in for a ZarrDataset leaf.

    `bad_local_idxs` is the set of local indices that should raise; all
    other indices return a trivial sample dict. We do not set `embodiment`,
    so MultiDataset._check_bounds short-circuits to None and returns the
    sample as-is — the only retry path exercised here is the
    `except Exception` branch in __getitem__.
    """

    def __init__(self, name: str, length: int, bad_local_idxs: set[int] | None = None):
        self.name = name
        self._length = length
        self._bad = bad_local_idxs or set()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, local_idx: int) -> dict:
        if local_idx in self._bad:
            raise ValueError(f"intentional failure at {self.name}[{local_idx}]")
        return {"leaf": self.name, "local_idx": local_idx}


def _make_mds(datasets: dict) -> MultiDataset:
    return MultiDataset(datasets=datasets, mode="total")


def test_retry_escapes_fully_bad_leaf():
    """A leaf where every local index fails must be escapable via retry."""
    bad_len = 4
    good_len = 6
    bad = _DummyLeaf("bad_ep", bad_len, bad_local_idxs=set(range(bad_len)))
    good = _DummyLeaf("good_ep", good_len)

    mds = _make_mds({"bad_ep": bad, "good_ep": good})
    # Sanity: index_map covers both leaves.
    assert len(mds) == bad_len + good_len

    # Hit an index that lives in the bad leaf. With per-leaf retry this would
    # exhaust attempts inside "bad_ep" and raise RuntimeError. With global
    # retry it should land somewhere in "good_ep" instead.
    bad_global_idx = mds._global_indices_by_dataset["bad_ep"][0]

    random.seed(0)
    sample = mds[bad_global_idx]
    assert sample["leaf"] == "good_ep"


def test_retry_pool_is_global(monkeypatch):
    """The candidate pool passed to random.choice should be the whole
    index_map (minus the failing index), not just the failing leaf's
    indices."""
    bad = _DummyLeaf("bad_ep", 3, bad_local_idxs={0, 1, 2})
    good = _DummyLeaf("good_ep", 5)
    mds = _make_mds({"bad_ep": bad, "good_ep": good})

    captured: list[list[int]] = []

    real_choice = random.choice

    def spy_choice(seq):
        # Materialize whatever was passed so we can inspect it.
        captured.append(list(seq))
        return real_choice(captured[-1])

    monkeypatch.setattr(random, "choice", spy_choice)

    bad_global_idx = mds._global_indices_by_dataset["bad_ep"][0]
    mds[bad_global_idx]

    assert captured, "retry never sampled — bad leaf wasn't exercised"
    # First retry's pool should include indices from the good leaf.
    good_globals = set(mds._global_indices_by_dataset["good_ep"])
    first_pool = set(captured[0])
    assert first_pool & good_globals, (
        f"first retry pool {first_pool} did not include any good-leaf indices "
        f"{good_globals} — retry is still scoped to the failing leaf"
    )


def test_retry_exhausts_when_everything_is_bad():
    """If every leaf is fully bad, the bounded while-loop must raise rather
    than spin forever."""
    a = _DummyLeaf("a", 2, bad_local_idxs={0, 1})
    b = _DummyLeaf("b", 2, bad_local_idxs={0, 1})
    mds = _make_mds({"a": a, "b": b})

    with pytest.raises(RuntimeError, match="Entire MultiDataset bad"):
        mds[0]


def _cartesian_stats(low: float = -1.0, high: float = 1.0) -> dict:
    q_low = torch.full((1, 12), low)
    q_high = torch.full((1, 12), high)
    return {
        "quantile_0_01": q_low,
        "quantile_99_99": q_high,
        "quantile_1": q_low,
        "quantile_99": q_high,
    }


def _make_cartesian_mds(reject_outliers: bool) -> MultiDataset:
    mds = MultiDataset(
        datasets={"ep": _DummyLeaf("ep", 1)},
        mode="total",
        reject_outliers=reject_outliers,
    )
    mds.key_types = {9: {"actions_cartesian": "action_keys"}}
    mds.zarr_keys = {9: {"actions_cartesian": "actions_cartesian"}}
    mds.norm_stats = {9: {"actions_cartesian": _cartesian_stats()}}
    return mds


def test_cartesian_rotation_columns_are_not_quantile_checked():
    mds = _make_cartesian_mds(reject_outliers=True)
    actions = torch.zeros(1, 12)
    actions[..., list(MultiDataset.CARTESIAN_ACTION_YPR_INDICES)] = 10.0
    data = {"embodiment": 9, "actions_cartesian": actions}

    assert mds._check_bounds(data, _DummyLeaf("ep", 1), 0, "ep") is None


def test_reject_outliers_false_accepts_finite_xyz_quantile_violation():
    mds = _make_cartesian_mds(reject_outliers=False)
    actions = torch.zeros(1, 12)
    actions[..., 0] = 2.0
    data = {"embodiment": 9, "actions_cartesian": actions}

    assert mds._check_bounds(data, _DummyLeaf("ep", 1), 0, "ep") is None


def test_reject_outliers_true_rejects_xyz_quantile_violation():
    mds = _make_cartesian_mds(reject_outliers=True)
    actions = torch.zeros(1, 12)
    actions[..., 0] = 2.0
    data = {"embodiment": 9, "actions_cartesian": actions}

    assert "Bounds violation" in mds._check_bounds(data, _DummyLeaf("ep", 1), 0, "ep")


# --- 6D (18D human / 20D robot) bounds checks: rotation columns excluded -----
from egomimic.utils.pose_utils import bimanual_cartesian_layout  # noqa: E402


def _make_cartesian_mds_width(reject_outliers: bool, width: int) -> MultiDataset:
    mds = MultiDataset(
        datasets={"ep": _DummyLeaf("ep", 1)},
        mode="total",
        reject_outliers=reject_outliers,
    )
    q_low = torch.full((1, width), -1.0)
    q_high = torch.full((1, width), 1.0)
    stats = {
        "quantile_0_01": q_low,
        "quantile_99_99": q_high,
        "quantile_1": q_low,
        "quantile_99": q_high,
    }
    mds.key_types = {9: {"actions_cartesian": "action_keys"}}
    mds.zarr_keys = {9: {"actions_cartesian": "actions_cartesian"}}
    mds.norm_stats = {9: {"actions_cartesian": stats}}
    return mds


@pytest.mark.parametrize("width", [18, 20])
def test_6d_rotation_columns_are_not_quantile_checked(width):
    mds = _make_cartesian_mds_width(reject_outliers=True, width=width)
    actions = torch.zeros(1, width)
    # Push every rotation column way out of [-1, 1]; must still pass.
    actions[..., list(bimanual_cartesian_layout(width)["rot"])] = 9.0
    data = {"embodiment": 9, "actions_cartesian": actions}
    assert mds._check_bounds(data, _DummyLeaf("ep", 1), 0, "ep") is None


@pytest.mark.parametrize("width", [18, 20])
def test_6d_xyz_outlier_still_rejected(width):
    mds = _make_cartesian_mds_width(reject_outliers=True, width=width)
    actions = torch.zeros(1, width)
    actions[..., bimanual_cartesian_layout(width)["xyz"][0]] = 5.0
    data = {"embodiment": 9, "actions_cartesian": actions}
    assert "Bounds violation" in mds._check_bounds(data, _DummyLeaf("ep", 1), 0, "ep")


def test_6d_robot_gripper_outlier_still_rejected():
    # 20D robot keeps grippers in the bounds check.
    mds = _make_cartesian_mds_width(reject_outliers=True, width=20)
    actions = torch.zeros(1, 20)
    actions[..., bimanual_cartesian_layout(20)["grip"][0]] = 5.0
    data = {"embodiment": 9, "actions_cartesian": actions}
    assert "Bounds violation" in mds._check_bounds(data, _DummyLeaf("ep", 1), 0, "ep")
