"""Tests for egomimic/ricl/metrics.py. Import-light (numpy[, torch]).

Run under pytest or standalone:  python egomimic/ricl/metrics_test.py
"""

from __future__ import annotations

import numpy as np

try:
    from egomimic.ricl import metrics as M
except Exception:
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location(
        "ricl_metrics", pathlib.Path(__file__).with_name("metrics.py")
    )
    M = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(M)


def test_cartesian_mse_l1():
    pred = np.zeros((4, 14), np.float32)
    gt = np.ones((4, 14), np.float32)
    assert abs(M.cartesian_mse(pred, gt) - 1.0) < 1e-6
    assert abs(M.cartesian_l1(pred, gt) - 1.0) < 1e-6


def test_gripper_accuracy_14d():
    pred = np.zeros((2, 14), np.float32)
    gt = np.zeros((2, 14), np.float32)
    # gripper dims 6,13. Make pred match gt on dim 6, differ on dim 13.
    pred[:, 6] = 1.0
    gt[:, 6] = 1.0  # agree (both > 0)
    pred[:, 13] = 1.0
    gt[:, 13] = -1.0  # disagree
    # 2 dims x 2 samples = 4 comparisons; dim6 agree (2), dim13 disagree (2) -> 0.5
    assert abs(M.gripper_accuracy(pred, gt) - 0.5) < 1e-6


def test_gripper_accuracy_non_14d_is_nan():
    pred = np.zeros((2, 6), np.float32)
    gt = np.zeros((2, 6), np.float32)
    assert np.isnan(M.gripper_accuracy(pred, gt))


def test_compare_to_floor_helps_and_hurts():
    # retrieval lower loss/mse than floor -> helps
    retr = {"Valid/eva_loss": 0.5, "Valid/eva_paired_mse_avg": 0.10, "other": "x"}
    floor = {"Valid/eva_loss": 0.8, "Valid/eva_paired_mse_avg": 0.20, "other": "x"}
    cmp = M.compare_to_floor(retr, floor)
    assert cmp["retrieval_helps"] is True
    assert cmp["mean_improvement"] > 0
    assert abs(cmp["delta_Valid/eva_loss"] - (-0.3)) < 1e-6

    # retrieval worse -> does not help
    cmp2 = M.compare_to_floor(floor, retr)
    assert cmp2["retrieval_helps"] is False
    assert cmp2["mean_improvement"] < 0


def test_strip_ricl_keys():
    batch = {
        0: {
            "actions_cartesian": 1,
            "ricl_retrieved_images": 2,
            "ricl_retrieved_state": 3,
            "tokenized_prompt": 4,
        }
    }
    out = M.strip_ricl_keys(batch)
    assert set(out[0].keys()) == {"actions_cartesian", "tokenized_prompt"}


def test_compare_to_floor_torch_scalars():
    try:
        import torch
    except Exception:
        return  # torch optional
    retr = {"Valid/loss": torch.tensor(0.4)}
    floor = {"Valid/loss": torch.tensor(0.6)}
    cmp = M.compare_to_floor(retr, floor)
    assert cmp["retrieval_helps"] is True
    assert abs(cmp["delta_Valid/loss"] - (-0.2)) < 1e-5


def _run_all():
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    print(f"== metrics_test: running {len(fns)} tests ==")
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print("ALL METRICS TESTS PASSED")


if __name__ == "__main__":
    _run_all()
