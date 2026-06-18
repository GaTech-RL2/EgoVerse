"""Tests for egomimic/ricl/data.py (RICL collate + query-frame wrapper).

Import-light: torch + numpy only (mock cache / provider / base collate), so it
runs on a CPU-only box. Run under pytest or standalone:

    python egomimic/ricl/tests/data_test.py
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from egomimic.ricl import data as rdata
except Exception:
    import importlib.util
    import pathlib

    _spec = importlib.util.spec_from_file_location(
        "ricl_data", pathlib.Path(__file__).with_name("data.py")
    )
    rdata = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rdata)


class _MockCache:
    """Minimal stand-in for RetrievalCache: ._entries + .neighbors(qh, fi)."""

    def __init__(self, entries):
        self._entries = entries  # {query_hash: (bank_hash[k], bank_frame[k], dist[k])}

    def neighbors(self, query_hash, frame_idx):
        return self._entries[query_hash]


def _mock_provider(image_hw=(224, 224), state_dim=14, action_shape=(3, 32)):
    H, W = image_hw

    def provider(h, f):
        return {
            "image": np.full((H, W, 3), 123, np.uint8),  # HWC uint8
            "state": np.ones(state_dim, np.float32) * 0.5,
            "action": np.ones(action_shape, np.float32) * 0.25,
        }

    return provider


def _base_collate(batch):
    return {"episode_hash": [s["episode_hash"] for s in batch]}


def test_query_dataset_injects_frame_idx():
    class _Base:
        def __init__(self):
            self.index_map = {0: ("eva", 5), 1: ("eva", 9)}

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {"episode_hash": "H", "x": torch.zeros(3)}

    wrapped = rdata.RiclQueryDataset(_Base())
    assert wrapped[0]["frame_idx"] == 5
    assert wrapped[1]["frame_idx"] == 9
    assert len(wrapped) == 2


def _make_collate(k=4, **kw):
    entries = {
        # "H": 2 valid neighbors (A,B) + 2 padded slots
        "H": (
            np.array(["A", "B", "", ""], dtype="U"),
            np.array([0, 1, -1, -1], dtype=np.int32),
            np.array([0.1, 0.2, np.inf, np.inf], dtype=np.float32),
        )
    }
    cache = _MockCache(entries)
    return rdata.build_ricl_collate(
        cache, _mock_provider(), k=k, base_collate=_base_collate, **kw
    )


def test_collate_shapes_and_mask():
    k = 4
    collate = _make_collate(k=k, state_dim=32, action_dim=32, action_horizon=1)
    batch = [
        {"episode_hash": "H", "frame_idx": 0},  # in cache: 2 valid
        {"episode_hash": "MISSING", "frame_idx": 3},  # not in cache: all padded
    ]
    out = collate(batch)
    B = 2
    assert out["ricl_retrieved_images"].shape == (B, k, 3, 224, 224)
    assert out["ricl_retrieved_state"].shape == (B, k, 32)
    assert out["ricl_retrieved_action"].shape == (B, k, 1, 32)
    assert out["ricl_retrieved_mask"].shape == (B, k)
    assert out["ricl_retrieved_dist"].shape == (B, k)
    # sample 0: first two valid, last two padded
    assert out["ricl_retrieved_mask"][0].tolist() == [True, True, False, False]
    # sample 1: not in cache -> all masked, images zero
    assert out["ricl_retrieved_mask"][1].tolist() == [False, False, False, False]
    assert torch.count_nonzero(out["ricl_retrieved_images"][1]) == 0


def test_collate_image_normalized_and_resized():
    # provider returns 100x100 uint8 -> collate resizes to 224 and scales to [0,1]
    cache = _MockCache(
        {
            "H": (
                np.array(["A"], dtype="U"),
                np.array([0], dtype=np.int32),
                np.array([0.1], dtype=np.float32),
            )
        }
    )
    collate = rdata.build_ricl_collate(
        cache,
        _mock_provider(image_hw=(100, 100)),
        k=1,
        base_collate=_base_collate,
        image_hw=(224, 224),
        state_dim=32,
        action_dim=32,
        action_horizon=1,
    )
    out = collate([{"episode_hash": "H", "frame_idx": 0}])
    img = out["ricl_retrieved_images"][0, 0]
    assert img.shape == (3, 224, 224)
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0  # scaled from uint8
    assert float(img.mean()) > 0.0  # non-zero (valid neighbor)


def test_collate_state_and_action_fit():
    # provider state_dim=14 -> fit to 32 (pad); action (3,32) -> horizon 1 (truncate)
    collate = _make_collate(k=2, state_dim=32, action_dim=32, action_horizon=1)
    out = collate([{"episode_hash": "H", "frame_idx": 0}])
    st = out["ricl_retrieved_state"][0, 0]  # valid slot A
    assert st.shape == (32,)
    assert torch.count_nonzero(st[:14]) == 14 and torch.count_nonzero(st[14:]) == 0
    act = out["ricl_retrieved_action"][0, 0]
    assert act.shape == (1, 32)
    assert torch.allclose(act, torch.full((1, 32), 0.25), atol=1e-6)


def _run_all():
    fns = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
    print(f"== data_test: running {len(fns)} tests ==")
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print("ALL DATA TESTS PASSED")


if __name__ == "__main__":
    _run_all()
