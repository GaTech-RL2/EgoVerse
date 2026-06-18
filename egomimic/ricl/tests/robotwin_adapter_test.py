"""Unit tests for the RoboTwin closed-loop adapter glue (robotwin_adapter.py).

Exercises the pure-Python pieces with the model + SAPIEN sim MOCKED: observation
mapping, quantile (un)normalization round-trip, slot-filled model input, action
un-normalization to qpos, and online retrieval against a tiny stub bank. No
checkpoint, no DINOv2, no sim.
"""

from __future__ import annotations

import numpy as np

from egomimic.ricl import robotwin_adapter as A
from egomimic.ricl.retrieval import BankRefs, RetrievalIndex

D = 14  # aloha-agilex qpos dim
H = 15
K = 4


def _fake_obs(d=D, hw=(48, 64)):
    H_, W_ = hw
    rng = np.random.default_rng(0)
    return {
        "observation": {
            "head_camera": {"rgb": np.full((H_, W_, 3), 10, np.uint8)},
            "left_camera": {"rgb": np.full((H_, W_, 3), 20, np.uint8)},
            "right_camera": {"rgb": np.full((H_, W_, 3), 30, np.uint8)},
        },
        "joint_action": {"vector": rng.standard_normal(d).astype(np.float32)},
    }


def test_obs_to_state_order_and_shape():
    obs = _fake_obs()
    rgb, state = A.obs_to_state(obs)
    assert len(rgb) == 3
    # order is [head, right, left] (matches pi05 encode_obs)
    assert rgb[0][0, 0, 0] == 10  # head
    assert rgb[1][0, 0, 0] == 30  # right
    assert rgb[2][0, 0, 0] == 20  # left
    assert state.shape == (D,)


def test_quantile_norm_roundtrip():
    q01 = np.full(D, -2.0, np.float32)
    q99 = np.full(D, 2.0, np.float32)
    x = np.linspace(-1.5, 1.5, D).astype(np.float32)
    x_norm = A.quantile_norm(x, q01, q99)
    x_back = A.quantile_unnorm(x_norm, q01, q99)
    assert np.allclose(x, x_back, atol=1e-4)


def test_state_to_model_input_slotfill():
    q = (np.full(D, -1.0, np.float32), np.full(D, 1.0, np.float32))
    state = np.zeros(D, np.float32)
    out = A.state_to_model_input(state, q)
    assert out.shape == (32,)
    assert np.allclose(out[D:], 0.0)  # pad dims zero


def test_unnormalize_action_to_qpos():
    q01 = np.full(D, -3.0, np.float32)
    q99 = np.full(D, 3.0, np.float32)
    qpos = np.linspace(-2.0, 2.0, D).astype(np.float32)
    # forward: qpos -> normalized -> slot-fill to 32 (what the model predicts)
    norm = A.quantile_norm(qpos, q01, q99)
    pred32 = np.concatenate([norm, np.zeros(32 - D, np.float32)])
    pred_chunk = np.tile(pred32, (H, 1))  # (H, 32)
    out = A.unnormalize_action(pred_chunk, (q01, q99), state_dim=D)
    assert out.shape == (H, D)
    assert np.allclose(out[0], qpos, atol=1e-4)  # inverts back to qpos


class _FakeEmbedder:
    """Returns a fixed descriptor equal to bank row 0 (so kNN is deterministic)."""

    def __init__(self, vec):
        self._vec = np.asarray(vec, np.float32)

    def embed(self, images_thwc):
        return np.tile(self._vec, (len(images_thwc), 1)).astype(np.float32)


def _fake_bank_provider(episode_hash, frame_idx):
    return {
        "image": np.full((32, 32, 3), 100, np.uint8),
        "state": np.full(D, 0.5, np.float32),
        "action": np.full((H, D), 0.25, np.float32),
    }


def test_online_retriever_blocks():
    dim = 8
    vectors = np.eye(5, dim, dtype=np.float32)  # 5 distinct bank rows
    refs = BankRefs(
        episode_hash=np.array([f"ep{i}" for i in range(5)]).astype("U"),
        frame_idx=np.zeros(5, np.int32),
    )
    index = RetrievalIndex(vectors, refs)
    retriever = A.OnlineRetriever(
        index,
        _fake_bank_provider,
        _FakeEmbedder(vectors[0]),  # query matches ep0
        k=K,
        action_horizon=H,
        state_dim=D,
    )
    out = retriever.retrieve(np.full((48, 64, 3), 7, np.uint8))
    assert out["ricl_retrieved_images"].shape == (1, K, 3, 224, 224)
    assert out["ricl_retrieved_state"].shape == (1, K, D)
    assert out["ricl_retrieved_action"].shape == (1, K, H, D)
    assert out["ricl_retrieved_mask"].shape == (1, K)
    assert bool(out["ricl_retrieved_mask"].all())  # 5 bank rows >= k -> all valid
    imgs = out["ricl_retrieved_images"]
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0  # scaled to [0,1]
    # nearest neighbor is bank row 0 (query == vectors[0]) at ~zero distance
    assert float(out["ricl_retrieved_dist"][0, 0]) < 1e-4
