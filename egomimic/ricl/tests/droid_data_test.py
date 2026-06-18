"""CPU smoke tests for the DROID RICL shim (egomimic/ricl/droid_data.py).

Skips automatically if the DROID corpus isn't on disk. Validates the query
dataset keys/shapes/normalization range, the bank provider, and a tiny
leave-one-out retrieval cache + the ricl collate output shapes/masks.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from egomimic.ricl import droid_data as D
from egomimic.ricl.data import build_ricl_collate

pytestmark = pytest.mark.skipif(
    not os.path.isdir(D.DROID_ROOT), reason="DROID corpus not on this machine"
)


@pytest.fixture(scope="module")
def corpus():
    all_g = sorted(
        d
        for d in os.listdir(D.DROID_ROOT)
        if os.path.isdir(os.path.join(D.DROID_ROOT, d)) and not d.startswith(".")
    )
    return D.DroidCorpus(groups=all_g[:2])


def test_query_sample_keys_and_norm(corpus):
    ds = D.DroidQueryDataset(corpus, corpus.hashes[:1], action_horizon=15)
    s = ds[0]
    assert s[D.STATE_KEY].shape == (32,)
    assert s[D.AC_KEY].shape == (15, 32)
    # slot-fill: dims 8..31 are zero on both state and action
    assert np.allclose(s[D.STATE_KEY][8:], 0.0)
    assert np.allclose(s[D.AC_KEY][:, 8:], 0.0)
    # quantile-normalized payload roughly within [-1, 1] (q01/q99 clip the bulk)
    assert s[D.STATE_KEY][:8].min() > -5 and s[D.STATE_KEY][:8].max() < 5
    for cam in D.CAM_KEYS:
        assert s[cam].shape == (224, 224, 3)
        assert s[cam].dtype == np.uint8
    assert isinstance(s["annotations"], list) and isinstance(s["annotations"][0], str)
    assert s["episode_hash"] == corpus.hashes[0]
    assert s["frame_idx"] == 0


def test_bank_provider(corpus):
    prov = D.make_droid_bank_provider(corpus)
    blk = prov(corpus.hashes[0], 3)
    assert blk["image"].shape == (224, 224, 3)
    assert blk["state"].shape == (8,)
    assert blk["action"].shape == (1, 8)


def test_cache_and_collate(corpus):
    k = 4
    cache = D.build_droid_retrieval_cache(corpus, k=k)
    D._verify_cache(cache, corpus)

    prov = D.make_droid_bank_provider(corpus)
    collate = build_ricl_collate(
        cache,
        prov,
        k=k,
        action_horizon=1,
        state_dim=32,
        action_dim=32,
    )
    ds = D.DroidQueryDataset(corpus, corpus.hashes[:2], action_horizon=15)
    batch = collate([ds[0], ds[10], ds[20]])
    B = 3
    assert batch["ricl_retrieved_images"].shape == (B, k, 3, 224, 224)
    assert batch["ricl_retrieved_state"].shape == (B, k, 32)
    assert batch["ricl_retrieved_action"].shape == (B, k, 1, 32)
    assert batch["ricl_retrieved_mask"].shape == (B, k)
    # retrieved images scaled to [0,1]; padded slots zero
    imgs = batch["ricl_retrieved_images"]
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    # at least one real neighbor for an in-group query frame
    assert batch["ricl_retrieved_mask"].any()
    # padded state slots 8..31 are zero
    assert torch.allclose(
        batch["ricl_retrieved_state"][..., 8:],
        torch.zeros_like(batch["ricl_retrieved_state"][..., 8:]),
    )
    # base query keys survived the collate
    assert batch[D.AC_KEY].shape == (B, 15, 32)
    assert batch[D.STATE_KEY].shape == (B, 32)
    assert len(batch["annotations"]) == B
