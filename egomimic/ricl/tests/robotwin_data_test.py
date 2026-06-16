"""Unit tests for the RoboTwin RICL shim (egomimic/ricl/robotwin_data.py).

Builds a tiny *synthetic* RoboTwin corpus (no archive download, no simulator) and validates the
corpus accessors, the query dataset keys/shapes/normalization, the bank provider,
the within-task leave-one-out retrieval cache + random controls, and the RICL
collate output shapes/masks. Uses a deterministic fake embedding provider so no
DINOv2 weights are needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.ricl import robotwin_data as R
from egomimic.ricl.data import build_ricl_collate
from egomimic.ricl.scripts.download_robotwin import make_synthetic_robotwin_task

N_EPS = 4
N_FRAMES = 40
HORIZON = 15
K = 4


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = tmp_path_factory.mktemp("robotwin_fixture")
    # two tasks (groups) so within-task LOO and cross-task random-bank are testable
    make_synthetic_robotwin_task(
        str(d), task="pick_cube", n_episodes=N_EPS, n_frames=N_FRAMES, seed=0
    )
    make_synthetic_robotwin_task(
        str(d), task="stack_blocks", n_episodes=N_EPS, n_frames=N_FRAMES, seed=100
    )
    return str(d)


@pytest.fixture(scope="module")
def corpus(root):
    return R.RoboTwinCorpus(root)


def test_corpus_discovery(corpus):
    assert len(corpus.group_to_hashes) == 2
    assert set(corpus.group_to_hashes) == {"pick_cube", "stack_blocks"}
    assert len(corpus.hashes) == 2 * N_EPS
    for h in corpus.hashes:
        assert corpus.num_frames(h) == N_FRAMES
        assert corpus.group_of(h) in corpus.group_to_hashes
    # detected qpos layout: 6-DOF arms -> 14-D state, grippers at slots 6 and 13
    assert corpus.state_dim == 14
    assert corpus.gripper_slots == (6, 13)
    # quantiles computed for state + actions, per-dim over the qpos dims
    for which in ("state", "actions"):
        q01, q99 = corpus.quantiles[which]
        assert q01.shape == (corpus.state_dim,) and q99.shape == (corpus.state_dim,)


def test_state_action_shapes_and_chunk_padding(corpus):
    h = corpus.hashes[0]
    D = corpus.state_dim
    assert corpus.state(h, 0).shape == (D,)
    chunk = corpus.action_chunk(h, 0, HORIZON)
    assert chunk.shape == (HORIZON, D)
    # near the episode tail the chunk repeat-pads the last action
    tail = corpus.action_chunk(h, N_FRAMES - 2, HORIZON)
    assert tail.shape == (HORIZON, D)
    assert np.allclose(tail[-1], tail[-2])  # padded with the last frame


def test_image_decode(corpus):
    img = corpus.image(corpus.hashes[0], 0, "base_0_rgb")
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8


def test_query_sample_keys_and_norm(corpus):
    D = corpus.state_dim
    ds = R.RoboTwinQueryDataset(corpus, corpus.hashes[:1], action_horizon=HORIZON)
    s = ds[0]
    assert s[R.STATE_KEY].shape == (32,)
    assert s[R.AC_KEY].shape == (HORIZON, 32)
    # slot-fill: pad dims D..31 are zero on both state and action
    assert np.allclose(s[R.STATE_KEY][D:], 0.0)
    assert np.allclose(s[R.AC_KEY][:, D:], 0.0)
    # quantile-normalized payload roughly within [-1, 1] (q01/q99 clip the bulk)
    assert s[R.STATE_KEY][:D].min() > -5 and s[R.STATE_KEY][:D].max() < 5
    assert s[R.PROMPT_STATE_KEY].shape == (D,)
    for cam in R.CAM_KEYS:
        assert s[cam].shape == (224, 224, 3)
        assert s[cam].dtype == np.uint8
    assert isinstance(s["annotations"], list) and isinstance(s["annotations"][0], str)
    assert s["episode_hash"] == corpus.hashes[0]
    assert s["frame_idx"] == 0


def test_bank_provider(corpus):
    D = corpus.state_dim
    prov = R.make_robotwin_bank_provider(corpus, action_horizon=HORIZON)
    blk = prov(corpus.hashes[0], 3)
    assert blk["image"].shape == (224, 224, 3)
    assert blk["state"].shape == (D,)
    assert blk["action"].shape == (HORIZON, D)


def test_retrieval_cache_and_collate(corpus):
    D = corpus.state_dim
    embed = R.make_fake_embedding_provider(corpus, dim=128)
    cache = R.build_robotwin_retrieval_cache(corpus, k=K, embeddings_provider=embed)
    R._verify_cache(cache, corpus)

    prov = R.make_robotwin_bank_provider(corpus, action_horizon=HORIZON)
    collate = build_ricl_collate(
        cache,
        prov,
        k=K,
        action_horizon=HORIZON,
        state_dim=D,
        action_dim=D,
    )
    ds = R.RoboTwinQueryDataset(corpus, corpus.hashes[:2], action_horizon=HORIZON)
    batch = collate([ds[0], ds[N_FRAMES // 2], ds[N_FRAMES + 1]])
    B = 3
    assert batch["ricl_retrieved_images"].shape == (B, K, 3, 224, 224)
    assert batch["ricl_retrieved_state"].shape == (B, K, D)
    assert batch["ricl_retrieved_action"].shape == (B, K, HORIZON, D)
    assert batch["ricl_retrieved_mask"].shape == (B, K)
    imgs = batch["ricl_retrieved_images"]
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    assert batch["ricl_retrieved_mask"].any()  # in-task frames have neighbors
    # base query keys survived the collate
    assert batch[R.AC_KEY].shape == (B, HORIZON, 32)
    assert batch[R.STATE_KEY].shape == (B, 32)
    assert len(batch["annotations"]) == B


def test_random_neighbor_caches(corpus):
    embed = R.make_fake_embedding_provider(corpus, dim=64)
    within = R.build_random_neighbor_cache(
        corpus, k=K, scope="within", seed=1, embeddings_provider=embed
    )
    bank = R.build_random_neighbor_cache(
        corpus, k=K, scope="bank", seed=1, embeddings_provider=embed
    )
    # within-scope neighbors stay in the query's own group; bank-scope may cross
    for qh in within.query_hashes:
        bh, _, dd = within.neighbors(qh, 0)
        grp = set(corpus.group_to_hashes[corpus.group_of(qh)])
        for h in (str(x) for x in bh if str(x)):
            assert h in grp and h != qh
        assert np.all(dd >= 0.0)  # true embedding L2 carried (not fake 0)
    crossed = False
    for qh in bank.query_hashes:
        bh, _, _ = bank.neighbors(qh, 0)
        grp = set(corpus.group_to_hashes[corpus.group_of(qh)])
        if any(str(x) and str(x) not in grp for x in bh):
            crossed = True
    assert crossed, "bank scope should sometimes pull cross-task demos"


def test_shared_quantiles_across_corpora(root):
    """An eval corpus can reuse the train corpus's quantiles for matched norm."""
    train = R.RoboTwinCorpus(root, groups=["pick_cube"])
    evald = R.RoboTwinCorpus(root, groups=["stack_blocks"], quantiles=train.quantiles)
    for which in ("state", "actions"):
        assert np.array_equal(train.quantiles[which][0], evald.quantiles[which][0])
        assert np.array_equal(train.quantiles[which][1], evald.quantiles[which][1])
