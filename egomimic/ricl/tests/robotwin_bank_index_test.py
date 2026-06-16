"""Bank-index builder + OnlineRetriever round-trip (CPU, synthetic, no DINOv2).

Covers the eval-side glue: ``build_robotwin_bank_index.build_bank_index`` writes a
consolidated ``{vectors.npy, refs.npz, manifest.json}`` index over a RoboTwin corpus,
which ``build_embedding_index.build_retrieval_index`` loads and
``robotwin_adapter.OnlineRetriever`` queries online. Uses fake embeddings + a stub
query embedder so it runs on a login node.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from egomimic.ricl import robotwin_data as R
from egomimic.ricl.robotwin_adapter import OnlineRetriever
from egomimic.ricl.scripts.build_embedding_index import build_retrieval_index
from egomimic.ricl.scripts.build_robotwin_bank_index import build_bank_index
from egomimic.ricl.scripts.download_robotwin import make_synthetic_robotwin_task

DIM = 128


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("robotwin_bank"))
    make_synthetic_robotwin_task(root, "pick_cube", n_episodes=4, n_frames=30)
    return R.RoboTwinCorpus(root)


class _StubEmbedder:
    """Deterministic query embedder matching the fake bank-index dim."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, imgs):
        return np.zeros((np.asarray(imgs).shape[0], self.dim), dtype=np.float32)


def test_build_bank_index_artifacts(corpus, tmp_path):
    out = str(tmp_path / "bank_index")
    build_bank_index(corpus, out, R.make_fake_embedding_provider(corpus, dim=DIM))

    assert os.path.isfile(os.path.join(out, "vectors.npy"))
    assert os.path.isfile(os.path.join(out, "refs.npz"))
    with open(os.path.join(out, "manifest.json")) as f:
        manifest = json.load(f)

    vectors = np.load(os.path.join(out, "vectors.npy"))
    refs = np.load(os.path.join(out, "refs.npz"))
    n_total = sum(corpus.num_frames(h) for h in corpus.hashes)
    assert vectors.shape == (n_total, DIM)
    assert manifest["n_frames"] == n_total
    assert manifest["embed_dim"] == DIM
    # refs reference only real corpus episodes, frame_idx within range
    hashes = set(corpus.hashes)
    assert set(refs["episode_hash"].tolist()) <= hashes
    for h, fi in zip(refs["episode_hash"], refs["frame_idx"]):
        assert 0 <= int(fi) < corpus.num_frames(str(h))


def test_online_retriever_roundtrip(corpus, tmp_path):
    out = str(tmp_path / "bank_index2")
    build_bank_index(corpus, out, R.make_fake_embedding_provider(corpus, dim=DIM))
    index = build_retrieval_index(out)
    bank_provider = R.make_robotwin_bank_provider(corpus, action_horizon=15)
    ret = OnlineRetriever(
        index,
        bank_provider,
        _StubEmbedder(DIM),
        k=4,
        action_horizon=15,
        state_dim=corpus.state_dim,
    )
    q = corpus.image(corpus.hashes[0], 0, "base_0_rgb")
    blocks = ret.retrieve(q)
    d = corpus.state_dim
    assert blocks["ricl_retrieved_images"].shape == (1, 4, 3, 224, 224)
    assert blocks["ricl_retrieved_state"].shape == (1, 4, d)
    assert blocks["ricl_retrieved_action"].shape == (1, 4, 15, d)
    assert blocks["ricl_retrieved_mask"].shape == (1, 4)
    assert bool(blocks["ricl_retrieved_mask"].any())
