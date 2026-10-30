"""Streaming leaves (batched MP4 byte-range reads) match the local per-sample MP4 path.

Needs EGOVERSE_TEST_LANCE_ROOT (pilot tables). CPU only.
"""

import os
import random
from pathlib import Path

import pytest
import torch

pytest.importorskip("lance")
pytest.importorskip("torchcodec")

from egomimic.rldb.embodiment.human import Human  # noqa: E402
from egomimic.rldb.lance.dataset import (  # noqa: E402
    LanceEpisodeResolver,
    LanceStreamingResolver,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset  # noqa: E402


@pytest.fixture(scope="module")
def root():
    src = os.environ.get("EGOVERSE_TEST_LANCE_ROOT")
    if not src:
        pytest.skip("set EGOVERSE_TEST_LANCE_ROOT")
    return Path(src)


def _dataset(root, resolver=LanceEpisodeResolver, **kw):
    leaves = resolver(
        root,
        key_map=Human.get_keymap(keymap_mode="keypoints"),
        transform_list=Human.get_transform_list(
            mode="keypoints_headframe_6d", stride=3
        ),
        cameras={"images.front_1": "front_1"},
        **kw,
    ).resolve(expected_embodiment="human_bimanual")
    return MultiDataset(datasets=leaves, mode="total")


def test_range_batch_matches_per_sample_mp4(root):
    per_sample = _dataset(root)
    batched = _dataset(root, LanceStreamingResolver)
    assert len(batched) == len(per_sample)
    rng = random.Random(0)
    idxs = [rng.randrange(len(batched)) for _ in range(48)]
    for i, got in zip(idxs, batched.__getitems__(idxs), strict=True):
        want = per_sample[i]
        assert got.keys() == want.keys()
        for k, v in want.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(got[k], v), (i, k)
            else:
                assert got[k] == v, (i, k)
