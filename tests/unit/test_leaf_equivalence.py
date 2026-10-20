"""The same recording read as a zarr leaf and as a Lance leaf gives the same samples."""

import numpy as np
import pytest
import torch

lance = pytest.importorskip("lance")

from fixtures.lance_tables import write_catalog  # noqa: E402

from egomimic.rldb.lance.dataset import (  # noqa: E402
    ZARR_TO_LANCE,
    LanceEpisodeResolver,
    _column,
)
from egomimic.rldb.leaf import EpisodeLeaf  # noqa: E402
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset  # noqa: E402
from egomimic.rldb.zarr.zarr_writer import ZarrWriter  # noqa: E402

KEY_MAP = {
    "left.action_ee_pose": {
        "key_type": "action_keys",
        "zarr_key": "left.obs_ee_pose",
        "horizon": 3,
    },
    "right.obs_ee_pose": {
        "key_type": "proprio_keys",
        "zarr_key": "right.obs_ee_pose",
        "history": 3,
        "history_stride": 2,
    },
    "obs_head_pose": {"key_type": "proprio_keys", "zarr_key": "obs_head_pose"},
    "annotations": {"key_type": "annotation_keys", "zarr_key": "annotations"},
}
TASK = "folding clothes"


def _zarr_twin(root, leaf) -> ZarrDataset:
    """A zarr episode holding the Lance leaf's steps rows, intrinsics and task."""
    steps = lance.dataset(leaf.steps_uri, version=leaf.steps_version).to_table(
        offset=leaf.row_start, limit=leaf.total_frames
    )
    keys = {s["zarr_key"] for s in KEY_MAP.values()} - {"annotations"}
    numeric = {zk: _column(steps, ZARR_TO_LANCE[zk]) for zk in keys}
    path = ZarrWriter.create_and_write(
        root / f"{leaf.episode_path}.zarr",
        numeric_data=numeric,
        embodiment=leaf.embodiment,
        fps=int(leaf.metadata["fps"]),
        annotations=[(TASK, 0, leaf.total_frames)],
        intrinsics={
            "front_1": np.c_[np.asarray(leaf.intrinsics).reshape(3, 3), np.zeros(3)]
        },
    )
    return ZarrDataset(path, key_map=KEY_MAP)


def test_zarr_and_lance_leaves_agree(tmp_path):
    lance_root = tmp_path / "lance"
    write_catalog(lance_root)
    leaves = LanceEpisodeResolver(lance_root, key_map=KEY_MAP, cameras={}).resolve()
    assert leaves
    for leaf in leaves.values():
        twin = _zarr_twin(tmp_path, leaf)
        assert isinstance(leaf, EpisodeLeaf) and isinstance(twin, EpisodeLeaf)
        for i in range(len(leaf)):
            a, b = leaf[i], twin[leaf.anchor(i)]
            # Only Lance tracks per-hand validity; every action chunk sampled here
            # has both hands (history can reach back to an untracked frame).
            assert bool((a.pop("action_hand_mask") == 1).all())
            a.pop("proprio_hand_valid")
            assert a.keys() == b.keys()
            for k in a:
                if isinstance(a[k], torch.Tensor):
                    torch.testing.assert_close(a[k], b[k], equal_nan=True, msg=k)
                else:
                    assert a[k] == b[k], k


def test_only_streaming_leaves_batch_fetch():
    from egomimic.rldb.lance.dataset import (
        LanceRecordingDataset,
        LanceStreamingDataset,
    )

    assert not EpisodeLeaf.batched_fetch
    assert not ZarrDataset.batched_fetch
    assert not LanceRecordingDataset.batched_fetch
    assert LanceStreamingDataset.batched_fetch
