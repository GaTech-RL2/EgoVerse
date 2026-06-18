"""Integration test for the RoboTwin -> EgoVerse Zarr converter.

Converts a synthetic RoboTwin task to Zarr, then loads it back through the *real*
EgoVerse pipeline — ``ZarrDataset`` + ``Eva.get_keymap("cartesian_pi")`` +
``Eva.get_transform_list("cartesian")`` — proving the produced store is a valid
``eva_bimanual`` cartesian episode (the standard MultiDataset path) without any
S3/SQL. Also checks the converter errors clearly when ``endpose`` is absent.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from egomimic.ricl.scripts.download_robotwin import (
    make_synthetic_robotwin_episode,
    make_synthetic_robotwin_task,
)
from egomimic.ricl.scripts.robotwin_to_zarr import convert_episode, convert_task
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

CHUNK = 15
N_EPS = 2
N_FRAMES = 60


@pytest.fixture(scope="module")
def zarr_stores(tmp_path_factory):
    src = tmp_path_factory.mktemp("rt_src")
    out = tmp_path_factory.mktemp("rt_zarr")
    make_synthetic_robotwin_task(
        str(src), task="pick_cube", n_episodes=N_EPS, n_frames=N_FRAMES, seed=7
    )
    written = convert_task(str(src), str(out))
    assert len(written) == N_EPS
    return written


def test_zarr_loads_through_eva_pipeline(zarr_stores):
    keymap = Eva.get_keymap("cartesian_pi")
    transform_list = Eva.get_transform_list("cartesian", chunk_length=CHUNK)
    ds = ZarrDataset(zarr_stores[0], key_map=keymap, transform_list=transform_list)

    assert ds.embodiment == "eva_bimanual"
    assert len(ds) == N_FRAMES

    for idx in (0, N_FRAMES // 2):
        sample = ds[idx]
        # cartesian action chunk: per-arm xyz+ypr (6) + gripper (1) = 7, x2 = 14
        act = sample["actions_cartesian"]
        assert tuple(act.shape) == (CHUNK, 14)
        state = sample["observations.state.ee_pose"]
        assert state.shape[-1] == 14
        # PI/PaliGemma camera keys are present and look like images (H, W, 3)
        for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            img = np.asarray(sample[cam])
            assert img.ndim == 3 and 3 in img.shape
        assert np.isfinite(np.asarray(act)).all()
        assert np.isfinite(np.asarray(state)).all()


def test_zarr_metadata_and_keys(zarr_stores):
    import zarr

    store = zarr.open(zarr_stores[0], mode="r")
    attrs = dict(store.attrs)
    assert attrs["embodiment"] == "eva_bimanual"
    assert attrs["total_frames"] == N_FRAMES
    for key in (
        "images.front_1",
        "images.left_wrist",
        "images.right_wrist",
        "left.obs_ee_pose",
        "right.obs_ee_pose",
        "left.cmd_ee_pose",
        "right.cmd_ee_pose",
        "left.obs_gripper",
        "right.obs_gripper",
        "left.cmd_gripper",
        "right.cmd_gripper",
    ):
        assert key in attrs["features"], f"missing feature {key}"
    # ee poses are 7-D XYZWXYZ, grippers 1-D
    assert attrs["features"]["left.obs_ee_pose"]["shape"] == [7]
    assert attrs["features"]["left.obs_gripper"]["shape"] == [1]


def test_missing_endpose_raises(tmp_path):
    """An episode with only joint_action (no endpose) errors clearly."""
    ep = str(tmp_path / "noendpose" / "data" / "episode0.hdf5")
    make_synthetic_robotwin_episode(ep, n_frames=20, with_endpose=False)
    # sanity: the fixture really has no endpose group
    with h5py.File(ep, "r") as f:
        assert "endpose" not in f
    with pytest.raises(ValueError, match="endpose"):
        convert_episode(ep, str(tmp_path / "out.zarr"), task_name="noendpose")
