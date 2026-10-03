"""Joint-space Eva mode (ABC YAM / abc_sim action space): the keymap reads
``{left,right}.{obs,cmd}_{joints,gripper}``, the transform emits a 14-D state
and a (100, 14) action chunk in ABC order, the robot keymap supports the same
``_hist`` camera as Human, and joint chunks get joint metrics, not the
cartesian ones their 14-D width would otherwise trigger."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from egomimic.eval.action_metrics import layout_metrics
from egomimic.rldb.embodiment.eva import (
    JOINT_ACTION_KEY,
    JOINT_STATE_KEY,
    Eva,
)
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

T = 120
CAM = Eva.VIZ_IMAGE_KEY


def _write_episode(path, fps=30):
    rng = np.random.default_rng(0)
    t = np.arange(T, dtype=np.float32)[:, None]
    numeric = {}
    # Each channel carries a distinct offset so ordering is checkable.
    for a, arm in enumerate(("left", "right")):
        base = 10.0 * a
        numeric[f"{arm}.obs_joints"] = base + np.arange(6, dtype=np.float32) + 0 * t
        numeric[f"{arm}.cmd_joints"] = base + np.arange(6, dtype=np.float32) + 0.01 * t
        numeric[f"{arm}.obs_gripper"] = np.full((T, 1), 0.25 + a * 0.5, np.float32)
        numeric[f"{arm}.cmd_gripper"] = np.full((T, 1), 0.3 + a * 0.5, np.float32)
    images = {
        k: rng.integers(0, 255, (T, 24, 32, 3), np.uint8)
        for k in ("images.front_1", "images.left_wrist", "images.right_wrist")
    }
    ZarrWriter.create_and_write(
        path,
        numeric_data=numeric,
        image_data=images,
        embodiment="eva_bimanual",
        fps=fps,
        task_name="synthetic",
        intrinsics={"front_1": np.eye(3, 4)},
    )


def _leaf(tmp_path, **keymap_kwargs):
    _write_episode(tmp_path / "ep.zarr")
    key_map = Eva.get_keymap("joints", **keymap_kwargs)
    resolver = LocalEpisodeResolver(
        tmp_path, key_map=key_map, transform_list=Eva.get_transform_list("joints")
    )
    ds = MultiDataset._from_resolver(resolver, mode="total")
    return next(iter(ds.datasets.values()))


def test_joint_sample_shapes_and_abc_order(tmp_path):
    sample = _leaf(tmp_path)[10]
    state = sample[JOINT_STATE_KEY].float()
    actions = sample[JOINT_ACTION_KEY].float()
    assert state.shape == (14,) and actions.shape == (100, 14)
    expected = torch.tensor(
        [0, 1, 2, 3, 4, 5, 0.25, 10, 11, 12, 13, 14, 15, 0.75], dtype=torch.float32
    )
    assert torch.allclose(state, expected)
    # Chunk start = the commanded pose at idx 10 (+0.01 * 10 on the joints).
    assert torch.allclose(actions[0, :6], expected[:6] + 0.1, atol=1e-5)
    assert torch.allclose(actions[:, 6], torch.full((100,), 0.3))
    assert torch.allclose(actions[:, 13], torch.full((100,), 0.8))
    # 45 raw steps resampled onto 100: the last step is idx 10 + 44.
    assert torch.allclose(actions[-1, 0], torch.tensor(0.54), atol=1e-5)
    assert sample["fps"].item() == 30
    assert "left.cmd_joints" not in sample and "left.obs_joints" not in sample


def test_joint_keymap_supports_the_hist_camera(tmp_path):
    leaf = _leaf(tmp_path, image_history_gap_s=0.1)
    assert torch.equal(leaf[20][f"{CAM}_hist"], leaf[17][CAM])


def test_joint_actions_get_joint_metrics_not_cartesian():
    gt = torch.zeros(2, 100, 14)
    pred = gt.clone()
    pred[..., 0] = np.deg2rad(2.0)  # left j1 off by 2 degrees
    pred[..., 6] = 0.5  # left gripper off by 0.5
    m = layout_metrics(pred, gt, "Valid/x", ac_key=JOINT_ACTION_KEY)
    assert not any("xyz" in k for k in m)
    assert m["Valid/x_joint_abs_err_deg_avg"].item() == pytest.approx(2.0 / 12, rel=1e-4)
    assert m["Valid/x_joint_abs_err_deg_max_joint"].item() == pytest.approx(2.0, rel=1e-4)
    assert m["Valid/x_grip_paired_mse_avg"].item() == pytest.approx(0.125, rel=1e-4)
    # Without the key name the width alone reads as cartesian ypr.
    assert any("xyz" in k for k in layout_metrics(pred, gt, "Valid/x"))
