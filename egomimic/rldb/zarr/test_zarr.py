from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.rldb.utils import S3RLDBDataset
from egomimic.rldb.zarr.action_chunk_transforms import (
    build_aria_bimanual_transform_list,
    build_eva_bimanual_transform_list,
    _matrix_to_xyzypr,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset
from egomimic.utils.egomimicUtils import EXTRINSICS


ZARR_EPISODE_PATH = Path(
    "/coc/flash7/rco3/EgoVerse/egomimic/rldb/zarr/zarr/new/1769460905119.zarr"
)
LEROBOT_EPISODE_HASH = "2026-01-26-20-55-05-119000"
LEROBOT_CACHE_ROOT = "/coc/flash7/skareer6/CacheEgoVerse/.cache"
EMBODIMENT = "eva_bimanual"
ACTION_HORIZON_REAL = 45
ACTION_CHUNK_LENGTH = 100
KEYS_TO_COMPARE = (
    "observations.images.front_img_1",
    "observations.images.right_wrist_img",
    "observations.images.left_wrist_img",
    "actions_cartesian",
    "observations.state.ee_pose",
)
IMAGE_KEYS = {
    "observations.images.front_img_1",
    "observations.images.right_wrist_img",
    "observations.images.left_wrist_img",
}

ARIA_ZARR_EPISODE_PATH = Path(
    "/coc/flash7/scratch/egoverseDebugDatasets/proc_zarr/1764285211791.zarr/"
)
ARIA_LEROBOT_EPISODE_HASH = "2025-11-27-23-13-31-791000"
ARIA_EMBODIMENT = "aria_bimanual"
ARIA_ACTION_HORIZON_REAL = 30
ARIA_ACTION_CHUNK_LENGTH = 100
ARIA_ACTION_STRIDE = 3
ARIA_KEYS_TO_COMPARE = (
    "observations.images.front_img_1",
    "actions_cartesian",
    "observations.state.ee_pose",
)
ARIA_IMAGE_KEYS = {"observations.images.front_img_1"}


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


def _check_equal_dict(left: dict, right: dict, path: str = "root") -> None:
    assert set(left.keys()) == set(right.keys()), (
        f"{path}: key mismatch. left_only={set(left.keys()) - set(right.keys())}, "
        f"right_only={set(right.keys()) - set(left.keys())}"
    )

    for key in left:
        left_value = left[key]
        right_value = right[key]
        key_path = f"{path}.{key}"

        if isinstance(left_value, dict) and isinstance(right_value, dict):
            _check_equal_dict(left_value, right_value, key_path)
            continue

        left_np = _to_numpy(left_value)
        right_np = _to_numpy(right_value)

        if isinstance(left_np, np.ndarray) or isinstance(right_np, np.ndarray):
            assert isinstance(left_np, np.ndarray) and isinstance(right_np, np.ndarray), (
                f"{key_path}: expected both values to be tensor/ndarray, "
                f"got {type(left_value)} vs {type(right_value)}"
            )
            assert left_np.shape == right_np.shape, (
                f"{key_path}: shape mismatch {left_np.shape} vs {right_np.shape}"
            )

            if np.issubdtype(left_np.dtype, np.floating) or np.issubdtype(
                right_np.dtype, np.floating
            ):
                np.testing.assert_allclose(
                    left_np,
                    right_np,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"{key_path}: floating values differ",
                )
            else:
                np.testing.assert_array_equal(
                    left_np, right_np, err_msg=f"{key_path}: values differ"
                )
            continue

        assert left_value == right_value, (
            f"{key_path}: value mismatch {left_value!r} vs {right_value!r}"
        )


def _build_zarr_dataset_eva() -> MultiDataset:
    key_map = {
        "observations.images.front_img_1": {"zarr_key": "images.front_1"},
        "observations.images.right_wrist_img": {"zarr_key": "images.right_wrist"},
        "observations.images.left_wrist_img": {"zarr_key": "images.left_wrist"},
        "right.obs_ee_pose": {"zarr_key": "right.obs_ee_pose"},
        "right.obs_gripper": {"zarr_key": "right.gripper"},
        "left.obs_ee_pose": {"zarr_key": "left.obs_ee_pose"},
        "left.obs_gripper": {"zarr_key": "left.gripper"},
        "right.gripper": {"zarr_key": "right.gripper", "horizon": ACTION_HORIZON_REAL},
        "left.gripper": {"zarr_key": "left.gripper", "horizon": ACTION_HORIZON_REAL},
        "right.cmd_ee_pose": {"zarr_key": "right.cmd_ee_pose", "horizon": ACTION_HORIZON_REAL},
        "left.cmd_ee_pose": {"zarr_key": "left.cmd_ee_pose", "horizon": ACTION_HORIZON_REAL},
    }

    extrinsics = EXTRINSICS["x5Dec13_2"]
    left_extrinsics_pose = _matrix_to_xyzypr(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzypr(extrinsics["right"][None, :])[0]

    transform_list = build_eva_bimanual_transform_list(
        chunk_length=ACTION_CHUNK_LENGTH,
        stride=1,
        left_extra_batch_key={"left_extrinsics_pose": left_extrinsics_pose},
        right_extra_batch_key={"right_extrinsics_pose": right_extrinsics_pose},
    )

    single_dataset = ZarrDataset(
        Episode_path=ZARR_EPISODE_PATH,
        key_map=key_map,
        transform_list=transform_list,
    )
    return MultiDataset(datasets={"single_episode": single_dataset}, mode="total")


def _build_lerobot_dataset() -> S3RLDBDataset:
    return S3RLDBDataset(
        filters={"episode_hash": LEROBOT_EPISODE_HASH},
        mode="total",
        cache_root=LEROBOT_CACHE_ROOT,
        embodiment=EMBODIMENT,
    )


def _build_zarr_dataset_aria() -> MultiDataset:
    key_map = {
        "observations.images.front_img_1": {"zarr_key": "images.front_1"},
        "left.obs_ee_pose": {"zarr_key": "left.obs_ee_pose"},
        "right.obs_ee_pose": {"zarr_key": "right.obs_ee_pose"},
        "left.action_ee_pose": {"zarr_key": "left.obs_ee_pose", "horizon": ARIA_ACTION_HORIZON_REAL},
        "right.action_ee_pose": {"zarr_key": "right.obs_ee_pose", "horizon": ARIA_ACTION_HORIZON_REAL},
        "obs_head_pose": {"zarr_key": "obs_head_pose"},
    }

    transform_list = build_aria_bimanual_transform_list(
        chunk_length=ARIA_ACTION_CHUNK_LENGTH,
        stride=ARIA_ACTION_STRIDE,
        left_action_world="left.action_ee_pose",
        right_action_world="right.action_ee_pose",
        actions_key="actions_cartesian",
        obs_key="observations.state.ee_pose",
    )

    single_dataset = ZarrDataset(
        Episode_path=ARIA_ZARR_EPISODE_PATH,
        key_map=key_map,
        transform_list=transform_list,
    )
    return MultiDataset(datasets={"single_episode": single_dataset}, mode="total")


def _build_lerobot_dataset_aria() -> S3RLDBDataset:
    return S3RLDBDataset(
        filters={"episode_hash": ARIA_LEROBOT_EPISODE_HASH},
        mode="total",
        cache_root=LEROBOT_CACHE_ROOT,
        embodiment=ARIA_EMBODIMENT,
    )


def _first_batch(dataset: torch.utils.data.Dataset) -> dict:
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return next(iter(loader))


def test_zarr_batch_matches_lerobot_batch_eva() -> None:
    if not ZARR_EPISODE_PATH.exists():
        pytest.skip(f"Zarr path not found: {ZARR_EPISODE_PATH}")

    zarr_batch = _first_batch(_build_zarr_dataset_eva())
    lerobot_batch = _first_batch(_build_lerobot_dataset())

    missing_keys = [key for key in KEYS_TO_COMPARE if key not in lerobot_batch]
    assert not missing_keys, (
        f"Lerobot batch missing keys required for comparison: {missing_keys}"
    )

    missing_zarr_keys = [key for key in KEYS_TO_COMPARE if key not in zarr_batch]
    assert not missing_zarr_keys, (
        f"Zarr batch missing keys required for comparison: {missing_zarr_keys}"
    )

    zarr_subset = {key: zarr_batch[key] for key in KEYS_TO_COMPARE}
    lerobot_subset = {key: lerobot_batch[key] for key in KEYS_TO_COMPARE}

    for key in IMAGE_KEYS:
        lerobot_arr = _to_numpy(lerobot_subset[key])
        zarr_arr = _to_numpy(zarr_subset[key])
        assert isinstance(lerobot_arr, np.ndarray) and isinstance(zarr_arr, np.ndarray), (
            f"{key}: expected array/tensor values, got {type(lerobot_subset[key])} "
            f"and {type(zarr_subset[key])}"
        )
        assert lerobot_arr.shape == zarr_arr.shape, (
            f"{key}: image shape mismatch {lerobot_arr.shape} vs {zarr_arr.shape}"
        )

    non_image_lerobot = {
        key: value for key, value in lerobot_subset.items() if key not in IMAGE_KEYS
    }
    non_image_zarr = {
        key: value for key, value in zarr_subset.items() if key not in IMAGE_KEYS
    }

    lerobot_actions = _to_numpy(non_image_lerobot.pop("actions_cartesian"))
    zarr_actions = _to_numpy(non_image_zarr.pop("actions_cartesian"))
    assert isinstance(lerobot_actions, np.ndarray) and isinstance(zarr_actions, np.ndarray), (
        "actions_cartesian must be tensors/arrays"
    )
    assert lerobot_actions.shape == zarr_actions.shape, (
        f"actions_cartesian shape mismatch: {lerobot_actions.shape} vs {zarr_actions.shape}"
    )

    np.testing.assert_allclose(
        lerobot_actions,
        zarr_actions,
        rtol=0.0,
        atol=2e-3,
        err_msg="actions_cartesian mismatch",
    )

    _check_equal_dict(non_image_lerobot, non_image_zarr)


def test_zarr_batch_matches_lerobot_batch_aria() -> None:
    if not ARIA_ZARR_EPISODE_PATH.exists():
        pytest.skip(f"Aria zarr path not found: {ARIA_ZARR_EPISODE_PATH}")

    zarr_batch = _first_batch(_build_zarr_dataset_aria())
    lerobot_batch = _first_batch(_build_lerobot_dataset_aria())

    missing_keys = [key for key in ARIA_KEYS_TO_COMPARE if key not in lerobot_batch]
    assert not missing_keys, (
        f"Lerobot Aria batch missing keys required for comparison: {missing_keys}"
    )

    missing_zarr_keys = [key for key in ARIA_KEYS_TO_COMPARE if key not in zarr_batch]
    assert not missing_zarr_keys, (
        f"Zarr Aria batch missing keys required for comparison: {missing_zarr_keys}"
    )

    zarr_subset = {key: zarr_batch[key] for key in ARIA_KEYS_TO_COMPARE}
    lerobot_subset = {key: lerobot_batch[key] for key in ARIA_KEYS_TO_COMPARE}

    for key in ARIA_IMAGE_KEYS:
        lerobot_arr = _to_numpy(lerobot_subset[key])
        zarr_arr = _to_numpy(zarr_subset[key])
        assert isinstance(lerobot_arr, np.ndarray) and isinstance(zarr_arr, np.ndarray), (
            f"{key}: expected array/tensor values, got {type(lerobot_subset[key])} "
            f"and {type(zarr_subset[key])}"
        )
        assert lerobot_arr.shape == zarr_arr.shape, (
            f"{key}: image shape mismatch {lerobot_arr.shape} vs {zarr_arr.shape}"
        )

    non_image_lerobot = {
        key: value for key, value in lerobot_subset.items() if key not in ARIA_IMAGE_KEYS
    }
    non_image_zarr = {
        key: value for key, value in zarr_subset.items() if key not in ARIA_IMAGE_KEYS
    }

    _check_equal_dict(non_image_lerobot, non_image_zarr)
