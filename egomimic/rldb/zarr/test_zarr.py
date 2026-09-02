import numpy as np
import torch

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva import Eva, _build_eva_bimanual_transform_list
from egomimic.rldb.embodiment.human import (
    _build_human_cartesian_bimanual_transform_list,
)
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter


def _pose_sequence(length: int, *, x_offset: float = 0.0) -> np.ndarray:
    poses = np.zeros((length, 7), dtype=np.float64)
    poses[:, 0] = x_offset + np.arange(length, dtype=np.float64) * 0.01
    poses[:, 3] = 1.0
    return poses


def _camera_intrinsics() -> dict[str, np.ndarray]:
    return {
        "front_1": np.array(
            [
                [200.0, 0.0, 160.0, 0.0],
                [0.0, 200.0, 120.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    }


def test_eva_zarr_dataset_runs_current_transform_pipeline(tmp_path) -> None:
    length = 5
    episode_path = tmp_path / "eva_episode.zarr"
    left_cmd_gripper = np.linspace(0.0, 1.0, length, dtype=np.float64)[:, None]
    right_cmd_gripper = np.linspace(1.0, 0.0, length, dtype=np.float64)[:, None]

    ZarrWriter.create_and_write(
        episode_path=episode_path,
        numeric_data={
            "left.obs_ee_pose": _pose_sequence(length),
            "right.obs_ee_pose": _pose_sequence(length, x_offset=0.2),
            "left.obs_gripper": np.full((length, 1), 0.25, dtype=np.float64),
            "right.obs_gripper": np.full((length, 1), 0.75, dtype=np.float64),
            "left.cmd_ee_pose": _pose_sequence(length, x_offset=0.1),
            "right.cmd_ee_pose": _pose_sequence(length, x_offset=0.3),
            "left.cmd_gripper": left_cmd_gripper,
            "right.cmd_gripper": right_cmd_gripper,
        },
        embodiment="eva_bimanual",
        intrinsics=_camera_intrinsics(),
        extrinsics=Eva.EXTRINSICS,
        chunk_timesteps=4,
    )

    key_map = {
        "left.obs_ee_pose": {"zarr_key": "left.obs_ee_pose"},
        "right.obs_ee_pose": {"zarr_key": "right.obs_ee_pose"},
        "left.obs_gripper": {"zarr_key": "left.obs_gripper"},
        "right.obs_gripper": {"zarr_key": "right.obs_gripper"},
        "left.cmd_ee_pose": {"zarr_key": "left.cmd_ee_pose", "horizon": 4},
        "right.cmd_ee_pose": {"zarr_key": "right.cmd_ee_pose", "horizon": 4},
        "left.cmd_gripper": {"zarr_key": "left.cmd_gripper", "horizon": 4},
        "right.cmd_gripper": {"zarr_key": "right.cmd_gripper", "horizon": 4},
    }
    leaf = ZarrDataset(
        Episode_path=episode_path,
        key_map=key_map,
        transform_list=_build_eva_bimanual_transform_list(
            chunk_length=6, stride=1, is_quat=True
        ),
    )
    dataset = MultiDataset(datasets={"eva_episode": leaf}, mode="total")

    sample = dataset[1]

    for arm, base_T_cam in Eva.EXTRINSICS.items():
        np.testing.assert_allclose(
            np.asarray(leaf.metadata["extrinsics"][arm]), base_T_cam
        )
    assert sample["actions_cartesian"].shape == (6, 14)
    assert sample["observations.state.ee_pose"].shape == (14,)
    assert sample["embodiment"] == get_embodiment_id("eva_bimanual")
    assert sample["episode_hash"] == "eva_episode"
    assert torch.isfinite(sample["actions_cartesian"]).all()
    np.testing.assert_allclose(
        sample["intrinsics"].numpy(), _camera_intrinsics()["front_1"]
    )
    np.testing.assert_allclose(
        sample["actions_cartesian"][[0, -1], 6].numpy(),
        left_cmd_gripper[[1, 4], 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample["actions_cartesian"][[0, -1], 13].numpy(),
        right_cmd_gripper[[1, 4], 0],
        atol=1e-6,
    )


def test_human_zarr_dataset_runs_current_transform_pipeline(tmp_path) -> None:
    length = 4
    episode_path = tmp_path / "human_episode.zarr"
    left_poses = _pose_sequence(length, x_offset=0.1)
    right_poses = _pose_sequence(length, x_offset=0.4)
    head_poses = np.zeros((length, 7), dtype=np.float64)
    head_poses[:, 3] = 1.0

    ZarrWriter.create_and_write(
        episode_path=episode_path,
        numeric_data={
            "left.obs_ee_pose": left_poses,
            "right.obs_ee_pose": right_poses,
            "obs_head_pose": head_poses,
        },
        embodiment="human_bimanual",
        intrinsics=_camera_intrinsics(),
        extrinsics=None,
        chunk_timesteps=4,
    )

    key_map = {
        "left.action_ee_pose": {
            "zarr_key": "left.obs_ee_pose",
            "horizon": length,
        },
        "right.action_ee_pose": {
            "zarr_key": "right.obs_ee_pose",
            "horizon": length,
        },
        "left.obs_ee_pose": {"zarr_key": "left.obs_ee_pose"},
        "right.obs_ee_pose": {"zarr_key": "right.obs_ee_pose"},
        "obs_head_pose": {"zarr_key": "obs_head_pose"},
    }
    dataset = ZarrDataset(
        Episode_path=episode_path,
        key_map=key_map,
        transform_list=_build_human_cartesian_bimanual_transform_list(
            chunk_length=6,
            stride=1,
            target_world_is_quat=True,
        ),
    )

    sample = dataset[0]

    assert sample["actions_cartesian"].shape == (6, 12)
    assert sample["observations.state.ee_pose"].shape == (12,)
    assert sample["embodiment"] == get_embodiment_id("human_bimanual")
    assert torch.isfinite(sample["actions_cartesian"]).all()
    np.testing.assert_allclose(
        sample["observations.state.ee_pose"][[0, 6]].numpy(),
        np.array([left_poses[0, 0], right_poses[0, 0]]),
        atol=1e-6,
    )
