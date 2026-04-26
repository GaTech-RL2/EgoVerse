from pathlib import Path

import cv2
import numpy as np
import torch
import zarr
from numcodecs import VLenBytes

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.so100 import So100SingleArm
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset


def _encode_jpeg(frame_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    assert ok
    return encoded.tobytes()


def _write_so100_zarr(root: Path) -> Path:
    episode_path = root / "so100_episode_000000.zarr"
    store = zarr.open_group(str(episode_path), mode="w")

    obs = np.array(
        [
            [0.10, 0.00, 0.30, 0.0, 0.0, 0.00, 1.0],
            [0.11, 0.01, 0.31, 0.0, 0.0, 0.05, 2.0],
            [0.12, 0.02, 0.32, 0.0, 0.0, 0.10, 3.0],
            [0.13, 0.03, 0.33, 0.0, 0.0, 0.15, 4.0],
            [0.14, 0.04, 0.34, 0.0, 0.0, 0.20, 5.0],
        ],
        dtype=np.float32,
    )
    cmd = obs + np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.05, 10.0], dtype=np.float32)
    store.create_dataset("obs_ee_pose_cam_rotvec", data=obs, chunks=(5, 7))
    store.create_dataset("cmd_ee_pose_cam_rotvec", data=cmd, chunks=(5, 7))

    encoded = np.empty((5,), dtype=object)
    for idx in range(5):
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[..., 0] = idx * 20
        frame[..., 1] = 40
        encoded[idx] = _encode_jpeg(frame)
    store.create_dataset(
        "images.front_1",
        shape=(5,),
        chunks=(1,),
        dtype=object,
        object_codec=VLenBytes(),
        fill_value=None,
    )
    store["images.front_1"][:] = encoded

    store.attrs.update(
        {
            "embodiment": "so100_singlearm",
            "robot_name": "so100_singlearm",
            "total_frames": 5,
            "fps": 30,
            "features": {
                "obs_ee_pose_cam_rotvec": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"],
                },
                "cmd_ee_pose_cam_rotvec": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"],
                },
                "images.front_1": {
                    "dtype": "jpeg",
                    "shape": [12, 16, 3],
                    "names": ["height", "width", "channel"],
                },
            },
        }
    )
    return episode_path


def test_so100_zarr_loads_and_emits_future_chunk(tmp_path: Path) -> None:
    _write_so100_zarr(tmp_path)
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=So100SingleArm.get_keymap(mode="camera_frame_ypr"),
        transform_list=So100SingleArm.get_transform_list(
            mode="camera_frame_ypr",
            chunk_length=64,
        ),
    )
    dataset = MultiDataset._from_resolver(
        resolver=resolver,
        filters=DatasetFilter(),
        mode="total",
        valid_ratio=0.0,
    )

    sample = dataset[0]

    assert tuple(sample["observations.images.front_img_1"].shape) == (3, 12, 16)
    assert tuple(sample["observations.state.ee_pose"].shape) == (7,)
    assert tuple(sample["actions_cartesian"].shape) == (64, 7)
    assert int(sample["embodiment"]) == get_embodiment_id("so100_singlearm")
    assert torch.isfinite(sample["observations.state.ee_pose"]).all()
    assert torch.isfinite(sample["actions_cartesian"]).all()

    np.testing.assert_allclose(
        sample["observations.state.ee_pose"].numpy(),
        np.array([0.10, 0.00, 0.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sample["actions_cartesian"][0].numpy(),
        np.array([0.60, 0.00, 0.30, 0.05, 0.0, 0.0, 11.0], dtype=np.float32),
        atol=1e-6,
    )
    assert sample["actions_cartesian"][-1, 0] > sample["actions_cartesian"][0, 0]
