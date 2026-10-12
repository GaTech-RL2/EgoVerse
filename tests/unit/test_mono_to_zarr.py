"""Mono hand-pose tracks become the mecka zarr layout."""

import sys
import types
from importlib.machinery import ModuleSpec

import numpy as np
from scipy.spatial.transform import Rotation


def _stub_cv2() -> None:
    if "cv2" in sys.modules:
        return
    try:
        import cv2  # noqa: F401
    except Exception:
        stub = types.ModuleType("cv2")
        stub.__spec__ = ModuleSpec("cv2", loader=None)
        sys.modules["cv2"] = stub


_stub_cv2()

from egomimic.scripts.data_download.mono_to_zarr import (  # noqa: E402
    head_pose_from_camera,
    intrinsics_matrix,
    select_shard,
    tracks_from_hand_pose,
)
from egomimic.scripts.mecka_process.mecka_to_zarr import (  # noqa: E402
    MeckaExtractor,
)


def _hand(side: str, origin: tuple[float, float, float]) -> dict:
    landmarks = []
    for index in range(21):
        landmarks.append(
            {
                "landmark_index": index,
                "x": origin[0] + 0.01 * (index % 5),
                "y": origin[1] + 0.01 * (index % 3),
                "z": origin[2] + 0.005 * (index % 4),
            }
        )
    return {"hand_type": side, "landmarks": landmarks}


def _pose() -> dict:
    quat = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_quat()
    return {
        "total_frames": 2,
        "frames": {
            "0": {
                "camera_transform": {
                    "position": {"x": 0.2, "y": 0.0, "z": 0.6},
                    "orientation": {
                        "x": float(quat[0]),
                        "y": float(quat[1]),
                        "z": float(quat[2]),
                        "w": float(quat[3]),
                    },
                },
                "hands": [
                    _hand("LEFT", (0.4, 0.1, 0.1)),
                    _hand("RIGHT", (0.1, 0.2, 0.15)),
                ],
            },
            "1": {
                "camera_transform": {
                    "position": {"x": 0.2, "y": 0.0, "z": 0.6},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "hands": [],
            },
        },
    }


def test_tracks_keep_left_and_right_and_fill_a_missing_frame():
    tracks = tracks_from_hand_pose(_pose())
    assert tracks["left.obs_keypoints"].shape == (2, 63)
    assert tracks["right.obs_wrist_pose"].shape == (2, 7)
    assert tracks["left.obs_keypoints"][0, 0] == np.float32(0.4)
    assert tracks["right.obs_keypoints"][0, 0] == np.float32(0.1)
    assert np.all(tracks["left.obs_keypoints"][1] == 0)
    # A frame with no hands still gets an identity quaternion, not a zero quat.
    assert tracks["left.obs_wrist_pose"][1, 3] == np.float32(1.0)
    assert abs(float(np.linalg.norm(tracks["left.obs_ee_pose"][0, 3:]))) - 1 < 1e-5


def test_head_pose_matches_the_mecka_axis_remap():
    quat = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_quat()
    position = {"x": 0.2, "y": -0.1, "z": 0.55}
    orientation = {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quat[2]),
        "w": float(quat[3]),
    }
    got = head_pose_from_camera(position, orientation)
    egomotion = np.zeros((1, 11), dtype=np.float64)
    egomotion[0, 1:4] = [0.2, -0.1, 0.55]
    egomotion[0, 7:11] = quat
    expected = MeckaExtractor._extract_head_poses(egomotion)[0]
    assert np.allclose(got, expected, atol=1e-5)


def test_shards_cover_every_episode_once():
    ids = [f"ep{i}" for i in range(10)]
    covered = []
    for shard in range(4):
        covered.extend(select_shard(ids, shard, 4))
    assert sorted(covered) == sorted(ids)
    assert len(covered) == len(ids)
    assert select_shard(ids, 0, 1) == ids


def test_intrinsics_scale_with_the_resized_frame():
    matrix = intrinsics_matrix(
        {"w": 1920, "h": 1080, "fl_x": 760.0, "fl_y": 759.0, "cx": 960.0, "cy": 540.0},
        640,
        360,
    )
    assert matrix.shape == (3, 4)
    assert matrix[0, 0] == 760.0 / 3
    assert matrix[1, 2] == 180.0
    assert matrix[2, 2] == 1.0
