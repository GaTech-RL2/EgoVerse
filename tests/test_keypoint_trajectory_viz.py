import numpy as np
import pytest

from egomimic.rldb.embodiment.human import Aria
from egomimic.utils.viz_utils import _split_hand_keypoint_chunks


def _action_chunk(horizon=100):
    actions = np.zeros((horizon, 126), dtype=np.float32)
    for hand_offset in (0, 63):
        for keypoint in range(21):
            actions[:, hand_offset + 3 * keypoint] = np.linspace(-0.08, 0.08, horizon)
            actions[:, hand_offset + 3 * keypoint + 1] = (keypoint - 10) * 0.002
            actions[:, hand_offset + 3 * keypoint + 2] = 0.5
    return actions


def test_split_hand_keypoint_chunks_preserves_horizon():
    chunks = _split_hand_keypoint_chunks(_action_chunk())

    assert chunks["left"].shape == (100, 21, 3)
    assert chunks["right"].shape == (100, 21, 3)


def test_keypoint_trajectory_defaults_to_first_20_steps(monkeypatch):
    projected_lengths = []

    def record_projection(points, _intrinsics):
        projected_lengths.append(len(points))
        return np.column_stack((points[:, :2] * 100 + 32, points[:, 2]))

    monkeypatch.setattr(
        "egomimic.utils.viz_utils.cam_frame_to_cam_pixels", record_projection
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    rendered = Aria.viz(
        image,
        _action_chunk(),
        mode="keypoint_traj",
        intrinsics=np.eye(3),
    )

    assert rendered.shape == image.shape
    assert projected_lengths == [20] * 12
    assert np.any(rendered)


def test_keypoint_trajectory_rejects_invalid_step_count():
    with pytest.raises(ValueError, match="must be positive"):
        Aria.viz(
            np.zeros((64, 64, 3), dtype=np.uint8),
            _action_chunk(),
            mode="keypoint_traj",
            intrinsics=np.eye(3),
            keypoint_traj_steps=0,
        )


def test_clipped_trajectory_remains_visible(monkeypatch):
    def project_offscreen_path(points, _intrinsics):
        x = np.linspace(-20, 84, len(points))
        return np.column_stack((x, np.full(len(points), 32), points[:, 2]))

    monkeypatch.setattr(
        "egomimic.utils.viz_utils.cam_frame_to_cam_pixels", project_offscreen_path
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    rendered = Aria.viz(
        image,
        _action_chunk(),
        mode="keypoint_traj",
        intrinsics=np.eye(3),
        keypoint_traj_steps=100,
    )

    assert np.any(rendered[:, 0])
    assert np.any(rendered[:, -1])
