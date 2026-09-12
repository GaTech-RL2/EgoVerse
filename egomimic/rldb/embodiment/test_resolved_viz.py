"""Resolved robots and asymmetric hands share the keypoint renderer."""

from dataclasses import replace

import numpy as np

from egomimic.rldb.embodiment import Embodiment, Eva
from egomimic.rldb.embodiment.registry import KeypointSpec

K = np.array([[100, 0, 100, 0], [0, 100, 100, 0], [0, 0, 1, 0]])


def _data():
    points = np.zeros((2, 21, 3))
    points[:, :, 2] = 1
    points[0, :, 0] = -0.4
    points[1, :, 0] = 0.4
    points[:, 1, 1] = 0.4
    return points.reshape(1, 126)


def test_robot_class_exposes_keypoint_mode():
    image = np.zeros((200, 200, 3), np.uint8)
    assert Eva.viz(image, _data(), mode="keypoints", intrinsics=K).any()
    assert (
        Embodiment.resolve("dexmate_bimanual").viz(image, _data(), intrinsics=K).any()
    )


def test_per_side_masks_and_missing_side_are_respected():
    resolved = Embodiment.resolve("dexmate_bimanual")
    left = replace(
        resolved.end_effectors["left"], keypoints=KeypointSpec("mano21", (0,))
    )
    resolved = replace(
        resolved, end_effectors={"left": left, "right": resolved.end_effectors["right"]}
    )
    image = np.zeros((200, 200, 3), np.uint8)
    vis = resolved.viz(image, _data(), intrinsics=K)
    assert not vis[140, 60].any() and vis[140, 140].any()
    left_only = replace(resolved, end_effectors={"left": left}).viz(
        image, _data(), intrinsics=K
    )
    assert left_only[100, 60].any() and not left_only[:, 120:].any()


def test_jaw_resolved_spec_masks_absent_finger_slots():
    image = np.zeros((200, 200, 3), np.uint8)
    vis = Embodiment.resolve("eva_bimanual").viz(image, _data(), intrinsics=K)
    assert vis[100, 60].any()
    assert not vis[140].any()
