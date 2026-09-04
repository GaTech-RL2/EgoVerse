"""Test the overridable cartesian action split and the viz path that reads it."""

import numpy as np
import pytest

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human

INTRINSICS = np.array([[40.0, 0, 40, 0], [0, 40.0, 40, 0], [0, 0, 1.0, 0]])


def _image():
    return np.zeros((80, 80, 3), dtype=np.uint8)


def _ypr_actions(width: int) -> np.ndarray:
    """Return a moving ``[L xyz ypr (g), R xyz ypr (g)]`` chunk."""
    steps = np.linspace(0.0, 0.1, 5)
    actions = np.zeros((5, width))
    stride = 7 if width == 14 else 6
    for side, offset in enumerate((0, stride)):
        actions[:, offset] = 0.1 * side + steps
        actions[:, offset + 2] = 1.0
    return actions


def _quat_actions() -> np.ndarray:
    """Return the 16 wide ``[L xyz quat g, R xyz quat g]`` wrist-frame chunk."""
    steps = np.linspace(0.0, 0.1, 5)
    actions = np.zeros((5, 16))
    for side, offset in enumerate((0, 8)):
        actions[:, offset] = 0.1 * side + steps
        actions[:, offset + 2] = 1.0
        actions[:, offset + 3] = 1.0  # identity quaternion, [qw, qx, qy, qz]
    return actions


def test_the_base_split_reads_the_shared_cartesian_widths() -> None:
    for width in (12, 14):
        blocks = Eva.split_action_pose(_ypr_actions(width))
        assert [b.shape for b in blocks] == [(5, 3)] * 4


def test_the_base_split_refuses_a_width_it_cannot_interpret() -> None:
    with pytest.raises(ValueError, match="Unsupported action dim 16"):
        Human.split_action_pose(_quat_actions())


def test_eva_reads_its_own_quaternion_wrist_frame_width() -> None:
    left_xyz, left_ypr, right_xyz, right_ypr = Eva.split_action_pose(_quat_actions())

    assert left_xyz.shape == (5, 3)
    assert np.allclose(left_ypr, 0.0), "an identity quaternion is a zero rotation"
    assert np.allclose(right_xyz[:, 0] - left_xyz[:, 0], 0.1)
    assert right_ypr.shape == (5, 3)


def test_eva_still_reads_the_shared_width() -> None:
    blocks = Eva.split_action_pose(_ypr_actions(14))

    assert [b.shape for b in blocks] == [(5, 3)] * 4


def test_eva_can_visualize_its_quaternion_wrist_frame_output() -> None:
    """`cartesian_wristframe_quat` emits 16 dims, which used to raise here."""
    for mode in ("traj", "traj+rotation", "axes"):
        image = Eva.viz(
            _image(), _quat_actions(), mode=mode, intrinsics=INTRINSICS
        )
        assert image.shape == (80, 80, 3)


def test_an_embodiment_that_declines_a_width_still_says_so_through_viz() -> None:
    with pytest.raises(ValueError, match="Unsupported action dim 16"):
        Human.viz(_image(), _quat_actions(), mode="traj", intrinsics=INTRINSICS)


def test_the_viz_path_calls_the_subclass_split() -> None:
    calls = []

    class _Counting(Eva):
        @classmethod
        def split_action_pose(cls, actions):
            calls.append(actions.shape[-1])
            return super().split_action_pose(actions)

    _Counting.viz(_image(), _quat_actions(), mode="axes", intrinsics=INTRINSICS)

    assert calls == [16]


def test_a_new_platform_overrides_only_the_split() -> None:
    """A layout the shared function cannot guess costs one classmethod."""

    class _SixteenWide(Eva):
        @classmethod
        def split_action_pose(cls, actions):
            # [L xyz ypr, L aux(2), R xyz ypr, R aux(2)]
            left, right = actions[..., :8], actions[..., 8:]
            return left[..., :3], left[..., 3:6], right[..., :3], right[..., 3:6]

    actions = np.zeros((5, 16))
    actions[:, 2] = actions[:, 10] = 1.0

    assert _SixteenWide.viz(
        _image(), actions, mode="traj", intrinsics=INTRINSICS
    ).shape == (80, 80, 3)
