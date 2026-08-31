import numpy as np

from egomimic.rldb.embodiment.human import Human


def test_aria_keypoint_trajectory_uses_only_wrist_and_fingertips():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    actions = np.zeros((3, 126), dtype=np.float32)
    left = np.tile([40.0, 40.0, 1.0], (3, 21, 1))
    right = np.tile([60.0, 60.0, 1.0], (3, 21, 1))
    actions[:, :63] = left.reshape(3, -1)
    actions[:, 63:] = right.reshape(3, -1)
    intrinsics = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    baseline = Human.viz(
        image,
        actions,
        mode="keypoint_traj",
        keypoint_layout="aria",
        intrinsics=intrinsics,
        color="Reds",
    )
    ignored_joint = actions.copy()
    ignored_joint[:, 3 * 10] = [10.0, 50.0, 90.0]
    ignored = Human.viz(
        image,
        ignored_joint,
        mode="keypoint_traj",
        keypoint_layout="aria",
        intrinsics=intrinsics,
        color="Reds",
    )
    fingertip = actions.copy()
    fingertip[:, 3 * 1] = [10.0, 50.0, 90.0]
    moved = Human.viz(
        image,
        fingertip,
        mode="keypoint_traj",
        keypoint_layout="aria",
        intrinsics=intrinsics,
        color="Reds",
    )

    assert np.array_equal(baseline, ignored)
    assert not np.array_equal(baseline, moved)
