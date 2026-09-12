"""View-specific coverage and the displayed-frame camera convention."""

import numpy as np
import zarr

from egomimic.rldb.zarr.camera_coverage import camera_coverage, camera_coverage_report
from egomimic.rldb.zarr.overlay import keypoint_chunk
from egomimic.rldb.zarr.test_render_check import overlay_episode as overlay_episode


def test_moving_camera_is_used_once_for_the_entire_horizon(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    head = np.asarray(group["obs_head_pose"][:])
    head[:, 0] = np.arange(len(head)) * 0.1
    group["obs_head_pose"][:] = head
    first = keypoint_chunk(group, 0, 3)[2]
    second = keypoint_chunk(group, 1, 3)[2]
    np.testing.assert_allclose(first[0], first[2])
    np.testing.assert_allclose(second[:, 0], first[:, 0] - 0.1)
    head[1:, 0] += 10
    group["obs_head_pose"][:] = head
    np.testing.assert_allclose(keypoint_chunk(group, 0, 3)[2], first)


def test_ego_coverage_does_not_require_wrist_calibration(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    # A numeric image array is sufficient for coverage inspection.
    group.create_array("images.left_wrist", data=np.zeros((4, 1)))
    report = camera_coverage_report(group)
    assert report["front_1"]["available"]
    assert not report["left_wrist"]["available"]
    assert (
        report["left_wrist"]["pose_source"] != "obs_head_pose at displayed observation"
    )


def test_missing_trajectory_blocks_only_the_transform_that_needs_it(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs["embodiment"] = "dexmate_bimanual"
    group.attrs["calibration"] = {
        "reference_frame": "robot_base",
        "cameras": {"front_1": {"K": group.attrs["intrinsics"]["front_1"]}},
    }
    del group["obs_head_pose"]
    assert not camera_coverage(group).available
    calibration = dict(group.attrs["calibration"])
    calibration["cameras"]["front_1"]["ref_T_cam"] = np.eye(4).tolist()
    group.attrs["calibration"] = calibration
    assert camera_coverage(group).available


def test_unsupported_optics_and_bad_timeline_are_explicit(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs["calibration"] = {
        "reference_frame": "slam_world",
        "cameras": {
            "front_1": {
                "K": group.attrs["intrinsics"]["front_1"],
                "model": "KANNALA_BRANDT",
                "distortion": [0, 0, 0, 0],
                "rectified": False,
            }
        },
    }
    assert "unrectified" in " ".join(camera_coverage(group).missing)
    del group.attrs["calibration"]
    group["obs_head_pose"].resize((3, 7))
    assert "timeline" in " ".join(camera_coverage(group).missing)


def test_invalid_quaternions_and_resolution_are_not_silently_projected(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group["obs_head_pose"][0] = np.zeros(7)
    assert not camera_coverage(group).available
    group.attrs["calibration"] = {
        "reference_frame": "slam_world",
        "cameras": {
            "front_1": {
                "K": group.attrs["intrinsics"]["front_1"],
                "resolution": [640, 480],
            }
        },
    }
    assert "resolution" in " ".join(
        camera_coverage(group, frame=1, image_shape=(64, 64, 3)).missing
    )


def test_bad_wrist_metadata_does_not_gate_an_ego_overlay(overlay_episode):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs["calibration"] = {
        "reference_frame": "slam_world",
        "cameras": {
            "front_1": {"K": group.attrs["intrinsics"]["front_1"]},
            "left_wrist": {"K": "missing"},
        },
    }
    assert camera_coverage(group).available
    assert not camera_coverage(group, "left_wrist").available


def test_legacy_aria_fallback_is_limited_to_known_rectified_image_sizes(
    overlay_episode,
):
    group = zarr.open_group(overlay_episode, mode="a")
    group.attrs.pop("intrinsics")
    group.attrs["embodiment"] = "aria_bimanual"
    coverage = camera_coverage(group, image_shape=(480, 640, 3))
    assert coverage.available
    assert "Aria intrinsics fallback" in " ".join(coverage.limitations)
    half = camera_coverage(group, image_shape=(240, 320, 3))
    np.testing.assert_allclose(half.K[:2], coverage.K[:2] / 2)
    assert not camera_coverage(group, image_shape=(360, 640, 3)).available
    group.attrs["embodiment"] = "human_bimanual"
    assert not camera_coverage(group, image_shape=(480, 640, 3)).available


def test_inspector_uses_displayed_camera_when_reference_and_ego_are_different(overlay_episode):
    from egomimic.rldb.zarr.overlay import decode_frame, render_keypoints
    from egomimic.scripts.data_visualization.inspector_lib.dataset_view import (
        _draw_overlay,
    )

    group = zarr.open_group(overlay_episode, mode="a")
    image = decode_frame(group, 0)
    group.create_array("images.left_wrist", data=np.tile(image, (4, 1, 1, 1)))
    group.attrs["calibration"] = {
        "reference_frame": "camera:left_wrist",
        "cameras": {
            "front_1": {"K": group.attrs["intrinsics"]["front_1"]},
            "left_wrist": {"K": [[20, 0, 32, 0], [0, 20, 32, 0], [0, 0, 1, 0]]},
        },
    }
    group["obs_head_pose"][0, 0] = 0.2
    expected, _ = render_keypoints(group, 0, image=image, camera="left_wrist", horizon=3)
    actual, ok, _ = _draw_overlay(image, group, 0, "keypoint", horizon=3, camera="left_wrist")
    assert ok
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, render_keypoints(group, 0, image=image, horizon=3)[0])
