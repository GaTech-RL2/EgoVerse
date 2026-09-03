"""Test the ``calibration`` attribute block and the legacy read shim."""

import numpy as np
import pytest

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.calibration import (
    Calibration,
    CalibrationError,
    CameraCalibration,
    camera_name,
    lift_legacy_calibration,
    parse_calibration,
    read_calibration,
    uncalibrated_cameras,
)
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

K_FRONT = np.array(
    [
        [200.0, 0.0, 160.0, 0.0],
        [0.0, 200.0, 120.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


def _calibration_block() -> dict:
    return {
        "reference_frame": "robot_base",
        "cameras": {
            "front_1": {
                "K": K_FRONT.tolist(),
                "resolution": [320, 240],
                "rectified": True,
                "ref_T_cam": np.eye(4).tolist(),
            },
            "left_wrist": {"K": K_FRONT.tolist(), "rectified": False},
        },
        "arm_bases": {"left": np.eye(4).tolist()},
    }


def test_parse_round_trips_through_jsonable() -> None:
    calibration = parse_calibration(_calibration_block())
    assert calibration.reference_frame == "robot_base"
    assert calibration.reference_camera is None
    assert calibration.cameras["front_1"].resolution == (320, 240)
    assert calibration.cameras["left_wrist"].rectified is False
    reparsed = parse_calibration(calibration.to_jsonable())
    assert reparsed.reference_frame == calibration.reference_frame
    assert set(reparsed.cameras) == set(calibration.cameras)
    for name, camera in calibration.cameras.items():
        other = reparsed.cameras[name]
        assert (other.resolution, other.rectified) == (
            camera.resolution,
            camera.rectified,
        )
        np.testing.assert_array_equal(other.K, camera.K)
    np.testing.assert_array_equal(
        reparsed.arm_bases["left"], calibration.arm_bases["left"]
    )


def test_default_camera_prefers_the_front_camera() -> None:
    calibration = parse_calibration(
        {
            "reference_frame": "robot_base",
            "cameras": {
                "left_wrist": {"K": K_FRONT.tolist()},
                "front_1": {"K": (2 * K_FRONT).tolist()},
            },
        }
    )
    assert calibration.default_camera() == "front_1"
    np.testing.assert_allclose(calibration.K(), 2 * K_FRONT)


def test_reference_camera_pose_is_the_identity() -> None:
    calibration = parse_calibration(
        {
            "reference_frame": "camera:front_1",
            "cameras": {"front_1": {"K": K_FRONT.tolist()}},
        }
    )
    np.testing.assert_allclose(calibration.ref_T_cam("front_1"), np.eye(4))
    assert calibration.ref_T_cam("left_wrist") is None


@pytest.mark.parametrize(
    "block, message",
    [
        ({"cameras": {"front_1": {}}}, "reference_frame"),
        ({"reference_frame": "robot_base", "cameras": {}}, "non-empty mapping"),
        (
            {"reference_frame": "camera:missing", "cameras": {"front_1": {}}},
            "does not declare",
        ),
        (
            {"reference_frame": "elbow", "cameras": {"front_1": {}}},
            "unknown reference_frame",
        ),
        (
            {"reference_frame": "robot_base", "cameras": {"front_1": {"fx": 1}}},
            "unknown field",
        ),
        (
            {
                "reference_frame": "robot_base",
                "cameras": {"front_1": {"K": [[1.0, 2.0], [3.0, 4.0]]}},
            },
            "3x4 camera matrix",
        ),
        (
            {
                "reference_frame": "robot_base",
                "cameras": {"front_1": {}},
                "arm_bases": {"left": np.eye(3).tolist()},
            },
            r"expected shape \(4, 4\)",
        ),
    ],
)
def test_parse_rejects_malformed_blocks(block, message) -> None:
    with pytest.raises(CalibrationError, match=message):
        parse_calibration(block)


def test_a_bare_3x3_camera_matrix_is_padded() -> None:
    calibration = parse_calibration(
        {
            "reference_frame": "robot_base",
            "cameras": {"front_1": {"K": K_FRONT[:, :3].tolist()}},
        }
    )
    np.testing.assert_allclose(calibration.K("front_1"), K_FRONT)


def test_legacy_attributes_lift_without_changing_base_T_cam() -> None:
    calibration = lift_legacy_calibration(
        intrinsics={"front_1": K_FRONT}, extrinsics=Eva.EXTRINSICS
    )
    assert calibration.legacy
    assert calibration.reference_frame == "camera:front_1"
    np.testing.assert_allclose(calibration.K(), K_FRONT)
    for side, base_T_cam in Eva.EXTRINSICS.items():
        # `arm_bases` holds ref_T_armbase, the inverse of the stored matrix.
        np.testing.assert_allclose(
            calibration.arm_bases[side], np.linalg.inv(base_T_cam), atol=1e-12
        )
        np.testing.assert_allclose(
            calibration.base_T_cam(side), base_T_cam, atol=1e-12
        )


def test_legacy_extrinsics_without_intrinsics_still_compose() -> None:
    calibration = lift_legacy_calibration(extrinsics=Eva.EXTRINSICS)
    assert calibration.K() is None
    np.testing.assert_allclose(
        calibration.base_T_cam("left"), Eva.EXTRINSICS["left"], atol=1e-12
    )


def test_read_calibration_prefers_the_block_over_the_legacy_pair() -> None:
    attrs = {
        "calibration": _calibration_block(),
        "intrinsics": {"front_1": (7 * K_FRONT).tolist()},
    }
    calibration = read_calibration(attrs)
    assert calibration is not None and not calibration.legacy
    np.testing.assert_allclose(calibration.K(), K_FRONT)
    assert read_calibration({}) is None


def test_camera_name_reads_the_stream_out_of_an_array_key() -> None:
    assert camera_name("images.front_1") == "front_1"
    assert camera_name("observations.images.left_wrist") == "left_wrist"
    assert camera_name("left.obs_ee_pose") is None
    assert camera_name("images.") is None


def test_uncalibrated_cameras_lists_streams_without_a_matrix() -> None:
    calibration = Calibration(
        reference_frame="robot_base",
        cameras={
            "front_1": CameraCalibration(name="front_1", K=K_FRONT),
            "left_wrist": CameraCalibration(name="left_wrist"),
        },
    )
    keys = ["images.front_1", "images.left_wrist", "images.right_wrist"]
    assert uncalibrated_cameras(keys, calibration) == ["left_wrist", "right_wrist"]
    assert uncalibrated_cameras(keys, None) == [
        "front_1",
        "left_wrist",
        "right_wrist",
    ]
    assert uncalibrated_cameras(["left.obs_gripper"], None) == []


def test_coverage_is_a_warning_by_default_and_an_error_under_strict(
    tmp_path, caplog
) -> None:
    images = {
        "images.front_1": np.zeros((4, 8, 8, 3), dtype=np.uint8),
        "images.left_wrist": np.zeros((4, 8, 8, 3), dtype=np.uint8),
    }
    with caplog.at_level("WARNING"):
        _write_episode(
            tmp_path / "partial.zarr",
            image_data=images,
            intrinsics={"front_1": K_FRONT},
        )
    assert "left_wrist" in caplog.text

    with pytest.raises(ValueError, match=r"no camera matrix.*left_wrist"):
        _write_episode(
            tmp_path / "strict.zarr",
            image_data=images,
            intrinsics={"front_1": K_FRONT},
            strict=True,
        )


def test_full_coverage_passes_under_strict(tmp_path, caplog) -> None:
    with caplog.at_level("WARNING"):
        _write_episode(
            tmp_path / "covered.zarr",
            image_data={"images.front_1": np.zeros((4, 8, 8, 3), dtype=np.uint8)},
            calibration={
                "reference_frame": "camera:front_1",
                "cameras": {"front_1": {"K": K_FRONT.tolist()}},
            },
            strict=True,
        )
    assert "no camera matrix" not in caplog.text


def _write_episode(episode_path, **kwargs) -> None:
    ZarrWriter.create_and_write(
        episode_path=episode_path,
        numeric_data={"left.obs_gripper": np.zeros((4, 1))},
        embodiment="eva_bimanual",
        chunk_timesteps=4,
        **kwargs,
    )


def test_the_writer_stores_the_block_and_the_legacy_pair(tmp_path) -> None:
    episode_path = tmp_path / "calibrated.zarr"
    _write_episode(episode_path, calibration=_calibration_block())

    episode = ZarrEpisode(episode_path)
    assert episode.metadata["calibration"]["reference_frame"] == "robot_base"
    # Readers that predate the block still find the pair they expect.
    np.testing.assert_allclose(
        np.asarray(episode.metadata["intrinsics"]["front_1"]), K_FRONT
    )
    np.testing.assert_allclose(
        np.asarray(episode.metadata["extrinsics"]["left"]), np.eye(4)
    )
    np.testing.assert_allclose(episode.calibration.K(), K_FRONT)


def test_a_legacy_episode_reads_through_the_shim(tmp_path) -> None:
    episode_path = tmp_path / "legacy.zarr"
    _write_episode(
        episode_path, intrinsics={"front_1": K_FRONT}, extrinsics=Eva.EXTRINSICS
    )

    episode = ZarrEpisode(episode_path)
    assert "calibration" not in episode.metadata
    calibration = episode.calibration
    assert calibration is not None and calibration.legacy
    np.testing.assert_allclose(calibration.K(), K_FRONT)
    np.testing.assert_allclose(
        calibration.base_T_cam("right"), Eva.EXTRINSICS["right"], atol=1e-12
    )


def test_the_writer_rejects_an_episode_with_no_camera_matrix(tmp_path) -> None:
    with pytest.raises(ValueError, match="Camera intrinsics"):
        _write_episode(tmp_path / "blank.zarr")
