"""Test the schema-driven episode validator."""

import numpy as np
import pytest

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.validate import (
    ERROR,
    OK,
    WARNING,
    load_schema,
    main,
    validate_episode,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

K = np.array([[200.0, 0.0, 160.0, 0.0], [0.0, 200.0, 120.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
LENGTH = 4


def _levels(report) -> dict[str, str]:
    return {f.check: f.level for f in report.findings}


def _poses(x_offset: float = 0.0) -> np.ndarray:
    """Return a moving ``[x, y, z, qw, qx, qy, qz]`` pose track."""
    angles = np.linspace(0.1, 0.4, LENGTH)
    poses = np.zeros((LENGTH, 7))
    poses[:, 0] = x_offset + np.arange(LENGTH) * 0.01
    poses[:, 3] = np.cos(angles / 2)
    poses[:, 6] = np.sin(angles / 2)
    return poses


def _eva_numeric() -> dict:
    data = {}
    for side, offset in (("left", 0.0), ("right", 0.2)):
        data[f"{side}.obs_ee_pose"] = _poses(offset)
        data[f"{side}.cmd_ee_pose"] = _poses(offset + 0.05)
        data[f"{side}.obs_joints"] = np.zeros((LENGTH, 6))
        data[f"{side}.cmd_joints"] = np.zeros((LENGTH, 6))
        data[f"{side}.obs_gripper"] = np.zeros((LENGTH, 1))
        data[f"{side}.cmd_gripper"] = np.zeros((LENGTH, 1))
    data["obs_rgb_timestamps_ns"] = np.arange(LENGTH, dtype=np.int64)
    return data


def _write_eva(path, *, numeric=None, images=None, **kwargs) -> None:
    kwargs.setdefault("calibration", {
        "reference_frame": "camera:front_1",
        "cameras": {"front_1": {"K": K.tolist()}},
        "arm_bases": {
            side: np.linalg.inv(T).tolist() for side, T in Eva.EXTRINSICS.items()
        },
    })
    kwargs.setdefault("metadata_override", {"schema_version": "v3.1"})
    kwargs.setdefault("annotations", [("fold the towel", 0, LENGTH)])
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data=_eva_numeric() if numeric is None else numeric,
        image_data=images,
        embodiment=kwargs.pop("embodiment", "eva_bimanual"),
        chunk_timesteps=LENGTH,
        **kwargs,
    )


def test_a_complete_eva_episode_passes_under_strict(tmp_path) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path, images={"images.front_1": np.zeros((LENGTH, 8, 8, 3), np.uint8)})

    report = validate_episode(path, strict=True)

    assert report.ok, report.text()
    assert not report.warnings
    assert _levels(report)["embodiment"] == OK


def test_a_missing_required_array_is_an_error(tmp_path) -> None:
    numeric = _eva_numeric()
    del numeric["right.obs_ee_pose"]
    _write_eva(tmp_path / "eva.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "eva.zarr")

    assert _levels(report)["right.obs_ee_pose"] == ERROR
    assert not report.ok


def test_a_wrong_width_is_reported_with_the_dimension_that_set_it(tmp_path) -> None:
    numeric = _eva_numeric()
    numeric["left.obs_joints"] = np.zeros((LENGTH, 5))
    _write_eva(tmp_path / "eva.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "eva.zarr")

    message = next(
        f.message for f in report.findings if f.check == "left.obs_joints"
    )
    assert "axis 1 is 5, expected 6 (arm_dof)" in message


def test_a_padded_tail_is_not_an_error(tmp_path) -> None:
    path = tmp_path / "eva.zarr"
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data=_eva_numeric(),
        embodiment="eva_bimanual",
        chunk_timesteps=3,  # 4 frames pad out to 6
        annotations=[("fold the towel", 0, LENGTH)],
        intrinsics={"front_1": K},
        extrinsics=Eva.EXTRINSICS,
    )

    report = validate_episode(path)

    assert _levels(report)["left.obs_ee_pose"] == OK
    assert report.ok, report.text()


def test_an_array_shorter_than_total_frames_is_an_error(tmp_path) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path)
    store = __import__("zarr").open_group(str(path), mode="a")
    store.attrs["total_frames"] = LENGTH + 10

    report = validate_episode(path)

    assert "holds 4 frames for total_frames 14" in report.text()


def test_strict_promotes_the_rules_the_corpus_does_not_meet_yet(tmp_path) -> None:
    path = tmp_path / "legacy.zarr"
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data=_eva_numeric(),
        image_data={
            "images.front_1": np.zeros((LENGTH, 8, 8, 3), np.uint8),
            "images.left_wrist": np.zeros((LENGTH, 8, 8, 3), np.uint8),
        },
        embodiment="eva_bimanual",
        chunk_timesteps=LENGTH,
        annotations=[("fold the towel", 0, LENGTH)],
        intrinsics={"front_1": K},
        extrinsics=Eva.EXTRINSICS,
    )

    lenient = validate_episode(path)
    strict = validate_episode(path, strict=True)

    assert lenient.ok
    assert _levels(lenient)["camera_coverage"] == WARNING
    assert _levels(lenient)["attrs.calibration"] == WARNING
    assert not strict.ok
    assert _levels(strict)["camera_coverage"] == ERROR
    assert "left_wrist" in strict.text()


def test_a_single_arm_episode_owes_only_its_own_arm(tmp_path) -> None:
    numeric = {
        k: v for k, v in _eva_numeric().items() if not k.startswith("right.")
    }
    _write_eva(tmp_path / "left.zarr", numeric=numeric, embodiment="eva_left_arm")

    report = validate_episode(tmp_path / "left.zarr", strict=True)

    assert report.ok, report.text()
    assert "right.obs_ee_pose" not in _levels(report)


def test_a_human_episode_owes_keypoints_and_a_head_pose(tmp_path) -> None:
    path = tmp_path / "human.zarr"
    numeric = {
        "left.obs_ee_pose": _poses(),
        "right.obs_ee_pose": _poses(0.2),
        "left.obs_keypoints": np.zeros((LENGTH, 63)),
        "obs_head_pose": _poses(),
        "obs_rgb_timestamps_ns": np.arange(LENGTH, dtype=np.int64),
    }
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data=numeric,
        embodiment="human_bimanual",
        chunk_timesteps=LENGTH,
        annotations=[("wave", 0, LENGTH)],
        intrinsics={"front_1": K},
        metadata_override={"schema_version": "v3.1"},
    )

    report = validate_episode(path)

    levels = _levels(report)
    assert levels["left.obs_keypoints"] == OK
    assert levels["obs_head_pose"] == OK
    # The registry declares 21 MANO slots with three coordinates per slot.
    assert levels["right.obs_keypoints"] == ERROR
    # The human platform has neither an arm chain nor a parallel jaw, so those
    # conditional rules do not run.
    assert "left.obs_gripper" not in levels
    assert "left.obs_joints" not in levels


def test_an_unknown_embodiment_stops_before_the_array_rules(tmp_path) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path)
    store = __import__("zarr").open_group(str(path), mode="a")
    store.attrs["embodiment"] = "sharpa_bimanual"

    report = validate_episode(path)

    levels = _levels(report)
    assert levels["attrs.embodiment"] == ERROR
    assert levels["embodiment"] == ERROR
    assert "left.obs_ee_pose" not in levels


def test_a_morphology_block_must_agree_with_the_embodiment_name(tmp_path) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path)
    store = __import__("zarr").open_group(str(path), mode="a")
    store.attrs["morphology"] = {
        "platform": "human_body",
        "end_effector": "mano_hand",
    }

    report = validate_episode(path)

    assert _levels(report)["attrs.morphology"] == ERROR
    assert "disagrees" in report.text()


def test_a_morphology_block_selects_the_end_effector(tmp_path) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path)
    store = __import__("zarr").open_group(str(path), mode="a")
    store.attrs["morphology"] = {
        "platform": "eva_x5",
        "end_effector": {"left": "eva_parallel_jaw", "right": "eva_parallel_jaw"},
        "vendor": "rl2",
    }

    report = validate_episode(path, strict=True)

    assert report.ok, report.text()
    assert "eva_parallel_jaw" in report.text(verbose=True)


def test_a_directory_that_is_not_an_episode_reports_one_error(tmp_path) -> None:
    report = validate_episode(tmp_path / "absent.zarr")

    assert not report.ok
    assert _levels(report)["episode"] == ERROR


def test_the_cli_exit_code_follows_the_findings(tmp_path, capsys) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path)

    assert main([str(path)]) == 0
    assert main([str(path), "--strict"]) == 0
    assert main([str(tmp_path / "absent.zarr")]) == 1
    assert "cannot open as a zarr group" in capsys.readouterr().out


def test_the_cli_can_report_json(tmp_path, capsys) -> None:
    path = tmp_path / "eva.zarr"
    _write_eva(path)

    main([str(path), "--json"])

    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload[0]["ok"] is True
    assert payload[0]["path"] == str(path)


@pytest.mark.parametrize("section", ["attributes", "checks", "arrays"])
def test_every_schema_rule_declares_a_usable_requirement(section) -> None:
    schema = load_schema()
    rules = schema[section]
    entries = rules.values() if isinstance(rules, dict) else rules
    assert entries
    for rule in entries:
        assert rule.get("required", False) in (True, False, "strict")


# --------------------------------------------------------------------------
# Degeneracy, timestamps and annotations
# --------------------------------------------------------------------------


def _finding(report, check: str) -> str:
    return next(f.message for f in report.findings if f.check == check)


def test_a_pose_track_that_never_moves_is_an_error(tmp_path) -> None:
    numeric = _eva_numeric()
    numeric["left.obs_ee_pose"] = np.tile(_poses()[0], (LENGTH, 1))
    _write_eva(tmp_path / "still.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "still.zarr")

    assert _levels(report)["pose_degeneracy"] == ERROR
    assert "constant across all 4 frames" in _finding(report, "pose_degeneracy")


def test_identity_rotations_beyond_the_limit_are_an_error(tmp_path) -> None:
    numeric = _eva_numeric()
    identity = numeric["left.obs_ee_pose"].copy()
    identity[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
    numeric["left.obs_ee_pose"] = identity
    _write_eva(tmp_path / "identity.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "identity.zarr")

    assert _levels(report)["pose_degeneracy"] == ERROR
    assert "identity rotation on 100% of frames" in _finding(report, "pose_degeneracy")


def test_an_identity_extrinsic_is_an_error(tmp_path) -> None:
    path = tmp_path / "identity_rig.zarr"
    ZarrWriter.create_and_write(
        episode_path=path,
        numeric_data=_eva_numeric(),
        embodiment="eva_bimanual",
        chunk_timesteps=LENGTH,
        annotations=[("fold the towel", 0, LENGTH)],
        intrinsics={"front_1": K},
        extrinsics={"left": np.eye(4), "right": np.eye(4)},
    )

    report = validate_episode(path)

    assert _levels(report)["calibration_degeneracy"] == ERROR
    assert "left arm base" in _finding(report, "calibration_degeneracy")


def test_a_synthesized_camera_matrix_is_an_error(tmp_path) -> None:
    width = height = 480
    synthetic = np.array(
        [
            [width, 0.0, width / 2, 0.0],
            [0.0, width, height / 2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    _write_eva(
        tmp_path / "synthetic.zarr",
        calibration={
            "reference_frame": "camera:front_1",
            "cameras": {
                "front_1": {"K": synthetic.tolist(), "resolution": [width, height]}
            },
        },
    )

    report = validate_episode(tmp_path / "synthetic.zarr")

    assert _levels(report)["intrinsics_signature"] == ERROR
    assert "synthesized centred pinhole" in _finding(report, "intrinsics_signature")


def test_a_rectified_aria_camera_matrix_stays_out_of_the_net(tmp_path) -> None:
    # ``fx`` is 266.5 at width 640, so the full synthetic signature is false.
    aria = np.array(
        [[266.5, 0.0, 320.0, 0.0], [0.0, 266.5, 240.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    _write_eva(
        tmp_path / "aria.zarr",
        calibration={
            "reference_frame": "camera:front_1",
            "cameras": {"front_1": {"K": aria.tolist(), "resolution": [640, 480]}},
        },
    )

    report = validate_episode(tmp_path / "aria.zarr")

    assert _levels(report)["intrinsics_signature"] == OK


def test_a_stalled_clock_is_an_error(tmp_path) -> None:
    numeric = _eva_numeric()
    numeric["obs_rgb_timestamps_ns"] = np.array(
        [1_000, 2_000, 2_000, 1_500], dtype=np.int64
    )
    _write_eva(tmp_path / "clock.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "clock.zarr")

    assert _levels(report)["timestamps"] == ERROR
    assert "does not increase on 2 of 3 steps" in _finding(report, "timestamps")


def test_a_float64_quantized_clock_is_an_error(tmp_path) -> None:
    numeric = _eva_numeric()
    numeric["obs_rgb_timestamps_ns"] = (
        np.int64(1_700_000_000_000_000_000) + np.arange(LENGTH, dtype=np.int64) * 256
    )
    _write_eva(tmp_path / "quantized.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "quantized.zarr")

    assert "quantized to 256 ns" in _finding(report, "timestamps")


def test_a_second_time_base_is_an_error(tmp_path) -> None:
    numeric = _eva_numeric()
    numeric["relative_timestamp_s"] = np.linspace(0.0, 0.1, LENGTH)[:, None]
    _write_eva(tmp_path / "two_clocks.zarr", numeric=numeric)

    report = validate_episode(tmp_path / "two_clocks.zarr")

    assert _levels(report)["timestamps"] == ERROR
    assert "one clock per episode" in _finding(report, "timestamps")


def test_annotation_coverage_below_the_minimum_is_reported(tmp_path) -> None:
    _write_eva(
        tmp_path / "thin.zarr", annotations=[("fold the towel", 0, LENGTH - 2)]
    )

    lenient = validate_episode(tmp_path / "thin.zarr")
    strict = validate_episode(tmp_path / "thin.zarr", strict=True)

    assert _levels(lenient)["annotation_coverage"] == WARNING
    assert _levels(strict)["annotation_coverage"] == ERROR
    assert "cover 50% of the episode" in _finding(strict, "annotation_coverage")


def test_an_annotation_span_past_the_episode_is_reported(tmp_path) -> None:
    _write_eva(tmp_path / "over.zarr", annotations=[("fold", 0, LENGTH + 5)])

    report = validate_episode(tmp_path / "over.zarr", strict=True)

    assert "outside [0, 4)" in _finding(report, "annotation_coverage")


def test_delimiter_encoded_metadata_is_an_error(tmp_path) -> None:
    _write_eva(
        tmp_path / "skill.zarr",
        annotations=[("pick up the cup | Skill: pick", 0, LENGTH)],
    )

    report = validate_episode(tmp_path / "skill.zarr")

    assert _levels(report)["annotation_text"] == ERROR
    assert "Skill: pick" in _finding(report, "annotation_text")


def test_delimiter_encoded_metadata_in_the_task_description_is_an_error(
    tmp_path,
) -> None:
    _write_eva(tmp_path / "desc.zarr", task_description="fold | Skill: fold")

    report = validate_episode(tmp_path / "desc.zarr")

    assert _levels(report)["annotation_text"] == ERROR
    assert "task_description" in _finding(report, "annotation_text")
