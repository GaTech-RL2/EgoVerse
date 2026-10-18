"""Test dexterous array rules and forward-kinematics residual validation."""

import numpy as np
import pytest

from egomimic.rldb.conftest import HAND_DOF
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.zarr.validate import ERROR, OK, WARNING, validate_episode
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

K = np.array([[200.0, 0.0, 160.0, 0.0], [0.0, 200.0, 120.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
LENGTH = 6
AUX_DOF = 7
ARM_DOF = 7


def test_missing_optional_model_is_a_limitation(
    write_dexterous, hand_spec, monkeypatch, tmp_path
):
    from dataclasses import replace

    from egomimic.rldb.embodiment.registry import load_end_effectors

    path = write_dexterous()
    entries = {
        **load_end_effectors(),
        hand_spec.name: replace(hand_spec, urdf=str(tmp_path / "missing.urdf")),
    }
    monkeypatch.setattr(
        "egomimic.rldb.embodiment.embodiment.load_end_effectors", lambda: entries
    )
    report = validate_episode(path)
    assert report.ok, report.text()
    assert any(f.check == "fk_unavailable" for f in report.warnings)


def _levels(report) -> dict[str, str]:
    return {f.check: f.level for f in report.findings}


def _message(report, check: str) -> str:
    return next(f.message for f in report.findings if f.check == check)


@pytest.fixture
def write_dexterous(tmp_path, dexterous_registry, hand_track):
    """Write one dexterous episode and return its path."""

    def write(**overrides):
        numeric = {}
        for side in ("left", "right"):
            joints, keypoints, poses = hand_track(LENGTH)
            numeric[f"{side}.obs_ee_pose"] = poses
            numeric[f"{side}.obs_hand_joints"] = joints
            numeric[f"{side}.cmd_hand_joints"] = joints
            numeric[f"{side}.obs_hand_keypoints"] = keypoints
            numeric[f"{side}.obs_joints"] = np.zeros((LENGTH, ARM_DOF))
            numeric[f"{side}.cmd_joints"] = np.zeros((LENGTH, ARM_DOF))
        numeric["obs_aux_joints"] = np.zeros((LENGTH, AUX_DOF))
        numeric["cmd_aux_joints"] = np.zeros((LENGTH, AUX_DOF))
        numeric["obs_rgb_timestamps_ns"] = np.arange(LENGTH, dtype=np.int64)
        numeric.update(overrides.pop("numeric", {}))
        for key in overrides.pop("drop", ()):
            numeric.pop(key, None)

        path = tmp_path / "dexterous.zarr"
        ZarrWriter.create_and_write(
            episode_path=path,
            numeric_data=numeric,
            embodiment="eva_bimanual",
            chunk_timesteps=LENGTH,
            calibration={
                "reference_frame": "camera:front_1",
                "cameras": {"front_1": {"K": K.tolist()}},
                "arm_bases": {
                    side: np.linalg.inv(T).tolist()
                    for side, T in Eva.EXTRINSICS.items()
                },
            },
            annotations=[("grasp the mug", 0, LENGTH)],
            metadata_override={"schema_version": "v3.1"},
            **overrides,
        )
        return path

    return write


def test_a_consistent_dexterous_episode_passes(write_dexterous) -> None:
    report = validate_episode(write_dexterous())

    assert report.ok, report.text()
    levels = _levels(report)
    assert levels["left.obs_hand_joints"] == OK
    assert levels["right.obs_hand_keypoints"] == OK
    assert levels["obs_aux_joints"] == OK
    assert levels["fk_residual"] == OK


def test_hand_joints_are_required_for_a_dexterous_hand(write_dexterous) -> None:
    report = validate_episode(write_dexterous(drop=["left.obs_hand_joints"]))

    assert _levels(report)["left.obs_hand_joints"] == ERROR


def test_hand_keypoints_are_required_alongside_the_joints(write_dexterous) -> None:
    report = validate_episode(write_dexterous(drop=["right.obs_hand_keypoints"]))

    assert _levels(report)["right.obs_hand_keypoints"] == ERROR


def test_the_hand_joint_width_comes_from_the_registry(write_dexterous) -> None:
    path = write_dexterous(
        numeric={"left.obs_hand_joints": np.zeros((LENGTH, HAND_DOF - 1))}
    )

    report = validate_episode(path)

    assert f"expected {HAND_DOF} (ee_dof)" in _message(report, "left.obs_hand_joints")


def test_the_aux_chain_is_required_when_the_platform_declares_one(
    write_dexterous,
) -> None:
    report = validate_episode(write_dexterous(drop=["cmd_aux_joints"]))

    assert _levels(report)["cmd_aux_joints"] == ERROR


def test_a_wrong_joint_order_fails_the_residual_gate(
    write_dexterous, hand_track
) -> None:
    joints, keypoints, poses = hand_track(LENGTH)
    path = write_dexterous(
        numeric={
            "left.obs_hand_joints": np.roll(joints, 1, axis=1),
            "left.obs_hand_keypoints": keypoints,
            "left.obs_ee_pose": poses,
        }
    )

    report = validate_episode(path)

    assert _levels(report)["fk_residual"] == ERROR
    assert "check joint order, units, handedness" in _message(report, "fk_residual")


def test_degrees_stored_where_radians_belong_fail_the_residual_gate(
    write_dexterous, hand_track
) -> None:
    joints, keypoints, poses = hand_track(LENGTH)
    path = write_dexterous(
        numeric={
            "left.obs_hand_joints": np.degrees(joints),
            "left.obs_hand_keypoints": keypoints,
            "left.obs_ee_pose": poses,
        }
    )

    report = validate_episode(path)

    assert _levels(report)["fk_residual"] == ERROR
    assert "left: forward kinematics" in _message(report, "fk_residual")


def test_the_gate_reports_no_finding_for_an_end_effector_with_no_urdf(
    tmp_path,
) -> None:
    """An end-effector without a URDF produces no FK-residual finding."""
    from egomimic.rldb.zarr.test_validate import _write_eva

    _write_eva(tmp_path / "eva.zarr")
    report = validate_episode(tmp_path / "eva.zarr")

    assert "fk_residual" not in _levels(report)
    assert report.ok, report.text()


def test_a_stored_tactile_array_matches_the_registry_width(write_dexterous) -> None:
    report = validate_episode(
        write_dexterous(numeric={"left.obs_tactile": np.zeros((LENGTH, 30))})
    )

    assert report.ok, report.text()
    assert _levels(report)["tactile_declaration"] == OK


def test_a_tactile_array_of_the_wrong_width_is_an_error(write_dexterous) -> None:
    report = validate_episode(
        write_dexterous(numeric={"left.obs_tactile": np.zeros((LENGTH, 31))})
    )

    assert "axis 1 is 31, expected 30 (tactile)" in _message(report, "left.obs_tactile")


def test_a_tactile_array_needs_a_registry_declaration(
    monkeypatch, write_dexterous, hand_spec
) -> None:
    from dataclasses import replace

    path = write_dexterous(numeric={"left.obs_tactile": np.zeros((LENGTH, 30))})
    silent = replace(hand_spec, tactile=None)
    monkeypatch.setattr(
        "egomimic.rldb.embodiment.embodiment.load_end_effectors",
        lambda: {silent.name: silent},
    )

    report = validate_episode(path)

    assert _levels(report)["tactile_declaration"] == ERROR
    assert "declares no `tactile:` block" in _message(report, "tactile_declaration")


@pytest.fixture
def limited_hand(monkeypatch, write_dexterous, hand_spec):
    """Give the generated hand registry limits of [-1, 1] rad on every joint."""
    from dataclasses import replace

    from egomimic.rldb.embodiment.registry import load_end_effectors

    limited = replace(
        hand_spec, joint_limits=tuple((-1.0, 1.0) for _ in range(HAND_DOF))
    )
    entries = load_end_effectors() | {limited.name: limited}
    monkeypatch.setattr(
        "egomimic.rldb.embodiment.embodiment.load_end_effectors", lambda: entries
    )
    return limited


def _hand_with(hand_track, column: int, value: float):
    """Return FK-consistent tracks whose joint ``column`` reads ``value`` at frame 2."""
    joints = np.random.default_rng(0).uniform(-0.6, 0.6, size=(LENGTH, HAND_DOF))
    joints[2, column] = value
    joints, keypoints, poses = hand_track(LENGTH, joints=joints)
    return {
        "left.obs_hand_joints": joints,
        "left.obs_hand_keypoints": keypoints,
        "left.obs_ee_pose": poses,
    }


def test_a_small_observed_limit_excursion_is_a_warning(
    write_dexterous, limited_hand, hand_track
) -> None:
    report = validate_episode(write_dexterous(numeric=_hand_with(hand_track, 3, 1.1)))

    assert report.ok, report.text()
    levels = _levels(report)
    assert levels["joint_limit_excursion"] == WARNING
    assert levels["joint_limit_gross"] == OK
    assert "j_0_3 by 0.1000" in _message(report, "joint_limit_excursion")


def test_a_gross_observed_limit_excursion_is_an_error(
    write_dexterous, limited_hand, hand_track
) -> None:
    report = validate_episode(write_dexterous(numeric=_hand_with(hand_track, 3, 1.3)))

    assert _levels(report)["joint_limit_gross"] == ERROR
    assert "check units, joint order" in _message(report, "joint_limit_gross")
    assert _levels(report)["fk_residual"] == OK


def test_a_commanded_overshoot_is_only_a_warning(write_dexterous, limited_hand) -> None:
    joints = np.zeros((LENGTH, HAND_DOF))
    joints[:, 5] = 1.4
    report = validate_episode(write_dexterous(numeric={"left.cmd_hand_joints": joints}))

    assert report.ok, report.text()
    assert "left.cmd_hand_joints" in _message(report, "joint_limit_excursion")


def test_degrees_stored_where_radians_belong_fail_the_gross_limit_tier(
    write_dexterous, limited_hand, hand_track
) -> None:
    joints, keypoints, poses = hand_track(LENGTH)
    path = write_dexterous(numeric={"left.obs_hand_joints": np.degrees(joints)})

    assert _levels(validate_episode(path))["joint_limit_gross"] == ERROR


def test_within_limit_joints_pass_both_tiers(write_dexterous, limited_hand) -> None:
    levels = _levels(validate_episode(write_dexterous()))

    assert levels["joint_limit_gross"] == levels["joint_limit_excursion"] == OK


def test_a_one_frame_joint_spike_is_a_warning(write_dexterous) -> None:
    joints = np.zeros((LENGTH, ARM_DOF))
    joints[3, 2] = 3.0
    report = validate_episode(write_dexterous(numeric={"right.cmd_joints": joints}))

    assert report.ok, report.text()
    message = _message(report, "joint_spikes")
    assert "right.cmd_joints: 1 frame(s)" in message and "first at frame 3" in message


def test_a_value_beyond_two_pi_is_a_spike(write_dexterous) -> None:
    joints = np.zeros((LENGTH, AUX_DOF))
    joints[:, 0] = 96.5
    report = validate_episode(write_dexterous(numeric={"cmd_aux_joints": joints}))

    assert f"cmd_aux_joints: {LENGTH} frame(s)" in _message(report, "joint_spikes")


def test_smooth_motion_has_no_spikes(write_dexterous) -> None:
    joints = np.linspace(0.0, 2.0, LENGTH)[:, None] * np.ones((1, ARM_DOF))
    report = validate_episode(write_dexterous(numeric={"left.obs_joints": joints}))

    assert _levels(report)["joint_spikes"] == OK
