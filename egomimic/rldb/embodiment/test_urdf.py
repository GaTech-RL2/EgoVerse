"""Test the URDF reader and the joints-versus-keypoints residual."""

import numpy as np
import pytest

from egomimic.rldb.conftest import HAND_DOF, HAND_JOINT_NAMES
from egomimic.rldb.embodiment.hand_kinematics import fk_keypoints, keypoint_residuals
from egomimic.rldb.embodiment.registry import RegistryError, _parse_end_effector
from egomimic.rldb.embodiment.urdf import UrdfError, load_urdf

_TWO_LINK = """
<robot name="two_link">
  <link name="base"/>
  <link name="upper"/>
  <link name="tip"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="wrist" type="fixed">
    <parent link="upper"/><child link="tip"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
</robot>
"""


def _write(tmp_path, text, name="robot.urdf"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_the_root_link_is_the_link_that_is_no_joints_child(tmp_path) -> None:
    chain = load_urdf(_write(tmp_path, _TWO_LINK))

    assert chain.root == "base"
    assert chain.actuated_joint_names == ("shoulder",)


def test_a_quarter_turn_swings_the_tip_into_the_x_axis(tmp_path) -> None:
    chain = load_urdf(_write(tmp_path, _TWO_LINK))

    poses = chain.link_transforms({"shoulder": np.pi / 2})

    # The tip starts 2 m above the base; rotating +90 deg about y sends the
    # upper segment along +x while its own 1 m offset stays below it.
    assert np.allclose(poses["tip"][:3, 3], [1.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(poses["upper"][:3, 3], [0.0, 0.0, 1.0], atol=1e-9)


def test_an_omitted_joint_value_is_zero(tmp_path) -> None:
    chain = load_urdf(_write(tmp_path, _TWO_LINK))

    assert np.allclose(chain.link_transforms({})["tip"][:3, 3], [0.0, 0.0, 2.0])


def test_a_mimic_joint_follows_its_source(tmp_path) -> None:
    text = _TWO_LINK.replace(
        '<joint name="wrist" type="fixed">',
        '<joint name="wrist" type="revolute">\n'
        '    <axis xyz="0 1 0"/>\n'
        '    <mimic joint="shoulder" multiplier="1.0" offset="0"/>',
    )
    chain = load_urdf(_write(tmp_path, text))

    assert chain.actuated_joint_names == ("shoulder",)
    poses = chain.link_transforms({"shoulder": np.pi / 2})
    # The tip sits where the fixed version put it, because its own rotation
    # happens after its offset. The mimic shows in the orientation: both joints
    # turned 90 deg about y, so the tip frame is a half turn from the base.
    assert np.allclose(poses["tip"][:3, 3], [1.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(
        poses["tip"][:3, :3], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=1e-9
    )


def test_a_prismatic_joint_slides_along_its_axis(tmp_path) -> None:
    text = _TWO_LINK.replace('type="revolute"', 'type="prismatic"')
    chain = load_urdf(_write(tmp_path, text))

    poses = chain.link_transforms({"shoulder": 0.25})

    assert np.allclose(poses["upper"][:3, 3], [0.0, 0.25, 1.0])


def test_an_unsupported_joint_type_is_refused(tmp_path) -> None:
    text = _TWO_LINK.replace('type="revolute"', 'type="floating"')

    with pytest.raises(UrdfError, match="unsupported type 'floating'"):
        load_urdf(_write(tmp_path, text))


def test_two_root_links_are_refused(tmp_path) -> None:
    text = _TWO_LINK.replace("<link name=\"base\"/>", '<link name="base"/><link name="orphan"/>')

    with pytest.raises(UrdfError, match="exactly one root link"):
        load_urdf(_write(tmp_path, text))


def test_naming_an_absent_link_says_which_links_exist(tmp_path) -> None:
    chain = load_urdf(_write(tmp_path, _TWO_LINK))

    with pytest.raises(UrdfError, match=r"\['fingertip'\] are not in this URDF"):
        chain.link_positions({}, ["fingertip"])


# ---------------------------------------------------------------------------
# The residual gate
# ---------------------------------------------------------------------------


def test_matching_joints_and_keypoints_leave_no_residual(hand_spec, hand_track) -> None:
    joints, keypoints, poses = hand_track()

    residuals = keypoint_residuals(hand_spec, joints, keypoints, poses)

    assert residuals.shape == (len(joints), 21)
    assert residuals.max() < 1e-9


def test_a_rotated_joint_order_shows_up_as_a_residual(hand_spec, hand_track) -> None:
    joints, keypoints, poses = hand_track()

    residuals = keypoint_residuals(
        hand_spec, np.roll(joints, 1, axis=1), keypoints, poses
    )

    assert residuals.max() > 1e-3


def test_degrees_stored_as_radians_show_up_as_a_residual(hand_spec, hand_track) -> None:
    joints, keypoints, poses = hand_track()

    residuals = keypoint_residuals(
        hand_spec, np.degrees(joints), keypoints, poses
    )

    assert residuals.max() > 1e-3


def test_keypoints_left_in_the_episode_frame_show_up_as_a_residual(
    hand_spec, hand_track
) -> None:
    joints, keypoints, poses = hand_track()
    root_frame = fk_keypoints(hand_spec, joints).reshape(len(joints), -1)

    residuals = keypoint_residuals(hand_spec, joints, root_frame, poses)

    assert residuals.max() > 1e-3


def test_a_wrong_joint_width_names_the_expected_one(hand_spec, hand_track) -> None:
    joints, keypoints, poses = hand_track()

    with pytest.raises(ValueError, match=rf"expected a \(T, {HAND_DOF}\) joint array"):
        keypoint_residuals(hand_spec, joints[:, :-1], keypoints, poses)


def test_a_joint_name_the_urdf_does_not_actuate_is_refused(hand_spec, hand_track):
    from dataclasses import replace

    joints, _, _ = hand_track()
    renamed = replace(hand_spec, joint_names=("typo",) + tuple(HAND_JOINT_NAMES[1:]))

    with pytest.raises(UrdfError, match=r"\['typo'\] are not actuated joints"):
        fk_keypoints(renamed, joints)


# ---------------------------------------------------------------------------
# Registry rules that keep the gate from going dormant
# ---------------------------------------------------------------------------


def _entry(**overrides) -> dict:
    block = {
        "class": "dexterous_hand",
        "dof": 2,
        "action_space": "keypoints",
        "keypoints": {"topology": "mano21", "valid": [0, 4]},
        "joint_names": ["a", "b"],
        "urdf": "hand.urdf",
        "keypoint_links": {0: "palm", 4: "thumb_tip"},
        "fk_tolerance_m": 0.01,
    }
    return block | overrides


def test_a_urdf_without_joint_names_is_refused() -> None:
    with pytest.raises(RegistryError, match="`urdf` needs `joint_names`"):
        _parse_end_effector("h", _entry(joint_names=None))


def test_a_urdf_without_keypoint_links_is_refused() -> None:
    with pytest.raises(RegistryError, match="`urdf` needs `keypoint_links`"):
        _parse_end_effector("h", _entry(keypoint_links=None))


def test_a_urdf_without_a_tolerance_is_refused() -> None:
    with pytest.raises(RegistryError, match="`urdf` needs `fk_tolerance_m`"):
        _parse_end_effector("h", _entry(fk_tolerance_m=None))


def test_keypoint_links_must_cover_exactly_the_valid_slots() -> None:
    with pytest.raises(RegistryError, match="maps slots \\[0\\], but the valid slots"):
        _parse_end_effector("h", _entry(keypoint_links={0: "palm"}))


def test_a_tolerance_without_a_urdf_is_refused() -> None:
    with pytest.raises(RegistryError, match="no effect without a `urdf`"):
        _parse_end_effector(
            "h", _entry(urdf=None, keypoint_links=None, joint_names=None)
        )


def test_tactile_needs_a_unit() -> None:
    with pytest.raises(RegistryError, match="missing required field 'units'"):
        _parse_end_effector("h", _entry(tactile={"taxels": 30}))
    with pytest.raises(RegistryError, match="physical unit of one value"):
        _parse_end_effector("h", _entry(tactile={"taxels": 30, "units": "  "}))


def test_tactile_width_is_taxels_times_channels() -> None:
    spec = _parse_end_effector(
        "h", _entry(tactile={"taxels": 30, "channels": 2, "units": "N"})
    )

    assert spec.tactile.width == 60
