"""The shipped Sharpa kinematic models and the tracks derived from them."""

import hashlib
import shutil

import numpy as np
import pytest
import pytorch_kinematics as pk
import torch

from egomimic.rldb.zarr import dexmate_rules
from egomimic.scripts.sharpa_process import kinematics as kin_module
from egomimic.scripts.sharpa_process import sharpa_tracks
from egomimic.scripts.sharpa_process.kinematics import (
    UrdfError,
    asset_path,
    load_spec,
    load_urdf,
    verified_urdf,
)
from egomimic.scripts.sharpa_process.sharpa_tracks import (
    TCP_LINK,
    TCP_MODEL,
    SharpaKinematics,
    derive_tracks,
    fk_residuals,
    joint_limit_excursions,
    tcp_side_check,
)
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix

SIDES = ("left", "right")


@pytest.fixture(scope="module")
def kinematics():
    return SharpaKinematics()


def _in_limit_vector(kinematics, frames, seed):
    """Return deterministic joint vectors strictly inside every limit."""
    rng = np.random.default_rng(seed)
    lower, upper = kinematics.limits.T
    return lower + (upper - lower) * rng.uniform(0.1, 0.9, size=(frames, 65))


def _urdfs():
    spec = load_spec()
    return {
        "platform": spec["platform"],
        **{side: spec["hands"][side] for side in SIDES},
    }


def test_shipped_files_match_the_spec_sha256():
    for entry in _urdfs().values():
        data = asset_path(entry["urdf"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["urdf_sha256"]
    for name in ("LICENSE.txt", "NOTICE.txt"):
        assert asset_path(name).read_text().strip()


def test_a_modified_urdf_is_refused(tmp_path, monkeypatch):
    entry = load_spec()["hands"]["left"]
    copy = tmp_path / entry["urdf"]
    shutil.copy(asset_path(entry["urdf"]), copy)
    copy.write_bytes(copy.read_bytes().replace(b"1.3963", b"1.3964", 1))
    monkeypatch.setattr(kin_module, "asset_path", lambda name: tmp_path / name)
    with pytest.raises(UrdfError, match="sha256"):
        verified_urdf(entry["urdf"], entry["urdf_sha256"])


def test_urdf_joints_and_limits_match_dexmate_rules():
    spec = load_spec()
    for side in SIDES:
        hand = load_urdf(asset_path(spec["hands"][side]["urdf"]).read_bytes())
        assert hand.actuated_joint_names == dexmate_rules.HAND_JOINT_NAMES[side]
        limits = tuple(hand.joint_limits[n] for n in hand.actuated_joint_names)
        assert limits == dexmate_rules.HAND_JOINT_LIMITS[side]
        links = spec["hands"][side]
        named = [links["ee_pose_link"], *links["keypoint_links"].values()]
        assert set(named + links["tactile"]["taxel_links"]) <= hand.links
        assert hand.root == f"{side}_hand_flange"
    platform = load_urdf(asset_path(spec["platform"]["urdf"]).read_bytes())
    expected = set(dexmate_rules.AUX_JOINT_NAMES)
    for side in SIDES:
        expected |= set(dexmate_rules.ARM_JOINT_NAMES[side])
        assert f"{side}_hand_base_link" in platform.links
        assert TCP_LINK.format(side=side) in platform.links
    assert set(platform.actuated_joint_names) == expected
    assert tuple(spec["platform"]["aux_joint_names"]) == dexmate_rules.AUX_JOINT_NAMES
    assert spec["platform"]["head_link"] in platform.links


def test_unsupported_urdfs_are_refused():
    two_roots = b'<robot name="r"><link name="a"/><link name="b"/></robot>'
    with pytest.raises(UrdfError, match="exactly one root"):
        load_urdf(two_roots)
    mimic = (
        b'<robot name="r"><link name="a"/><link name="b"/><link name="c"/>'
        b'<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
        b'<axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>'
        b'<joint name="k" type="revolute"><parent link="b"/><child link="c"/>'
        b'<axis xyz="0 0 1"/><mimic joint="j"/>'
        b'<limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    with pytest.raises(UrdfError, match="mimic"):
        load_urdf(mimic)


def test_batched_fk_matches_pytorch_kinematics(kinematics):
    # An independent reference: pytorch-kinematics' own URDF builder. It parses
    # origins in float32, hence the 1e-6 tolerance.
    values = _in_limit_vector(kinematics, 5, 0)
    named = dict(zip(kinematics.column_names, values.T, strict=True))
    chains = {"platform": kinematics.platform, **kinematics.hands}
    for key, entry in _urdfs().items():
        reference = pk.build_chain_from_urdf(asset_path(entry["urdf"]).read_bytes())
        reference = reference.to(dtype=torch.float64)
        angles = torch.from_numpy(
            np.stack([named[n] for n in reference.get_joint_parameter_names()], -1)
        )
        expected = reference.forward_kinematics(angles)
        ours = chains[key].batch_link_transforms(named, len(values))
        assert set(ours) == set(expected), key
        for link, transform in expected.items():
            np.testing.assert_allclose(
                ours[link], transform.get_matrix().numpy(), atol=1e-6, err_msg=link
            )


def test_tracks_compose_platform_and_hand_fk(kinematics, monkeypatch):
    observed = _in_limit_vector(kinematics, 7, 2)
    commanded = _in_limit_vector(kinematics, 7, 3).astype(np.float32)
    links = ("head_base_link", "left_hand_wrist", TCP_LINK.format(side="right"))
    tracks, poses = derive_tracks(
        observed, commanded, kinematics=kinematics, links=links
    )
    named = dict(zip(kinematics.column_names, observed.T, strict=True))
    platform = kinematics.platform.batch_link_transforms(named, 7)
    for side in SIDES:
        hand = kinematics.hands[side].batch_link_transforms(named, 7)
        base_T_flange = platform[f"{side}_hand_base_link"]
        palm = base_T_flange @ hand[f"{side}_hand_C_MC"]
        np.testing.assert_allclose(
            _xyzwxyz_to_matrix(tracks[f"{side}.obs_ee_pose"]), palm, atol=1e-12
        )
        tip = (base_T_flange @ hand[f"{side}_pinky_fingertip"])[:, :3, 3]
        np.testing.assert_array_equal(
            tracks[f"{side}.obs_hand_keypoints"][:, 60:63], tip
        )
        # Slot 0 is the palm link itself.
        np.testing.assert_array_equal(
            tracks[f"{side}.obs_hand_keypoints"][:, :3], palm[:, :3, 3]
        )
        assert tracks[f"{side}.cmd_hand_keypoints"].shape == (7, 63)
        assert tracks[f"{side}.cmd_hand_joints"].dtype == np.float32
        assert tracks[f"{side}.obs_ee_pose"].dtype == np.float64
    np.testing.assert_array_equal(poses["head_base_link"], platform["head_base_link"])
    np.testing.assert_array_equal(tracks["obs_aux_joints"], observed[:, 58:65])
    np.testing.assert_array_equal(tracks["right.obs_joints"], observed[:, 29:36])
    np.testing.assert_array_equal(tracks["right.obs_hand_joints"], observed[:, 36:58])

    # FK blocks are independent.
    monkeypatch.setattr(sharpa_tracks, "FK_BLOCK", 3)
    blocked, blocked_poses = derive_tracks(
        observed, commanded, kinematics=kinematics, links=links
    )
    for key, values in tracks.items():
        np.testing.assert_array_equal(blocked[key], values, err_msg=key)
    for link in links:
        np.testing.assert_array_equal(blocked_poses[link], poses[link])
    with pytest.raises(ValueError, match="unknown link"):
        derive_tracks(observed, observed, kinematics=kinematics, links=("nope",))
    with pytest.raises(ValueError, match="non-finite"):
        bad = observed.copy()
        bad[1, 3] = np.nan
        derive_tracks(bad, observed, kinematics=kinematics)


def test_excursions_are_reported_per_column_with_their_tier(kinematics):
    raw = _in_limit_vector(kinematics, 4, 3).astype(np.float32)
    column = kinematics.column_names.index("right_middle_DIP")
    upper = kinematics.limits[column, 1]
    raw[1:3, column] = upper + 0.15
    [record] = joint_limit_excursions(raw, "obs", kinematics)
    assert record["array"] == "right.obs_hand_joints"
    assert record["frames_above"] == 2 and record["frames_below"] == 0
    assert record["max_above"] == pytest.approx(0.15, abs=1e-6)
    # 0.15 rad is past main's warning margin and inside its error margin.
    assert record["validator_check"] == "warning"
    assert not record["validator_checked"]
    raw[1, column] = upper + 0.3
    [gross] = joint_limit_excursions(raw, "obs", kinematics)
    assert gross["validator_check"] == "error" and gross["validator_checked"]
    # Commanded joints are never an error.
    [command] = joint_limit_excursions(raw, "cmd", kinematics)
    assert command["array"] == "right.cmd_hand_joints"
    assert command["validator_check"] == "warning"
    # Arm columns are reported, with their URDF limits, but not checked.
    arm = kinematics.column_names.index("left_arm_joint_2")
    raw[0, arm] = kinematics.limits[arm, 0] - 0.5
    arm_record = joint_limit_excursions(raw, "obs", kinematics)[0]
    assert arm_record["array"] == "left.obs_joints"
    assert arm_record["limit_source"].startswith("urdf:")
    assert arm_record["validator_check"] is None


def test_fk_residuals_are_zero_for_derived_keypoints_only(kinematics):
    observed = _in_limit_vector(kinematics, 6, 4)
    tracks, _ = derive_tracks(observed, observed, kinematics=kinematics)
    for side in SIDES:
        args = [
            tracks[f"{side}.obs_hand_joints"],
            tracks[f"{side}.obs_hand_keypoints"],
            tracks[f"{side}.obs_ee_pose"],
        ]
        residuals = fk_residuals(side, *args, kinematics)
        assert residuals.shape == (6, 21) and residuals.max() < 1e-9
        # Joints in the wrong order no longer explain the keypoints.
        shuffled = args[0][:, ::-1]
        assert fk_residuals(side, shuffled, *args[1:], kinematics).max() > 0.005
    other = fk_residuals(
        "left",
        tracks["right.obs_hand_joints"],
        tracks["right.obs_hand_keypoints"],
        tracks["right.obs_ee_pose"],
        kinematics,
    )
    assert other.max() > 0.005
    with pytest.raises(ValueError, match="expected"):
        fk_residuals("left", observed[:, :21], *args[1:], kinematics)


def _tcp_pose(kinematics, observed):
    """Sharpa's TCP for ``observed`` joints, from the fitted TCP model."""
    from scipy.spatial.transform import Rotation

    links = [TCP_LINK.format(side=side) for side in SIDES]
    _, poses = derive_tracks(observed, observed, kinematics=kinematics, links=links)
    out = []
    for side in SIDES:
        model, link = TCP_MODEL[side], poses[TCP_LINK.format(side=side)]
        rotation = Rotation.from_matrix(link[:, :3, :3])
        position = (
            link[:, :3, 3] + rotation.apply(model["tool_m"]) + model["base_shift_m"]
        )
        tool = Rotation.from_rotvec(np.radians(model["tool_rotvec_deg"]))
        out += [position, (rotation * tool).as_rotvec()]
    return np.concatenate(out, axis=1), poses


def test_tcp_side_check_catches_swapped_arms(kinematics):
    observed = _in_limit_vector(kinematics, 8, 5)
    tcp, poses = _tcp_pose(kinematics, observed)
    result = tcp_side_check(tcp, poses)
    assert result["passed"], result["problems"]
    assert result["sides"]["left"]["rotation_error_deg_median"] < 1e-6
    swapped = tcp_side_check(np.concatenate([tcp[:, 6:], tcp[:, :6]], 1), poses)
    assert not swapped["passed"]
    assert any("look swapped" in p for p in swapped["problems"])
