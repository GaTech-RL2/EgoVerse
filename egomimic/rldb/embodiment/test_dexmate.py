"""Check the delivered models, palm frames and current-camera action chunks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egomimic.rldb.embodiment import Embodiment
from egomimic.rldb.embodiment.hand_kinematics import (
    fk_keypoints,
    keypoint_residuals,
    load_chain,
)
from egomimic.rldb.embodiment.registry import RegistryError, load_platforms
from egomimic.rldb.embodiment.urdf import load_urdf
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz, _xyzwxyz_to_matrix


def test_defaults_follow_active_sides_and_explicit_morphology():
    resolved = Embodiment.resolve("dexmate_bimanual")
    for side in resolved.sides:
        assert resolved.end_effectors[side].name == f"sharpa_wave_v1_{side}"
    assert set(Embodiment.resolve("eva_left_arm").end_effectors) == {"left"}
    explicit = Embodiment.from_attrs(
        {
            "embodiment": "eva_left_arm",
            "morphology": {"platform": "eva_x5", "end_effector": {"left": "mano_hand"}},
        }
    )
    assert explicit.end_effectors["left"].name == "mano_hand"
    with pytest.raises(ValueError, match="active sides"):
        Embodiment.from_attrs(
            {
                "embodiment": "eva_left_arm",
                "morphology": {
                    "platform": "eva_x5",
                    "end_effector": {"right": "mano_hand"},
                },
            }
        )


def test_morphology_defaults_follow_platform_arity_without_a_name(monkeypatch):
    platform = replace(
        load_platforms()["eva_x5"],
        arity=("left_arm",),
        default_end_effector={"left": "eva_parallel_jaw"},
    )
    monkeypatch.setattr(
        "egomimic.rldb.embodiment.embodiment.load_platforms",
        lambda: {platform.name: platform},
    )
    assert set(Embodiment.resolve({"platform": platform.name}).end_effectors) == {
        "left"
    }
    with pytest.raises(ValueError, match="unsupported arity"):
        Embodiment.resolve(
            {
                "platform": platform.name,
                "end_effector": {"left": "mano_hand", "right": "mano_hand"},
            }
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_palm_pose_matches_keypoints_and_root_convention_remains_available(side):
    spec = Embodiment.resolve("dexmate_bimanual").end_effectors[side]
    chain = load_chain(spec)
    assert chain.actuated_joint_names == spec.joint_names
    joints = np.linspace(0, 0.3, 66).reshape(3, 22)
    palm = fk_keypoints(spec, joints)
    np.testing.assert_allclose(palm[:, 0], 0, atol=1e-15)
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[:, :3, :3] = Rotation.from_euler("xyz", [0.4, -0.2, 0.8]).as_matrix()
    transforms[:, :3, 3] = [0.3, 0.2, 0.8]
    stored = (
        np.einsum("tij,tkj->tki", transforms[:, :3, :3], palm)
        + transforms[:, None, :3, 3]
    )
    poses = _matrix_to_xyzwxyz(transforms)
    np.testing.assert_allclose(
        keypoint_residuals(spec, joints, stored.reshape(3, 63), poses), 0, atol=1e-12
    )
    root_spec = replace(spec, ee_pose_link=None)
    assert (
        keypoint_residuals(root_spec, joints, stored.reshape(3, 63), poses).max()
        > 0.029
    )
    expected = chain.link_positions(
        dict(zip(spec.joint_names, joints[0])), list(spec.keypoint_link_map.values())
    )
    np.testing.assert_allclose(fk_keypoints(root_spec, joints)[0], expected)


def test_models_are_optional_and_hashes_are_checked_at_asset_load(tmp_path):
    spec = load_platforms()["dexmate_nth_poc1"]
    assert len(load_urdf(spec.asset_path(verify=True)).actuated_joint_names) == 21
    bad = replace(spec, urdf_sha256="0" * 64)
    assert bad.urdf_path.is_file()  # resolution does not load the model
    with pytest.raises(RegistryError, match="SHA-256 mismatch"):
        bad.asset_path(verify=True)
    optional = replace(spec, urdf=None, urdf_sha256=None)
    assert optional.asset_path(verify=True) is None
    model = tmp_path / "model.xml"
    model.write_text("<mujoco/>")
    assert replace(optional, robot_model=str(model)).robot_model_path == model


@pytest.mark.parametrize("quat,width", [(True, 140), (False, 138)])
def test_moving_camera_anchors_whole_chunk_without_fk(quat, width, monkeypatch):
    resolved = Embodiment.resolve("dexmate_bimanual")
    keymap = resolved.get_keymap("keypoints")
    assert "horizon" not in keymap["obs_head_pose"]
    assert keymap["left.action_wrist_pose"]["zarr_key"] == "left.obs_ee_pose"
    assert keymap["left.action_keypoints"]["zarr_key"] == "left.obs_hand_keypoints"
    cameras = np.repeat(np.eye(4)[None], 31, axis=0)
    cameras[:, :3, :3] = Rotation.from_euler(
        "z", np.linspace(0.2, 0.8, 31)[:, None]
    ).as_matrix()
    cameras[:, :3, 3] = np.arange(31)[:, None] * [0.01, -0.02, 0.005]
    poses = np.tile([0.3, 0.1, 1.0, 1, 0, 0, 0], (31, 1))
    poses[:, 0] += np.arange(31) * 0.01
    points = np.tile(poses[:, :3, None], (1, 1, 21)).transpose(0, 2, 1)
    monkeypatch.setattr(
        "egomimic.rldb.embodiment.hand_kinematics.load_chain",
        lambda *a: pytest.fail("loader opened URDF"),
    )
    for start in (0, 1):
        batch = {"obs_head_pose": _matrix_to_xyzwxyz(cameras)[start]}
        for side in ("left", "right"):
            batch.update(
                {
                    f"{side}.action_keypoints": points[start : start + 30].reshape(
                        30, 63
                    ),
                    f"{side}.obs_keypoints": points[start].reshape(63),
                    f"{side}.action_wrist_pose": poses[start : start + 30],
                    f"{side}.obs_wrist_pose": poses[start],
                }
            )
        for transform in resolved.get_transform_list(
            f"keypoints_headframe_{'quat' if quat else 'ypr'}"
        ):
            batch = transform.transform(batch)
        actions = batch["actions_keypoints"]
        assert actions.shape == (100, width)
        expected = np.linalg.inv(cameras[start]) @ _xyzwxyz_to_matrix(
            poses[[start, start + 29]]
        )
        np.testing.assert_allclose(actions[[0, -1], :3], expected[:, :3, 3], atol=1e-6)
        pose_width = 7 if quat else 6
        np.testing.assert_allclose(
            actions[[0, -1], pose_width : pose_width + 3], expected[:, :3, 3], atol=1e-6
        )
