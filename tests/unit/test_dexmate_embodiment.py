"""dexmate_bimanual: a valid embodiment with its own id (the existing ids are
unchanged; norm-stat caches and precomputed stats are keyed by them), and the
Dexmate keymap / transform list on synthetic episodes."""

import numpy as np
import pytest
import zarr
from fixtures.synthetic_episodes import write_dexmate_episode

from egomimic.rldb.embodiment.dexmate import Dexmate
from egomimic.rldb.embodiment.embodiment import (
    EMBODIMENT,
    get_embodiment,
    get_embodiment_id,
)
from egomimic.rldb.zarr import dexmate_rules
from egomimic.rldb.zarr.episode_rules import EMBODIMENTS, arm_sides, embodiment_error
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.utils.pose_utils import _xyzwxyz_to_matrix


def test_existing_ids_are_unchanged_and_dexmate_is_8():
    assert {e.name: e.value for e in EMBODIMENT} == {
        "HUMAN_RIGHT_ARM": 1,
        "HUMAN_LEFT_ARM": 2,
        "HUMAN_BIMANUAL": 3,
        "EVA_RIGHT_ARM": 4,
        "EVA_LEFT_ARM": 5,
        "EVA_BIMANUAL": 6,
        "DEXMATE_BIMANUAL": 8,
    }


def test_dexmate_name_round_trips():
    assert dexmate_rules.EMBODIMENT == "dexmate_bimanual"
    assert dexmate_rules.EMBODIMENT in EMBODIMENTS
    assert embodiment_error("dexmate_bimanual") is None
    assert get_embodiment_id("dexmate_bimanual") == 8
    assert get_embodiment(8) == "DEXMATE_BIMANUAL"
    assert arm_sides("dexmate_bimanual") == dexmate_rules.SIDES


def test_keymap_reads_dexmate_keys_into_human_batch_names():
    km = Dexmate.get_keymap("keypoints")
    assert km["left.action_keypoints"]["zarr_key"] == "left.obs_hand_keypoints"
    assert km["right.obs_keypoints"]["zarr_key"] == "right.obs_hand_keypoints"
    assert km["left.action_wrist_pose"]["zarr_key"] == "left.obs_ee_pose"
    assert km["right.obs_wrist_pose"]["zarr_key"] == "right.obs_ee_pose"
    assert km[dexmate_rules.HEAD_LINK_POSE] == {
        "key_type": "proprio_keys",
        "zarr_key": dexmate_rules.HEAD_LINK_POSE,
    }
    assert "obs_head_pose" not in km
    cameras = {
        k: v["zarr_key"] for k, v in km.items() if v["key_type"] == "camera_keys"
    }
    assert cameras == {
        Dexmate.VIZ_IMAGE_KEY: "images.front_1",
        "observations.images.left_wrist_img": "images.left_wrist",
        "observations.images.right_wrist_img": "images.right_wrist",
    }
    no_wrists = Dexmate.get_keymap("keypoints", wrist_cameras=False)
    assert [k for k in no_wrists if "wrist_img" in k] == []
    norm = Dexmate.get_keymap("keypoints", norm_mode=True, annotation_key="annotations")
    assert {v["key_type"] for v in norm.values()} == {"proprio_keys", "action_keys"}


@pytest.mark.parametrize("mode", ["cartesian", "keypoints_pi"])
def test_keymap_has_only_the_keypoints_mode(mode):
    with pytest.raises(ValueError, match="Unsupported keymap_mode"):
        Dexmate.get_keymap(mode)


@pytest.mark.parametrize("mode", ["keypoints_wristframe_quat", "cartesian_6d"])
def test_transform_list_has_only_head_frame_keypoint_modes(mode):
    with pytest.raises(ValueError, match="Unsupported transform_list mode"):
        Dexmate.get_transform_list(mode)


def _in_frame(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``points`` (N, 3) in the frame of the xyz+wxyz ``pose``."""
    inv = np.linalg.inv(_xyzwxyz_to_matrix(pose[None])[0])
    return points @ inv[:3, :3].T + inv[:3, 3]


@pytest.mark.parametrize(
    "mode, width", [("keypoints_headframe_quat", 140), ("keypoints_headframe_ypr", 138)]
)
@pytest.mark.parametrize("wrist_cameras", [True, False])
def test_padded_episode_with_a_moving_head_loads_in_the_head_link_frame(
    tmp_path, mode, width, wrist_cameras
):
    T = 48  # stored padded to 100 rows of zeros
    angle = np.linspace(0.0, 0.6, T)
    head = np.zeros((T, 7))
    head[:, 0] = np.linspace(0.0, 0.2, T)
    head[:, 2] = 1.2
    head[:, 3] = np.cos(angle / 2)  # wxyz, a turn about z
    head[:, 6] = np.sin(angle / 2)
    # The hands move too (a drift plus a turn about x), so a chunk that repeated
    # the current hand configuration would fail the future-frame checks below.
    t = np.arange(T)[:, None]
    moving = {dexmate_rules.HEAD_LINK_POSE: head}
    for i, side in enumerate(dexmate_rules.SIDES):
        palm = np.zeros((T, 7))
        palm[:, :3] = [0.4, 0.2 - 0.4 * i, 0.9] + 0.004 * t * [1.0, 0.5, -0.5]
        palm[:, 3] = np.cos(0.01 * (i + 1) * t[:, 0])  # wxyz, a turn about x
        palm[:, 4] = np.sin(0.01 * (i + 1) * t[:, 0])
        offsets = np.random.default_rng(i).normal(0, 0.05, (21, 3))
        kp = (palm[:, None, :3] + offsets + 0.002 * t[:, :, None]).reshape(T, 63)
        moving.update(
            {
                f"{side}.obs_ee_pose": palm,
                f"{side}.cmd_ee_pose": palm,
                f"{side}.obs_hand_keypoints": kp,
                f"{side}.cmd_hand_keypoints": kp,
            }
        )
    ep = write_dexmate_episode(
        tmp_path,
        T=T,
        H=32,
        W=32,
        wrist_cameras=wrist_cameras,
        arrays=moving,
    )
    store = zarr.open_group(str(ep), mode="r")
    assert store[dexmate_rules.HEAD_LINK_POSE].shape[0] > T
    keypoints = {
        side: store[f"{side}.obs_hand_keypoints"][:T].reshape(T, 21, 3)
        for side in dexmate_rules.SIDES
    }
    palms = {side: store[f"{side}.obs_ee_pose"][:T] for side in dexmate_rules.SIDES}
    resolver = LocalEpisodeResolver(
        tmp_path,
        key_map=Dexmate.get_keymap(
            "keypoints", wrist_cameras=wrist_cameras, annotation_key="annotations"
        ),
        transform_list=Dexmate.get_transform_list(mode, stride=1),
    )
    ds = MultiDataset._from_resolver(resolver, mode="total")
    pose_width = (width - 2 * 63) // 2  # 7 (quaternion) or 6 (YPR)
    for idx in (0, T - 5):  # T - 5: the chunk runs past the last frame
        sample = ds[idx]
        obs = np.asarray(sample["observations.state.keypoints"])
        actions = np.asarray(sample["actions_keypoints"])
        assert obs.shape == (width,) and actions.shape == (100, width)
        assert np.isfinite(actions).all()
        images = sorted(k for k in sample if k.startswith("observations.images."))
        assert len(images) == (3 if wrist_cameras else 1), images
        assert dexmate_rules.HEAD_LINK_POSE not in sample  # consumed
        last = min(idx + Dexmate.ACTION_HORIZON, T) - 1
        for i, side in enumerate(dexmate_rules.SIDES):
            start = i * (pose_width + 63)
            kp = slice(start + pose_width, start + pose_width + 63)
            np.testing.assert_allclose(
                obs[kp], _in_frame(head[idx], keypoints[side][idx]).ravel(), atol=1e-5
            )
            np.testing.assert_allclose(
                obs[start : start + 3],
                _in_frame(head[idx], palms[side][idx, None, :3])[0],
                atol=1e-5,
            )
            # The whole chunk is in the CURRENT head frame; its last step is the
            # future frame `last` (past the episode end it repeats the last
            # frame, never the zero padding).
            np.testing.assert_allclose(actions[0], obs, atol=1e-5)
            assert not np.allclose(actions[-1, kp], obs[kp], atol=1e-3)
            np.testing.assert_allclose(
                actions[-1, kp],
                _in_frame(head[idx], keypoints[side][last]).ravel(),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                actions[-1, start : start + 3],
                _in_frame(head[idx], palms[side][last, None, :3])[0],
                atol=1e-5,
            )
            if pose_width == 7:  # head_T_palm = inverse(base_T_head) @ base_T_palm
                head_T_palm = np.linalg.inv(
                    _xyzwxyz_to_matrix(head[idx][None])[0]
                ) @ _xyzwxyz_to_matrix(palms[side][[idx, last]])
                for step, want in (
                    (obs, head_T_palm[0]),
                    (actions[-1], head_T_palm[1]),
                ):
                    got = _xyzwxyz_to_matrix(step[None, start : start + 7])[0]
                    np.testing.assert_allclose(got[:3, :3], want[:3, :3], atol=1e-5)
