"""Independent end-to-end checks of the 144-D wrist-frame hand-keypoint
pipeline (the default human action):

  raw world-frame keypoints + wrist / head poses
    -> Human.get_transform_list("keypoints_wristframe_6d", include_ee_pose=True)
    -> actions_keypoints (T, 144), observations.state.keypoints (144,),
       observations.state.ee_pose (18 / 20,)
    -> _build_human_keypoints_revert_6d_wristframe_transform_list (evaluator)
    -> 126-D head-frame keypoints  ==  independent inv(T_head) @ kp_world

Ground truth is plain numpy/scipy SE(3) math sharing no code with the
pipeline. Interpolation endpoints are exact, so the chunk is compared at
t = 0 and t = -1 against raw frames 0 and horizon-1.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_keypoints_revert_6d_transform_list,
    _build_human_keypoints_revert_6d_wristframe_transform_list,
)
from egomimic.rldb.zarr.action_chunk_transforms import (
    KeypointsRot6DToYPR,
    KeypointsYPRToRot6D,
)
from egomimic.utils.pose_utils import bimanual_keypoint_layout

HORIZON = Human.ACTION_HORIZON  # raw frames per chunk (30)
CHUNK = 100


def _rng(seed):
    return np.random.default_rng(seed)


def _rand_pose(rng, scale=1.0):
    q = R.random(random_state=int(rng.integers(1 << 31))).as_quat()  # xyzw
    return np.concatenate([rng.uniform(-scale, scale, 3), q[[3, 0, 1, 2]]])


def _rand_chunk(rng, start, n):
    out = np.zeros((n, 7))
    p = start[:3].copy()
    r = R.from_quat(start[[4, 5, 6, 3]])
    for t in range(n):
        if t > 0:
            p = p + rng.normal(0, 0.01, 3)
            r = R.from_rotvec(rng.normal(0, 0.05, 3)) * r
        q = r.as_quat()
        out[t] = np.concatenate([p, q[[3, 0, 1, 2]]])
    return out


def _T(p7):
    M = np.eye(4)
    M[:3, :3] = R.from_quat(p7[[4, 5, 6, 3]]).as_matrix()
    M[:3, 3] = p7[:3]
    return M


def _in_frame(T_frame, pts):
    """Points (..., 3) world -> frame."""
    inv = np.linalg.inv(T_frame)
    return pts @ inv[:3, :3].T + inv[:3, 3]


def _apply(transform_list, sample):
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
    for t in transform_list:
        s = t.transform(s)
    return s


def _raw_sample(rng):
    head = _rand_pose(rng)
    raw = {"obs_head_pose": head}
    world = {}
    for side in ("left", "right"):
        wrist_obs = _rand_pose(rng)
        wrist_act = _rand_chunk(rng, wrist_obs, HORIZON)  # (30, 7) world
        offsets = rng.uniform(-0.1, 0.1, size=(HORIZON, 21, 3))
        kp_act = wrist_act[:, None, :3] + offsets  # (30, 21, 3) world
        kp_obs = kp_act[0]
        ee_obs = _rand_pose(rng)
        raw[f"{side}.obs_wrist_pose"] = wrist_obs
        raw[f"{side}.action_wrist_pose"] = wrist_act
        raw[f"{side}.obs_keypoints"] = kp_obs.reshape(63)
        raw[f"{side}.action_keypoints"] = kp_act.reshape(HORIZON, 63)
        raw[f"{side}.obs_ee_pose"] = ee_obs
        raw[f"{side}.action_ee_pose"] = _rand_chunk(rng, ee_obs, HORIZON)
        world[side] = (wrist_obs, wrist_act, kp_obs, kp_act)
    return raw, head, world


def test_keypoint_transform_round_trips_138_144():
    rng = _rng(0)
    x = np.concatenate(
        [
            rng.uniform(-1, 1, (5, 3)),
            rng.uniform(-1, 1, (5, 3)),
            rng.uniform(-1, 1, (5, 63)),
            rng.uniform(-1, 1, (5, 3)),
            rng.uniform(-1, 1, (5, 3)),
            rng.uniform(-1, 1, (5, 63)),
        ],
        axis=-1,
    )
    assert x.shape == (5, 138)
    y = KeypointsYPRToRot6D(action_key="k").transform({"k": x.copy()})["k"]
    assert y.shape == (5, 144)
    layout = bimanual_keypoint_layout(144)
    # keypoint blocks pass through untouched
    np.testing.assert_allclose(y[:, list(layout["keypoints"])][:, :63], x[:, 6:69])
    back = KeypointsRot6DToYPR(action_key="k").transform({"k": y})["k"]
    np.testing.assert_allclose(back, x, atol=1e-9)
    # a single proprio vector works the same way
    single = KeypointsYPRToRot6D(action_key="k").transform({"k": x[0].copy()})["k"]
    assert single.shape == (144,)
    t = KeypointsYPRToRot6D(action_key="k").transform({"k": torch.from_numpy(x)})["k"]
    assert isinstance(t, torch.Tensor) and t.shape == (5, 144)


@pytest.mark.parametrize("pad", [False, True])
def test_keypoints_wristframe_6d_pipeline_matches_independent_math(pad):
    rng = _rng(3 + int(pad))
    raw, head, world = _raw_sample(rng)
    fwd = Human.get_transform_list(
        "keypoints_wristframe_6d",
        stride=1,
        include_ee_pose=True,
        pad_proprio_gripper=pad,
    )
    out = _apply(fwd, raw)
    assert set(out) == {
        "actions_keypoints",
        "observations.state.keypoints",
        "observations.state.ee_pose",
    }, sorted(out)
    act = np.asarray(out["actions_keypoints"])
    obs = np.asarray(out["observations.state.keypoints"])
    ee = np.asarray(out["observations.state.ee_pose"])
    assert act.shape == (CHUNK, 144) and obs.shape == (144,)
    assert ee.shape == ((20,) if pad else (18,))
    if pad:
        assert ee[9] == 0.0 and ee[19] == 0.0

    layout = bimanual_keypoint_layout(144)
    per_hand = layout["per_hand"]
    Th = _T(head)
    for si, side in enumerate(("left", "right")):
        wrist_obs, wrist_act, kp_obs, kp_act = world[side]
        Tw = _T(wrist_obs)
        o = si * per_hand
        # proprio: wrist pose in HEAD frame (xyz + rot6d), keypoints in the
        # wrist's own frame
        Twh = np.linalg.inv(Th) @ Tw
        np.testing.assert_allclose(obs[o : o + 3], Twh[:3, 3], atol=1e-9)
        np.testing.assert_allclose(obs[o + 3 : o + 6], Twh[:3, 0], atol=1e-9)
        np.testing.assert_allclose(obs[o + 6 : o + 9], Twh[:3, 1], atol=1e-9)
        np.testing.assert_allclose(
            obs[o + 9 : o + per_hand].reshape(21, 3), _in_frame(Tw, kp_obs), atol=1e-9
        )
        # action: wrist pose relative to the obs wrist (identity at t=0),
        # keypoints in the obs-wrist frame; endpoints are exact under
        # interpolation.
        for t_chunk, t_raw in ((0, 0), (CHUNK - 1, HORIZON - 1)):
            Ta = _T(wrist_act[t_raw])
            Trel = np.linalg.inv(Tw) @ Ta
            row = act[t_chunk]
            np.testing.assert_allclose(row[o : o + 3], Trel[:3, 3], atol=1e-6)
            np.testing.assert_allclose(row[o + 3 : o + 6], Trel[:3, 0], atol=1e-6)
            np.testing.assert_allclose(row[o + 6 : o + 9], Trel[:3, 1], atol=1e-6)
            np.testing.assert_allclose(
                row[o + 9 : o + per_hand].reshape(21, 3),
                _in_frame(Tw, kp_act[t_raw]),
                atol=1e-6,
            )
    np.testing.assert_allclose(act[0, list(layout["wrist_xyz"])], 0.0, atol=1e-12)

    # evaluator revert: back to 126-D head-frame keypoints (batched)
    rev = _build_human_keypoints_revert_6d_wristframe_transform_list()
    reverted = Embodiment.apply_transform(
        {
            "actions_keypoints": torch.from_numpy(act[None]).float(),
            "observations.state.keypoints": torch.from_numpy(obs[None]).float(),
        },
        rev,
    )
    kp_head = np.asarray(reverted["actions_keypoints"])
    assert kp_head.shape == (1, CHUNK, 126)
    for si, side in enumerate(("left", "right")):
        _, _, _, kp_act = world[side]
        for t_chunk, t_raw in ((0, 0), (CHUNK - 1, HORIZON - 1)):
            np.testing.assert_allclose(
                kp_head[0, t_chunk, 63 * si : 63 * si + 63].reshape(21, 3),
                _in_frame(Th, kp_act[t_raw]),
                atol=1e-4,
            )


def test_keypoints_headframe_6d_pipeline_and_revert():
    rng = _rng(9)
    raw, head, world = _raw_sample(rng)
    for k in list(raw):
        if "ee_pose" in k:
            del raw[k]
    fwd = Human.get_transform_list("keypoints_headframe_6d", stride=1)
    out = _apply(fwd, raw)
    assert set(out) == {"actions_keypoints", "observations.state.keypoints"}
    act = np.asarray(out["actions_keypoints"])
    assert act.shape == (CHUNK, 144)
    Th = _T(head)
    layout = bimanual_keypoint_layout(144)
    per_hand = layout["per_hand"]
    for si, side in enumerate(("left", "right")):
        _, wrist_act, _, kp_act = world[side]
        o = si * per_hand
        Tah = np.linalg.inv(Th) @ _T(wrist_act[0])
        np.testing.assert_allclose(act[0, o : o + 3], Tah[:3, 3], atol=1e-6)
        np.testing.assert_allclose(act[0, o + 3 : o + 6], Tah[:3, 0], atol=1e-6)
        np.testing.assert_allclose(
            act[0, o + 9 : o + per_hand].reshape(21, 3),
            _in_frame(Th, kp_act[0]),
            atol=1e-6,
        )
    rev = _build_human_keypoints_revert_6d_transform_list()
    reverted = Embodiment.apply_transform(
        {
            "actions_keypoints": torch.from_numpy(act[None]).float(),
            "observations.state.keypoints": torch.from_numpy(
                np.asarray(out["observations.state.keypoints"])[None]
            ).float(),
        },
        rev,
    )
    assert np.asarray(reverted["actions_keypoints"]).shape == (1, CHUNK, 138)
    assert np.asarray(reverted["observations.state.keypoints"]).shape == (1, 138)


def test_keypoint_modes_without_ee_pose_emit_no_ee_pose():
    rng = _rng(12)
    raw, _, _ = _raw_sample(rng)
    for k in list(raw):
        if "ee_pose" in k:
            del raw[k]
    for mode in ("keypoints_wristframe_6d", "keypoints_wristframe_ypr"):
        out = _apply(
            Human.get_transform_list(mode, stride=1, allow_legacy_rotation=True), raw
        )
        assert set(out) == {"actions_keypoints", "observations.state.keypoints"}
    ypr = _apply(
        Human.get_transform_list(
            "keypoints_wristframe_ypr", stride=1, allow_legacy_rotation=True
        ),
        raw,
    )
    assert np.asarray(ypr["actions_keypoints"]).shape == (CHUNK, 138)


def test_keymap_include_ee_pose_adds_the_ee_pose_keys():
    km = Human.get_keymap("keypoints")
    assert "left.obs_ee_pose" not in km
    km = Human.get_keymap("keypoints", include_ee_pose=True)
    for side in ("left", "right"):
        assert km[f"{side}.obs_ee_pose"]["key_type"] == "proprio_keys"
        assert km[f"{side}.action_ee_pose"]["key_type"] == "action_keys"
        assert km[f"{side}.action_ee_pose"]["horizon"] == HORIZON
    with pytest.raises(ValueError, match="include_ee_pose"):
        Human.get_transform_list("cartesian_6d", include_ee_pose=True)
    with pytest.raises(ValueError, match="pad_proprio_gripper"):
        Human.get_transform_list("keypoints_wristframe_6d", pad_proprio_gripper=True)
