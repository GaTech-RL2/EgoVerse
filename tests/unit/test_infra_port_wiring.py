"""Wiring checks for the hand-keypoint default: recipes, stems, the mecka
left-wrist fix in keypoint modes, and the packed-width loss reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

import egomimic
from egomimic.rldb.embodiment.human import Human

CONFIG_DIR = Path(egomimic.__file__).parent / "hydra_configs"
TRAIN_TOP_LEVEL = sorted(p.stem for p in CONFIG_DIR.glob("train_*.yaml"))
HUMAN = "human_bimanual"

# The all-cartesian cam-frame cotrain recipe is cotrain_pi_lang plus these
# overrides (model/pi0.5_cotrain_eva_aria_6d.yaml); there is no data config
# that differs from its parent only in the transform mode.
CAM_FRAME_COTRAIN = [
    "data.train_datasets.eva_bimanual.resolver.transform_list.mode=cartesian_6d",
    f"data.train_datasets.{HUMAN}.resolver.key_map.keymap_mode=cartesian",
    f"data.train_datasets.{HUMAN}.resolver.key_map.include_ee_pose=false",
    f"data.train_datasets.{HUMAN}.resolver.transform_list.mode=cartesian_6d",
    f"data.train_datasets.{HUMAN}.resolver.transform_list.include_ee_pose=false",
]


def _human_dataset_nodes(cfg):
    for split in (
        "train_datasets",
        "valid_datasets",
        "train_viz_datasets",
        "unseen_op_valid_datasets",
    ):
        node = (cfg.data.get(split) or {}).get(HUMAN)
        if node is not None and node.get("resolver") is not None:
            yield split, node.resolver


@pytest.mark.parametrize("top", TRAIN_TOP_LEVEL)
def test_human_action_width_matches_data_mode(top, compose_resolve):
    """A model declaring the 144-D human action must be fed the
    keypoints_*_6d data (and vice versa); a stale ypr/cartesian mode would
    only fail at the first forward pass."""
    cfg = compose_resolve(top, [])
    rm = cfg.model.robomimic_model
    if HUMAN not in (rm.get("domains") or []):
        pytest.skip(f"{top}: no human domain")
    dims = (rm.get("dims") or {}).get(HUMAN)
    ac_key = rm.ac_keys[HUMAN]
    for split, resolver in _human_dataset_nodes(cfg):
        mode = resolver.transform_list.mode
        keymap_mode = resolver.key_map.keymap_mode
        if ac_key == "actions_keypoints":
            assert keymap_mode == "keypoints", (top, split, keymap_mode)
            assert mode.startswith("keypoints_") and mode.endswith("_6d"), (
                top,
                split,
                mode,
            )
            if dims is not None:
                assert dims.action == 144, (top, split, dims)
        else:
            assert keymap_mode == "cartesian", (top, split, keymap_mode)
            assert mode.startswith("cartesian"), (top, split, mode)


@pytest.mark.parametrize(
    "model,data,extra",
    [
        ("hpt_bc_flow_aria", "aria", []),
        ("hpt_bc_flow_mecka", "mecka", []),
        ("hpt_bc_flow_scale", "scale", []),
        ("hpt_bc_flow_human", "human", []),
        ("pi0.5_bc_aria", "aria", []),
        ("pi0.5_bc_mecka", "mecka", []),
        ("pi0.5_bc_scale", "scale", []),
        ("pi0.5_cotrain_eva_aria", "cotrain_pi_base", []),
        ("pi0.5_cotrain_mecka_scale", "mecka_scale_cotrain_pi", []),
        ("pi0.5_cotrain_eva_aria_6d", "cotrain_pi_lang", CAM_FRAME_COTRAIN),
    ],
)
def test_vendor_pairings_agree_on_the_human_action(model, data, extra, compose_resolve):
    cfg = compose_resolve(
        "train_zarr_cartesian", [f"model={model}", f"data={data}", *extra]
    )
    rm = cfg.model.robomimic_model
    ac_key = rm.ac_keys[HUMAN]
    resolver = cfg.data.train_datasets[HUMAN].resolver
    mode = resolver.transform_list.mode
    if ac_key == "actions_keypoints":
        assert mode.endswith("_6d") and mode.startswith("keypoints_"), (model, mode)
        if rm.get("dims"):
            assert rm.dims[HUMAN].action == 144
        if "pi0.5" in model:
            assert rm.config.model.action_dim == 144
            assert "Keypoints" in rm.action_converters.rules.HUMAN_BIMANUAL._target_
            # the pi0.5 prompt state is the cartesian ee_pose
            assert resolver.key_map.get("include_ee_pose") is True
            assert resolver.transform_list.get("include_ee_pose") is True
    else:
        assert mode.startswith("cartesian"), (model, mode)
        if "pi0.5" in model:
            assert mode.endswith("_6d"), (model, mode)
            assert rm.config.model.action_dim == 32


def test_hpt_stems_follow_the_human_action(compose_resolve):
    """HPT human stems: keypoint models expose exactly ``state_keypoints``
    (a base's ``state_ee_pose`` is nulled out, not inherited)."""
    from egomimic.algo.hpt import HPTModel

    for model in ("hpt_bc_flow_mecka", "hpt_cotrain_flow_seperate_head"):
        cfg = compose_resolve("train_zarr_cartesian", [f"model={model}"])
        stems = cfg.model.robomimic_model.stem_specs[HUMAN]
        live = {k for k, v in stems.items() if v is not None}
        assert live == {"state_keypoints"}, (model, live)
    cfg = compose_resolve(
        "train_zarr_cartesian", ["model=hpt_cotrain_flow_shared_head"]
    )
    stems = cfg.model.robomimic_model.stem_specs[HUMAN]
    assert {k for k, v in stems.items() if v is not None} == {"state_ee_pose"}

    # the model side drops null stems instead of registering None
    m = HPTModel.__new__(HPTModel)
    torch.nn.Module.__init__(m)
    m.stem_spec, m.modalities, m.stems = {}, {}, {}
    m.init_domain_stem("human_bimanual", {"state_ee_pose": None, "x": object()})
    assert m.modalities["human_bimanual"] == ["x"]
    assert set(m.stems) == {"human_bimanual_x"}


# ------------------------------------------------------ mecka left-wrist fix
def _pose(rng):
    q = R.random(random_state=int(rng.integers(1 << 31))).as_quat()
    return np.concatenate([rng.uniform(-1, 1, 3), q[[3, 0, 1, 2]]])


def _chunk(rng, start, n):
    out = np.zeros((n, 7))
    p, r = start[:3].copy(), R.from_quat(start[[4, 5, 6, 3]])
    for t in range(n):
        if t:
            p = p + rng.normal(0, 0.01, 3)
            r = R.from_rotvec(rng.normal(0, 0.05, 3)) * r
        out[t] = np.concatenate([p, r.as_quat()[[3, 0, 1, 2]]])
    return out


def _rz180(pose7):
    """What the fixed converter would have written: the same pose with its
    local axes relabelled by Rz(180 deg)."""
    out = np.array(pose7, dtype=np.float64, copy=True)
    q = R.from_quat(out[..., [4, 5, 6, 3]]) * R.from_euler("z", np.pi)
    out[..., 3:7] = q.as_quat()[..., [3, 0, 1, 2]]
    return out


def _apply(tl, s):
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    for t in tl:
        s = t.transform(s)
    return s


def test_fix_mecka_left_wrist_equals_reconverting_in_keypoint_mode():
    rng = np.random.default_rng(31)
    H = Human.ACTION_HORIZON
    raw = {"obs_head_pose": _pose(rng)}
    for side in ("left", "right"):
        wrist = _pose(rng)
        raw[f"{side}.obs_wrist_pose"] = wrist
        raw[f"{side}.action_wrist_pose"] = _chunk(rng, wrist, H)
        kp = raw[f"{side}.action_wrist_pose"][:, None, :3] + rng.uniform(
            -0.1, 0.1, (H, 21, 3)
        )
        raw[f"{side}.action_keypoints"] = kp.reshape(H, 63)
        raw[f"{side}.obs_keypoints"] = kp[0].reshape(63)
        ee = _pose(rng)
        raw[f"{side}.obs_ee_pose"] = ee
        raw[f"{side}.action_ee_pose"] = _chunk(rng, ee, H)
    # "reconverted" twin: the fixed converter relabels the LEFT hand's axes on
    # every pose it writes (wrist_pose and ee_pose share the rotation).
    fixed = dict(raw)
    for k in (
        "left.obs_wrist_pose",
        "left.action_wrist_pose",
        "left.obs_ee_pose",
        "left.action_ee_pose",
    ):
        fixed[k] = _rz180(raw[k])

    kw = dict(stride=1, include_ee_pose=True, pad_proprio_gripper=True)
    with_flag = _apply(
        Human.get_transform_list(
            "keypoints_wristframe_6d", fix_mecka_left_wrist=True, **kw
        ),
        raw,
    )
    reconverted = _apply(
        Human.get_transform_list("keypoints_wristframe_6d", **kw), fixed
    )
    assert set(with_flag) == set(reconverted)
    for k in with_flag:
        np.testing.assert_allclose(with_flag[k], reconverted[k], atol=1e-9, err_msg=k)
    # and the flag really changes the left hand's frame
    unfixed = _apply(Human.get_transform_list("keypoints_wristframe_6d", **kw), raw)
    assert (
        np.abs(unfixed["actions_keypoints"] - with_flag["actions_keypoints"]).max()
        > 1e-3
    )
    # the same equivalence without the ee_pose side (plain HPT keypoint data)
    plain = {k: v for k, v in raw.items() if "ee_pose" not in k}
    plain_fixed = {k: v for k, v in fixed.items() if "ee_pose" not in k}
    a = _apply(
        Human.get_transform_list(
            "keypoints_wristframe_6d", stride=1, fix_mecka_left_wrist=True
        ),
        plain,
    )
    b = _apply(
        Human.get_transform_list("keypoints_wristframe_6d", stride=1), plain_fixed
    )
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-9, err_msg=k)


def test_pi_loss_is_reduced_over_the_packed_width():
    from egomimic.algo.pi import PI
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT
    from egomimic.utils.action_utils import (
        ConverterRegistry,
        HumanBimanualKeypoints,
        RobotBimanualCartesianEuler,
    )

    pi = PI.__new__(PI)
    pi.action_registry = ConverterRegistry()
    eva, human = EMBODIMENT.EVA_BIMANUAL.value, EMBODIMENT.HUMAN_BIMANUAL.value
    pi.action_registry.register(eva, "actions_cartesian", RobotBimanualCartesianEuler())
    pi.action_registry.register(human, "actions_keypoints", HumanBimanualKeypoints())
    losses = torch.zeros(2, 5, 144)
    losses[..., 32:] = 1.0  # error only in the zero-padded slots
    eva_loss = pi._reduce_loss(losses, torch.zeros(2, 5, 20), eva, "actions_cartesian")
    assert eva_loss.item() == 0.0
    kp_loss = pi._reduce_loss(
        losses, torch.zeros(2, 5, 144), human, "actions_keypoints"
    )
    assert abs(kp_loss.item() - 112 / 144) < 1e-6
    # already-reduced losses pass through unchanged
    assert (
        pi._reduce_loss(torch.tensor([1.0, 3.0]), None, eva, "actions_cartesian").item()
        == 2.0
    )
