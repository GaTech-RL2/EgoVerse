"""An untracked hand (Lance `action_hand_mask` / `proprio_hand_valid`) is left out of
the loss, the norm stats and the bounds check, and reads as the mean pose in proprio."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from egomimic.algo.hpt import HPT
from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment_id
from egomimic.rldb.multi import MultiDataset, hand_keep
from egomimic.utils.pose_utils import hand_channel_mask

LEFT_ONLY = torch.tensor([[1.0, 0.0], [1.0, 0.0]])


def test_hand_channel_mask_splits_bimanual_layouts_in_halves():
    for width in (14, 20, 144):
        m = hand_channel_mask(np.array([1.0, 0.0]), width)
        assert m[: width // 2].all() and not m[width // 2 :].any()
    assert hand_channel_mask(np.array([1.0, 0.0]), 7) is None


class _FixedPred(DenoisingPolicy):
    def __init__(self, pred):
        torch.nn.Module.__init__(self)
        self.action_horizon = pred.shape[1]
        self.pooling = None
        self.padding = None
        self._pred = pred

    def predict(self, actions, global_cond):
        return self._pred, actions


def test_denoising_loss_leaves_out_the_untracked_hand():
    g = torch.Generator().manual_seed(0)
    target, pred = (
        torch.randn(2, 4, 144, generator=g),
        torch.randn(2, 4, 144, generator=g),
    )
    policy = _FixedPred(pred)
    cond = torch.zeros(2, 3)
    pad = torch.ones(2, 4, 1)
    loss = policy.compute_loss(
        cond, {"action": target, "pad_mask": pad, "action_hand_mask": LEFT_ONLY}
    )
    torch.testing.assert_close(loss, F.mse_loss(pred[..., :72], target[..., :72]))
    both = policy.compute_loss(
        cond, {"action": target, "pad_mask": pad, "action_hand_mask": torch.ones(2, 2)}
    )
    torch.testing.assert_close(
        both, policy.compute_loss(cond, {"action": target, "pad_mask": pad})
    )


def test_denoising_loss_refuses_a_hand_mask_it_cannot_place():
    policy = _FixedPred(torch.zeros(2, 4, 7))
    with pytest.raises(ValueError, match="bimanual layout"):
        policy.compute_loss(
            torch.zeros(2, 3),
            {"action": torch.zeros(2, 4, 7), "action_hand_mask": LEFT_ONLY},
        )


class _NoRename:
    def zarr_key_to_keyname(self, zarr_key, embodiment_id):
        return None


def test_hpt_process_batch_keeps_the_hand_mask():
    algo = HPT.__new__(HPT)
    algo.norm_stats = _NoRename()
    algo.device = "cpu"
    algo.annotation_key = None
    algo.use_pad_mask = False
    emb = "human_bimanual"
    algo.ac_keys = {get_embodiment_id(emb): "actions_keypoints"}
    out = HPT.process_batch_for_training(
        algo,
        {
            emb: {
                "actions_keypoints": torch.zeros(2, 6, 144),
                "action_hand_mask": LEFT_ONLY,
                "proprio_hand_valid": LEFT_ONLY,
            }
        },
    )[get_embodiment_id(emb)]
    torch.testing.assert_close(out["action_hand_mask"], LEFT_ONLY)
    assert "proprio_hand_valid" not in out


def test_hpt_heads_get_the_hand_mask():
    algo = HPT.__new__(HPT)
    algo.is_6dof = True
    algo.shared_ac_key = None
    g = torch.Generator().manual_seed(0)
    target, pred = (
        torch.randn(2, 4, 144, generator=g),
        torch.randn(2, 4, 144, generator=g),
    )
    batch = {
        "actions_keypoints": target,
        "pad_mask": torch.ones(2, 4, 1),
        "embodiment": torch.zeros(2),
        "action_hand_mask": LEFT_ONLY,
    }
    policy, cond = _FixedPred(pred), torch.zeros(2, 3)

    def loss(b):
        data = HPT._robomimic_to_hpt_data(
            algo, b, [], [], [], "actions_keypoints", domain="human_bimanual"
        )
        return policy.compute_loss(cond, data)

    moved = target.clone()
    moved[..., 72:] += 5.0
    torch.testing.assert_close(loss(batch), loss({**batch, "actions_keypoints": moved}))


def _pi():
    from egomimic.algo.pi import PI
    from egomimic.utils.action_utils import (
        ConverterRegistry,
        HumanBimanualCartesian6D,
        HumanBimanualKeypoints,
    )

    pi = PI.__new__(PI)
    pi.action_registry = ConverterRegistry()
    pi._packed_widths = {}
    human = EMBODIMENT.HUMAN_BIMANUAL.value
    pi.action_registry.register(human, "actions_keypoints", HumanBimanualKeypoints())
    pi.action_registry.register(human, "actions_cartesian", HumanBimanualCartesian6D())
    return pi, human


def test_pi_loss_leaves_out_the_untracked_hand():
    pytest.importorskip("openpi")
    pi, human = _pi()
    losses = torch.zeros(2, 5, 160)
    losses[..., 72:144] = 1.0  # error only on the right hand
    kp = pi._reduce_loss(
        losses, torch.zeros(2, 5, 144), human, "actions_keypoints", LEFT_ONLY
    )
    assert kp.item() == 0.0
    plain = pi._reduce_loss(losses, torch.zeros(2, 5, 144), human, "actions_keypoints")
    both = pi._reduce_loss(
        losses, torch.zeros(2, 5, 144), human, "actions_keypoints", torch.ones(2, 2)
    )
    torch.testing.assert_close(both, plain)
    # Cartesian packs [L 9 | g | R 9 | g | pad 12]; the gripper slots stay in.
    losses = torch.zeros(2, 5, 32)
    losses[..., 0:10] = 1.0
    right_only = LEFT_ONLY.flip(-1)
    cart = pi._reduce_loss(
        losses, torch.zeros(2, 5, 18), human, "actions_cartesian", right_only
    )
    assert abs(cart.item() - 1 / 23) < 1e-6


def _stats_md():
    md = MultiDataset(state={})
    emb = get_embodiment_id("human_bimanual")
    X = np.random.default_rng(0).normal(size=(64, 144)).astype(np.float32)
    md.norm_mode = "zscore"
    md.norm_stats = {
        emb: {
            "state": MultiDataset._compute_stats_for_array(X),
            "act": MultiDataset._compute_stats_for_array(X[:, None, :]),
        }
    }
    md.zarr_keys = {
        emb: {"state": "observations.state.keypoints", "act": "actions_keypoints"}
    }
    md.key_types = {emb: {"state": "proprio_keys", "act": "action_keys"}}
    return md, emb


def test_untracked_proprio_reads_as_the_mean_pose():
    md, emb = _stats_md()
    data = {
        "embodiment": emb,
        "observations.state.keypoints": torch.full((144,), 5.0),
        "proprio_hand_valid": torch.tensor([1.0, 0.0]),
    }
    out = md.normalize(data, emb)["observations.state.keypoints"]
    assert (out[:72] != 0).all()
    rot = list(range(75, 81))
    other = [c for c in range(72, 144) if c not in rot]
    assert (out[other] == 0).all()
    # rot6d passes through normalization: its neutral value is the mean, not 0.
    mean = md.norm_stats[emb]["state"]["mean"]
    np.testing.assert_allclose(out[rot].numpy(), mean[rot], rtol=1e-6)


def test_bounds_check_skips_the_untracked_hand():
    md, emb = _stats_md()
    act = torch.zeros(1, 144)
    act[0, 100] = 1e6  # an identity-pose fill value far outside the stats
    data = {"embodiment": emb, "actions_keypoints": act}
    assert md._check_bounds(dict(data), None, 0, "ep") is not None
    data["action_hand_mask"] = torch.tensor([1.0, 0.0])
    assert md._check_bounds(data, None, 0, "ep") is None


def test_norm_stats_use_only_tracked_cells():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3, 144)).astype(np.float32)
    flags = np.ones((40, 2), np.float32)
    flags[:10, 1] = 0.0
    X[:10, :, 72:] = 1e6  # fill values of the untracked right hand
    X[20, 0, 0] = np.nan  # a genuinely bad sample is still dropped
    keep = hand_keep(
        {"action_hand_mask": torch.from_numpy(flags)},
        "action_keys",
        "actions_keypoints",
        torch.from_numpy(X),
    )
    keep = np.broadcast_to(keep.numpy(), X.shape)
    kept = MultiDataset._drop_nonfinite_rows(X, "actions_keypoints", keep)
    assert kept.shape[0] == 39
    stats = MultiDataset._compute_stats_for_array(kept)
    good = np.delete(X, 20, axis=0)
    np.testing.assert_allclose(
        stats["mean"][:, 72:], good[10:, :, 72:].mean(0), rtol=1e-5
    )
    np.testing.assert_allclose(
        stats["mean"][:, :72], good[:, :, :72].mean(0), rtol=1e-5
    )
    assert all(np.isfinite(a).all() for a in stats.values())


def test_norm_stats_refuse_a_cell_no_tracked_hand_reached():
    X = np.ones((4, 144), np.float32)
    keep = np.ones_like(X, bool)
    keep[:, 72:] = False
    with pytest.raises(ValueError, match="tracked hand"):
        MultiDataset._drop_nonfinite_rows(X, "k", keep)


def test_eval_takes_an_untracked_hands_gt_from_the_prediction():
    from egomimic.eval.action_metrics import fill_untracked_from_pred

    gt, pred = torch.zeros(2, 3, 144), torch.ones(2, 3, 144)
    out = fill_untracked_from_pred(
        {"actions_keypoints": gt, "action_hand_mask": LEFT_ONLY},
        "actions_keypoints",
        pred,
    )["actions_keypoints"]
    assert (out[..., :72] == 0).all() and (out[..., 72:] == 1).all()
    plain = {"actions_keypoints": gt}
    assert fill_untracked_from_pred(plain, "actions_keypoints", pred) is plain


def test_eval_metrics_average_over_the_tracked_hands_only():
    from egomimic.eval.action_metrics import (
        fill_untracked_from_pred,
        layout_metrics,
        per_tracked_hand,
        tracked_hand_fraction,
    )
    from egomimic.utils.pose_utils import bimanual_keypoint_layout

    g = torch.Generator().manual_seed(0)
    gt, pred = torch.randn(2, 3, 144, generator=g), torch.randn(2, 3, 144, generator=g)
    key = "actions_keypoints"
    batch = fill_untracked_from_pred(
        {key: gt, "action_hand_mask": LEFT_ONLY}, key, pred
    )
    tracked = tracked_hand_fraction(batch, key)
    assert tracked == 0.5
    got = per_tracked_hand(layout_metrics(pred, batch[key], "v", key), tracked)
    left = [i for i in bimanual_keypoint_layout(144)["wrist_xyz"] if i < 72]
    torch.testing.assert_close(
        got["v_wrist_xyz_paired_mse_avg"],
        (pred[..., left] - gt[..., left]).pow(2).mean(),
    )
    assert got.keys() == layout_metrics(pred, gt, "v", key).keys()
    assert tracked_hand_fraction({key: gt}, key) is None
