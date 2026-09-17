"""Episode-tail padding must not supervise "stop moving".

The dataset emits a real ``action_pad_mask`` next to the action chunk, the
interpolators carry it to the model horizon, and ``DenoisingPolicy.loss_fn``
drops the padded steps -- but only behind the ``use_pad_mask`` algo flag, so
the flag-off path is bit-for-bit today's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from fixtures.synthetic_episodes import write_episode
from omegaconf import OmegaConf

import egomimic
from egomimic.algo.hpt import HPT
from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalAnnotationCutoffEpisodeResolver,
    LocalEpisodeResolver,
    MultiDataset,
)

T_FRAMES = 48  # write_episode default
HORIZON = Human.ACTION_HORIZON  # 30 raw action frames
CHUNK = 100  # model horizon after interpolation


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _pad_mask_transform_cls():
    """Imported inside the tests so the module still collects (and every test
    fails on its own assertion) before the transform exists."""
    from egomimic.rldb.zarr.action_chunk_transforms import InterpolatePadMask

    return InterpolatePadMask


def _leaf(ds: MultiDataset):
    """The single per-episode ZarrDataset behind a one-episode MultiDataset."""
    return next(iter(ds.datasets.values()))


def _mecka_dataset(tmp_path, *, transforms, cutoff: bool = False, n: int = 1):
    for i in range(n):
        write_episode(tmp_path, "mecka", seed=i)
    # No annotation_key: the cutoff reads the zarr's spans directly, and the
    # default collate cannot stack the per-sample annotation string lists.
    key_map = Human.get_keymap(keymap_mode="cartesian")
    resolver_cls = (
        LocalAnnotationCutoffEpisodeResolver if cutoff else LocalEpisodeResolver
    )
    resolver = resolver_cls(tmp_path, key_map=key_map, transform_list=transforms)
    return MultiDataset._from_resolver(resolver, mode="total")


def _expected_upsampled_mask(mask: np.ndarray, new_len: int) -> np.ndarray:
    """Independent restatement of the rule: an output step is valid only if the
    two source frames it interpolates between are both real; a step landing
    exactly on a source frame inherits that frame."""
    T = mask.shape[0]
    old = np.linspace(0, 1, T)
    new = np.linspace(0, 1, new_len)
    out = np.empty(new_len, dtype=np.float32)
    for j, t in enumerate(new):
        i = int(np.searchsorted(old, t))
        if i < T and np.isclose(old[i], t):
            out[j] = mask[i]
        else:
            out[j] = mask[i - 1] * mask[i]
    return out


# ---------------------------------------------------------------------------
# 1. dataset emits the real mask
# ---------------------------------------------------------------------------


def test_dataset_emits_action_pad_mask_for_a_tail_chunk(tmp_path):
    ds = _mecka_dataset(tmp_path, transforms=[])
    leaf = _leaf(ds)
    assert leaf.total_frames == T_FRAMES

    sample = leaf[T_FRAMES - 5]
    mask = np.asarray(sample["action_pad_mask"])
    assert mask.shape == (HORIZON,)
    expected = np.zeros(HORIZON, dtype=np.float32)
    expected[:5] = 1.0
    np.testing.assert_array_equal(mask, expected)

    head = np.asarray(leaf[0]["action_pad_mask"])
    np.testing.assert_array_equal(head, np.ones(HORIZON, dtype=np.float32))


def test_action_pad_mask_is_a_float32_tensor(tmp_path):
    ds = _mecka_dataset(tmp_path, transforms=[])
    value = _leaf(ds)[0]["action_pad_mask"]
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.float32


def test_no_action_pad_mask_without_an_action_horizon(tmp_path):
    """Keymaps with no horizoned ``action_keys`` entry must not grow a key."""
    write_episode(tmp_path, "mecka", seed=0)
    key_map = Human.get_keymap(keymap_mode="cartesian")
    for spec in key_map.values():
        if spec.get("key_type") == "action_keys":
            spec["key_type"] = "proprio_keys"
            spec.pop("horizon", None)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=[])
    ds = MultiDataset._from_resolver(resolver, mode="total")
    assert "action_pad_mask" not in _leaf(ds)[0]


def test_mask_survives_bounds_check_and_normalization(tmp_path):
    """``MultiDataset.__getitem__`` runs _check_bounds + normalize; the mask is
    not in the norm stats and must come through untouched."""
    transforms = Human.get_transform_list(mode="cartesian", stride=1)
    ds = _mecka_dataset(tmp_path, transforms=transforms, n=3)

    stats = MultiDataset(state={}, norm_mode="quantile")
    stats.populate_from_datasets({"human_bimanual": ds})
    stats.infer_shapes_from_batch(ds[0])
    stats.infer_norm_from_dataset(ds, "human_bimanual", sample_frac=1.0, num_workers=0)
    emb = get_embodiment_id("human_bimanual")
    assert "action_pad_mask" not in stats.norm_stats[emb]

    raw = _leaf(ds)[0]["action_pad_mask"].clone()
    ds.set_norm_stats_from(stats)
    normalized = ds[0]
    assert "action_pad_mask" in normalized
    assert normalized["action_pad_mask"].shape == (CHUNK,)
    torch.testing.assert_close(normalized["action_pad_mask"], raw)


# ---------------------------------------------------------------------------
# 2. annotation cutoff
# ---------------------------------------------------------------------------


def test_annotation_cutoff_is_padding(tmp_path):
    """Synthetic episodes carry spans [0, T//2) and [T//2, T-1); a chunk that
    starts inside the first span is real only up to that span's end."""
    ds = _mecka_dataset(tmp_path, transforms=[], cutoff=True)
    leaf = _leaf(ds)
    ann_end = T_FRAMES // 2
    idx = ann_end - 4
    assert idx + HORIZON > ann_end  # the cutoff, not the episode end, bites

    mask = np.asarray(leaf[idx]["action_pad_mask"])
    expected = np.zeros(HORIZON, dtype=np.float32)
    expected[: ann_end - idx] = 1.0
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# 3. InterpolatePadMask
# ---------------------------------------------------------------------------


def test_interpolate_pad_mask_30_to_100_with_five_real_frames():
    mask = np.zeros(HORIZON, dtype=np.float32)
    mask[:5] = 1.0
    batch = {"action_pad_mask": mask.copy()}
    out = _pad_mask_transform_cls()(new_chunk_length=CHUNK).transform(batch)[
        "action_pad_mask"
    ]

    expected = _expected_upsampled_mask(mask, CHUNK)
    assert out.shape == (CHUNK,)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)
    assert out.sum() == expected.sum()
    # The tail is padding and the head is real: the mask is a prefix of ones.
    assert out[0] == 1.0 and out[-1] == 0.0
    assert np.all(np.diff(out) <= 0)


def test_interpolate_pad_mask_all_real_stays_all_ones():
    batch = {"action_pad_mask": np.ones(HORIZON, dtype=np.float32)}
    out = _pad_mask_transform_cls()(new_chunk_length=CHUNK).transform(batch)[
        "action_pad_mask"
    ]
    np.testing.assert_array_equal(out, np.ones(CHUNK, dtype=np.float32))


def test_interpolate_pad_mask_honours_stride():
    mask = np.zeros(HORIZON, dtype=np.float32)
    mask[:16] = 1.0
    out = _pad_mask_transform_cls()(new_chunk_length=CHUNK, stride=3).transform(
        {"action_pad_mask": mask.copy()}
    )["action_pad_mask"]
    np.testing.assert_array_equal(out, _expected_upsampled_mask(mask[::3], CHUNK))


def test_interpolate_pad_mask_is_a_noop_without_the_key():
    batch = {"actions_cartesian": np.zeros((3, 4))}
    assert _pad_mask_transform_cls()(new_chunk_length=CHUNK).transform(batch) == batch


def test_interpolate_pad_mask_is_idempotent():
    """``include_ee_pose`` chains two builders that both carry the mask."""
    mask = np.zeros(HORIZON, dtype=np.float32)
    mask[:5] = 1.0
    t = _pad_mask_transform_cls()(new_chunk_length=CHUNK, stride=3)
    once = t.transform({"action_pad_mask": mask.copy()})
    twice = t.transform(dict(once))
    np.testing.assert_array_equal(twice["action_pad_mask"], once["action_pad_mask"])


def test_human_transform_lists_upsample_the_mask():
    modes = [
        "cartesian",
        "cartesian_6d",
        "cartesian_wristframe_6d",
        "keypoints_headframe_6d",
        "keypoints_wristframe_6d",
    ]
    for mode in modes:
        tl = Human.get_transform_list(mode=mode, stride=3)
        assert any(
            isinstance(t, _pad_mask_transform_cls()) for t in tl
        ), f"{mode} does not carry the pad mask"


def test_keypoints_with_ee_pose_upsamples_the_mask_once(tmp_path):
    """Two builders chain here; the mask must still come out (100,)."""
    tl = Human.get_transform_list(
        mode="keypoints_wristframe_6d", stride=1, include_ee_pose=True
    )
    mask = np.zeros(HORIZON, dtype=np.float32)
    mask[:5] = 1.0
    batch = {"action_pad_mask": mask.copy()}
    for t in tl:
        if isinstance(t, _pad_mask_transform_cls()):
            batch = t.transform(batch)
    assert batch["action_pad_mask"].shape == (CHUNK,)
    np.testing.assert_array_equal(
        batch["action_pad_mask"], _expected_upsampled_mask(mask, CHUNK)
    )


# ---------------------------------------------------------------------------
# 4. loss_fn
# ---------------------------------------------------------------------------


def _policy() -> DenoisingPolicy:
    return DenoisingPolicy.__new__(DenoisingPolicy)


def test_loss_fn_without_mask_matches_mse_loss():
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(3, 7, 5, generator=g)
    target = torch.randn(3, 7, 5, generator=g)
    p = _policy()
    torch.testing.assert_close(p.loss_fn(pred, target), F.mse_loss(pred, target))
    ones = torch.ones(3, 7, 1)
    torch.testing.assert_close(
        p.loss_fn(pred, target, pad_mask=ones), F.mse_loss(pred, target)
    )


def test_loss_fn_ignores_masked_steps():
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(2, 6, 4, generator=g)
    target = torch.randn(2, 6, 4, generator=g)
    mask = torch.ones(2, 6, 1)
    mask[:, 4:] = 0.0
    p = _policy()
    before = p.loss_fn(pred, target, pad_mask=mask)

    perturbed = pred.clone()
    perturbed[:, 4:] += 100.0
    torch.testing.assert_close(p.loss_fn(perturbed, target, pad_mask=mask), before)

    # ... and it is exactly the mse over the kept steps.
    torch.testing.assert_close(before, F.mse_loss(pred[:, :4], target[:, :4]))


def test_loss_fn_all_zero_mask_does_not_divide_by_zero():
    p = _policy()
    loss = p.loss_fn(
        torch.randn(2, 6, 4), torch.randn(2, 6, 4), pad_mask=torch.zeros(2, 6, 1)
    )
    assert torch.isfinite(loss) and float(loss) == 0.0


class _FixedPredPolicy(DenoisingPolicy):
    """Minimal DenoisingPolicy: ``predict`` returns a canned prediction."""

    def __init__(self, action_horizon: int, pred: torch.Tensor):
        nn.Module.__init__(self)
        self.action_horizon = action_horizon
        self.pooling = None
        self.padding = None
        self._pred = pred

    def predict(self, actions, global_cond):
        return self._pred, actions


def test_compute_loss_forwards_the_pad_mask():
    g = torch.Generator().manual_seed(2)
    target = torch.randn(2, 6, 4, generator=g)
    pred = torch.randn(2, 6, 4, generator=g)
    policy = _FixedPredPolicy(6, pred)
    mask = torch.ones(2, 6, 1)
    mask[:, 4:] = 0.0
    global_cond = torch.zeros(2, 3)
    loss = policy.compute_loss(global_cond, {"action": target, "pad_mask": mask})
    torch.testing.assert_close(loss, F.mse_loss(pred[:, :4], target[:, :4]))


# ---------------------------------------------------------------------------
# 5. process_batch_for_training + the config flag
# ---------------------------------------------------------------------------


class _NoRenameNormStats:
    def zarr_key_to_keyname(self, zarr_key, embodiment_id):
        return None


def _process(use_pad_mask: bool, action_pad_mask: torch.Tensor | None, S: int = 6):
    algo = HPT.__new__(HPT)
    algo.norm_stats = _NoRenameNormStats()
    algo.device = "cpu"
    algo.annotation_key = None
    algo.use_pad_mask = use_pad_mask
    emb = "human_bimanual"
    emb_id = get_embodiment_id(emb)
    algo.ac_keys = {emb_id: "actions_cartesian"}
    inner = {"actions_cartesian": torch.zeros(2, S, 18)}
    if action_pad_mask is not None:
        inner["action_pad_mask"] = action_pad_mask
    return HPT.process_batch_for_training(algo, {emb: inner})[emb_id]


def test_process_batch_keeps_ones_when_the_flag_is_off():
    mask = torch.ones(2, 6)
    mask[:, 3:] = 0.0
    out = _process(False, mask)
    torch.testing.assert_close(out["pad_mask"], torch.ones(2, 6, 1))
    assert "action_pad_mask" not in out


def test_process_batch_uses_the_dataset_mask_when_the_flag_is_on():
    mask = torch.ones(2, 6)
    mask[:, 3:] = 0.0
    out = _process(True, mask)
    assert out["pad_mask"].shape == (2, 6, 1)
    torch.testing.assert_close(out["pad_mask"], mask[..., None])
    assert "action_pad_mask" not in out


def test_process_batch_falls_back_to_ones_without_the_dataset_key():
    out = _process(True, None)
    torch.testing.assert_close(out["pad_mask"], torch.ones(2, 6, 1))


def test_process_batch_raises_on_a_length_mismatch():
    with pytest.raises(ValueError, match="6"):
        _process(True, torch.ones(2, 5))


def test_use_pad_mask_defaults_to_false():
    import inspect

    assert inspect.signature(HPT.__init__).parameters["use_pad_mask"].default is False


@pytest.mark.parametrize(
    "yaml_name",
    [
        "hpt_bc_mecka_6d_300M",
        "hpt_bc_keypoints_wrist_300M",
        "hpt_bc_pickplace_qwen_pooled",
        "hpt_bc_pickplace_qwen_pertoken",
    ],
)
def test_algo_yaml_declares_use_pad_mask_off(yaml_name):
    from pathlib import Path

    path = (
        Path(egomimic.__file__).parent / "hydra_configs" / "model" / f"{yaml_name}.yaml"
    )
    cfg = OmegaConf.load(path)
    assert cfg.robomimic_model.use_pad_mask is False
