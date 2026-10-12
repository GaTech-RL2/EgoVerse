"""Learned null tokens for missing history: a dropped (history dropout,
training only) or front-padded (episode start, train and eval) past step /
frame becomes a learned null token, never the current one. The legacy
copy-current behaviour stays bit-identical behind the flags, and a checkpoint
from before the null tokens falls back to it on load."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from egomimic.algo.hpt import HPT, HPTModel, history_mask_kwargs
from egomimic.models.hpt_nets import MLPPolicyStem
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from tests.unit.test_rdt import DOMAIN, B, D, _data, _model, _pairing_algo

IN, OUT, K = 5, 7, 4


def _stem(**kwargs):
    torch.manual_seed(0)
    return MLPPolicyStem(
        input_dim=IN, output_dim=OUT, widths=[16], history_len=K, **kwargs
    )


def _with_time_embed(stem):
    with torch.no_grad():
        stem.time_embed.normal_()
    return stem


def _legacy_forward(stem, x):
    """MLPPolicyStem.forward before null tokens, verbatim."""
    if stem.training and stem.history_dropout > 0:
        drop = torch.rand(x.shape[0], device=x.device) < stem.history_dropout
        current = x[:, -1:].expand_as(x)
        x = torch.where(drop.view(-1, *([1] * (x.ndim - 1))), current, x)
    y = stem.net(x)
    return y + stem.time_embed.view(1, K, *([1] * (y.ndim - 3)), y.shape[-1])


# ---------------------------------------------------------------------------
# proprio stem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train", [True, False])
def test_flag_off_stem_is_bit_identical_to_the_legacy_stem(train):
    stem = _with_time_embed(_stem(history_dropout=0.5, history_null_token=False))
    assert not hasattr(stem, "null_token")
    stem.train(train)
    x = torch.randn(16, K, IN)
    mask = torch.zeros(16, K)
    torch.manual_seed(3)
    got = stem(x, history_mask=mask)  # the legacy stem ignores the mask
    torch.manual_seed(3)
    assert torch.equal(got, _legacy_forward(stem, x))


def test_flag_on_without_mask_or_drop_matches_the_legacy_stem():
    on = _with_time_embed(_stem())
    off = _stem(history_null_token=False)
    off.load_state_dict({k: v for k, v in on.state_dict().items() if k != "null_token"})
    x = torch.randn(3, K, IN)
    assert torch.equal(on.train()(x), off.train()(x))


def test_padded_steps_are_the_null_token_in_train_and_eval():
    stem = _with_time_embed(_stem())
    x = torch.randn(2, K, IN)
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    real = stem.net(x) + stem.time_embed
    null = stem.null_token + stem.time_embed
    for mode in (True, False):
        y = stem.train(mode)(x, history_mask=mask)
        torch.testing.assert_close(y[0, :2], null[:2])
        torch.testing.assert_close(y[0, 2:], real[0, 2:])
        torch.testing.assert_close(y[1], real[1])


def test_the_current_step_is_never_nulled():
    stem = _stem(history_dropout=1.0).train()
    x = torch.randn(3, K, IN)
    y = stem(x, history_mask=torch.zeros(3, K))
    torch.testing.assert_close(y[:, -1], stem.net(x[:, -1]) + stem.time_embed[-1])


def test_dropout_nulls_every_past_step_in_train_only():
    stem = _with_time_embed(_stem(history_dropout=1.0))
    x = torch.randn(3, K, IN)
    y = stem.train()(x)
    null = (stem.null_token + stem.time_embed[:-1]).expand(3, -1, -1)
    torch.testing.assert_close(y[:, :-1], null)
    torch.testing.assert_close(stem.eval()(x), stem.net(x) + stem.time_embed)


def test_null_token_is_small_and_reaches_the_graph_even_when_unused():
    """Plain DDP (trainer=ddp) crashes on a parameter without a gradient."""
    stem = _stem().train()
    assert stem.null_token.shape == (OUT,) and stem.null_token.abs().max() < 0.2
    stem(torch.randn(3, K, IN)).sum().backward()
    assert stem.null_token.grad is not None


def test_history_len_one_has_no_null_token():
    stem = MLPPolicyStem(input_dim=IN, output_dim=OUT, widths=[16])
    assert not hasattr(stem, "null_token") and not stem.history_null_token


def test_a_pre_null_checkpoint_falls_back_to_legacy_on_load():
    legacy = _with_time_embed(_stem(history_dropout=1.0, history_null_token=False))
    stem = _stem(history_dropout=1.0)
    with pytest.warns(UserWarning, match="null_token"):
        stem.load_state_dict(legacy.state_dict())
    assert not stem.history_null_token and not hasattr(stem, "null_token")
    x = torch.randn(4, K, IN)
    torch.manual_seed(5)
    got = stem.train()(x)
    torch.manual_seed(5)
    assert torch.equal(got, legacy.train()(x))


def test_a_null_checkpoint_loads_silently():
    src = _stem()
    dst = _stem()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dst.load_state_dict(src.state_dict())
    assert dst.history_null_token
    assert torch.equal(dst.null_token, src.null_token)


# ---------------------------------------------------------------------------
# proprio mask plumbing: ZarrDataset sample -> HPT batch -> stem
# ---------------------------------------------------------------------------


class _NoRenameNormStats:
    def zarr_key_to_keyname(self, zarr_key, embodiment_id):
        return None


def _proprio_leaf(tmp_path, T=24):
    t = np.arange(T, dtype=np.float64)[:, None]
    pose = np.concatenate(
        [t * 0.001 + np.zeros((T, 3)), np.tile([[1.0, 0.0, 0.0, 0.0]], (T, 1))], 1
    )
    ZarrWriter.create_and_write(
        tmp_path / "ep.zarr",
        numeric_data={
            "left.obs_ee_pose": pose,
            "right.obs_ee_pose": pose + 0.1,
            "obs_head_pose": pose,
        },
        image_data={"images.front_1": np.zeros((T, 8, 8, 3), np.uint8)},
        embodiment="human_bimanual",
        fps=30,
        task_name="synthetic",
        intrinsics={"front_1": np.eye(3, 4)},
    )
    key_map = Human.get_keymap(keymap_mode="cartesian", proprio_history=K)
    resolver = LocalEpisodeResolver(tmp_path, key_map=key_map, transform_list=[])
    ds = MultiDataset._from_resolver(resolver, mode="total")
    return next(iter(ds.datasets.values()))


def test_dataset_mask_reaches_the_stem_through_the_hpt_batch(tmp_path):
    leaf = _proprio_leaf(tmp_path)
    proprio = "left.obs_ee_pose"
    raw = default_collate([leaf[1], leaf[10]])
    assert raw["proprio_history_mask"].tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    ac_key = next(k for k, v in raw.items() if "action" in k and v.ndim == 3)

    algo = HPT.__new__(HPT)
    algo.norm_stats = _NoRenameNormStats()
    algo.device = "cpu"
    algo.annotation_key = None
    algo.use_pad_mask = False
    algo.ac_keys = {get_embodiment_id(DOMAIN): ac_key}
    algo.is_6dof, algo.shared_ac_key = True, None
    algo.annotation_modality, algo.eval_image_augs = "annotation", None
    stem = _with_time_embed(
        MLPPolicyStem(input_dim=7, output_dim=OUT, widths=[16], history_len=K)
    ).eval()
    policy = HPTModel.__new__(HPTModel)
    policy.stems = {f"{DOMAIN}_state_obs_ee_pose": stem}
    algo.nets = {"policy": policy}

    batch = HPT.process_batch_for_training(algo, {DOMAIN: raw})
    batch = batch[get_embodiment_id(DOMAIN)]
    data = HPT._robomimic_to_hpt_data(
        algo, batch, [], [proprio], [], ac_key, domain=DOMAIN
    )
    y = stem(data["state_obs_ee_pose"], **history_mask_kwargs(stem, data))
    null = stem.null_token + stem.time_embed
    torch.testing.assert_close(y[0, :2], null[:2])
    assert not torch.allclose(y[0, 2], null[2])
    assert not torch.allclose(y[1, :2], null[:2])


# ---------------------------------------------------------------------------
# image history (RDT)
# ---------------------------------------------------------------------------


def _null_frame_tokens(model):
    """What a nulled past frame (slot 0) looks like after the embeddings."""
    from egomimic.models.rdt_nets import sincos_2d

    grid = model.encoders["front_img_1"].grid_size
    pos = sincos_2d(D, grid)
    return (
        model.null_frame
        + pos
        + model.frame_embed[0, 0]
        + model.camera_embed["shared_front_img_1"][0, 0]
    )


def _tag(model):
    with torch.no_grad():
        model.frame_embed.normal_()
        model.camera_embed["shared_front_img_1"].normal_()
    return model


def test_image_dropout_nulls_the_past_frame_under_its_embeddings():
    model = _tag(_model(frame_dropout=1.0)).train()
    cond, _ = model.forward_features(DOMAIN, _data())
    expected = _null_frame_tokens(model).expand(B, -1, -1)
    torch.testing.assert_close(cond.img[:, :6], expected)
    assert not torch.allclose(cond.img[:, 6:], expected, atol=1e-3)


def test_eval_nulls_padded_frames_but_never_drops():
    model = _tag(_model(frame_dropout=1.0)).eval()
    data = _data()
    ref, _ = model.forward_features(DOMAIN, data)
    data["front_img_1_mask"] = torch.tensor([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    cond, _ = model.forward_features(DOMAIN, data)
    null = _null_frame_tokens(model)
    torch.testing.assert_close(cond.img[0, :6], null)
    torch.testing.assert_close(cond.img[2, :6], null)
    torch.testing.assert_close(cond.img[1], ref.img[1])  # real past frame kept
    torch.testing.assert_close(cond.img[:, 6:], ref.img[:, 6:])  # never the current


def test_flag_off_image_tokens_ignore_the_mask():
    model = _model(image_null=False).eval()
    data = _data()
    ref, _ = model.forward_features(DOMAIN, data)
    data["front_img_1_mask"] = torch.zeros(B, 2)
    cond, _ = model.forward_features(DOMAIN, data)
    assert torch.equal(cond.img, ref.img)


def test_single_frame_model_has_no_null_frame():
    assert not hasattr(_model(frames=1), "null_frame")


def test_null_params_get_gradients_when_nothing_is_nulled():
    model = _model(history_len=2).train()
    import torch.nn as nn

    nn.init.normal_(model.heads[DOMAIN].model.ffn_final.fc2.weight, std=0.1)
    model.compute_loss({"domain": DOMAIN, "data": _data(history_len=2)}).backward()
    assert model.null_frame.grad is not None
    assert model.stems[f"{DOMAIN}_state_ee_pose"].null_token.grad is not None


def test_rdt_routes_the_proprio_mask_to_the_state_stem():
    model = _model(history_len=2).eval()
    stem = model.stems[f"{DOMAIN}_state_ee_pose"]
    data = _data(history_len=2)
    data["proprio_history_mask"] = torch.tensor([[0.0, 1.0]] * B)
    cond, _ = model.forward_features(DOMAIN, data)
    torch.testing.assert_close(
        cond.state[:, 0], (stem.null_token + stem.time_embed[0]).expand(B, -1)
    )


def test_a_pre_null_rdt_checkpoint_falls_back_to_legacy_on_load():
    legacy = _model(frame_dropout=1.0, image_null=False)
    model = _model(frame_dropout=1.0)
    with pytest.warns(UserWarning, match="null_frame"):
        model.load_state_dict(legacy.state_dict())
    assert not model.image_history_null_token and not hasattr(model, "null_frame")
    data = _data()
    torch.manual_seed(4)
    got, _ = model.train().forward_features(DOMAIN, data)
    torch.manual_seed(4)
    want, _ = legacy.train().forward_features(DOMAIN, _data())
    assert torch.equal(got.img, want.img)


def test_algo_builds_the_frame_mask_current_last():
    algo = _pairing_algo(torch.nn.Identity())
    cam = "observations.images.front_img_1"
    batch = {
        cam: torch.rand(B, 3, 8, 8),
        f"{cam}_hist": torch.rand(B, 3, 8, 8),
        f"{cam}_hist_mask": torch.tensor([[0.0], [1.0], [1.0]]),
        "actions": torch.zeros(B, 5, 6),
        "pad_mask": torch.ones(B, 5, 1),
        "embodiment": torch.tensor([3]),
        "fps": torch.full((B,), 30.0),
    }
    data = algo._robomimic_to_hpt_data(batch, [cam, f"{cam}_hist"], [], [], "actions")
    assert data["front_img_1_mask"].tolist() == [[0, 1], [1, 1], [1, 1]]
    del batch[f"{cam}_hist_mask"]
    data = algo._robomimic_to_hpt_data(batch, [cam, f"{cam}_hist"], [], [], "actions")
    assert "front_img_1_mask" not in data
