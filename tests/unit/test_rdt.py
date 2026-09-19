"""RDT on the HPT chassis: the shared DiT backbone and per-embodiment denoiser,
``RDTModel``'s condition streams (random-init tiny DINOv3 tower, no network)
and the algo's image-history batch layout."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from egomimic.algo.rdt import RDT, RDTModel
from egomimic.models.fm_policy import FMPolicy
from egomimic.models.hpt_nets import DINOv3Stem, MLPPolicyStem, PolicyStem
from egomimic.models.rdt_nets import (
    RDTBackbone,
    RDTConditions,
    RDTDenoiser,
    sincos_2d,
)

D, ACT, SEQ, B = 32, 6, 5, 3
DOMAIN = "human_bimanual"
TINY_TOWER = {"embed_dim": 32, "depth": 1, "num_heads": 2}


def _backbone(depth=4):
    torch.manual_seed(0)
    return RDTBackbone(hidden_dim=D, depth=depth, n_heads=4).eval()


def _denoiser(**kwargs):
    torch.manual_seed(0)
    net = RDTDenoiser(act_dim=ACT, act_seq=SEQ, hidden_dim=D, **kwargs)
    # the zero-init output layer would hide every upstream effect
    nn.init.normal_(net.ffn_final.fc2.weight, std=0.1)
    return net.eval()


def _cond(backbone=None, lang_len=4, n_state=1):
    torch.manual_seed(1)
    return RDTConditions(
        img=torch.randn(B, 7, D),
        freq=torch.full((B,), 30.0),
        backbone=backbone or _backbone(),
        state=torch.randn(B, n_state, D) if n_state else None,
        lang=torch.randn(B, lang_len, D),
        lang_mask=torch.ones(B, lang_len, dtype=torch.bool),
    )


def test_denoiser_shape_and_zero_init():
    net = RDTDenoiser(act_dim=ACT, act_seq=SEQ, hidden_dim=D)
    out = net(torch.randn(B, SEQ, ACT), torch.rand(B), _cond())
    assert out.shape == (B, SEQ, ACT)
    assert torch.count_nonzero(out) == 0


def test_denoiser_ignores_masked_language_tokens():
    net, x, t = _denoiser(), torch.randn(B, SEQ, ACT), torch.rand(B)
    cond = _cond()
    cond.lang_mask[:, -2:] = False
    ref = net(x, t, cond)
    cond.lang[:, -2:] += 10.0
    assert torch.allclose(net(x, t, cond), ref, atol=1e-6)
    cond.lang[:, 0] += 10.0
    assert not torch.allclose(net(x, t, cond), ref, atol=1e-4)


def test_denoiser_uses_every_condition_time_and_frequency():
    net, x, t = _denoiser(), torch.randn(B, SEQ, ACT), torch.rand(B)
    backbone = _backbone()
    ref = net(x, t, _cond(backbone))
    for field in ("img", "state", "lang", "freq"):
        cond = _cond(backbone)
        setattr(cond, field, getattr(cond, field) + 1.0)
        assert not torch.allclose(net(x, t, cond), ref, atol=1e-4), field
    assert not torch.allclose(net(x, t * 0.5, _cond(backbone)), ref, atol=1e-4)


def test_denoisers_of_different_widths_share_one_backbone():
    """Co-training: per-embodiment adaptors, one set of DiT blocks."""
    backbone = _backbone()
    wide = RDTDenoiser(act_dim=ACT + 3, act_seq=SEQ, hidden_dim=D)
    cond = _cond(backbone)
    assert _denoiser()(torch.randn(B, SEQ, ACT), torch.rand(B), cond).shape[-1] == ACT
    assert wide(torch.randn(B, SEQ, ACT + 3), torch.rand(B), cond).shape[-1] == ACT + 3
    assert not any("blocks" in n for n, _ in wide.named_parameters())


def test_denoiser_without_language_or_state():
    net = _denoiser(n_state_tokens=0)
    cond = RDTConditions(
        img=torch.randn(B, 7, D), freq=torch.full((B,), 30.0), backbone=_backbone()
    )
    assert net(torch.randn(B, SEQ, ACT), torch.rand(B), cond).shape == (B, SEQ, ACT)


def test_denoiser_rejects_wrong_state_token_count():
    with pytest.raises(ValueError, match="n_state_tokens"):
        _denoiser()(torch.randn(B, SEQ, ACT), torch.rand(B), _cond(n_state=2))


def test_fm_head_samples_through_the_conditions_container():
    head = FMPolicy(
        model=_denoiser(),
        action_horizon=SEQ,
        infer_ac_dims={DOMAIN: ACT},
        num_inference_steps=2,
    )
    assert head((_cond(), DOMAIN)).shape == (B, SEQ, ACT)


def test_sincos_2d_is_resolution_agnostic_at_the_corners():
    small, large = sincos_2d(D, (2, 3)), sincos_2d(D, (14, 25))
    assert small.shape == (6, D) and large.shape == (350, D)
    for a, b in ((0, 0), (2, 24), (3, 325), (5, 349)):  # the four corners
        assert torch.allclose(small[a], large[b], atol=1e-6)
    # vertical neighbours share their column half
    assert torch.equal(large[0, D // 2 :], large[25, D // 2 :])


def _dino(**kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("freeze_backbone", True)
    return DINOv3Stem(
        model_name="vit_small_patch16_dinov3",
        output_dim=D,
        image_size=[32, 48],
        pretrained=False,
        tower_kwargs=TINY_TOWER,
        **kwargs,
    )


# --------------------------------------------------------------------------
# RDTModel
# --------------------------------------------------------------------------


class _StubText(PolicyStem):
    """A per-token text stem: prompt i has i + 1 real tokens, left-padded."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, D)

    def forward_with_mask(self, prompts):
        n = len(prompts)
        ids = torch.arange(n)[None, :].expand(n, -1)
        mask = ids >= (n - 1 - torch.arange(n))[:, None]
        return self.embed(ids), mask


def _model(cond_drop=None, text=True, history_len=1, frames=2, frame_dropout=0.0):
    torch.manual_seed(0)
    model = RDTModel(
        embed_dim=D,
        depth=2,
        num_heads=4,
        cond_drop=cond_drop,
        image_history=frames,
        image_history_dropout=frame_dropout,
    )
    shared = {"front_img_1": MLPPolicyStem(input_dim=D, output_dim=D, widths=[D])}
    if text:
        shared["annotation"] = _StubText()
    model.init_domain_stem("shared", shared)
    model.shared_keys = list(shared)
    model.init_domain_stem(
        DOMAIN,
        {
            "state_ee_pose": MLPPolicyStem(
                input_dim=4, output_dim=D, widths=[D], history_len=history_len
            )
        },
    )
    model.init_domain_head(
        DOMAIN,
        FMPolicy(
            model=RDTDenoiser(
                act_dim=ACT, act_seq=SEQ, hidden_dim=D, n_state_tokens=history_len
            ),
            action_horizon=SEQ,
            infer_ac_dims={DOMAIN: ACT},
            num_inference_steps=2,
        ),
    )
    model.init_encoder("front_img_1", _dino())
    model.auxiliary_ac_keys = {}
    model.diffusion = True
    model.device = torch.device("cpu")
    model.finalize_modules()
    return model


def _data(history_len=1, text=True, frames=2):
    torch.manual_seed(2)
    data = {
        "front_img_1": torch.rand(B, frames, 1, 3, 32, 48),
        "state_ee_pose": torch.randn(B, history_len, 4),
        "action": torch.randn(B, SEQ, ACT),
        "pad_mask": torch.ones(B, SEQ, 1),
        "fps": torch.full((B,), 30.0),
    }
    if text:
        data["annotation"] = ["a", "b", "c"]
    return data


ALL_DROPS = {"state": 1.0, "lang": 1.0, "img": 1.0}


def test_model_streams_and_shapes():
    model = _model(history_len=2).eval()
    cond, _ = model.forward_features(DOMAIN, _data(history_len=2))
    assert cond.img.shape == (B, 2 * 6, D)  # 2 frames x 6 patches
    assert cond.state.shape == (B, 2, D)
    assert cond.lang.shape == (B, B, D)
    assert cond.lang_mask.sum(1).tolist() == [1, 2, 3]
    assert cond.freq.tolist() == [30.0] * B
    assert cond.backbone is model.trunk["trunk"]


def test_model_shares_the_dit_in_the_trunk_not_the_head():
    names = [n for n, _ in _model().named_parameters()]
    assert any(n.startswith("trunk.trunk.blocks.") for n in names)
    assert not any(n.startswith("heads.") and ".blocks." in n for n in names)
    assert not any(".tokens" in n or "cross_attention" in n for n in names)


def test_model_keeps_rdt_init_after_hpt_xavier_pass():
    net = _model().heads[DOMAIN].model
    assert torch.count_nonzero(net.ffn_final.fc2.weight) == 0
    assert net.x_pos_embed.abs().sum() > 0


def test_model_loss_reaches_every_trainable_parameter():
    """Plain DDP crashes on a parameter that gets no gradient."""
    model = _model(cond_drop={"state": 0.5, "lang": 0.5}, frame_dropout=0.5).train()
    nn.init.normal_(model.heads[DOMAIN].model.ffn_final.fc2.weight, std=0.1)
    model.compute_loss({"domain": DOMAIN, "data": _data()}).backward()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, missing


def test_model_without_text_stem_and_single_frame():
    model = _model(cond_drop=ALL_DROPS, text=False, frames=1).train()
    data = _data(text=False, frames=1)
    cond, _ = model.forward_features(DOMAIN, data)
    assert cond.lang is None and cond.img.shape == (B, 6, D)
    model.compute_loss({"domain": DOMAIN, "data": data}).backward()
    out = model.eval().forward(DOMAIN, _data(text=False, frames=1))
    assert out[DOMAIN].shape == (B, SEQ, ACT)


def test_model_rejects_a_frame_count_the_config_did_not_declare():
    with pytest.raises(ValueError, match="image_history"):
        _model(frames=2).forward_features(DOMAIN, _data(frames=1))


def test_condition_dropout_is_train_only_and_never_blinds_one_camera():
    model = _model(cond_drop=ALL_DROPS)
    assert not hasattr(model, "null_img")  # one camera: img dropout is inert
    seen = []
    stem = model.stems["shared_annotation"]
    real = stem.forward_with_mask
    stem.forward_with_mask = lambda prompts: (
        seen.append(list(prompts)),
        real(prompts),
    )[1]

    ref, _ = model.eval().forward_features(DOMAIN, _data())
    assert seen[-1] == ["a", "b", "c"]
    cond, _ = model.train().forward_features(DOMAIN, _data())
    assert seen[-1] == ["", "", ""]  # a dropped prompt is the empty prompt
    assert torch.allclose(cond.state, model.null_state.expand_as(cond.state))
    assert torch.allclose(cond.img, ref.img)  # tower frozen + in eval: same tokens


def test_image_history_dropout_replaces_the_past_frame_by_the_current():
    model = _model(frame_dropout=1.0).train()
    with torch.no_grad():
        model.frame_embed.zero_()
    cond, _ = model.forward_features(DOMAIN, _data())
    past, current = cond.img[:, :6], cond.img[:, 6:]
    assert torch.allclose(past, current, atol=1e-6)
    cond, _ = model.eval().forward_features(DOMAIN, _data())
    assert not torch.allclose(cond.img[:, :6], cond.img[:, 6:], atol=1e-4)


def test_frame_and_camera_tags_start_at_zero():
    model = _model()
    assert torch.count_nonzero(model.frame_embed) == 0
    assert all(torch.count_nonzero(p) == 0 for p in model.camera_embed.values())


def test_model_needs_fps():
    data = _data()
    del data["fps"]
    with pytest.raises(ValueError, match="fps"):
        _model().forward_features(DOMAIN, data)


# --------------------------------------------------------------------------
# RDT algo
# --------------------------------------------------------------------------


def test_rdt_rejects_hpt_trunk_features():
    with pytest.raises(ValueError, match="HPT-only"):
        RDT(None, {}, None, None, ot=True)


def _pairing_algo(augs):
    from egomimic.models.image_augs import PerSampleAugs

    algo = RDT.__new__(RDT)
    algo.nets = nn.ModuleDict().train()
    algo.encoders = {"front_img_1": None}
    algo.train_image_augs = PerSampleAugs(augs)
    algo.eval_image_augs = None
    algo.annotation_modality, algo.is_6dof, algo.shared_ac_key = (
        "annotation",
        True,
        None,
    )
    return algo


class _Shift(nn.Module):  # a "random" aug with no batched form: one draw per call
    def forward(self, image):
        return image + torch.rand(())


@pytest.mark.parametrize("vectorized", [True, False])
def test_algo_pairs_history_frames_under_one_augmentation(vectorized):
    """(past, current) reach the model as one (B, 2, 1, 3, H, W) input, jittered
    with the same per-sample draw, on the batched and the per-image aug path."""
    from torchvision.transforms import ColorJitter

    algo = _pairing_algo(ColorJitter(0.4, 0.4, 0.4, 0.1) if vectorized else _Shift())
    assert (algo.train_image_augs.vectorized is not None) == vectorized
    cam = "observations.images.front_img_1"
    frame = torch.rand(1, 3, 8, 8).expand(B, -1, -1, -1).contiguous()
    batch = {
        cam: frame,
        f"{cam}_hist": frame.clone(),
        "actions": torch.zeros(B, SEQ, ACT),
        "pad_mask": torch.ones(B, SEQ, 1),
        "embodiment": torch.tensor([3]),
        "fps": torch.full((B,), 30.0),
    }
    data = algo._robomimic_to_hpt_data(batch, [cam, f"{cam}_hist"], [], [], "actions")
    frames = data["front_img_1"]
    assert frames.shape == (B, 2, 1, 3, 8, 8)
    assert "front_img_1_hist" not in data
    # identical inputs: the pair stays identical, the samples do not
    assert torch.allclose(frames[:, 0], frames[:, 1], atol=1e-6)
    assert not torch.allclose(frames[0], frames[1], atol=1e-4)
    assert data["fps"].tolist() == [30.0] * B


def test_algo_keeps_past_and_current_in_order():
    algo = _pairing_algo(nn.Identity())
    cam = "observations.images.front_img_1"
    batch = {
        cam: torch.ones(B, 3, 8, 8),
        f"{cam}_hist": torch.zeros(B, 3, 8, 8),
        "actions": torch.zeros(B, SEQ, ACT),
        "pad_mask": torch.ones(B, SEQ, 1),
        "embodiment": torch.tensor([3]),
        "fps": torch.full((B,), 30.0),
    }
    frames = algo._robomimic_to_hpt_data(
        batch, [cam, f"{cam}_hist"], [], [], "actions"
    )["front_img_1"]
    assert frames[:, 0].sum() == 0 and frames[:, 1].min() == 1  # current frame last
