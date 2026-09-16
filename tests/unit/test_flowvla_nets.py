"""QwenVLBackbone (Qwen 3.5, multi-frame, all layers) + FlowVLAModel on a fake VLM."""

from __future__ import annotations

import pytest
import torch
from fixtures.fake_qwen35 import (
    STUB_HIDDEN,
    STUB_IMAGE_SIZE,
    STUB_LAYERS,
    STUB_TOKENS_PER_FRAME,
    install_fake_qwen35,
)

from egomimic.models.flowvla_nets import FlowVLAModel, QwenVLBackbone
from egomimic.models.hpt_nets import verify_pretrained_weights
from egomimic.models.layerwise_dit import LayerwiseFMHead

B, T, W = 2, 4, 6


@pytest.fixture()
def snapshot(tmp_path, monkeypatch):
    return install_fake_qwen35(tmp_path, monkeypatch)


def _backbone(snapshot, **kwargs) -> QwenVLBackbone:
    kwargs.setdefault("image_size", STUB_IMAGE_SIZE)
    kwargs.setdefault("dtype", "float32")
    return QwenVLBackbone(model_name=snapshot, **kwargs)


def _frames(frames: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, frames, 3, *STUB_IMAGE_SIZE, generator=g)


def test_backbone_returns_every_text_layer_and_the_padding_mask(snapshot):
    backbone = _backbone(snapshot).eval()
    prompts = ["Embodiment: human bimanual. Task: fold", "Embodiment: human bimanual."]
    contexts, mask = backbone(_frames(), prompts)
    assert len(contexts) == STUB_LAYERS == backbone.num_layers
    assert all(c.shape == (B, mask.shape[1], STUB_HIDDEN) for c in contexts)
    assert mask.dtype == torch.bool
    # 8 image tokens + 5 words vs 8 + 3 words: right padding shows in the mask
    assert mask[0].sum() == STUB_TOKENS_PER_FRAME + 5
    assert mask[1].sum() == STUB_TOKENS_PER_FRAME + 3
    assert mask[1, -2:].tolist() == [False, False]


def test_backbone_accepts_a_single_frame_batch_and_multi_frame(snapshot):
    backbone = _backbone(snapshot).eval()
    one, mask1 = backbone(_frames()[:, 0], ["a b"] * B)
    two, mask2 = backbone(_frames(2), ["a b"] * B)
    assert mask1.shape[1] == STUB_TOKENS_PER_FRAME + 2
    assert mask2.shape[1] == 2 * STUB_TOKENS_PER_FRAME + 2


def test_backbone_resizes_frames_to_image_size(snapshot):
    backbone = _backbone(snapshot).eval()
    big = torch.rand(B, 1, 3, 8, 16)
    contexts, mask = backbone(big, ["x"] * B)
    assert mask.shape[1] == STUB_TOKENS_PER_FRAME + 1


def test_num_layers_takes_the_last_layers(snapshot):
    backbone = _backbone(snapshot, num_layers=1).eval()
    full = _backbone(snapshot).eval()
    contexts, _ = backbone(_frames(), ["x"] * B)
    all_layers, _ = full(_frames(), ["x"] * B)
    assert len(contexts) == 1
    assert torch.allclose(contexts[0], all_layers[-1])
    with pytest.raises(ValueError):
        _backbone(snapshot, num_layers=STUB_LAYERS + 1)


def test_image_tokens_depend_on_the_pixels(snapshot):
    backbone = _backbone(snapshot).eval()
    a, _ = backbone(_frames(seed=0), ["x"] * B)
    b, _ = backbone(_frames(seed=1), ["x"] * B)
    assert not torch.allclose(
        a[-1][:, :STUB_TOKENS_PER_FRAME], b[-1][:, :STUB_TOKENS_PER_FRAME]
    )


def test_fine_tune_exposes_trainable_backbone_parameters(snapshot):
    tuned = _backbone(snapshot, freeze=False)
    params = tuned.backbone_parameters()
    assert params and all(p.requires_grad for p in params)
    assert getattr(tuned.model, "gradient_checkpointing", False) is True
    frozen = _backbone(snapshot, freeze=True)
    assert not any(p.requires_grad for p in frozen.backbone_parameters())
    contexts, _ = frozen.eval()(_frames(), ["x"] * B)
    assert not contexts[-1].requires_grad


def test_backbone_passes_the_weight_hash_check(snapshot):
    verify_pretrained_weights(_backbone(snapshot), verbose=False)


def _model(snapshot, **head_kwargs) -> FlowVLAModel:
    backbone = _backbone(snapshot)
    torch.manual_seed(0)
    head = LayerwiseFMHead(
        action_width=W,
        action_horizon=T,
        vlm_hidden=backbone.hidden_size,
        num_layers=backbone.num_layers,
        state_dims={"human_bimanual": 3},
        head_dim=8,
        num_register_tokens=2,
        dropout=0.0,
        num_inference_steps=2,
        **head_kwargs,
    )
    return FlowVLAModel(backbone, head)


def _data() -> dict:
    g = torch.Generator().manual_seed(3)
    return {
        "images": _frames(),
        "prompts": ["Embodiment: human bimanual. Task: fold"] * B,
        "action": torch.randn(B, T, W, generator=g),
        "loss_mask": torch.ones(B, T, W),
        "state": torch.randn(B, 1, 3, generator=g),
        "embodiment_name": "human_bimanual",
    }


def test_model_loss_and_sample(snapshot):
    model = _model(snapshot)
    assert model.encoders["backbone"] is model.backbone
    model.train()
    loss = model.compute_loss(_data())
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    g = torch.Generator().manual_seed(0)
    actions = model.sample(_data(), generator=g)
    assert actions.shape == (B, T, W)


# ==========================================================================
# ResNetTextBackbone: one ResNet per camera role + frozen text -> one context
# ==========================================================================

from fixtures.stub_text_encoder import StubTextEncoder  # noqa: E402

from egomimic.models.resnet_text_backbone import ResNetTextBackbone  # noqa: E402

RT_HIDDEN = 32
RT_TEXT_HIDDEN = 16
FRONT = "observations.images.front_img_1"
LEFT = "observations.images.left_wrist_img"
PROMPT_LENGTHS = {"fold clothes": 2, "pick up the cup": 3}
# 64x64 through resnet18's children[:-2] downsamples by 32 -> 2x2 = 4 tokens
TOKENS_PER_CAMERA = 4


def _rt_backbone(roles=(FRONT,), hidden=RT_HIDDEN, num_layers=4, modality_embed=True):
    text_encoder = StubTextEncoder(output_dim=hidden, hidden_size=RT_TEXT_HIDDEN)
    text_encoder.lengths = dict(PROMPT_LENGTHS)  # fixed lengths for the assertions
    backbone = ResNetTextBackbone(
        camera_roles=list(roles),
        text_encoder=text_encoder,
        hidden_size=hidden,
        num_layers=num_layers,
        weights=None,  # no ImageNet download in a unit test
        modality_embed=modality_embed,
    )
    return backbone.eval()


def _rt_images(batch=2, cameras=1, size=64):
    return torch.randn(batch, cameras, 3, size, size)


def test_rt_one_encoder_per_role_keyed_by_role():
    backbone = _rt_backbone(roles=(FRONT, LEFT))
    assert set(backbone.image_encoders) == {
        ResNetTextBackbone._role_key(FRONT),
        ResNetTextBackbone._role_key(LEFT),
    }


def test_rt_image_tokens_are_camera_order_concatenated():
    backbone = _rt_backbone(roles=(FRONT, LEFT))
    images = _rt_images(batch=2, cameras=2)
    out = backbone._encode_images(images)
    assert out.shape == (2, 2 * TOKENS_PER_CAMERA, RT_HIDDEN)
    front = backbone.image_encoders[ResNetTextBackbone._role_key(FRONT)]
    expected_first = front(images[:, 0:1]) + backbone.modality_embed[0]
    torch.testing.assert_close(out[:, :TOKENS_PER_CAMERA], expected_first)


def test_rt_each_encoder_sees_exactly_one_frame():
    backbone = _rt_backbone(roles=(FRONT, LEFT))
    seen = []
    for module in backbone.image_encoders.values():
        module.register_forward_pre_hook(
            lambda _m, args, seen=seen: seen.append(tuple(args[0].shape))
        )
    backbone._encode_images(_rt_images(batch=2, cameras=2))
    assert seen == [(2, 1, 3, 64, 64), (2, 1, 3, 64, 64)]


def test_rt_wrong_camera_count_raises():
    backbone = _rt_backbone(roles=(FRONT, LEFT))
    with pytest.raises(ValueError, match="2 cameras"):
        backbone._encode_images(_rt_images(batch=2, cameras=1))


def test_rt_duplicate_roles_raise():
    with pytest.raises(ValueError, match="duplicate"):
        _rt_backbone(roles=(FRONT, FRONT))


def test_rt_backbone_parameters_are_the_resnet_trunks_only():
    backbone = _rt_backbone(roles=(FRONT,))
    ids = {id(p) for p in backbone.backbone_parameters()}
    front = backbone.image_encoders[ResNetTextBackbone._role_key(FRONT)]
    assert ids == {id(p) for p in front.net.parameters()}
    assert not ids & {id(p) for p in backbone.text_encoder.parameters()}
    assert not ids & {id(p) for p in front.proj.parameters()}


def test_rt_forward_returns_one_context_per_layer():
    backbone = _rt_backbone(roles=(FRONT,), num_layers=4)
    contexts, mask = backbone(_rt_images(batch=2, cameras=1), ["fold clothes"] * 2)
    assert len(contexts) == 4
    expected = (2, TOKENS_PER_CAMERA + 2, RT_HIDDEN)  # "fold clothes" -> 2 tokens
    assert all(context.shape == expected for context in contexts)
    assert mask.shape == (2, TOKENS_PER_CAMERA + 2)
    assert mask.dtype == torch.bool


def test_rt_images_come_before_text_in_the_context():
    backbone = _rt_backbone(roles=(FRONT,), num_layers=2)
    images = _rt_images(batch=2, cameras=1)
    contexts, _ = backbone(images, ["fold clothes"] * 2)
    torch.testing.assert_close(
        contexts[0][:, :TOKENS_PER_CAMERA], backbone._encode_images(images)
    )


def test_rt_mask_is_true_on_images_and_on_real_text_only():
    backbone = _rt_backbone(roles=(FRONT,), num_layers=2)
    # Two prompts of different length -> the shorter one is left-padded.
    _, mask = backbone(
        _rt_images(batch=2, cameras=1), ["fold clothes", "pick up the cup"]
    )
    assert mask[:, :TOKENS_PER_CAMERA].all()
    text = mask[:, TOKENS_PER_CAMERA:]
    assert text.shape[1] == 3  # longest prompt
    assert text[0].tolist() == [False, True, True]  # left-padded to 3
    assert text[1].all()


def test_rt_contexts_share_one_tensor():
    backbone = _rt_backbone(roles=(FRONT,), num_layers=4)
    contexts, _ = backbone(_rt_images(), ["fold clothes"] * 2)
    assert all(context is contexts[0] for context in contexts)


def test_rt_text_dim_mismatch_raises():
    with pytest.raises(ValueError, match="hidden_size"):
        ResNetTextBackbone(
            camera_roles=[FRONT],
            text_encoder=StubTextEncoder(output_dim=8, hidden_size=RT_TEXT_HIDDEN),
            hidden_size=RT_HIDDEN,
            weights=None,
        )
