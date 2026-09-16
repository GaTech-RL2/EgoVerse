"""QwenVLBackbone (Qwen 3.5, multi-frame, all layers) + QwenVLAModel on a fake VLM."""

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

from egomimic.models.hpt_nets import verify_pretrained_weights
from egomimic.models.layerwise_dit import LayerwiseFMHead
from egomimic.models.qwenvla_nets import QwenVLAModel, QwenVLBackbone

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


def _model(snapshot, **head_kwargs) -> QwenVLAModel:
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
    return QwenVLAModel(backbone, head)


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
    assert model.encoders["vlm"] is model.backbone
    model.train()
    loss = model.compute_loss(_data())
    assert torch.isfinite(loss)
    loss.backward()
    model.eval()
    g = torch.Generator().manual_seed(0)
    actions = model.sample(_data(), generator=g)
    assert actions.shape == (B, T, W)
