"""``DINOv3Stem`` on a random-init tiny tower (no network, no weights)."""

from __future__ import annotations

import pytest
import torch

from egomimic.models.hpt_nets import DINOv3Stem

D, B = 32, 3
TINY_TOWER = {"embed_dim": 32, "depth": 1, "num_heads": 2}


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


def test_dinov3_stem_returns_patch_tokens_only():
    stem = _dino()
    assert stem.grid_size == (2, 3)
    assert stem(torch.rand(B, 1, 1, 3, 90, 160)).shape == (B, 2 * 3, D)
    # 2 frames on the time axis: past frame's patches first
    assert stem(torch.rand(B, 2, 1, 3, 90, 160)).shape == (B, 2 * 2 * 3, D)


def test_dinov3_stem_frozen_tower():
    stem = _dino().train()
    assert not stem.tower.training
    stem(torch.rand(B, 1, 1, 3, 32, 48)).sum().backward()
    assert all(p.grad is None for p in stem.tower.parameters())
    assert stem.proj.weight.grad is not None
    assert stem.pretrained_submodules() == [stem.tower]


def test_dinov3_stem_finetuned_tower_gets_gradients():
    stem = _dino(freeze_backbone=False).train()
    stem(torch.rand(B, 1, 1, 3, 32, 48)).sum().backward()
    assert any(p.grad is not None for p in stem.tower.parameters())
    assert {id(p) for p in stem.backbone_parameters()} == {
        id(p) for p in stem.tower.parameters()
    }


def test_dinov3_stem_rejects_off_grid_size():
    with pytest.raises(ValueError, match="multiple"):
        DINOv3Stem(
            model_name="vit_small_patch16_dinov3",
            image_size=250,
            pretrained=False,
            tower_kwargs=TINY_TOWER,
        )


def test_a_random_tower_is_not_verified():
    assert _dino().pretrained_reference_state_dict() is None


def test_the_verifier_checks_the_tower_against_its_cached_checkpoint(
    tmp_path, monkeypatch
):
    """The stem reads its timm checkpoint back from the HF cache, so #645's
    hash check covers it like the Qwen stems."""
    import huggingface_hub
    from safetensors.torch import save_file

    from egomimic.models.hpt_nets import verify_pretrained_weights

    stem = _dino()
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in stem.tower.state_dict().items()}, path)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    stem.pretrained = True

    assert list(verify_pretrained_weights(stem, verbose=False)) == ["DINOv3Stem"]
    with torch.no_grad():
        next(stem.tower.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="do not"):
        verify_pretrained_weights(stem, verbose=False)


def test_an_uncached_checkpoint_is_skipped_not_fetched(monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def offline(*a, **k):
        assert k.get("local_files_only") is True
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    stem = _dino()
    stem.pretrained = True
    assert stem.pretrained_reference_state_dict() is None
