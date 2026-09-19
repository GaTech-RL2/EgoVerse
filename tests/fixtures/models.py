"""Tiny, network-free model parts shared by the unit tests."""

from __future__ import annotations

import torch

TINY_TOWER = {"embed_dim": 32, "depth": 1, "num_heads": 2}


def tiny_dinov3_stem(output_dim: int, **kwargs):
    """Random-init one-block DINOv3 tower on 32x48 frames (a 2x3 patch grid)."""
    # lazy: recipes imports TINY_TOWER and must stay cheap to import
    from egomimic.models.hpt_nets import DINOv3Stem

    torch.manual_seed(0)
    kwargs.setdefault("freeze_backbone", True)
    return DINOv3Stem(
        model_name="vit_small_patch16_dinov3",
        output_dim=output_dim,
        image_size=[32, 48],
        pretrained=False,
        tower_kwargs=TINY_TOWER,
        **kwargs,
    )
