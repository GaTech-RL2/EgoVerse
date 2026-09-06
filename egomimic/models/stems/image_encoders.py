"""
Image encoders for HNetPolicy conditioning.

Designed to be instantiated via Hydra `_target_:`:

    img_encoders:
      front_cam:
        _target_: egomimic.models.stems.image_encoders.SimpleConv
        in_channels: 3
        channels: [32, 64, 128]
        embed_dim: 256

Each encoder must expose an `embed_dim` attribute so `HNetPolicy` can size
the fusion MLP without inspecting weights.

Shape contract: accept any tensor of shape `(..., C, H, W)` and return
`(..., embed_dim)`. Leading dims are flattened for the conv stack then
unflattened on output — so `(B, T, C, H, W) -> (B, T, embed_dim)` works
out of the box.
"""

from typing import Sequence

import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    """A small convnet for per-frame image conditioning.

    Architecture (configurable via YAML):
        for c_out in channels:
            Conv2d(stride) -> GroupNorm -> GELU
        AdaptiveAvgPool2d(1) -> Flatten -> Linear -> embed_dim
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 3,
        stride: int = 2,
        embed_dim: int = 256,
        norm_groups: int = 8,
    ):
        super().__init__()
        layers = []
        c = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(
                    c, c_out, kernel_size, stride=stride, padding=kernel_size // 2
                ),
                nn.GroupNorm(min(norm_groups, c_out), c_out),
                nn.GELU(),
            ]
            c = c_out
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(c, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W) -> (..., embed_dim)
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        feat = self.conv(x).flatten(1)  # (N, C_last)
        feat = self.head(feat)  # (N, embed_dim)
        return feat.reshape(*leading, self.embed_dim)
