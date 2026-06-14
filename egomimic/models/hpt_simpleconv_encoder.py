"""From-scratch conv image encoder that is a drop-in for HPT's ``ResNet`` stem.

Purpose: the "HPT except the image encoder" ablation. HPT's image encoder is
config-selected via ``encoder_specs.front_img_1._target_``; pointing it at this
class (instead of ``egomimic.models.hpt_nets.ResNet``) keeps *everything else*
about HPT identical — the cross-attention stem, the action-token trunk, the
chunk + temporal-ensemble rollout, the image augmentations — and replaces only
the (ImageNet-pretrained ResNet-18) vision backbone with a small from-scratch
convnet. If the model still reaches ~HPT coverage, HPT's success is the recipe,
not the pretrained encoder; if it collapses, the encoder is the load-bearing
part.

Output contract matches ``ResNet.forward`` exactly: input ``[B, T, N, 3, H, W]``
(or any leading dims that flatten to ``N`` images of ``3xHxW``) ->
``[B, M, output_dim]`` spatial tokens, where ``M`` is the conv grid size. The
downstream ``MLPPolicyStem`` cross-attention pools those ``M`` tokens to its
fixed latent count, so ``M`` differing from ResNet's is fine.
"""

from typing import Sequence

import torch
import torch.nn as nn

from egomimic.models.hpt_nets import PolicyStem


class SimpleConvEncoder(PolicyStem):
    """Small GroupNorm convnet -> spatial tokens, drop-in for ``ResNet``."""

    def __init__(
        self,
        output_dim: int = 128,
        in_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: int = 3,
        stride: int = 2,
        norm_groups: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        layers = []
        c = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c, c_out, kernel_size, stride=stride, padding=kernel_size // 2),
                nn.GroupNorm(min(norm_groups, c_out), c_out),
                nn.GELU(),
            ]
            c = c_out
        self.net = nn.Sequential(*layers)
        self.out_dim = output_dim
        self.proj = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mirror ResNet.forward: [B, T, N, 3, H, W] -> [B, M, output_dim].
        B, *_, H, W = x.shape
        x = x.reshape(-1, 3, H, W)               # (N, 3, H, W), N = B*T*views
        feat = self.net(x)                       # (N, C', h, w)
        feat = feat.reshape(B, feat.shape[1], -1).transpose(1, 2)  # (B, M, C')
        feat = self.proj(feat)                   # (B, M, output_dim)
        return feat
