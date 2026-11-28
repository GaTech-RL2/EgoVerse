"""Patch tokens + a small transformer trained from scratch, for HPT ``encoder_specs`` (10-31 multisource cart; the user chose this
fast point-transformer option for a fourth policy).

= PatchPointEncoder (Point-BERT / Point-MAE tokens: G farthest-point centres, each with its K nearest points relative to the
centre, a shared mini-PointNet, the centre's xyz added back through a positional MLP; all train-only augmentation lives there)
followed by a pre-norm Transformer encoder over each cloud's G tokens (PyTorch scaled-dot-product attention), then LayerNorm.
Same contract as PatchPointEncoder: ``(B, T, I, N*in_dim)`` -> ``(B, T*I*G, output_dim)``; HPT's cross-attention stem reads the
tokens. Measured encoder-only training step (A40, batch 32, 2048/1024/1024 points, 4 layers): 64 ms vs 342 ms Utonia frozen,
818 ms Utonia LoRA, 12 ms PointNet (train/encoder_speed.txt).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from egomimic.models.patch_pcd_encoder import PatchPointEncoder


class PatchTransformerEncoder(PatchPointEncoder):
    def __init__(self, *args, tf_layers: int = 4, tf_heads: int = 4, tf_ff_mult: int = 4, tf_dropout: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer = nn.TransformerEncoderLayer(self.output_dim, tf_heads, tf_ff_mult * self.output_dim, dropout=tf_dropout,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
        self.tf_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = super().forward(x)                                   # (B, F*G, D), F = clouds per sample
        B, FG, D = tok.shape
        t = tok.reshape(B * (FG // self.num_groups), self.num_groups, D)   # attention within each cloud
        return self.tf_norm(self.tf(t)).reshape(B, FG, D)
