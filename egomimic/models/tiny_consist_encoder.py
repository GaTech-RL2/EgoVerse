"""Very small point-cloud encoder for the real-sim consistency side study (10-31, user request).

= PatchTransformerEncoder (patch tokens + transformer, egomimic/models/patch_tf_encoder.py) at a tiny size: 32 FPS centres x 32 points,
mini-PointNet hidden 64, d 128, 2 pre-norm layers, 4 heads, FF x2 (~0.2 M parameters). Pretrained by side/pretrain.py with a real-sim
InfoNCE consistency loss + action-chunk BC on paired human-view clouds (the real Aria cloud and the sim twin render of the same frame).
In a policy: ``ckpt_path`` loads the pretrained encoder weights and ``freeze=True`` stops their gradients; every sub-module then carries
``_hpt_keep_init`` so HPT.finalize_modules does not re-initialise the loaded weights (that reset hit every earlier pretrained encoder,
see memory hpt-finalize-reinit-pretrained). The train-only augmentation inside PatchPointEncoder still runs when HPT is training.
"""
from __future__ import annotations

import torch

from egomimic.models.patch_tf_encoder import PatchTransformerEncoder

TINY = dict(output_dim=128, hidden=64, num_groups=32, group_size=32, tf_layers=2, tf_heads=4, tf_ff_mult=2)


class TinyConsistEncoder(PatchTransformerEncoder):
    def __init__(self, *args, ckpt_path: str = '', freeze: bool = False, any_points: bool = False, **kwargs) -> None:
        for k, v in TINY.items():
            kwargs.setdefault(k, v)
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        self.any_points = any_points   # pretraining shares one encoder across the 2048- and 1024-point clouds
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            sd = sd.get('encoder', sd)
            self.load_state_dict(sd, strict=True)
            for m in self.modules():
                m._hpt_keep_init = True
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.any_points:
            self.num_points = x.shape[-1] // self.in_dim if x.dim() in (2, 4) else x.shape[1]
        if self.freeze:
            with torch.no_grad():
                return super().forward(x)
        return super().forward(x)
