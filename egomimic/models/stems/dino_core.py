"""DINO (v2/v3) visual encoder, a DROP-IN for ``VisualCore`` — same shape contract
``(..., C, H, W) -> (..., feature_dimension)`` so the H-Net trunk is untouched; only
the ``_target_`` in the obs-encoder config changes.

Design (confirmed by the 2026-06-24 encoder sweep): interpolate input -> ``image_size``
(DINO/14|16 wants >=224, not 96), ImageNet-normalize, take the FROZEN backbone's patch
tokens and pool them on a SPATIAL grid (``feature='grid7'`` -> 7x7), flatten, then
``Linear(g*g*dino_dim -> feature_dimension) + LayerNorm`` to the trunk width. The sweep
showed spatial-grid features ~HALVE object orient/pos error vs mean-pool (mean-pool is
the worst mode), so grid is the default. Freeze by default: cheap, and it preserves the
frozen-encoder property the exact-logp PPO pipeline relies on (features can be cached).

DINOv3 weights are loaded by PATH (the torch-hub entrypoint needs ``weights=<.pth>``;
``pretrained=True`` only works for v2). Ungated mirror weights live under
``~/.cache/dinov3_weights/`` (see ``V3W``). To use v3: hub_repo='facebookresearch/dinov3',
model_name='dinov3_vitl16' (or vitb16/vits16), image_size a multiple of 16 (224).
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Ungated DINOv3 weight files (downloaded from the Fanqi-Lin-IR mirror). Keyed by
# torch-hub model name; the v3 entrypoint takes these via ``weights=``.
V3W = {
    "dinov3_vits16": "~/.cache/dinov3_weights/dinov3_vits16_pretrain.pth",
    "dinov3_vitb16": "~/.cache/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "~/.cache/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _load_dino(hub_repo: str, model_name: str, weights_path: str = ""):
    """Load a DINO backbone, preferring the local torch-hub cache so it works on
    offline compute nodes; fall back to the normal github fetch. DINOv3 is loaded by
    weight path; DINOv2 via ``pretrained=True``."""
    cached = os.path.join(torch.hub.get_dir(), hub_repo.replace("/", "_") + "_main")
    src = "local" if os.path.isdir(cached) else "github"
    repo = cached if src == "local" else hub_repo
    if "dinov3" in model_name:
        w = weights_path or V3W.get(model_name, "")
        assert w, f"no DINOv3 weights path for {model_name}"
        return torch.hub.load(repo, model_name, source=src,
                              weights=os.path.expanduser(w), verbose=False)
    return torch.hub.load(repo, model_name, source=src, pretrained=True, verbose=False)


class DinoVisualCore(nn.Module):
    """Frozen DINO spatial-grid encoder. Output: (..., feature_dimension)."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 224,
        feature_dimension: int = 96,
        hub_repo: str = "facebookresearch/dinov3",
        model_name: str = "dinov3_vitb16",
        weights_path: str = "",            # DINOv3 .pth; defaults from V3W by model_name
        freeze: bool = True,
        feature: str = "grid7",            # grid<g> (spatial) | patch_mean | cls
        unfreeze_last_n: int = 0,          # >0 -> fine-tune last N transformer blocks
    ):
        super().__init__()
        assert in_channels == 3, "DINO expects 3-channel RGB"
        assert feature in ("patch_mean", "cls") or feature.startswith("grid"), \
            f"bad feature {feature}"
        self.image_size = image_size
        self.feature = feature
        self.freeze = freeze and unfreeze_last_n == 0
        self.embed_dim = feature_dimension          # matches VisualCore's output-dim attr

        self.dino = _load_dino(hub_repo, model_name, weights_path)
        dino_dim = self.dino.embed_dim
        for p in self.dino.parameters():
            p.requires_grad_(False)
        if unfreeze_last_n > 0:
            for blk in self.dino.blocks[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        if self.freeze:
            self.dino.eval()

        if feature.startswith("grid"):
            self.grid = int(feature[4:])
            proj_in = dino_dim * self.grid * self.grid
        else:
            self.grid = 0
            proj_in = dino_dim
        self.proj = nn.Sequential(nn.Linear(proj_in, feature_dimension),
                                  nn.LayerNorm(feature_dimension))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:                 # keep frozen backbone in eval (BN/dropout-free)
            self.dino.eval()
        return self

    def _pool(self, out) -> torch.Tensor:
        if self.feature == "cls":
            return out["x_norm_clstoken"]
        pt = out["x_norm_patchtokens"]                  # (B, N, D)
        if self.feature == "patch_mean":
            return pt.mean(1)
        B, N, D = pt.shape                              # grid<g>: spatial avg-pool to g x g
        s = int(round(N ** 0.5))
        pt = pt.transpose(1, 2).reshape(B, D, s, s)
        return F.adaptive_avg_pool2d(pt, self.grid).reshape(B, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 3, H, W), RGB in [0, 1]
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.dino.forward_features(x)
        feat = self.proj(self._pool(out))
        return feat.reshape(*leading, self.embed_dim)
