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
        spatial: bool = False,
        image_size: int = 96,
        return_tokens: bool = False,
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
        # spatial=False: global-avg-pool -> ONE vector (loses "where").
        # spatial=True: keep the conv feature map; flatten it so the embedding
        # encodes the spatial layout (where the T / goal / pusher are).
        self.spatial = bool(spatial)
        # return_tokens (cross-attn cond): keep the conv feature map as a SET of
        # spatial tokens (..., M, embed_dim) instead of one pooled/flattened
        # vector. Gated + default False -> the pooled/spatial-flatten paths
        # below are byte-identical to before this knob existed. Implies spatial.
        self.return_tokens = bool(return_tokens)
        if self.return_tokens:
            self.spatial = True
        if not self.spatial:
            layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            _d = torch.zeros(1, in_channels, image_size, image_size)
            _cm = self.conv(_d)
            _feat_dim = _cm.flatten(1).shape[1]
            self._n_tok = int(_cm.shape[2] * _cm.shape[3])
            self._tok_dim = int(_cm.shape[1])
        if self.return_tokens:
            self.head = nn.Linear(self._tok_dim, embed_dim)
            self.n_tokens = self._n_tok
        else:
            self.head = nn.Linear(_feat_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W) -> (..., embed_dim) | (..., M, embed_dim) tokens.
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        cm = self.conv(x)
        if self.return_tokens:
            N, C, h, w = cm.shape
            tok = cm.reshape(N, C, h * w).transpose(1, 2)
            tok = self.head(tok)
            return tok.reshape(*leading, self._n_tok, self.embed_dim)
        feat = cm.flatten(1)  # (N, C_last)
        feat = self.head(feat)  # (N, embed_dim)
        return feat.reshape(*leading, self.embed_dim)


class ResNetEncoder(nn.Module):
    """ResNet image encoder for H-Net conditioning — drop-in for ``SimpleConv``.

    Reuses HPT's ``egomimic.models.stems.hpt_stems.ResNet`` (optionally
    ImageNet-pretrained), which the ablation found is a ~2x vision helper over
    the global-pooled ``SimpleConv``. Obeys the same shape contract:
    ``(..., C, H, W) -> (..., embed_dim)`` and exposes ``.embed_dim`` so
    ``CondEncoderModule`` can size its fusion MLP.

    spatial=False (default): mean-pool the ResNet spatial tokens -> one vector
        (a ResNet global feature; better backbone than SimpleConv's pool).
    spatial=True: flatten the tokens and project -> embed_dim, keeping the
        "where" (T / goal / pusher layout), like ``SimpleConv(spatial=True)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        resnet_model: str = "resnet18",
        pretrained: bool = True,
        spatial: bool = False,
        image_size: int = 96,
        return_tokens: bool = False,
    ):
        super().__init__()
        from egomimic.models.stems.hpt_stems import ResNet

        self.resnet = ResNet(
            output_dim=embed_dim,
            resnet_model=resnet_model,
            weights="DEFAULT" if pretrained else None,
        )
        self.spatial = bool(spatial)
        # return_tokens (cross-attn cond): expose the ResNet spatial tokens AS
        # tokens (..., M, embed_dim) -- no flatten/pool -- so a cross-attn cond
        # path can localize over them (HPT-style). Gated + default False -> the
        # pooled/spatial-flatten paths stay byte-identical. Implies spatial.
        self.return_tokens = bool(return_tokens)
        if self.return_tokens:
            self.spatial = True
        # Probe the inner ResNet's real per-token dim. gmm's
        # ``hpt_stems.ResNet.forward`` returns RAW backbone features (4D
        # ``(N,512,h,w)``, no internal projection), while other ResNet variants
        # may return projected ``(N,M,output_dim)`` tokens. Sizing the head from
        # the probed token dim makes this drop-in robust to both: when the token
        # dim already equals ``embed_dim`` the head is an Identity (same as a
        # pre-projecting ResNet), otherwise it projects down.
        with torch.no_grad():
            _f = self.resnet(torch.zeros(1, in_channels, image_size, image_size))
            if _f.dim() != 3:  # (1, C, h, w) -> (1, M, C)
                _f = _f.flatten(2).transpose(1, 2)
            self._n_tok = int(_f.shape[1])
            _tok_in = int(_f.shape[2])
        if self.return_tokens:
            self.head = (
                nn.Identity() if _tok_in == embed_dim else nn.Linear(_tok_in, embed_dim)
            )
            self.n_tokens = self._n_tok
        elif self.spatial:
            self.head = nn.Linear(self._n_tok * _tok_in, embed_dim)
        else:  # pooled
            self.head = (
                nn.Identity() if _tok_in == embed_dim else nn.Linear(_tok_in, embed_dim)
            )
        self.embed_dim = int(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, H, W) -> (..., embed_dim) | (..., M, embed_dim) tokens.
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        tok = self.resnet(x)  # (N, M, Cin) spatial tokens (Cin may != embed_dim)
        if tok.dim() != 3:
            tok = tok.flatten(2).transpose(1, 2)
        if self.return_tokens:
            tok = self.head(tok)
            return tok.reshape(*leading, self._n_tok, self.embed_dim)
        if self.spatial:
            feat = self.head(tok.flatten(1))  # (N, embed_dim)
        else:
            feat = self.head(tok.mean(dim=1))  # (N, embed_dim) pooled ResNet feature
        return feat.reshape(*leading, self.embed_dim)
