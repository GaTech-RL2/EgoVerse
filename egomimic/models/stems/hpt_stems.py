"""HPT policy STEM modules relocated from ``models/hpt_nets.py``.

Role home for the HPT input-side encoders: the :class:`PolicyStem` base (with
its cross-attention latent pooling), the :class:`MLPPolicyStem` proprio encoder,
and the :class:`ResNet` image encoder. Moved here verbatim (class bodies
byte-identical) in the models/ hierarchy pass; the ``CrossAttention`` primitive
used by ``PolicyStem.init_cross_attn`` now imports from
``egomimic.models.cores.hpt_transformer``.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from egomimic.models.cores.hpt_transformer import CrossAttention

INIT_CONST = 0.02

class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.specs = kwargs.get("specs")

    def init_cross_attn(self, stem_spec):
        """initialize cross attention module and the learnable tokens"""
        token_num = stem_spec.crossattn_latent
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, stem_spec.modality_embed_dim) * INIT_CONST
        )

        self.cross_attention = CrossAttention(
            stem_spec.modality_embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        stem_feat = self(x)
        stem_feat = stem_feat.reshape(
            stem_feat.shape[0], -1, stem_feat.shape[-1]
        )  # (32, 147, 128)
        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens


class MLPPolicyStem(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: Optional[List[int]] = None,
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        input_slice: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """vanilla MLP class. ``input_slice=[start, end]`` picks ``x[..., start:end]``
        in forward, letting a multi-component proprio tensor feed only a subset
        into the stem without resaving the zarr."""
        if widths is None:
            widths = [512]
        super().__init__(**kwargs)
        self.input_slice = (
            slice(int(input_slice[0]), int(input_slice[1])) if input_slice else None
        )
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*modules) for _ in range(num_of_copy)]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        if self.input_slice is not None:
            x = x[..., self.input_slice]
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class ResNet(PolicyStem):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        """ResNet Encoder for Images"""
        super().__init__(**kwargs)
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)

        # by default we use a separate image encoder for each view in downstream evaluation
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [
                    nn.Sequential(*list(pretrained_model.children())[:-2])
                    for _ in range(num_of_copy)
                ]
            )
        self.input = input
        self.out_dim = output_dim
        self.to_tensor = transforms.ToTensor()
        self.proj = nn.Linear(512, output_dim)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # Freeze the backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all parameters in the ResNet backbone"""
        if isinstance(self.net, nn.ModuleList):
            for net in self.net:
                for param in net.parameters():
                    param.requires_grad = False
        else:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size,
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]
        """
        B, *_, H, W = x.shape
        x = x.view(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.reshape(-1, 3, H, W)
            feat = self.net(x)
        # concat along time
        feat = feat.reshape(B, feat.shape[1], -1).transpose(1, 2)
        feat = self.proj(feat)
        return feat
