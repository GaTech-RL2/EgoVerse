"""Small ConvVAE for PushShapes 96x96 RGB frames.

Trained once as a separate stage (see ``scripts/train_pushshapes_vae.py``)
and then frozen. Downstream usage:

* ``encode(x) -> (mu, logvar)`` then sample / take mean.
* ``decode(z) -> x_rec`` to turn predicted latents back into pixel
  frames at self-rollout-eval time.

Architecture (96 -> 6 -> 96):
  encoder: 3 -> 32 -> 64 -> 128 -> 256 channels, stride-2 each, GroupNorm
           + SiLU; ending at 6x6x256 spatial. Flatten + linear to mu, logvar.
  decoder: linear from latent to 6x6x256, four ConvTranspose stride-2
           blocks (256 -> 128 -> 64 -> 32 -> 3), final sigmoid.

Loss: per-pixel MSE + beta * KL. ``beta`` is set at the training script
level (default 1.0; lower if the latents start collapsing to a posterior
that doesn't carry image info).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
        self.norm = _gn(c_out)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class _ConvTransposeBlock(nn.Module):
    """Legacy ConvTranspose-based 2x upsample block. Produces checkerboard
    artifacts on sharp edges. Kept for back-compat with v1/v2 checkpoints
    (``decoder_upsample="transpose"``). For new training prefer
    ``_NNUpsampleBlock`` (``decoder_upsample="nearest"``).
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            c_in,
            c_out,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.norm = _gn(c_out)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class _NNUpsampleBlock(nn.Module):
    """Nearest-neighbor upsample + stride-1 Conv2d. Eliminates the
    checkerboard pattern that stride-2 ConvTranspose tends to produce
    on sharp edges, at no extra param cost (Conv2d is the same shape as
    the original ConvTranspose's kernel).
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
        self.norm = _gn(c_out)

    def forward(self, x):
        return F.silu(self.norm(self.conv(self.up(x))))


class _ResBlock(nn.Module):
    """Residual conv block with optional stride-2 downsample on the first
    conv. Two stacked (Conv + GN + SiLU) layers + a skip connection
    (projected when shapes change). Standard ResNet pre-block pattern.

    Compared to ``_ConvBlock`` (single conv), residual blocks give much
    better gradient flow and let capacity actually fit the data —
    plain stacked convs hit "barely-learning" plateaus much sooner.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.norm1 = _gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(c_out)
        if stride != 1 or c_in != c_out:
            self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class _ResUpBlock(nn.Module):
    """Residual NN-Upsample-Conv block. Mirrors ``_ResBlock`` but does
    a 2x spatial upsample first (via nearest neighbor) so the decoder
    can run at the larger spatial resolution. Same param shape as a
    pair of standard 3x3 convs.
    """

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
        self.norm1 = _gn(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(c_out)
        self.skip = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class ImageVAE(nn.Module):
    """ConvVAE for HxWx3 inputs.

    Supports two latent topologies:
      * **flat** (default, ``spatial_latent=False``): encoder output is
        flattened and projected via Linear to a single ``latent_dim``
        vector. Decoder un-projects symmetrically. Used by v1/v2/v3.
      * **spatial** (``spatial_latent=True``): encoder's last feature
        map is reduced to ``latent_channels`` channels via 1x1 Conv,
        keeping the bottleneck H'x W' spatial extent. The latent is
        ``(latent_channels, H', W')`` (e.g. 4x6x6 = 144D for 96px input).
        Decoder starts directly from the spatial latent — gives the
        decoder spatial structure to ground its outputs in. Used by SD-VAE
        / VQ-GAN / MAGVIT. Set ``latent_channels`` to control the
        capacity.

    Set ``residual=True`` to swap the plain (Conv+GN+SiLU) blocks for
    2-conv residual blocks with skip connections. Same channels per
    stage, ~2x params per stage, much better gradient flow.

    Args:
        in_channels: input image channels (3 for RGB).
        image_size: side length; must be divisible by 16.
        latent_dim: width of the flat latent (only used when
            ``spatial_latent=False``).
        channels: encoder channel widths (4 stages).
        decoder_upsample: ``"transpose"`` (v1/v2 default) or ``"nearest"``
            (v3+; avoids checkerboard).
        residual: use 2-conv residual blocks instead of plain conv blocks.
        spatial_latent: keep the latent as a spatial (C, H', W') feature
            map instead of flattening to a vector.
        latent_channels: number of channels in the spatial latent. Only
            used when ``spatial_latent=True``. The total flat dim is
            ``latent_channels * (image_size/16) ** 2``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 96,
        latent_dim: int = 64,
        channels: tuple = (32, 64, 128, 256),
        decoder_upsample: str = "transpose",
        residual: bool = False,
        spatial_latent: bool = False,
        latent_channels: int = 4,
    ):
        super().__init__()
        if decoder_upsample not in {"transpose", "nearest"}:
            raise ValueError(
                f"decoder_upsample must be 'transpose' or 'nearest', "
                f"got {decoder_upsample!r}."
            )
        self.decoder_upsample = decoder_upsample
        self.residual = bool(residual)
        self.spatial_latent = bool(spatial_latent)
        self.latent_channels = int(latent_channels)
        if image_size % 16 != 0:
            raise ValueError(
                f"image_size must be divisible by 16 (4 stride-2 stages); "
                f"got {image_size}."
            )
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.bottleneck_size = self.image_size // 16  # e.g. 96 -> 6
        self.bottleneck_c = int(channels[-1])

        # ---- Encoder ----
        enc_layers = []
        c_prev = int(in_channels)
        for c in channels:
            if self.residual:
                enc_layers.append(_ResBlock(c_prev, int(c), stride=2))
            else:
                enc_layers.append(_ConvBlock(c_prev, int(c)))
            c_prev = int(c)
        self.encoder = nn.Sequential(*enc_layers)
        flat_dim = self.bottleneck_c * self.bottleneck_size * self.bottleneck_size

        # ---- Latent projection ----
        if self.spatial_latent:
            # Keep the bottleneck spatial; just reduce channel count.
            self.to_mu = nn.Conv2d(
                self.bottleneck_c,
                self.latent_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.to_logvar = nn.Conv2d(
                self.bottleneck_c,
                self.latent_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.from_z = nn.Conv2d(
                self.latent_channels,
                self.bottleneck_c,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        else:
            self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(flat_dim, self.latent_dim)
            self.fc_z = nn.Linear(self.latent_dim, flat_dim)

        # ---- Decoder ----
        if self.residual:
            up_block_cls = _ResUpBlock  # always uses NN-upsample
        elif decoder_upsample == "nearest":
            up_block_cls = _NNUpsampleBlock
        else:
            up_block_cls = _ConvTransposeBlock

        dec_layers = []
        rev = list(reversed(channels))
        for i, c in enumerate(rev):
            c_in = int(c)
            c_out = int(rev[i + 1]) if i + 1 < len(rev) else int(in_channels)
            if c_out == int(in_channels):
                # Last layer: no norm; sigmoid applied in ``decode``.
                if self.residual or decoder_upsample == "nearest":
                    dec_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    dec_layers.append(
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)
                    )
                else:
                    dec_layers.append(
                        nn.ConvTranspose2d(
                            c_in,
                            c_out,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
                    )
            else:
                dec_layers.append(up_block_cls(c_in, c_out))
        self.decoder = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------
    # Standard VAE API.
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, 3, H, W) -> (mu, logvar)``.

        Output shape depends on ``spatial_latent``:
          * flat: each is ``(B, latent_dim)``
          * spatial: each is ``(B, latent_channels, H', W')`` where
            ``H' = W' = image_size // 16``.
        """
        h = self.encoder(x)
        if self.spatial_latent:
            return self.to_mu(h), self.to_logvar(h)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z -> (B, 3, H, W)`` in [0,1].

        Accepts either a flat latent ``(B, latent_dim)`` (when the model
        was built with ``spatial_latent=False``) or a spatial latent
        ``(B, latent_channels, H', W')``. Downstream code is free to
        flatten / unflatten the spatial latent; we sniff the input rank
        to figure out which path.
        """
        if self.spatial_latent:
            if z.dim() == 2:
                # Caller flattened — reshape back to the canonical spatial
                # form before the 1x1 projection.
                z = z.view(
                    z.shape[0],
                    self.latent_channels,
                    self.bottleneck_size,
                    self.bottleneck_size,
                )
            h = self.from_z(z)
        else:
            h = self.fc_z(z)
            h = h.view(
                -1,
                self.bottleneck_c,
                self.bottleneck_size,
                self.bottleneck_size,
            )
        return torch.sigmoid(self.decoder(h))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # ------------------------------------------------------------------
    # Loss helper.
    # ------------------------------------------------------------------

    @staticmethod
    def vae_loss(
        x: torch.Tensor,
        x_rec: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = F.mse_loss(x_rec, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl
