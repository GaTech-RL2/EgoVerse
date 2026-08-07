"""``PixelVideoDFoTOuterStage`` — Clean pixel-space DFoT for video clip batches.

Designed for ``ZarrVideoClipDataset`` which provides standard batches of
``(B, T, 3, H, W)`` fixed-length video clips. No packing, no per-episode
loops — the backbone processes the whole batch at once, exactly like the
reference DFoT repo.

This keeps fused_min_snr working correctly since ``k`` stays ``(B, T)``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from egomimic.algo.diffusion.outer_stages.outer_stage import DFoTOuterStage
from egomimic.models.diffusion.diffusion.discrete_diffusion import DiscreteDiffusion


class PixelVideoDFoTOuterStage(DFoTOuterStage):
    """Clean pixel-space DFoT outer stage for fixed-length video clip batches.

    Expects ``batch[video_key]`` to be ``(B, T, 3, H, W)`` float [0, 1].
    No packed mode support — use ``ZarrVideoClipDataset`` instead of packed datasets.

    Args:
        action_dim: unused, kept for interface compat.
        cond_encoder: empty CondEncoderModule for ABI parity.
        backbone: DFoTDiT3DBackbone configured for pixel space.
        diffusion: DiscreteDiffusion instance.
        video_key: batch key for video clips.
    """

    def __init__(
        self,
        action_dim: int,
        cond_encoder,
        backbone,
        diffusion,
        video_key: str = "videos",
        cond_output_key: str = "fused_cond",
    ):
        super().__init__(
            action_dim=action_dim,
            cond_encoder=cond_encoder,
            backbone=backbone,
            diffusion=diffusion,
            cond_output_key=cond_output_key,
        )
        self.video_key = str(video_key)
        # Infer shape from backbone config
        self._channels = int(backbone.latent_channels)
        self._size = int(backbone.latent_size)
        self._bundle_shape = (self._channels, self._size, self._size)

    @property
    def bundle_shape(self) -> tuple:
        return self._bundle_shape

    @property
    def bundle_dim(self) -> int:
        c, h, w = self._bundle_shape
        return c * h * w

    @property
    def action_slice(self) -> slice:
        return slice(0, 0)

    def encode(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        """Extract video clips and apply forward noising.

        Expects ``batch[video_key]``: ``(B, T, C, H, W)`` float [0, 1].
        """
        # Get videos directly from batch (not from ctx.obs)
        videos = batch.get(self.video_key)
        if videos is None:
            raise KeyError(
                f"'{self.video_key}' not found in batch. "
                f"Keys: {list(batch.keys())}"
            )

        # Ensure float [0, 1]
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
        elif videos.max() > 1.5:
            videos = videos.float() / 255.0
        else:
            videos = videos.float()

        B, T = videos.shape[:2]

        # Sample per-token noise levels: (B, T)
        k = torch.randint(
            0, self.diffusion.timesteps, (B, T),
            device=videos.device, dtype=torch.long,
        )

        # Forward noising
        noise = torch.randn_like(videos).clamp_(
            -self.diffusion.clip_noise, self.diffusion.clip_noise
        )
        x_t = self.diffusion.q_sample(videos, k, noise=noise)

        ctx.q_state = {
            "x_t": x_t,
            "k": k,
            "time_cond": k,
            "noise": noise,
            "x_start": videos,
        }
        ctx.external_cond = None
        ctx.latent_clean = videos
        return x_t

    def forward(self, batch: dict, ctx: SimpleNamespace) -> torch.Tensor:
        """Single forward pass on the whole batch — no per-episode loops."""
        x_t = self.encode(batch, ctx)
        k = ctx.q_state["k"]  # (B, T) — 2D, fused_min_snr works

        v_pred = self.inner_stage(x_t, k, external_cond=None)

        self.decode(v_pred, batch, ctx)
        return v_pred
