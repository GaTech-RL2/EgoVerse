"""DP3-style point-cloud encoder for HPT ``encoder_specs``.

Faithful port of the point-cloud encoder from 3D-Diffusion-Policy (DP3):
    https://github.com/YanjieZe/3D-Diffusion-Policy
    (3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py,
     class ``PointNetEncoderXYZ``)

DP3's observation encoder is deliberately minimal: a per-point MLP
(3 -> 64 -> 128 -> 256, LayerNorm + ReLU between layers), a max-pool over
points, and a LayerNorm-ed linear projection. No color, no FPS inside the
network, no attention — the paper's claim is that this simplicity, applied to
a workspace-cropped, downsampled cloud, is enough.

Differences from the original, all interface-level (the math is identical):

* Wrapped for HPT's ``encoder_specs`` slot: input arrives as the camera-key
  tensor ``(B, T, I, N*3)`` (the cloud is stored flattened because LeRobot
  persists only 1-D non-image features) and output is ``(B, T*I, output_dim)``
  — the same contract ``Adapt3R3DEncoder`` follows, so the trunk / stems /
  flow-matching head are untouched and any comparison isolates the encoder.
* Optional train-time point-cloud augmentation (off by default), matching the
  jitter added to Adapt3R after its no-aug run overfit (held-out val 0.129 vs
  v4's 0.086): random SE(3) jitter of the whole cloud, per-point Gaussian
  noise, and random point dropout (resampled with replacement, so N is
  preserved).

The cloud is expected in the GLASS (Aria device) frame, metric metres,
workspace-cropped and FPS-downsampled to a fixed N at dataset-build time.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class DP3PointNetEncoder(nn.Module):
    """DP3 ``PointNetEncoderXYZ`` behind the HPT encoder interface.

    Args:
        output_dim: Final token width. Must equal the stem's ``input_dim``.
        num_points: Points per cloud (N). Input is validated against this.
        block_channels: Per-point MLP widths. DP3 default ``[64, 128, 256]``.
        use_layernorm: LayerNorm between MLP layers (DP3 uses True).
        final_norm: ``"layernorm"`` (DP3 default) or ``"none"`` on the
            output projection.
        pose_jitter_deg / pose_jitter_m: train-mode random rigid jitter of the
            whole cloud (rotation about the cloud centroid, then translation).
        point_noise_std_m: train-mode per-point Gaussian noise, metres.
        point_dropout: train-mode fraction of points replaced by duplicates of
            surviving points (keeps N fixed). 0.0 disables.
    """

    def __init__(
        self,
        output_dim: int = 256,
        num_points: int = 2048,
        in_dim: int = 3,
        block_channels: List[int] = (64, 128, 256),
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        pose_jitter_deg: float = 0.0,
        pose_jitter_m: float = 0.0,
        point_noise_std_m: float = 0.0,
        point_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_points = num_points
        self.pose_jitter_deg = pose_jitter_deg
        self.pose_jitter_m = pose_jitter_m
        self.point_noise_std_m = point_noise_std_m
        self.point_dropout = point_dropout

        c0, c1, c2 = block_channels
        norm = nn.LayerNorm if use_layernorm else None
        self.in_dim = in_dim
        layers: List[nn.Module] = [nn.Linear(in_dim, c0)]
        if norm: layers.append(norm(c0))
        layers += [nn.ReLU(), nn.Linear(c0, c1)]
        if norm: layers.append(norm(c1))
        layers += [nn.ReLU(), nn.Linear(c1, c2)]
        if norm: layers.append(norm(c2))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(c2, output_dim), nn.LayerNorm(output_dim)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(c2, output_dim)
        else:
            raise ValueError(f"final_norm must be 'layernorm' or 'none', got {final_norm!r}")

    # ------------------------------------------------------------------
    # Train-mode augmentation (all default-off; eval() is always a no-op)
    # ------------------------------------------------------------------
    def _augment(self, pc: torch.Tensor) -> torch.Tensor:
        """pc: (B, N, in_dim); first 3 dims metric xyz (only these are augmented)."""
        if pc.shape[-1] > 3:
            xyz = self._augment(pc[..., :3])
            return torch.cat([xyz, pc[..., 3:]], dim=-1)
        B, N, _ = pc.shape
        device, dtype = pc.device, pc.dtype
        rot_std = math.radians(getattr(self, "pose_jitter_deg", 0.0))
        trans_std = getattr(self, "pose_jitter_m", 0.0)
        noise_std = getattr(self, "point_noise_std_m", 0.0)
        dropout = getattr(self, "point_dropout", 0.0)

        if rot_std > 0.0 or trans_std > 0.0:
            # Rotate about the per-cloud centroid (not the camera origin) so a
            # pure-rotation jitter doesn't double as a large translation of a
            # far-away workspace; then translate.
            axis = torch.randn(B, 3, device=device, dtype=dtype)
            axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
            angle = torch.randn(B, device=device, dtype=dtype) * rot_std
            ax, ay, az = axis.unbind(-1)
            zero = torch.zeros_like(ax)
            Kx = torch.stack([
                torch.stack([zero, -az, ay], -1),
                torch.stack([az, zero, -ax], -1),
                torch.stack([-ay, ax, zero], -1),
            ], -2)
            I3 = torch.eye(3, device=device, dtype=dtype).expand(B, -1, -1)
            sin_a = angle.sin().view(B, 1, 1)
            cos_a = angle.cos().view(B, 1, 1)
            R = I3 + sin_a * Kx + (1 - cos_a) * torch.bmm(Kx, Kx)
            centroid = pc.mean(dim=1, keepdim=True)
            pc = torch.bmm(pc - centroid, R.transpose(1, 2)) + centroid
            pc = pc + torch.randn(B, 1, 3, device=device, dtype=dtype) * trans_std

        if noise_std > 0.0:
            pc = pc + torch.randn_like(pc) * noise_std

        if dropout > 0.0:
            # Replace a random fraction of points with duplicates of random
            # survivors — N stays fixed, max-pool sees fewer distinct points.
            n_drop = int(N * dropout)
            if n_drop > 0:
                drop_idx = torch.rand(B, N, device=device).argsort(dim=1)[:, :n_drop]
                src_idx = torch.randint(0, N, (B, n_drop), device=device)
                batch_ar = torch.arange(B, device=device).unsqueeze(1)
                pc = pc.clone()
                pc[batch_ar, drop_idx] = pc[batch_ar, src_idx]

        return pc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of point clouds.

        Args:
            x: ``(B, T, I, N*3)`` flattened clouds (HPT camera-key layout), or
               ``(B, N*3)`` / ``(B, N, 3)`` for direct use in tests.
        Returns:
            ``(B, T*I, output_dim)``.
        """
        if x.dim() == 4:
            B, T, I, D = x.shape
            N_frames = T * I
            pc = x.reshape(B * N_frames, -1, self.in_dim)
        elif x.dim() == 3 and x.shape[-1] == self.in_dim:
            B, N_frames = x.shape[0], 1
            pc = x
        elif x.dim() == 2:
            B, N_frames = x.shape[0], 1
            pc = x.reshape(B, -1, self.in_dim)
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        if pc.shape[1] != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} points per cloud, got {pc.shape[1]} "
                f"(input shape {tuple(x.shape)}). Check the dataset build."
            )

        if self.training:
            pc = self._augment(pc)

        feat = self.mlp(pc)                # (B*NF, N, C2) per-point features
        feat = torch.max(feat, dim=1)[0]   # (B*NF, C2)   DP3's max-pool
        token = self.final_projection(feat)  # (B*NF, output_dim)
        return token.reshape(B, N_frames, self.output_dim)
