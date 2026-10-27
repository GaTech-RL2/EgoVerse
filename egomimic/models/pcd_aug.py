"""Train-time geometric jitter of a point cloud, factored out of ``DP3PointNetEncoder._augment`` (same math, same
order of random draws) so the other point encoders can run the dp3c_dual V0 recipe's augmentation
(``pose_jitter_deg`` 5, ``pose_jitter_m`` 0.02, ``point_noise_std_m`` 0.015) and differ from DP3 only in architecture.
"""

from __future__ import annotations

import math

import torch


def dp3_geometric_jitter(xyz: torch.Tensor, rot_deg: float, trans_m: float, noise_std_m: float) -> torch.Tensor:
    """xyz: (B, N, 3) metres. Rotation about each cloud's centroid (random axis, angle ~ N(0, rot_deg)), then one
    translation ~ N(0, trans_m) per cloud, then per-point noise ~ N(0, noise_std_m)."""
    B = xyz.shape[0]
    device, dtype = xyz.device, xyz.dtype
    rot_std = math.radians(rot_deg)
    if rot_std > 0.0 or trans_m > 0.0:
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
        centroid = xyz.mean(dim=1, keepdim=True)
        xyz = torch.bmm(xyz - centroid, R.transpose(1, 2)) + centroid
        xyz = xyz + torch.randn(B, 1, 3, device=device, dtype=dtype) * trans_m
    if noise_std_m > 0.0:
        xyz = xyz + torch.randn_like(xyz) * noise_std_m
    return xyz
