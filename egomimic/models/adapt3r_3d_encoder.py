"""Adapt3R 3D observation token encoder for HPT.

Overview
--------
This module ports the core Adapt3R contribution into EgoVerse's encoder_specs
interface **without touching the HPT backbone (HPTModel)**. The key idea:

  Instead of encoding each camera image into 2D spatial tokens (as HPT's
  ResNet does), we lift the paired depth map into a 3D point cloud, encode
  each point's XYZ position with NeRF sinusoidal embeddings, fuse those
  position features with co-located 2D RGB backbone features, run Farthest
  Point Sampling (FPS) to reduce to a fixed point budget, then attention-pool
  the entire cloud into a single rich token per camera per frame.

  That single 3D-aware token is then passed through the existing
  MLPPolicyStem cross-attention, which expands it to the usual 16 trunk
  tokens — so the HPT backbone, stems, and heads are completely unchanged.

Data flow
---------
    _robomimic_to_hpt_data (HPT outer class)
        concatenates front_img_1 [B,3,H,W] + front_depth_1 [B,1,H,W]
        → data["front_img_1"] [B,1,1,4,H,W]

    Adapt3R3DEncoder.forward([B,1,1,4,H,W])
        RGB  → ResNet18 backbone → rgb_proj → [BN, h*w, hidden_dim]
        depth → lift_point_cloud_batch → XYZ [BN, h*w, 3]
        XYZ  → NeRFSinusoidalPosEmb → [BN, h*w, hidden_dim]
        FPS(512) on XYZ → subsample RGB & pos features
        cat([pos_emb, rgb_feat]) → [BN, 512, 2*hidden_dim]
        AttentionExtractor → [BN, 1, output_dim]
        reshape → [B, 1, output_dim]  ← 1 token per frame

    MLPPolicyStem.compute_latent([B, 1, output_dim])
        cross-attention (16 learned queries over 1 point-cloud token)
        → [B, 16, 256] trunk tokens  ← same as plain ResNet path

Dependencies
------------
All self-contained — no Adapt3R package install required.
Uses only: torch, torchvision, einops (all present in EgoVerse venv).

Source attribution
------------------
The following classes are adapted from Adapt3R
(https://github.com/user/adapt3r, MIT licence):
  - NeRFSinusoidalPosEmb  ← adapt3r/algos/utils/position_encodings.py
  - AttentionExtractor    ← adapt3r/algos/utils/pointnet_extractor.py
  - depth2fgpcd_batch,
    batch_transform_point_cloud,
    lift_point_cloud_batch  ← adapt3r/utils/point_cloud_utils.py
The pure-torch FPS and Adapt3R3DEncoder wrapper are new.
"""

from __future__ import annotations

import math
from typing import List, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

__all__ = [
    "NeRFSinusoidalPosEmb",
    "AttentionExtractor",
    "depth2fgpcd_batch",
    "batch_transform_point_cloud",
    "lift_point_cloud_batch",
    "fps_pytorch",
    "Adapt3R3DEncoder",
]


# ---------------------------------------------------------------------------
# NeRF sinusoidal positional encoding
# Adapted from adapt3r/algos/utils/position_encodings.py
# ---------------------------------------------------------------------------

class NeRFSinusoidalPosEmb(nn.Module):
    """NeRF-style sinusoidal positional embedding for 3-D spatial coordinates.

    Encodes each XYZ point as a concatenation of sin/cos at geometrically
    spaced frequencies — the same encoding used in the original NeRF paper.
    This gives the downstream network a smooth, distance-aware representation
    of where each point lives in 3D space, without any learned parameters.

    Args:
        dim: Output embedding dimension. **Must be divisible by 6** because
             the encoding splits evenly across 3 spatial coordinates
             (X, Y, Z) × 2 (sin + cos) × n_freqs.

    Shape:
        Input:  ``(..., 3)`` — float tensor of XYZ coordinates, any batch shape.
        Output: ``(..., dim)`` — sinusoidal embedding.

    Example::

        pos_enc = NeRFSinusoidalPosEmb(dim=60)
        xyz = torch.randn(8, 512, 3)   # batch of 512 points
        emb = pos_enc(xyz)             # → (8, 512, 60)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 6 == 0, (
            f"NeRFSinusoidalPosEmb: dim must be divisible by 6, got {dim}. "
            "Reason: dim // 6 frequency bands are applied to each of X, Y, Z "
            "and then sin+cos doubles the count → 6 * n_freqs = dim."
        )
        self.dim = dim

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Encode XYZ coordinates into sinusoidal embeddings.

        Args:
            x: Float tensor of shape ``(..., 3)``.

        Returns:
            Float tensor of shape ``(..., dim)``.
        """
        device = x.device
        n_freqs = self.dim // 6       # frequency bands per spatial axis
        max_freq = n_freqs - 1

        # Geometrically spaced frequencies: 2^0, 2^1, …, 2^(n_freqs-1)
        freq_bands = torch.pow(
            torch.tensor(2, device=device),
            torch.linspace(0, max_freq, steps=n_freqs, device=device),
        )                                        # [n_freqs]
        freq_bands = freq_bands.view(1, 1, 1, -1)   # broadcast: [1, 1, 1, n_freqs]

        x = x.unsqueeze(-1)                          # [..., 3, 1]
        emb = freq_bands * x                         # [..., 3, n_freqs]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)   # [..., 3, 2*n_freqs]
        emb = einops.rearrange(emb, "... i j -> ... (i j)")  # [..., dim]
        return emb


# ---------------------------------------------------------------------------
# Attention-based point-cloud pooling
# Adapted from adapt3r/algos/utils/pointnet_extractor.py
# ---------------------------------------------------------------------------

class AttentionExtractor(nn.Module):
    """Pool a variable-size point set to a single embedding via learned attention.

    Uses ``num_heads`` learnable query vectors (one per head) that attend over
    all N input points via scaled dot-product attention.  The per-head outputs
    are concatenated and projected to ``out_channels``.  This is order-invariant
    (permutation-equivariant in points) and produces a fixed-size output
    regardless of how many points are in the cloud.

    Compared to simple max/mean pooling, the attention mechanism allows the
    model to learn *which* points to focus on (e.g. object surfaces vs.
    background), making it much better at capturing task-relevant 3D structure.

    Args:
        in_shape: Per-point input feature dimension (``int`` or shape tuple,
                  only the last element is used).
        out_channels: Output embedding dimension per frame.
        use_layernorm: Whether to apply LayerNorm after each MLP hidden layer.
                       Improves training stability at the cost of a small speed
                       penalty.
        final_norm: Normalisation applied after the final projection.
                    ``"layernorm"`` | ``"none"`` (default).
        hidden_dim: Internal dimension of the key/value MLPs and the attention
                    heads.  Must be divisible by ``num_heads``.
        num_heads: Number of parallel attention heads. Each head produces an
                   independent ``hidden_dim``-dimensional read-out; the heads
                   are concatenated before the final projection.
        **kwargs: Absorbs extra Hydra config keys.

    Shape:
        Input:  ``(B, F, N, in_channels)`` where F = frame_stack (typically 1)
                and N = number of points.
        Output: ``(B, F, out_channels)``

    Example::

        extractor = AttentionExtractor(in_shape=120, out_channels=256,
                                       hidden_dim=256, num_heads=4)
        pc = torch.randn(4, 1, 512, 120)   # batch=4, 1 frame, 512 points
        token = extractor(pc)              # → (4, 1, 256)
    """

    def __init__(
        self,
        in_shape: int | tuple,
        out_channels: int = 256,
        use_layernorm: bool = False,
        final_norm: str = "none",
        hidden_dim: int = 256,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        in_channels = in_shape if isinstance(in_shape, int) else in_shape[-1]
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # One learnable query vector per head; broadcast over batch and frame dims.
        # Each head independently decides which points to focus on.
        self.queries = nn.Parameter(torch.randn(num_heads, hidden_dim))

        def _mlp(in_c: int, out_c: int, use_ln: bool) -> nn.Sequential:
            """4-layer MLP with optional LayerNorm after each hidden layer."""
            layers: List[nn.Module] = []
            for i in range(4):
                c_in = in_c if i == 0 else out_c
                layers.append(nn.Linear(c_in, out_c))
                if use_ln:
                    layers.append(nn.LayerNorm(out_c))
                if i < 3:          # no activation after the last linear
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        # Separate key and value projections allow the network to learn
        # different representations for "where to look" vs "what to extract".
        self.K_mlp = _mlp(in_channels, hidden_dim, use_layernorm)
        self.V_mlp = _mlp(in_channels, hidden_dim, use_layernorm)

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(hidden_dim * num_heads, out_channels)
        else:
            raise ValueError(
                f"Unsupported final_norm='{final_norm}'. Choose 'layernorm' or 'none'."
            )

    def forward(
        self, pc: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool the point cloud to a single token per frame.

        Args:
            pc: Point features, shape ``(B, F, N, in_channels)``.
            mask: Optional boolean attention mask, shape ``(B, F, N)``.
                  ``True`` keeps a point, ``False`` suppresses it.

        Returns:
            Pooled token, shape ``(B, F, out_channels)``.
        """
        # Project each point to keys and values independently
        K = self.K_mlp(pc)    # [B, F, N, hidden_dim]
        V = self.V_mlp(pc)    # [B, F, N, hidden_dim]

        # Q: [1, 1, num_heads, hidden_dim] — broadcast over B and F.
        # scaled_dot_product_attention treats the last two dims as (seq_len, embed),
        # so num_heads acts as the query sequence length → each head gives one
        # hidden_dim read-out → concat → num_heads * hidden_dim total.
        Q = self.queries.unsqueeze(0).unsqueeze(0)    # [1, 1, num_heads, hidden_dim]

        if mask is not None:
            mask = mask.unsqueeze(2)    # [B, F, 1, N] — broadcast over query heads

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False
        )    # [B, F, num_heads, hidden_dim]

        # Concatenate all heads, then project to final dimension
        x = einops.rearrange(attn_out, "b f h d -> b f (h d)")
        return self.final_projection(x)    # [B, F, out_channels]


# ---------------------------------------------------------------------------
# Point cloud lifting utilities  (pinhole camera model)
# Adapted from adapt3r/utils/point_cloud_utils.py
# ---------------------------------------------------------------------------

def depth2fgpcd_batch(
    depth: torch.Tensor,
    cam_params: torch.Tensor,
    keepdims: bool = False,
) -> torch.Tensor:
    """Unproject a depth image to camera-frame 3-D points (pinhole model).

    For each pixel (u, v) with measured depth z:
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z

    Args:
        depth: Depth values in metres, shape ``(B, ncam, H, W)``.
        cam_params: Camera intrinsic matrix K, shape ``(B, ncam, 3, 3)``.
            Layout: ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
        keepdims: If ``True`` return ``(B, ncam, H, W, 3)``; if ``False``
            (default) flatten spatial dims → ``(B, ncam*H*W, 3)``.

    Returns:
        3-D point cloud in camera frame with shape
        ``(B, ncam, H, W, 3)`` or ``(B, ncam*H*W, 3)``.
    """
    B, ncam, H, W = depth.shape

    # Extract intrinsic parameters from the K matrix
    fx = cam_params[..., 0, 0]    # [B, ncam]
    fy = cam_params[..., 1, 1]
    cx = cam_params[..., 0, 2]
    cy = cam_params[..., 1, 2]

    # Build pixel coordinate grids — (u, v) in image space
    # indexing="ij" then transpose gives the standard (row=H, col=W) layout
    pos_u, pos_v = torch.meshgrid(
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pos_u = pos_u.T.unsqueeze(0).unsqueeze(0).expand(B, ncam, H, W)   # pixel col
    pos_v = pos_v.T.unsqueeze(0).unsqueeze(0).expand(B, ncam, H, W)   # pixel row

    # Reshape scalars for broadcasting over spatial dims
    fx = fx.reshape(B, ncam, 1, 1)
    fy = fy.reshape(B, ncam, 1, 1)
    cx = cx.reshape(B, ncam, 1, 1)
    cy = cy.reshape(B, ncam, 1, 1)

    # Pinhole back-projection
    x3d = (pos_u - cx) * depth / fx    # [B, ncam, H, W]
    y3d = (pos_v - cy) * depth / fy
    z3d = depth

    pcd = torch.stack([x3d, y3d, z3d], dim=-1)    # [B, ncam, H, W, 3]
    if keepdims:
        return pcd
    return einops.rearrange(pcd, "b ncam h w c -> b (ncam h w) c")


def batch_transform_point_cloud(
    pcd: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """Apply a batched rigid body transform to a point cloud.

    Converts points to homogeneous coordinates, applies the 4×4 transform
    matrix, then strips the homogeneous component.

    Args:
        pcd: Point cloud in source frame, shape ``(..., N, 3)``.
        transform: Rigid transform (cam-to-world extrinsics), shape
            ``(..., 4, 4)``.  The leading batch dims of ``pcd`` and
            ``transform`` must broadcast together.

    Returns:
        Transformed point cloud in target frame, shape ``(..., N, 3)``.
    """
    ones = torch.ones((*pcd.shape[:-1], 1), device=pcd.device, dtype=pcd.dtype)
    pcd_h = torch.cat([pcd, ones], dim=-1)    # [..., N, 4]  homogeneous

    transform = transform.to(dtype=pcd.dtype)
    # Einsum: for each point n in the cloud, multiply by the 4×4 matrix.
    # "...nd,...id->...ni" contracts over the 'd' (homogeneous) dimension.
    transformed = torch.einsum("...nd,...id->...ni", pcd_h, transform)
    return transformed[..., :3]    # drop homogeneous coordinate


def lift_point_cloud_batch(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    keepdims: bool = False,
) -> torch.Tensor:
    """Full pinhole point-cloud lifting: pixel depth → world-frame XYZ.

    Combines ``depth2fgpcd_batch`` (camera-frame unprojection) with
    ``batch_transform_point_cloud`` (cam-to-world extrinsics application).

    Args:
        depths: Depth maps in metres, shape ``(B, ncam, H, W)``.
        intrinsics: Camera K matrices, shape ``(B, ncam, 3, 3)``.
        extrinsics: Camera-to-world transform matrices, shape ``(B, ncam, 4, 4)``.
            Pass the identity matrix to stay in camera frame.
        keepdims: If ``True`` return ``(B, ncam, H, W, 3)``; if ``False``
            flatten spatial dims → ``(B, ncam*H*W, 3)``.

    Returns:
        World-frame point cloud with shape
        ``(B, ncam*H*W, 3)`` or ``(B, ncam, H, W, 3)``.
    """
    B, ncam, H, W = depths.shape

    # Step 1: back-project depth to camera-frame XYZ
    pcd = depth2fgpcd_batch(depths, intrinsics, keepdims=False)
    pcd = einops.rearrange(pcd, "b (ncam hw) c -> b ncam hw c", ncam=ncam)

    # Step 2: transform from camera frame to world frame
    trans_pcd = batch_transform_point_cloud(pcd, extrinsics)

    if keepdims:
        return einops.rearrange(trans_pcd, "b ncam (h w) c -> b ncam h w c", h=H)
    return einops.rearrange(trans_pcd, "b ncam (h w) c -> b (ncam h w) c")


# ---------------------------------------------------------------------------
# Farthest Point Sampling — pure PyTorch (no DGL required)
# ---------------------------------------------------------------------------

def fps_pytorch(pcd: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Farthest Point Sampling (FPS) in pure PyTorch.

    Iteratively selects the point farthest from all already-selected points.
    This gives a well-spread, geometry-preserving subsample that is much
    better than random sampling for retaining fine-grained 3D structure.

    Complexity: O(N × n_samples) — acceptable for N ≤ 16 384 and
    n_samples ≤ 1 024 on modern GPUs.  For very large clouds consider
    installing DGL and using ``dgl.geometry.farthest_point_sampler`` instead.

    Args:
        pcd: Input point cloud, shape ``(B, N, 3)``.
        n_samples: Number of points to select. Must be ≤ N.

    Returns:
        Long tensor of selected indices, shape ``(B, n_samples)``.

    Note:
        Always starts from index 0 for reproducibility.  Pass a shuffled
        cloud if you need stochastic starts.
    """
    B, N, _ = pcd.shape
    assert n_samples <= N, f"n_samples ({n_samples}) > number of input points ({N})"

    device = pcd.device
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    # min-distance to any already-selected point; initialised to ∞
    min_dists = torch.full((B, N), float("inf"), device=device)

    # Seed with the first point (index 0) for determinism
    selected[:, 0] = 0
    last = pcd[:, 0:1, :]    # [B, 1, 3]

    for i in range(1, n_samples):
        # Distance from every point to the most recently selected point
        new_dists = ((pcd - last) ** 2).sum(-1)        # [B, N]
        # Update the running minimum distance
        min_dists = torch.min(min_dists, new_dists)    # [B, N]
        # Select the point with the largest minimum distance
        selected[:, i] = min_dists.argmax(dim=1)       # [B]
        last = pcd[torch.arange(B, device=device), selected[:, i]].unsqueeze(1)

    return selected


# ---------------------------------------------------------------------------
# Main encoder — plugs into HPT encoder_specs
# ---------------------------------------------------------------------------

class Adapt3R3DEncoder(nn.Module):
    """Adapt3R-style 3D observation token encoder for HPT's ``encoder_specs``.

    Converts a 4-channel RGBD input (RGB + metric depth) into a single
    3D-aware token per camera per frame, which is then processed by the
    existing ``MLPPolicyStem`` cross-attention in HPT.

    The encoder is **registered in the model YAML under** ``encoder_specs``
    and wired to a camera key (e.g. ``front_img_1``).  The paired depth is
    concatenated onto the RGB in ``HPT._robomimic_to_hpt_data`` via the
    ``depth_key_map`` YAML config — no changes to the HPT backbone.

    Pipeline (per frame per camera):

    1. **RGB backbone** — frozen (by default) ImageNet-pretrained ResNet-18
       without its final avgpool and fc layers.  Produces a spatial feature
       map ``[BN, 512, h, w]`` which is projected to ``hidden_dim`` channels.

    2. **Depth → point cloud** — standard pinhole back-projection using the
       camera-specific intrinsics (K) and extrinsics (E) provided in the
       YAML.  The resulting XYZ cloud is bilinearly down-sampled to the
       backbone spatial resolution so each spatial location has both an RGB
       feature and a 3D position.

    3. **FPS** — Farthest Point Sampling selects ``num_points`` well-spread
       points from the full spatial grid, keeping both position and RGB
       feature for each chosen point.

    4. **Feature assembly** — concatenate per-point features:
       ``[NeRF_pos_emb(XYZ), RGB_backbone_feat]`` → ``[BN, num_points, pc_in_dim]``.

    5. **Attention pooling** — ``AttentionExtractor`` reduces the ``num_points``
       point features to one embedding vector per frame: ``[BN, 1, output_dim]``.

    6. **Reshape** → ``[B, T*I, output_dim]``, matching HPT's encoder output
       contract.  With T=I=1 this is ``[B, 1, output_dim]``.

    Args:
        output_dim: Dimension of the output token.  Must equal
            ``shared_stem_specs.<key>.input_dim`` in the model YAML
            (default 256 for the standard RBY1 config).
        hidden_dim: Internal feature width used by the RGB projection, NeRF
            encoding, and AttentionExtractor MLP.  **Must be divisible by 6**
            (NeRF encoding constraint).  Adapt3R default: 60.
        num_points: FPS target — number of points kept per frame after
            down-sampling.  Adapt3R default: 512.
        intrinsics: Camera intrinsic matrix K as a nested 3×3 list
            ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.  Fixed per embodiment;
            stored as a non-trainable buffer.
        extrinsics: Camera-to-world 4×4 transform as a nested list.  Pass
            the identity to keep points in camera frame.  Stored as a buffer.
        do_pos: Include NeRF-encoded XYZ position features (recommended: True).
        do_image: Include RGB backbone features (recommended: True).
        do_rgb: Include raw RGB pixel values as additional point features.
            Usually False — the backbone features already carry colour info.
        finetune_backbone: If ``False`` (default), the ResNet-18 backbone is
            frozen.  Set ``True`` to allow gradients through the backbone;
            increases VRAM usage and training time noticeably.
        **kwargs: Absorbs extra Hydra config keys (e.g. ``_target_``).

    Raises:
        AssertionError: If ``hidden_dim % 6 != 0``, or if ``intrinsics`` /
            ``extrinsics`` are not provided.

    Shape:
        Input:  ``(B, T, I, 4, H, W)`` — 4-channel RGBD tensor.
            Channels ``[:3]`` are RGB in ``[0, 1]``; channel ``[3]`` is
            depth in metres (positive, non-zero).  T and I are temporal and
            instance (camera view) dimensions; both are typically 1.
        Output: ``(B, T*I, output_dim)``

    Example (unit test)::

        K = [[600, 0, 64], [0, 600, 64], [0, 0, 1]]
        E = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        enc = Adapt3R3DEncoder(output_dim=256, hidden_dim=60, num_points=512,
                               intrinsics=K, extrinsics=E)
        x = torch.cat([torch.rand(2,1,1,3,128,128),
                        torch.rand(2,1,1,1,128,128)*2+0.1], dim=3)
        out = enc(x)   # → (2, 1, 256)
    """

    def __init__(
        self,
        output_dim: int = 256,
        hidden_dim: int = 60,
        num_points: int = 512,
        intrinsics: Optional[List] = None,
        extrinsics: Optional[List] = None,
        do_pos: bool = True,
        do_image: bool = True,
        do_rgb: bool = False,
        finetune_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # ----- Validate configuration -----
        assert hidden_dim % 6 == 0, (
            f"hidden_dim must be divisible by 6 for NeRF encoding, got {hidden_dim}. "
            "Set hidden_dim to 60 (Adapt3R default) or any multiple of 6."
        )
        assert intrinsics is not None, (
            "intrinsics (3×3 camera K matrix) must be provided in the model YAML. "
            "Format: [[fx,0,cx],[0,fy,cy],[0,0,1]]"
        )
        assert extrinsics is not None, (
            "extrinsics (4×4 cam-to-world matrix) must be provided in the model YAML. "
            "Use the identity matrix to keep points in camera frame."
        )
        assert do_pos or do_image, "At least one of do_pos or do_image must be True."

        # ----- Camera calibration (non-trainable) -----
        # Registered as buffers so they move to the right device automatically
        # and are saved/restored with the checkpoint.
        K = torch.tensor(intrinsics, dtype=torch.float32).reshape(1, 1, 3, 3)
        E = torch.tensor(extrinsics, dtype=torch.float32).reshape(1, 1, 4, 4)
        self.register_buffer("K", K)    # [1, 1, 3, 3]
        self.register_buffer("E", E)    # [1, 1, 4, 4]

        # ----- RGB backbone: ResNet-18 without avgpool + fc -----
        # Removing the last 2 children (avgpool, fc) gives a spatial feature
        # map [B, 512, h, w] where h = H//32 and w = W//32 for standard stride.
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        if not finetune_backbone:
            # Freeze backbone: faster training, avoids overfitting on small datasets.
            # Set finetune_backbone=True if the backbone features are not general
            # enough for your domain (e.g. very different from ImageNet).
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Project ResNet-18 output (512-ch) to the internal hidden_dim
        self.rgb_proj = nn.Linear(512, hidden_dim)

        # ----- NeRF positional encoding -----
        # Only instantiated when do_pos=True; stored as None otherwise so
        # it is excluded from the parameter count.
        self.xyz_proj = NeRFSinusoidalPosEmb(hidden_dim) if do_pos else None

        # ----- Feature dimension into AttentionExtractor -----
        # Each point carries: [pos_emb (hidden_dim)] + [rgb_feat (hidden_dim)]
        # + optionally raw RGB (3). Compute statically to avoid graph issues.
        pc_in_dim = (int(do_pos) + int(do_image)) * hidden_dim + (3 if do_rgb else 0)

        # ----- AttentionExtractor -----
        # Internal hidden_dim of the MLP is scaled up 4× relative to the point
        # feature width (common transformer convention); floor at 256 to avoid
        # degenerate small dims when hidden_dim is very small (e.g. 60 → 240 < 256).
        self.attention_extractor = AttentionExtractor(
            in_shape=pc_in_dim,
            out_channels=output_dim,
            hidden_dim=max(hidden_dim * 4, 256),
            num_heads=4,
            use_layernorm=False,
            final_norm="none",
        )

        # ----- ImageNet normalisation constants -----
        # Stored as buffers (not parameters) — move with the model but not trained.
        # Applied *inside* the encoder so the external image_augs in the YAML
        # (which still apply to the plain RGB path) are not used for RGBD keys.
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

        # Store flags needed at forward time
        self.do_pos = do_pos
        self.do_image = do_image
        self.do_rgb = do_rgb
        self.num_points = num_points
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an RGBD observation into a 3D-aware token.

        Args:
            x: 4-channel RGBD tensor of shape ``(B, T, I, 4, H, W)``.
               Channels ``[:3]`` = RGB in ``[0, 1]``;
               channel ``[3]`` = depth in metres (must be > 0).
               Assembled by ``HPT._robomimic_to_hpt_data`` via
               ``depth_key_map`` — both T and I are typically 1.

        Returns:
            Token tensor of shape ``(B, T*I, output_dim)``.
            With T=I=1 this is ``(B, 1, output_dim)``.
        """
        B, T, I, C, H, W = x.shape
        assert C == 4, (
            f"Adapt3R3DEncoder expects 4-channel RGBD input (C=4), got C={C}. "
            "Check that depth_key_map is set correctly in the model YAML."
        )

        rgb   = x[..., :3, :, :]    # [B, T, I, 3, H, W]  — normalised in [0,1]
        depth = x[..., 3:4, :, :]   # [B, T, I, 1, H, W]  — depth in metres

        # Flatten temporal and instance dims into the batch for parallel processing
        N_frames = T * I
        rgb_flat   = rgb.reshape(B * N_frames, 3, H, W)
        depth_flat = depth.reshape(B * N_frames, 1, H, W)

        # Expand calibration buffers to match the batch size
        K = self.K.expand(B * N_frames, -1, -1, -1)    # [BN, 1, 3, 3]
        E = self.E.expand(B * N_frames, -1, -1, -1)    # [BN, 1, 4, 4]

        # ------------------------------------------------------------------
        # Step 1 — RGB features
        # Normalise to ImageNet statistics before passing through ResNet-18.
        # Disable gradients when the backbone is frozen to save memory.
        # ------------------------------------------------------------------
        rgb_norm = (rgb_flat - self.rgb_mean) / self.rgb_std
        backbone_trainable = self.training and any(
            p.requires_grad for p in self.backbone.parameters()
        )
        with torch.set_grad_enabled(backbone_trainable):
            feat_map = self.backbone(rgb_norm)    # [BN, 512, h, w]

        h, w = feat_map.shape[-2:]
        # Flatten spatial grid and project channels: [BN, h*w, 512] → [BN, h*w, hidden_dim]
        feat = feat_map.flatten(2).transpose(1, 2)    # [BN, h*w, 512]
        feat = self.rgb_proj(feat)                    # [BN, h*w, hidden_dim]

        # ------------------------------------------------------------------
        # Step 2 — Point cloud lifting
        # Unproject depth to world-frame XYZ using pinhole model + extrinsics.
        # Downsample the full-res cloud to the backbone spatial resolution (h×w)
        # so each spatial position has a paired RGB feature and 3D coordinate.
        # ------------------------------------------------------------------
        pcd_full = lift_point_cloud_batch(
            depth_flat.reshape(B * N_frames, 1, H, W),    # [BN, 1, H, W]
            K,                                             # [BN, 1, 3, 3]
            E,                                             # [BN, 1, 4, 4]
            keepdims=True,                                 # [BN, 1, H, W, 3]
        )
        # Rearrange to channel-first for F.interpolate, then downsample
        pcd_chw = einops.rearrange(pcd_full, "b 1 h w c -> b c h w")    # [BN, 3, H, W]
        pcd_ds = F.interpolate(pcd_chw.float(), size=(h, w), mode="nearest")
        pcd = einops.rearrange(pcd_ds, "b c h w -> b (h w) c")    # [BN, h*w, 3]

        # ------------------------------------------------------------------
        # Step 3 — Farthest Point Sampling
        # Selects num_points well-spread 3D locations from the h×w spatial grid.
        # We then gather the corresponding RGB features and XYZ positions.
        # ------------------------------------------------------------------
        n_spatial = pcd.shape[1]
        n_sample = min(self.num_points, n_spatial)
        fps_idx = fps_pytorch(pcd, n_sample)                   # [BN, n_sample]
        idx_exp = fps_idx.unsqueeze(-1)                        # [BN, n_sample, 1]

        pcd_fps  = torch.gather(pcd,  1, idx_exp.expand(-1, -1, 3))
        feat_fps = torch.gather(feat, 1, idx_exp.expand(-1, -1, feat.shape[-1]))

        # ------------------------------------------------------------------
        # Step 4 — Per-point feature assembly
        # Concatenate whichever feature types are enabled.
        # Default (do_pos=True, do_image=True): [NeRF(xyz), rgb_feat]
        # This lets the attention extractor see both *where* a point is and
        # *what* it looks like, which is the core Adapt3R insight.
        # ------------------------------------------------------------------
        parts: List[torch.Tensor] = []

        if self.do_pos:
            pos_emb = self.xyz_proj(pcd_fps)    # [BN, n_sample, hidden_dim]
            parts.append(pos_emb)

        if self.do_image:
            parts.append(feat_fps)              # [BN, n_sample, hidden_dim]

        if self.do_rgb:
            # Raw pixel colour at each sampled location (cheap additional signal)
            rgb_ds = F.interpolate(
                rgb_flat, size=(h, w), mode="bilinear", align_corners=False
            )
            rgb_ds = einops.rearrange(rgb_ds, "b c h w -> b (h w) c")
            rgb_fps = torch.gather(rgb_ds, 1, idx_exp.expand(-1, -1, 3))
            parts.append(rgb_fps)               # [BN, n_sample, 3]

        cat_cloud = torch.cat(parts, dim=-1)    # [BN, n_sample, pc_in_dim]

        # ------------------------------------------------------------------
        # Step 5 — Attention pooling → 1 token per frame
        # AttentionExtractor expects [B, F, N, D] where F is a frame-stack dim.
        # We add a dummy F=1 dim, extract the token, then remove it.
        # ------------------------------------------------------------------
        cat_cloud = cat_cloud.unsqueeze(1)               # [BN, 1, n_sample, pc_in_dim]
        tokens = self.attention_extractor(cat_cloud)     # [BN, 1, output_dim]

        # ------------------------------------------------------------------
        # Step 6 — Reshape to HPT encoder output format [B, N_frames, output_dim]
        # With T=I=1 this produces [B, 1, output_dim].
        # ------------------------------------------------------------------
        return tokens.squeeze(1).reshape(B, N_frames, self.output_dim)
