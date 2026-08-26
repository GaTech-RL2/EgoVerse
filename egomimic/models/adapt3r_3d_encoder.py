"""Adapt3R 3D observation token encoder for HPT (DINOv2 / ResNet18 backbones).

Overview
--------
Ports the core Adapt3R contribution into EgoVerse's ``encoder_specs`` interface
**without touching the HPT backbone (HPTModel)**.

Per camera per frame:

  RGB  ── backbone (DINOv2 ViT-S/14 by default; ResNet-18 as alternative)
                 │
                 ├── spatial feature map  [BN, h, w, F]
                 │
  Depth ── pinhole back-projection (intrinsics K + extrinsics E)
                 │
                 ├── XYZ point cloud     [BN, h, w, 3]    (downsampled to (h,w))
                 │
        FPS to num_points  ──►  [BN, N, 3] + [BN, N, F]
                                       │
            NeRF positional encoding   ▼
            on XYZ (hidden_dim)        +  rgb_proj  (→ hidden_dim)
                                       │
            concat per point  ──►  [BN, N, pc_in_dim]
                                       │
            AttentionExtractor   ──►  [BN, 1, output_dim]
                                       │
                          reshape ──► [B, T*I, output_dim]

That single 3D-aware token feeds the existing ``MLPPolicyStem`` cross-attention,
which expands it to the standard 16 trunk tokens — so HPT's trunk, stems,
heads, and stem cross-attention are completely unchanged.

Backbones
---------
* ``backbone="dino"`` (default): DINOv2 ViT-S/14 from torch.hub. 384-d patch
  features. Input is internally resized to 224×224 (16×16 patches → 256 spatial
  locations). Frozen by default; pass ``finetune_backbone=True`` to allow
  fine-tuning. ``get_finetune_param_groups(base_lr, backbone_lr_scale=...)``
  exposes a parameter-group split for low-LR backbone finetuning.

* ``backbone="resnet18"``: TorchVision ImageNet-pretrained ResNet-18 with the
  final avgpool/fc removed. 512-d features. Output spatial size is ``H/32 × W/32``.

Camera calibration
------------------
Intrinsics ``K`` are configured for the original sensor resolution (e.g. 2016×1512
for the Aria RGB cam) and are auto-rescaled at init time to the encoder input
resolution (e.g. 360×360) using the YAML fields ``image_size_orig`` and
``image_size_input``.

Dummy depth
-----------
For datasets without a depth stream, set ``dummy_depth: <metres>``. The encoder
will overwrite the depth channel of every input with that constant before
unprojection. The lifted cloud is a flat plane at z = ``dummy_depth`` — the 3D
encoding becomes uninformative, but the entire encoder pipeline (shape, dtype,
device, gradient flow, integration with HPT) is exercised end-to-end.

Future TODOs (intentionally not implemented for this minimal port)
------------------------------------------------------------------
* **FISHEYE624 distortion** — Aria RGB is a fisheye lens with 12 distortion
  coefficients. We currently use a pinhole approximation. For real-world
  deployment, undistort the depth/RGB into a pinhole image first or replace
  ``depth2fgpcd_batch`` with a fisheye-aware unprojection.
* **Scene bounding-box crop** — Adapt3R clips points outside per-task scene
  bounds (LIBERO/MimicGen). Add ``crop_bounds: [[xmin,ymin,zmin],[xmax,ymax,zmax]]``
  here when needed.
* **Hand-frame transform** — Adapt3R's cross-embodiment trick re-frames the
  cloud in the gripper frame each step. Requires per-batch hand poses.

Source attribution
------------------
The following classes are adapted from Adapt3R (MIT licence):

* ``NeRFSinusoidalPosEmb``  ← ``adapt3r/algos/utils/position_encodings.py``
* ``AttentionExtractor``    ← ``adapt3r/algos/utils/pointnet_extractor.py``
* ``depth2fgpcd_batch``,
  ``batch_transform_point_cloud``,
  ``lift_point_cloud_batch``  ← ``adapt3r/utils/point_cloud_utils.py``

The pure-torch FPS, the DINO/ResNet18 backbone selection, the K rescaling, the
dummy-depth path, and the ``Adapt3R3DEncoder`` wrapper are new.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    Encodes each XYZ point as a concatenation of sin/cos at geometrically spaced
    frequencies — the same encoding used in the original NeRF paper. Provides a
    smooth, distance-aware representation of where each point lives in 3D space
    without any learned parameters.

    Args:
        dim: Output embedding dimension. Must be divisible by 6
            (3 axes × 2 (sin+cos) × n_freqs).

    Shape:
        Input:  ``(..., 3)`` float tensor of XYZ coordinates.
        Output: ``(..., dim)`` sinusoidal embedding.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 6 == 0, (
            f"NeRFSinusoidalPosEmb: dim must be divisible by 6, got {dim}."
        )
        self.dim = dim

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        device = x.device
        n_freqs = self.dim // 6
        max_freq = n_freqs - 1
        freq_bands = torch.pow(
            torch.tensor(2.0, device=device),
            torch.linspace(0, max_freq, steps=n_freqs, device=device),
        )                                          # [n_freqs]
        # Broadcast: leading dims of x will be (..., 3, 1); freq_bands to (..., 1, n_freqs)
        # We choose a generic-rank-safe broadcast.
        for _ in range(x.dim() - 1):
            freq_bands = freq_bands.unsqueeze(0)
        x = x.unsqueeze(-1)                        # [..., 3, 1]
        emb = freq_bands * x                       # [..., 3, n_freqs]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)   # [..., 3, 2*n_freqs]
        emb = einops.rearrange(emb, "... i j -> ... (i j)")
        return emb


# ---------------------------------------------------------------------------
# Attention-based point-cloud pooling
# Adapted from adapt3r/algos/utils/pointnet_extractor.py
# ---------------------------------------------------------------------------

class AttentionExtractor(nn.Module):
    """Pool a variable-size point set to a single embedding via learned attention.

    Uses ``num_heads`` learnable query vectors that attend over all N input
    points via scaled dot-product attention. Order-invariant in points,
    fixed-size output. See module docstring for design notes.

    Shape:
        Input:  ``(B, F, N, in_channels)`` (F = frame_stack, typically 1).
        Output: ``(B, F, out_channels)``.
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
        self.queries = nn.Parameter(torch.randn(num_heads, hidden_dim))

        def _mlp(in_c: int, out_c: int, use_ln: bool) -> nn.Sequential:
            layers: List[nn.Module] = []
            for i in range(4):
                c_in = in_c if i == 0 else out_c
                layers.append(nn.Linear(c_in, out_c))
                if use_ln:
                    layers.append(nn.LayerNorm(out_c))
                if i < 3:
                    layers.append(nn.ReLU())
            return nn.Sequential(*layers)

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
            raise ValueError(f"Unsupported final_norm='{final_norm}'.")

    def forward(
        self, pc: torch.Tensor, mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        K = self.K_mlp(pc)
        V = self.V_mlp(pc)
        Q = self.queries.unsqueeze(0).unsqueeze(0)    # [1, 1, num_heads, hidden_dim]

        if mask is not None:
            mask = mask.unsqueeze(2)

        if return_attn:
            # Manual attention so we can return the attn map for visualization.
            scale = 1.0 / math.sqrt(K.shape[-1])
            attn_logits = torch.einsum("b f h d, b f n d -> b f h n", Q.expand(K.shape[0], K.shape[1], -1, -1), K) * scale
            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
            attn = attn_logits.softmax(dim=-1)        # [B, F, num_heads, N]
            attn_out = torch.einsum("b f h n, b f n d -> b f h d", attn, V)
        else:
            attn_out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
            attn = None

        x = einops.rearrange(attn_out, "b f h d -> b f (h d)")
        out = self.final_projection(x)
        if return_attn:
            return out, attn
        return out


# ---------------------------------------------------------------------------
# Point cloud lifting utilities  (pinhole camera model)
# Adapted from adapt3r/utils/point_cloud_utils.py
# ---------------------------------------------------------------------------

def depth2fgpcd_batch(
    depth: torch.Tensor,
    cam_params: torch.Tensor,
    keepdims: bool = False,
) -> torch.Tensor:
    """Pinhole back-projection of a depth image to camera-frame XYZ.

    For each pixel (u, v) with depth z:
        X = (u - cx) * z / fx;  Y = (v - cy) * z / fy;  Z = z.

    Args:
        depth: ``(B, ncam, H, W)`` depth in metres.
        cam_params: ``(B, ncam, 3, 3)`` intrinsic matrices.
        keepdims: If ``True`` return ``(B, ncam, H, W, 3)``; else ``(B, ncam*H*W, 3)``.
    """
    B, ncam, H, W = depth.shape

    fx = cam_params[..., 0, 0]
    fy = cam_params[..., 1, 1]
    cx = cam_params[..., 0, 2]
    cy = cam_params[..., 1, 2]

    pos_u, pos_v = torch.meshgrid(
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pos_u = pos_u.T.unsqueeze(0).unsqueeze(0).expand(B, ncam, H, W)
    pos_v = pos_v.T.unsqueeze(0).unsqueeze(0).expand(B, ncam, H, W)

    fx = fx.reshape(B, ncam, 1, 1)
    fy = fy.reshape(B, ncam, 1, 1)
    cx = cx.reshape(B, ncam, 1, 1)
    cy = cy.reshape(B, ncam, 1, 1)

    x3d = (pos_u - cx) * depth / fx
    y3d = (pos_v - cy) * depth / fy
    z3d = depth

    pcd = torch.stack([x3d, y3d, z3d], dim=-1)
    if keepdims:
        return pcd
    return einops.rearrange(pcd, "b ncam h w c -> b (ncam h w) c")


def batch_transform_point_cloud(
    pcd: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """Apply a batched 4×4 rigid transform to a point cloud ``(..., N, 3)``."""
    ones = torch.ones((*pcd.shape[:-1], 1), device=pcd.device, dtype=pcd.dtype)
    pcd_h = torch.cat([pcd, ones], dim=-1)
    transform = transform.to(dtype=pcd.dtype)
    transformed = torch.einsum("...nd,...id->...ni", pcd_h, transform)
    return transformed[..., :3]


def lift_point_cloud_batch(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    keepdims: bool = False,
) -> torch.Tensor:
    """Lift depth → cam-frame XYZ → world-frame via extrinsics.

    Args:
        depths: ``(B, ncam, H, W)``.
        intrinsics: ``(B, ncam, 3, 3)``.
        extrinsics: ``(B, ncam, 4, 4)`` cam-to-world (identity = camera frame).
    """
    B, ncam, H, W = depths.shape
    pcd = depth2fgpcd_batch(depths, intrinsics, keepdims=False)
    pcd = einops.rearrange(pcd, "b (ncam hw) c -> b ncam hw c", ncam=ncam)
    trans_pcd = batch_transform_point_cloud(pcd, extrinsics)
    if keepdims:
        return einops.rearrange(trans_pcd, "b ncam (h w) c -> b ncam h w c", h=H)
    return einops.rearrange(trans_pcd, "b ncam (h w) c -> b (ncam h w) c")


# ---------------------------------------------------------------------------
# Farthest Point Sampling — pure PyTorch (no DGL required)
# ---------------------------------------------------------------------------

def fps_pytorch(pcd: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Farthest Point Sampling. ``pcd`` ``(B, N, 3)`` → indices ``(B, n_samples)``.

    Deterministic: always seeds at index 0 (matches DGL default ``start_idx=0``).
    Complexity O(N × n_samples).
    """
    B, N, _ = pcd.shape
    assert n_samples <= N, f"n_samples ({n_samples}) > N ({N})"
    device = pcd.device
    selected = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float("inf"), device=device)
    selected[:, 0] = 0
    last = pcd[:, 0:1, :]
    for i in range(1, n_samples):
        new_dists = ((pcd - last) ** 2).sum(-1)
        min_dists = torch.min(min_dists, new_dists)
        selected[:, i] = min_dists.argmax(dim=1)
        last = pcd[torch.arange(B, device=device), selected[:, i]].unsqueeze(1)
    return selected


# ---------------------------------------------------------------------------
# Backbone wrappers
# ---------------------------------------------------------------------------

class _DinoV2Wrapper(nn.Module):
    """Wraps DINOv2 ViT-S/14 from torch.hub to expose a spatial feature map.

    Internally resizes input to a fixed (default 224×224) so the patch grid is
    a round 16×16. Returns a ``[B, F, h, w]`` tensor where ``F=384`` and
    ``h=w=img_size//14`` (16 for img_size=224).
    """

    def __init__(self, img_size: int = 224, model_name: str = "dinov2_vits14") -> None:
        super().__init__()
        # Load via torch.hub. First call downloads the checkpoint to ~/.cache/torch/hub.
        self.vit = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.feat_dim = self.vit.embed_dim                     # 384 for ViT-S/14
        self.patch_size = self.vit.patch_embed.patch_size      # 14
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        assert img_size % self.patch_size == 0, (
            f"img_size ({img_size}) must be a multiple of DINO patch_size ({self.patch_size})"
        )
        self.img_size = img_size
        self.grid = img_size // self.patch_size                 # 16 for 224/14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [0,1], ImageNet-mean/std-normalised by the caller.
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        out = self.vit.forward_features(x)
        # DINOv2 returns dict with `x_norm_patchtokens`: [B, N_patches, D]
        patches = out["x_norm_patchtokens"]                     # [B, grid*grid, D]
        h = w = self.grid
        return einops.rearrange(patches, "b (h w) d -> b d h w", h=h, w=w)


class _ResNet18Wrapper(nn.Module):
    """ImageNet-pretrained ResNet-18 with avgpool/fc removed → spatial features."""

    def __init__(self) -> None:
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class Adapt3R3DEncoder(nn.Module):
    """Adapt3R-style 3D observation token encoder for HPT ``encoder_specs``.

    See module docstring for the full pipeline. Plugs into ``encoder_specs``
    and produces ``[B, T*I, output_dim]``.

    Args:
        output_dim: Output token width. Must equal ``shared_stem_specs.<key>.input_dim``.
        hidden_dim: Internal feature width. Must be divisible by 6 (NeRF constraint).
        num_points: FPS target. Silently clamped (with warning) if the spatial grid
            is smaller than this.
        intrinsics: 3×3 camera K matrix as nested list. Configured for the
            ``image_size_orig`` resolution; auto-rescaled to ``image_size_input``.
        extrinsics: 4×4 cam-to-world transform.
        image_size_orig: ``[W, H]`` of the sensor at which intrinsics were
            calibrated (e.g. ``[2016, 1512]`` for Aria RGB).
        image_size_input: ``[W, H]`` reaching the encoder (e.g. ``[360, 360]``).
            If equal to ``image_size_orig`` (or unset), no rescaling.
        backbone: ``"dino"`` (default) or ``"resnet18"``.
        dino_model: Hub model name when ``backbone="dino"``. Default
            ``"dinov2_vits14"``.
        dino_img_size: Internal resize for DINO. Must be a multiple of 14.
            Default 224 → 16×16 patch grid.
        do_pos: Include NeRF-encoded XYZ.
        do_image: Include backbone features.
        do_rgb: Include raw RGB at sampled points.
        finetune_backbone: If False (default), backbone params are frozen.
        dummy_depth: If set, replaces the depth channel with this constant
            (metres). Useful for verifying integration on RGB-only datasets.
    """

    def __init__(
        self,
        output_dim: int = 256,
        hidden_dim: int = 60,
        num_points: int = 512,
        intrinsics: Optional[List] = None,
        extrinsics: Optional[List] = None,
        image_size_orig: Optional[List[int]] = None,
        image_size_input: Optional[List[int]] = None,
        backbone: str = "dino",
        dino_model: str = "dinov2_vits14",
        dino_img_size: int = 224,
        do_pos: bool = True,
        do_image: bool = True,
        do_rgb: bool = False,
        finetune_backbone: bool = False,
        dummy_depth: Optional[float] = None,
        # ----- Point-cloud-space augmentation (train-mode only; see forward()) -----
        # Rationale: image-space crop/rotate is NOT used on the RGBD path (it would
        # invalidate the fixed camera intrinsics K used for back-projection). These
        # instead perturb the LIFTED CLOUD, which is geometry, not image-space, so no
        # intrinsics coupling issue. All default to 0.0 (off) => identical behaviour to
        # before this was added, including for OLD checkpoints unpickled without these
        # attributes (see the getattr() reads in forward()).
        pose_jitter_deg: float = 0.0,   # random small rotation applied to extrinsics E
        pose_jitter_m: float = 0.0,     # random small translation applied to E
        depth_scale_jitter: float = 0.0,  # multiplicative, e.g. 0.05 = +/-5%
        depth_noise_std_m: float = 0.0,   # additive Gaussian noise, metres
        **kwargs,
    ) -> None:
        super().__init__()

        assert hidden_dim % 6 == 0, (
            f"hidden_dim must be divisible by 6, got {hidden_dim}."
        )
        assert intrinsics is not None and extrinsics is not None, (
            "intrinsics (3×3) and extrinsics (4×4) must be provided in YAML."
        )
        assert do_pos or do_image, "At least one of do_pos / do_image must be True."

        # ----- Camera calibration with optional rescale -----
        # Rescale fx, cx by W_in/W_orig and fy, cy by H_in/H_orig.
        # (FISHEYE624 distortion is currently ignored; see module TODO.)
        K = torch.tensor(intrinsics, dtype=torch.float32).clone()
        if image_size_orig is not None and image_size_input is not None:
            W_o, H_o = image_size_orig
            W_i, H_i = image_size_input
            sx = float(W_i) / float(W_o)
            sy = float(H_i) / float(H_o)
            K[0, 0] *= sx        # fx
            K[0, 2] *= sx        # cx
            K[1, 1] *= sy        # fy
            K[1, 2] *= sy        # cy
        K = K.reshape(1, 1, 3, 3)
        E = torch.tensor(extrinsics, dtype=torch.float32).reshape(1, 1, 4, 4)
        self.register_buffer("K", K)
        self.register_buffer("E", E)
        # Per-batch extrinsics override (eef-frame variant): set by HPT from
        # extrinsics_key_map just before forward. None -> static self.E (all
        # existing camera-frame checkpoints restore and run unchanged).
        self._batch_E: Optional[torch.Tensor] = None

        # ----- Backbone -----
        self.backbone_type = backbone
        if backbone == "dino":
            self.backbone = _DinoV2Wrapper(img_size=dino_img_size, model_name=dino_model)
        elif backbone == "resnet18":
            self.backbone = _ResNet18Wrapper()
        else:
            raise ValueError(f"Unknown backbone='{backbone}'. Choose 'dino' or 'resnet18'.")
        self.feat_dim = self.backbone.feat_dim

        self.finetune_backbone = finetune_backbone
        if not finetune_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ----- Projections -----
        self.rgb_proj = nn.Linear(self.feat_dim, hidden_dim)
        self.xyz_proj = NeRFSinusoidalPosEmb(hidden_dim) if do_pos else None

        # ----- Per-point feature dim -----
        pc_in_dim = (int(do_pos) + int(do_image)) * hidden_dim + (3 if do_rgb else 0)

        # ----- Attention pooling -----
        self.attention_extractor = AttentionExtractor(
            in_shape=pc_in_dim,
            out_channels=output_dim,
            hidden_dim=max(hidden_dim * 4, 256),
            num_heads=4,
            use_layernorm=False,
            final_norm="none",
        )

        # ----- ImageNet normalisation (applied internally) -----
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

        # ----- Flags -----
        self.do_pos = do_pos
        self.do_image = do_image
        self.do_rgb = do_rgb
        self.num_points = num_points
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dummy_depth = dummy_depth
        self.pose_jitter_deg = pose_jitter_deg
        self.pose_jitter_m = pose_jitter_m
        self.depth_scale_jitter = depth_scale_jitter
        self.depth_noise_std_m = depth_noise_std_m

    # ------------------------------------------------------------------
    # Param-group helper for low-LR backbone fine-tuning
    # ------------------------------------------------------------------
    def get_finetune_param_groups(
        self, base_lr: float, backbone_lr_scale: float = 0.1
    ) -> List[dict]:
        """Return parameter groups suitable for AdamW with low-LR backbone.

        Use this when ``finetune_backbone=True`` and you want the DINO/ResNet
        backbone to train at a smaller LR than the rest of the encoder
        (typical convention for vision-foundation fine-tuning).

        Example::

            param_groups = encoder.get_finetune_param_groups(base_lr=1e-4,
                                                              backbone_lr_scale=0.1)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        """
        backbone_params = list(self.backbone.parameters())
        other_params = [
            p for n, p in self.named_parameters()
            if not n.startswith("backbone.") and p.requires_grad
        ]
        return [
            {"params": [p for p in backbone_params if p.requires_grad],
             "lr": base_lr * backbone_lr_scale, "name": "encoder.backbone"},
            {"params": other_params, "lr": base_lr, "name": "encoder.other"},
        ]

    # ------------------------------------------------------------------
    # Point-cloud-space augmentation (train-mode only)
    # ------------------------------------------------------------------
    def _jitter_extrinsics(self, E: torch.Tensor) -> torch.Tensor:
        """Compose E (N,ncam,4,4 cam-to-world) with a random small per-sample
        rigid transform. Regularises against the encoder memorising absolute XYZ
        coordinates tied to one fixed camera mounting pose across all demos —
        the NeRF positional embedding sees raw XYZ, so a fixed extrinsic makes
        "this coordinate value" a learnable proxy for "this point in the fixed
        rig geometry", which does not generalise. Rotation via small-angle
        Rodrigues; exact for the few-degree magnitudes this is meant to be used at.
        """
        rot_std = math.radians(getattr(self, "pose_jitter_deg", 0.0))
        trans_std = getattr(self, "pose_jitter_m", 0.0)
        if rot_std == 0.0 and trans_std == 0.0:
            return E
        orig_shape = E.shape  # (N, ncam, 4, 4); ncam is always 1 here but keep general
        E = E.reshape(-1, 4, 4)
        N = E.shape[0]
        device, dtype = E.device, E.dtype
        axis = torch.randn(N, 3, device=device, dtype=dtype)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        angle = torch.randn(N, device=device, dtype=dtype) * rot_std
        ax, ay, az = axis.unbind(-1)
        zero = torch.zeros_like(ax)
        Kx = torch.stack([
            torch.stack([zero, -az, ay], -1),
            torch.stack([az, zero, -ax], -1),
            torch.stack([-ay, ax, zero], -1),
        ], -2)  # (N, 3, 3) cross-product matrix
        I3 = torch.eye(3, device=device, dtype=dtype).expand(N, -1, -1)
        sin_a = angle.sin().view(N, 1, 1)
        cos_a = angle.cos().view(N, 1, 1)
        R = I3 + sin_a * Kx + (1 - cos_a) * torch.bmm(Kx, Kx)
        t = torch.randn(N, 3, device=device, dtype=dtype) * trans_std
        J = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
        J[:, :3, :3] = R
        J[:, :3, 3] = t
        return torch.bmm(E, J).reshape(orig_shape)

    def _jitter_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Multiplicative scale + additive Gaussian noise on valid (>0) depth
        pixels. Motivated by real, measured uncertainty in this pipeline: the
        Aria stereo baseline used to compute depth was later found to disagree
        with a nominally-identical device by ~5%, and per-pixel triangulation
        noise is inherent to stereo. Training through comparable noise should
        make the policy robust to exactly the kind of calibration slack we hit.
        """
        scale_j = getattr(self, "depth_scale_jitter", 0.0)
        noise_std = getattr(self, "depth_noise_std_m", 0.0)
        if scale_j == 0.0 and noise_std == 0.0:
            return depth
        valid = depth > 0
        out = depth
        if scale_j > 0:
            scale = 1.0 + (torch.rand(depth.shape[0], 1, 1, 1, device=depth.device, dtype=depth.dtype) * 2 - 1) * scale_j
            out = torch.where(valid, out * scale, out)
        if noise_std > 0:
            noise = torch.randn_like(out) * noise_std
            out = torch.where(valid, out + noise, out)
        return out.clamp(min=0)

    def set_batch_extrinsics(self, E: torch.Tensor) -> None:
        """Per-row cam->target extrinsics for the NEXT forward ([B, 16] or [B, 4, 4]).

        Called by HPT (extrinsics_key_map) each batch, immediately before the
        encoder forward. Cleared implicitly by being overwritten every batch; a
        stale value can never cross batches because HPT sets it unconditionally
        whenever the key is present, and configs without the key never call this.
        """
        self._batch_E = E

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_internals: bool = False
    ) -> torch.Tensor:
        """Encode an RGBD observation.

        Args:
            x: ``(B, T, I, 4, H, W)`` — channels [:3] = RGB in [0,1];
                channel [3] = depth in metres (or 0 when ``dummy_depth`` is set).
            return_internals: If ``True``, return a dict with intermediate
                tensors useful for visualisation: ``{token, pcd_full, pcd_fps,
                feat_grid, feat_fps, attn}``.
        Returns:
            ``(B, T*I, output_dim)`` token, or ``(token, internals)`` when
            ``return_internals=True``.
        """
        B, T, I, C, H, W = x.shape
        assert C == 4, f"Adapt3R3DEncoder expects 4-channel RGBD; got C={C}."

        rgb = x[..., :3, :, :]
        depth = x[..., 3:4, :, :]

        if self.dummy_depth is not None:
            depth = torch.full_like(depth, float(self.dummy_depth))

        N_frames = T * I
        rgb_flat = rgb.reshape(B * N_frames, 3, H, W)
        depth_flat = depth.reshape(B * N_frames, 1, H, W)

        K = self.K.expand(B * N_frames, -1, -1, -1)
        # getattr, NOT self._batch_E: HPT is pickled whole; pre-eef Adapt3R
        # checkpoints restore WITHOUT this attribute and a bare access raises
        # AttributeError on first inference (reproduced 2026-08-18, a3r_human
        # ep1999). LOCAL FIX — re-apply after every snapshot apply until
        # upstream adopts it (their in_dim fix used the same pattern).
        batch_E = getattr(self, "_batch_E", None)
        if batch_E is not None:
            # [B, 16] flattened row-major 4x4 (LeRobot stores 1-D features), one
            # per row; T=I=1 in this pipeline so B*N_frames == B.
            E = batch_E.reshape(-1, 1, 4, 4).to(dtype=self.E.dtype)
            assert E.shape[0] == B * N_frames, (
                f"batch extrinsics {tuple(batch_E.shape)} vs B*N_frames {B * N_frames}"
            )
        else:
            E = self.E.expand(B * N_frames, -1, -1, -1)

        if self.training:
            E = self._jitter_extrinsics(E)
            depth_flat = self._jitter_depth(depth_flat)

        # ----- Step 1: backbone features -----
        rgb_norm = (rgb_flat - self.rgb_mean) / self.rgb_std
        backbone_trainable = self.training and any(
            p.requires_grad for p in self.backbone.parameters()
        )
        with torch.set_grad_enabled(backbone_trainable):
            feat_map = self.backbone(rgb_norm)             # [BN, F, h, w]

        h, w = feat_map.shape[-2:]
        feat = feat_map.flatten(2).transpose(1, 2)         # [BN, h*w, F]
        feat = self.rgb_proj(feat)                         # [BN, h*w, hidden_dim]

        # ----- Step 2: lift point cloud (at full input resolution) → downsample to (h, w) -----
        pcd_full = lift_point_cloud_batch(
            depth_flat, K, E, keepdims=True
        )                                                   # [BN, 1, H, W, 3]
        pcd_chw = einops.rearrange(pcd_full, "b 1 h w c -> b c h w")
        pcd_ds = F.interpolate(pcd_chw.float(), size=(h, w), mode="nearest")
        pcd = einops.rearrange(pcd_ds, "b c h w -> b (h w) c")    # [BN, h*w, 3]

        # ----- Step 3: FPS -----
        n_spatial = pcd.shape[1]
        n_sample = self.num_points
        if n_sample > n_spatial:
            warnings.warn(
                f"Adapt3R3DEncoder.num_points={n_sample} > spatial grid {n_spatial} "
                f"({h}×{w}); clamping to {n_spatial}. Consider lowering num_points "
                "or using a larger backbone output (e.g. ResNet18 instead of DINO ViT-S/14).",
                stacklevel=2,
            )
            n_sample = n_spatial
        fps_idx = fps_pytorch(pcd, n_sample)
        idx_exp = fps_idx.unsqueeze(-1)
        pcd_fps = torch.gather(pcd, 1, idx_exp.expand(-1, -1, 3))
        feat_fps = torch.gather(feat, 1, idx_exp.expand(-1, -1, feat.shape[-1]))

        # ----- Step 4: per-point feature assembly -----
        parts: List[torch.Tensor] = []
        if self.do_pos:
            pos_emb = self.xyz_proj(pcd_fps)               # [BN, n_sample, hidden_dim]
            parts.append(pos_emb)
        if self.do_image:
            parts.append(feat_fps)
        if self.do_rgb:
            rgb_ds = F.interpolate(
                rgb_flat, size=(h, w), mode="bilinear", align_corners=False
            )
            rgb_ds = einops.rearrange(rgb_ds, "b c h w -> b (h w) c")
            rgb_fps = torch.gather(rgb_ds, 1, idx_exp.expand(-1, -1, 3))
            parts.append(rgb_fps)
        cat_cloud = torch.cat(parts, dim=-1)               # [BN, n_sample, pc_in_dim]

        # ----- Step 5: attention pooling -----
        cat_cloud = cat_cloud.unsqueeze(1)                 # [BN, 1, n_sample, pc_in_dim]
        if return_internals:
            tokens, attn = self.attention_extractor(cat_cloud, return_attn=True)
        else:
            tokens = self.attention_extractor(cat_cloud)
            attn = None

        token = tokens.squeeze(1).reshape(B, N_frames, self.output_dim)

        if not return_internals:
            return token

        internals = {
            "token": token,                                                     # [B, N_frames, output_dim]
            "feat_grid": feat_map.reshape(B, N_frames, *feat_map.shape[1:]),    # [B, N_frames, F, h, w]
            "pcd_full": pcd_full.reshape(B, N_frames, H, W, 3),                 # [B, N_frames, H, W, 3]
            "pcd_ds": pcd.reshape(B, N_frames, h, w, 3),                        # [B, N_frames, h, w, 3]
            "pcd_fps": pcd_fps.reshape(B, N_frames, n_sample, 3),               # [B, N_frames, N, 3]
            "feat_fps": feat_fps.reshape(B, N_frames, n_sample, -1),            # [B, N_frames, N, hidden_dim]
            "fps_idx": fps_idx.reshape(B, N_frames, n_sample),                  # [B, N_frames, N]
            "attn": attn.reshape(B, N_frames, *attn.shape[2:]) if attn is not None else None,
        }
        return token, internals
