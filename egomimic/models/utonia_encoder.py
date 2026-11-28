"""Pretrained Utonia (PTv3) point-cloud encoder for HPT ``encoder_specs``.

Utonia (https://github.com/Pointcept/Utonia, weights https://huggingface.co/Pointcept/Utonia) is a self-supervised,
cross-domain Point Transformer V3, encoder only. This module puts it behind the contract of ``DP3PointNetEncoder`` /
``PatchPointEncoder``: flattened clouds ``(B, T, I, N*in_dim)`` (xyz metres in the stream's own frame + rgb in [0, 1])
in, ``(B, T*I*G, output_dim)`` tokens out, so stems / trunk / flow head stay as they are and a comparison isolates the
encoder. N may differ between batches (nothing is sized by it).

Per cloud, the steps of the official ``utonia.transform.default(scale)``, done batched on the GPU: optional fixed
rotation -> xyz * coord_scale -> CenterShift (xy box centre, z min) -> GridSample at grid_size (one point per voxel:
random in train mode, the first in eval) -> feat = [coord, colour, normal = 0] (9 channels). The backbone's levels are
concatenated back to the finest voxels (the README's "upcast", ``concat_levels`` of them) and mapped to every input
point through the grid-sample inverse. Tokens: G farthest-point centres in the ORIGINAL stream frame (absolute position
kept for navigation), each the max over its K nearest points of a learned projection of their features, plus a
positional MLP of the centre xyz (as in PatchPointEncoder).

``finetune_stages=()`` and ``lora_rank=0`` freeze the backbone (no_grad). Otherwise those encoder stages train, or
rank-``lora_rank`` adapters on every block's qkv / proj / fc1 / fc2 (base weights frozen), and
``pl_model.configure_optimizers`` gives them ``backbone_lr``. The backbone always runs in eval mode (no stochastic depth),
in ``backbone_dtype`` under autocast and with serialization-order shuffling off, so a cloud always gets the same
features. Train through it in bf16: fp16 gradients underflow without a loss scaler.

``attention="whole_cloud"`` (default): each cloud is ONE attention patch, which is what the pretrained model computes
for it (flash attention: patches of 1024 points, no padding for smaller clouds) and what the stock non-flash path
computes at batch size 1. Every cloud fits, as no stage has more points than the input's 1024. The stock non-flash path
(``attention="batched"``, the 10-27 wave-1 runs) instead cuts each cloud into windows the size of the batch's smallest
cloud, so features depend on the batch, and loops over the clouds in Python (~2k GPU syncs per call, the bulk of the
step time). Attention over a whole cloud is independent of point order (RoPE encodes coordinates), so serialization
is reduced to the cheap z order.
HPT.finalize_modules re-initialises every Linear / LayerNorm; the backbone's modules carry ``_hpt_keep_init`` and
HPT._init_weights skips them. Needs the utonia package (utonia_src) plus spconv, torch_scatter and addict on the path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.adapt3r_3d_encoder import fps_pytorch
from egomimic.models.pcd_aug import dp3_geometric_jitter

def _cpu_bias_off(m, args):  # spconv conv forward pre-hook on CPU: run the conv without its bias ...
    m._cpu_bias, m.bias = m.bias, None


def _cpu_bias_on(m, args, out):  # ... and add it to the output features afterwards
    bias, m.bias = m._cpu_bias, m._cpu_bias
    return out.replace_feature(out.features + bias)


def _lora_hook(m, args, out):  # forward hook of a frozen backbone Linear: + low-rank update (B starts at 0)
    return out + (args[0] @ m.lora_a.T @ m.lora_b.T) * m.lora_scale


def _cloud_patches(point, patch_size: int):
    """Per stage (cached in the Point): each point's slot in a (clouds, K) padded layout, the key mask (None when all
    clouds have K points) and K = the largest cloud, rounded up to 16 for the attention kernel's alignment."""
    if "wc_index" not in point.keys():
        offset = point.offset
        counts = torch.diff(offset, prepend=offset.new_zeros(1))
        lo, hi = torch.stack([counts.min(), counts.max()]).tolist()  # the stage's one host sync
        assert hi <= patch_size, f"a cloud has {hi} > {patch_size} points at some stage; use attention='batched'"
        K = -(-hi // 16) * 16
        start = offset - counts
        local = torch.arange(len(point.batch), device=offset.device) - start[point.batch]
        point["wc_index"] = point.batch * K + local
        point["wc_mask"] = None if lo == K else (torch.arange(K, device=offset.device) < counts[:, None])[:, None, None]
        point["wc_K"] = K
    return point.wc_index, point.wc_mask, point.wc_K


class _WholeCloudAttention(nn.Module):
    """Drop-in for a block's utonia SerializedAttention (same submodules, so the same state_dict keys): every cloud
    attends over all of its own points. Same dtype handling as the stock SDPA path."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        assert not attn.enable_rpe
        self.num_heads, self.channels, self.scale = attn.num_heads, attn.channels, attn.scale
        self.patch_size = getattr(attn, "patch_size_max", attn.patch_size)
        self.upcast_attention = attn.upcast_attention
        self.qkv, self.proj, self.proj_drop, self.rope = attn.qkv, attn.proj, attn.proj_drop, attn.rope

    def forward(self, point):
        H, C = self.num_heads, self.channels
        D = C // H
        index, mask, K = _cloud_patches(point, self.patch_size)
        nc = point.offset.shape[0]
        qkv = self.qkv(point.feat)
        q, k, v = qkv.reshape(-1, 3, H, D).unbind(dim=1)
        q, k = self.rope(q, k, point.coord)
        if self.upcast_attention:
            q, k = q.float(), k.float()
        v = v.to(q.dtype)
        Dp = D + (-D % 8)  # the memory-efficient kernel needs head dim % 8 == 0; zero padding leaves q.k unchanged

        def pack(x):
            out = x.new_zeros(nc * K, H, Dp)
            out[index, :, :D] = x
            return out.view(nc, K, H, Dp).transpose(1, 2)

        feat = F.scaled_dot_product_attention(pack(q), pack(k), pack(v), attn_mask=mask, scale=self.scale)
        feat = feat.transpose(1, 2).reshape(nc * K, H, Dp)[index, :, :D].to(qkv.dtype).reshape(-1, C)
        point.feat = self.proj_drop(self.proj(feat))
        return point


_ROTATIONS = {
    "none": None,
    # camera / glass frame (x right, y down, z forward) -> z up (x right, y forward, z up)
    "cam_to_zup": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
}


class UtoniaPointEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 256,
        num_points: int | None = None,
        in_dim: int = 6,
        ckpt_path: str = "",
        num_groups: int = 16,
        group_size: int = 64,
        hidden: int = 256,
        coord_scale: float = 1.0,
        grid_size: float = 0.01,
        input_rotation: str = "none",
        concat_levels: int = 4,
        finetune_stages: tuple = (),
        backbone_lr: float | None = None,
        backbone_dtype: str = "fp32",
        serialization_order: tuple | None = None,
        attention: str = "whole_cloud",
        enc_patch_size: int | None = None,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        colour_brightness_jitter: float = 0.0,
        colour_channel_jitter: float = 0.0,
        pose_jitter_deg: float = 0.0,
        pose_jitter_m: float = 0.0,
        point_noise_std_m: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        import utonia  # noqa: E402  (utonia_src + spconv / torch_scatter / addict on PYTHONPATH)

        self.output_dim = output_dim
        self.num_points = num_points
        self.in_dim = in_dim
        self.num_groups = num_groups
        self.group_size = group_size
        self.coord_scale = coord_scale
        self.grid_size = grid_size
        self.concat_levels = concat_levels
        self.backbone_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[backbone_dtype]
        self.colour_brightness_jitter = colour_brightness_jitter
        self.colour_channel_jitter = colour_channel_jitter
        self.pose_jitter_deg, self.pose_jitter_m, self.point_noise_std_m = pose_jitter_deg, pose_jitter_m, point_noise_std_m
        rot = _ROTATIONS[input_rotation]
        self.register_buffer("rotation", None if rot is None else torch.tensor(rot), persistent=False)

        assert attention in ("whole_cloud", "batched"), attention
        self.attention = attention
        custom = dict(enable_flash=False, shuffle_orders=False)
        if serialization_order:  # "batched" only: the order picks the attention windows
            custom["order"] = tuple(serialization_order)
        if attention == "whole_cloud":
            custom["order"] = ("z",)  # unused by whole-cloud attention; z is the cheapest to compute
        ckpt = utonia.model.load(ckpt_path, custom_config=custom, ckpt_only=True)
        if enc_patch_size:  # 10-28 cart (user: 2048-point gaze cloud): attention windows of this size at every stage (pretraining: 1024)
            ckpt["config"]["enc_patch_size"] = (int(enc_patch_size),) * len(ckpt["config"]["enc_patch_size"])
        self.backbone = utonia.model.PointTransformerV3(**ckpt["config"])
        self.backbone.load_state_dict(ckpt["state_dict"])
        print(f"[utonia] pretrained weights loaded from {ckpt_path}", flush=True)
        blocks = [m for m in self.backbone.modules() if isinstance(m, utonia.model.Block)]
        if attention == "whole_cloud":
            for blk in blocks:
                blk.attn = _WholeCloudAttention(blk.attn)
        for m in self.backbone.modules():
            if hasattr(m, "shuffle_orders"):  # GridPooling defaults to True whatever the config says
                m.shuffle_orders = False
            m._hpt_keep_init = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        stages = dict(self.backbone.enc.named_children())
        self.finetune_stages = tuple(int(s) for s in finetune_stages)
        for s in self.finetune_stages:
            for p in stages[f"enc{s}"].parameters():
                p.requires_grad = True
        self.lora_rank = int(lora_rank)
        if self.lora_rank:
            for blk in blocks:
                for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp[0].fc1, blk.mlp[0].fc2):
                    lin.lora_a = nn.Parameter(torch.empty(self.lora_rank, lin.in_features))
                    nn.init.kaiming_uniform_(lin.lora_a, a=math.sqrt(5))
                    lin.lora_b = nn.Parameter(torch.zeros(lin.out_features, self.lora_rank))
                    lin.lora_scale = (lora_alpha or self.lora_rank) / self.lora_rank
                    lin.register_forward_hook(_lora_hook)
        self.backbone_lr = backbone_lr if (self.finetune_stages or self.lora_rank) else None

        channels = list(ckpt["config"]["enc_channels"])
        feat_dim = sum(channels[len(channels) - 1 - concat_levels:])
        self.point_proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.proj = nn.Linear(hidden, output_dim)
        self.pos = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def backbone_trainable_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def compile(self, *args, **kwargs):
        # EGOMIMIC_COMPILE compiles every encoder; spconv + per-batch voxel counts don't fit CUDA graphs. Stay eager.
        return None

    def _match_spconv_algo(self, device_type: str) -> None:
        """spconv's default MaskImplicitGemm is CUDA-only; CPU (the sim evals' policy servers) needs Native. spconv keeps
        every weight in KRSC layout whatever the algorithm (spconv.constants.ALL_WEIGHT_IS_KRSC), so switching is safe.
        spconv's CPU kernel also refuses a fused bias, so on CPU each conv runs without it and the bias is added after."""
        if getattr(self, "_spconv_device", None) == device_type:
            return
        from spconv.pytorch import ConvAlgo
        from spconv.pytorch.conv import SparseConvolution
        for m in self.backbone.modules():
            if isinstance(m, SparseConvolution):
                if not hasattr(m, "_algo_cuda"):
                    m._algo_cuda = m.algo
                m.algo = ConvAlgo.Native if device_type == "cpu" else m._algo_cuda
                for h in getattr(m, "_cpu_bias_hooks", ()):
                    h.remove()
                m._cpu_bias_hooks = ()
                if device_type == "cpu" and m.bias is not None:
                    m._cpu_bias_hooks = (m.register_forward_pre_hook(_cpu_bias_off), m.register_forward_hook(_cpu_bias_on))
        self._spconv_device = device_type

    def _grid_sample(self, coord: torch.Tensor):
        """coord (BF, N, 3), already scaled + centre-shifted. Returns, per voxel, the index of its representative point
        in the flattened (BF*N) cloud, its cloud and int grid coordinate, and every point's voxel index."""
        BF, N, _ = coord.shape
        dev = coord.device
        g = torch.floor(coord / self.grid_size).long()
        g = g - g.amin(dim=1, keepdim=True)
        dims = g.reshape(-1, 3).amax(0) + 1
        b = torch.arange(BF, device=dev).view(BF, 1).expand(BF, N)
        key = ((b * dims[0] + g[..., 0]) * dims[1] + g[..., 1]) * dims[2] + g[..., 2]
        _, inv = torch.unique(key.reshape(-1), return_inverse=True)  # sorted keys -> voxels grouped by cloud, in order
        M, P = int(inv.max()) + 1, BF * N
        idx = torch.arange(P, device=dev)
        cand = idx
        if self.training:  # GridSample(mode="train"): a random point per voxel
            prio = torch.rand(P, device=dev)
            best = torch.full((M,), 2.0, device=dev).scatter_reduce(0, inv, prio, "amin")
            cand = torch.where(prio == best[inv], idx, P)
        rep = torch.full((M,), P, dtype=torch.long, device=dev).scatter_reduce(0, inv, cand, "amin")
        return rep, rep // N, g.reshape(-1, 3)[rep].int(), inv

    def _point_features(self, xyz: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """(BF, N, 3) xyz + (BF, N, 3) rgb -> (BF, N, feat_dim) Utonia features of every input point."""
        BF, N, _ = xyz.shape
        self._match_spconv_algo(xyz.device.type)
        with torch.autocast(device_type=xyz.device.type, enabled=False):
            coord = xyz.float()
            if self.rotation is not None:
                coord = coord @ self.rotation.T
            coord = coord * self.coord_scale
            lo, hi = coord.amin(1, keepdim=True), coord.amax(1, keepdim=True)
            coord = coord - torch.cat([(lo[..., :2] + hi[..., :2]) / 2, lo[..., 2:]], -1)
            rep, batch, grid_coord, inv = self._grid_sample(coord)
            coord_v = coord.reshape(-1, 3)[rep]
            feat = torch.cat([coord_v, rgb.float().reshape(-1, 3)[rep], torch.zeros_like(coord_v)], -1)
            offset = torch.cumsum(torch.bincount(batch, minlength=BF), 0)
            data = dict(coord=coord_v, grid_coord=grid_coord, feat=feat, offset=offset, grid_size=self.grid_size)
            half = self.backbone_dtype is not None and xyz.device.type == "cuda"
            trains = bool(self.finetune_stages) or bool(getattr(self, "lora_rank", 0))  # getattr: pre-LoRA checkpoints
            with torch.set_grad_enabled(torch.is_grad_enabled() and trains), \
                    torch.autocast("cuda", dtype=self.backbone_dtype or torch.bfloat16, enabled=half):
                point = self.backbone(data)
                level = 0
                while "pooling_parent" in point.keys():
                    parent, inverse = point.pop("pooling_parent"), point.pop("pooling_inverse")
                    up = point.feat[inverse]
                    parent.feat = torch.cat([parent.feat, up], -1) if level < self.concat_levels else up
                    point, level = parent, level + 1
            return point.feat[inv].reshape(BF, N, -1).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, I, _ = x.shape
            n_frames = T * I
            pc = x.reshape(B * n_frames, -1, self.in_dim)
        elif x.dim() == 3 and x.shape[-1] == self.in_dim:
            B, n_frames, pc = x.shape[0], 1, x
        elif x.dim() == 2:
            B, n_frames = x.shape[0], 1
            pc = x.reshape(B, -1, self.in_dim)
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
        if self.num_points is not None and pc.shape[1] != self.num_points:
            raise ValueError(f"Expected {self.num_points} points per cloud, got {pc.shape[1]}")
        if self.training and (self.pose_jitter_deg > 0.0 or self.pose_jitter_m > 0.0 or self.point_noise_std_m > 0.0):
            pc = torch.cat([dp3_geometric_jitter(pc[..., :3], self.pose_jitter_deg, self.pose_jitter_m,
                                                 self.point_noise_std_m), pc[..., 3:]], -1)
        b, c = self.colour_brightness_jitter, self.colour_channel_jitter
        if self.training and (b > 0.0 or c > 0.0):
            # PatchPointEncoder's train-only photometric gain: one brightness and one per-channel gain per cloud
            BF = pc.shape[0]
            gain = torch.ones(BF, 1, 1, device=pc.device, dtype=pc.dtype)
            if b > 0.0:
                gain = gain * (1.0 + (torch.rand(BF, 1, 1, device=pc.device, dtype=pc.dtype) * 2 - 1) * b)
            if c > 0.0:
                gain = gain * (1.0 + (torch.rand(BF, 1, 3, device=pc.device, dtype=pc.dtype) * 2 - 1) * c)
            pc = torch.cat([pc[..., :3], (pc[..., 3:6] * gain).clamp(0.0, 1.0)], -1)
        xyz = pc[..., :3]
        f = self.point_proj(self._point_features(xyz, pc[..., 3:6]))                     # (BF, N, hidden)
        with torch.no_grad():
            cidx = fps_pytorch(xyz, self.num_groups)                                     # (BF, G)
            centres = torch.gather(xyz, 1, cidx.unsqueeze(-1).expand(-1, -1, 3))         # (BF, G, 3)
            knn = torch.cdist(centres, xyz).topk(self.group_size, largest=False).indices  # (BF, G, K)
        grp = torch.gather(f.unsqueeze(1).expand(-1, self.num_groups, -1, -1), 2,
                           knn.unsqueeze(-1).expand(-1, -1, -1, f.shape[-1]))            # (BF, G, K, hidden)
        tok = self.norm(self.proj(grp.max(dim=2)[0]) + self.pos(centres))
        return tok.reshape(B, n_frames * self.num_groups, self.output_dim)
