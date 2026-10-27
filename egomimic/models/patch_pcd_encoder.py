"""Point-patch tokenizer for HPT ``encoder_specs`` -- an object-aware alternative to DP3's single token.

DP3 max-pools a whole cloud into ONE vector: it keeps *what* is present but only implicitly *where*,
which caps how precisely a policy can localize an object (the basket). This encoder follows the
Point-BERT / Point-MAE patch tokenization: G farthest-point-sampled centres, each with its K nearest
points expressed relative to the centre, encoded by a shared mini-PointNet (max-pool inside the patch);
the centre's xyz is added back through a learned positional MLP. Output is G tokens per cloud, so each
token states what is there AND where, and the stem's cross-attention can attend to the patch holding the
object. A patch centroid of K points localizes to ~spacing/sqrt(K), well below the cloud's point spacing.

Same input contract as ``DP3PointNetEncoder`` (flattened ``(B, T, I, N*in_dim)``), but returns G tokens
per cloud: ``(B, T*I*G, output_dim)``. ``use_rgb=False`` (default) reads xyz only, so it is
lighting-immune by construction. With ``use_rgb=True``, ``colour_brightness_jitter`` /
``colour_channel_jitter`` apply DP3PointNetEncoder's train-only photometric gain (default off).
``pose_jitter_deg`` / ``pose_jitter_m`` / ``point_noise_std_m`` apply DP3PointNetEncoder's train-only geometric jitter
to xyz before patching (``pcd_aug.dp3_geometric_jitter``; default off).

Optional VoteNet-style supervision of the tokens (``vote_weight > 0``, train only): HPT passes the raw
per-sample target position in this cloud's frame (``set_vote_target``); every patch whose centre lies within
``vote_r_pos`` of it is an object patch (balanced BCE objectness vs patches beyond ``vote_r_neg``), and object
patches regress their offset to the target (smooth-L1, in units of 10 cm). The loss is local and
translation-equivariant, so it cannot be met by recalling a per-episode position. HPT adds
``pop_vote_loss()`` to the training loss. Default off: no extra modules, identical state_dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.models.adapt3r_3d_encoder import fps_pytorch
from egomimic.models.pcd_aug import dp3_geometric_jitter


class PatchPointEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 256,
        num_points: int = 1024,
        in_dim: int = 6,
        use_rgb: bool = False,
        num_groups: int = 64,
        group_size: int = 32,
        hidden: int = 128,
        colour_brightness_jitter: float = 0.0,
        colour_channel_jitter: float = 0.0,
        pose_jitter_deg: float = 0.0,
        pose_jitter_m: float = 0.0,
        point_noise_std_m: float = 0.0,
        vote_weight: float = 0.0,
        vote_slice: tuple = (0, 3),
        vote_r_pos: float = 0.30,
        vote_r_neg: float = 0.45,
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_points = num_points
        self.in_dim = in_dim
        self.use_rgb = use_rgb
        self.num_groups = num_groups
        self.group_size = group_size
        self.colour_brightness_jitter = colour_brightness_jitter
        self.colour_channel_jitter = colour_channel_jitter
        self.pose_jitter_deg, self.pose_jitter_m, self.point_noise_std_m = pose_jitter_deg, pose_jitter_m, point_noise_std_m
        fin = 6 if use_rgb else 3
        self.point_mlp = nn.Sequential(
            nn.Linear(fin, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.proj = nn.Linear(hidden, output_dim)
        self.pos = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.vote_weight = vote_weight
        self.vote_slice = tuple(vote_slice)
        self.vote_r_pos, self.vote_r_neg = vote_r_pos, vote_r_neg
        self._vote_target, self._vote_loss = None, None
        if vote_weight > 0:
            self.obj_head = nn.Sequential(nn.Linear(output_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.vote_head = nn.Sequential(nn.Linear(output_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def set_vote_target(self, target: torch.Tensor) -> None:
        """target: (B, >=3) raw metric positions; this cloud's frame is target[:, vote_slice]."""
        self._vote_target = target[:, self.vote_slice[0]:self.vote_slice[1]]

    def pop_vote_loss(self):
        loss, self._vote_loss = self._vote_loss, None
        return loss

    def _compute_vote_loss(self, tok, centres, n_frames):
        tgt = self._vote_target.to(tok.dtype).repeat_interleave(n_frames, 0)       # (BF, 3)
        off = tgt[:, None, :] - centres                                               # (BF, G, 3)
        dist = off.norm(dim=-1)
        pos, neg = dist < self.vote_r_pos, dist > self.vote_r_neg
        bce = F.binary_cross_entropy_with_logits(self.obj_head(tok).squeeze(-1), pos.to(tok.dtype), reduction="none")
        obj = 0.5 * (bce * pos).sum() / pos.sum().clamp(min=1) + 0.5 * (bce * neg).sum() / neg.sum().clamp(min=1)
        vl = F.smooth_l1_loss(self.vote_head(tok), off, beta=0.05, reduction="none").sum(-1)
        vote = (vl * pos).sum() / pos.sum().clamp(min=1) / 0.1
        return self.vote_weight * (obj + vote)

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
        if pc.shape[1] != self.num_points:
            raise ValueError(f"Expected {self.num_points} points per cloud, got {pc.shape[1]}")
        if self.training and (self.pose_jitter_deg > 0.0 or self.pose_jitter_m > 0.0 or self.point_noise_std_m > 0.0):
            pc = torch.cat([dp3_geometric_jitter(pc[..., :3], self.pose_jitter_deg, self.pose_jitter_m,
                                                 self.point_noise_std_m), pc[..., 3:]], -1)
        b, c = self.colour_brightness_jitter, self.colour_channel_jitter
        if self.training and self.use_rgb and (b > 0.0 or c > 0.0):
            # DP3PointNetEncoder._augment's photometric gain: one brightness and one per-channel gain per cloud
            BF = pc.shape[0]
            gain = torch.ones(BF, 1, 1, device=pc.device, dtype=pc.dtype)
            if b > 0.0:
                gain = gain * (1.0 + (torch.rand(BF, 1, 1, device=pc.device, dtype=pc.dtype) * 2 - 1) * b)
            if c > 0.0:
                gain = gain * (1.0 + (torch.rand(BF, 1, 3, device=pc.device, dtype=pc.dtype) * 2 - 1) * c)
            pc = torch.cat([pc[..., :3], (pc[..., 3:6] * gain).clamp(0.0, 1.0)], -1)
        xyz = pc[..., :3]
        with torch.no_grad():
            cidx = fps_pytorch(xyz, self.num_groups)                                    # (BF, G)
            centres = torch.gather(xyz, 1, cidx.unsqueeze(-1).expand(-1, -1, 3))        # (BF, G, 3)
            knn = torch.cdist(centres, xyz).topk(self.group_size, largest=False).indices  # (BF, G, K)
        C = pc.shape[-1]
        grp = torch.gather(pc.unsqueeze(1).expand(-1, self.num_groups, -1, -1), 2,
                           knn.unsqueeze(-1).expand(-1, -1, -1, C))                     # (BF, G, K, C)
        local = grp[..., :3] - centres.unsqueeze(2)
        feat = torch.cat([local, grp[..., 3:6]], -1) if self.use_rgb else local
        f = self.point_mlp(feat).max(dim=2)[0]                                          # (BF, G, hidden)
        tok = self.norm(self.proj(f) + self.pos(centres))
        if self.training and self.vote_weight > 0 and self._vote_target is not None:
            self._vote_loss = self._compute_vote_loss(tok, centres, n_frames)
        self._vote_target = None
        return tok.reshape(B, n_frames * self.num_groups, self.output_dim)
