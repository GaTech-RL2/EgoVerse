"""Robotics Diffusion Transformer (RDT, https://github.com/thu-ml/RoboticsDiffusionTransformer)
building blocks. ``RDTBackbone`` is the DiT shared by every embodiment (the
policy's trunk); ``RDTDenoiser`` is one embodiment's adaptor / output layer
around it, used as the denoiser of a ``DenoisingPolicy`` head."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_1d(embed_dim: int, positions) -> np.ndarray:
    """(M,) positions -> (M, embed_dim) sin-cos table."""
    assert embed_dim % 2 == 0
    omega = 1.0 / 10000 ** (
        np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2)
    )
    out = np.einsum("m,d->md", np.asarray(positions, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def multimodal_sincos(embed_dim: int, lens: list) -> torch.Tensor:
    """RDT's input position table: the first half of each row encodes which
    segment the token belongs to, the second half its index inside it."""
    half = embed_dim // 2
    segment = sincos_1d(half, np.arange(len(lens)))
    rows = []
    for idx, length in enumerate(lens):
        row = np.zeros((length, embed_dim))
        row[:, :half] = segment[idx]
        row[:, half:] = sincos_1d(embed_dim - half, np.arange(length))
        rows.append(row)
    return torch.from_numpy(np.concatenate(rows, axis=0)).float()


def sincos_2d(embed_dim: int, grid_hw, ref_grid: float = 16.0) -> torch.Tensor:
    """(h * w, embed_dim) raster-order table, half the channels per axis.

    Coordinates are normalized to [0, ref_grid] on both axes, so an edge or the
    centre gets the same code at any resolution or aspect ratio.
    """
    h, w = grid_hw
    half = embed_dim // 2
    rows = np.linspace(0.0, ref_grid, h) if h > 1 else np.zeros(1)
    cols = np.linspace(0.0, ref_grid, w) if w > 1 else np.zeros(1)
    table = np.concatenate(
        [
            np.repeat(sincos_1d(half, rows), w, axis=0),
            np.tile(sincos_1d(embed_dim - half, cols), (h, 1)),
        ],
        axis=1,
    )
    return torch.from_numpy(table).float()


def mlp_gelu(in_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    """RDT's ``mlp{depth}x_gelu`` condition adaptor."""
    layers = [nn.Linear(in_dim, out_dim)]
    for _ in range(1, depth):
        layers += [nn.GELU(approximate="tanh"), nn.Linear(out_dim, out_dim)]
    return nn.Sequential(*layers)


@dataclass
class RDTConditions:
    """What the denoiser is conditioned on, already at the DiT width.

    Quacks like the single conditioning tensor ``DenoisingPolicy`` expects
    (``len`` / ``dtype`` / ``device``), so the FM / diffusion heads take it
    unchanged.
    """

    img: torch.Tensor  # (B, N_img, D)
    freq: torch.Tensor  # (B,) control frequency, Hz
    backbone: "RDTBackbone"  # the shared trunk the denoiser runs through
    state: Optional[torch.Tensor] = None  # (B, K, D)
    lang: Optional[torch.Tensor] = None  # (B, L, D)
    lang_mask: Optional[torch.Tensor] = None  # (B, L) bool, True = attend
    memory: Optional[torch.Tensor] = None  # (B, M, D), self-attended
    memory_mask: Optional[torch.Tensor] = None  # (B, M) bool, True = attend

    def __len__(self):
        return self.img.shape[0]

    @property
    def dtype(self):
        return self.img.dtype

    @property
    def device(self):
        return self.img.device


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class _Attention(nn.Module):
    """Self- (``context=None``) or cross-attention with QK RMSNorm."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape
        context = x if context is None else context
        L = context.shape[1]
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = (
            self.kv(context)
            .view(B, L, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if mask is not None:
            mask = mask.view(B, 1, 1, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, out_dim: Optional[int] = None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden, out_dim or dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RDTBlock(nn.Module):
    """Pre-RMSNorm self-attention, cross-attention to one condition, FFN."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 1.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = _Attention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.cross_attn = _Attention(hidden_size, num_heads)
        self.norm3 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.ffn = _Mlp(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, x, c, mask=None, self_mask=None):
        x = x + self.attn(self.norm1(x), mask=self_mask)
        x = x + self.cross_attn(self.norm2(x), c, mask)
        return x + self.ffn(self.norm3(x))


def _xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class RDTBackbone(nn.Module):
    """The embodiment-agnostic part of RDT: timestep / control-frequency
    embedders and the blocks, which alternate the condition they cross-attend
    to (language, image, language, ...; image only when there is no language).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 28,
        n_heads: int = 16,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.freq_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [RDTBlock(hidden_dim, n_heads, mlp_ratio) for _ in range(depth)]
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(_xavier)
        for embedder in (self.t_embedder, self.freq_embedder):
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)

    def prefix(self, t: torch.Tensor, freq: torch.Tensor, batch_size: int):
        """(B, 2, D): the timestep and control-frequency tokens."""
        t = self.t_embedder(t.reshape(-1)).unsqueeze(1).expand(batch_size, -1, -1)
        return torch.cat([t, self.freq_embedder(freq.reshape(-1)).unsqueeze(1)], dim=1)

    def forward(
        self,
        h: torch.Tensor,
        cond: "RDTConditions",
        self_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``self_mask`` (B, N) bool: which of ``h``'s tokens may be attended
        to (padded memory steps may not)."""
        conds = [(cond.img, None)]
        if cond.lang is not None:
            conds.insert(0, (cond.lang, cond.lang_mask))
        for i, block in enumerate(self.blocks):
            c, mask = conds[i % len(conds)]
            h = block(h, c, mask, self_mask)
        return h


class RDTDenoiser(nn.Module):
    """One embodiment's view of RDT: the action adaptor, the position table of
    ``[timestep; ctrl freq; state tokens; memory tokens; noisy action chunk]``
    and the output layer. The blocks in between are ``cond.backbone``, shared across
    embodiments. ``forward`` has the ``model(x_t, t, cond)`` signature of the
    other denoising nets, with ``cond`` an ``RDTConditions``.

    ``time_scale`` maps the head's time onto RDT's [0, 1000) embedding range:
    1000 for flow matching (t in [0, 1]), 1 for a DDPM head (integer steps).
    """

    def __init__(
        self,
        act_dim: int,
        act_seq: int,
        hidden_dim: int = 1024,
        n_state_tokens: int = 1,
        time_scale: float = 1000.0,
        n_memory_tokens: int = 0,
    ):
        super().__init__()
        self.act_seq = act_seq
        self.n_state_tokens = n_state_tokens
        self.n_memory_tokens = n_memory_tokens
        self.time_scale = time_scale
        self.action_adaptor = mlp_gelu(act_dim, hidden_dim, depth=3)
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, 2 + n_state_tokens + n_memory_tokens + act_seq, hidden_dim)
        )
        self.norm_final = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.ffn_final = _Mlp(hidden_dim, hidden_dim, act_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """RDT's init. ``RDTModel.finalize_modules`` calls it again, after HPT's
        xavier pass over every Linear."""
        self.apply(_xavier)
        lens = [1, 1, self.n_state_tokens, self.n_memory_tokens, self.act_seq]
        self.x_pos_embed.data.copy_(
            multimodal_sincos(self.x_pos_embed.shape[-1], [n for n in lens if n > 0])
        )
        nn.init.constant_(self.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.ffn_final.fc2.bias, 0)

    def forward(self, x, timesteps, cond: RDTConditions, *args, **kwargs):
        n_state = 0 if cond.state is None else cond.state.shape[1]
        if n_state != self.n_state_tokens:
            raise ValueError(
                f"RDTDenoiser got {n_state} state tokens but n_state_tokens="
                f"{self.n_state_tokens}; set it to (proprio stems x history_len)"
            )
        n_memory = 0 if cond.memory is None else cond.memory.shape[1]
        if n_memory != self.n_memory_tokens:
            raise ValueError(
                f"RDTDenoiser got {n_memory} memory tokens but n_memory_tokens="
                f"{self.n_memory_tokens}; set it to (memory frames x latents)"
            )
        tokens = [cond.backbone.prefix(timesteps * self.time_scale, cond.freq, len(x))]
        if n_state:
            tokens.append(cond.state)
        self_mask = None
        if n_memory:
            tokens.append(cond.memory)
            B = len(x)
            keep = torch.ones(B, 2 + n_state, dtype=torch.bool, device=x.device)
            self_mask = torch.cat(
                [keep, cond.memory_mask, keep.new_ones(B, self.act_seq)], dim=1
            )
        tokens.append(self.action_adaptor(x))
        h = torch.cat(tokens, dim=1) + self.x_pos_embed
        h = cond.backbone(h, cond, self_mask)
        return self.ffn_final(self.norm_final(h))[:, -self.act_seq :]


class _ResamplerLayer(nn.Module):
    """Latents cross-attend to one frame's patch tokens, then an FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.RMSNorm(dim, eps=1e-6)
        self.norm_kv = nn.RMSNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm_ffn = nn.RMSNorm(dim, eps=1e-6)
        self.ffn = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, latents, feats):
        latents = latents + self.attn(self.norm_q(latents), self.norm_kv(feats))
        return latents + self.ffn(self.norm_ffn(latents))


class RDTMemory(nn.Module):
    """Long-range visual memory: ``frames`` past frames of one camera, each
    compressed to ``latents`` tokens for the DiT's self-attention prefix.

    A frozen image tower (a ``DINOv3Stem`` with ``freeze_backbone``; only its
    ``proj`` trains) encodes every frame; a Perceiver resampler, shared across
    frames, pools the patch tokens into learned latents; a fixed sin-cos code
    of each frame's age in seconds (``stride_s`` apart, the newest
    ``stride_s`` back) tags its latents. ``fuse_proprio`` adds a projection of
    that step's proprio token, which needs the proprio history on the same
    grid: ``frames + 1`` steps ``stride_s`` apart, the current one last.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int,
        frames: int = 10,
        stride_s: float = 1.0,
        latents: int = 8,
        depth: int = 1,
        num_heads: int = 16,
        fuse_proprio: bool = False,
    ):
        super().__init__()
        if not getattr(encoder, "freeze_backbone", False):
            raise ValueError(
                "the memory encoder must be frozen (freeze_backbone=true): its "
                "only gradient would come through a few pooled tokens"
            )
        self.encoder = encoder
        self.frames = int(frames)
        self.latents_per_frame = int(latents)
        self.fuse_proprio = bool(fuse_proprio)
        self.latents = nn.Parameter(torch.randn(1, latents, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [_ResamplerLayer(hidden_dim, num_heads) for _ in range(depth)]
        )
        self.norm_out = nn.RMSNorm(hidden_dim, eps=1e-6)
        if self.fuse_proprio:
            self.proprio_proj = nn.Linear(hidden_dim, hidden_dim)
        grid = getattr(encoder, "grid_size", None)
        self.register_buffer(
            "patch_pos",
            sincos_2d(hidden_dim, grid) if grid is not None else None,
            persistent=False,
        )
        # ages in 0.1 s units, oldest first, so 0.1 s apart is one sin-cos step
        age_s = float(stride_s) * np.arange(self.frames, 0, -1)
        self.register_buffer(
            "age_embed",
            torch.from_numpy(sincos_1d(hidden_dim, age_s * 10.0)).float(),
            persistent=False,
        )

    @property
    def n_tokens(self) -> int:
        return self.frames * self.latents_per_frame

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, N, 3, H, W)`` frames, oldest first, and their ``(B, N)``
        validity -> ``(B, N * latents, D)`` tokens and their ``(B, N * latents)``
        bool mask."""
        B, N = images.shape[:2]
        if N != self.frames:
            raise ValueError(
                f"memory window has {N} frames but trunk.memory.frames="
                f"{self.frames}; set it to the data config's key_map.image_memory"
            )
        feats = self.encoder(images)
        feats = feats.reshape(B * N, -1, feats.shape[-1])
        if self.patch_pos is not None and self.patch_pos.shape[0] == feats.shape[1]:
            feats = feats + self.patch_pos.to(feats)
        lat = self.latents.expand(B * N, -1, -1).to(feats)
        for layer in self.layers:
            lat = layer(lat, feats)
        lat = self.norm_out(lat).reshape(B, N, self.latents_per_frame, -1)
        lat = lat + self.age_embed.to(lat)[None, :, None]
        if self.fuse_proprio:
            if state is None or state.shape[1] != N + 1:
                got = None if state is None else state.shape[1]
                raise ValueError(
                    f"fuse_proprio needs {N + 1} proprio steps on the memory's "
                    f"grid (got {got}); set key_map.proprio_history={N + 1} and "
                    "history_stride_s to the memory stride"
                )
            lat = lat + self.proprio_proj(state[:, :N]).unsqueeze(2)
        tokens = lat.reshape(B, N * self.latents_per_frame, -1)
        mask = mask.bool().repeat_interleave(self.latents_per_frame, dim=1)
        return tokens, mask
