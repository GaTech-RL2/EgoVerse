"""
CondEncoderModule: build the per-token conditioning tensor for the H-Net
from a raw obs dict.

Replaces the inline obs_encoders / img_encoders / cond_proj fusion that
previously lived on ``HNetPolicy``. The algo now holds a single
``CondEncoderModule`` instance and calls ``encode(obs, T)`` to produce a
``cond_dict`` it can hand to ``HNetContext``.

Output:
    cond_dict[output_key] = (B, T, d_cond)  — fused per-token cond.
    optionally per-obs raw embeddings keyed by their obs name when
    ``per_obs_keys=True`` (useful if a stage wants un-fused cond).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _mlp(
    in_dim: int, out_dim: int, widths: Optional[List[int]] = None
) -> nn.Sequential:
    widths = widths or []
    layers: List[nn.Module] = []
    prev = in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.GELU()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CondEncoderModule(nn.Module):
    """
    Args:
        obs_specs:        dict of obs_key -> {input_dim, embed_dim, widths}.
        img_encoders:     dict of img_key -> nn.Module with .embed_dim.
        d_cond:           output cond width per token.
        cond_proj_widths: hidden widths for the fusion MLP. If None, defaults
                          to ``[max(d_cond, fused_in_dim)]``.
        output_key:       key under which the fused cond is exposed.
        per_obs_keys:     when True, also expose each obs's individual embedding
                          under its own obs_key in the output dict (useful for
                          stages that want un-fused cond).
    """

    def __init__(
        self,
        d_cond: int,
        obs_specs: Optional[Dict[str, dict]] = None,
        img_encoders: Optional[Dict[str, nn.Module]] = None,
        cond_proj_widths: Optional[List[int]] = None,
        output_key: str = "fused_cond",
        per_obs_keys: bool = False,
    ):
        super().__init__()
        self.d_cond = int(d_cond)
        self.output_key = output_key
        self.per_obs_keys = per_obs_keys

        obs_specs = obs_specs or {}
        self.obs_keys = sorted(obs_specs.keys())
        self.obs_encoders = nn.ModuleDict()
        # Optional per-key input slice: spec["input_slice"] = [start, end] picks
        # x[..., start:end] before the MLP. Lets a multi-component proprio
        # tensor feed only a subset into the encoder without resaving the zarr.
        self.obs_input_slices: Dict[str, slice] = {}
        fused_dim = 0
        for key in self.obs_keys:
            spec = obs_specs[key]
            self.obs_encoders[key] = _mlp(
                spec["input_dim"], spec["embed_dim"], spec.get("widths", [])
            )
            if "input_slice" in spec:
                start, end = spec["input_slice"]
                self.obs_input_slices[key] = slice(int(start), int(end))
            fused_dim += spec["embed_dim"]

        img_encoders = img_encoders or {}
        self.img_keys = sorted(img_encoders.keys())
        self.img_encoders = nn.ModuleDict({k: img_encoders[k] for k in self.img_keys})
        for key in self.img_keys:
            enc = self.img_encoders[key]
            if not hasattr(enc, "embed_dim"):
                raise AttributeError(
                    f"img_encoders['{key}'] must expose `.embed_dim` (got "
                    f"{type(enc).__name__})."
                )
            fused_dim += enc.embed_dim

        if fused_dim == 0:
            self.cond_proj = None
        else:
            widths = cond_proj_widths
            if widths is None:
                widths = [max(self.d_cond, fused_dim)]
            self.cond_proj = _mlp(fused_dim, self.d_cond, widths=list(widths))

    def encode(
        self, obs: Dict[str, torch.Tensor], T_action: int
    ) -> Dict[str, torch.Tensor]:
        """Returns a cond_dict. Single-frame obs are broadcast across T_action."""
        if self.cond_proj is None:
            return {}

        out: Dict[str, torch.Tensor] = {}
        feats: List[torch.Tensor] = []

        for key in self.obs_keys:
            if key not in obs:
                continue
            x = obs[key]
            if key in self.obs_input_slices:
                x = x[..., self.obs_input_slices[key]]
            if x.dim() == 2:  # (B, D) -> (B, T, D)
                x = x.unsqueeze(1).expand(-1, T_action, -1)
            emb = self.obs_encoders[key](x)  # (B, T, embed_dim)
            feats.append(emb)
            if self.per_obs_keys:
                out[key] = emb

        for key in self.img_keys:
            if key not in obs:
                continue
            x = obs[key]
            if x.dim() == 4:  # (B, C, H, W) -> (B, T, ...)
                x = x.unsqueeze(1).expand(-1, T_action, -1, -1, -1)
            emb = self.img_encoders[key](x)  # (B, T, embed_dim)
            feats.append(emb)
            if self.per_obs_keys:
                out[key] = emb

        if not feats:
            return out
        fused = torch.cat(feats, dim=-1)
        out[self.output_key] = self.cond_proj(fused)
        return out


class _LatentCrossAttn(nn.Module):
    """HPT-style latent compressor: ``n_latents`` learnable queries cross-attend
    over a variable-length set of spatial tokens to produce a fixed small set
    of cond tokens (localizes the T / goal like HPT's ``compute_latent``)."""

    def __init__(self, d_cond: int, n_latents: int, num_heads: int = 4):
        super().__init__()
        assert d_cond % num_heads == 0
        self.d_cond = int(d_cond)
        self.n_latents = int(n_latents)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_cond // self.num_heads
        self.latents = nn.Parameter(torch.zeros(1, self.n_latents, self.d_cond))
        nn.init.normal_(self.latents, std=0.02)
        self.q_proj = nn.Linear(self.d_cond, self.d_cond)
        self.kv_proj = nn.Linear(self.d_cond, 2 * self.d_cond)
        self.out_proj = nn.Linear(self.d_cond, self.d_cond)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (BT, M, d_cond) -> (BT, n_latents, d_cond).
        import torch.nn.functional as F

        BT = tokens.shape[0]
        q = self.q_proj(self.latents.expand(BT, -1, -1))
        q = q.reshape(BT, self.n_latents, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(tokens).reshape(
            BT, tokens.shape[1], 2, self.num_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(BT, self.n_latents, self.d_cond)
        return self.out_proj(o)


class SpatialCondEncoderModule(CondEncoderModule):
    """CondEncoderModule that ALSO emits per-frame SPATIAL cond tokens for the
    cross-attention (run D) conditioning path.

    Keeps the parent's pooled ``fused_cond`` (B, T, d_cond) UNTOUCHED -- the
    input-layer cond token / AdaLN path is byte-identical. ADDS:

        cond_dict["spatial_cond_tokens"] = (B, T, n_spatial, d_cond)

    built from a SEPARATE set of image encoders configured with
    ``return_tokens=True`` (so the image stays ~M spatial tokens instead of a
    globally pooled vector -> the backbone can cross-attend to LOCALIZE the T /
    goal, HPT-style). Optionally compresses M -> ``compress_tokens`` latents via
    a small latent cross-attention (HPT ``crossattn_latent``).

    Args (in addition to the parent's):
        spatial_img_encoders: dict img_key -> nn.Module emitting (..., M, E)
                              spatial tokens (set ``return_tokens=true`` in the
                              encoder yaml). Its per-token width E is projected
                              to ``d_cond`` by a per-key linear.
        spatial_output_key:   dict key for the spatial tokens.
        compress_tokens:      if set, compress each frame's tokens to this many
                              via latent cross-attention (default None -> raw M).
        compress_heads:       heads for the latent cross-attention.
    """

    def __init__(
        self,
        d_cond: int,
        obs_specs=None,
        img_encoders=None,
        cond_proj_widths=None,
        output_key: str = "fused_cond",
        per_obs_keys: bool = False,
        spatial_img_encoders=None,
        spatial_output_key: str = "spatial_cond_tokens",
        compress_tokens=None,
        compress_heads: int = 4,
    ):
        super().__init__(
            d_cond=d_cond,
            obs_specs=obs_specs,
            img_encoders=img_encoders,
            cond_proj_widths=cond_proj_widths,
            output_key=output_key,
            per_obs_keys=per_obs_keys,
        )
        self.spatial_output_key = spatial_output_key
        spatial_img_encoders = spatial_img_encoders or {}
        self.spatial_img_keys = sorted(spatial_img_encoders.keys())
        self.spatial_img_encoders = nn.ModuleDict(
            {k: spatial_img_encoders[k] for k in self.spatial_img_keys}
        )
        # Per-key projection token-width -> d_cond (skip if already d_cond).
        self.spatial_proj = nn.ModuleDict()
        # Learnable SPATIAL positional embedding per camera, sized to that
        # encoder's token count M. WITHOUT this the cross-attention over the
        # spatial tokens is permutation-invariant -> it CANNOT tell "T on the
        # left" from "T on the right" (a translated patch is the same token SET)
        # and so cannot localize -- the whole point of run D. (HPT adds the same
        # spatial pos emb to its tokens before cross-attn.)
        self.spatial_pos_emb = nn.ParameterDict()
        for k in self.spatial_img_keys:
            enc = self.spatial_img_encoders[k]
            ed = int(getattr(enc, "embed_dim", self.d_cond))
            self.spatial_proj[k] = (
                nn.Identity() if ed == self.d_cond else nn.Linear(ed, self.d_cond)
            )
            n_tok = int(getattr(enc, "n_tokens", 0) or 0)
            if n_tok > 0:
                pe = nn.Parameter(torch.zeros(1, n_tok, self.d_cond))
                nn.init.normal_(pe, std=0.02)
                self.spatial_pos_emb[k] = pe
        self.compress_tokens = int(compress_tokens) if compress_tokens else 0
        if self.compress_tokens > 0:
            self.compressor = _LatentCrossAttn(
                self.d_cond, self.compress_tokens, num_heads=int(compress_heads)
            )
        else:
            self.compressor = None

    def encode(self, obs, T_action: int):
        out = super().encode(obs, T_action)  # fused_cond untouched
        if not self.spatial_img_keys:
            return out
        tok_per_key = []
        B = T = None
        for key in self.spatial_img_keys:
            if key not in obs:
                continue
            x = obs[key]
            if x.dim() == 4:  # (B, C, H, W) -> (B, T, C, H, W)
                x = x.unsqueeze(1).expand(-1, T_action, -1, -1, -1)
            # encoder returns (B, T, M, E) spatial tokens.
            tok = self.spatial_img_encoders[key](x)
            tok = self.spatial_proj[key](tok)  # (B, T, M, d_cond)
            B, T = tok.shape[0], tok.shape[1]
            M = tok.shape[2]
            # ADD the spatial positional embedding so cross-attn can localize
            # (broadcast over B, T). (1, M, d_cond) -> (1, 1, M, d_cond).
            if key in self.spatial_pos_emb:
                tok = tok + self.spatial_pos_emb[key].unsqueeze(1).to(tok.dtype)
            if self.compressor is not None:
                tok = self.compressor(tok.reshape(B * T, M, self.d_cond))
                tok = tok.reshape(B, T, self.compress_tokens, self.d_cond)
            tok_per_key.append(tok)
        if tok_per_key:
            # Concatenate multiple cameras along the token axis.
            out[self.spatial_output_key] = torch.cat(tok_per_key, dim=2)
        return out

