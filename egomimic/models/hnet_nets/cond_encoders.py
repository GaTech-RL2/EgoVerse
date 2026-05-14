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


def _mlp(in_dim: int, out_dim: int, widths: Optional[List[int]] = None) -> nn.Sequential:
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
        fused_dim = 0
        for key in self.obs_keys:
            spec = obs_specs[key]
            self.obs_encoders[key] = _mlp(
                spec["input_dim"], spec["embed_dim"], spec.get("widths", [])
            )
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

    def encode(self, obs: Dict[str, torch.Tensor], T_action: int) -> Dict[str, torch.Tensor]:
        """Returns a cond_dict. Single-frame obs are broadcast across T_action."""
        if self.cond_proj is None:
            return {}

        out: Dict[str, torch.Tensor] = {}
        feats: List[torch.Tensor] = []

        for key in self.obs_keys:
            if key not in obs:
                continue
            x = obs[key]
            if x.dim() == 2:                       # (B, D) -> (B, T, D)
                x = x.unsqueeze(1).expand(-1, T_action, -1)
            emb = self.obs_encoders[key](x)         # (B, T, embed_dim)
            feats.append(emb)
            if self.per_obs_keys:
                out[key] = emb

        for key in self.img_keys:
            if key not in obs:
                continue
            x = obs[key]
            if x.dim() == 4:                       # (B, C, H, W) -> (B, T, ...)
                x = x.unsqueeze(1).expand(-1, T_action, -1, -1, -1)
            emb = self.img_encoders[key](x)         # (B, T, embed_dim)
            feats.append(emb)
            if self.per_obs_keys:
                out[key] = emb

        if not feats:
            return out
        fused = torch.cat(feats, dim=-1)
        out[self.output_key] = self.cond_proj(fused)
        return out
