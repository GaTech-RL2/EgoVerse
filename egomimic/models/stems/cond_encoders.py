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
        self,
        obs: Dict[str, torch.Tensor],
        T_action: int,
        embodiment_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns a cond_dict. Single-frame obs are broadcast across T_action.

        ``embodiment_id`` is accepted (and ignored) so the call site can pass
        it uniformly; ``MultiEmbodimentCondEncoder`` is the one that dispatches
        on it.
        """
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


class MultiEmbodimentCondEncoder(nn.Module):
    """Per-embodiment cond encoder dispatch.

    Holds a ``nn.ModuleDict[str, CondEncoderModule]`` (one encoder per
    embodiment name) and routes ``encode()`` calls to the matching encoder
    based on ``embodiment_id``.

    Hydra config layout:
        cond_encoder:
          _target_: egomimic.models.stems.cond_encoders.MultiEmbodimentCondEncoder
          encoders:
            pushshapes_sim:
              _target_: egomimic.models.stems.cond_encoders.CondEncoderModule
              d_cond: 128
              ...
            pushshapes_sim_stick:
              _target_: egomimic.models.stems.cond_encoders.CondEncoderModule
              d_cond: 128
              ...

    The first encoder's ``output_key`` is exposed as ``self.output_key`` so
    callers that need it can read the fused-cond key. All encoders must
    share the same output_key.
    """

    def __init__(self, encoders: Dict[str, "CondEncoderModule"]):
        super().__init__()
        if not encoders:
            raise ValueError("MultiEmbodimentCondEncoder needs at least one encoder.")
        self.encoders = nn.ModuleDict(encoders)
        first = next(iter(self.encoders.values()))
        self.output_key = first.output_key
        for emb, enc in self.encoders.items():
            if enc.output_key != self.output_key:
                raise ValueError(
                    f"All per-embodiment encoders must share output_key; "
                    f"got {enc.output_key!r} for {emb!r} vs {self.output_key!r}."
                )

    def encode(
        self,
        obs: Dict[str, torch.Tensor],
        T_action: int,
        embodiment_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if embodiment_id is None:
            raise RuntimeError(
                "MultiEmbodimentCondEncoder.encode requires embodiment_id."
            )
        if embodiment_id not in self.encoders:
            raise KeyError(
                f"No cond encoder for embodiment {embodiment_id!r}; "
                f"available: {list(self.encoders.keys())}."
            )
        return self.encoders[embodiment_id].encode(obs, T_action)
