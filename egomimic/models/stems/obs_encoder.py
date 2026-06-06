"""Per-frame observation encoder for BC-RNN.

Encodes the raw obs dict at EACH timestep of a sequence into a single
per-frame embedding vector. This is the robomimic ``ObservationGroupEncoder``
analogue: low-dim obs (agent xy) go through a small MLP, image obs go through
a conv/ResNet backbone, and the per-modality features are concatenated then
projected to ``embed_dim``.

Shape contract: every value in ``obs`` is either
  * ``(B, T, D)`` low-dim  or ``(B, T, C, H, W)`` image (sequence inputs), or
  * ``(B, D)``    low-dim  or ``(B, C, H, W)``    image (single-frame inputs,
    auto-unsqueezed to T=1).
Returns ``(B, T, embed_dim)``.

Reuses ``egomimic.models.hnet.image_encoders`` (``SimpleConv`` /
``ResNetEncoder``) for vision -- they already obey the ``(..., C, H, W) ->
(..., embed_dim)`` contract -- and a local MLP for the state, so no H-Net algo
code is imported.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _mlp(in_dim: int, out_dim: int, widths: Optional[List[int]] = None) -> nn.Sequential:
    widths = widths or []
    layers: List[nn.Module] = []
    prev = in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ObsEncoder(nn.Module):
    """Fuse per-frame low-dim + image obs into one ``embed_dim`` vector.

    Args:
        embed_dim:    output per-frame embedding width.
        obs_specs:    dict of obs_key -> {input_dim, embed_dim, widths,
                      [input_slice]}. ``input_slice`` ``[start, end]`` picks
                      ``x[..., start:end]`` before the MLP (e.g. the agent xy
                      out of a 5-dim state_agent_obj) without resaving zarr.
        img_encoders: dict of img_key -> nn.Module exposing ``.embed_dim``
                      (instantiated by Hydra from the image_encoders module).
        fuse_widths:  hidden widths for the final fusion MLP (defaults to one
                      hidden layer of ``max(embed_dim, fused_in_dim)``).
    """

    def __init__(
        self,
        embed_dim: int,
        obs_specs: Optional[Dict[str, dict]] = None,
        img_encoders: Optional[Dict[str, nn.Module]] = None,
        fuse_widths: Optional[List[int]] = None,
        paper_exact: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        # paper_exact (robomimic ObservationEncoder, v0.2 obs_nets.py:230-235):
        #   (a) low-dim state is fed RAW (no MLP embed) -- robomimic gives it no
        #       obs_net, so `self.obs_nets[k] is None` and ReLU is NOT applied;
        #   (b) the image VisualCore output gets feature_activation=nn.ReLU
        #       (applied ONLY to modalities WITH an encoder net);
        #   (c) there is NO post-concat fusion MLP -- the concatenated features
        #       feed the LSTM directly (RNN input_dim == sum of modality dims).
        # Gated + default False -> the existing (MLP-embed + fusion-MLP) path is
        # byte-identical. In paper_exact mode `embed_dim` becomes the SUM of the
        # raw low-dim slice widths + image embed_dims (no projection), so the
        # LSTM input_dim must be set to that sum (e.g. 64 + 2 = 66).
        self.paper_exact = bool(paper_exact)

        obs_specs = obs_specs or {}
        self.obs_keys = sorted(obs_specs.keys())
        self.obs_encoders = nn.ModuleDict()
        self.obs_input_slices: Dict[str, slice] = {}
        self.obs_raw_dims: Dict[str, int] = {}
        fused_dim = 0
        for key in self.obs_keys:
            spec = obs_specs[key]
            if "input_slice" in spec:
                start, end = spec["input_slice"]
                self.obs_input_slices[key] = slice(int(start), int(end))
            if self.paper_exact:
                # raw low-dim: no MLP, contributes its sliced width to the concat
                if "input_slice" in spec:
                    start, end = spec["input_slice"]
                    raw_dim = int(end) - int(start)
                else:
                    raw_dim = int(spec["input_dim"])
                self.obs_raw_dims[key] = raw_dim
                fused_dim += raw_dim
            else:
                self.obs_encoders[key] = _mlp(
                    spec["input_dim"], spec["embed_dim"], spec.get("widths", [])
                )
                fused_dim += int(spec["embed_dim"])

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
            fused_dim += int(enc.embed_dim)

        if fused_dim == 0:
            raise ValueError("ObsEncoder needs at least one obs or image key.")
        if self.paper_exact:
            # no fusion MLP; concat feeds the LSTM. embed_dim == fused_dim.
            self.fuse = nn.Identity()
            self.embed_dim = int(fused_dim)
            self._relu = nn.ReLU()
        else:
            widths = (
                fuse_widths
                if fuse_widths is not None
                else [max(self.embed_dim, fused_dim)]
            )
            self.fuse = _mlp(fused_dim, self.embed_dim, widths=list(widths))

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """obs dict -> per-frame embedding ``(B, T, embed_dim)``.

        Single-frame inputs (``(B, D)`` / ``(B, C, H, W)``) are treated as T=1.
        """
        feats: List[torch.Tensor] = []

        for key in self.obs_keys:
            if key not in obs:
                continue
            x = obs[key]
            if x.dim() == 2:  # (B, D) -> (B, 1, D)
                x = x.unsqueeze(1)
            if key in self.obs_input_slices:
                x = x[..., self.obs_input_slices[key]]
            if self.paper_exact:
                # robomimic: low-dim has no obs_net, fed RAW (no ReLU).
                feats.append(x)
            else:
                feats.append(self.obs_encoders[key](x))  # (B, T, embed)

        for key in self.img_keys:
            if key not in obs:
                continue
            x = obs[key]
            if x.dim() == 4:  # (B, C, H, W) -> (B, 1, C, H, W)
                x = x.unsqueeze(1)
            f = self.img_encoders[key](x)  # (B, T, embed)
            if self.paper_exact:
                # robomimic: feature_activation=nn.ReLU applied after the obs net
                # (image VisualCore), v0.2 obs_nets.py:233-234.
                f = self._relu(f)
            feats.append(f)

        if not feats:
            raise KeyError(
                f"ObsEncoder got no usable keys. Expected some of "
                f"{self.obs_keys + self.img_keys}, got {list(obs.keys())}."
            )
        fused = torch.cat(feats, dim=-1)  # (B, T, fused_dim)
        return self.fuse(fused)  # (B, T, embed_dim)
