"""
Build an ``Isotropic`` stack from a self-contained spec dict.

The vendored ``Isotropic`` constructor expects an ``HNetConfig`` indexed by a
``stage_idx`` / ``pos_idx`` pair. In the new stage-based architecture each
stage owns its own Isotropic(s) directly, so we synthesise a single-stage
``HNetConfig`` on the fly.

A spec dict looks like::

    {
      "arch_layout": "T4",       # str — passed verbatim to Isotropic
      "d_model": 128,
      "d_intermediate": 512,
      "num_heads": 4,
      "cond": true,              # whether AdaLN should be wired in this stack
      "ssm_cfg": {...},          # optional, only used for m/M arch
    }
"""

from typing import Any, Dict

from egomimic.models.hnet_nets.blocks import Isotropic
from egomimic.models.hnet_nets.config import AttnConfig, HNetConfig, SSMConfig


def build_isotropic(
    spec: Dict[str, Any],
    d_cond: int = 0,
    causal: bool = True,
) -> Isotropic:
    """Construct an Isotropic stack from a flat spec dict.

    ``d_cond`` is the algo-level conditioning width; conditioning is wired
    only when ``spec.get("cond", False)`` is True AND ``d_cond > 0``.

    ``spec.get("cond_mode", "adaln")`` selects the conditioning mechanism
    inside each TransformerBlock:
      - ``"adaln"`` (default): per-token scale/shift modulation.
      - ``"cross_attn"``: extra cross-attention layer (queries from x,
        keys/values from cond tokens).
    """
    arch_layout = spec["arch_layout"]
    d_model = int(spec["d_model"])
    d_intermediate = int(spec.get("d_intermediate", 0))
    num_heads = int(spec.get("num_heads", 8))
    cond_here = bool(spec.get("cond", False))
    cond_mode = str(spec.get("cond_mode", "adaln"))
    ssm_kwargs = spec.get("ssm_cfg", {}) or {}

    # Wrap the per-stage scalars into the list-indexed config Isotropic expects.
    cfg = HNetConfig(
        arch_layout=[arch_layout],  # 1-level nesting; pos_idx=0 selects it
        d_model=[d_model],
        d_intermediate=[d_intermediate],
        attn_cfg=AttnConfig(num_heads=[num_heads]),
        ssm_cfg=SSMConfig(**{k: [v] for k, v in ssm_kwargs.items()}),
    )
    return Isotropic(
        cfg,
        pos_idx=0,
        stage_idx=0,
        d_cond=d_cond if cond_here else 0,
        cond_here=cond_here,
        causal=causal,
        cond_mode=cond_mode,
    )
