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

from egomimic.models.hnet.blocks import Isotropic
from egomimic.models.hnet.config import AttnConfig, HNetConfig, SSMConfig


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
    # F6: per-stack rotary_emb_dim (0 disables RoPE for this Isotropic).
    rotary_emb_dim = int(spec.get("rotary_emb_dim", 0) or 0)
    # Per-stack dropout knobs. ``dropout`` is the attention-softmax dropout
    # (forwarded into MHA); ``resid_dropout`` is the residual-branch dropout
    # applied to attn/ffn outputs in each TransformerBlock. Both default to
    # 0.0 — yaml specs that omit them are unaffected.
    attn_dropout = float(spec.get("dropout", 0.0) or 0.0)
    resid_dropout = float(spec.get("resid_dropout", 0.0) or 0.0)
    ffn_dropout = float(spec.get("ffn_dropout", 0.0) or 0.0)
    # Optional Mixture-of-Experts FFN. ``spec["moe"]`` (dict or None) swaps this
    # stack's block MLPs for MoEFFN. None / absent => dense SwiGLU (default).
    moe_spec = spec.get("moe", None)
    if moe_spec is not None:
        moe_spec = dict(moe_spec)  # DictConfig -> plain dict for MoEFFN(**kwargs)

    # Wrap the per-stage scalars into the list-indexed config Isotropic expects.
    cfg = HNetConfig(
        arch_layout=[arch_layout],  # 1-level nesting; pos_idx=0 selects it
        d_model=[d_model],
        d_intermediate=[d_intermediate],
        attn_cfg=AttnConfig(
            num_heads=[num_heads],
            rotary_emb_dim=[rotary_emb_dim],
            dropout=[attn_dropout],
            resid_dropout=[resid_dropout],
            ffn_dropout=[ffn_dropout],
        ),
        ssm_cfg=SSMConfig(**{k: [v] for k, v in ssm_kwargs.items()}),
        moe=[moe_spec],
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
