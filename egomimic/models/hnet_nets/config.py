from dataclasses import dataclass, field
from typing import Any, List, Union


@dataclass
class AttnConfig:
    num_heads: List[int] = field(default_factory=list)
    rotary_emb_dim: List[int] = field(default_factory=list)
    window_size: List[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    """Per-stage Mamba2 kwargs. Each list is indexed by stage_idx."""

    d_conv: List[int] = field(default_factory=list)
    expand: List[int] = field(default_factory=list)
    d_state: List[int] = field(default_factory=list)
    chunk_size: List[int] = field(default_factory=list)
    headdim: List[int] = field(default_factory=list)


@dataclass
class HNetConfig:
    """
    Flexible config for the vendored HNet.

    arch_layout: nested list describing the hierarchy. At each level either
        [encoder_str, sub_layout, decoder_str] (3 elements) or [innermost_str] (1).
        Strings use HNet's notation but here we only support 't'/'T' (transformer
        without/with MLP). Mamba ('m'/'M') is not vendored.
    d_model:        per-stage list of hidden dims
    d_intermediate: per-stage list of FFN intermediate dims (0 = no MLP within block)
    attn_cfg:       dict-like (num_heads / window_size lists, indexed per stage)
    target_compression_ratios: per non-innermost stage, used by the ratio loss
    """

    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    d_intermediate: List[int] = field(default_factory=list)
    attn_cfg: Any = field(default_factory=AttnConfig)
    ssm_cfg: Any = field(default_factory=SSMConfig)
    target_compression_ratios: List[float] = field(default_factory=list)
    # AdaLN conditioning placement. Each list holds the stage indices that
    # should receive cond at that position. Defaults: outer encoder only.
    cond_encoder_stages: List[int] = field(default_factory=lambda: [0])
    cond_decoder_stages: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, cfg: dict) -> "HNetConfig":
        attn_cfg = cfg.get("attn_cfg", {})
        if isinstance(attn_cfg, dict):
            attn_cfg = AttnConfig(**attn_cfg)
        ssm_cfg = cfg.get("ssm_cfg", {})
        if isinstance(ssm_cfg, dict):
            ssm_cfg = SSMConfig(**ssm_cfg)
        return cls(
            arch_layout=cfg["arch_layout"],
            d_model=cfg["d_model"],
            d_intermediate=cfg.get("d_intermediate", [0] * len(cfg["d_model"])),
            attn_cfg=attn_cfg,
            ssm_cfg=ssm_cfg,
            target_compression_ratios=cfg.get(
                "target_compression_ratios",
                [2.0] * max(0, len(cfg["d_model"]) - 1),
            ),
            cond_encoder_stages=list(cfg.get("cond_encoder_stages", [0])),
            cond_decoder_stages=list(cfg.get("cond_decoder_stages", [])),
        )


def get_stage_cfg(cfg: AttnConfig, stage_idx: int) -> dict:
    """Slice the per-stage AttnConfig into a flat dict for a given stage."""
    out = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, list) and len(v) > stage_idx:
            out[k] = v[stage_idx]
    return out
