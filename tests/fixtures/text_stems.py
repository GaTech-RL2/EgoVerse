"""Shared bits of the Qwen text-stem tests: cross-attention specs for a tiny
stem and the locally cached Qwen3-Embedding-0.6B snapshot (tests needing it
skip without it)."""

from __future__ import annotations

import glob
import os

import pytest
from omegaconf import OmegaConf

EMBED_DIM = 32


def cross_attn_specs(latent: int = 4) -> OmegaConf:
    return OmegaConf.create(
        {
            "random_horizon_masking": False,
            "cross_attn": {
                "crossattn_latent": latent,
                "crossattn_heads": 2,
                "crossattn_dim_head": 8,
                "crossattn_modality_dropout": 0.0,
                "modality_embed_dim": EMBED_DIM,
            },
        }
    )


def _qwen_snapshot() -> str | None:
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    hits = sorted(
        glob.glob(
            os.path.join(cache, "models--Qwen--Qwen3-Embedding-0.6B", "snapshots", "*")
        )
    )
    return hits[-1] if hits else None


QWEN_SNAPSHOT = _qwen_snapshot()
requires_qwen = pytest.mark.skipif(
    QWEN_SNAPSHOT is None, reason="Qwen3-Embedding-0.6B snapshot not in the HF cache"
)
