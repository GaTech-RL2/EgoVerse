"""The `transformers` pin and openpi's `transformers_replace` patch move together.

`external/openpi/src/openpi/models_pytorch/transformers_replace/` holds verbatim
copies of specific `transformers` sources, so the installed version must be the
pinned one, and the copies must actually be installed over the library. Both are
easy to break silently: `uv sync` reinstalls `transformers` and undoes the `cp`.
"""

import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _pinned_version() -> str:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"transformers==([^"]+)"', text)
    assert match is not None, "no `transformers==` pin in pyproject.toml"
    return match.group(1)


def test_installed_transformers_matches_pyproject_pin():
    import transformers

    assert transformers.__version__ == _pinned_version()


def test_qwen3_5_is_available():
    """The reason for the upgrade: the Qwen 3.5 architecture (5.2.0+)."""
    module = importlib.import_module("transformers.models.qwen3_5")
    assert module is not None
    importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")


def test_openpi_patch_is_installed():
    """The AdaRMS / tuple-returning norm that openpi's pi0.5 code relies on."""
    pytest.importorskip(
        "openpi", reason="requires the openpi checkout (external/openpi)"
    )
    import torch
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    out = GemmaRMSNorm(8)(torch.zeros(1, 2, 8))
    assert isinstance(out, tuple) and len(out) == 2, (
        "GemmaRMSNorm must return (hidden_states, gate); "
        "the transformers_replace patch is not installed"
    )
    assert out[1] is None


def test_openpi_patch_config_fields():
    pytest.importorskip(
        "openpi", reason="requires the openpi checkout (external/openpi)"
    )
    from transformers.models.gemma.configuration_gemma import GemmaConfig

    plain = GemmaConfig()
    assert plain.use_adarms is False
    assert plain.adarms_cond_dim is None

    adaptive = GemmaConfig(use_adarms=True)
    assert adaptive.adarms_cond_dim == adaptive.hidden_size
    assert GemmaConfig(use_adarms=True, adarms_cond_dim=16).adarms_cond_dim == 16


def test_openpi_version_check_agrees_with_the_pin():
    pytest.importorskip(
        "openpi", reason="requires the openpi checkout (external/openpi)"
    )
    from transformers.models.siglip.check import (
        check_whether_transformers_replace_is_installed_correctly,
    )

    assert check_whether_transformers_replace_is_installed_correctly()


def test_gemma_embedding_normalizer_is_disabled():
    """pi0.5 does its own embedding scaling.

    4.53 applied `hidden_size ** 0.5` in `GemmaModel.forward` and the patch
    commented it out; 5.x moved the same scale into `GemmaTextScaledWordEmbedding`,
    so the patch neutralises it there instead (`embed_scale=1.0`).
    """
    pytest.importorskip(
        "openpi", reason="requires the openpi checkout (external/openpi)"
    )
    import torch
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaModel

    config = GemmaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=11,
    )
    model = GemmaModel(config)
    ids = torch.tensor([[1, 2, 3]])
    embedded = model.embed_tokens(ids)
    raw = model.embed_tokens.weight[ids]
    torch.testing.assert_close(embedded, raw)
