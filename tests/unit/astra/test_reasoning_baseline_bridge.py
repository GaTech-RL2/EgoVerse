from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from astra_reversal.reasoning_learning.rlinf_bridge import difference, match_native_gelu


def test_parity_rejects_small_systematic_drift_and_nonfinite_output():
    reference = torch.ones(2, 3)
    assert difference(reference, reference)["passed"]
    assert not difference(reference + 0.001, reference)["passed"]
    assert not difference(reference * float("nan"), reference)["passed"]


def test_activation_probe_is_explicit_and_restores_callables_after_failure():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Identity()
            self.fc2 = nn.Identity()
            self.dropout = nn.Identity()

        def forward(self, x):
            return F.gelu(x)

    core = SimpleNamespace(gemma=SimpleNamespace(gelu_glu=lambda g, v: F.gelu(g) * v))
    mlp = MLP()
    model = SimpleNamespace(
        img=SimpleNamespace(encoder=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
    )
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(hidden_activation="gelu_pytorch_tanh"),
        vision_config=SimpleNamespace(hidden_act="gelu_pytorch_tanh"),
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(config=cfg))
        )
    )
    original = core.gemma.gelu_glu
    x = torch.tensor([1.5])
    before = mlp(x)
    with pytest.raises(RuntimeError, match="simulated"):
        with match_native_gelu(core, model, policy):
            assert torch.equal(mlp(x), F.gelu(x, approximate="tanh"))
            assert not torch.equal(mlp(x), before)
            raise RuntimeError("simulated interrupted probe")
    assert core.gemma.gelu_glu is original
    assert torch.equal(mlp(x), before)
