import numpy as np
import pytest
import torch

from astra_reversal.reasoning_learning.learning import LoRALinear, NativeLearner
from astra_reversal.records import digest


def test_zero_adapter_exact_and_updates_do_not_change_base():
    torch.manual_seed(1)
    base = torch.nn.Linear(3, 2)
    layer = LoRALinear(base, rank=2)
    x = torch.randn(4, 3)
    original = base.weight.detach().clone()
    assert torch.equal(layer(x), base(x))
    optimizer = torch.optim.AdamW([layer.lora_a, layer.lora_b], lr=0.01, weight_decay=0)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = (layer(x) - 0.5).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] / 2
    assert torch.equal(original, base.weight)
    assert base.weight.grad is None


class SmallPolicy:
    """Exercise learner wiring and real updates; not a π0.5 performance test."""

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.paligemma_with_expert = torch.nn.Module()
        self.model.paligemma_with_expert.gemma_expert = torch.nn.Module()
        root = torch.nn.Module()
        root.q_proj = torch.nn.Linear(7, 7)
        self.model.paligemma_with_expert.gemma_expert.model = root
        self.model.sample_noise = lambda shape, device: torch.randn(
            shape, device=device
        )
        self.model.sample_time = lambda n, device: (
            torch.rand(n, device=device) * 0.8 + 0.1
        )
        self.policy = self.model
        self.policy.prepare_action = lambda batch: torch.tensor(
            batch["actions"], dtype=torch.float32
        )[None]
        self.metadata = {"frozen": True}
        self.horizon, self.device = 10, torch.device("cpu")

    def _preprocess(self, raw):
        return raw


def test_native_learner_uses_fresh_times_updates_weights_then_freezes(monkeypatch):
    from astra_reversal.reasoning_learning import learning

    monkeypatch.setattr(
        learning,
        "prepare_velocity",
        lambda policy, batch, differentiable: (
            lambda x, t: policy.paligemma_with_expert.gemma_expert.model.q_proj(x)
        ),
    )
    p = SmallPolicy()
    learner = NativeLearner(p, rank=2, learning_rate=0.01)
    raw = {"prompt": "lift", "observation/state": np.zeros(8)}
    oid = digest(raw)
    row = {
        "window_id": "w",
        "source": "executed_commands",
        "evidence": "observed_useful",
        "observation_id": oid,
        "episode_id": "e",
        "event_ids": ["i"],
        "stages": ["correction"],
        "actions": np.ones((10, 7)).tolist(),
    }
    receipt = learner.update([row], {oid: raw}, updates=6)
    assert receipt["after_sha256"] != receipt["before_sha256"]
    assert len({r["flow_time"] for r in receipt["history"]}) == 6
    assert not any(
        p.requires_grad or p.grad is not None for p in learner.policy.model.parameters()
    )
    assert receipt["replay_beta"] == 0
    row["source"] = "unexecuted_candidate"
    with pytest.raises(ValueError, match="real windows"):
        learner.update([row], {oid: raw}, updates=1)
