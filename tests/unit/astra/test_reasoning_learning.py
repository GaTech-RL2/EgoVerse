import numpy as np
import pytest
import torch

from astra_reversal.reasoning_learning.learning import (
    LoRALinear,
    NativeLearner,
    flow_matching_loss,
)
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


def test_native_action_loss_ignores_errors_in_padding_channels():
    prediction = torch.ones(1, 10, 32, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[..., 7:] = 100
    loss = flow_matching_loss(prediction, target, 7)
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad[..., :7].abs().min() > 0
    assert prediction.grad[..., 7:].count_nonzero() == 0
    assert (
        flow_matching_loss(prediction, target).item() > 1000
    )  # Recorded legacy objective differs.


def test_evidence_mask_loss_ignores_unsupported_steps_and_preserves_full_loss():
    prediction = torch.ones(1, 10, 32, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 5:, :] = 100
    mask = [[True] * 5 + [False] * 5]
    loss = flow_matching_loss(prediction, target, 7, step_mask=mask)
    assert loss.item() == 1
    loss.backward()
    assert prediction.grad[:, :5, :7].abs().min() > 0
    assert prediction.grad[:, 5:, :].count_nonzero() == 0
    assert prediction.grad[:, :, 7:].count_nonzero() == 0
    torch.testing.assert_close(
        flow_matching_loss(prediction, target, 7, step_mask=[[True] * 10]),
        flow_matching_loss(prediction, target, 7),
        rtol=0,
        atol=0,
    )
    for invalid in ([[False] * 10], [[True] * 5], [[float("nan")] * 10], [[0.5] * 10]):
        with pytest.raises(ValueError, match="evidence mask"):
            flow_matching_loss(prediction, target, 7, step_mask=invalid)


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


@pytest.mark.parametrize(
    "masked, success_bc", [(False, False), (True, False), (False, True)]
)
def test_native_learner_uses_fresh_times_updates_weights_then_freezes(
    monkeypatch, masked, success_bc
):
    from astra_reversal.reasoning_learning import learning

    monkeypatch.setattr(
        learning,
        "prepare_velocity",
        lambda policy, batch, differentiable: (
            lambda x, t: policy.paligemma_with_expert.gemma_expert.model.q_proj(x)
        ),
    )
    p = SmallPolicy()
    learner = NativeLearner(p, rank=2, learning_rate=0.01, loss_action_dimensions=7)
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
    if masked:
        row.update(
            evidence="observed_useful_masked", step_loss_mask=[True] * 5 + [False] * 5
        )
    if success_bc:
        row.update(evidence="successful_episode", stages=["successful_episode"])
        with pytest.raises(ValueError, match="configured evidence"):
            learner.update([row], {oid: raw}, updates=6)
    receipt = learner.update(
        [row],
        {oid: raw},
        updates=6,
        sampling="uniform" if success_bc else "event_stage_balanced",
        allow_successful_episodes=success_bc,
    )
    assert receipt["after_sha256"] != receipt["before_sha256"]
    assert len({r["flow_time"] for r in receipt["history"]}) == 6
    assert not any(
        p.requires_grad or p.grad is not None for p in learner.policy.model.parameters()
    )
    assert receipt["replay_beta"] == 0
    assert receipt["evidence_masked_loss"] is masked
    assert receipt["successful_episode_evidence_enabled"] is success_bc
    assert receipt["window_sampling"] == (
        "uniform" if success_bc else "event_stage_balanced"
    )
    assert all(
        r["supervised_steps"] == (5 if masked else 10) for r in receipt["history"]
    )
    row["source"] = "unexecuted_candidate"
    with pytest.raises(ValueError, match="real windows"):
        learner.update([row], {oid: raw}, updates=1)
