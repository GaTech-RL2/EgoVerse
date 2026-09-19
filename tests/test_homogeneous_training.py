import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from egomimic.algo.hpt import HPT, HPTModel
from egomimic.algo.pi import PI
from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.models.hpt_nets import PolicyHead
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_sampler import MultiJActionSampler, NativeActionMSELoss
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.utils.action_utils import (
    ConverterRegistry,
    HumanBimanualCartesianEuler,
    RobotBimanualCartesianEuler,
)

DOMAINS = ["eva_bimanual", "human_bimanual"]
IDS = [get_embodiment_id(domain) for domain in DOMAINS]


def _assert_gradients_equal(first, second):
    for (name, left), (_, right) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(
                left.grad, right.grad, rtol=2e-4, atol=2e-6, msg=name
            )


class _PIPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7))
        self.observations = []

    def forward(self, observation, action):
        self.observations.append(observation)
        value = observation.state[:, :1] + observation.tokenized_prompt[:, :1]
        return (self.weight * value.unsqueeze(1) - action).square()


def _pi(homogeneous=True):
    pi = object.__new__(PI)
    pi.homogeneous_training = homogeneous
    pi.domains = DOMAINS
    pi.device = torch.device("cpu")
    pi.pi_cam_keys = ["front", "wrist"]
    pi.image_resolution = (4, 4)
    pi.ac_keys = dict.fromkeys(IDS, "actions")
    pi.camera_keys = dict.fromkeys(IDS, pi.pi_cam_keys)
    pi.proprio_keys = dict.fromkeys(IDS, ["state"])
    pi.lang_keys = dict.fromkeys(IDS, [])
    pi.action_registry = ConverterRegistry()
    pi.action_registry.register(IDS[0], "actions", RobotBimanualCartesianEuler())
    pi.action_registry.register(IDS[1], "actions", HumanBimanualCartesianEuler())
    pi.nets = nn.ModuleDict({"policy": _PIPolicy()})
    return pi


def _pi_batch(second_horizon=2):
    batches = {}
    for emb_id, size, width, horizon in [
        (IDS[0], 2, 14, 2),
        (IDS[1], 5, 12, second_horizon),
    ]:
        batch = {
            "actions": torch.randn(size, horizon, width),
            "state": torch.randn(size, width),
            "front": torch.rand(size, 3, 4, 4),
            "tokenized_prompt": torch.arange(size * 3).reshape(size, 3),
            "tokenized_mask": torch.ones(size, 3, dtype=torch.bool),
            "token_ar_mask": torch.zeros(size, 3, dtype=torch.bool),
            "token_loss_mask": torch.zeros(size, 3, dtype=torch.bool),
        }
        if emb_id == IDS[0]:
            batch["wrist"] = torch.rand(size, 3, 4, 4)
        batches[emb_id] = batch
    return batches


@pytest.mark.parametrize("second_horizon, expected_calls", [(2, 1), (3, 2)])
def test_pi_batches_compatible_inputs_and_preserves_loss_and_gradients(
    second_horizon, expected_calls
):
    torch.manual_seed(1)
    batch = _pi_batch(second_horizon)
    merged, sequential = _pi(), _pi(False)
    predictions = merged.forward_training(batch)
    reference = sequential.forward_training(batch)
    assert len(merged.nets["policy"].observations) == expected_calls
    assert len(sequential.nets["policy"].observations) == 2
    for domain in DOMAINS:
        torch.testing.assert_close(
            predictions[f"{domain}_loss"], reference[f"{domain}_loss"]
        )
    actual_loss = merged.compute_losses(predictions, batch)["action_loss"]
    expected_loss = sequential.compute_losses(reference, batch)["action_loss"]
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    expected_loss.backward()
    _assert_gradients_equal(merged.nets, sequential.nets)
    if expected_calls == 1:
        observation = merged.nets["policy"].observations[0]
        assert observation.state.shape == (7, 32)
        assert observation.image_masks["wrist"].tolist() == [
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ]
        assert torch.equal(
            observation.tokenized_prompt,
            torch.cat([part["tokenized_prompt"] for part in batch.values()]),
        )


@pytest.mark.parametrize("algo_type", [PI, HPT, PipelineAlgo])
def test_loss_respects_unequal_sample_counts_and_absent_domains(algo_type):
    algo = object.__new__(algo_type)
    algo.domains = DOMAINS
    algo.ac_keys = dict.fromkeys(IDS, "actions")
    algo.resolved_ac_keys = algo.ac_keys
    algo.ot = False
    batches = {
        IDS[0]: {"actions": torch.zeros(1, 2, 3)},
        IDS[1]: {"actions": torch.zeros(3, 2, 3)},
    }
    if algo_type is PipelineAlgo:
        predictions = {
            f"{IDS[0]}_action_loss": torch.tensor(2.0),
            f"{IDS[1]}_action_loss": torch.tensor(10.0),
        }
    else:
        predictions = {
            f"{DOMAINS[0]}_loss": torch.tensor(2.0),
            f"{DOMAINS[1]}_loss": torch.tensor(10.0),
        }
    assert algo.compute_losses(predictions, batches)["action_loss"].item() == 8.0
    assert (
        algo.compute_losses(predictions, {IDS[1]: batches[IDS[1]]})[
            "action_loss"
        ].item()
        == 10.0
    )


class _Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.specs = SimpleNamespace(random_horizon_masking=False)
        self.calls = []

    def compute_latent(self, value):
        self.calls.append(value.shape[0])
        return self.proj(value.reshape(value.shape[0], -1, 4))


class _Head(PolicyHead):
    def __init__(self, dim=2):
        super().__init__()
        self.proj = nn.Linear(4, dim)
        self.calls = []

    def forward(self, value):
        self.calls.append(value.shape[0])
        return self.proj(value)


def _hpt(shared_head=True, no_trunk=False):
    model = HPTModel(
        embed_dim=4, num_blocks=2, num_heads=2, action_horizon=2, no_trunk=no_trunk
    )
    model.device = torch.device("cpu")
    for domain in DOMAINS:
        model.init_domain_stem(domain, {"state": _Stem()})
    model.init_domain_stem("shared", {"front": _Stem()})
    model.shared_keys = ["front"]
    model.shared_action = shared_head
    model.auxiliary_ac_keys = {}
    if shared_head:
        model.init_domain_head("shared", _Head())
    else:
        for domain in DOMAINS:
            model.init_domain_head(domain, _Head())
    model.finalize_modules()
    return model


def _hpt_batches(extra_tokens=False):
    return {
        domain: {
            "domain": domain,
            "data": {
                "state": torch.randn(size, 1, 4),
                "front": torch.randn(
                    size, 1, 2 if extra_tokens and size == 3 else 1, 4
                ),
                "action": torch.randn(size, 2, 2),
            },
        }
        for domain, size in zip(DOMAINS, [2, 3])
    }


@pytest.mark.parametrize(
    "shared_head, depth, extra_tokens",
    [(True, None, False), (False, None, False), (True, 1, False), (True, None, True)],
)
def test_hpt_batches_shared_modules_and_preserves_gradients(
    shared_head, depth, extra_tokens
):
    torch.manual_seed(2)
    merged = _hpt(shared_head)
    sequential = copy.deepcopy(merged)
    batches = _hpt_batches(extra_tokens)
    trunk_calls = []
    merged.trunk["trunk"].register_forward_pre_hook(
        lambda module, args: trunk_calls.append(args[0].shape[0])
    )
    actual = merged.compute_loss_multi(copy.deepcopy(batches), depth=depth)
    expected = {
        domain: sequential.compute_loss(batch)
        if depth is None
        else sequential.compute_loss_depth(batch, depth)
        for domain, batch in copy.deepcopy(batches).items()
    }
    assert trunk_calls == ([2, 3] if extra_tokens else [5])
    assert merged.stems["shared_front"].calls == ([2, 3] if extra_tokens else [5])
    assert merged.stems[f"{DOMAINS[0]}_state"].calls == [2]
    assert merged.stems[f"{DOMAINS[1]}_state"].calls == [3]
    if shared_head:
        assert merged.heads["shared"].calls == [5]
    else:
        assert merged.heads[DOMAINS[0]].calls == [2]
        assert merged.heads[DOMAINS[1]].calls == [3]
    for domain in DOMAINS:
        torch.testing.assert_close(
            actual[domain], expected[domain], rtol=2e-5, atol=1e-6
        )
    sum(actual.values()).backward()
    sum(expected.values()).backward()
    _assert_gradients_equal(merged, sequential)


def test_hpt_without_trunk_backpropagates():
    model = _hpt(no_trunk=True)
    losses = model.compute_loss_multi(_hpt_batches())
    sum(losses.values()).backward()
    assert model.heads["shared"].proj.weight.grad is not None


class _DenoisingHead(DenoisingPolicy):
    def __init__(self):
        super().__init__(
            nn.Linear(4, 14),
            action_horizon=2,
            infer_ac_dims=dict.fromkeys(DOMAINS, 14),
            padding="zero",
        )
        self.calls = []

    def predict(self, actions, global_cond):
        self.calls.append(actions.shape[0])
        return self.model(global_cond), actions


def test_hpt_shared_denoising_head_batches_after_action_padding():
    merged = _hpt()
    merged.heads["shared"] = _DenoisingHead()
    sequential = copy.deepcopy(merged)
    batches = _hpt_batches()
    batches[DOMAINS[0]]["data"]["action"] = torch.randn(2, 2, 14)
    batches[DOMAINS[1]]["data"]["action"] = torch.randn(3, 2, 12)
    actual = merged.compute_loss_multi(copy.deepcopy(batches))
    expected = {
        domain: sequential.compute_loss(batch) for domain, batch in batches.items()
    }
    assert merged.heads["shared"].calls == [5]
    assert sequential.heads["shared"].calls == [2, 3]
    for domain in DOMAINS:
        torch.testing.assert_close(actual[domain], expected[domain])
    sum(actual.values()).backward()
    sum(expected.values()).backward()
    _assert_gradients_equal(merged, sequential)


class _Velocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.calls = []

    def forward(self, latent, time, condition):
        self.calls.append(latent.shape[0])
        return self.proj(condition).expand_as(latent) + latent * 0.1


def _sampler():
    return MultiJActionSampler(
        denoising_module=_Velocity(),
        condition_input_dim=4,
        condition_dim=4,
        action_horizon=2,
        action_dims=dict(zip(DOMAINS, [2, 6])),
        latent_dim=3,
        decoder_hidden_dim=8,
        denoiser_hidden_dim=8,
        num_inference_steps=2,
        sampling_schedule={1: {1: 1.0}},
        gradient_checkpointing=False,
    )


def test_pipeline_shared_denoiser_batches_different_action_spaces():
    torch.manual_seed(3)
    merged = _sampler().train()
    sequential = copy.deepcopy(merged)
    batches = {
        domain: {
            "embodiment": domain,
            "sampler/noise": torch.randn(size, 2, 3),
            "condition": torch.randn(size, 4),
            "target": torch.randn(size, 2, dim),
        }
        for domain, size, dim in zip(DOMAINS, [2, 3], [2, 6])
    }
    actual = Pipeline([merged, NativeActionMSELoss()]).forward_batches(
        copy.deepcopy(batches)
    )
    reference_pipeline = Pipeline([sequential, NativeActionMSELoss()])
    expected = {
        domain: reference_pipeline(batch)
        for domain, batch in copy.deepcopy(batches).items()
    }
    assert merged.denoising_module.calls == [5]
    assert sequential.denoising_module.calls == [2, 3]
    for domain in DOMAINS:
        torch.testing.assert_close(
            actual[domain]["pred_action"], expected[domain]["pred_action"]
        )
    sum(batch["loss/native_action"] for batch in actual.values()).backward()
    sum(batch["loss/native_action"] for batch in expected.values()).backward()
    _assert_gradients_equal(merged, sequential)
    assert merged.training_batches_seen.item() == 1


def test_pipeline_schedule_advances_without_anchor_domain():
    sampler = _sampler().train()
    for step in (1, 2):
        result = sampler.forward_batches(
            {
                "human": {
                    "embodiment": "human_bimanual",
                    "sampler/noise": torch.randn(2, 2, 3),
                    "condition": torch.randn(2, 4),
                }
            }
        )
        assert result["human"]["log/optimizer_step"] == step
