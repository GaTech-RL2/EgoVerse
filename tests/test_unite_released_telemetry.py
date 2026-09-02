"""Joint-loss and topology telemetry contracts for the UNITE register sweep."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from egomimic.pipeline.stages_sampler import TokenwiseMLPActionDecoder
from egomimic.pipeline.stages_unite_released import (
    ReleasedRecipeUniteLatentPolicy,
    ReleasedRecipeUniteObjective,
)
from egomimic.pipeline.stages_unite_separate import (
    build_configurable_unite_generative_encoder,
)
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_model_unite_released import ReleasedUniteModelWrapper
from egomimic.utils.unite_optim import (
    ReleasedUniteCompositeOptimizer,
    released_unite_two_stage_scheduler,
)

DOMAIN = "pushshapes_sim_u_socket"
SHARED_KEYS = {
    "log/unite_gradient_cosine",
    "log/unite_recon_grad_norm",
    "log/unite_denoise_grad_norm",
    "log/unite_gradient_parameter_count",
    "log/unite_gradient_tensor_count",
}
SEPARATE_KEYS = {
    "log/unite_tokenizer_recon_grad_norm",
    "log/unite_denoiser_flow_grad_norm",
}


class _TelemetryHarness(ReleasedUniteModelWrapper):
    def __init__(self, topology: str, num_latent_tokens: int):
        torch.nn.Module.__init__(self)
        self.topology = topology
        self.logged = {}
        shape = (num_latent_tokens, 16)
        if topology == "shared":
            self.shared_weight = torch.nn.Parameter(torch.randn(shape))
        elif topology == "separate":
            self.tokenizer_weight = torch.nn.Parameter(torch.randn(shape))
            self.denoiser_weight = torch.nn.Parameter(torch.randn(shape))
        else:
            raise ValueError(topology)

    @property
    def device(self):
        return next(self.parameters()).device

    def _log_train_metric(self, name, value, **_kwargs):
        self.logged[name] = torch.as_tensor(value).detach().float()

    def _unite_shared_parameters(self):
        return ("shared_weight",), (self.shared_weight,)

    def _unite_separate_parameters(self):
        return (
            (("tokenizer_weight", self.tokenizer_weight),),
            (("denoiser_weight", self.denoiser_weight),),
        )


def _real_policy(shared: bool):
    encoder = build_configurable_unite_generative_encoder(
        backbone_config={
            "_target_": "egomimic.models.unite_dit.UniteDiTBackbone",
            "input_dim": 16,
            "output_dim": 16,
            "horizon": 4,
            "condition_dim": 12,
            "max_condition_tokens": 1,
            "max_content_tokens": 4,
            "hidden_dim": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "time_fourier_dim": 16,
            "context_adaln_summary": False,
            "in_context_start": 1,
            "in_context_len": 3,
            "gradient_checkpointing": False,
        },
        share_encoder_denoiser=shared,
        action_dims={DOMAIN: 4},
        condition_input_dim=14,
        latent_dim=16,
        num_latent_tokens=4,
        condition_dim=12,
        denoiser_hidden_dim=32,
        gradient_checkpointing=False,
        in_context_start=1,
        in_context_len=3,
    )
    return ReleasedRecipeUniteLatentPolicy(
        generative_encoder=encoder,
        decoders={DOMAIN: TokenwiseMLPActionDecoder(16, 16, 4)},
        num_inference_steps=2,
        flow_steps_per_reconstruction=2,
        flow_mini_batch=1,
        timestep_shift_alpha=0.5,
    ).train()


class _RealTelemetryHarness(ReleasedUniteModelWrapper):
    def __init__(self, policy, shared: bool):
        torch.nn.Module.__init__(self)
        self.policy_ref = policy
        self.model = SimpleNamespace(
            policy=SimpleNamespace(stages=[policy]),
            domains=[DOMAIN],
        )
        self.share_encoder_denoiser = shared
        self.logged = {}

    @property
    def device(self):
        return next(self.policy_ref.parameters()).device

    def _log_train_metric(self, name, value, **_kwargs):
        self.logged[name] = torch.as_tensor(value).detach().float()


def _real_component_losses(policy):
    result = ReleasedRecipeUniteObjective()(
        policy(
            {
                "target": torch.randn(2, 4, 4),
                "condition": torch.randn(2, 14),
                "sampler/noise": torch.randn(2, 4, 16),
                "embodiment": DOMAIN,
            }
        )
    )
    return result["loss/unite_reconstruction"], result["loss/unite_latent"]


def _shared_component_gradient_norms(policy, reconstruction, flow):
    named = policy.shared_reconstruction_denoising_named_parameters([DOMAIN])
    parameters = tuple(parameter for _, parameter in named)
    reconstruction_gradients = torch.autograd.grad(
        reconstruction,
        parameters,
        retain_graph=True,
        allow_unused=False,
    )
    flow_gradients = torch.autograd.grad(
        flow,
        parameters,
        retain_graph=True,
        allow_unused=False,
    )

    def norm(gradients):
        return torch.stack(
            tuple(gradient.float().square().sum() for gradient in gradients)
        ).sum().sqrt()

    return norm(reconstruction_gradients), norm(flow_gradients)


@pytest.mark.parametrize("num_latent_tokens", [4, 8])
def test_shared_telemetry_is_read_only_for_joint_loss(num_latent_tokens):
    wrapper = _TelemetryHarness("shared", num_latent_tokens)
    weight = wrapper.shared_weight
    reconstruction = weight.square().sum()
    flow = ((1.5 * weight) + 0.2).square().sum()
    expected = torch.autograd.grad(
        reconstruction + flow, weight, retain_graph=True
    )[0]

    ModelWrapper._measure_unite_shared_gradients(
        wrapper, reconstruction, flow
    )

    assert set(wrapper.logged) == SHARED_KEYS
    assert weight.grad is None
    (reconstruction + flow).backward()
    torch.testing.assert_close(weight.grad, expected)
    assert all(torch.isfinite(value).all() for value in wrapper.logged.values())
    assert wrapper.logged["log/unite_recon_grad_norm"] > 0
    assert wrapper.logged["log/unite_denoise_grad_norm"] > 0


@pytest.mark.parametrize("num_latent_tokens", [4, 8])
def test_separate_telemetry_is_read_only_for_joint_loss(num_latent_tokens):
    wrapper = _TelemetryHarness("separate", num_latent_tokens)
    tokenizer = wrapper.tokenizer_weight
    denoiser = wrapper.denoiser_weight
    reconstruction = tokenizer.square().sum()
    flow = ((1.5 * denoiser) + 0.2).square().sum()
    expected_tokenizer, expected_denoiser = torch.autograd.grad(
        reconstruction + flow,
        (tokenizer, denoiser),
        retain_graph=True,
    )

    wrapper._measure_unite_separate_gradients(reconstruction, flow)

    assert set(wrapper.logged) == SEPARATE_KEYS
    assert tokenizer.grad is None and denoiser.grad is None
    (reconstruction + flow).backward()
    torch.testing.assert_close(tokenizer.grad, expected_tokenizer)
    torch.testing.assert_close(denoiser.grad, expected_denoiser)
    assert all(torch.isfinite(value).all() for value in wrapper.logged.values())
    assert all(wrapper.logged[key] > 0 for key in SEPARATE_KEYS)


@pytest.mark.parametrize("topology", ["shared", "separate"])
def test_telemetry_rejects_zero_component_gradient(topology):
    wrapper = _TelemetryHarness(topology, 4)
    if topology == "shared":
        reconstruction = (wrapper.shared_weight * 0.0).sum()
        flow = wrapper.shared_weight.square().sum()
        measure = ModelWrapper._measure_unite_shared_gradients
    else:
        reconstruction = (wrapper.tokenizer_weight * 0.0).sum()
        flow = wrapper.denoiser_weight.square().sum()
        measure = ReleasedUniteModelWrapper._measure_unite_separate_gradients
    with pytest.raises(RuntimeError, match="zero|non-finite"):
        measure(wrapper, reconstruction, flow)
    assert wrapper.logged == {}


def test_released_wrapper_rejects_alternating_update_mode():
    algo = torch.nn.Module()
    algo.nets = torch.nn.ModuleDict()
    with pytest.raises(ValueError, match="joint reconstruction"):
        ReleasedUniteModelWrapper(
            robomimic_model=algo,
            optimizer=SimpleNamespace(),
            scheduler=None,
            unite_flow_updates_per_reconstruction=1,
            unite_gradient_telemetry_every_n_steps=1,
        )


@pytest.mark.parametrize("shared", [True, False])
def test_real_policy_telemetry_after_one_joint_optimizer_step(shared):
    torch.manual_seed(17)
    policy = _real_policy(shared)
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0e-3)
    reconstruction, flow = _real_component_losses(policy)
    (reconstruction + flow).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    reconstruction, flow = _real_component_losses(policy)
    wrapper = _RealTelemetryHarness(policy, shared)
    wrapper._measure_topology_gradients(reconstruction, flow)

    expected = SHARED_KEYS if shared else SEPARATE_KEYS
    assert set(wrapper.logged) == expected
    assert all(parameter.grad is None for parameter in policy.parameters())
    (reconstruction + flow).backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(torch.isfinite(value).all() for value in wrapper.logged.values())


def test_smoke_telemetry_third_forward_follows_first_positive_lr_update():
    """Exercise the released zero-LR warmup boundary with the real components."""

    torch.manual_seed(17)
    policy = _real_policy(shared=True)
    wrapper = _RealTelemetryHarness(policy, shared=True)
    optimizer = ReleasedUniteCompositeOptimizer(
        named_params=tuple(policy.named_parameters()),
        lr=1.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-6,
        adamw_weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_momentum=0.95,
        muon_adjust_lr_fn="match_rms_adamw",
    )
    scheduler = released_unite_two_stage_scheduler(
        optimizer,
        warmup_steps=8_000,
        decay_start_1_steps=12_000,
        decay_end_1_steps=20_000,
        decay_start_2_steps=1_200_000,
        decay_end_2_steps=1_200_000,
        base_lr_1=1.0e-4,
        base_lr_2=5.0e-5,
        final_lr=1.0e-5,
    )
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in policy.named_parameters()
    }

    assert scheduler.get_last_lr()
    assert all(lr == 0.0 for lr in scheduler.get_last_lr())

    # Forward 1 is the pristine AdaLN-Zero graph. Its clean target and flow
    # prediction are both zero, so flow has no shared-core gradient.
    reconstruction, flow = _real_component_losses(policy)
    reconstruction_norm, flow_norm = _shared_component_gradient_norms(
        policy, reconstruction, flow
    )
    assert reconstruction_norm > 0.0
    assert flow_norm == 0.0
    with pytest.raises(RuntimeError, match="zero gradient norm"):
        wrapper._measure_topology_gradients(reconstruction, flow)
    (reconstruction + flow).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    # The first optimizer step used LR=0. The scheduler has now exposed a
    # positive LR, but no model parameter has changed yet.
    assert all(lr > 0.0 for lr in scheduler.get_last_lr())
    assert all(
        torch.equal(parameter, initial_parameters[name])
        for name, parameter in policy.named_parameters()
    )

    # Forward 2 is therefore still degenerate. Its joint backward is the first
    # update applied with a positive LR.
    reconstruction, flow = _real_component_losses(policy)
    reconstruction_norm, flow_norm = _shared_component_gradient_norms(
        policy, reconstruction, flow
    )
    assert reconstruction_norm > 0.0
    assert flow_norm == 0.0
    with pytest.raises(RuntimeError, match="zero gradient norm"):
        wrapper._measure_topology_gradients(reconstruction, flow)
    (reconstruction + flow).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    assert any(
        not torch.equal(parameter, initial_parameters[name])
        for name, parameter in policy.named_parameters()
    )

    # Forward 3 observes the first positive-LR parameter update and provides
    # finite, nonzero gradients for both components without changing .grad.
    reconstruction, flow = _real_component_losses(policy)
    reconstruction_norm, flow_norm = _shared_component_gradient_norms(
        policy, reconstruction, flow
    )
    assert reconstruction_norm > 0.0
    assert flow_norm > 0.0
    wrapper._measure_topology_gradients(reconstruction, flow)
    assert set(wrapper.logged) == SHARED_KEYS
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(torch.isfinite(value).all() for value in wrapper.logged.values())
