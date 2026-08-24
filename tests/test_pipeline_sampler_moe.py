import copy

import torch
from torch.nn import functional as F

from egomimic.models.moe_denoising_nets import MoECrossTransformer
from egomimic.models.moe_ffn import MoEFFN
from egomimic.pipeline.stages_sampler_moe import MoEMultiJActionSampler


def _reference_sparse_forward(module, x):
    """Previous dispatch implementation, retained only as an equivalence oracle."""
    original_shape = x.shape
    tokens = x.reshape(-1, original_shape[-1])
    probabilities = F.softmax(module.gate(tokens).float(), dim=-1)
    top_values, top_indices = torch.topk(probabilities, module.top_k, dim=-1)
    top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    routed_weights = torch.zeros_like(probabilities)
    routed_weights.scatter_(1, top_indices, top_weights)
    output = torch.zeros_like(tokens)
    for expert_index, expert in enumerate(module.experts):
        selected = routed_weights[:, expert_index] > 0
        if not bool(selected.any()):
            continue
        token_indices = selected.nonzero(as_tuple=True)[0]
        expert_output = expert(tokens.index_select(0, token_indices))
        weights = routed_weights[token_indices, expert_index].unsqueeze(-1)
        output.index_add_(
            0,
            token_indices,
            (weights.to(expert_output.dtype) * expert_output).to(output.dtype),
        )

    dispatch = (routed_weights > 0).to(probabilities.dtype)
    denominator = max(tokens.shape[0] * module.top_k, 1)
    fractions = dispatch.sum(dim=0) / denominator
    mean_probability = probabilities.mean(dim=0)
    aux_loss = (
        module.aux_weight
        * module.num_experts
        * torch.sum(fractions.detach() * mean_probability)
    )
    return output.reshape(original_shape), aux_loss


def _sampler():
    denoiser = MoECrossTransformer(
        nblocks=2,
        cond_dim=12,
        hidden_dim=16,
        act_dim=8,
        act_seq=5,
        n_heads=4,
        dropout=0.0,
        mlp_layers=2,
        mlp_ratio=2,
        time_conditioning="additive",
        moe_experts=4,
        moe_top_k=2,
        moe_d_expert=24,
        moe_aux_weight=1.0e-3,
    )
    return MoEMultiJActionSampler(
        denoising_module=denoiser,
        condition_input_dim=10,
        condition_dim=12,
        action_horizon=5,
        action_dims={"eva_bimanual": 3},
        latent_dim=8,
        decoder_hidden_dim=16,
        denoiser_hidden_dim=16,
        sampling_schedule={1: {2: 1.0}},
        gradient_checkpointing=False,
    )


def test_moe_sampler_emits_finite_aux_and_router_gradients():
    sampler = _sampler().train()
    output = sampler(
        {
            "sampler/noise": torch.randn(3, 5, 8),
            "condition": torch.randn(3, 10),
            "embodiment": "eva_bimanual",
        }
    )
    assert output["pred_action"].shape == (3, 5, 3)
    assert output["loss/moe_lb"].ndim == 0
    assert torch.isfinite(output["loss/moe_lb"])
    assert 0.0 < float(output["loss/moe_lb"]) < 0.004
    output["pred_action"].square().mean().add(output["loss/moe_lb"]).backward()

    gates = [module.gate for module in sampler.modules() if isinstance(module, MoEFFN)]
    assert len(gates) == 2
    assert all(gate.weight.grad is not None for gate in gates)
    assert all(
        parameter.grad is not None
        for module in sampler.modules()
        if isinstance(module, MoEFFN)
        for expert in module.experts
        for parameter in expert.parameters()
    )


def test_aux_scale_is_averaged_over_blocks_and_j_calls():
    sampler = _sampler().train()
    output = sampler(
        {
            "sampler/noise": torch.randn(2, 5, 8),
            "condition": torch.randn(2, 10),
            "embodiment": "eva_bimanual",
        }
    )
    assert float(output["loss/moe_lb"]) < 0.004
    assert len(sampler._moe_aux_calls) == 2


def test_batched_dispatch_matches_sparse_forward_and_gradients():
    torch.manual_seed(7)
    batched = MoEFFN(
        d_model=16,
        d_intermediate=24,
        num_experts=4,
        top_k=2,
        aux_weight=1.0e-3,
    ).double()
    reference = copy.deepcopy(batched)
    batched_input = torch.randn(3, 5, 16, dtype=torch.double, requires_grad=True)
    reference_input = batched_input.detach().clone().requires_grad_(True)

    batched_output = batched(batched_input)
    reference_output, reference_aux = _reference_sparse_forward(
        reference, reference_input
    )
    torch.testing.assert_close(batched_output, reference_output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        batched.last_aux_loss, reference_aux, rtol=1e-12, atol=1e-12
    )

    batched_output.square().mean().add(batched.last_aux_loss).backward()
    reference_output.square().mean().add(reference_aux).backward()
    torch.testing.assert_close(
        batched_input.grad, reference_input.grad, rtol=1e-11, atol=1e-12
    )
    for (batched_name, batched_parameter), (
        reference_name,
        reference_parameter,
    ) in zip(batched.named_parameters(), reference.named_parameters(), strict=True):
        assert batched_name == reference_name
        torch.testing.assert_close(
            batched_parameter.grad,
            reference_parameter.grad,
            rtol=1e-11,
            atol=1e-12,
        )


def test_batched_dispatch_keeps_unselected_experts_in_autograd_graph():
    module = MoEFFN(
        d_model=8,
        d_intermediate=12,
        num_experts=4,
        top_k=1,
    )
    with torch.no_grad():
        module.gate.weight.zero_()
        module.gate.weight[0].fill_(1.0)
        module.gate.weight[1:].fill_(-1.0)

    output = module(torch.ones(2, 3, 8))
    output.square().mean().backward()

    for expert_index, expert in enumerate(module.experts):
        for parameter in expert.parameters():
            assert parameter.grad is not None
            if expert_index > 0:
                assert torch.count_nonzero(parameter.grad) == 0


def test_batched_dispatch_preserves_checkpoint_parameter_names():
    module = MoEFFN(d_model=8, d_intermediate=12, num_experts=4, top_k=2)
    assert list(module.state_dict()) == [
        "gate.weight",
        "experts.0.gate_value.weight",
        "experts.0.gate_value.bias",
        "experts.0.output.weight",
        "experts.0.output.bias",
        "experts.1.gate_value.weight",
        "experts.1.gate_value.bias",
        "experts.1.output.weight",
        "experts.1.output.bias",
        "experts.2.gate_value.weight",
        "experts.2.gate_value.bias",
        "experts.2.output.weight",
        "experts.2.output.bias",
        "experts.3.gate_value.weight",
        "experts.3.gate_value.bias",
        "experts.3.output.weight",
        "experts.3.output.bias",
    ]
