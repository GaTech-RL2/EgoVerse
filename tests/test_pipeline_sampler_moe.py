import torch

from egomimic.models.moe_denoising_nets import MoECrossTransformer
from egomimic.models.moe_ffn import MoEFFN
from egomimic.pipeline.stages_sampler_moe import MoEMultiJActionSampler


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
    assert "sampler/endpoint" in sampler.writes
    assert output["sampler/endpoint"].shape == (3, 5, 8)
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
