import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from egomimic.models.unite_action_decoder import UniteActionDecoder
from egomimic.pipeline.stages_sampler import TokenwiseMLPActionDecoder
from egomimic.pipeline.stages_unite_released import (
    ReleasedRecipeUniteLatentPolicy,
    ReleasedRecipeUniteObjective,
)
from egomimic.pipeline.stages_unite_separate import (
    SeparateUniteGenerativeEncoder,
    build_configurable_unite_generative_encoder,
)

DOMAIN = "pushshapes_sim_u_socket"


def _backbone_config(num_latent_tokens):
    return {
        "_target_": "egomimic.models.unite_dit.UniteDiTBackbone",
        "input_dim": 16,
        "output_dim": 16,
        "horizon": num_latent_tokens,
        "condition_dim": 12,
        "max_condition_tokens": 1,
        "max_content_tokens": 16,
        "hidden_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "time_fourier_dim": 16,
        "context_adaln_summary": False,
        "trainable_position_embeddings": False,
        "trainable_register_position_embeddings": True,
        "trainable_content_position_embeddings": False,
        "trainable_condition_position_embeddings": False,
    }


def _encoder(shared, num_latent_tokens):
    return build_configurable_unite_generative_encoder(
        backbone_config=_backbone_config(num_latent_tokens),
        share_encoder_denoiser=shared,
        action_dims={DOMAIN: 4},
        condition_input_dim=14,
        latent_dim=16,
        num_latent_tokens=num_latent_tokens,
        condition_dim=12,
        denoiser_hidden_dim=32,
        gradient_checkpointing=False,
    )


def _policy(shared, num_latent_tokens):
    return ReleasedRecipeUniteLatentPolicy(
        generative_encoder=_encoder(shared, num_latent_tokens),
        decoders={DOMAIN: TokenwiseMLPActionDecoder(16, 16, 4)},
        num_inference_steps=2,
        flow_steps_per_reconstruction=14,
        flow_mini_batch=4,
        timestep_shift_alpha=0.5,
    )


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("num_latent_tokens", [4, 8, 16])
def test_six_rows_propagate_register_count_and_only_toggle_weight_tying(
    shared, num_latent_tokens
):
    encoder = _encoder(shared, num_latent_tokens)
    assert encoder.latent_dim == 16
    assert encoder.num_latent_tokens == num_latent_tokens
    assert encoder.denoising_module.input_dim == 16
    assert encoder.denoising_module.output_dim == 16
    assert encoder.denoising_module.horizon == num_latent_tokens
    assert encoder.output_norm.normalized_shape == (16,)
    assert isinstance(encoder.denoising_module.pos_emb, torch.nn.Parameter)
    assert encoder.denoising_module.max_content_tokens == 16
    assert encoder.denoising_module.max_condition_tokens == 1
    assert not isinstance(
        encoder.denoising_module.condition_pos_emb, torch.nn.Parameter
    )
    if shared:
        assert not isinstance(encoder, SeparateUniteGenerativeEncoder)
        named = encoder.shared_reconstruction_denoising_named_parameters([DOMAIN])
        assert named
    else:
        assert isinstance(encoder, SeparateUniteGenerativeEncoder)
        assert encoder.tokenization_module is not encoder.denoising_module
        tokenizer, denoiser = (
            encoder.separate_reconstruction_denoising_named_parameters([DOMAIN])
        )
        assert {id(p) for _, p in tokenizer}.isdisjoint({id(p) for _, p in denoiser})
        assert encoder.denoising_output_norm.normalized_shape == (16,)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("num_latent_tokens", [4, 8, 16])
def test_materialized_h16_rows_construct_and_forward_clean_content(
    shared, num_latent_tokens
):
    sharing = "shared" if shared else "separate"
    config_path = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/model/bf"
        / f"us_unite_register_{sharing}_nt{num_latent_tokens}_s42.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["robomimic_model"]["action_horizon"] == 16
    ge_config = config["robomimic_model"]["stages"][3]["generative_encoder"]
    backbone_config = dict(ge_config["backbone_config"])
    backbone_config.update(hidden_dim=32, depth=2, num_heads=4, time_fourier_dim=16)
    encoder = build_configurable_unite_generative_encoder(
        backbone_config=backbone_config,
        share_encoder_denoiser=ge_config["share_encoder_denoiser"],
        action_dims=ge_config["action_dims"],
        condition_input_dim=ge_config["condition_input_dim"],
        latent_dim=ge_config["latent_dim"],
        num_latent_tokens=ge_config["num_latent_tokens"],
        condition_dim=ge_config["condition_dim"],
        denoiser_hidden_dim=32,
        gradient_checkpointing=ge_config["gradient_checkpointing"],
        tokenization_time_max=ge_config["tokenization_time_max"],
    ).train()

    assert encoder.denoising_module.max_content_tokens == 16
    assert encoder.denoising_module.max_condition_tokens == 1
    assert encoder.denoising_module.gradient_checkpointing is True
    modules = [encoder.denoising_module]
    if not shared:
        modules.append(encoder.tokenization_module)
    assert all(module.gradient_checkpointing for module in modules)

    clean_actions = torch.randn(2, 16, 4, requires_grad=True)
    clean_latent = encoder.tokenize(clean_actions, DOMAIN)
    assert clean_latent.shape == (2, num_latent_tokens, 16)
    predicted = encoder.denoise(
        clean_latent.detach(),
        torch.full((2,), 0.5),
        torch.randn(2, 128),
        DOMAIN,
    )
    assert predicted.shape == clean_latent.shape
    (clean_latent.square().mean() + predicted.square().mean()).backward()


def test_cfg_training_dropout_uses_a_learned_null_condition():
    policy = _policy(True, 4).train()
    policy.condition_dropout_probability = 1.0
    condition = torch.randn(3, 14)
    dropped, fraction = policy._condition_with_dropout(condition)
    expected = policy._null_condition_like(condition)
    torch.testing.assert_close(dropped, expected)
    assert fraction.item() == 1.0
    assert policy.null_observation_condition.requires_grad


def test_cfg_uses_unconditional_plus_scale_times_conditional_delta(monkeypatch):
    policy = _policy(True, 4).eval()
    latent = torch.randn(2, 4, 16)
    condition = torch.randn(2, 14)
    time = torch.full((2,), 0.5)

    def denoise(value, _, context):
        level = context.reshape(context.shape[0], -1).mean(dim=-1).reshape(-1, 1, 1)
        return torch.zeros_like(value) + level

    monkeypatch.setattr(policy.generative_encoder, "denoise", denoise)
    conditioned = denoise(latent, time, condition)
    unconditional = denoise(latent, time, policy._null_condition_like(condition))
    actual = policy._guided_clean_prediction(latent, time, condition, 4.0)
    torch.testing.assert_close(
        actual, unconditional + 4.0 * (conditioned - unconditional)
    )


def test_dopri5_uses_released_50_point_shifted_grid_and_tolerances(monkeypatch):
    policy = _policy(True, 4).eval()
    policy.dopri5_num_steps = 50
    policy.dopri5_atol = 1.0e-6
    policy.dopri5_rtol = 1.0e-3
    captured = {}

    def odeint(function, initial, times, *, method, atol, rtol):
        captured.update(
            method=method, times=times.detach().clone(), atol=atol, rtol=rtol
        )
        derivative = function(times[0], initial)
        return torch.stack((initial, initial + derivative * 0.0))

    monkeypatch.setitem(sys.modules, "torchdiffeq", SimpleNamespace(odeint=odeint))
    noise = torch.randn(2, 4, 16)
    condition = torch.randn(2, 14)
    endpoint = policy.sample(noise, condition, sampling_method="dopri5")
    assert endpoint.shape == noise.shape
    assert captured["method"] == "dopri5"
    assert len(captured["times"]) == 50
    assert captured["times"][0].item() == 0.0
    assert captured["times"][-1].item() == 1.0
    assert captured["atol"] == 1.0e-6 and captured["rtol"] == 1.0e-3
    assert policy._last_sampler_nfe == 1


@pytest.mark.parametrize("shared", [True, False])
def test_released_step_is_one_reconstruction_plus_14_summed_flow_samples(shared):
    policy = _policy(shared, 4).train()
    calls = []
    modules = [policy.generative_encoder.denoising_module]
    if not shared:
        modules.append(policy.generative_encoder.tokenization_module)
    hooks = [
        module.register_forward_hook(
            lambda module, args, output: calls.append(args[0].shape[0])
        )
        for module in modules
    ]
    batch = {
        "target": torch.randn(2, 4, 4),
        "condition": torch.randn(2, 14),
        "sampler/noise": torch.randn(2, 4, 16),
        "embodiment": DOMAIN,
    }
    output = ReleasedRecipeUniteObjective()(policy(batch))
    for hook in hooks:
        hook.remove()
    assert sorted(calls) == sorted([2, 8, 8, 8, 4])
    assert policy._flow_chunks(14, 4) == (4, 4, 4, 2)
    total = output["loss/unite_reconstruction"] + output["loss/unite_latent"]
    assert total.ndim == 0 and torch.isfinite(total)
    total.backward()


def test_separate_flow_stop_gradient_does_not_reach_tokenizer():
    policy = _policy(False, 4).train()
    output = policy(
        {
            "target": torch.randn(2, 4, 4),
            "condition": torch.randn(2, 14),
            "sampler/noise": torch.randn(2, 4, 16),
            "embodiment": DOMAIN,
        }
    )
    output["unite/flow_loss"].backward()
    encoder = policy.generative_encoder
    assert all(p.grad is None for p in encoder.tokenization_module.parameters())
    assert all(p.grad is None for p in encoder.output_norm.parameters())
    assert any(p.grad is not None for p in encoder.denoising_module.parameters())
    tokenizer, denoiser = policy.separate_reconstruction_denoising_named_parameters(
        [DOMAIN]
    )
    assert tokenizer and denoiser


def test_paper_sized_action_decoder_contract_and_shape():
    decoder = UniteActionDecoder(
        latent_dim=16,
        action_dim=4,
        num_latent_tokens=4,
        action_horizon=16,
        hidden_dim=48,
        depth=2,
        num_heads=4,
        gradient_checkpointing=False,
    )
    assert decoder(torch.randn(2, 4, 16)).shape == (2, 16, 4)
    assert not isinstance(decoder.decoder_pos_embed, torch.nn.Parameter)
    config = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/model/bf/us_unite_register_shared_nt4_s42.yaml"
    ).read_text()
    assert "hidden_dim: 768" in config
    assert "depth: 12" in config
    assert "num_heads: 12" in config


def test_released_config_uses_paper_dopri5_and_classifier_free_guidance():
    import yaml

    config_path = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/model/bf/us_unite_register_shared_nt4_s42.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    policy = config["robomimic_model"]["stages"][3]
    assert policy["sampling_method"] == "dopri5"
    assert policy["condition_dropout_probability"] == 0.1
    assert policy["cfg_scale"] == 4.0
    assert policy["cfg_interval"] == [0.0, 1.0]
    assert policy["cfg_norm_order"] == "norm_first"
    assert policy["dopri5_num_steps"] == 50
    assert policy["dopri5_atol"] == 1.0e-6
    assert policy["dopri5_rtol"] == 1.0e-3
    # J=8 remains an explicit protocol-compatible fallback, not the paper sampler.
    assert policy["num_inference_steps"] == 8


def test_scheduler_second_decay_boundaries_are_ordered_in_canonical_config():
    import yaml

    config_path = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/model/bf/us_unite_register_shared_nt4_s42.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    scheduler = config["scheduler"]
    assert scheduler["decay_start_2_steps"] == 120000
    assert scheduler["decay_end_2_steps"] == 152000
    assert scheduler["decay_start_2_steps"] < scheduler["decay_end_2_steps"]
