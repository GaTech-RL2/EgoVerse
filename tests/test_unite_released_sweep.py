import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from egomimic.models.unite_action_decoder import UniteActionDecoder
from egomimic.models.unite_action_decoder import (
    _sincos_positions as _decoder_sincos_positions,
)
from egomimic.models.unite_dit import _sincos_positions as _encoder_sincos_positions
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


def test_released_1d_position_layout_concatenates_sin_then_cos():
    frequency = torch.tensor((1.0, 0.01))
    angle = frequency
    expected = torch.cat((angle.sin(), angle.cos()))
    for helper in (_encoder_sincos_positions, _decoder_sincos_positions):
        table = helper(2, 4)
        torch.testing.assert_close(table[0, 0], torch.tensor((0.0, 0.0, 1.0, 1.0)))
        torch.testing.assert_close(table[0, 1], expected)


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
        "in_context_start": 1,
        "in_context_len": 32,
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
        in_context_start=1,
        in_context_len=32,
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
def test_register_counts_propagate_and_only_toggle_weight_tying(
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
    assert encoder.denoising_module.in_context_start == 1
    assert encoder.denoising_module.in_context_len == 32
    assert isinstance(encoder.denoising_module.in_context_pos_emb, torch.nn.Parameter)
    in_context_std = encoder.denoising_module.in_context_pos_emb.detach().std(
        unbiased=False
    )
    assert 0.015 < float(in_context_std) < 0.025
    assert not isinstance(
        encoder.denoising_module.condition_pos_emb, torch.nn.Parameter
    )
    torch.testing.assert_close(
        encoder.denoising_module.condition_pos_emb,
        torch.zeros(1, 1, encoder.denoising_module.hidden_dim),
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


def test_only_four_active_register_configs_are_hydra_selectable():
    config_dir = (
        Path(__file__).parents[1] / "egomimic/hydra_configs/model/bf"
    )
    actual = sorted(path.name for path in config_dir.glob("us_unite_register_*_s42.yaml"))
    assert actual == [
        "us_unite_register_separate_nt4_s42.yaml",
        "us_unite_register_separate_nt8_s42.yaml",
        "us_unite_register_shared_nt4_s42.yaml",
        "us_unite_register_shared_nt8_s42.yaml",
    ]


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("num_latent_tokens", [4, 8])
def test_materialized_active_h16_rows_construct_and_forward_clean_content(
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
    assert ge_config["in_context_start"] == 4
    assert ge_config["in_context_len"] == 32
    backbone_config = dict(ge_config["backbone_config"])
    assert backbone_config["in_context_start"] == 4
    assert backbone_config["in_context_len"] == 32
    assert backbone_config["trainable_in_context_position_embeddings"] is True
    backbone_config.update(
        hidden_dim=32,
        depth=2,
        num_heads=4,
        time_fourier_dim=16,
        in_context_start=1,
    )
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
        in_context_start=1,
        in_context_len=ge_config["in_context_len"],
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


@pytest.mark.parametrize("shared", [True, False])
def test_tokenizer_uses_clean_content_and_learned_null_condition_together(shared):
    encoder = _encoder(shared, 4).eval()
    backbone = (
        encoder.denoising_module if shared else encoder.tokenization_module
    )
    captured = {}

    def capture(_, args, kwargs):
        captured["condition"] = kwargs.get("condition")
        captured["content_tokens"] = kwargs.get("content_tokens")

    handle = backbone.register_forward_pre_hook(capture, with_kwargs=True)
    output = encoder.tokenize(torch.randn(2, 16, 4), DOMAIN)
    handle.remove()
    assert output.shape == (2, 4, 16)
    assert captured["condition"].shape == (2, 1, 12)
    assert captured["content_tokens"].shape == (2, 16, 12)
    if shared:
        assert encoder.null_condition_inputs[DOMAIN].requires_grad
    else:
        assert encoder.tokenization_null_condition_inputs[DOMAIN].requires_grad
        assert (
            encoder.tokenization_null_condition_inputs[DOMAIN]
            is not encoder.null_condition_inputs[DOMAIN]
        )


def test_in_context_condition_is_inserted_at_configured_block():
    encoder = _encoder(True, 4).eval()
    sequence_lengths = []
    handles = [
        block.register_forward_pre_hook(
            lambda _, args: sequence_lengths.append(int(args[0].shape[1]))
        )
        for block in encoder.denoising_module.blocks
    ]
    encoder.tokenize(torch.randn(2, 16, 4), DOMAIN)
    for handle in handles:
        handle.remove()
    assert sequence_lengths == [20, 52]


def test_time_embedding_uses_released_normal_initialization():
    backbone = _encoder(True, 4).denoising_module
    for index in (0, 2):
        std = backbone.time_embedder.mlp[index].weight.detach().std(unbiased=False)
        assert 0.015 < float(std) < 0.025


def test_cfg_training_dropout_uses_a_learned_null_condition():
    policy = _policy(True, 4).train()
    policy.condition_dropout_probability = 1.0
    condition = torch.randn(3, 14)
    dropped, fraction = policy._condition_with_dropout(condition)
    expected = policy._null_condition_like(condition)
    torch.testing.assert_close(dropped, expected)
    assert fraction.item() == 1.0
    assert policy.generative_encoder.null_condition_inputs[DOMAIN].requires_grad


def test_shared_tokenizer_and_cfg_use_the_same_null_input():
    policy = _policy(True, 4).train()
    condition = torch.randn(3, 14)
    expanded = policy._null_condition_like(condition)
    expected = policy.generative_encoder.null_condition_inputs[DOMAIN]
    torch.testing.assert_close(expanded, expected.reshape(1, -1).expand_as(condition))
    shared_names = {
        name
        for name, _ in policy.shared_reconstruction_denoising_named_parameters([DOMAIN])
    }
    assert f"null_condition_inputs.{DOMAIN}" in shared_names
    assert "condition_projection.weight" in shared_names
    assert "denoising_module.condition_projection.weight" in shared_names


def test_flow_time_uses_full_logit_normal_support(monkeypatch):
    policy = _policy(True, 4)
    low_logit = torch.logit(torch.tensor(0.01)).item()

    def fixed_normal(*size, device=None, dtype=None):
        return torch.full(size, low_logit, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fixed_normal)
    sampled = policy._sample_flow_time(3, torch.device("cpu"))
    expected = policy.shift_time(torch.full((3,), 0.01))
    torch.testing.assert_close(sampled, expected)
    assert bool((sampled < policy.shift_time(torch.tensor(0.05))).all())


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


def test_dopri5_keeps_state_and_vector_field_fp32_under_bf16_autocast(monkeypatch):
    policy = _policy(True, 4).eval()
    captured = {}

    def guided(latent, time, condition, cfg_scale):
        return torch.zeros_like(latent, dtype=torch.bfloat16)

    def odeint(function, initial, times, *, method, atol, rtol):
        derivative = function(times[0], initial)
        captured["initial_dtype"] = initial.dtype
        captured["derivative_dtype"] = derivative.dtype
        captured["derivative_finite"] = bool(torch.isfinite(derivative).all())
        return torch.stack((initial, initial + derivative * 0.0))

    monkeypatch.setattr(policy, "_guided_clean_prediction", guided)
    monkeypatch.setitem(sys.modules, "torchdiffeq", SimpleNamespace(odeint=odeint))
    noise = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    condition = torch.randn(2, 14, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        endpoint = policy.sample(noise, condition, sampling_method="dopri5")
    assert endpoint.dtype == torch.float32
    assert captured == {
        "initial_dtype": torch.float32,
        "derivative_dtype": torch.float32,
        "derivative_finite": True,
    }
    assert policy._last_sampler_nfe == 1


def test_euler_fallback_integrates_the_full_ode_interval(monkeypatch):
    policy = _policy(True, 4).eval()
    evaluated_times = []

    def guided(latent, time, condition, cfg_scale):
        evaluated_times.append(time.detach().clone())
        return torch.zeros_like(latent)

    monkeypatch.setattr(policy, "_guided_clean_prediction", guided)
    noise = torch.randn(2, 4, 16)
    condition = torch.randn(2, 14)
    output = policy.sample(noise, condition, sampling_method="euler", cfg_scale=1.0)
    assert output.shape == noise.shape
    assert len(evaluated_times) == policy.num_inference_steps
    assert evaluated_times[0][0].item() == 0.0
    raw_grid = torch.linspace(0.0, 1.0, policy.num_inference_steps + 1)
    shifted_grid = policy.shift_time(raw_grid)
    torch.testing.assert_close(evaluated_times[-1][0], shifted_grid[-2])


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
    denoiser_names = {name for name, _ in denoiser}
    assert "denoising_module.condition_projection.weight" in denoiser_names
    assert "denoising_module.content_projection.weight" not in denoiser_names


def test_paper_sized_action_decoder_contract_and_shape():
    torch.manual_seed(123)
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
    assert not torch.equal(
        decoder.decoder.layers[0].qkv.weight,
        decoder.decoder.layers[1].qkv.weight,
    )
    assert decoder.decoder.layers[0].layernorm_before.eps == 1.0e-12
    assert decoder.decoder.layers[0].layernorm_after.eps == 1.0e-12
    assert decoder.decoder.norm.eps == 1.0e-12
    latent = torch.randn(2, 4, 16, requires_grad=True)
    output = decoder(latent)
    assert output.shape == (2, 16, 4)
    output.square().mean().backward()
    assert latent.grad is not None and bool(torch.isfinite(latent.grad).all())
    assert all(
        layer.qkv.weight.grad is not None
        and bool(torch.isfinite(layer.qkv.weight.grad).all())
        for layer in decoder.decoder.layers
    )
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


def test_scheduler_preserves_released_no_second_decay_within_run():
    import yaml

    config_path = (
        Path(__file__).parents[1]
        / "egomimic/hydra_configs/model/bf/us_unite_register_shared_nt4_s42.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    scheduler = config["scheduler"]
    assert scheduler["decay_start_2_steps"] == 1200000
    assert scheduler["decay_end_2_steps"] == 1200000
    assert scheduler["decay_start_2_steps"] > 240000
