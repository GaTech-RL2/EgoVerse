import copy

import torch
import torch.nn as nn

from egomimic.models.denoising_nets import CrossTransformer
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_sampler import (
    DPStyleObsEncoder,
    FusedObsEncoder,
    GaussianLatentNoise,
    LatentFlowSampler,
    MultiJActionSampler,
    NativeActionMSELoss,
    PerEmbodimentActionDecoder,
    TemporalConvActionDecoder,
    TokenwiseMLPActionDecoder,
)


def test_dp_style_obs_encoder_concatenates_agent_xy_and_image_feature():
    image_encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 64))
    encoder = DPStyleObsEncoder(
        obs_specs={"state_agent_obj": {"input_dim": 2, "input_slice": [0, 2]}},
        img_encoders={"front_img_1": image_encoder},
    )
    out = encoder.forward_packed(
        obs_packed={
            "state_agent_obj": torch.randn(3, 5),
            "front_img_1": torch.randn(3, 3, 4, 4),
        },
        T_total=3,
    )
    assert out.shape == (3, 66)


class _FakeObsEncoder(nn.Module):
    def forward_packed(self, *, obs_packed, cu_seqlens, T_total, **kwargs):
        assert T_total == 6
        assert torch.equal(cu_seqlens.cpu(), torch.tensor([0, 2, 4, 6]))
        assert obs_packed["state"].shape == (6, 5)
        return torch.arange(18, dtype=torch.float32).reshape(6, 3)


def test_standard_obs_encoder_keeps_packing_internal():
    stage = FusedObsEncoder(_FakeObsEncoder(), n_obs_steps=2)
    actions = torch.randn(3, 7, 4)
    out = stage(
        {
            "obs/state": torch.randn(3, 2, 5),
            "actions": actions,
            "embodiment": "eva_bimanual",
        }
    )
    assert out["condition"].shape == (3, 6)
    assert out["target"] is actions
    assert "cu_seqlens" not in out
    assert "actions" not in out


class _FakeSingleFrameEncoder(nn.Module):
    def forward_packed(self, *, obs_packed, cu_seqlens, T_total, **kwargs):
        assert T_total == 4
        assert obs_packed["image"].shape == (4, 3, 8, 8)
        assert obs_packed["state"].shape == (4, 5)
        assert torch.equal(cu_seqlens.cpu(), torch.arange(5))
        return torch.zeros(4, 6)


def test_single_frame_pusht_observation_has_no_public_history_grid():
    stage = FusedObsEncoder(_FakeSingleFrameEncoder(), n_obs_steps=1)
    out = stage(
        {
            "obs/image": torch.randn(4, 3, 8, 8),
            "obs/state": torch.randn(4, 5),
            "actions": torch.randn(4, 16, 2),
            "embodiment": "pushshapes_sim",
        }
    )
    assert out["condition"].shape == (4, 6)
    assert out["target"].shape == (4, 16, 2)
    assert "cu_seqlens" not in out


def _nodes():
    field = CrossTransformer(
        nblocks=2,
        cond_dim=12,
        hidden_dim=32,
        act_dim=8,
        act_seq=7,
        n_heads=4,
        dropout=0.0,
        mlp_layers=2,
        mlp_ratio=2,
        time_conditioning="additive",
    )
    return (
        GaussianLatentNoise(num_tokens=7, latent_dim=8),
        MultiJActionSampler(
            denoising_module=field,
            condition_input_dim=14,
            condition_dim=12,
            action_horizon=7,
            action_dims={"eva_bimanual": 4, "human_bimanual": 6},
            latent_dim=8,
            decoder_hidden_dim=16,
            decoder_extra_hidden_layers_by_domain={"human_bimanual": 2},
            denoiser_hidden_dim=32,
            num_inference_steps=2,
            sampling_schedule={1: {1: 0.5, 2: 0.5}, 5: {2: 0.75, 4: 0.25}},
            gradient_accumulation_steps=2,
        ),
        NativeActionMSELoss(),
    )


def _batch(embodiment="eva_bimanual", target=True):
    out = {"condition": torch.randn(3, 14), "embodiment": embodiment}
    if target:
        dim = 4 if embodiment == "eva_bimanual" else 6
        out["target"] = torch.randn(3, 7, dim)
    return out


def _temporal_nodes():
    field = CrossTransformer(
        nblocks=2,
        cond_dim=12,
        hidden_dim=32,
        act_dim=8,
        act_seq=8,
        n_heads=4,
        dropout=0.0,
        mlp_layers=2,
        mlp_ratio=2,
        time_conditioning="additive",
    )
    return (
        GaussianLatentNoise(num_tokens=8, latent_dim=8),
        LatentFlowSampler(
            denoising_module=field,
            condition_input_dim=14,
            condition_dim=12,
            domains=["eva_bimanual", "human_bimanual"],
            latent_dim=8,
            denoiser_hidden_dim=32,
            num_inference_steps=2,
            sampling_schedule={1: {1: 1.0}},
        ),
        PerEmbodimentActionDecoder(
            decoders={
                "eva_bimanual": TemporalConvActionDecoder(8, 16, 4),
                "human_bimanual": TemporalConvActionDecoder(8, 16, 6),
            }
        ),
        NativeActionMSELoss(),
    )


def _temporal_batch(embodiment="eva_bimanual", target=True):
    out = {"condition": torch.randn(3, 14), "embodiment": embodiment}
    if target:
        dim = 4 if embodiment == "eva_bimanual" else 6
        out["target"] = torch.randn(3, 16, dim)
    return out


def test_separate_noise_sampler_and_loss_nodes_backpropagate_end_to_end():
    noise, sampler, loss = _nodes()
    pipe = Pipeline([noise, sampler, loss]).train()
    out = pipe(_batch())
    assert out["sampler/noise"].shape == (3, 7, 8)
    assert out["pred_action"].shape == (3, 7, 4)
    assert out["log/sampler_unroll_steps"] == 1.0
    assert sampler.denoising_module is not None
    assert any(
        name.startswith("denoising_module.") for name, _ in sampler.named_parameters()
    )
    out["loss/native_action"].backward()
    assert sampler.denoising_module.proj_u.weight.grad is not None
    assert sampler.decoders["eva_bimanual"][-1].weight.grad is not None


def test_temporal_decoder_maps_h8_l8_to_h16_actions_and_backpropagates():
    noise, sampler, decoder_stage, loss = _temporal_nodes()
    pipe = Pipeline([noise, sampler, decoder_stage, loss]).train()
    out = pipe(_temporal_batch())

    assert out["sampler/noise"].shape == (3, 8, 8)
    assert out["sampler/endpoint"].shape == (3, 8, 8)
    assert out["pred_action"].shape == (3, 16, 4)
    decoder = decoder_stage.decoder_for("eva_bimanual")
    assert isinstance(decoder, TemporalConvActionDecoder)
    assert decoder.num_layers == 3
    assert decoder.channel_projection[-1].out_features == 4
    assert decoder.temporal_upsampler.in_channels == 4

    out["loss/native_action"].backward()
    assert sampler.denoising_module.proj_u.weight.grad is not None
    assert decoder.channel_projection[0].weight.grad is not None
    assert decoder.temporal_upsampler.weight.grad is not None


def test_temporal_decoder_rollout_preserves_per_domain_action_dims():
    noise, sampler, decoder, _ = _temporal_nodes()
    sampler.eval()
    decoder.eval()
    out = decoder(sampler(noise(_temporal_batch("human_bimanual", target=False))))
    assert out["pred_action"].shape == (3, 16, 6)


def test_default_token_mlp_decoder_remains_horizon_preserving():
    noise, sampler, _ = _nodes()
    sampler.eval()
    out = sampler(noise(_batch(target=False)))
    assert sampler.decoder_type == "token_mlp"
    assert sampler.latent_horizon == sampler.action_horizon == 7
    assert out["pred_action"].shape == (3, 7, 4)


def test_legacy_composite_preserves_optimizer_parameter_order():
    _, sampler, _ = _nodes()
    assert list(sampler._modules)[:4] == [
        "denoising_module",
        "decoders",
        "domain_embeddings",
        "condition_projection",
    ]


def test_split_tokenwise_decoder_routes_domains_and_backpropagates():
    decoder = PerEmbodimentActionDecoder(
        decoders={
            "eva_bimanual": TokenwiseMLPActionDecoder(8, 16, 4),
            "human_bimanual": TokenwiseMLPActionDecoder(8, 16, 6),
        }
    )
    endpoint = torch.randn(3, 7, 8, requires_grad=True)
    out = decoder(
        {
            "sampler/endpoint": endpoint,
            "embodiment": "human_bimanual",
        }
    )
    assert decoder.domains == ("eva_bimanual", "human_bimanual")
    assert decoder.latent_dim == 8
    assert decoder.temporal_factor == 1
    assert decoder.action_dims == {"eva_bimanual": 4, "human_bimanual": 6}
    assert decoder.output_num_tokens(7) == 7
    assert out["pred_action"].shape == (3, 7, 6)
    out["pred_action"].square().mean().backward()
    assert endpoint.grad is not None
    assert decoder.decoder_for("human_bimanual")[-1].weight.grad is not None


def test_per_embodiment_decoder_rejects_mixed_temporal_mappings():
    try:
        PerEmbodimentActionDecoder(
            decoders={
                "eva_bimanual": TokenwiseMLPActionDecoder(8, 16, 4),
                "human_bimanual": TemporalConvActionDecoder(8, 16, 6),
            }
        )
    except ValueError as exc:
        assert "temporal_factor" in str(exc)
    else:
        raise AssertionError("mixed temporal mappings were not rejected")


def test_per_embodiment_decoder_rejects_unknown_domain_and_malformed_endpoint():
    decoder = PerEmbodimentActionDecoder(
        decoders={"eva_bimanual": TokenwiseMLPActionDecoder(8, 16, 4)}
    )
    try:
        decoder(
            {
                "sampler/endpoint": torch.randn(2, 7, 8),
                "embodiment": "human_bimanual",
            }
        )
    except KeyError as exc:
        assert "Unknown embodiment" in str(exc)
    else:
        raise AssertionError("unknown embodiment was not rejected")

    for malformed in (torch.randn(2, 8), torch.randn(2, 7, 9)):
        try:
            decoder(
                {
                    "sampler/endpoint": malformed,
                    "embodiment": "eva_bimanual",
                }
            )
        except ValueError as exc:
            assert "sampler/endpoint shape" in str(exc)
        else:
            raise AssertionError("malformed latent endpoint was not rejected")


class _MalformedOutputDecoder(nn.Module):
    latent_dim = 8
    action_dim = 4
    temporal_factor = 1

    @staticmethod
    def output_num_tokens(input_num_tokens):
        return int(input_num_tokens)

    @staticmethod
    def forward(latent):
        return latent[..., :3]


def test_per_embodiment_decoder_rejects_malformed_branch_output():
    decoder = PerEmbodimentActionDecoder(
        decoders={"eva_bimanual": _MalformedOutputDecoder()}
    )
    try:
        decoder(
            {
                "sampler/endpoint": torch.randn(2, 7, 8),
                "embodiment": "eva_bimanual",
            }
        )
    except RuntimeError as exc:
        assert "produced" in str(exc)
    else:
        raise AssertionError("malformed decoder output was not rejected")


def test_legacy_composite_matches_split_nodes_numerically():
    noise_stage, legacy, _ = _nodes()
    legacy.eval()
    split_sampler = LatentFlowSampler(
        denoising_module=copy.deepcopy(legacy.denoising_module),
        condition_input_dim=legacy.condition_input_dim,
        domains=list(legacy.action_dims),
        latent_dim=legacy.latent_dim,
        condition_dim=legacy.condition_dim,
        denoiser_hidden_dim=legacy.denoiser_hidden_dim,
        num_inference_steps=legacy.num_inference_steps,
        sampling_schedule=legacy.sampling_schedule,
        gradient_checkpointing=legacy.gradient_checkpointing,
        gradient_accumulation_steps=legacy.gradient_accumulation_steps,
        schedule_anchor_domain=legacy.schedule_anchor_domain,
    ).eval()
    split_state = split_sampler.state_dict()
    legacy_state = legacy.state_dict()
    split_sampler.load_state_dict({key: legacy_state[key] for key in split_state})
    split_decoder = PerEmbodimentActionDecoder(
        decoders={
            domain: copy.deepcopy(legacy.decoder(domain))
            for domain in legacy.action_dims
        }
    ).eval()

    seeded = noise_stage(_batch(target=False))
    legacy_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in seeded.items()
    }
    split_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in seeded.items()
    }
    legacy_out = legacy(legacy_batch)
    split_out = split_decoder(split_sampler(split_batch))
    assert torch.equal(
        legacy_out["sampler/endpoint"], split_out["sampler/endpoint"]
    )
    assert torch.equal(legacy_out["pred_action"], split_out["pred_action"])
    for key in (
        "log/sampler_noise_rms",
        "log/sampler_endpoint_rms",
        "log/sampler_prediction_rms",
    ):
        assert torch.equal(legacy_out[key], split_out[key])


def test_temporal_decoder_is_horizon_free_and_always_doubles_tokens():
    decoder = TemporalConvActionDecoder(latent_dim=8, hidden_dim=16, action_dim=4)
    for input_tokens in (3, 8, 11):
        latent = torch.randn(2, input_tokens, 8)
        assert decoder(latent).shape == (2, 2 * input_tokens, 4)
        assert decoder.output_num_tokens(input_tokens) == 2 * input_tokens


def test_sampler_rejects_denoiser_positional_horizon_mismatch():
    field = CrossTransformer(
        nblocks=1,
        cond_dim=12,
        hidden_dim=32,
        act_dim=8,
        act_seq=7,
        n_heads=4,
        dropout=0.0,
        mlp_layers=1,
        mlp_ratio=2,
        time_conditioning="additive",
    )
    try:
        sampler = LatentFlowSampler(
            denoising_module=field,
            condition_input_dim=14,
            condition_dim=12,
            domains=["eva_bimanual"],
            latent_dim=8,
            denoiser_hidden_dim=32,
            schedule_anchor_domain="eva_bimanual",
        )
        sampler(
            {
                "condition": torch.randn(2, 14),
                "sampler/noise": torch.randn(2, 8, 8),
                "embodiment": "eva_bimanual",
            }
        )
    except ValueError as exc:
        assert "positional horizon" in str(exc)
    else:
        raise AssertionError("denoiser positional horizon mismatch was not rejected")


def test_gaussian_noise_depends_only_on_condition_shape():
    noise, _, _ = _nodes()
    out = noise({"condition": torch.randn(4, 14)})
    assert out["sampler/noise"].shape == (4, 7, 8)


def test_gaussian_noise_legacy_horizon_alias_is_read_only_and_unambiguous():
    legacy = GaussianLatentNoise(action_horizon=5, latent_dim=3)
    assert legacy.num_tokens == legacy.action_horizon == 5
    try:
        legacy.action_horizon = 6
    except AttributeError:
        pass
    else:
        raise AssertionError("legacy action_horizon alias must be read-only")
    try:
        GaussianLatentNoise(num_tokens=5, action_horizon=5, latent_dim=3)
    except ValueError as exc:
        assert "not both" in str(exc)
    else:
        raise AssertionError("dual token-count arguments were not rejected")


def test_sampler_node_counts_optimizer_steps_not_domain_forwards():
    noise, sampler, _ = _nodes()
    noise.train()
    sampler.train()
    for expected in (1, 1, 2, 2):
        for embodiment in ("eva_bimanual", "human_bimanual"):
            out = sampler(noise(_batch(embodiment, target=False)))
            assert out["log/optimizer_step"] == expected


def test_sampling_schedule_uses_start_step_and_exact_split():
    _, sampler, _ = _nodes()
    assert [sampler.unroll_steps_at(step) for step in range(1, 5)] == [1, 2, 1, 2]
    assert [sampler.unroll_steps_at(step) for step in range(5, 9)] == [2, 2, 2, 4]


def test_training_grid_is_sorted_uniform_breakpoints():
    _, sampler, _ = _nodes()
    reference = torch.zeros(2, 7, 8)
    generator = torch.Generator().manual_seed(17)
    actual = sampler.sample_step_sizes(2, 4, reference, generator=generator)
    replay = torch.Generator().manual_seed(17)
    internal = torch.rand(2, 3, dtype=torch.float64, generator=replay).sort(-1).values
    endpoints = torch.cat(
        (
            torch.zeros(2, 1, dtype=torch.float64),
            internal,
            torch.ones(2, 1, dtype=torch.float64),
        ),
        dim=-1,
    )
    expected = endpoints.diff(dim=-1).to(reference)
    assert torch.equal(actual, expected)
    assert torch.all(actual > 0)
    assert torch.allclose(actual.sum(-1), torch.ones(2))


def test_bfloat16_training_grid_passes_integration_validation():
    _, sampler, _ = _nodes()
    noise = torch.randn(2, 7, 8, dtype=torch.bfloat16)
    condition = torch.randn(2, 14, dtype=torch.bfloat16)
    grid = sampler.sample_step_sizes(2, 4, noise)
    assert grid.dtype == torch.float32
    assert torch.all(grid > 0)
    # Isolate the integration-grid contract from denoiser parameter dtype.
    sampler._velocity = lambda latent, time, cond: torch.zeros_like(latent)
    endpoint = sampler.integrate(noise, condition, num_steps=4, step_sizes=grid)
    assert endpoint.shape == noise.shape


def test_bfloat16_grid_does_not_round_small_positive_gaps_to_zero():
    _, sampler, _ = _nodes()
    noise = torch.randn(1, 7, 8, dtype=torch.bfloat16)
    # These are valid sorted-uniform gaps, but the second gap disappears if
    # endpoints or accumulated time are represented in bf16 near one.
    grid = torch.tensor([[0.9990, 0.0001, 0.0004, 0.0005]], dtype=torch.float32)
    sampler._velocity = lambda latent, time, cond: torch.zeros_like(latent)
    endpoint = sampler.integrate(
        noise,
        torch.randn(1, 14, dtype=torch.bfloat16),
        num_steps=4,
        step_sizes=grid,
    )
    assert endpoint.shape == noise.shape
    assert torch.all(sampler.last_integration_step_sizes > 0)


def test_rollout_plan_explicitly_excludes_train_only_loss_node():
    noise, sampler, loss = _nodes()
    pipe = Pipeline([noise, sampler, loss]).eval()
    runnable, excluded = pipe.plan(["condition", "embodiment"], mode="rollout")
    assert runnable == [noise, sampler]
    assert excluded == [(loss, ["<train-only>"])]
    out = _batch("human_bimanual", target=False)
    for stage in runnable:
        out = stage(out)
    assert out["pred_action"].shape == (3, 7, 6)
    assert "loss/native_action" not in out


def test_loss_node_rejects_native_shape_mismatch():
    loss = NativeActionMSELoss()
    try:
        loss({"pred_action": torch.zeros(2, 7, 4), "target": torch.zeros(2, 7, 6)})
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("shape mismatch was not rejected")
