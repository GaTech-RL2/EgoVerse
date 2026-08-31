import math

import pytest
import torch

from egomimic.models.denoising_nets import CrossTransformer
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.losses import conditional_energy_score
from egomimic.pipeline.stages_sampler import (
    ConditionalEnergyScoreLoss,
    GaussianLatentNoise,
    GroupedActionMSELoss,
    LatentEndpointGaugeLoss,
    LatentEndpointRadiusHingeLoss,
    LatentEndpointSmoothRMSCap,
    LatentFlowSampler,
    PerEmbodimentActionDecoder,
    TokenwiseMLPActionDecoder,
)


def test_u_statistic_excludes_diagonal_and_uses_k_k_minus_one():
    prediction = torch.tensor([[[[-1.0]], [[1.0]]]])
    target = torch.zeros(1, 1, 1)
    result = conditional_energy_score(prediction, target, normalize_by_dimension=False)

    assert result["attraction"].item() == pytest.approx(1.0)
    assert result["pairwise_distance"].item() == pytest.approx(2.0)
    assert result["repulsion"].item() == pytest.approx(1.0)
    assert result["score"].item() == pytest.approx(0.0)


def _explicit_energy_score(prediction, target, beta):
    scores = []
    for batch_index in range(prediction.shape[0]):
        samples = prediction[batch_index].flatten(1).float()
        truth = target[batch_index].flatten().float()
        attraction = torch.stack(
            [torch.linalg.vector_norm(sample - truth).pow(beta) for sample in samples]
        ).mean()
        ordered = []
        for first in range(samples.shape[0]):
            for second in range(samples.shape[0]):
                if first != second:
                    ordered.append(
                        torch.linalg.vector_norm(samples[first] - samples[second]).pow(
                            beta
                        )
                    )
        repulsion = torch.stack(ordered).sum() / (
            2 * samples.shape[0] * (samples.shape[0] - 1)
        )
        scores.append(attraction - repulsion)
    return torch.stack(scores).mean()


def test_matches_ordered_pair_loop_and_is_sample_permutation_invariant():
    generator = torch.Generator().manual_seed(19)
    prediction = torch.randn(2, 4, 3, 4, generator=generator)
    target = torch.randn(2, 3, 4, generator=generator)
    actual = conditional_energy_score(
        prediction, target, beta=1.3, normalize_by_dimension=False
    )["score"]
    expected = _explicit_energy_score(prediction, target, beta=1.3)
    permuted = conditional_energy_score(
        prediction[:, [2, 0, 3, 1]],
        target,
        beta=1.3,
        normalize_by_dimension=False,
    )["score"]

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(actual, permuted, atol=1e-6, rtol=1e-6)


def test_scores_complete_action_chunk_as_one_joint_event():
    prediction = torch.tensor([[[[-1.0], [-1.0]], [[1.0], [1.0]]]])
    target = torch.tensor([[[-1.0], [1.0]]])
    result = conditional_energy_score(prediction, target, normalize_by_dimension=False)

    assert result["score"].item() == pytest.approx(2.0 - math.sqrt(2.0))


def test_bimodal_candidate_beats_mean_under_energy_score_but_not_mse():
    target = torch.tensor([[[[-1.0]]], [[[1.0]]]]).squeeze(1)
    modes = torch.tensor(
        [
            [[[-1.0]], [[-1.0]], [[1.0]], [[1.0]]],
            [[[-1.0]], [[-1.0]], [[1.0]], [[1.0]]],
        ]
    )
    mean = torch.zeros_like(modes)

    mode_result = conditional_energy_score(modes, target)
    mean_result = conditional_energy_score(mean, target)

    assert mode_result["score"] < mean_result["score"]
    assert mode_result["mse"] > mean_result["mse"]


@pytest.mark.parametrize("beta", [-1.0, 0.0, 2.0, float("inf")])
def test_rejects_beta_outside_open_interval(beta):
    with pytest.raises(ValueError, match="0 < beta < 2"):
        conditional_energy_score(
            torch.zeros(1, 2, 1, 1), torch.zeros(1, 1, 1), beta=beta
        )


def test_rejects_k_less_than_two_and_event_shape_mismatch():
    with pytest.raises(ValueError, match="at least two"):
        conditional_energy_score(torch.zeros(1, 1, 2, 4), torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="event shape mismatch"):
        conditional_energy_score(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 6))


def test_fp32_accumulation_and_finite_backward_from_bfloat16_inputs():
    prediction = torch.tensor(
        [[[[0.0]], [[0.0]], [[0.25]], [[-0.25]]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    target = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
    result = conditional_energy_score(prediction, target, beta=0.5)

    for name in ("score", "attraction", "repulsion", "mse"):
        assert result[name].dtype == torch.float32
        assert torch.isfinite(result[name])
    result["score"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_exact_and_near_collapse_have_finite_gradients(beta):
    collapsed = torch.zeros(1, 4, 1, 1, requires_grad=True)
    target = torch.zeros(1, 1, 1)
    conditional_energy_score(collapsed, target, beta=beta)["score"].backward()
    assert torch.isfinite(collapsed.grad).all()

    near = torch.tensor([[[[0.0]], [[1e-4]], [[-1e-4]], [[2e-4]]]], requires_grad=True)
    conditional_energy_score(near, torch.ones_like(target), beta=beta)[
        "score"
    ].backward()
    assert torch.isfinite(near.grad).all()


@pytest.mark.parametrize("action_dim", [4, 6])
def test_action4_and_point6_are_scored_separately(action_dim):
    prediction = torch.randn(3, 4, 16, action_dim, requires_grad=True)
    target = torch.randn(3, 16, action_dim)
    score = conditional_energy_score(prediction, target)["score"]
    score.backward()

    assert score.ndim == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_pad_mask_removes_padded_timesteps_from_both_terms():
    target = torch.zeros(1, 2, 1)
    baseline = torch.tensor([[[[1.0], [0.0]], [[-1.0], [0.0]]]])
    changed_only_in_padding = baseline.clone()
    changed_only_in_padding[:, :, 1, 0] = torch.tensor([1000.0, -1000.0])
    mask = torch.tensor([[1.0, 0.0]])

    first = conditional_energy_score(baseline, target, pad_mask=mask)
    second = conditional_energy_score(changed_only_in_padding, target, pad_mask=mask)
    for name in ("score", "attraction", "repulsion", "pairwise_distance"):
        assert torch.equal(first[name], second[name])


def test_stage_optimizes_only_energy_score_and_keeps_mse_diagnostic():
    samples = torch.randn(2, 4, 3, 4, requires_grad=True)
    stage = ConditionalEnergyScoreLoss(
        beta=1.0, expected_num_samples=4, target_key="canonical_target"
    )
    out = stage(
        {
            "pred_action_samples": samples,
            "canonical_target": torch.randn(2, 3, 4),
        }
    )

    assert "loss/native_action" not in out
    assert torch.equal(sum_losses(out), out["loss/conditional_energy_score"])
    assert out["log/native_action"].requires_grad is False


def test_grouped_mse_control_optimizes_every_sample_and_logs_energy_detached():
    samples = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]], requires_grad=True)
    stage = GroupedActionMSELoss(
        beta=1.0, expected_num_samples=4, target_key="canonical_target"
    )
    out = stage(
        {
            "pred_action_samples": samples,
            "canonical_target": torch.ones(1, 1, 1),
        }
    )

    assert torch.equal(sum_losses(out), out["loss/grouped_action_mse"])
    assert out["loss/grouped_action_mse"].item() == pytest.approx(1.5)
    assert out["log/conditional_energy_score"].requires_grad is False
    out["loss/grouped_action_mse"].backward()
    assert samples.grad is not None
    assert torch.count_nonzero(samples.grad) == 3
    assert samples.grad[0, 0].item() < 0.0
    assert samples.grad[0, 2].item() > 0.0
    assert samples.grad[0, 3].item() > 0.0


def test_latent_endpoint_gauge_is_inactive_below_threshold_and_logs_fp32():
    endpoint = torch.full((2, 4, 3, 2), 2.0, dtype=torch.bfloat16, requires_grad=True)
    out = LatentEndpointGaugeLoss()({"sampler/endpoint": endpoint})

    assert out["loss/latent_endpoint_gauge"].dtype == torch.float32
    assert out["loss/latent_endpoint_gauge"].item() == 0.0
    assert out["log/latent_endpoint_m2"].item() == pytest.approx(4.0)
    assert out["log/latent_endpoint_total_rms"].item() == pytest.approx(2.0)
    assert out["log/latent_endpoint_group_mean_rms"].item() == pytest.approx(2.0)
    assert out["log/latent_endpoint_centered_within_k_rms"].item() == 0.0
    assert out["log/latent_endpoint_max_abs"].item() == pytest.approx(2.0)
    assert out["log/latent_endpoint_gauge_active"].item() == 0.0
    assert not out["log/latent_endpoint_m2"].requires_grad
    assert not out["log/latent_endpoint_total_rms"].requires_grad


def test_latent_endpoint_gauge_active_value_and_grouped_gradient_are_correct():
    endpoint = torch.full((2, 4, 3, 2), 10.0, requires_grad=True)
    stage = LatentEndpointGaugeLoss(weight=1.0e-4, second_moment_threshold=64.0)
    out = stage({"sampler/endpoint": endpoint})

    expected = torch.tensor(1.0e-4 * (100.0 - 64.0))
    torch.testing.assert_close(out["loss/latent_endpoint_gauge"], expected)
    assert out["log/latent_endpoint_gauge_active"].item() == 1.0
    assert torch.equal(sum_losses(out), out["loss/latent_endpoint_gauge"])

    out["loss/latent_endpoint_gauge"].backward()
    assert endpoint.grad is not None
    assert torch.isfinite(endpoint.grad).all()
    expected_gradient = torch.full_like(endpoint, 2.0e-4 * 10.0 / endpoint.numel())
    torch.testing.assert_close(endpoint.grad, expected_gradient)


def test_latent_endpoint_gauge_decomposes_grouped_scale_and_rank3_is_k1():
    grouped = torch.tensor([[[[1.0]], [[3.0]]]], requires_grad=True)
    grouped_out = LatentEndpointGaugeLoss()({"sampler/endpoint": grouped})

    assert grouped_out["log/latent_endpoint_total_rms"].item() == pytest.approx(
        math.sqrt(5.0)
    )
    assert grouped_out["log/latent_endpoint_group_mean_rms"].item() == pytest.approx(
        2.0
    )
    assert grouped_out[
        "log/latent_endpoint_centered_within_k_rms"
    ].item() == pytest.approx(1.0)
    assert grouped_out["log/latent_endpoint_max_abs"].item() == pytest.approx(3.0)

    rank3 = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
    rank3_out = LatentEndpointGaugeLoss()({"sampler/endpoint": rank3})
    assert torch.equal(
        rank3_out["log/latent_endpoint_group_mean_rms"],
        rank3_out["log/latent_endpoint_total_rms"],
    )
    assert rank3_out["log/latent_endpoint_centered_within_k_rms"].item() == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight": -1.0}, "weight must be finite and non-negative"),
        ({"weight": float("inf")}, "weight must be finite and non-negative"),
        (
            {"second_moment_threshold": 0.0},
            "second_moment_threshold must be finite and positive",
        ),
        (
            {"second_moment_threshold": float("nan")},
            "second_moment_threshold must be finite and positive",
        ),
    ],
)
def test_latent_endpoint_gauge_rejects_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LatentEndpointGaugeLoss(**kwargs)


def test_latent_endpoint_gauge_is_excluded_from_rollout_plan():
    stage = LatentEndpointGaugeLoss()
    runnable, excluded = Pipeline([stage]).plan(["sampler/endpoint"], mode="rollout")

    assert runnable == []
    assert excluded == [(stage, ["<train-only>"])]


def test_latent_endpoint_hinge_penalizes_each_raw_candidate_independently():
    endpoint = torch.tensor(
        [[[[3.0], [4.0]], [[10.0], [10.0]]]], requires_grad=True
    )
    stage = LatentEndpointRadiusHingeLoss(max_rms=8.0, weight=1.0e-4)
    out = stage({"sampler/endpoint": endpoint})

    # The two candidates have m2 12.5 and 100. The hinge is applied before
    # averaging candidates, so the second contributes (100 - 64) / 2.
    assert out["loss/latent_endpoint_gauge"].item() == pytest.approx(0.0018)
    assert out["log/latent_endpoint_hinge_active_fraction"].item() == 0.5
    assert out["log/latent_endpoint_hinge_excess_m2"].item() == pytest.approx(18.0)
    assert torch.equal(sum_losses(out), out["loss/latent_endpoint_gauge"])

    out["loss/latent_endpoint_gauge"].backward()
    torch.testing.assert_close(endpoint.grad[:, 0], torch.zeros_like(endpoint[:, 0]))
    torch.testing.assert_close(
        endpoint.grad[:, 1], torch.full_like(endpoint[:, 1], 5.0e-4)
    )


def test_latent_endpoint_hinge_and_cap_ignore_and_zero_padding():
    endpoint = torch.tensor(
        [[[[3.0], [4.0], [1000.0]], [[10.0], [10.0], [-1000.0]]]],
        requires_grad=True,
    )
    batch = {"sampler/endpoint": endpoint, "pad_mask": torch.tensor([[1, 1, 0]])}
    hinge_out = LatentEndpointRadiusHingeLoss(weight=1.0e-4)(dict(batch))
    cap_out = LatentEndpointSmoothRMSCap()(dict(batch))

    assert hinge_out["loss/latent_endpoint_gauge"].item() == pytest.approx(0.0018)
    stabilized = cap_out["sampler/stabilized_endpoint"]
    assert torch.equal(stabilized[:, :, 2], torch.zeros_like(stabilized[:, :, 2]))
    torch.testing.assert_close(stabilized[:, 0, :2], endpoint[:, 0, :2])
    expected_rms = 6.0 + 2.0 * math.tanh(2.0)
    actual_rms = stabilized[:, 1, :2].float().square().mean().sqrt()
    assert actual_rms.item() == pytest.approx(expected_rms)
    assert cap_out["log/latent_endpoint_candidate_rms_max"].item() == 10.0


def test_smooth_rms_cap_is_identity_through_knee_then_tanh_bounded():
    endpoint = torch.tensor(
        [[[[6.0], [6.0]], [[10.0], [10.0]]]], requires_grad=True
    )
    out = LatentEndpointSmoothRMSCap(
        soft_start_rms=6.0, max_rms=8.0
    )({"sampler/endpoint": endpoint})

    stabilized = out["sampler/stabilized_endpoint"]
    torch.testing.assert_close(stabilized[:, 0], endpoint[:, 0])
    expected_rms = 6.0 + 2.0 * math.tanh(2.0)
    candidate_rms = stabilized.float().square().mean(dim=(-2, -1)).sqrt()
    assert candidate_rms[0, 0].item() == 6.0
    assert candidate_rms[0, 1].item() == pytest.approx(expected_rms)
    assert 6.0 < candidate_rms[0, 1].item() < 8.0
    assert out["log/latent_endpoint_saturation_fraction"].item() == 0.5
    assert out["log/latent_endpoint_above_cap_fraction"].item() == 0.5
    assert out["log/latent_endpoint_stabilized_candidate_rms_max"].item() < 8.0

    stabilized[:, 1].sum().backward()
    assert endpoint.grad is not None
    assert torch.isfinite(endpoint.grad).all()
    assert torch.count_nonzero(endpoint.grad[:, 1]) == endpoint[:, 1].numel()
    assert (endpoint.grad[:, 1] > 0.0).all()


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 4, 3, 4)])
def test_smooth_rms_cap_zero_candidate_has_finite_identity_gradient(shape):
    endpoint = torch.zeros(shape, requires_grad=True)
    out = LatentEndpointSmoothRMSCap()({"sampler/endpoint": endpoint})

    stabilized = out["sampler/stabilized_endpoint"]
    torch.testing.assert_close(stabilized, endpoint)
    assert out["log/latent_endpoint_candidate_rms_max"].item() == 0.0
    assert out["log/latent_endpoint_stabilized_candidate_rms_max"].item() == 0.0

    stabilized.sum().backward()
    assert endpoint.grad is not None
    torch.testing.assert_close(endpoint.grad, torch.ones_like(endpoint))


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 4, 3, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_smooth_rms_cap_supports_rank3_rank4_and_preserves_dtype(shape, dtype):
    endpoint = torch.full(shape, 10.0, dtype=dtype)
    out = LatentEndpointSmoothRMSCap()({"sampler/endpoint": endpoint})

    stabilized = out["sampler/stabilized_endpoint"]
    candidate_rms = stabilized.float().square().mean(dim=(-2, -1)).sqrt()
    assert stabilized.shape == endpoint.shape
    assert stabilized.dtype == dtype
    assert torch.all(candidate_rms < 8.0)
    assert torch.allclose(
        candidate_rms,
        torch.full_like(candidate_rms, 6.0 + 2.0 * math.tanh(2.0)),
        atol=3.0e-2 if dtype == torch.bfloat16 else 1.0e-6,
        rtol=0.0,
    )


def test_rollout_excludes_hinge_but_runs_identical_smooth_cap():
    hinge = LatentEndpointRadiusHingeLoss(input_key="latent/raw")
    cap = LatentEndpointSmoothRMSCap(
        input_key="latent/raw",
        output_key="latent/stable",
        soft_start_rms=6.0,
        max_rms=8.0,
    )
    runnable, excluded = Pipeline([hinge, cap]).plan(["latent/raw"], mode="rollout")

    assert runnable == [cap]
    assert excluded == [(hinge, ["<train-only>"])]
    rollout_reads, rollout_writes = cap.contract("rollout")
    assert rollout_reads == ("latent/raw",)
    assert "latent/stable" in rollout_writes
    assert "loss/latent_endpoint_gauge" not in rollout_writes

    raw = torch.full((1, 3, 2), 10.0)
    cap.train()
    train_stable = cap({"latent/raw": raw.clone()})["latent/stable"]
    cap.eval()
    rollout_stable = cap({"latent/raw": raw.clone()})["latent/stable"]
    torch.testing.assert_close(rollout_stable, train_stable)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_key": ""}, "input_key must be non-empty"),
        ({"max_rms": 0.0}, "max_rms must be finite and positive"),
        ({"weight": -1.0}, "weight must be finite and non-negative"),
    ],
)
def test_latent_endpoint_hinge_rejects_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LatentEndpointRadiusHingeLoss(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_key": ""}, "input_key and output_key must be non-empty"),
        (
            {"input_key": "same", "output_key": "same"},
            "output_key must differ from input_key",
        ),
        ({"max_rms": 0.0}, "max_rms must be finite and positive"),
        (
            {"soft_start_rms": 8.0},
            "soft_start_rms must be finite, positive, and below max_rms",
        ),
        ({"eps": 0.0}, "eps must be finite and positive"),
    ],
)
def test_smooth_rms_cap_rejects_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LatentEndpointSmoothRMSCap(**kwargs)


@pytest.mark.parametrize("grouped", [False, True])
def test_decoder_logs_selected_first_linear_scale_diagnostics(grouped):
    decoder = PerEmbodimentActionDecoder(
        decoders={"u_socket": TokenwiseMLPActionDecoder(2, 8, 4)}
    )
    endpoint_shape = (2, 4, 3, 2) if grouped else (2, 3, 2)
    endpoint = torch.full(endpoint_shape, 3.0, requires_grad=True)
    out = decoder({"sampler/endpoint": endpoint, "embodiment": "u_socket"})

    first_linear = next(
        module
        for module in decoder.decoder_for("u_socket").modules()
        if isinstance(module, torch.nn.Linear)
    )
    expected_norm = torch.linalg.vector_norm(first_linear.weight.detach().float())
    actual_norm = out["log/decoder_first_linear_weight_frobenius_norm"]
    product = out["log/latent_decoder_scale_product"]
    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(product, 3.0 * expected_norm)
    assert actual_norm.dtype == torch.float32
    assert product.dtype == torch.float32
    assert not actual_norm.requires_grad
    assert not product.requires_grad


def test_decoder_accepts_a_generic_stabilized_endpoint_key():
    decoder = PerEmbodimentActionDecoder(
        decoders={"u_socket": TokenwiseMLPActionDecoder(2, 8, 4)},
        input_key="sampler/stabilized_endpoint",
    )
    endpoint = torch.randn(2, 3, 2)
    out = decoder(
        {
            "sampler/stabilized_endpoint": endpoint,
            "embodiment": "u_socket",
        }
    )

    assert out["pred_action"].shape == (2, 3, 4)
    assert decoder.contract("train")[0] == (
        "sampler/stabilized_endpoint",
        "embodiment",
    )


def test_decoder_without_linear_keeps_prior_public_contract():
    class IdentityDecoder(torch.nn.Module):
        latent_dim = 2
        action_dim = 2
        temporal_factor = 1

        @staticmethod
        def output_num_tokens(input_num_tokens):
            return int(input_num_tokens)

        @staticmethod
        def forward(latent):
            return latent

    decoder = PerEmbodimentActionDecoder(decoders={"custom": IdentityDecoder()})
    out = decoder(
        {
            "sampler/endpoint": torch.randn(2, 3, 2),
            "embodiment": "custom",
        }
    )

    assert out["pred_action"].shape == (2, 3, 2)
    assert "log/decoder_first_linear_weight_frobenius_norm" not in out
    assert "log/latent_decoder_scale_product" not in out


def _grouped_nodes(action_dim=4):
    field = CrossTransformer(
        nblocks=1,
        cond_dim=12,
        hidden_dim=16,
        act_dim=2,
        act_seq=3,
        n_heads=4,
        dropout=0.0,
        mlp_layers=2,
        mlp_ratio=2,
        time_conditioning="additive",
    )
    noise = GaussianLatentNoise(num_tokens=3, latent_dim=2, num_samples=4)
    sampler = LatentFlowSampler(
        denoising_module=field,
        condition_input_dim=5,
        condition_dim=12,
        domains=["u_socket", "chain_gripper"],
        latent_dim=2,
        denoiser_hidden_dim=16,
        num_inference_steps=2,
        sampling_schedule={1: {2: 1.0}},
        gradient_checkpointing=False,
        schedule_anchor_domain="u_socket",
    )
    decoder = PerEmbodimentActionDecoder(
        decoders={
            "u_socket": TokenwiseMLPActionDecoder(2, 8, 4),
            "chain_gripper": TokenwiseMLPActionDecoder(2, 8, 6),
        }
    )
    return noise, sampler, decoder, ConditionalEnergyScoreLoss(expected_num_samples=4)


def _grouped_batch(domain="u_socket", target=True):
    action_dim = 4 if domain == "u_socket" else 6
    batch = {"condition": torch.randn(2, 5), "embodiment": domain}
    if target:
        target_value = torch.randn(2, 3, action_dim)
        # GaussianLatentNoise uses the raw target's presence to distinguish
        # teacher-forced sampling from observation-only rollout. The loss
        # consumes the canonicalized counterpart produced later in the graph.
        batch["target"] = target_value
        batch["canonical_target"] = target_value
    return batch


def test_grouped_sampler_reuses_condition_and_integration_grid():
    noise, sampler, decoder, loss = _grouped_nodes()
    seen_conditions = []

    def zero_velocity(latent, time, condition):
        seen_conditions.append(condition.detach().clone())
        return torch.zeros_like(latent)

    sampler._velocity = zero_velocity
    out = Pipeline([noise, sampler, decoder, loss]).train()(_grouped_batch())

    assert out["sampler/noise"].shape == (2, 4, 3, 2)
    assert out["sampler/endpoint"].shape == (2, 4, 3, 2)
    assert out["pred_action_samples"].shape == (2, 4, 3, 4)
    assert out["pred_action"].shape == (2, 3, 4)
    assert not torch.equal(out["sampler/noise"][:, 0], out["sampler/noise"][:, 1])

    grids = sampler.last_integration_step_sizes.reshape(2, 4, 2)
    assert torch.equal(grids, grids[:, :1].expand_as(grids))
    for condition in seen_conditions:
        grouped = condition.reshape(2, 4, *condition.shape[1:])
        assert torch.equal(grouped, grouped[:, :1].expand_as(grouped))


def test_grouped_noise_is_target_value_independent_and_rollout_stays_rank_three():
    noise, sampler, decoder, _ = _grouped_nodes()
    first = _grouped_batch()
    second = {
        **first,
        "target": torch.randn_like(first["target"]) * 1000.0,
    }
    second["canonical_target"] = second["target"]
    torch.manual_seed(31)
    first_noise = noise(dict(first))["sampler/noise"]
    torch.manual_seed(31)
    second_noise = noise(dict(second))["sampler/noise"]
    assert torch.equal(first_noise, second_noise)

    rollout = _grouped_batch(target=False)
    noise.eval()
    sampler.eval()
    decoder.eval()
    out = decoder(sampler(noise(rollout)))
    assert out["sampler/noise"].shape == (2, 3, 2)
    assert out["pred_action"].shape == (2, 3, 4)
    assert out["pred_action_samples"].shape == (2, 1, 3, 4)


@pytest.mark.parametrize(
    ("domain", "action_dim"), [("u_socket", 4), ("chain_gripper", 6)]
)
def test_grouped_energy_backward_reaches_shared_field_and_domain_decoder(
    domain, action_dim
):
    noise, sampler, decoder, loss = _grouped_nodes(action_dim=action_dim)
    out = Pipeline([noise, sampler, decoder, loss]).train()(_grouped_batch(domain))
    out["loss/conditional_energy_score"].backward()

    assert sampler.denoising_module.proj_u.weight.grad is not None
    branch = decoder.decoder_for(domain)
    assert branch[-1].weight.grad is not None
    assert torch.isfinite(branch[-1].weight.grad).all()
