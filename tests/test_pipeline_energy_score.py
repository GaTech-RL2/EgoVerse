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
    stage = ConditionalEnergyScoreLoss(beta=1.0, expected_num_samples=4)
    out = stage(
        {
            "pred_action_samples": samples,
            "target": torch.randn(2, 3, 4),
        }
    )

    assert "loss/native_action" not in out
    assert torch.equal(sum_losses(out), out["loss/conditional_energy_score"])
    assert out["log/native_action"].requires_grad is False


def test_grouped_mse_control_optimizes_every_sample_and_logs_energy_detached():
    samples = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]]]], requires_grad=True)
    stage = GroupedActionMSELoss(beta=1.0, expected_num_samples=4)
    out = stage({"pred_action_samples": samples, "target": torch.ones(1, 1, 1)})

    assert torch.equal(sum_losses(out), out["loss/grouped_action_mse"])
    assert out["loss/grouped_action_mse"].item() == pytest.approx(1.5)
    assert out["log/conditional_energy_score"].requires_grad is False
    out["loss/grouped_action_mse"].backward()
    assert samples.grad is not None
    assert torch.count_nonzero(samples.grad) == 3
    assert samples.grad[0, 0].item() < 0.0
    assert samples.grad[0, 2].item() > 0.0
    assert samples.grad[0, 3].item() > 0.0


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
        batch["target"] = torch.randn(2, 3, action_dim)
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
    second = {**first, "target": torch.randn_like(first["target"]) * 1000.0}
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
