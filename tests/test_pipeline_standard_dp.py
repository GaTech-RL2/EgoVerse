import inspect
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from egomimic.models.ddim_scheduler import DDIMScheduler
from egomimic.models.denoising_nets import ConditionalUnet1D
from egomimic.models.diffusion_policy import DiffusionPolicy
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_diffusion import (
    DiffusionDenoiserStage,
    DiffusionEpsilonLossStage,
    DiffusionNoisingStage,
)
from egomimic.pipeline.stages_io import ActionTargetBuilder
from egomimic.pipeline.stages_sampler import FusedObsEncoder


def _tiny_scheduler(prediction_type: str = "epsilon", timesteps: int = 8):
    return DDIMScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type=prediction_type,
    )


def _tiny_policy(action_dim: int, prediction_type: str = "epsilon"):
    return DiffusionPolicy(
        model=ConditionalUnet1D(
            input_dim=action_dim,
            cond_dim=5,
            ac_latent_seq=1,
            diffusion_step_embed_dim=16,
            down_dims=[8, 16],
            kernel_size=3,
            n_groups=4,
            cond_predict_scale=True,
        ),
        noise_scheduler=_tiny_scheduler(prediction_type),
        action_horizon=4,
        num_inference_steps=2,
    )


def _factorized_stages(action_dim: int = 6):
    return (
        ActionTargetBuilder(),
        DiffusionNoisingStage(
            noise_scheduler=_tiny_scheduler(),
            action_dim=action_dim,
            action_horizon=4,
        ),
        DiffusionDenoiserStage(
            policy=_tiny_policy(action_dim),
            action_horizon=4,
            action_dim=action_dim,
            condition_input_dim=5,
        ),
        DiffusionEpsilonLossStage(),
    )


def test_factorized_training_uses_epsilon_loss_and_backpropagates():
    stages = _factorized_stages()
    output = {
        "condition": torch.randn(2, 5),
        "actions": torch.randn(2, 4, 6),
    }
    for stage in stages:
        output = stage.train()(output)

    assert output["diffusion/noisy_action"].shape == (2, 4, 6)
    assert output["diffusion/timestep"].shape == (2,)
    assert output["diffusion/predicted_noise"].shape == (2, 4, 6)
    assert output["loss/diffusion_noise"].ndim == 0
    assert torch.isfinite(output["loss/diffusion_noise"])
    output["loss/diffusion_noise"].backward()
    denoiser = stages[2]
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in denoiser.parameters()
    )


def test_factorized_training_matches_direct_scheduler_and_model_computation():
    policy = _tiny_policy(6)
    reference_policy = deepcopy(policy)
    target = torch.randn(2, 4, 6)
    condition = torch.randn(2, 5)

    torch.manual_seed(123)
    expected_noise = torch.randn(target.shape, device=target.device)
    expected_timesteps = torch.randint(
        0,
        reference_policy.noise_scheduler.num_train_timesteps,
        (target.shape[0],),
        device=target.device,
    ).long()
    expected_noisy_action = reference_policy.noise_scheduler.add_noise(
        target, expected_noise, expected_timesteps
    )
    expected_prediction = reference_policy.model(
        expected_noisy_action, expected_timesteps, condition
    )
    expected_loss = F.mse_loss(expected_prediction, expected_noise)

    stages = (
        DiffusionNoisingStage(
            noise_scheduler=_tiny_scheduler(),
            action_dim=6,
            action_horizon=4,
        ),
        DiffusionDenoiserStage(
            policy=policy,
            action_horizon=4,
            action_dim=6,
            condition_input_dim=5,
        ),
        DiffusionEpsilonLossStage(),
    )
    torch.manual_seed(123)
    output = {"condition": condition, "target": target}
    for stage in stages:
        output = stage.train()(output)

    assert torch.equal(output["diffusion/predicted_noise"], expected_prediction)
    assert torch.equal(output["diffusion/noise_target"], expected_noise)
    assert torch.equal(output["loss/diffusion_noise"], expected_loss)


@pytest.mark.parametrize("width", [6, 4])
def test_denoiser_inference_samples_the_configured_width(width):
    stage = DiffusionDenoiserStage(
        policy=_tiny_policy(width),
        action_horizon=4,
        action_dim=width,
        condition_input_dim=5,
    )
    with torch.inference_mode():
        output = stage.execute({"condition": torch.randn(2, 5)}, mode="inference")

    assert output["pred_action"].shape == (2, 4, width)
    assert torch.isfinite(output["pred_action"]).all()
    assert output["log/diffusion_inference_steps"] == 2.0


def test_nodes_reject_wrong_objective_width_and_scheduler_pair():
    with pytest.raises(ValueError, match="epsilon noise objective"):
        DiffusionNoisingStage(
            noise_scheduler=_tiny_scheduler("sample"),
            action_dim=6,
            action_horizon=4,
        )

    target, noising, denoiser, _ = _factorized_stages(4)
    with pytest.raises(ValueError, match="must be"):
        noising(
            target(
                {
                    "actions": torch.randn(2, 4, 6),
                }
            )
        )

    noising = DiffusionNoisingStage(
        noise_scheduler=_tiny_scheduler(timesteps=7),
        action_dim=6,
        action_horizon=4,
    )
    mismatched = noising({"target": torch.randn(2, 4, 6)})
    with pytest.raises(ValueError, match="schedulers differ"):
        denoiser = DiffusionDenoiserStage(
            policy=_tiny_policy(6),
            action_horizon=4,
            action_dim=6,
            condition_input_dim=5,
        )
        denoiser({**mismatched, "condition": torch.randn(2, 5)})


def test_diffusion_pipeline_contracts_plan_factorized_train_and_inference():
    encoder = FusedObsEncoder(
        torch.nn.Identity(),
        inputs={"state": "state"},
        n_obs_steps=1,
    )
    target, noising, denoiser, loss = _factorized_stages()
    pipeline = Pipeline([encoder, target, noising, denoiser, loss])

    assert encoder.contract("train") == (
        ("state",),
        ("condition",),
    )
    assert denoiser.contract("inference") == (
        ("condition",),
        ("pred_action", "log/*"),
    )
    train, train_excluded = pipeline.plan(["state", "actions"], mode="train")
    inference, inference_excluded = pipeline.plan(["state"], mode="inference")
    assert train == [encoder, target, noising, denoiser, loss]
    assert train_excluded == []
    assert inference == [encoder, denoiser]
    assert inference_excluded == [
        (target, ["<train-only>"]),
        (noising, ["<train-only>"]),
        (loss, ["<train-only>"]),
    ]


def test_diffusion_pipeline_mode_controls_topology_not_module_training_flag():
    pipeline = Pipeline(list(_factorized_stages()))
    initial_keys = tuple(pipeline.state_dict())

    pipeline.eval()
    training = pipeline.execute(
        {
            "condition": torch.randn(2, 5),
            "actions": torch.randn(2, 4, 6),
        },
        mode="train",
    )
    assert "loss/diffusion_noise" in training

    pipeline.train()
    inference = pipeline.execute(
        {"condition": torch.randn(2, 5)},
        mode="inference",
    )
    assert inference["pred_action"].shape == (2, 4, 6)
    assert "target" not in inference
    assert tuple(pipeline.state_dict()) == initial_keys


def test_generic_diffusion_nodes_ignore_unrelated_route_metadata():
    target = torch.randn(2, 4, 6)
    condition = torch.randn(2, 5)
    first = _factorized_stages()
    second = deepcopy(first)

    torch.manual_seed(91)
    first_output = {"target": target, "condition": condition, "selector": "first"}
    for stage in first[1:]:
        first_output = stage.train()(first_output)

    torch.manual_seed(91)
    second_output = {
        "target": target,
        "condition": condition,
        "selector": "arbitrary-unused-metadata",
    }
    for stage in second[1:]:
        second_output = stage.train()(second_output)

    assert torch.equal(
        first_output["loss/diffusion_noise"],
        second_output["loss/diffusion_noise"],
    )


def test_generic_denoiser_uses_explicit_width_without_a_route_map():
    policy = _tiny_policy(6)
    assert "infer_ac_dims" not in inspect.signature(DiffusionPolicy).parameters
    assert "infer_ac_dims" not in vars(policy)
    stage = DiffusionDenoiserStage(
        policy=policy,
        action_horizon=4,
        action_dim=6,
        condition_input_dim=5,
    )

    output = stage.execute({"condition": torch.randn(2, 5)}, mode="inference")

    assert output["pred_action"].shape == (2, 4, 6)


def test_generic_denoiser_uses_singular_policy_state_prefix():
    stage = DiffusionDenoiserStage(
        policy=_tiny_policy(6),
        action_horizon=4,
        action_dim=6,
        condition_input_dim=5,
    )
    keys = set(stage.state_dict())
    assert keys
    assert all(key.startswith("policy.") for key in keys)


def test_unet_has_one_conditioning_mechanism():
    assert "feature_concatenate" not in inspect.signature(ConditionalUnet1D).parameters
