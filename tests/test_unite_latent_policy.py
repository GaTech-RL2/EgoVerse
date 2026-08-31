from pathlib import Path

import torch

from egomimic.models.denoising_nets import CrossTransformer
from egomimic.pipeline.core import Pipeline, sum_losses
from egomimic.pipeline.stages_sampler import (
    GaussianLatentNoise,
    TokenwiseMLPActionDecoder,
)
from egomimic.pipeline.stages_unite import (
    UniteGenerativeEncoder,
    UniteLatentPolicy,
    UniteObjective,
)

DOMAINS = {
    "pushshapes_sim_u_socket": 4,
    "pushshapes_sim_chain_gripper": 6,
}


def _generative_encoder() -> UniteGenerativeEncoder:
    denoiser = CrossTransformer(
        nblocks=2,
        cond_dim=12,
        hidden_dim=32,
        act_dim=8,
        act_seq=4,
        n_heads=4,
        dropout=0.0,
        mlp_layers=2,
        mlp_ratio=2,
        time_conditioning="additive",
    )
    return UniteGenerativeEncoder(
        denoising_module=denoiser,
        action_dims=DOMAINS,
        condition_input_dim=14,
        latent_dim=8,
        condition_dim=12,
        denoiser_hidden_dim=32,
        gradient_checkpointing=False,
    )


def _policy(num_inference_steps: int = 3) -> UniteLatentPolicy:
    return UniteLatentPolicy(
        generative_encoder=_generative_encoder(),
        decoders={
            domain: TokenwiseMLPActionDecoder(8, 16, action_dim)
            for domain, action_dim in DOMAINS.items()
        },
        num_inference_steps=num_inference_steps,
        reconstruction_noise_std=0.1,
    )


def _batch(domain: str = "pushshapes_sim_u_socket", include_target: bool = True):
    batch = {
        "condition": torch.randn(2, 14),
        "sampler/noise": torch.randn(2, 4, 8),
        "embodiment": domain,
    }
    if include_target:
        batch["target"] = torch.randn(2, 4, DOMAINS[domain])
    return batch


def test_unite_training_runs_two_shared_encoder_passes_and_backpropagates():
    policy = _policy().train()
    objective = UniteObjective()
    shared_calls = []
    hook = policy.generative_encoder.denoising_module.register_forward_hook(
        lambda *args: shared_calls.append(1)
    )

    out = objective(policy(_batch()))
    hook.remove()

    assert len(shared_calls) == 2
    assert out["unite/clean_latent"].shape == (2, 4, 8)
    assert out["unite/predicted_clean_latent"].shape == (2, 4, 8)
    assert out["unite/reconstructed_action"].shape == (2, 4, 4)
    assert out["pred_action"].shape == (2, 4, 4)
    assert set(key for key in out if key.startswith("loss/")) == {
        "loss/unite_reconstruction",
        "loss/unite_latent",
    }

    sum_losses(out).backward()
    encoder = policy.generative_encoder
    decoder = policy.action_decoder.decoder_for("pushshapes_sim_u_socket")
    assert encoder.action_projections["pushshapes_sim_u_socket"].weight.grad is not None
    assert encoder.denoising_module.proj_u.weight.grad is not None
    assert decoder[-1].weight.grad is not None


def test_unite_latent_loss_stops_gradient_into_clean_target_branch():
    policy = _policy().train()
    out = policy(_batch())
    encoder = policy.generative_encoder
    latent_loss = (
        (out["unite/predicted_clean_latent"] - out["unite/clean_latent"].detach())
        .square()
        .mean()
    )
    latent_loss.backward()

    # The action projection and tokenization-only condition occur exclusively
    # before the detached clean target. Shared denoiser weights still learn
    # through the noisy-latent pass.
    assert encoder.action_projections["pushshapes_sim_u_socket"].weight.grad is None
    assert encoder.tokenization_conditions["pushshapes_sim_u_socket"].grad is None
    assert encoder.denoising_module.proj_u.weight.grad is not None


def test_unite_shared_parameter_set_has_finite_two_objective_gradients():
    policy = _policy().train()
    out = UniteObjective()(policy(_batch()))
    named = policy.shared_reconstruction_denoising_named_parameters(
        ["pushshapes_sim_u_socket"]
    )
    names = {name for name, _ in named}
    parameters = tuple(parameter for _, parameter in named)

    assert any(name.startswith("denoising_module.") for name in names)
    assert any(name.startswith("output_norm.") for name in names)
    assert "domain_embeddings.pushshapes_sim_u_socket" in names
    assert not any(name.startswith("action_projections.") for name in names)
    assert not any(name.startswith("condition_projection.") for name in names)

    reconstruction_gradients = torch.autograd.grad(
        out["loss/unite_reconstruction"], parameters, retain_graph=True
    )
    denoising_gradients = torch.autograd.grad(
        out["loss/unite_latent"], parameters, retain_graph=True
    )
    for gradient in (*reconstruction_gradients, *denoising_gradients):
        assert torch.isfinite(gradient).all()


def test_unite_update_schedule_is_exactly_fourteen_flow_then_one_reconstruction():
    from egomimic.pl_utils.pl_model import ModelWrapper

    reconstruction = torch.tensor(2.0)
    flow = torch.tensor(3.0)
    modes = []
    selected = []
    positions = []
    for step in range(30):
        loss, mode, position = ModelWrapper._select_unite_update_loss(
            reconstruction, flow, step, 14
        )
        modes.append(mode)
        selected.append(float(loss))
        positions.append(position)

    assert modes[:15] == ["flow"] * 14 + ["reconstruction"]
    assert modes[15:] == modes[:15]
    assert selected[:15] == [3.0] * 14 + [2.0]
    assert positions[:15] == list(range(15))


def test_unite_rollout_starts_from_noise_and_excludes_training_objective():
    noise = GaussianLatentNoise(num_tokens=4, latent_dim=8)
    policy = _policy(num_inference_steps=3).eval()
    objective = UniteObjective()
    pipeline = Pipeline([noise, policy, objective]).eval()

    runnable, excluded = pipeline.plan(["condition", "embodiment"], mode="rollout")
    assert runnable == [noise, policy]
    assert excluded == [(objective, ["<train-only>"])]

    batch = {
        "condition": torch.randn(2, 14),
        "embodiment": "pushshapes_sim_chain_gripper",
    }
    for stage in runnable:
        batch = stage(batch)
    assert batch["sampler/noise"].shape == (2, 4, 8)
    assert batch["sampler/endpoint"].shape == (2, 4, 8)
    assert batch["pred_action"].shape == (2, 4, 6)
    assert batch["log/sampler_unroll_steps"] == 3.0
    assert not any(key.startswith("loss/") for key in batch)


def test_unite_teacher_forced_eval_keeps_losses_and_uses_full_sampler():
    policy = _policy(num_inference_steps=3).eval()
    objective = UniteObjective().eval()
    out = objective(policy(_batch()))

    assert out["unite/clean_latent"].shape == (2, 4, 8)
    assert out["unite/predicted_clean_latent"].shape == (2, 4, 8)
    assert out["pred_action"].shape == (2, 4, 4)
    assert out["log/sampler_unroll_steps"] == 3.0
    assert torch.isfinite(sum_losses(out))


def test_unite_rejects_mismatched_encoder_decoder_contracts():
    try:
        UniteLatentPolicy(
            generative_encoder=_generative_encoder(),
            decoders={"pushshapes_sim_u_socket": TokenwiseMLPActionDecoder(8, 16, 4)},
        )
    except ValueError as exc:
        assert "embodiments" in str(exc)
    else:
        raise AssertionError("mismatched UNITE domains were accepted")


def test_unite_model_config_is_additive_and_uses_val01_experiment_data():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    model_path = root / (
        "egomimic/hydra_configs/model/bf/"
        "bf_pipeline_unite_usocket_chain_points_w128_transformer_cg32_"
        "per_emb_proprio_h16.yaml"
    )
    experiment_path = root / (
        "egomimic/hydra_configs/experiment/pusht/"
        "pipeline_unite_usocket_chain_newdata_val01_h16_per_emb_proprio.yaml"
    )
    model = OmegaConf.load(model_path)
    experiment = OmegaConf.load(experiment_path)

    stages = model.robomimic_model.stages
    assert [stage._target_.rsplit(".", 1)[-1] for stage in stages] == [
        "EmbodimentProprioProjection",
        "FusedObsEncoder",
        "GaussianLatentNoise",
        "UniteLatentPolicy",
        "UniteObjective",
    ]
    policy = stages[3]
    assert stages[2].latent_dim == 128
    assert policy.generative_encoder.latent_dim == 128
    assert policy.generative_encoder.denoising_module.act_dim == 128
    assert policy.num_inference_steps == 8
    assert policy.reconstruction_noise_std == 0.1
    assert stages[4].generated_action_weight == 0.0

    defaults = [str(value) for value in experiment.defaults]
    assert any("newdata_val01_h16" in value for value in defaults)
