import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.models.unite_action_decoder import UniteActionDecoder
from egomimic.pipeline.pushshapes import (
    USocketModelStateObservationAdapter,
    USocketRotVecNativeDecoder,
)
from egomimic.pipeline.stages_sampler import KeyedFeatureProjection
from egomimic.pipeline.stages_unite_released import (
    ReleasedRecipeUniteLatentPolicy,
    ReleasedRecipeUniteObjective,
)
from egomimic.pipeline.stages_unite_separate import (
    SeparateUniteGenerativeEncoder,
    build_configurable_unite_generative_encoder,
)
from egomimic.rldb.embodiment.pushshapes import (
    get_planar_keymap_per_source_proprio,
    get_usocket_rotvec_action_state_transform_list,
)

DOMAIN = "pushshapes_sim_u_socket"
ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "egomimic/hydra_configs"


def _backbone_config(num_latent_tokens: int) -> dict:
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
        "in_context_len": 8,
        "trainable_position_embeddings": False,
        "trainable_register_position_embeddings": True,
        "trainable_content_position_embeddings": False,
        "trainable_condition_position_embeddings": False,
        "trainable_in_context_position_embeddings": True,
        "gradient_checkpointing": False,
    }


def _encoder(shared: bool, num_latent_tokens: int):
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
        in_context_len=8,
    )


def _policy(shared: bool, num_latent_tokens: int, flow_mini_batch: int = 14):
    decoder = UniteActionDecoder(
        latent_dim=16,
        action_dim=4,
        num_latent_tokens=num_latent_tokens,
        action_horizon=16,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        gradient_checkpointing=False,
    )
    return ReleasedRecipeUniteLatentPolicy(
        generative_encoder=_encoder(shared, num_latent_tokens),
        decoders={DOMAIN: decoder},
        flow_steps_per_reconstruction=14,
        flow_mini_batch=flow_mini_batch,
    )


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("num_latent_tokens", [4, 8])
def test_register_topology_and_h16_round_trip(shared, num_latent_tokens):
    policy = _policy(shared, num_latent_tokens)
    encoder = policy.generative_encoder
    assert encoder.num_latent_tokens == num_latent_tokens
    assert encoder.denoising_module.horizon == num_latent_tokens
    if shared:
        assert not isinstance(encoder, SeparateUniteGenerativeEncoder)
    else:
        assert isinstance(encoder, SeparateUniteGenerativeEncoder)
        assert encoder.tokenization_module is not encoder.denoising_module

    actions = torch.randn(2, 16, 4, requires_grad=True)
    latent = encoder.tokenize(
        actions,
        DOMAIN,
        torch.randn(2, num_latent_tokens, 16),
    )
    prediction = encoder.denoise(
        latent.detach(),
        torch.full((2,), 0.5),
        torch.randn(2, 14),
        DOMAIN,
    )
    decoded = policy._decode(prediction, DOMAIN)
    assert latent.shape == prediction.shape == (2, num_latent_tokens, 16)
    assert decoded.shape == (2, 16, 4)
    (latent.square().mean() + decoded.square().mean()).backward()
    assert actions.grad is not None


def test_action_decoder_does_not_require_horizon_register_divisibility():
    decoder = UniteActionDecoder(
        latent_dim=16,
        action_dim=4,
        num_latent_tokens=3,
        action_horizon=7,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        gradient_checkpointing=False,
    )
    assert decoder(torch.randn(2, 3, 16)).shape == (2, 7, 4)


def test_clean_content_and_in_context_tokens_enter_shared_adaln_backbone():
    encoder = _encoder(True, 4).eval()
    captured = {}
    sequence_lengths = []

    def capture_inputs(_module, _args, kwargs):
        captured["registers"] = _args[0]
        captured["condition"] = kwargs["condition"]
        captured["content"] = kwargs["content_tokens"]

    input_hook = encoder.denoising_module.register_forward_pre_hook(
        capture_inputs, with_kwargs=True
    )
    block_hooks = [
        block.register_forward_pre_hook(
            lambda _module, args: sequence_lengths.append(int(args[0].shape[1]))
        )
        for block in encoder.denoising_module.blocks
    ]
    registers = torch.randn(2, 4, 16)
    encoder.tokenize(torch.randn(2, 16, 4), DOMAIN, registers)
    input_hook.remove()
    for hook in block_hooks:
        hook.remove()

    assert captured["condition"].shape == (2, 1, 12)
    assert captured["content"].shape == (2, 16, 12)
    torch.testing.assert_close(captured["registers"], registers)
    assert sequence_lengths == [20, 28]
    assert all(
        hasattr(block, "adaln_modulation") for block in encoder.denoising_module.blocks
    )
    assert (
        "CrossTransformer"
        not in Path(
            encoder.denoising_module.__class__.__module__.replace(".", "/")
        ).name
    )


@pytest.mark.parametrize("shared", [True, False])
def test_joint_step_uses_one_reconstruction_and_fourteen_flow_samples(shared):
    policy = _policy(shared, 4, flow_mini_batch=4).train()
    calls = []
    modules = [policy.generative_encoder.denoising_module]
    if not shared:
        modules.append(policy.generative_encoder.tokenization_module)
    hooks = [
        module.register_forward_hook(
            lambda _module, args, _output: calls.append(int(args[0].shape[0]))
        )
        for module in modules
    ]
    batch = {
        "target": torch.randn(2, 16, 4),
        "condition": torch.randn(2, 14),
        "sampler/noise": torch.randn(2, 4, 16),
        "embodiment": torch.tensor([19, 19]),
    }
    output = ReleasedRecipeUniteObjective()(policy(batch))
    for hook in hooks:
        hook.remove()

    assert policy._flow_chunks(14, 4) == (4, 4, 4, 2)
    assert sorted(calls) == sorted([2, 8, 8, 8, 4])
    total = output["loss/unite_reconstruction"] + output["loss/unite_latent"]
    assert total.ndim == 0 and bool(torch.isfinite(total))
    assert "sampler/endpoint" not in output
    assert "pred_action" not in output
    total.backward()


def test_separate_flow_target_is_stop_gradient():
    policy = _policy(False, 4).train()
    output = policy(
        {
            "target": torch.randn(2, 16, 4),
            "condition": torch.randn(2, 14),
            "sampler/noise": torch.randn(2, 4, 16),
            "embodiment": torch.tensor([19, 19]),
        }
    )
    output["unite/flow_loss"].backward()
    encoder = policy.generative_encoder
    assert all(
        parameter.grad is None for parameter in encoder.tokenization_module.parameters()
    )
    assert all(parameter.grad is None for parameter in encoder.output_norm.parameters())
    assert any(
        parameter.grad is not None
        for parameter in encoder.denoising_module.parameters()
    )


def test_unite_graph_execution_mode_is_explicit(monkeypatch):
    policy = _policy(True, 4).train()
    monkeypatch.setattr(
        policy,
        "sample",
        lambda noise, _condition, _embodiment: torch.zeros_like(noise),
    )
    monkeypatch.setattr(
        policy.generative_encoder,
        "tokenize",
        lambda *_args, **_kwargs: pytest.fail("inference called the training path"),
    )
    output = policy.execute(
        {
            "target": torch.randn(2, 16, 4),
            "condition": torch.randn(2, 14),
            "sampler/noise": torch.randn(2, 4, 16),
            "embodiment": torch.tensor([19, 19]),
        },
        mode="inference",
    )
    assert output["pred_action"].shape == (2, 16, 4)
    assert "unite/flow_loss" not in output


def test_dopri5_has_one_fp32_integrator_with_released_grid(monkeypatch):
    policy = _policy(True, 4).eval()
    captured = {}

    monkeypatch.setattr(
        policy,
        "_guided_clean_prediction",
        lambda latent, _time, _condition, _scale, _embodiment: torch.zeros_like(
            latent, dtype=torch.bfloat16
        ),
    )

    def fake_odeint(function, initial, times, *, method, atol, rtol):
        derivative = function(times[0], initial)
        captured.update(
            initial_dtype=initial.dtype,
            derivative_dtype=derivative.dtype,
            times=times.detach().clone(),
            method=method,
            atol=atol,
            rtol=rtol,
        )
        return torch.stack([initial for _ in times])

    monkeypatch.setitem(sys.modules, "torchdiffeq", SimpleNamespace(odeint=fake_odeint))
    endpoint = policy.sample(
        torch.randn(2, 4, 16, dtype=torch.bfloat16),
        torch.randn(2, 14, dtype=torch.bfloat16),
        DOMAIN,
    )
    assert endpoint.shape == (2, 4, 16)
    assert endpoint.dtype == torch.float32
    assert captured["initial_dtype"] == captured["derivative_dtype"] == torch.float32
    assert captured["method"] == "dopri5"
    assert captured["times"].shape == (50,)
    assert captured["times"][0] == 0.0 and captured["times"][-1] == 1.0
    assert captured["atol"] == 1.0e-6 and captured["rtol"] == 1.0e-3
    assert policy.dopri5_output_points == 50


def test_released_unite_diagnostics_capture_dopri5_and_shared_blocks(monkeypatch):
    policy = _policy(True, 4).eval()

    def fake_odeint(function, initial, times, *, method, atol, rtol):
        assert method == "dopri5"
        assert atol == 1.0e-6 and rtol == 1.0e-3
        assert bool(torch.isfinite(function(times[1], initial)).all())
        return torch.stack([initial + float(index) / 100.0 for index in range(len(times))])

    monkeypatch.setitem(sys.modules, "torchdiffeq", SimpleNamespace(odeint=fake_odeint))
    diagnostics = policy.validation_diagnostics(
        noise=torch.randn(12, 4, 16),
        condition=torch.randn(12, 14),
        target=torch.randn(12, 16, 4),
        embodiment=DOMAIN,
        raw_noise_levels=[0.0, 0.5, 1.0],
    )
    assert diagnostics["sampler_latents"].shape == (50, 12, 4, 16)
    assert diagnostics["decoded_actions_normalized"].shape == (50, 12, 16, 4)
    assert diagnostics["noise_level_final_predictions"].shape == (3, 12, 4, 16)
    assert set(diagnostics["tokenization_activations"]) == {"block_00", "block_01"}
    for layer in diagnostics["tokenization_activations"]:
        assert diagnostics["tokenization_activations"][layer].shape[0] == 12
        assert diagnostics["denoising_activations"][layer].shape[0] == 3
        assert not policy.generative_encoder.denoising_module.blocks[
            int(layer.rsplit("_", 1)[1])
        ]._forward_hooks
    assert bool(torch.isfinite(diagnostics["sampler_latents"]).all())


def test_cfg_threads_the_dataset_selected_branch_to_encoder(monkeypatch):
    policy = _policy(True, 4).eval()
    selected = []

    def null_condition(condition, embodiment):
        selected.append(("null", embodiment))
        return torch.zeros_like(condition)

    def denoise(latent, _time, _condition, embodiment):
        selected.append(("denoise", embodiment))
        return torch.zeros_like(latent)

    monkeypatch.setattr(
        policy.generative_encoder, "null_condition_like", null_condition
    )
    monkeypatch.setattr(policy.generative_encoder, "denoise", denoise)
    policy._guided_clean_prediction(
        torch.randn(2, 4, 16),
        torch.full((2,), 0.5),
        torch.randn(2, 14),
        4.0,
        DOMAIN,
    )
    assert selected == [("null", DOMAIN), ("denoise", DOMAIN)]


@pytest.mark.parametrize(
    ("topology", "num_latent_tokens"),
    [("shared", 4), ("shared", 8), ("separate", 4), ("separate", 8)],
)
def test_four_rows_are_thin_overlays_of_one_resolved_base(topology, num_latent_tokens):
    model_dir = CONFIG_DIR / "model/bf"
    canonical_path = model_dir / "us_unite_register_shared_nt4_s42.yaml"
    canonical = OmegaConf.load(canonical_path)
    row_path = (
        model_dir / f"us_unite_register_{topology}_nt{num_latent_tokens}_s42.yaml"
    )
    if row_path == canonical_path:
        row = canonical
        assert "defaults" not in row
    else:
        overlay = OmegaConf.load(row_path)
        assert set(overlay) <= {
            "defaults",
            "share_encoder_denoiser",
            "num_latent_tokens",
        }
        assert set(overlay) - {"defaults"}
        del overlay["defaults"]
        row = OmegaConf.merge(canonical, overlay)
    root = OmegaConf.create({"model": row})
    OmegaConf.resolve(root)
    model = root.model
    assert model.share_encoder_denoiser is (topology == "shared")
    assert model.num_latent_tokens == num_latent_tokens
    stages = model.pipeline.stages
    assert not {
        "action_horizon",
        "domains",
        "ac_keys",
        "rollout_adapter",
        "rollout_adapters",
        "rollout_observation_adapters",
        "norm_stats",
    } & set(model.pipeline)
    assert [stage._target_.rsplit(".", 1)[-1] for stage in stages[:4]] == [
        "KeyedFeatureProjection",
        "FusedObsEncoder",
        "ActionTargetBuilder",
        "GaussianLatentNoise",
    ]
    assert stages[3].num_tokens == num_latent_tokens
    ge = stages[4].generative_encoder
    assert ge.share_encoder_denoiser is (topology == "shared")
    assert ge.num_latent_tokens == ge.backbone_config.horizon == num_latent_tokens
    assert stages[4].flow_mini_batch == 4
    assert ge.gradient_checkpointing is True
    assert ge.backbone_config.gradient_checkpointing is True
    assert stages[4].decoders[DOMAIN].action_horizon == 16
    assert stages[4].decoders[DOMAIN].gradient_checkpointing is True


@pytest.mark.parametrize("topology", ["shared", "separate"])
@pytest.mark.parametrize("num_latent_tokens", [4, 8])
def test_four_rows_compose_with_supported_data_and_evaluator_schema(
    topology, num_latent_tokens
):
    row = f"us_unite_register_{topology}_nt{num_latent_tokens}_s42"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                f"model=bf/{row}",
                "+experiment=pusht/unite_usocket_register_sweep_val01_h16",
                "++paths.root_dir=.",
            ],
        )
    OmegaConf.resolve(cfg.model)
    OmegaConf.resolve(cfg.data)
    assert cfg.model.share_encoder_denoiser is (topology == "shared")
    assert cfg.model.num_latent_tokens == num_latent_tokens
    for section in (cfg.data, cfg.evaluator):
        target = get_class(section._target_)
        accepted = set(inspect.signature(target.__init__).parameters)
        assert set(section) - accepted - {"_target_"} == set()


def test_only_registered_rows_and_no_legacy_or_diagnostic_surface():
    model_dir = CONFIG_DIR / "model/bf"
    rows = sorted(path.name for path in model_dir.glob("us_unite_register_*_s42.yaml"))
    assert rows == [
        "us_unite_register_separate_nt4_s42.yaml",
        "us_unite_register_separate_nt8_s42.yaml",
        "us_unite_register_shared_nt16_s42.yaml",
        "us_unite_register_shared_nt4_s42.yaml",
        "us_unite_register_shared_nt8_flow42_s42.yaml",
        "us_unite_register_shared_nt8_s42.yaml",
    ]
    assert (
        sorted(path.name for path in model_dir.glob("us_unite_register*.yaml")) == rows
    )


def test_ice_launcher_explicitly_wires_unite_diagnostics():
    launcher = (ROOT / "scripts/ice/launch_unite.sbatch").read_text()
    assert "ICE_UNITE_DIAGNOSTICS=${ICE_UNITE_DIAGNOSTICS:-false}" in launcher
    assert "evaluator.unite_diagnostics.enabled=true" in launcher
    assert (
        "evaluator.unite_diagnostics.validation_view.split_manifest_sha256="
        "$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" in launcher
    )
    assert '"$ICE_EXPECTED_SPLIT_MANIFEST_SHA256" "$ICE_UNITE_DIAGNOSTICS"' in launcher
    assert "launch_contract.json" in launcher
    assert '"world_size": int(world_size)' in launcher
    assert '"unite_diagnostics": unite_diagnostics' in launcher
    assert "ICE_WORLD_SIZE=1" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --constraint=H100|H200" in launcher
    assert "ICE_UNITE_FAST=${ICE_UNITE_FAST:-false}" in launcher
    assert "ICE_UNITE_H100_FAST" not in launcher
    assert "ICE_EXPECTED_GPU_NAME" not in launcher
    assert "bf/us_unite_register_shared_nt8_flow42_s42" in launcher
    assert "+runtime.slurm_requeue_owner" not in launcher
    assert "runtime.slurm_requeue_owner=runner" in launcher
    assert '"trainer.devices=$ICE_WORLD_SIZE"' in launcher
    assert "export ICE_WORLD_SIZE ICE_TRAINER_STRATEGY" in launcher
    assert '"evaluator.unite_diagnostics.validation_view.world_size=$ICE_WORLD_SIZE"' in launcher
    assert '--ntasks="${ICE_WORLD_SIZE:?}"' in launcher
    assert 'test "$ICE_MAX_STEPS" -ge 100' in launcher
    assert 'test "$ICE_VAL_CHECK_INTERVAL" -le "$ICE_MAX_STEPS"' in launcher
    assert not (ROOT / "egomimic/pipeline/stages_unite.py").exists()
    policy_parameters = inspect.signature(
        ReleasedRecipeUniteLatentPolicy.__init__
    ).parameters
    assert not {
        "sampling_method",
        "cfg_norm_order",
        "dopri5_num_steps",
        "reconstruction_noise_std",
    } & set(policy_parameters)
    experiment = yaml.safe_load(
        (
            CONFIG_DIR / "experiment/pusht/unite_usocket_register_sweep_val01_h16.yaml"
        ).read_text()
    )
    assert experiment["trainer"]["max_steps"] == 240_000
    assert experiment["trainer"]["check_val_every_n_epoch"] is None
    assert experiment["trainer"]["gradient_clip_val"] == 3.0
    assert experiment["trainer"]["gradient_clip_algorithm"] == "norm"
    assert experiment["callbacks"]["model_checkpoint"]["every_n_train_steps"] == 20_000
    assert experiment["callbacks"]["model_checkpoint"]["every_n_epochs"] is None
    assert experiment["eval_checkpoint"] == {"use_ema": True}
    assert {"override /evaluator": "eval_planar_v2"} in experiment["defaults"]
    assert experiment["evaluator"]["energy_score_max_batches_per_rank"] == 1
    assert experiment["evaluator"]["deterministic_seed"] == 420_042
    assert experiment["evaluator"]["energy_score_validation_view"]["definition"] == (
        "first_deterministic_validation_batch_per_ddp_rank"
    )
    assert experiment["evaluator"]["energy_score_provenance"]["sampler"] == "dopri5"
    assert experiment["evaluator"]["semantic_blocks"] == [[0, 2], [2, 4]]
    assert experiment["evaluator"]["native_decoder"] == {
        "_target_": "egomimic.pipeline.pushshapes.USocketRotVecNativeDecoder"
    }
    assert set(experiment["deployment"]) == {
        "observation_adapter",
        "action_decoder",
    }
    assert experiment["run_provenance"]["energy_score_contract"]["sample_count"] == 32
    assert "train_only_normalization_sha256" not in experiment["run_provenance"]
    assert "unite_diagnostics" not in experiment["evaluator"]
    assert "unite_training_diagnostics" not in experiment["run_provenance"]


def test_unite_energy_score_is_limited_without_truncating_mse_validation(tmp_path):
    evaluator = PlanarActionEval(
        seed_bank_path=str(CONFIG_DIR / "evaluator/energy_score_seed_bank_k32_v1.json"),
        seed_bank_sha256=(
            "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
        ),
        artifact_root=str(tmp_path),
        semantic_blocks=((0, 2), (2, 4)),
        energy_score_max_batches_per_rank=1,
    )

    class IdentityNorm:
        @staticmethod
        def unnormalize(values, _embodiment_id):
            return values

    target = torch.zeros(2, 16, 4)
    evaluator.model = SimpleNamespace(
        forward_eval=lambda _batch: {19: {"pred_action": target}},
    )
    evaluator.bind_data_context(normalizer=IdentityNorm())
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        global_step=0,
        global_rank=0,
        lightning_module=SimpleNamespace(log_dict=lambda *_args, **_kwargs: None),
    )
    score_calls = []
    evaluator._seeded_predictions = lambda _batch: (
        score_calls.append(True)
        or (
            {19: target.unsqueeze(0).repeat(32, 1, 1, 1)},
            {19: {"pred_action": target}},
        )
    )
    evaluator._save_artifact = lambda *_args: None
    batch = {
        19: {
            "actions": target,
            "embodiment": torch.full((target.shape[0],), 19),
        }
    }

    evaluator.on_validation_step(batch, batch_idx=1)
    assert score_calls == []
    evaluator.on_validation_step(batch, batch_idx=0)
    assert score_calls == [True]


@pytest.mark.parametrize("as_torch", [False, True])
def test_usocket_training_and_rollout_action_contract_match(as_torch):
    raw = np.array([10.0, 20.0, -1.2, 30.0, 40.0, 0.5], dtype=np.float32)
    actions = np.tile(np.array([1.0, 2.0, -0.4], dtype=np.float32), (16, 1))
    if as_torch:
        raw, actions = torch.from_numpy(raw), torch.from_numpy(actions)
    sample = {
        "state_agent_obj": raw,
        "state_agent_model": raw.clone() if as_torch else raw.copy(),
        "actions": actions,
    }
    for transform in get_usocket_rotvec_action_state_transform_list():
        sample = transform.transform(sample)
    assert sample["state_agent_model"].shape == (4,)
    assert sample["actions"].shape == (16, 4)

    rollout_obs = USocketModelStateObservationAdapter().encode({"state_agent_obj": raw})
    assert rollout_obs["state_agent_obj"] is raw
    assert rollout_obs["state_agent_model"].shape == (4,)
    decoded = USocketRotVecNativeDecoder().decode(sample["actions"])
    if as_torch:
        torch.testing.assert_close(decoded, actions)
    else:
        np.testing.assert_allclose(decoded, actions, atol=1e-6)


def test_usocket_keymap_and_proprio_projection_are_explicit():
    keymap = get_planar_keymap_per_source_proprio(action_horizon=16)
    assert keymap["state_agent_obj"]["key_type"] == "metadata_keys"
    assert keymap["state_agent_model"]["key_type"] == "proprio_keys"
    assert keymap["actions"]["horizon"] == 16

    projection = KeyedFeatureProjection(
        selector_key="embodiment",
        selector_aliases={19: DOMAIN},
        input_key="state_agent_model",
        output_key="proprio_condition",
        projections={DOMAIN: {"input_dim": 4, "hidden_dim": 8}},
        output_dim=6,
    )
    assert projection.contract() == (
        ("state_agent_model", "embodiment"),
        ("proprio_condition",),
    )
    batch = {
        "embodiment": torch.tensor([19, 19, 19]),
        "state_agent_model": torch.randn(3, 4),
    }
    assert projection(batch)["proprio_condition"].shape == (3, 6)
    with pytest.raises(ValueError, match="expected 4"):
        projection(
            {
                "embodiment": torch.tensor([19, 19, 19]),
                "state_agent_model": torch.randn(3, 6),
            }
        )
    with pytest.raises(ValueError, match="homogeneous"):
        projection(
            {
                "embodiment": torch.tensor([19, 20, 19]),
                "state_agent_model": torch.randn(3, 4),
            }
        )
