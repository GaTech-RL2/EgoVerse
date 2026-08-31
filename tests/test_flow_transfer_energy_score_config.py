from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
EXPERIMENT = (
    "pusht/pipeline_sampler_usocket_chain_newdata_r01_l4_energy_score_val01_h16"
)
DOMAINS = ("pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper")


def _compose(experiment=EXPERIMENT, extra_overrides=()):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", *extra_overrides],
        )


def test_energy_score_model_keeps_u4_chain_point6_h16_contract():
    cfg = _compose()
    model = cfg.model.robomimic_model
    (
        projection,
        _,
        noise,
        sampler,
        hinge,
        cap,
        decoder,
        canonicalizer,
        loss,
    ) = model.stages

    assert len(model.stages) == 9
    assert list(model.domains) == list(DOMAINS)
    assert model.action_horizon == 16
    assert projection.projections[DOMAINS[0]].source_dim == 4
    assert projection.projections[DOMAINS[1]].source_dim == 6
    assert noise.num_tokens == 16
    assert noise.latent_dim == 4
    assert noise.num_samples == 4
    assert sampler.latent_dim == 4
    assert sampler.denoiser_hidden_dim == 768
    assert sampler.num_inference_steps == 8
    assert sampler.denoising_module.nblocks == 13
    assert sampler.denoising_module.hidden_dim == 768
    assert sampler.denoising_module.n_heads == 12
    assert sampler.denoising_module.mlp_layers == 2
    assert sampler.denoising_module.dropout == 0.0
    with torch.device("meta"):
        denoiser = instantiate(sampler.denoising_module)
    assert sum(parameter.numel() for parameter in denoiser.parameters()) == 249_459_460
    assert hinge._target_.endswith("LatentEndpointRadiusHingeLoss")
    assert hinge.input_key == "sampler/endpoint"
    assert hinge.max_rms == 8.0
    assert hinge.weight == 1.0e-4
    assert cap._target_.endswith("LatentEndpointSmoothRMSCap")
    assert cap.input_key == "sampler/endpoint"
    assert cap.output_key == "sampler/stabilized_endpoint"
    assert cap.soft_start_rms == 6.0
    assert cap.max_rms == 8.0
    assert decoder.decoders[DOMAINS[0]].action_dim == 4
    assert decoder.decoders[DOMAINS[1]].action_dim == 6
    assert decoder.decoders[DOMAINS[0]].hidden_dim == 16
    assert decoder.decoders[DOMAINS[1]].hidden_dim == 16
    assert decoder.decoders[DOMAINS[0]].num_layers == 3
    assert decoder.decoders[DOMAINS[1]].num_layers == 3
    assert decoder.input_key == "sampler/stabilized_endpoint"
    assert decoder.output_key == "raw_pred_action"
    assert canonicalizer._target_.endswith("PerEmbodimentActionCanonicalizer")
    assert canonicalizer.input_key == "raw_pred_action"
    assert canonicalizer.target_output_key == "canonical_target"
    assert canonicalizer.require_samples is True
    assert canonicalizer.representation_loss_weight == 1.0e-3
    assert canonicalizer.canonicalizers[DOMAINS[0]]._target_.endswith(
        "USocketRotVecActionCanonicalizer"
    )
    assert canonicalizer.canonicalizers[DOMAINS[1]]._target_.endswith(
        "ChainGripperPointActionCanonicalizer"
    )
    assert loss._target_.endswith("ConditionalEnergyScoreLoss")
    assert loss.beta == 1.0
    assert loss.normalize_by_dimension is True
    assert loss.expected_num_samples == 4
    assert loss.target_key == "canonical_target"
    assert model.rollout_adapters[DOMAINS[1]].input_is_canonical is True

    model_yaml = OmegaConf.to_yaml(cfg.model).lower()
    assert "nativeactionmseloss" not in model_yaml
    assert "action_encoder" not in model_yaml


def test_energy_score_data_is_disjoint_seed42_val01_and_train_only_norm():
    cfg = _compose()
    assert cfg.seed == 42
    assert cfg.data.valid_combined_mode == "max_size"
    assert cfg.data.manage_distributed_samplers is True
    assert cfg.trainer.use_distributed_sampler is False
    for domain in DOMAINS:
        train = cfg.data.train_datasets[domain]
        valid = cfg.data.valid_datasets[domain]
        assert train.mode == "train"
        assert valid.mode == "valid"
        assert train.valid_ratio == valid.valid_ratio == 0.01
        assert OmegaConf.to_container(
            train.resolver, resolve=True
        ) == OmegaConf.to_container(valid.resolver, resolve=True)
    chain_roots = list(cfg.data.train_datasets[DOMAINS[1]].resolver.folder_paths)
    assert chain_roots[-1].endswith(
        "/chain_gripper_gen_flow_transfer_frozen719_20260829"
    )
    assert cfg.data.train_dataloader_params[DOMAINS[0]].batch_size == 32
    assert cfg.data.train_dataloader_params[DOMAINS[1]].batch_size == 32
    assert "disjoint1pct_trainonly" in cfg.norm_stats.precomputed_norm_path


def test_energy_score_training_validation_logging_and_checkpoint_contract():
    cfg = _compose()
    assert cfg.launch_params.gpus_per_node == cfg.trainer.devices == 2
    assert cfg.trainer.strategy == "ddp"
    assert cfg.trainer.precision == "bf16"
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.gradient_clip_val is None
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 1.0
    assert cfg.model.train_metrics_on_step is True
    assert cfg.model.optimizer.lr == 3.0e-5
    assert cfg.model.scheduler.eta_min == 3.0e-6
    assert cfg.evaluator.deterministic_seed == 42
    assert cfg.evaluator.exact_epoch_metrics is True
    assert cfg.callbacks.model_checkpoint.every_n_train_steps == 20_000
    assert cfg.callbacks.model_checkpoint.monitor is None
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert "epoch=" in cfg.callbacks.model_checkpoint.filename
    assert "step=" in cfg.callbacks.model_checkpoint.filename
    tags = set(cfg.logger.wandb.tags)
    assert {
        "conditional-energy-score",
        "model-260m",
        "denoiser-w768-d13-mlp2",
        "latent-candidate-hinge-rms8-w1e4",
        "latent-smooth-rms6to8",
        "global-step-val-10k",
        "stability-gated-pair",
        "checkpoint-20k",
    } <= tags
    assert cfg.logger.wandb.group.endswith("260m_smoothcap_20260831")


def test_legacy_mse_recipe_remains_the_default_for_existing_experiment():
    cfg = _compose(
        "pusht/pipeline_sampler_usocket_chain_newdata_cotrain12_per_emb_proprio_h16"
    )
    assert cfg.model.robomimic_model.stages[-1]._target_.endswith("NativeActionMSELoss")
    assert cfg.model.robomimic_model.stages[2].get("num_samples", 1) == 1


def test_matched_control_changes_only_the_grouped_objective_target():
    energy_cfg = _compose()
    control_cfg = _compose(
        extra_overrides=[
            "model.robomimic_model.stages.8._target_="
            "egomimic.pipeline.stages_sampler.GroupedActionMSELoss"
        ]
    )
    stage = control_cfg.model.robomimic_model.stages[8]
    assert stage._target_.endswith("GroupedActionMSELoss")
    assert stage.expected_num_samples == 4
    assert control_cfg.model.robomimic_model.stages[2].num_samples == 4
    assert control_cfg.data.train_dataloader_params[DOMAINS[0]].batch_size == 32

    energy_model = OmegaConf.to_container(
        energy_cfg.model.robomimic_model, resolve=True
    )
    control_model = OmegaConf.to_container(
        control_cfg.model.robomimic_model, resolve=True
    )
    control_model["stages"][8]["_target_"] = energy_model["stages"][8]["_target_"]
    assert control_model == energy_model
    for section in ("data", "trainer", "evaluator", "norm_stats"):
        assert OmegaConf.to_container(
            control_cfg[section], resolve=False
        ) == OmegaConf.to_container(energy_cfg[section], resolve=False)
