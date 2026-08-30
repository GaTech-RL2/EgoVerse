from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "egomimic" / "hydra_configs"


def _config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/pipeline_diffusion_usocket_chain_newdata_val01_h16_20m"
            ],
        )


def _launch_config():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/pipeline_diffusion_usocket_chain_newdata_val01_h16_20m",
                "logger.wandb.id=dp20m_smoke_test",
                "+logger.wandb.name=dp20m_smoke_test",
                "+logger.wandb.resume=never",
            ],
        )


def test_dp20m_architecture_and_training_contract():
    cfg = _config()
    pipeline = cfg.model.robomimic_model
    assert pipeline.action_horizon == 16
    assert pipeline.stages[0].n_obs_steps == 1
    assert pipeline.stages[1].condition_input_dim == 67
    assert cfg.model.optimizer.lr == 3e-5
    assert cfg.model.scheduler.eta_min == 3e-6
    assert cfg.trainer.max_steps == 240000
    assert cfg.trainer.val_check_interval == 10000
    assert cfg.trainer.limit_val_batches == 8
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.evaluator.exact_epoch_metrics is True
    for domain, action_dim in {
        "pushshapes_sim_u_socket": 4,
        "pushshapes_sim_chain_gripper": 6,
    }.items():
        policy = pipeline.stages[1].policies[domain]
        assert policy.model.input_dim == action_dim
        assert list(policy.model.down_dims) == [72, 144, 288]
        assert policy.noise_scheduler.prediction_type == "epsilon"


def test_dp20m_disjoint_split_and_checkpoint_contract():
    cfg = _config()
    assert cfg.data.valid_combined_mode == "max_size"
    domains = set(cfg.model.robomimic_model.domains)
    assert set(cfg.data.train_datasets) == set(cfg.data.valid_datasets) == domains
    for domain in domains:
        train = cfg.data.train_datasets[domain]
        valid = cfg.data.valid_datasets[domain]
        assert train.mode == "train" and valid.mode == "valid"
        assert train.valid_ratio == valid.valid_ratio == 0.01
    checkpoint = cfg.callbacks.model_checkpoint
    assert checkpoint.save_top_k == -1
    assert checkpoint.save_last is True
    assert checkpoint.every_n_train_steps == 20000
    assert checkpoint.auto_insert_metric_name is False


def test_dp20m_launcher_adds_optional_wandb_fields():
    cfg = _launch_config()
    assert cfg.logger.wandb.id == cfg.logger.wandb.name == "dp20m_smoke_test"
    assert cfg.logger.wandb.resume == "never"


def test_dp20m_launcher_hardens_hardware_and_strict_reload_contracts():
    launcher = (
        ROOT / "scripts/train/flow_transfer_dp_capacity_skynet_l40sx2.sbatch"
    ).read_text()
    assert "EXPECTED_SLURM_ACCOUNT=${EXPECTED_SLURM_ACCOUNT:?" in launcher
    assert "EXPECTED_SLURM_PARTITION=${EXPECTED_SLURM_PARTITION:?" in launcher
    assert "EXPECTED_GPU_MODEL=${EXPECTED_GPU_MODEL:?" in launcher
    assert "strict=True, weights_only=False" in launcher
