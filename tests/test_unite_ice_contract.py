from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.ice.unite.unite_contract import validate_unite_config


@pytest.fixture
def resolved_config():
    config_dir = Path("egomimic/hydra_configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "model=bf/us_unite_register_shared_nt4_s42",
                "+experiment=pusht/unite_usocket_register_sweep_val01_h16",
                "trainer.devices=2",
                "++paths.root_dir=.",
            ],
        )


@pytest.mark.parametrize(
    ("shared", "tokens"),
    [(True, 4), (True, 8), (False, 4), (False, 8)],
)
def test_clean_rows_bind_energy_view_to_world_size(resolved_config, shared, tokens):
    config = OmegaConf.create(OmegaConf.to_container(resolved_config, resolve=False))
    config.model.share_encoder_denoiser = shared
    config.model.num_latent_tokens = tokens
    result = validate_unite_config(config, expected_world_size=2)
    assert result["status"] == "READY"
    assert result["row"] == {
        "share_encoder_denoiser": shared,
        "num_latent_tokens": tokens,
        "latent_dim": 16,
    }
    assert result["energy_score_conditions"] == 32
    assert result["diagnostics"] == "ABSENT"


def test_single_gpu_view_is_supported_when_total_is_32(resolved_config):
    config = OmegaConf.create(OmegaConf.to_container(resolved_config, resolve=False))
    config.trainer.devices = 1
    config.evaluator.energy_score_validation_view.world_size = 1
    config.evaluator.energy_score_validation_view.per_rank_batch_size = 32
    assert validate_unite_config(config, expected_world_size=1)["world_size"] == 1


def test_energy_view_rejects_world_size_or_total_mismatch(resolved_config):
    config = OmegaConf.create(OmegaConf.to_container(resolved_config, resolve=False))
    config.evaluator.energy_score_validation_view.per_rank_batch_size = 8
    with pytest.raises(ValueError, match="exactly 32 conditions"):
        validate_unite_config(config, expected_world_size=2)


def test_optional_diagnostics_use_the_same_world_size_contract(resolved_config):
    config = OmegaConf.create(OmegaConf.to_container(resolved_config, resolve=False))
    config.evaluator.unite_diagnostics = {
        "validation_view": {"world_size": 2, "per_rank_batch_size": 16}
    }
    assert (
        validate_unite_config(config, expected_world_size=2)["diagnostics"]
        == "BOUND_TO_WORLD_SIZE"
    )
    config.evaluator.unite_diagnostics.validation_view.per_rank_batch_size = 15
    with pytest.raises(ValueError, match="diagnostic validation view"):
        validate_unite_config(config, expected_world_size=2)


def test_contract_rejects_scientific_drift(resolved_config):
    config = OmegaConf.create(OmegaConf.to_container(resolved_config, resolve=False))
    policy = next(
        stage
        for stage in config.model.pipeline.stages
        if str(stage.get("_target_", "")).endswith("ReleasedRecipeUniteLatentPolicy")
    )
    policy.flow_steps_per_reconstruction = 13
    with pytest.raises(ValueError, match="unsupported UNITE"):
        validate_unite_config(config, expected_world_size=2)


def test_n8_flow42_ablation_is_explicit_and_contract_valid():
    config_dir = Path("egomimic/hydra_configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        config = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "model=bf/us_unite_register_shared_nt8_flow42_s42",
                "+experiment=pusht/unite_usocket_register_sweep_val01_h16",
                "trainer.devices=1",
                "evaluator.energy_score_validation_view.per_rank_batch_size=32",
                "++paths.root_dir=.",
            ],
        )
    result = validate_unite_config(config, expected_world_size=1)
    assert result["row"]["num_latent_tokens"] == 8
    assert result["flow_steps_per_reconstruction"] == 42
    assert config.run_provenance.unite_update_schedule.flow_samples_per_reconstruction == 42
