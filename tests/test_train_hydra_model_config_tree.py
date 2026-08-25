from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.trainHydra import _build_model_config_tree

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def test_model_tree_materializes_parent_interpolations_before_detaching():
    cfg = OmegaConf.create(
        {
            "dense_latent_dim": 64,
            "model": {
                "robomimic_model": {
                    "norm_stats": "${missing_norm_stats}",
                    "stages": [
                        {"latent_dim": "${oc.select:dense_latent_dim,128}"},
                        {
                            "latent_dim": (
                                "${model.robomimic_model.stages.0.latent_dim}"
                            )
                        },
                    ],
                }
            },
        }
    )

    model_tree = _build_model_config_tree(cfg)

    assert model_tree.model.robomimic_model.stages[0].latent_dim == 64
    assert model_tree.model.robomimic_model.stages[1].latent_dim == 64
    assert model_tree.model.robomimic_model.norm_stats is None
    unresolved_original = OmegaConf.to_container(cfg, resolve=False)
    assert (
        unresolved_original["model"]["robomimic_model"]["norm_stats"]
        == "${missing_norm_stats}"
    )


@pytest.mark.parametrize(
    ("experiment", "latent_dim", "denoiser_hidden_dim"),
    [
        ("pipeline_sampler_robot_bc_latent64", 64, 512),
        ("pipeline_sampler_robot_bc_latent64_denoiser256", 64, 256),
        ("pipeline_sampler_robot_bc_latent96_denoiser384", 96, 384),
    ],
)
def test_dense_capacity_survives_model_tree_detachment(
    experiment: str,
    latent_dim: int,
    denoiser_hidden_dim: int,
):
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR.resolve()),
        version_base=None,
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=fold/rh/{experiment}"],
        )

    model_tree = _build_model_config_tree(cfg)
    stages = model_tree.model.robomimic_model.stages
    sampler = stages[2]

    assert stages[1].latent_dim == latent_dim
    assert sampler.latent_dim == latent_dim
    assert sampler.denoiser_hidden_dim == denoiser_hidden_dim
    assert sampler.denoising_module.act_dim == latent_dim
    assert sampler.denoising_module.hidden_dim == denoiser_hidden_dim
