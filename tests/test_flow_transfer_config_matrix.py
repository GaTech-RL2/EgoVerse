from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"


def _compose(experiment: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


@pytest.mark.parametrize(
    ("experiment", "latent_dim", "hidden_dim"),
    [
        (
            "pusht/pipeline_sampler_chain_gripper_points_arc_length_medium",
            96,
            384,
        ),
        (
            "pusht/pipeline_sampler_chain_gripper_points_arc_length_large",
            128,
            512,
        ),
        (
            "pusht/pipeline_sampler_usocket_chain_points_arc_length_medium",
            96,
            384,
        ),
        (
            "pusht/pipeline_sampler_usocket_chain_points_arc_length_large",
            128,
            512,
        ),
    ],
)
def test_flow_transfer_latent_capacity_and_horizon_contract(
    experiment: str,
    latent_dim: int,
    hidden_dim: int,
) -> None:
    cfg = _compose(experiment)
    model = cfg.model.robomimic_model
    noise = model.stages[1]
    sampler = model.stages[2]

    assert model.action_horizon == 26
    assert noise.action_horizon == 26
    assert noise.latent_dim == latent_dim
    assert sampler.action_horizon == 26
    assert sampler.latent_dim == latent_dim
    assert sampler.denoiser_hidden_dim == hidden_dim
    assert sampler.denoising_module.act_dim == latent_dim
    assert sampler.denoising_module.hidden_dim == hidden_dim
    assert sampler.denoising_module.nblocks == 16
    assert sampler.denoising_module.act_seq == 26
    assert sampler.denoising_module.time_conditioning == "additive"
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
    assert cfg.norm_stats.reduce_all_but_last is False

    # Decoder-only refers to the action path. The camera/state condition encoder
    # remains present by design.
    model_yaml = OmegaConf.to_yaml(cfg.model)
    assert "action_encoder" not in model_yaml.lower()
    assert "latentactionencoder" not in model_yaml.lower()


def test_chain_full_data_composes_native_fk_then_anchored_phi_arc() -> None:
    cfg = _compose("pusht/pipeline_sampler_chain_gripper_points_arc_length_medium")
    data = cfg.data.train_datasets.pushshapes_sim_chain_gripper
    resolver = instantiate(data.resolver)

    assert str(resolver.folder_path).endswith("/chain_gripper_3000_v2")
    assert resolver.key_map["actions"]["zarr_key"] == "actions"
    assert resolver.key_map["actions"]["horizon"] == 100
    assert [x.__class__.__name__ for x in resolver.transform_list] == [
        "ChainGripperNative4ToPoints6",
        "TokenizeChainGripperPointArcLength",
    ]

    controls = np.column_stack(
        [
            np.linspace(100.0, 300.0, 100),
            np.full(100, 240.0),
            np.linspace(-0.3, 0.5, 100),
            np.linspace(0.1, 0.9, 100),
        ]
    ).astype(np.float32)
    sample = {"actions": controls}
    for transform in resolver.transform_list:
        sample = transform.transform(sample)
    assert sample["actions"].shape == (26, 6)
    assert np.isfinite(sample["actions"]).all()


def test_chain_direct_loader_and_revert_list_are_native_source_only() -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="data/pusht/chain_gripper_pipeline_h16_points_full")
    data = cfg.data.pusht.train_datasets.pushshapes_sim_chain_gripper
    resolver = instantiate(data.resolver)

    assert resolver.key_map["actions"]["zarr_key"] == "actions"
    assert [x.__class__.__name__ for x in resolver.transform_list] == [
        "ChainGripperNative4ToPoints6"
    ]

    from egomimic.rldb.embodiment.pushshapes import (
        get_chain_gripper_point_revert_transform_list,
    )

    revert = get_chain_gripper_point_revert_transform_list(keys=["actions"])
    assert [x.__class__.__name__ for x in revert] == [
        "ChainGripperPoints6ToNative4"
    ]


@pytest.mark.parametrize(
    "experiment",
    [
        "pusht/pipeline_sampler_usocket_chain_points_arc_length_medium",
        "pusht/pipeline_sampler_usocket_chain_points_arc_length_large",
    ],
)
def test_cotrain_has_two_decoders_adapters_and_no_fake_obstacle(experiment: str) -> None:
    cfg = _compose(experiment)
    model = cfg.model.robomimic_model
    sampler = model.stages[2]

    assert list(model.domains) == [
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    ]
    assert dict(sampler.action_dims) == {
        "pushshapes_sim_u_socket": 4,
        "pushshapes_sim_chain_gripper": 6,
    }
    assert set(model.rollout_adapters) == {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    assert model.rollout_adapters.pushshapes_sim_u_socket._target_.endswith(
        "USocketArcLengthRolloutAdapter"
    )
    assert model.rollout_adapters.pushshapes_sim_chain_gripper._target_.endswith(
        "ChainGripperPointArcLengthRolloutAdapter"
    )
    assert set(cfg.data.train_datasets) == {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    assert "960" not in OmegaConf.to_yaml(cfg.data)
    assert "obstacle" not in cfg.name.lower()

    u_resolver = instantiate(
        cfg.data.train_datasets.pushshapes_sim_u_socket.resolver
    )
    chain_resolver = instantiate(
        cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
    )
    assert [x.__class__.__name__ for x in u_resolver.transform_list] == [
        "TokenizeUSocketArcLength"
    ]
    assert [x.__class__.__name__ for x in chain_resolver.transform_list] == [
        "ChainGripperNative4ToPoints6",
        "TokenizeChainGripperPointArcLength",
    ]


def test_no_active_pushshapes_recipe_reads_materialized_chain_points() -> None:
    data_dir = CONFIG_DIR / "data" / "pusht"
    offenders = []
    for path in data_dir.glob("*.yaml"):
        text = path.read_text()
        if "actions.points" in text or "dual_control" in text:
            offenders.append(path.name)
    assert offenders == []
