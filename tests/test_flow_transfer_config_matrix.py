from pathlib import Path

import numpy as np
import pytest
import zarr
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
    assert [x.__class__.__name__ for x in revert] == ["ChainGripperPoints6ToNative4"]


@pytest.mark.parametrize(
    "experiment",
    [
        "pusht/pipeline_sampler_usocket_chain_points_arc_length_medium",
        "pusht/pipeline_sampler_usocket_chain_points_arc_length_large",
    ],
)
def test_cotrain_has_two_decoders_adapters_and_no_fake_obstacle(
    experiment: str,
) -> None:
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

    u_resolver = instantiate(cfg.data.train_datasets.pushshapes_sim_u_socket.resolver)
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


@pytest.mark.parametrize(
    ("smoke", "full"),
    [
        (
            "pusht/pipeline_sampler_chain_gripper_points_arc_length_medium_smoke",
            "pusht/pipeline_sampler_chain_gripper_points_arc_length_medium",
        ),
        (
            "pusht/pipeline_sampler_usocket_chain_points_arc_length_medium_smoke",
            "pusht/pipeline_sampler_usocket_chain_points_arc_length_medium",
        ),
    ],
)
def test_flow_transfer_smoke_preserves_full_model_data_and_evaluator(
    smoke: str, full: str
) -> None:
    smoke_cfg = _compose(smoke)
    full_cfg = _compose(full)

    assert OmegaConf.to_container(
        smoke_cfg.model, resolve=False
    ) == OmegaConf.to_container(full_cfg.model, resolve=False)
    assert OmegaConf.to_container(
        smoke_cfg.data, resolve=False
    ) == OmegaConf.to_container(full_cfg.data, resolve=False)
    assert OmegaConf.to_container(
        smoke_cfg.evaluator, resolve=False
    ) == OmegaConf.to_container(full_cfg.evaluator, resolve=False)
    assert smoke_cfg.trainer.precision == full_cfg.trainer.precision == "bf16"
    assert smoke_cfg.launch_params == full_cfg.launch_params
    assert smoke_cfg.trainer.max_steps == 2
    assert smoke_cfg.trainer.val_check_interval == 1
    assert smoke_cfg.trainer.limit_val_batches == 1
    assert smoke_cfg.callbacks.model_checkpoint.every_n_train_steps == 1
    assert smoke_cfg.callbacks.model_checkpoint.save_last is True
    assert smoke_cfg.norm_stats.sample_frac == 0.002


@pytest.mark.parametrize(
    ("experiment", "domains", "action_dims"),
    [
        (
            "pusht/pipeline_sampler_usocket_dense_medium",
            ["pushshapes_sim_u_socket"],
            {"pushshapes_sim_u_socket": 4},
        ),
        (
            "pusht/pipeline_sampler_chain_gripper_points_dense_medium",
            ["pushshapes_sim_chain_gripper"],
            {"pushshapes_sim_chain_gripper": 6},
        ),
        (
            "pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium",
            ["pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"],
            {"pushshapes_sim_u_socket": 4, "pushshapes_sim_chain_gripper": 6},
        ),
    ],
)
def test_direct_dense_medium_uses_fold_topology_without_arc_tokens(
    monkeypatch, experiment, domains, action_dims
):
    monkeypatch.setenv("CHAIN_OBSTACLE_ROOT", "/tmp/chain-obstacle-audited")
    cfg = _compose(experiment)
    model = cfg.model.robomimic_model
    noise = model.stages[1]
    sampler = model.stages[2]

    assert list(model.domains) == domains
    assert model.action_horizon == 100
    assert noise.action_horizon == 100
    assert noise.latent_dim == 96
    assert sampler.action_horizon == 100
    assert sampler.latent_dim == 96
    assert sampler.decoder_hidden_dim == 512
    assert sampler.denoiser_hidden_dim == 384
    assert dict(sampler.action_dims) == action_dims
    assert sampler.denoising_module.act_seq == 100
    assert sampler.denoising_module.act_dim == 96
    assert sampler.denoising_module.hidden_dim == 384
    assert sampler.denoising_module.nblocks == 16
    assert sampler.num_inference_steps == 16
    assert cfg.norm_stats.reduce_all_but_last is True
    resolved = OmegaConf.to_yaml(cfg)
    assert "arc_length" not in resolved
    assert "velocity" not in resolved.lower()


@pytest.mark.parametrize(
    ("experiment", "domains", "width"),
    [
        (
            "pusht/pipeline_diffusion_usocket_h16",
            ["pushshapes_sim_u_socket"],
            4,
        ),
        (
            "pusht/pipeline_diffusion_chain_gripper_points_h16",
            ["pushshapes_sim_chain_gripper"],
            6,
        ),
    ],
)
def test_single_domain_dp_controls_are_genuine_action_diffusion(
    experiment, domains, width
):
    cfg = _compose(experiment)
    model = cfg.model.robomimic_model
    stage = model.stages[1]
    policy = stage.policies[domains[0]]

    assert list(model.domains) == domains
    assert model.action_horizon == stage.action_horizon == 16
    assert policy.model._target_.endswith("ConditionalUnet1D")
    assert policy.model.input_dim == width
    assert policy.noise_scheduler.prediction_type == "epsilon"
    assert policy.num_inference_steps == 100
    assert "GaussianLatentNoise" not in OmegaConf.to_yaml(cfg.model)


def test_obstacle_cotrain_config_pins_all_audited_sources(monkeypatch):
    root = "/audit/chain-obstacle-output-128"
    monkeypatch.setenv("CHAIN_OBSTACLE_ROOT", root)
    cfg = _compose("pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium")
    chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver

    assert chain._target_.endswith("LocalEpisodeResolverManyWithEmbodimentOverride")
    assert len(chain.folder_paths) == 31
    assert chain.folder_paths[0].endswith("/chain_gripper_3000_v2")
    assert chain.folder_paths[1] == f"{root}/level_01/chain_gripper/T"
    assert chain.folder_paths[-1] == f"{root}/level_30/chain_gripper/T"
    assert chain.key_map.action_horizon == 100
    assert cfg.launch_params.gpus_per_node == 2

    dp = _compose("pusht/pipeline_diffusion_usocket_chain_obstacle_h16")
    assert (
        dp.data.train_datasets.pushshapes_sim_u_socket.resolver.key_map.action_horizon
        == 16
    )
    assert (
        dp.data.train_datasets.pushshapes_sim_chain_gripper.resolver.key_map.action_horizon
        == 16
    )
    assert (
        len(dp.data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_paths)
        == 31
    )


def test_many_root_resolver_namespaces_colliding_episode_names(tmp_path):
    from egomimic.rldb.zarr.zarr_dataset_multi import (
        LocalEpisodeResolverManyWithEmbodimentOverride,
    )

    roots = [tmp_path / "clean", tmp_path / "obstacle"]
    for root in roots:
        group = zarr.open_group(str(root / "same.zarr"), mode="w")
        group.attrs["embodiment"] = "pushshapes_sim"

    class DummyDataset:
        def __init__(self, path, key_map=None, transform_list=None):
            self.path = path
            self.key_map = key_map
            self.transform_list = transform_list
            self.embodiment = "pushshapes_sim"

    resolver = LocalEpisodeResolverManyWithEmbodimentOverride(
        folder_paths=roots,
        embodiment_override="pushshapes_sim_chain_gripper",
    )
    resolver._dataset_class = DummyDataset
    datasets = resolver.resolve()

    assert set(datasets) == {"source_000/same", "source_001/same"}
    assert {dataset.embodiment for dataset in datasets.values()} == {
        "pushshapes_sim_chain_gripper"
    }
