import hashlib
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROWS = {
    "usocket_arc_bc": ("pushshapes_sim_u_socket", 101, 1, "PlanarFlowSampler"),
    "usocket_arc_hybrid_D40_M16_R24deg_bc": (
        "pushshapes_sim_u_socket",
        17,
        1,
        "PlanarFlowSampler",
    ),
    "chain_arc_bc": ("pushshapes_sim_chain_gripper", 101, 1, "PlanarFlowSampler"),
    "usocket_direct_bc": ("pushshapes_sim_u_socket", 16, 1, "PlanarFlowSampler"),
    "chain_direct_bc": ("pushshapes_sim_chain_gripper", 16, 1, "PlanarFlowSampler"),
    "usocket_dp_standard": ("pushshapes_sim_u_socket", 16, 1, "ConditionalUnet1D"),
    "chain_dp_standard": ("pushshapes_sim_chain_gripper", 16, 1, "ConditionalUnet1D"),
    "usocket_dp_paper": ("pushshapes_sim_u_socket", 16, 2, "PaperConditionalUnet1D"),
    "chain_dp_paper": ("pushshapes_sim_chain_gripper", 16, 2, "PaperConditionalUnet1D"),
}

DATASETS = {
    "pushshapes_sim_u_socket": {
        "root": "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/u_socket_3000_v2_clean",
        "total": 2999,
        "train": 2970,
        "valid": 29,
        "manifest": "planar_v2_usocket_dp_3k_split_seed42_v1.json",
        "manifest_sha256": "3683e3461596eef8df2432fa865779b3c77b2a2057dabd0fea125595729cf313",
    },
    "pushshapes_sim_chain_gripper": {
        "root": "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
        "total": 3000,
        "train": 2970,
        "valid": 30,
        "manifest": "planar_v2_chain_gripper_dp_3k_split_seed42_v1.json",
        "manifest_sha256": "3ced944ea3af8e875ea88fc5c2df3a5d2865a9f95223d109fb4bd28c8be7cf69",
    },
}

DP_DOMAIN_BASES = {
    "chain_dp_standard": "chain_base",
    "chain_dp_paper": "chain_base",
    "usocket_dp_standard": "usocket_base",
    "usocket_dp_paper": "usocket_base",
}

DOMAIN_BASE_DATA = {
    "chain_base": "chain_gripper",
    "usocket_base": "usocket",
}


@pytest.mark.parametrize("row,expected", ROWS.items())
def test_planar_row_composes_without_pipeline_routing_metadata(row, expected):
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/planar_v2_{row}", "++paths.root_dir=."],
        )
    for section in (cfg.planar, cfg.model, cfg.data):
        OmegaConf.resolve(section)
    domain, horizon, observation_horizon, implementation = expected
    assert list(cfg.data.train_datasets) == [domain]
    assert not {
        "action_horizon",
        "domains",
        "ac_keys",
        "rollout_adapter",
        "rollout_adapters",
        "rollout_observation_adapters",
        "norm_stats",
    } & set(cfg.model.pipeline)
    dataset = cfg.data.train_datasets[domain]
    assert dataset.valid_ratio == cfg.data.valid_datasets[domain].valid_ratio == 0.01
    assert dataset.split_seed == cfg.seed == 42
    expected_data = DATASETS[domain]
    assert dataset.resolver.folder_path == expected_data["root"]
    assert dataset.expected_train_episode_count == expected_data["train"]
    assert dataset.expected_valid_episode_count == expected_data["valid"]
    assert dataset.resolver.expected_episode_count == expected_data["total"]
    assert cfg.run_provenance.split_manifest_sha256 == expected_data["manifest_sha256"]
    assert cfg.run_provenance.dataset_observation_alignment == "pre_step"
    assert cfg.planar.observation_horizon == observation_horizon
    decoder = cfg.planar.eval_native_decoder
    if "arc_bc" in row:
        assert decoder._target_.endswith("PlanarArcWaypointZeroNativeDecoder")
        assert "action_horizon" not in decoder
    elif "arc_hybrid" in row:
        assert decoder._target_.endswith("PlanarArcWaypointZeroNativeDecoder")
        assert decoder.resampled_vector_length == 16
        assert cfg.planar.arc_distance == 40.0
        assert cfg.planar.arc_rotation_radius == 0.0
        assert cfg.planar.hybrid_rotation_unit == 0.14776
        assert dataset.resolver.transform_list.rotation_radius == 0.0
        assert dataset.resolver.transform_list.hybrid_rotation_unit == 0.14776
    else:
        assert decoder._target_.endswith("PlanarCommon5NativeDecoder")
        assert decoder.action_horizon == 16
    if "arc_hybrid" not in row:
        assert dataset.resolver.transform_list.rotation_radius == 30.0
        assert dataset.resolver.transform_list.hybrid_rotation_unit is None
    model_yaml = OmegaConf.to_yaml(cfg.model)
    assert implementation in model_yaml
    stage_names = [
        stage._target_.rsplit(".", 1)[-1] for stage in cfg.model.pipeline.stages
    ]
    assert stage_names[0:2] == ["FusedObsEncoder", "ActionTargetBuilder"]
    assert cfg.model.pipeline.stages[0].inputs == {
        "front_img_1": "front_img_1",
        "state_agent_obj": "state_agent_obj",
    }
    sampler = cfg.model.pipeline.stages[3]
    assert sampler.action_horizon == horizon
    if "dp_paper" not in row:
        assert "eval_checkpoint" not in cfg
    if "dp_paper" in row:
        assert cfg.callbacks.ema.use_warmup is True
        assert cfg.eval_checkpoint.use_ema is True
        assert "DDPMScheduler" in model_yaml
        assert stage_names == [
            "FusedObsEncoder",
            "ActionTargetBuilder",
            "DiffusionNoisingStage",
            "DiffusionDenoiserStage",
            "DiffusionEpsilonLossStage",
        ]
        assert cfg.planar.action_target_offset == 1
        assert cfg.model.pipeline.stages[3].action_dim == 5
        assert OmegaConf.to_container(cfg.run_provenance.action_contract) == {
            "observation_alignment": "pre_step",
            "observation_horizon": 2,
            "training_action_target_offset": 1,
            "prediction_horizon": 16,
            "rollout_action_chunk_start_index": 0,
            "replan_every": 8,
            "execution_slice": "[0,8)",
        }
    elif "dp_standard" in row:
        assert "DDIMScheduler" in model_yaml
        assert "ema" not in cfg.callbacks
        assert cfg.model.pipeline.stages[3].action_dim == 5
        assert stage_names == [
            "FusedObsEncoder",
            "ActionTargetBuilder",
            "DiffusionNoisingStage",
            "DiffusionDenoiserStage",
            "DiffusionEpsilonLossStage",
        ]


@pytest.mark.parametrize("row,domain_base", DP_DOMAIN_BASES.items())
def test_dp_rows_inherit_model_neutral_domain_base(row, domain_base):
    experiment_dir = (
        Path(__file__).parents[1]
        / "egomimic"
        / "hydra_configs"
        / "experiment"
        / "pusht"
    )
    experiment = OmegaConf.load(experiment_dir / f"planar_v2_{row}.yaml")
    defaults = OmegaConf.to_container(experiment.defaults)
    assert defaults[0] == f"/experiment/pusht/planar_v2_{domain_base}"
    assert all("direct_bc" not in str(item) for item in defaults)

    base = OmegaConf.load(experiment_dir / f"planar_v2_{domain_base}.yaml")
    base_defaults = OmegaConf.to_container(base.defaults)
    assert base_defaults[0] == "/experiment/pusht/planar_v2_base"
    assert base_defaults[1] == {
        "override /data": f"pusht/planar_v2_{DOMAIN_BASE_DATA[domain_base]}"
    }
    assert "model" not in base


def test_standard_dp_clean_cotrain_composes_both_domains_without_obstacles():
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/planar_v2_cotrain_clean_dp_standard",
                "++paths.root_dir=.",
            ],
        )
    for section in (cfg.planar, cfg.model, cfg.data, cfg.run_provenance):
        OmegaConf.resolve(section)
    expected_domains = {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    assert set(cfg.data.train_datasets) == expected_domains
    assert set(cfg.data.valid_datasets) == expected_domains
    assert set(cfg.evaluator.native_decoders) == expected_domains
    assert cfg.run_provenance.obstacle_data is False
    assert all(
        "obstacle" not in dataset.resolver.folder_path.lower()
        for dataset in cfg.data.train_datasets.values()
    )
    assert all(
        dataset.valid_ratio == 0.01 and dataset.split_seed == 42
        for dataset in cfg.data.train_datasets.values()
    )
    assert cfg.model.pipeline.stages[3].action_dim == 5


def test_diffusion_mse_is_visible_during_long_training_epochs():
    model_wrapper = (
        Path(__file__).parents[1] / "egomimic" / "pl_utils" / "pl_model.py"
    ).read_text()
    aggregate = model_wrapper.index('"Train/MSE",')
    per_source = model_wrapper.index('f"Train/MSE/{source}",')
    assert "on_step=True" in model_wrapper[aggregate : aggregate + 220]
    assert "on_step=True" in model_wrapper[per_source : per_source + 220]


def test_ddp_device_count_is_per_node_without_eval_resolver():
    config_dir = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/planar_v2_cotrain_clean_dp_standard",
                "launch_params.gpus_per_node=2",
                "launch_params.nodes=3",
                "++paths.root_dir=.",
            ],
        )
    assert cfg.trainer.devices == 2
    assert cfg.trainer.num_nodes == 3
    assert "${eval:" not in OmegaConf.to_yaml(cfg.trainer, resolve=False)


@pytest.mark.parametrize("domain,expected", DATASETS.items())
def test_three_thousand_episode_split_manifest_is_exact(domain, expected):
    data_dir = (
        Path(__file__).parents[1] / "egomimic" / "hydra_configs" / "data" / "pusht"
    )
    path = data_dir / expected["manifest"]
    payload = path.read_bytes()
    manifest = json.loads(payload)
    split = manifest["domains"][domain]

    assert hashlib.sha256(payload).hexdigest() == expected["manifest_sha256"]
    assert split["folder_path"] == expected["root"]
    assert split["total_count"] == expected["total"]
    assert split["train_count"] == expected["train"]
    assert split["valid_count"] == expected["valid"]
    assert split["id_overlap_count"] == 0
    assert split["resolved_path_overlap_count"] == 0


def test_runtime_targets_are_declared_dependencies():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '"diffusers==0.30.3"' in pyproject
