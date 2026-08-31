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
            "pusht/pipeline_sampler_chain_gripper_obstacle_points_dense_medium",
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
            "pusht/pipeline_diffusion_chain_gripper_obstacle_points_h16",
            ["pushshapes_sim_chain_gripper"],
            6,
        ),
    ],
)
def test_single_domain_dp_controls_are_genuine_action_diffusion(
    monkeypatch, experiment, domains, width
):
    monkeypatch.setenv("CHAIN_OBSTACLE_ROOT", "/tmp/chain-obstacle-audited")
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


@pytest.mark.parametrize(
    "experiment",
    [
        "pusht/pipeline_sampler_usocket_dense_medium",
        "pusht/pipeline_sampler_chain_gripper_obstacle_points_dense_medium",
        "pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium",
        "pusht/pipeline_diffusion_usocket_h16",
        "pusht/pipeline_diffusion_chain_gripper_obstacle_points_h16",
        "pusht/pipeline_diffusion_usocket_chain_obstacle_h16",
    ],
)
def test_direct_dense_runtime_safety_contract(monkeypatch, experiment):
    monkeypatch.setenv("CHAIN_OBSTACLE_ROOT", "/tmp/chain-obstacle-audited")
    cfg = _compose(experiment)

    assert cfg.model.enable_grad_norm is False
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.limit_val_batches == 0
    expected_train_batch = 32 if len(cfg.data.train_datasets) == 2 else 64
    for domain in cfg.data.train_datasets:
        assert cfg.data.train_datasets[domain].valid_ratio == 0.0
        assert cfg.data.valid_datasets[domain].valid_ratio == 0.02
        assert (
            cfg.data.train_dataloader_params[domain].batch_size == expected_train_batch
        )
        assert cfg.data.valid_dataloader_params[domain].batch_size == 16
    world_size = cfg.launch_params.gpus_per_node
    global_batch_per_domain = {
        domain: cfg.data.train_dataloader_params[domain].batch_size * world_size
        for domain in cfg.data.train_datasets
    }
    assert set(global_batch_per_domain.values()) == {64}
    assert sum(global_batch_per_domain.values()) == (
        128 if len(global_batch_per_domain) == 2 else 64
    )

    terminal = cfg.callbacks.terminal_checkpoint
    assert terminal._target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
    assert terminal.filename == "step-{step}"
    assert terminal.monitor is None
    assert terminal.every_n_train_steps == cfg.trainer.max_steps
    assert terminal.every_n_epochs is None
    assert terminal.train_time_interval is None
    assert terminal.save_top_k == 1
    assert terminal.save_last is False
    assert terminal.save_on_train_epoch_end is False
    assert terminal.save_on_exception is False
    assert terminal.save_weights_only is False
    assert terminal.auto_insert_metric_name is False
    assert terminal.enable_version_counter is False

    expected_best = {}
    if "pushshapes_sim_u_socket" in cfg.data.train_datasets:
        expected_best["best_usocket_checkpoint"] = (
            "best_usocket",
            "Valid/emb19_actions_action_mse",
        )
    if "pushshapes_sim_chain_gripper" in cfg.data.train_datasets:
        expected_best["best_chain_checkpoint"] = (
            "best_chain",
            "Valid/emb20_actions_action_mse",
        )
    actual_best = {name for name in cfg.callbacks if name.startswith("best_")}
    assert actual_best == set(expected_best)
    for name, (directory, monitor) in expected_best.items():
        callback = cfg.callbacks[name]
        assert callback._target_ == "lightning.pytorch.callbacks.ModelCheckpoint"
        assert callback._get_node("dirpath")._value() == (
            f"${{paths.output_dir}}/checkpoints/{directory}"
        )
        assert callback.filename == "best-step-{step}"
        assert callback.monitor == monitor
        assert callback.mode == "min"
        assert callback.every_n_epochs == 1
        assert callback.every_n_train_steps is None
        assert callback.train_time_interval is None
        assert callback.save_top_k == 1
        assert callback.save_last is False
        assert callback.save_on_train_epoch_end is False
        assert callback.save_on_exception is False
        assert callback.save_weights_only is False
        assert callback.auto_insert_metric_name is False
        assert callback.enable_version_counter is False

    model = cfg.model.robomimic_model
    if len(model.stages) == 3:
        assert model.stages[2].gradient_accumulation_steps == 1


def test_obstacle_cotrain_config_pins_all_audited_sources(monkeypatch):
    root = "/audit/chain-obstacle-output-3000-balanced"
    monkeypatch.setenv("CHAIN_OBSTACLE_ROOT", root)
    cfg = _compose("pusht/pipeline_sampler_usocket_chain_obstacle_dense_medium")
    chain = cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver

    assert chain._target_.endswith("LocalEpisodeResolverManyWithEmbodimentOverride")
    assert len(chain.folder_paths) == 30
    assert chain.folder_paths[0] == f"{root}/level_01/chain_gripper/T"
    assert chain.folder_paths[1] == f"{root}/level_02/chain_gripper/T"
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
        == 30
    )
    for cotrain in (cfg, dp):
        assert cotrain.model.enable_grad_norm is False
        assert cotrain.model.optimizer.lr == pytest.approx(1.0e-4)
        assert cotrain.trainer.accumulate_grad_batches == 1
        scheduler = cotrain.model.scheduler
        assert scheduler.max_steps == 240_000
        assert scheduler.warmup_steps == 3_000
        assert scheduler.warmup_start_factor == 0.1
        assert scheduler.eta_min == 1.0e-5


@pytest.mark.parametrize(
    ("experiment", "horizon"),
    [
        (
            "pusht/pipeline_sampler_chain_gripper_obstacle_points_dense_medium",
            100,
        ),
        (
            "pusht/pipeline_diffusion_chain_gripper_obstacle_points_h16",
            16,
        ),
    ],
)
def test_chain_bc_uses_only_all_obstacle_roots(monkeypatch, experiment, horizon):
    root = "/audit/chain-obstacle-output-3000-balanced"
    monkeypatch.setenv("CHAIN_OBSTACLE_ROOT", root)
    cfg = _compose(experiment)
    assert set(cfg.data.train_datasets) == {"pushshapes_sim_chain_gripper"}
    for split in (cfg.data.train_datasets, cfg.data.valid_datasets):
        resolver = split.pushshapes_sim_chain_gripper.resolver
        assert resolver._target_.endswith(
            "LocalEpisodeResolverManyWithEmbodimentOverride"
        )
        assert len(resolver.folder_paths) == 30
        assert resolver.folder_paths[0] == f"{root}/level_01/chain_gripper/T"
        assert resolver.folder_paths[1] == f"{root}/level_02/chain_gripper/T"
        assert resolver.folder_paths[-1] == f"{root}/level_30/chain_gripper/T"
        assert resolver.key_map.action_horizon == horizon


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


def test_obstacle_launchers_do_not_gate_excluded_clean_chain_data() -> None:
    repo_root = Path(__file__).parents[1]
    matrix = (
        repo_root / "scripts" / "train" / "flow_transfer_direct_dense_matrix.sbatch"
    ).read_text()
    precompute = (
        repo_root / "scripts" / "train" / "flow_transfer_norm_precompute.sbatch"
    ).read_text()

    for launcher in (matrix, precompute):
        assert "CHAIN_DATA=" not in launcher
        assert "chain_clean_inventory" not in launcher
        assert "output_3000_balanced_v1" in launcher
        assert "EXPECTED_OBSTACLE_AUDIT_SHA=1f7f341b" in launcher
        assert "EXPECTED_OBSTACLE_MANIFEST_SHA=b5c9385b" in launcher
        assert "EXPECTED_OBSTACLE_INVENTORY_SHA=1c143396" in launcher
        assert "= 3000" in launcher
        assert "= 100" in launcher
        assert "-xtype d -name '*.zarr'" in launcher
        assert "EXPECTED_U_TRAIN_FRAMES=521676" in launcher
        assert "EXPECTED_CHAIN_TRAIN_FRAMES=1272946" in launcher
        assert "float(dataset.valid_ratio) == 0.0" in launcher
        assert "float(dataset.valid_ratio) == 0.02" in launcher
        assert 'dataset.mode == "train"' in launcher
        assert 'dataset.mode == "valid"' in launcher

    assert "mode=norm_stats" in precompute
    assert "trainer.strategy=" not in precompute
    assert 'if mode == "full":\n        assert not files' in matrix
    assert 'assert mode == "smoke"\n    assert len(files) == 1' in matrix


def test_matrix_submit_helper_is_six_arm_smoke_gated_and_world2_safe() -> None:
    helper = (
        Path(__file__).parents[1]
        / "scripts"
        / "train"
        / "submit_flow_transfer_direct_dense_matrix_when_ready.sh"
    ).read_text()

    assert "flow-transfer-direct-dense-obstacle-dp-schedulefix-20260826" in helper
    for arm in (
        "bc_usocket_latent",
        "bc_usocket_dp",
        "bc_chain_latent",
        "bc_chain_dp",
        "cotrain_obstacle_latent",
        "cotrain_obstacle_dp",
    ):
        assert arm in helper
    assert "LATENT_NORM_ARTIFACT=${LATENT_NORM_ARTIFACT:?" in helper
    assert "DP_NORM_ARTIFACT=${DP_NORM_ARTIFACT:?" in helper
    assert '--ntasks-per-node="$gpus"' in helper
    assert "--cpus-per-task=8" in helper
    assert "'hoffman-lab hoffman-lab a40 1 96G'" in helper
    assert "'rl2-lab rl2-lab l40s 2 128G'" in helper
    assert "partition=hoffman-lab" in helper
    assert "partition=rl2-lab" in helper
    assert "account=hoffman-lab" in helper
    assert "account=rl2-lab" in helper
    assert "gpu_type=a40" in helper
    assert "gpu_type=l40s" in helper
    assert '--partition="$partition"' in helper
    assert '--account="$account"' in helper
    assert '--gres="gpu:$gpu_type:$gpus"' in helper
    assert "--open-mode=append" in helper
    assert "afterok" not in helper
    assert "smoke_identity.tsv" in helper
    assert 'validation["status"] == "PASS"' in helper
    assert 'validation["gradient_clipping_enabled"] is False' in helper
    assert "full jobs are never chained automatically" in helper


def test_matrix_requeue_selects_newest_recovery_checkpoint() -> None:
    repo_root = Path(__file__).parents[1]
    train_hydra = (repo_root / "egomimic" / "trainHydra.py").read_text()
    matrix = (
        repo_root / "scripts" / "train" / "flow_transfer_direct_dense_matrix.sbatch"
    ).read_text()

    assert "SLURMEnvironment(requeue_signal=signal.SIGUSR1)" in train_hydra
    assert "plugins=plugins" in train_hydra
    assert "resuming from 'last.ckpt'" not in train_hydra
    assert 'RESUME_CANDIDATES=("$RUN_DIR"/hpc_ckpt_*.ckpt)' in matrix
    assert 'RESUME_CANDIDATES+=("$LAST_CKPT")' in matrix
    assert "path.stat().st_mtime_ns" in matrix
    assert 'COMMON_OVERRIDES+=("ckpt_path=$RESUME_CKPT")' in matrix
    assert '"++paths.root_dir=$RUN_DIR"' in matrix
    assert '"paths.output_dir=$RUN_DIR"' in matrix
    assert '"paths.work_dir=$REPO"' in matrix
    assert '--cfg job --resolve > "$RESOLVED_CONFIG"' in matrix


@pytest.mark.parametrize(
    ("experiment", "model_family"),
    [
        (
            "pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16",
            "latent",
        ),
        ("pusht/pipeline_diffusion_usocket_chain_newdata_h16", "dp"),
    ],
)
def test_newdata_cotrain_configs_are_h16_world2_and_config_only(
    experiment: str,
    model_family: str,
) -> None:
    cfg = _compose(experiment)
    expected_roots = [
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
    ]

    assert set(cfg.data.train_datasets) == {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    assert set(cfg.data.valid_datasets) == set(cfg.data.train_datasets)
    for split in ("train_datasets", "valid_datasets"):
        datasets = cfg.data[split]
        chain = datasets.pushshapes_sim_chain_gripper
        usocket = datasets.pushshapes_sim_u_socket
        assert chain.resolver._target_.endswith(
            "LocalEpisodeResolverManyWithEmbodimentOverride"
        )
        assert list(chain.resolver.folder_paths) == expected_roots
        assert chain.resolver.key_map.action_horizon == 16
        assert usocket.resolver.key_map.action_horizon == 16
        assert (
            usocket.resolver.folder_path == "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/"
            "u_socket_3000_v2_clean"
        )

    assert cfg.data.train_datasets.pushshapes_sim_u_socket.valid_ratio == 0.0
    assert cfg.data.train_datasets.pushshapes_sim_chain_gripper.valid_ratio == 0.0
    assert cfg.data.valid_datasets.pushshapes_sim_u_socket.valid_ratio == 0.02
    assert cfg.data.valid_datasets.pushshapes_sim_chain_gripper.valid_ratio == 0.02
    assert cfg.launch_params.gpus_per_node == 2
    assert cfg.launch_params.nodes == 1
    for domain in cfg.data.train_datasets:
        per_rank = cfg.data.train_dataloader_params[domain].batch_size
        assert per_rank == 32
        assert per_rank * cfg.launch_params.gpus_per_node == 64

    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.model.enable_grad_norm is False
    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.max_steps == 240_000
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)
    assert (
        cfg.model.scheduler._target_
        == "egomimic.utils.schedulers.warmup_cosine_scheduler"
    )
    assert cfg.model.get("scheduler_interval", "step") == "step"
    assert cfg.model.get("scheduler_frequency", 1) == 1
    assert (
        cfg.model.optimizer.lr * cfg.model.scheduler.warmup_start_factor
        == pytest.approx(cfg.model.scheduler.eta_min)
    )
    assert cfg.model.train_metrics_on_step is True
    assert cfg.model.train_metrics_on_epoch is True
    assert cfg.logger.wandb.project == "pushshapes-flow-transfer"
    assert not any(name.startswith("best_") for name in cfg.callbacks)

    model = cfg.model.robomimic_model
    assert model.action_horizon == 16
    assert set(model.domains) == {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    if model_family == "latent":
        assert model.stages[1].action_horizon == 16
        assert model.stages[1].latent_dim == 96
        assert model.stages[2].action_horizon == 16
        assert model.stages[2].denoising_module.act_seq == 16
        assert model.stages[2].gradient_accumulation_steps == 1
        assert "action_encoder" not in OmegaConf.to_yaml(cfg.model).lower()
    else:
        assert model.stages[1].action_horizon == 16
        assert set(model.stages[1].policies) == set(model.domains)
        for policy in model.stages[1].policies.values():
            assert policy.action_horizon == 16


@pytest.mark.parametrize(
    "experiment",
    [
        "pusht/pipeline_sampler_usocket_chain_newdata_cotrain12_per_emb_proprio_h16",
        "pusht/pipeline_sampler_usocket_chain_newdata_temporal_h8_l8_w256_d12_dec64_per_emb_proprio",
    ],
)
def test_current_flow_transfer_configs_use_frozen719_without_filter(
    experiment: str,
) -> None:
    cfg = _compose(experiment)
    expected_roots = [
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
        "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/"
        "chain_gripper_gen_flow_transfer_frozen719_20260829",
    ]
    for split in ("train_datasets", "valid_datasets"):
        chain = cfg.data[split].pushshapes_sim_chain_gripper
        assert list(chain.resolver.folder_paths) == expected_roots
        assert "filters" not in chain


def test_temporal_compression_config_denoises_h8_l8_and_outputs_native_h16() -> None:
    cfg = _compose(
        "pusht/pipeline_sampler_usocket_chain_newdata_temporal_h8_l8_w256_d12_dec64_per_emb_proprio"
    )
    model = cfg.model.robomimic_model
    projection = model.stages[0]
    fused = model.stages[1]
    noise = model.stages[2]
    sampler = model.stages[3]
    decoder = model.stages[4]
    denoiser = sampler.denoising_module

    assert model.action_horizon == 16
    assert projection._target_.endswith("EmbodimentProprioProjection")
    assert projection.output_dim == 64
    assert projection.projections.pushshapes_sim_u_socket.source_dim == 4
    assert projection.projections.pushshapes_sim_chain_gripper.source_dim == 6
    assert fused.required_obs_keys == ["front_img_1", "proprio_condition"]
    assert fused.encoder.obs_specs.proprio_condition.input_dim == 64
    assert noise.num_tokens == 8
    assert noise.latent_dim == 8
    assert sampler._target_.endswith("LatentFlowSampler")
    assert sampler.latent_dim == 8
    assert sampler.condition_input_dim == 128
    assert sampler.condition_dim == 256
    assert sampler.denoiser_hidden_dim == 256
    for decoder_field in (
        "action_horizon",
        "action_dims",
        "decoder_type",
        "decoder_hidden_dim",
        "latent_horizon",
    ):
        assert decoder_field not in sampler
    assert decoder._target_.endswith("PerEmbodimentActionDecoder")
    u_decoder = decoder.decoders.pushshapes_sim_u_socket
    chain_decoder = decoder.decoders.pushshapes_sim_chain_gripper
    for leaf, action_dim in ((u_decoder, 4), (chain_decoder, 6)):
        assert leaf._target_.endswith("TemporalConvActionDecoder")
        assert leaf.latent_dim == 8
        assert leaf.hidden_dim == 64
        assert leaf.action_dim == action_dim
        assert leaf.num_layers == 4
        assert leaf.project_to_action_before_temporal is True
        assert "extra_hidden_layers" not in leaf
        assert "latent_horizon" not in leaf
        assert "action_horizon" not in leaf
    assert denoiser.act_seq == 8
    assert denoiser.act_dim == 8
    assert denoiser.hidden_dim == 256
    assert denoiser.nblocks == 12
    assert set(model.rollout_observation_adapters) == {
        "pushshapes_sim_u_socket",
        "pushshapes_sim_chain_gripper",
    }
    assert model.rollout_adapters.pushshapes_sim_chain_gripper.action_horizon == 16
    for split in ("train_datasets", "valid_datasets"):
        datasets = cfg.data[split]
        assert datasets.pushshapes_sim_u_socket.resolver.key_map.action_horizon == 16
        assert (
            datasets.pushshapes_sim_chain_gripper.resolver.key_map.action_horizon == 16
        )
        assert list(datasets.pushshapes_sim_chain_gripper.resolver.folder_paths) == [
            "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
            "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/"
            "chain_gripper_gen_flow_transfer_frozen719_20260829",
        ]
        assert "filters" not in datasets.pushshapes_sim_chain_gripper
        assert datasets.pushshapes_sim_u_socket.resolver.key_map._target_.endswith(
            "get_keymap_hpt_per_emb_proprio"
        )
        assert (
            datasets.pushshapes_sim_u_socket.resolver.transform_list._target_.endswith(
                "get_usocket_rotvec_action_state_transform_list"
            )
        )
        assert datasets.pushshapes_sim_chain_gripper.resolver.key_map._target_.endswith(
            "get_keymap_hpt_per_emb_proprio"
        )

    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == 8
    assert cfg.evaluator.limit_val_batches == 8
    assert cfg.evaluator._target_.endswith("HumanRobotOverlayEval")
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)
    assert cfg.logger.wandb.project == "pushshapes-flow-transfer"


def test_newdata_world2_launcher_is_smoke_only_and_fail_closed() -> None:
    repo_root = Path(__file__).parents[1]
    launcher = (
        repo_root
        / "scripts"
        / "train"
        / "flow_transfer_newdata_h16_world2_matrix.sbatch"
    ).read_text()

    for value in (
        "ARM=${ARM:?",
        "EXPECTED_HEAD=${EXPECTED_HEAD:?",
        "EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?",
        "EXPECTED_SLURM_PARTITION=${EXPECTED_SLURM_PARTITION:?",
        "EXPECTED_SLURM_ACCOUNT=${EXPECTED_SLURM_ACCOUNT:?",
        "EXPECTED_GPU_MODEL=${EXPECTED_GPU_MODEL:?",
        "NORM_ARTIFACT=${NORM_ARTIFACT:?",
        "EXPECTED_NORM_SHA=${EXPECTED_NORM_SHA:?",
        "U_INVENTORY=${U_INVENTORY:?",
        "U_EPISODE_METADATA=${U_EPISODE_METADATA:?",
        "CHAIN_BASE_INVENTORY=${CHAIN_BASE_INVENTORY:?",
        "CHAIN_BASE_EPISODE_METADATA=${CHAIN_BASE_EPISODE_METADATA:?",
        "CHAIN_GEN_INVENTORY=${CHAIN_GEN_INVENTORY:?",
        "CHAIN_GEN_EPISODE_METADATA=${CHAIN_GEN_EPISODE_METADATA:?",
        "EXPECTED_U_INVENTORY_SHA=${EXPECTED_U_INVENTORY_SHA:?",
        "EXPECTED_U_EPISODE_METADATA_SHA=${EXPECTED_U_EPISODE_METADATA_SHA:?",
        "EXPECTED_U_TRAIN_FRAMES=${EXPECTED_U_TRAIN_FRAMES:?",
        "EXPECTED_CHAIN_BASE_INVENTORY_SHA=${EXPECTED_CHAIN_BASE_INVENTORY_SHA:?",
        "EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_BASE_EPISODE_METADATA_SHA:?",
        "EXPECTED_CHAIN_GEN_INVENTORY_SHA=${EXPECTED_CHAIN_GEN_INVENTORY_SHA:?",
        "EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA=${EXPECTED_CHAIN_GEN_EPISODE_METADATA_SHA:?",
        "EXPECTED_CHAIN_TRAIN_FRAMES=${EXPECTED_CHAIN_TRAIN_FRAMES:?",
    ):
        assert value in launcher
    assert "trap report_error ERR" in launcher
    assert "[smoke] FAIL line=%s status=%s command=%q" in launcher
    assert "[smoke] SOURCE_DATA_PASS" in launcher
    assert "[smoke] ALLOCATION_ENV_PASS" in launcher
    assert "pusht/pipeline_diffusion_usocket_chain_newdata_h16" in launcher
    assert "pusht/pipeline_sampler_usocket_chain_newdata_dense_medium_h16" in launcher
    assert (
        "pusht/pipeline_sampler_usocket_chain_newdata_temporal_h8_l8_w256_d12_dec64_per_emb_proprio"
        in launcher
    )
    assert "temporal_h8_l8)" in launcher
    assert 'decoder._target_.endswith("PerEmbodimentActionDecoder")' in launcher
    assert 'u_decoder._target_.endswith("TemporalConvActionDecoder")' in launcher
    assert "u_decoder.num_layers == chain_decoder.num_layers == 4" in launcher
    assert "sampler.decoder_type" not in launcher
    assert (
        "CHAIN_GEN_FROZEN_DATA=/coc/flash7/paphiwetsa3/datasets/Tsim_v2/"
        "chain_gripper_gen_flow_transfer_frozen719_20260829"
    ) in launcher
    assert "CHAIN_GEN_DATA=$CHAIN_GEN_FROZEN_DATA" in launcher
    assert "CHAIN_EXCLUDED_FRAMES=0" in launcher
    assert 'if arm == "temporal_h8_l8":' in launcher
    assert 'assert "filters" not in chain_dataset' in launcher
    assert '"$EXCLUDED_CHAIN_EPISODE" "$CHAIN_GEN_DATA"' in launcher
    assert 'test -z "$(git -C "$REPO" status --porcelain=v1' in launcher
    assert launcher.count("validate_all_inventories") >= 3
    assert 'episode / "zarr.json"' in launcher
    assert "hashlib.sha256(raw).hexdigest()" in launcher
    assert "metadata_path.read_bytes() == metadata_bytes" in launcher
    assert "live episode metadata differs from pinned artifact" in launcher
    assert "effective_chain_frames" in launcher
    assert "EXCLUDED_CHAIN_EPISODE=episode_T_chain_gripper_obs7_000050" in launcher
    assert "EXCLUDED_CHAIN_FRAMES=3118" in launcher
    assert "lambda row: row.get('episode_hash') != " in launcher
    assert 'for split in ("train_datasets", "valid_datasets")' in launcher
    assert 'filters._target_ == "egomimic.rldb.filters.DatasetFilter"' in launcher
    assert "list(filters.filter_lambdas) == [expected_chain_filter]" in launcher
    assert (
        'expected_frames = {"19": int(sys.argv[3]), "20": int(sys.argv[4])}' in launcher
    )
    assert 'payload["frames"] == sum(expected_frames.values())' in launcher
    assert (
        'metadata["total_dataset_frames"] == sum(expected_frames.values())' in launcher
    )
    assert '"19": {"state_agent_model": 4, "actions": 4}' in launcher
    assert '"20": {"state_agent_model": 6, "actions": 6}' in launcher
    assert "state_agent_obj" not in launcher
    assert "EXPECTED_WORLD_SIZE=2" in launcher
    assert 'test "${SLURM_NTASKS:?}" = "$EXPECTED_WORLD_SIZE"' in launcher
    assert 'test "${SLURM_JOB_PARTITION:?}" = "$EXPECTED_SLURM_PARTITION"' in launcher
    assert 'test "${SLURM_JOB_ACCOUNT:?}" = "$EXPECTED_SLURM_ACCOUNT"' in launcher
    assert 'grep -Fci "$EXPECTED_GPU_MODEL"' in launcher
    assert "allocation_contract.tsv" in launcher
    assert "trainer.max_steps=2" in launcher
    assert "trainer.limit_train_batches=2" in launcher
    assert "trainer.val_check_interval=1" in launcher
    assert "trainer.limit_val_batches=1" in launcher
    assert "trainer.num_sanity_val_steps=0" in launcher
    assert "trainer.log_every_n_steps=1" in launcher
    assert '"~callbacks.terminal_checkpoint"' in launcher
    assert 'assert "terminal_checkpoint" not in cfg.callbacks' in launcher
    assert '--expected-world-size "$EXPECTED_WORLD_SIZE"' in launcher
    assert launcher.count("--required-embodiments 19,20") == 2
    assert launcher.count("--dry-run") == 1
    dry_run_index = launcher.index("--dry-run")
    postflight_index = launcher.index("# A live append or artifact mutation")
    durable_result_index = launcher.index("# Persist PASS only after")
    assert dry_run_index < postflight_index < durable_result_index
    assert "MODE=${MODE:?" not in launcher
    assert '"$SLURM_BIN/sbatch"' not in launcher


def test_training_smoke_verifier_checks_world2_and_dense_step_history() -> None:
    verifier = (
        Path(__file__).parents[1] / "egomimic" / "scripts" / "verify_training_smoke.py"
    ).read_text()

    assert 'parser.add_argument("--expected-world-size"' in verifier
    assert "config.trainer.limit_train_batches" in verifier
    assert "config.model.train_metrics_on_step is True" in verifier
    assert 'checkpoint.get("optimizer_states", [])' in verifier
    assert 'checkpoint.get("lr_schedulers", [])' in verifier
    assert "scheduler_last_epoch == global_step" in verifier
    assert 'key.startswith("Timing/")' in verifier
    assert 'not key.endswith("_epoch")' in verifier
    assert '"Train/Loss"' in verifier
    assert '"Timing/Process_Batch_Sec"' in verifier
    assert '"Timing/Forward_Pass_Sec"' in verifier
    assert '"Timing/Compute_Losses_Sec"' in verifier
    assert '"Optimizer/param_group_0_lr"' in verifier
    assert "dense_training_history" in verifier
    assert "training_steps == list(range(expected_steps))" in verifier
    assert "all(math.isfinite(value) for value in values)" in verifier
    assert "scheduled_history" in verifier
