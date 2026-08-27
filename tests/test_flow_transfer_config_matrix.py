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


def test_pace_cotrain_helper_is_world1_batch64_per_domain_and_smoke_gated() -> None:
    repo_root = Path(__file__).parents[1]
    matrix = (
        repo_root / "scripts" / "train" / "flow_transfer_direct_dense_matrix.sbatch"
    ).read_text()
    helper = (
        repo_root
        / "scripts"
        / "train"
        / "submit_flow_transfer_pace_cotrain_when_ready.sh"
    ).read_text()

    assert "pace_world1" in matrix
    assert "PACE profile is cotrain-only" in matrix
    assert "64 // expected_gpus if cotrain else 64" in matrix
    assert "64 / GPUS_EXPECTED" in matrix
    assert "FLOW_TRANSFER_U_DATA" in matrix
    assert "data.train_datasets.pushshapes_sim_u_socket.resolver.folder_path" in matrix
    assert "world1_pace_normfix" in matrix
    assert "gts-dxu345-rl2" in matrix
    assert "A100:gpu-a100 | H200:gpu-h200" in matrix

    assert "ARMS=(cotrain_obstacle_latent cotrain_obstacle_dp)" in helper
    assert "bc_usocket" not in helper
    assert "bc_chain" not in helper
    assert "ACCOUNT=gts-dxu345-rl2" in helper
    assert "--ntasks-per-node=1" in helper
    assert '--gres="gpu:$GPU_TYPE:1"' in helper
    assert "PARTITION=gpu-a100" in helper
    assert "PARTITION=gpu-h200" in helper
    assert "GPU_CONSTRAINT=A100-80GB" in helper
    assert '--constraint="$GPU_CONSTRAINT"' in helper
    assert 'validation["expected_world_size"] == 1' in helper
    assert 'set(validation["global_batch_per_domain"].values()) == {64}' in helper
    assert 'validation["total_global_batch"] == 128' in helper
    assert 'validation["status"] == "PASS"' in helper
    assert "full jobs are never chained automatically" in helper
    assert "--account=rl2-dxu" not in helper


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
