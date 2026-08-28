from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
CHAIN_DOMAIN = "pushshapes_sim_chain_gripper"
CHAIN_ROOTS = [
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_3000_v2",
    "/coc/flash7/paphiwetsa3/datasets/Tsim_v2/chain_gripper_gen",
]
CHAIN_FILTER = (
    "lambda row: row.get('episode_hash') != 'episode_T_chain_gripper_obs7_000050'"
)
BC_EXPERIMENTS = [
    "pusht/pipeline_sampler_chain_gripper_newdata_points_dense_medium_h16",
    "pusht/pipeline_diffusion_chain_gripper_newdata_points_h16",
]


def _compose(experiment: str, extra_overrides: list[str] | None = None):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}", *(extra_overrides or [])],
        )


@pytest.mark.parametrize("experiment", BC_EXPERIMENTS)
@pytest.mark.parametrize(
    ("mode", "overrides"),
    [
        (
            "smoke",
            [
                "trainer.max_steps=2",
                "callbacks.model_checkpoint.every_n_train_steps=2",
                "callbacks.model_checkpoint.train_time_interval=null",
                "callbacks.model_checkpoint.save_on_train_epoch_end=false",
                "callbacks.terminal_checkpoint.every_n_train_steps=1",
            ],
        ),
        (
            "full",
            [
                "trainer.max_steps=240000",
                "callbacks.model_checkpoint.every_n_train_steps=null",
                "callbacks.model_checkpoint.save_on_train_epoch_end=false",
            ],
        ),
    ],
)
def test_chain_bc_checkpoint_state_keys_are_unique(
    experiment: str, mode: str, overrides: list[str]
) -> None:
    cfg = _compose(experiment, overrides)
    cfg.paths.output_dir = "/tmp/flow_transfer_chain_bc_callback_state_keys"
    checkpoint = cfg.callbacks.model_checkpoint
    terminal = cfg.callbacks.terminal_checkpoint

    assert instantiate(checkpoint).state_key != instantiate(terminal).state_key
    if mode == "smoke":
        assert checkpoint.every_n_train_steps == 2
        assert checkpoint.train_time_interval is None
        assert terminal.every_n_train_steps == 1
    else:
        assert mode == "full"
        assert checkpoint.every_n_train_steps is None
        assert checkpoint.train_time_interval.hours == 1
        assert terminal.every_n_train_steps == cfg.trainer.max_steps == 240_000


@pytest.mark.parametrize("experiment", BC_EXPERIMENTS)
def test_chain_bc_newdata_contract_is_world1_batch64_h16(experiment: str) -> None:
    cfg = _compose(experiment)
    model = cfg.model.robomimic_model

    assert cfg.launch_params.gpus_per_node == 1
    assert cfg.launch_params.nodes == 1
    assert cfg.trainer.precision == "bf16"
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.limit_val_batches == 0
    assert cfg.trainer.get("gradient_clip_val") is None
    assert cfg.model.enable_grad_norm is False
    assert cfg.model.train_metrics_on_step is True
    assert cfg.model.train_metrics_on_epoch is True
    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.optimizer.weight_decay == pytest.approx(1.0e-4)
    assert cfg.model.scheduler.warmup_steps == 3_000
    assert cfg.model.scheduler.max_steps == 240_000
    assert cfg.model.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)

    assert list(model.domains) == [CHAIN_DOMAIN]
    assert model.action_horizon == 16
    assert set(cfg.data.train_datasets) == {CHAIN_DOMAIN}
    assert set(cfg.data.valid_datasets) == {CHAIN_DOMAIN}
    assert cfg.data.train_dataloader_params[CHAIN_DOMAIN].batch_size == 64
    assert cfg.data.valid_dataloader_params[CHAIN_DOMAIN].batch_size == 16

    for split_name, mode, ratio in (
        ("train_datasets", "train", 0.0),
        ("valid_datasets", "valid", 0.02),
    ):
        dataset = cfg.data[split_name][CHAIN_DOMAIN]
        assert dataset.mode == mode
        assert float(dataset.valid_ratio) == ratio
        assert list(dataset.resolver.folder_paths) == CHAIN_ROOTS
        assert dataset.resolver.key_map.action_horizon == 16
        assert dataset.resolver.key_map.action_zarr_key == "actions"
        assert dataset.resolver.transform_list._target_.endswith(
            "get_chain_gripper_point_transform_list"
        )
        assert float(dataset.resolver.transform_list.world_size) == 512.0
        assert list(dataset.filters.filter_lambdas) == [CHAIN_FILTER]


def test_chain_bc_data_subtree_matches_cotrain_chain_subtree() -> None:
    cotrain = _compose("pusht/pipeline_diffusion_usocket_chain_newdata_h16")

    for experiment in BC_EXPERIMENTS:
        bc = _compose(experiment)
        for split_name in ("train_datasets", "valid_datasets"):
            assert OmegaConf.to_container(
                bc.data[split_name][CHAIN_DOMAIN], resolve=True
            ) == OmegaConf.to_container(
                cotrain.data[split_name][CHAIN_DOMAIN], resolve=True
            )


def test_chain_bc_latent_is_decoder_only_medium_h16() -> None:
    cfg = _compose(
        "pusht/pipeline_sampler_chain_gripper_newdata_points_dense_medium_h16"
    )
    model = cfg.model.robomimic_model

    assert model.stages[1].action_horizon == 16
    assert model.stages[1].latent_dim == 96
    sampler = model.stages[2]
    assert sampler.action_horizon == 16
    assert sampler.action_dims[CHAIN_DOMAIN] == 6
    assert sampler.latent_dim == 96
    assert sampler.decoder_hidden_dim == 512
    assert sampler.denoiser_hidden_dim == 384
    assert sampler.denoising_module.hidden_dim == 384
    assert sampler.denoising_module.act_dim == 96
    assert sampler.denoising_module.act_seq == 16
    assert sampler.denoising_module.nblocks == 16
    assert "action_encoder" not in OmegaConf.to_yaml(cfg.model).lower()


def test_chain_bc_dp_is_epsilon_h16() -> None:
    cfg = _compose("pusht/pipeline_diffusion_chain_gripper_newdata_points_h16")
    model = cfg.model.robomimic_model
    stage = model.stages[1]
    policy = stage.policies[CHAIN_DOMAIN]

    assert stage.action_horizon == 16
    assert policy.action_horizon == 16
    assert policy.infer_ac_dims[CHAIN_DOMAIN] == 6
    assert policy.model.input_dim == 6
    assert policy.noise_scheduler.prediction_type == "epsilon"
