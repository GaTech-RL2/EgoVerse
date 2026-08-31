import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from egomimic.eval.human_robot_overlay_eval import HumanRobotOverlayEval
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.pushshapes import (
    get_keymap_hpt,
    get_planar_dense_transform_list,
)

REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = REPO_ROOT / "egomimic" / "hydra_configs"
SEED_BANK = CONFIG_DIR / "evaluator" / "energy_score_seed_bank_k32_v1.json"
SEED_BANK_SHA256 = "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
SPECS = {
    "pipeline_sampler_usocket_planar_v2_arc_D50_M16": {
        "domain": "pushshapes_sim_u_socket",
        "distance": 50.0,
        "waypoints": 16,
        "horizon": 17,
        "native_dim": 3,
    },
    "pipeline_sampler_usocket_planar_v2_arc_D100_M25": {
        "domain": "pushshapes_sim_u_socket",
        "distance": 100.0,
        "waypoints": 25,
        "horizon": 26,
        "native_dim": 3,
    },
    "pipeline_sampler_chain_gripper_planar_v2_arc_D40_M16": {
        "domain": "pushshapes_sim_chain_gripper",
        "distance": 40.0,
        "waypoints": 16,
        "horizon": 17,
        "native_dim": 4,
    },
    "pipeline_sampler_chain_gripper_planar_v2_arc_D80_M25": {
        "domain": "pushshapes_sim_chain_gripper",
        "distance": 80.0,
        "waypoints": 25,
        "horizon": 26,
        "native_dim": 4,
    },
}
DP_SPECS = {
    "pipeline_diffusion_usocket_planar_v2_common5_h16": (
        "pushshapes_sim_u_socket",
        3,
    ),
    "pipeline_diffusion_chain_gripper_planar_v2_common5_h16": (
        "pushshapes_sim_chain_gripper",
        4,
    ),
}


def _compose(experiment: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment=pusht/{experiment}"],
        )


def _assert_shared_gate(cfg, domain: str) -> None:
    assert set(cfg.data.train_datasets) == set(cfg.data.valid_datasets) == {domain}
    for split_name, mode in (("train_datasets", "train"), ("valid_datasets", "valid")):
        dataset = cfg.data[split_name][domain]
        assert dataset.mode == mode
        assert dataset.valid_ratio == pytest.approx(0.01)
        assert dataset.split_seed == cfg.seed == 42
        assert dataset.expected_train_episode_count == 990
        assert dataset.expected_valid_episode_count == 10
        assert dataset.resolver.expected_episode_count == 1000
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == pytest.approx(1.0)
    assert cfg.model.optimizer.lr == pytest.approx(1.0e-4)
    if "ema_contract" in cfg.run_provenance:
        assert cfg.model.optimizer.weight_decay == pytest.approx(1.0e-6)
        assert list(cfg.model.optimizer.betas) == pytest.approx([0.95, 0.999])
        assert cfg.model.scheduler.warmup_steps == 500
        assert cfg.model.scheduler.eta_min == pytest.approx(0.0)
    else:
        assert cfg.model.optimizer.weight_decay == pytest.approx(1.0e-4)
        assert cfg.model.scheduler.warmup_steps == 3000
        assert cfg.model.scheduler.eta_min == pytest.approx(1.0e-5)
    assert cfg.launch_params.gpus_per_node == 2
    assert cfg.model.train_metrics_on_step is True
    assert cfg.evaluator.energy_score.sample_count == 32
    assert cfg.evaluator.energy_score.seed_bank_sha256 == SEED_BANK_SHA256
    assert set(cfg.evaluator.energy_score.distance_blocks) == {
        "translation",
        "rotation",
        "grip",
    }
    required = set(cfg.run_provenance.required_wandb_metrics)
    for prefix in (
        "Train/MSE",
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    ):
        assert prefix in required
        assert f"{prefix}/{domain}" in required
    for callback in (
        cfg.callbacks.model_checkpoint,
        cfg.callbacks.validation_checkpoint,
        cfg.callbacks.terminal_checkpoint,
    ):
        assert callback.save_top_k == -1
        assert callback.filename == "epoch-{epoch}-step-{step}"
        assert callback.enable_version_counter is False


@pytest.mark.parametrize("experiment", sorted(SPECS))
def test_arc_sixrun_contract(experiment: str) -> None:
    spec = SPECS[experiment]
    cfg = _compose(experiment)
    _assert_shared_gate(cfg, spec["domain"])
    model = cfg.model.robomimic_model
    noise, sampler = model.stages[1:3]
    assert model.action_horizon == spec["horizon"]
    assert noise.action_horizon == sampler.action_horizon == spec["horizon"]
    assert sampler.denoising_module.act_seq == spec["horizon"]
    assert sampler.latent_dim == noise.latent_dim == 96
    assert sampler.num_inference_steps == 8
    assert dict(sampler.action_dims) == {spec["domain"]: 5}
    adapter = model.rollout_adapters[spec["domain"]]
    assert adapter._target_.endswith(".PlanarArcWaypointZeroRolloutAdapter")
    assert adapter.resampled_vector_length == spec["waypoints"]
    assert adapter.native_action_dim == spec["native_dim"]
    transform = cfg.data.train_datasets[spec["domain"]].resolver.transform_list
    assert transform.min_distance_unit == pytest.approx(spec["distance"])
    assert transform.resampled_vector_length == spec["waypoints"]
    assert transform.rotation_radius == pytest.approx(30.0)
    assert cfg.run_provenance.norm_contract.domains[spec["domain"]].actions_shape == [
        spec["horizon"],
        5,
    ]
    smoke = _compose(f"{experiment}_smoke")
    for key in ("model", "data", "evaluator", "run_provenance"):
        assert OmegaConf.to_container(
            smoke[key], resolve=False
        ) == OmegaConf.to_container(cfg[key], resolve=False)
    assert smoke.trainer.max_steps == smoke.trainer.limit_train_batches == 2
    assert smoke.trainer.val_check_interval == smoke.trainer.limit_val_batches == 1


@pytest.mark.parametrize("experiment", sorted(DP_SPECS))
def test_genuine_dp_baseline_contract(experiment: str) -> None:
    domain, native_dim = DP_SPECS[experiment]
    cfg = _compose(experiment)
    _assert_shared_gate(cfg, domain)
    model = cfg.model.robomimic_model
    assert model.action_horizon == 16
    assert len(model.stages) == 2
    diffusion = model.stages[1]
    assert diffusion._target_.endswith(".MultiDomainDiffusionPolicyStage")
    policy = diffusion.policies[domain]
    assert policy._target_.endswith(".DiffusionPolicy")
    assert policy.model._target_.endswith(".PaperConditionalUnet1D")
    assert policy.model.input_dim == 5
    assert policy.model.global_cond_dim == 134
    assert policy.model.diffusion_step_embed_dim == 128
    assert policy.model.down_dims == [512, 1024, 2048]
    assert policy.model.kernel_size == 5
    assert model.stages[0].n_obs_steps == 2
    assert diffusion.condition_input_dim == 134
    assert (
        policy.noise_scheduler._target_
        == "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"
    )
    assert policy.noise_scheduler.prediction_type == "epsilon"
    assert policy.noise_scheduler.num_train_timesteps == 100
    assert policy.num_inference_steps == 100
    assert dict(policy.infer_ac_dims) == {domain: 5}
    adapter = model.rollout_adapters[domain]
    assert adapter._target_.endswith(".PlanarCommon5RolloutAdapter")
    assert adapter.action_horizon == 16
    assert adapter.native_action_dim == native_dim
    assert cfg.data.train_datasets[domain].resolver.key_map.action_horizon == 16
    assert cfg.data.train_datasets[domain].resolver.key_map.observation_horizon == 2
    assert cfg.data.train_datasets[domain].resolver.key_map.action_target_offset == 1
    assert cfg.data.train_dataloader_params[domain].batch_size == 32
    assert cfg.run_provenance.action_contract.execution_horizon == 8
    assert cfg.run_provenance.diffusion_contract.scheduler == "DDPM"
    assert cfg.run_provenance.diffusion_contract.unet_parameter_count == 251580037
    assert (
        cfg.run_provenance.diffusion_contract.online_model_parameter_count
        == 262777125
    )
    assert cfg.model.ema.enabled is True
    assert cfg.model.ema.power == pytest.approx(0.75)
    assert cfg.model.ema.max_value == pytest.approx(0.9999)
    assert cfg.run_provenance.norm_contract.domains[domain].actions_shape == [16, 5]
    assert not any(
        "GaussianLatentNoise" in str(stage._target_) for stage in model.stages
    )


def test_paper_dp_observation_and_action_windows_are_causally_aligned() -> None:
    keymap = get_keymap_hpt(
        action_horizon=16,
        observation_horizon=2,
        action_target_offset=1,
    )
    assert keymap["front_img_1"]["horizon"] == 2
    assert keymap["state_agent_obj"]["horizon"] == 2
    assert keymap["actions"]["horizon"] == 17

    native = np.zeros((17, 3), dtype=np.float32)
    native[:, 0] = np.arange(17)
    native[:, 1] = 100 + np.arange(17)
    native[:, 2] = np.linspace(-0.4, 0.4, 17)
    batch = {"actions": native.copy()}
    for transform in get_planar_dense_transform_list(
        action_horizon=16,
        action_target_offset=1,
    ):
        batch = transform.transform(batch)

    actions = batch["actions"]
    assert actions.shape == (16, 5)
    assert actions[0, 0] == pytest.approx(native[1, 0])
    assert actions[0, 1] == pytest.approx(native[1, 1])
    assert actions[0, 2] == pytest.approx(np.cos(native[1, 2]))
    assert actions[0, 3] == pytest.approx(np.sin(native[1, 2]))
    assert actions[0, 4] == pytest.approx(0.0)


def test_energy_score_seed_bank_and_prediction_artifact(tmp_path: Path) -> None:
    assert hashlib.sha256(SEED_BANK.read_bytes()).hexdigest() == SEED_BANK_SHA256
    domain = "pushshapes_sim_u_socket"
    emb_id = get_embodiment_id(domain)
    target = torch.zeros(2, 3, 5)
    evaluator = HumanRobotOverlayEval(
        viz_func=None,
        energy_score={
            "enabled": True,
            "sample_count": 32,
            "seed_bank_path": str(SEED_BANK),
            "seed_bank_sha256": SEED_BANK_SHA256,
            "action_dim": 5,
            "max_batches_per_rank": 1,
            "artifact_root": str(tmp_path),
            "validation_view": {"definition": "test"},
            "provenance": {"source": "unit-test"},
            "distance_blocks": {
                "translation": {"indices": [0, 1], "weight": 1.0},
                "rotation": {"indices": [2, 3], "weight": 1.0},
                "grip": {"indices": [4], "weight": 1.0},
            },
        },
    )
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: "actions"},
        forward_eval=lambda _batch: {f"emb{emb_id}_actions": torch.randn_like(target)},
    )
    evaluator.trainer = SimpleNamespace(
        global_step=7,
        current_epoch=2,
        global_rank=0,
        precision="bf16",
    )
    metrics = evaluator._energy_metrics_and_artifact({emb_id: {"actions": target}}, 0)
    assert set(metrics) == {
        "Valid/EnergyScore@32",
        f"Valid/EnergyScore@32/{domain}",
        "Valid/EnergyScoreAccuracy@32",
        f"Valid/EnergyScoreAccuracy@32/{domain}",
        "Valid/EnergyScoreDiversity@32",
        f"Valid/EnergyScoreDiversity@32/{domain}",
    }
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    artifacts = list(tmp_path.glob("epoch-2-step-7/rank-0-batch-0.pt"))
    assert len(artifacts) == 1
    payload = torch.load(artifacts[0], map_location="cpu", weights_only=False)
    assert payload["domains"][domain]["predictions"].shape == (32, 2, 3, 5)
