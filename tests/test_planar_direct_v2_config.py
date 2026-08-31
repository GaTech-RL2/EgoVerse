import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.human_robot_overlay_eval import HumanRobotOverlayEval
from egomimic.pipeline.pushshapes import PlanarCommon5RolloutAdapter
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
HORIZON = 16
SPECS = {
    "pushshapes_sim_u_socket": {
        "direct": "pusht/pipeline_sampler_usocket_planar_v2_dense_h16",
        "arc": "pusht/pipeline_sampler_usocket_planar_v2_arc_D200_M100",
        "split": "planar_v2_usocket_split_seed42_v1.json",
        "native_dim": 3,
    },
    "pushshapes_sim_chain_gripper": {
        "direct": "pusht/pipeline_sampler_chain_gripper_planar_v2_dense_h16",
        "arc": "pusht/pipeline_sampler_chain_gripper_planar_v2_arc_D200_M100",
        "split": "planar_v2_chain_gripper_split_seed42_v1.json",
        "native_dim": 4,
    },
}


def _compose(experiment: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


def _apply_transforms(resolver, actions: np.ndarray) -> np.ndarray:
    sample = {"actions": actions.copy()}
    for transform in resolver.transform_list:
        sample = transform.transform(sample)
    return sample["actions"]


@pytest.mark.parametrize("domain", sorted(SPECS))
def test_direct_singleton_contract_and_arc_matching(domain: str) -> None:
    spec = SPECS[domain]
    cfg = _compose(spec["direct"])
    arc = _compose(spec["arc"])
    model = cfg.model.robomimic_model
    arc_model = arc.model.robomimic_model
    noise, sampler = model.stages[1:3]
    arc_noise, arc_sampler = arc_model.stages[1:3]

    assert list(model.domains) == [domain]
    assert set(model.ac_keys) == set(model.rollout_adapters) == {domain}
    assert set(cfg.data.train_datasets) == set(cfg.data.valid_datasets) == {domain}
    assert model.action_horizon == noise.action_horizon == sampler.action_horizon == HORIZON
    assert sampler.denoising_module.act_seq == HORIZON
    assert dict(sampler.action_dims) == {domain: 5}
    assert sampler.schedule_anchor_domain == domain
    assert model.rollout_adapters[domain]._target_.endswith(
        ".PlanarCommon5RolloutAdapter"
    )
    assert model.rollout_adapters[domain].action_horizon == HORIZON
    assert model.rollout_adapters[domain].native_action_dim == spec["native_dim"]

    # Capacity, condition path, curriculum, optimizer and train/validation
    # gates are inherited from the paired arc singleton. Only the temporal
    # target representation and its horizon differ.
    assert OmegaConf.to_container(model.stages[0], resolve=True) == (
        OmegaConf.to_container(arc_model.stages[0], resolve=True)
    )
    for field in (
        "condition_input_dim",
        "condition_dim",
        "gradient_accumulation_steps",
        "latent_dim",
        "decoder_hidden_dim",
        "denoiser_hidden_dim",
        "num_inference_steps",
        "sampling_schedule",
        "gradient_checkpointing",
    ):
        direct_value = sampler[field]
        arc_value = arc_sampler[field]
        if OmegaConf.is_config(direct_value):
            direct_value = OmegaConf.to_container(direct_value, resolve=True)
            arc_value = OmegaConf.to_container(arc_value, resolve=True)
        assert direct_value == arc_value
    for field in (
        "nblocks",
        "cond_dim",
        "hidden_dim",
        "act_dim",
        "n_heads",
        "dropout",
        "mlp_layers",
        "mlp_ratio",
        "time_conditioning",
    ):
        assert sampler.denoising_module[field] == arc_sampler.denoising_module[field]
    assert noise.latent_dim == arc_noise.latent_dim == sampler.latent_dim == 96
    assert cfg.model.optimizer == arc.model.optimizer
    assert cfg.model.scheduler == arc.model.scheduler
    assert cfg.trainer == arc.trainer
    assert cfg.launch_params == arc.launch_params
    assert cfg.evaluator == arc.evaluator

    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.max_steps == cfg.trainer.max_steps == 240_000
    assert cfg.model.scheduler.warmup_steps == 3000
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)
    assert cfg.launch_params.gpus_per_node == 2
    assert cfg.trainer.val_check_interval == 10_000
    assert cfg.trainer.limit_val_batches == pytest.approx(1.0)
    assert cfg.trainer.log_every_n_steps == 1

    for split_name, mode in (("train_datasets", "train"), ("valid_datasets", "valid")):
        dataset = cfg.data[split_name][domain]
        arc_dataset = arc.data[split_name][domain]
        assert dataset.mode == mode
        assert dataset.valid_ratio == arc_dataset.valid_ratio == pytest.approx(0.01)
        assert dataset.split_seed == arc_dataset.split_seed == cfg.seed == 42
        assert dataset.expected_train_episode_count == 990
        assert dataset.expected_valid_episode_count == 10
        assert dataset.expected_train_episode_names_sha256 == (
            arc_dataset.expected_train_episode_names_sha256
        )
        assert dataset.expected_valid_episode_names_sha256 == (
            arc_dataset.expected_valid_episode_names_sha256
        )
        assert dataset.resolver.folder_path == arc_dataset.resolver.folder_path
        assert dataset.resolver.expected_episode_count == 1000
        assert dataset.resolver.expected_episode_names_sha256 == (
            arc_dataset.resolver.expected_episode_names_sha256
        )
        assert dataset.resolver.key_map.action_horizon == HORIZON
        assert arc_dataset.resolver.key_map.action_horizon == 200
        assert dataset.resolver.transform_list._target_.endswith(
            ".get_planar_dense_transform_list"
        )

    assert cfg.run_provenance.split_manifest_path == (
        arc.run_provenance.split_manifest_path
    )
    assert cfg.run_provenance.split_manifest_sha256 == (
        arc.run_provenance.split_manifest_sha256
    )
    assert cfg.run_provenance.id_overlap_count == 0
    assert cfg.run_provenance.resolved_path_overlap_count == 0
    assert cfg.run_provenance.norm_contract.domains[domain].actions_shape == [16, 5]
    assert cfg.run_provenance.action_contract.model_action_shape == [16, 5]
    assert cfg.run_provenance.action_contract.native_action_shape == [
        16,
        spec["native_dim"],
    ]
    assert set(cfg.run_provenance.required_wandb_metrics) == {
        "Train/MSE",
        f"Train/MSE/{domain}",
        "Valid/MSE",
        f"Valid/MSE/{domain}",
        "Valid/Native_MSE",
        f"Valid/Native_MSE/{domain}",
    }

    split_path = CONFIG_DIR / "data" / "pusht" / spec["split"]
    manifest = json.loads(split_path.read_text())
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == (
        cfg.run_provenance.split_manifest_sha256
    )
    assert set(manifest["domains"]) == {domain}
    assert len(manifest["domains"][domain]["train_ids"]) == 990
    assert len(manifest["domains"][domain]["valid_ids"]) == 10

    smoke = _compose(f"{spec['direct']}_smoke")
    for key in ("model", "data", "evaluator"):
        assert OmegaConf.to_container(smoke[key], resolve=False) == (
            OmegaConf.to_container(cfg[key], resolve=False)
        )
    assert smoke.trainer.max_steps == smoke.trainer.limit_train_batches == 2
    assert smoke.trainer.val_check_interval == smoke.trainer.limit_val_batches == 1
    assert smoke.callbacks.model_checkpoint.every_n_train_steps == 1
    assert smoke.callbacks.terminal_checkpoint.every_n_train_steps == 2


@pytest.mark.parametrize("domain", sorted(SPECS))
def test_direct_resolver_emits_common5(domain: str) -> None:
    spec = SPECS[domain]
    cfg = _compose(spec["direct"])
    resolver = instantiate(cfg.data.train_datasets[domain].resolver)
    time = np.linspace(0.0, 1.0, HORIZON, dtype=np.float32)
    native = np.column_stack((100.0 * time, 20.0 * time, -0.5 + time))
    if spec["native_dim"] == 4:
        native = np.column_stack((native, 0.25 + 0.5 * time))
    common = _apply_transforms(resolver, native.astype(np.float32))

    assert common.shape == (HORIZON, 5)
    assert common.dtype == np.float32
    np.testing.assert_allclose(
        np.square(common[:, 2]) + np.square(common[:, 3]), 1.0, atol=1e-6
    )
    if spec["native_dim"] == 3:
        np.testing.assert_allclose(common[:, 4], 0.0)
    else:
        np.testing.assert_allclose(common[:, 4], native[:, 3])


def test_common5_adapter_preserves_full_dense_chunk_and_dtype() -> None:
    theta = np.linspace(-np.pi, np.pi, HORIZON, dtype=np.float32)
    common = np.zeros((2, HORIZON, 5), dtype=np.float32)
    common[..., 0] = np.arange(HORIZON, dtype=np.float32)
    common[..., 1] = 7.0
    common[..., 2] = np.cos(theta)
    common[..., 3] = np.sin(theta)
    common[..., 4] = 0.75

    u_adapter = PlanarCommon5RolloutAdapter(HORIZON, 3)
    chain_adapter = PlanarCommon5RolloutAdapter(HORIZON, 4)
    u_native = u_adapter.decode(common)
    chain_native = chain_adapter.decode(torch.from_numpy(common).to(torch.bfloat16))

    assert u_adapter.preserves_decoded_timing is True
    assert u_native.shape == (2, HORIZON, 3)
    assert u_native.dtype == np.float32
    np.testing.assert_allclose(u_native[..., :2], common[..., :2])
    np.testing.assert_allclose(
        np.angle(np.exp(1j * u_native[..., 2])),
        np.broadcast_to(np.angle(np.exp(1j * theta)), (2, HORIZON)),
        atol=1e-6,
    )
    assert chain_native.shape == (2, HORIZON, 4)
    assert chain_native.dtype == torch.bfloat16
    torch.testing.assert_close(
        chain_native.float()[..., 4 - 1],
        torch.full((2, HORIZON), 0.75),
        atol=0.01,
        rtol=0.0,
    )
    with pytest.raises(ValueError, match="expects"):
        u_adapter.decode(np.zeros((2, HORIZON - 1, 5), dtype=np.float32))


def test_native_mse_uses_the_full_dense_horizon() -> None:
    domain = "pushshapes_sim_u_socket"
    emb_id = get_embodiment_id(domain)
    target = torch.zeros(1, HORIZON, 5)
    target[..., 2] = 1.0
    prediction = target.clone()
    prediction[:, -1, 0] = 4.0

    class IdentityNorm:
        @staticmethod
        def unnormalize(value, _emb_id):
            return value

    evaluator = HumanRobotOverlayEval(
        viz_func=None,
        log_normalized_mse=True,
        log_native_mse=True,
    )
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={emb_id: "actions"},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: {f"emb{emb_id}_actions": prediction},
        rollout_adapter_for=lambda _emb_id: PlanarCommon5RolloutAdapter(HORIZON, 3),
    )

    metrics, _ = evaluator.compute_metrics_and_viz({emb_id: {"actions": target}})
    assert metrics[f"Valid/MSE/{domain}"] == pytest.approx(16.0 / (16 * 5))
    assert metrics[f"Valid/Native_MSE/{domain}"] == pytest.approx(16.0 / (16 * 3))
