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
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.pushshapes import PlanarArcWaypointZeroRolloutAdapter
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.zarr.zarr_dataset_multi import (
    LocalEpisodeResolver,
    MultiDataset,
    episode_names_sha256,
    split_dataset_names,
)

CONFIG_DIR = Path(__file__).parents[1] / "egomimic" / "hydra_configs"
EXPERIMENT = "pusht/pipeline_sampler_usocket_chain_planar_v2_arc_D200_M100"
SMOKE_EXPERIMENT = f"{EXPERIMENT}_smoke"
DOMAINS = {"pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"}
TOKEN_HORIZON = 101


def _compose(experiment: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[f"+experiment={experiment}"],
        )


def _apply_resolver_transforms(resolver, actions: np.ndarray) -> np.ndarray:
    sample = {"actions": actions.copy()}
    for transform in resolver.transform_list:
        sample = transform.transform(sample)
    return sample["actions"]


def test_planar_v2_two_domain_config_contract() -> None:
    cfg = _compose(EXPERIMENT)
    model = cfg.model.robomimic_model
    noise = model.stages[1]
    sampler = model.stages[2]

    assert set(model.domains) == DOMAINS
    assert set(model.ac_keys) == DOMAINS
    assert set(cfg.data.train_datasets) == DOMAINS
    assert set(cfg.data.valid_datasets) == DOMAINS
    assert dict(sampler.action_dims) == {domain: 5 for domain in DOMAINS}
    assert sampler.schedule_anchor_domain == "pushshapes_sim_u_socket"

    assert model.action_horizon == TOKEN_HORIZON
    assert noise.action_horizon == TOKEN_HORIZON
    assert sampler.action_horizon == TOKEN_HORIZON
    assert sampler.denoising_module.act_seq == TOKEN_HORIZON
    assert noise.latent_dim == sampler.latent_dim == 96
    assert sampler.denoising_module.act_dim == 96
    assert sampler.denoising_module.hidden_dim == 384
    assert sampler.denoising_module.time_conditioning == "additive"
    assert sampler.num_inference_steps == 8
    assert (
        max(int(j) for phase in sampler.sampling_schedule.values() for j in phase) == 8
    )

    assert cfg.model.optimizer.lr == pytest.approx(3.0e-5)
    assert cfg.model.scheduler.warmup_steps == 3000
    assert cfg.model.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert cfg.model.scheduler.eta_min == pytest.approx(3.0e-6)
    assert cfg.model.train_metrics_on_step is True
    assert cfg.model.train_metrics_on_epoch is True
    assert cfg.launch_params.gpus_per_node == 2
    assert cfg.launch_params.nodes == 1
    assert cfg.trainer.max_steps == 240_000
    assert cfg.trainer.log_every_n_steps == 1
    assert cfg.trainer.limit_val_batches == pytest.approx(1.0)
    assert cfg.norm_stats.norm_mode == "quantile"
    assert cfg.norm_stats.reduce_all_but_last is False
    assert cfg.evaluator.log_normalized_mse is True
    assert cfg.evaluator.log_native_mse is True
    assert cfg.evaluator.limit_val_batches == pytest.approx(1.0)
    assert cfg.callbacks.terminal_checkpoint.every_n_train_steps == 240_000
    assert cfg.run_provenance.split_seed == cfg.seed == 42
    assert cfg.run_provenance.valid_ratio == pytest.approx(0.01)
    assert cfg.run_provenance.id_overlap_count == 0
    assert cfg.run_provenance.resolved_path_overlap_count == 0
    assert set(cfg.run_provenance.required_wandb_metrics) == {
        "Train/MSE",
        "Train/MSE/pushshapes_sim_u_socket",
        "Train/MSE/pushshapes_sim_chain_gripper",
        "Valid/MSE",
        "Valid/MSE/pushshapes_sim_u_socket",
        "Valid/MSE/pushshapes_sim_chain_gripper",
        "Valid/Native_MSE",
        "Valid/Native_MSE/pushshapes_sim_u_socket",
        "Valid/Native_MSE/pushshapes_sim_chain_gripper",
    }

    adapters = model.rollout_adapters
    assert set(adapters) == DOMAINS
    assert adapters.pushshapes_sim_u_socket.native_action_dim == 3
    assert adapters.pushshapes_sim_chain_gripper.native_action_dim == 4
    assert adapters.pushshapes_sim_u_socket.resampled_vector_length == 100
    assert adapters.pushshapes_sim_chain_gripper.resampled_vector_length == 100

    for split_name, mode in (("train_datasets", "train"), ("valid_datasets", "valid")):
        split = cfg.data[split_name]
        for domain in DOMAINS:
            dataset = split[domain]
            resolver = dataset.resolver
            transform = resolver.transform_list
            assert dataset.mode == mode
            assert dataset.valid_ratio == pytest.approx(0.01)
            assert dataset.split_seed == cfg.seed == 42
            assert dataset.expected_train_episode_count == 990
            assert dataset.expected_valid_episode_count == 10
            assert resolver.expected_episode_count == 1000
            assert "${PUSHSHAPES_ROOT}" not in str(resolver.folder_path)
            assert resolver.key_map.action_horizon == 200
            assert transform.min_distance_unit == pytest.approx(200.0)
            assert transform.resampled_vector_length == 100
            assert transform.rotation_radius == pytest.approx(30.0)
            assert transform.hybrid_rotation_unit is None
            assert transform.velocity_mode == "mean_scalar"
            assert transform.velocity_layout == "append"


def test_planar_v2_resolvers_emit_shared_tokens() -> None:
    cfg = _compose(EXPERIMENT)
    u_resolver = instantiate(cfg.data.train_datasets.pushshapes_sim_u_socket.resolver)
    chain_resolver = instantiate(
        cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
    )

    time = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    u_actions = np.column_stack((200.0 * time, 40.0 * time, -0.4 + 0.8 * time)).astype(
        np.float32
    )
    chain_actions = np.column_stack(
        (200.0 * time, 40.0 * time, -0.4 + 0.8 * time, 0.2 + 0.6 * time)
    ).astype(np.float32)

    u_token = _apply_resolver_transforms(u_resolver, u_actions)
    chain_token = _apply_resolver_transforms(chain_resolver, chain_actions)

    for token in (u_token, chain_token):
        assert token.shape == (TOKEN_HORIZON, 5)
        assert token.dtype == np.float32
        assert np.isfinite(token).all()
        np.testing.assert_allclose(
            np.square(token[:-1, 2]) + np.square(token[:-1, 3]),
            1.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(token[-1, 1:], 0.0, atol=1e-7)
        assert token[-1, 0] > 0.0

    np.testing.assert_allclose(
        u_token[0],
        [
            u_actions[0, 0],
            u_actions[0, 1],
            np.cos(u_actions[0, 2]),
            np.sin(u_actions[0, 2]),
            0.0,
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        chain_token[0],
        [
            chain_actions[0, 0],
            chain_actions[0, 1],
            np.cos(chain_actions[0, 2]),
            np.sin(chain_actions[0, 2]),
            chain_actions[0, 3],
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(u_token[:-1, 4], 0.0, atol=1e-7)


def test_planar_v2_rotation_and_stationary_grip_edge_cases() -> None:
    cfg = _compose(EXPERIMENT)
    resolver = instantiate(
        cfg.data.train_datasets.pushshapes_sim_chain_gripper.resolver
    )
    tokenizer = resolver.transform_list[0]

    theta = np.linspace(np.pi - 0.4, np.pi + 0.4, 200, dtype=np.float32)
    wrapped = np.angle(np.exp(1j * theta)).astype(np.float32)
    rotating = np.column_stack(
        (
            np.full(200, 100.0, dtype=np.float32),
            np.full(200, 200.0, dtype=np.float32),
            wrapped,
            np.full(200, 0.25, dtype=np.float32),
        )
    )
    rotation_token = tokenizer.tokenize_at(rotating, 0)

    assert rotation_token[-1, 0] > 0.0
    assert not np.allclose(rotation_token[0, 2:4], rotation_token[-2, 2:4])
    np.testing.assert_allclose(
        np.square(rotation_token[:-1, 2]) + np.square(rotation_token[:-1, 3]),
        1.0,
        atol=1e-5,
    )

    stationary = np.column_stack(
        (
            np.full(200, 100.0, dtype=np.float32),
            np.full(200, 200.0, dtype=np.float32),
            np.zeros(200, dtype=np.float32),
            np.concatenate(
                (np.zeros(100, dtype=np.float32), np.ones(100, dtype=np.float32))
            ),
        )
    )
    before_grip = tokenizer.tokenize_at(stationary, 0)
    after_grip = tokenizer.tokenize_at(stationary, 100)

    np.testing.assert_allclose(before_grip[:-1, 4], 0.0)
    np.testing.assert_allclose(after_grip[:-1, 4], 1.0)
    np.testing.assert_allclose(before_grip[-1], 0.0)
    np.testing.assert_allclose(after_grip[-1], 0.0)


def test_planar_v2_waypoint_zero_rollout_adapter() -> None:
    tokens = np.zeros((2, TOKEN_HORIZON, 5), dtype=np.float32)
    tokens[:, 0] = np.array(
        [[10.0, 20.0, 0.0, 1.0, 0.75], [30.0, 40.0, -1.0, 0.0, 0.25]],
        dtype=np.float32,
    )
    tokens[:, 1:] = 999.0

    u_adapter = PlanarArcWaypointZeroRolloutAdapter(100, 3)
    chain_adapter = PlanarArcWaypointZeroRolloutAdapter(100, 4)
    u_actions = u_adapter.decode(tokens)
    chain_actions = chain_adapter.decode(torch.from_numpy(tokens).to(torch.bfloat16))

    assert u_adapter.action_horizon == chain_adapter.action_horizon == 1
    assert u_actions.shape == (2, 1, 3)
    assert u_actions.dtype == np.float32
    np.testing.assert_allclose(
        u_actions[:, 0],
        [[10.0, 20.0, np.pi / 2.0], [30.0, 40.0, np.pi]],
        atol=1e-6,
    )
    assert chain_actions.shape == (2, 1, 4)
    assert chain_actions.dtype == torch.bfloat16
    torch.testing.assert_close(
        chain_actions.float()[:, 0],
        torch.tensor([[10.0, 20.0, np.pi / 2.0, 0.75], [30.0, 40.0, np.pi, 0.25]]),
        atol=0.02,
        rtol=0.0,
    )

    with pytest.raises(ValueError, match="expects"):
        u_adapter.decode(np.zeros((2, 100, 5), dtype=np.float32))


def test_planar_v2_split_manifest_matches_config() -> None:
    cfg = _compose(EXPERIMENT)
    path = (
        CONFIG_DIR / "data" / "pusht" / "planar_v2_usocket_chain_split_seed42_v1.json"
    )
    manifest = json.loads(path.read_text())

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        cfg.run_provenance.split_manifest_sha256
    )
    assert manifest["status"] == "PASS"
    assert manifest["split_seed"] == cfg.seed == 42
    assert manifest["valid_ratio"] == pytest.approx(0.01)
    assert manifest["cross_domain_train_valid_resolved_path_overlap_count"] == 0
    for domain in DOMAINS:
        entry = manifest["domains"][domain]
        dataset = cfg.data.train_datasets[domain]
        resolver = dataset.resolver
        assert entry["total_count"] == resolver.expected_episode_count == 1000
        assert entry["inventory_names_sha256"] == resolver.expected_episode_names_sha256
        assert entry["train_count"] == len(entry["train_ids"]) == 990
        assert entry["valid_count"] == len(entry["valid_ids"]) == 10
        assert entry["train_names_sha256"] == (
            dataset.expected_train_episode_names_sha256
        )
        assert entry["valid_names_sha256"] == (
            dataset.expected_valid_episode_names_sha256
        )
        assert episode_names_sha256(entry["train_ids"]) == entry["train_names_sha256"]
        assert episode_names_sha256(entry["valid_ids"]) == entry["valid_names_sha256"]
        assert set(entry["train_ids"]).isdisjoint(entry["valid_ids"])
        assert entry["id_overlap_count"] == 0
        assert entry["resolved_path_overlap_count"] == 0
        assert entry["union_count"] == 1000
        assert entry["union_matches_inventory"] is True
        assert len(set(entry["valid_ids"])) == 10
        assert all(episode.startswith("episode_T_") for episode in entry["valid_ids"])


def test_local_resolver_inventory_pin_fails_closed(tmp_path: Path) -> None:
    names = ["episode_000", "episode_001"]
    digest = hashlib.sha256("".join(f"{name}\n" for name in names).encode()).hexdigest()
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        expected_episode_count=2,
        expected_episode_names_sha256=digest,
    )
    paths = [(str(tmp_path / name), name) for name in reversed(names)]

    resolver._validate_expected_inventory(paths)
    with pytest.raises(ValueError, match="Expected 2 episodes"):
        resolver._validate_expected_inventory(paths[:1])
    resolver.expected_episode_names_sha256 = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolver._validate_expected_inventory(paths)


def test_local_resolver_pin_rejects_silent_zarr_load_failure(tmp_path: Path) -> None:
    names = {"episode_000", "episode_001"}
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        expected_episode_count=2,
        expected_episode_names_sha256=episode_names_sha256(names),
    )

    resolver._validate_loaded_inventory({name: object() for name in names}, names)
    with pytest.raises(ValueError, match="did not fully load"):
        resolver._validate_loaded_inventory({"episode_000": object()}, names)


def test_multidataset_split_hashes_fail_closed() -> None:
    datasets = {f"episode_{index:03d}": [index] for index in range(20)}
    train, valid = split_dataset_names(datasets, valid_ratio=0.2, seed=42)
    kwargs = {
        "datasets": datasets,
        "mode": "train",
        "valid_ratio": 0.2,
        "split_seed": 42,
        "expected_train_episode_count": len(train),
        "expected_train_episode_names_sha256": episode_names_sha256(train),
        "expected_valid_episode_count": len(valid),
        "expected_valid_episode_names_sha256": episode_names_sha256(valid),
    }

    dataset = MultiDataset(**kwargs)
    assert dataset.train_collections == train
    assert dataset.valid_collections == valid

    kwargs["expected_valid_episode_names_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="validation episode-ID SHA-256 mismatch"):
        MultiDataset(**kwargs)


def test_planar_v2_logs_required_normalized_and_native_mse() -> None:
    u_id = get_embodiment_id("pushshapes_sim_u_socket")
    chain_id = get_embodiment_id("pushshapes_sim_chain_gripper")
    target_u = torch.zeros(2, TOKEN_HORIZON, 5)
    target_chain = torch.zeros_like(target_u)
    target_u[:, 0, 2] = 1.0
    target_chain[:, 0, 2] = 1.0
    prediction_u = target_u.clone()
    prediction_chain = target_chain.clone()
    prediction_u[:, 0, 0] += 1.0
    prediction_chain[:, 0, 0] += 2.0

    class IdentityNorm:
        @staticmethod
        def unnormalize(value, _emb_id):
            return value

    adapters = {
        u_id: PlanarArcWaypointZeroRolloutAdapter(100, 3),
        chain_id: PlanarArcWaypointZeroRolloutAdapter(100, 4),
    }
    predictions = {
        f"emb{u_id}_actions": prediction_u,
        f"emb{chain_id}_actions": prediction_chain,
    }
    evaluator = HumanRobotOverlayEval(
        viz_func=None,
        log_normalized_mse=True,
        log_native_mse=True,
    )
    evaluator.model = SimpleNamespace(
        resolved_ac_keys={u_id: "actions", chain_id: "actions"},
        norm_stats=IdentityNorm(),
        forward_eval=lambda _batch: predictions,
        rollout_adapter_for=lambda emb_id: adapters[emb_id],
    )

    metrics, images = evaluator.compute_metrics_and_viz(
        {
            u_id: {"actions": target_u},
            chain_id: {"actions": target_chain},
        }
    )

    assert metrics["Valid/MSE/pushshapes_sim_u_socket"] == pytest.approx(1.0 / 505)
    assert metrics["Valid/MSE/pushshapes_sim_chain_gripper"] == pytest.approx(4.0 / 505)
    assert metrics["Valid/MSE"] == pytest.approx(5.0 / 1010)
    assert metrics["Valid/Native_MSE/pushshapes_sim_u_socket"] == pytest.approx(1.0 / 3)
    assert metrics["Valid/Native_MSE/pushshapes_sim_chain_gripper"] == pytest.approx(
        1.0
    )
    assert metrics["Valid/Native_MSE"] == pytest.approx(5.0 / 7)
    assert images == {}

    train_metrics = PipelineAlgo.log_info(
        SimpleNamespace(
            domain_by_id={
                u_id: "pushshapes_sim_u_socket",
                chain_id: "pushshapes_sim_chain_gripper",
            }
        ),
        {
            "losses": {
                "action_loss": torch.tensor(2.5),
                f"{u_id}_log_native_action": torch.tensor(1.0),
                f"{chain_id}_log_native_action": torch.tensor(4.0),
            }
        },
    )
    assert train_metrics["MSE"] == pytest.approx(2.5)
    assert train_metrics["MSE/pushshapes_sim_u_socket"] == pytest.approx(1.0)
    assert train_metrics["MSE/pushshapes_sim_chain_gripper"] == pytest.approx(4.0)


def test_planar_v2_full_and_smoke_preserve_model_data_and_evaluator() -> None:
    full = _compose(EXPERIMENT)
    smoke = _compose(SMOKE_EXPERIMENT)

    for key in ("model", "data", "evaluator"):
        assert OmegaConf.to_container(
            smoke[key], resolve=False
        ) == OmegaConf.to_container(full[key], resolve=False)
    assert smoke.trainer.precision == full.trainer.precision == "bf16"
    assert smoke.launch_params == full.launch_params
    assert smoke.trainer.max_steps == 2
    assert smoke.trainer.val_check_interval == 1
    assert smoke.trainer.limit_val_batches == 1
    assert smoke.callbacks.model_checkpoint.every_n_train_steps == 1
    assert smoke.callbacks.model_checkpoint.save_last is True
    assert smoke.callbacks.terminal_checkpoint.every_n_train_steps == 2
    assert OmegaConf.to_container(
        smoke.norm_stats, resolve=False
    ) == OmegaConf.to_container(full.norm_stats, resolve=False)
    assert smoke.norm_stats.sample_frac == pytest.approx(0.05)
