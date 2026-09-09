from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from egomimic.eval.planar_action_eval import PlanarActionEval


class _IdentityNormalizer:
    @staticmethod
    def unnormalize(values, embodiment_id):
        assert embodiment_id == 19
        return values


class _NestedRandomModel:
    @staticmethod
    def forward_eval(batch):
        return {
            source_id: {"pred_action": torch.rand_like(source_batch["actions"])}
            for source_id, source_batch in batch.items()
        }


def _write_seed_bank(tmp_path, seeds=range(32), filename="energy-seeds.json"):
    payload = json.dumps({"seeds": list(seeds)}, sort_keys=True).encode()
    path = tmp_path / filename
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _evaluator(tmp_path, **kwargs):
    path, digest = _write_seed_bank(tmp_path)
    return PlanarActionEval(
        seed_bank_path=str(path),
        seed_bank_sha256=digest,
        artifact_root=str(tmp_path / "artifacts"),
        semantic_blocks=((0, 2), (2, 4)),
        deterministic_seed=0,
        **kwargs,
    )


def _batch(target):
    return {
        "validation/usocket": {
            "actions": target,
            "embodiment": torch.full((target.shape[0],), 19),
        }
    }


def test_energy_seed_bank_hash_and_seed_identity_are_strict(tmp_path):
    path, digest = _write_seed_bank(tmp_path)
    with pytest.raises(ValueError, match="seed-bank identity mismatch"):
        PlanarActionEval(
            seed_bank_path=str(path),
            seed_bank_sha256="0" * 64,
            artifact_root=str(tmp_path / "bad-hash"),
        )

    duplicate_path, duplicate_digest = _write_seed_bank(
        tmp_path, [7] * 32, "duplicate-energy-seeds.json"
    )
    with pytest.raises(ValueError, match="32 unique seeds"):
        PlanarActionEval(
            seed_bank_path=str(duplicate_path),
            seed_bank_sha256=duplicate_digest,
            artifact_root=str(tmp_path / "duplicate-seeds"),
        )

    evaluator = PlanarActionEval(
        seed_bank_path=str(path),
        seed_bank_sha256=digest,
        artifact_root=str(tmp_path / "artifacts"),
        semantic_blocks=((0, 2), (2, 4)),
        deterministic_seed=0,
    )
    evaluator.model = _NestedRandomModel()
    target = torch.zeros(2, 2, 4)
    batch = _batch(target)

    torch.manual_seed(8675309)
    original_rng_state = torch.random.get_rng_state()
    sampled, first_result = evaluator._seeded_predictions(batch)
    sampled = sampled["validation/usocket"]
    actual_next_random = torch.rand(4)
    torch.random.set_rng_state(original_rng_state)
    expected_next_random = torch.rand(4)

    with torch.random.fork_rng():
        torch.manual_seed(evaluator.seeds[0])
        expected_first_sample = torch.rand_like(target)
    assert sampled.shape == (32, 2, 2, 4)
    torch.testing.assert_close(sampled[0], expected_first_sample)
    torch.testing.assert_close(
        first_result["validation/usocket"]["pred_action"], expected_first_sample
    )
    torch.testing.assert_close(actual_next_random, expected_next_random)


def test_energy_artifact_records_provenance_and_per_condition_outputs(tmp_path):
    validation_view = {
        "definition": "first_energy_score_batch_per_ddp_rank",
        "split_manifest_sha256": "split-sha",
        "per_rank_batch_size": 2,
        "world_size": 4,
    }
    provenance = {
        "action_representations": {
            "pushshapes_sim_u_socket": "x_y_cos_theta_sin_theta"
        },
        "prediction_horizon": 2,
        "sampler": "dopri5",
        "sampler_evaluation_points": 50,
        "sampler_atol": 1.0e-6,
        "sampler_rtol": 1.0e-3,
        "model_autocast_precision": "bf16",
        "dopri5_state_precision": "fp32",
        "dopri5_derivative_precision": "fp32",
        "dopri5_error_control_precision": "fp32",
    }
    evaluator = _evaluator(
        tmp_path,
        energy_score_max_batches_per_rank=1,
        energy_score_validation_view=validation_view,
        energy_score_provenance=provenance,
    )
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    target = torch.zeros(2, 2, 4)
    batch = _batch(target)
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source_id: {"pred_action": source_batch["actions"]}
            for source_id, source_batch in grouped.items()
        }
    )
    logged = {}
    evaluator.trainer = SimpleNamespace(
        current_epoch=3,
        global_step=41,
        global_rank=2,
        precision="bf16-mixed",
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.update(metrics)
        ),
    )

    samples = target.unsqueeze(0).repeat(32, 1, 1, 1)
    condition_values = torch.linspace(0.0, 1.0, 32)
    samples[:, 1] = condition_values[:, None, None]
    evaluator._seeded_predictions = lambda _batch: (
        {"validation/usocket": samples},
        {"validation/usocket": {"pred_action": samples[0]}},
    )

    evaluator.on_validation_step(batch, batch_idx=0)

    artifact_path = tmp_path / "artifacts" / "epoch-3-step-41" / "rank-2-batch-0.pt"
    assert artifact_path.is_file()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    assert artifact["schema_version"] == 1
    assert artifact["metric"] == "EnergyScore@32"
    assert artifact["sample_count"] == 32
    assert artifact["seed_bank"] == list(range(32))
    assert artifact["seed_bank_sha256"] == evaluator.seed_bank_sha256
    assert artifact["deterministic_seed"] == 0
    assert artifact["distance"] == {
        "space": "normalized_action_chunk",
        "formula": "mean_equal_weight_semantic_block_rms",
        "semantic_blocks": ((0, 2), (2, 4)),
    }
    assert artifact["aggregation"] == ("condition_mean_then_equal_domain_macro_mean")
    assert artifact["global_step"] == 41
    assert artifact["epoch"] == 3
    assert artifact["rank"] == 2
    assert artifact["batch_idx"] == 0
    assert artifact["precision"] == "bf16-mixed"
    assert artifact["validation_view"] == validation_view
    assert artifact["provenance"] == provenance

    domain = artifact["domains"]["pushshapes_sim_u_socket"]
    assert domain["source_id"] == "validation/usocket"
    assert domain["embodiment_id"] == 19
    assert domain["action_key"] == "actions"
    assert domain["predictions"].shape == (32, 2, 2, 4)
    assert domain["targets"].shape == (2, 2, 4)
    for name in (
        "accuracy_by_condition",
        "diversity_by_condition",
        "score_by_condition",
    ):
        assert domain[name].shape == (2,)
        assert bool(torch.isfinite(domain[name]).all())

    off_diagonal = ~torch.eye(32, dtype=torch.bool)
    expected_diversity = (
        (condition_values[:, None] - condition_values[None, :])
        .abs()[off_diagonal]
        .mean()
    )
    expected_accuracy = torch.tensor([0.0, condition_values.mean()])
    expected_diversity_by_condition = torch.tensor([0.0, expected_diversity])
    expected_score = expected_accuracy - 0.5 * expected_diversity_by_condition
    torch.testing.assert_close(domain["accuracy_by_condition"], expected_accuracy)
    torch.testing.assert_close(
        domain["diversity_by_condition"], expected_diversity_by_condition
    )
    torch.testing.assert_close(domain["score_by_condition"], expected_score)
    torch.testing.assert_close(logged["Valid/EnergyScore@32"], expected_score.mean())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluator.on_validation_step(batch, batch_idx=0)


def test_energy_batch_cap_does_not_cap_normal_mse_validation(tmp_path):
    evaluator = _evaluator(tmp_path, energy_score_max_batches_per_rank=1)
    evaluator.bind_data_context(normalizer=_IdentityNormalizer())
    target = torch.zeros(2, 2, 4)
    batch = _batch(target)
    evaluator.model = SimpleNamespace(
        forward_eval=lambda grouped: {
            source_id: {"pred_action": source_batch["actions"]}
            for source_id, source_batch in grouped.items()
        }
    )
    logged = []
    evaluator.trainer = SimpleNamespace(
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **_kwargs: logged.append(metrics)
        )
    )
    energy_calls = []
    evaluator._seeded_predictions = lambda _batch: (
        energy_calls.append(True)
        or (
            {"validation/usocket": target.unsqueeze(0).repeat(32, 1, 1, 1)},
            {"validation/usocket": {"pred_action": target}},
        )
    )
    evaluator._save_artifact = lambda *_args: None

    evaluator.on_validation_step(batch, batch_idx=1)
    evaluator.on_validation_step(batch, batch_idx=0)

    assert energy_calls == [True]
    assert "Valid/MSE" in logged[0] and "Valid/EnergyScore@32" not in logged[0]
    assert "Valid/MSE" in logged[1] and "Valid/EnergyScore@32" in logged[1]
