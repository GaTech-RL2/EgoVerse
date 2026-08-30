import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from egomimic.scripts import verify_training_full as verifier

HEAD = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _full_fixture(tmp_path: Path):
    run_dir = tmp_path / "run"
    config_path = run_dir / ".hydra" / "config.yaml"
    checkpoint_path = run_dir / "checkpoints" / "final" / "step-10.ckpt"
    wandb_stream = run_dir / "wandb" / "run-final" / "run-test.wandb"
    checkpoint_path.parent.mkdir(parents=True)
    wandb_stream.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"terminal checkpoint")
    wandb_stream.write_bytes(b"native wandb stream")

    norm_stats = {
        "19": {
            "state_agent_obj": {
                "quantile_1": [0.0, 1.0],
                "quantile_99": [2.0, 3.0],
            },
            "actions": {
                "quantile_1": [[0.0, 1.0], [2.0, 3.0]],
                "quantile_99": [[4.0, 5.0], [6.0, 7.0]],
            },
        }
    }
    norm_payload = {
        "norm_mode": "quantile",
        "reduce_all_but_last": False,
        "stats": norm_stats,
    }
    norm_artifact = _write_json(tmp_path / "bundle" / "norm_stats.json", norm_payload)
    bundle_manifest = _write_json(
        tmp_path / "bundle" / "manifest.json",
        {
            "source": {"head": HEAD, "clean": True},
            "outputs": {"full_run": str(run_dir)},
            "wandb": {
                "entity": "rl2-group",
                "project": "project",
                "group": "group",
                "full_id": "full-id",
            },
        },
    )
    required_metrics = ["Train/MSE", "Valid/MSE"]
    smoke_result = _write_json(
        tmp_path / "smoke" / "SMOKE_RESULT.json",
        {
            "status": "passed",
            "repo_head": HEAD,
            "world_size": 2,
            "global_step": 2,
            "wandb_exit_code": 0,
            "required_wandb_metrics": required_metrics,
            "model_wrapper_load": {
                "status": "passed",
                "strict": True,
                "map_location": "cpu",
            },
        },
    )

    provenance = {
        "norm_contract": {
            "norm_mode": "quantile",
            "reduce_all_but_last": False,
            "sample_frac": 0.05,
            "domains": {
                "domain_a": {
                    "embodiment_id": 19,
                    "state_agent_obj_shape": [2],
                    "actions_shape": [2, 2],
                }
            },
        },
        "training_contract": {
            "world_size": 2,
            "precision": "bf16",
            "max_steps": 10,
            "validation_interval_steps": 5,
            "validation_fraction": 1.0,
            "optimizer": "AdamW",
            "peak_lr": 3.0e-5,
            "warmup_steps": 2,
            "warmup_start_lr": 3.0e-6,
            "cosine_floor_lr": 3.0e-6,
        },
        "required_wandb_metrics": required_metrics,
    }
    config = OmegaConf.create(
        {
            "mode": "train",
            "ckpt_path": None,
            "paths": {
                "output_dir": str(run_dir),
                "work_dir": str(tmp_path / "repo"),
            },
            "launch_params": {"gpus_per_node": 2, "nodes": 1},
            "trainer": {
                "max_steps": 10,
                "devices": 2,
                "num_nodes": 1,
                "precision": "bf16",
                "val_check_interval": 5,
                "limit_val_batches": 1.0,
            },
            "norm_stats": {
                "norm_mode": "quantile",
                "reduce_all_but_last": False,
                "sample_frac": 0.05,
                "save_cache_dir": None,
                "precomputed_norm_path": str(norm_artifact),
            },
            "model": {
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 3.0e-5,
                },
                "scheduler": {
                    "warmup_steps": 2,
                    "warmup_start_factor": 0.1,
                    "eta_min": 3.0e-6,
                },
            },
            "logger": {
                "wandb": {
                    "entity": "rl2-group",
                    "project": "project",
                    "group": "group",
                    "id": "full-id",
                    "name": "full-id",
                    "resume": "never",
                }
            },
            "run_provenance": provenance,
        }
    )
    config_path.parent.mkdir(parents=True)
    OmegaConf.save(config, config_path)
    pinned_config = tmp_path / "bundle" / "resolved_full.yaml"
    OmegaConf.save(config, pinned_config)
    bundle_payload = json.loads(bundle_manifest.read_text())
    bundle_payload["configs"] = {
        "full": {
            "path": str(pinned_config),
            "sha256": _sha256(pinned_config),
            "bytes": pinned_config.stat().st_size,
        }
    }
    _write_json(bundle_manifest, bundle_payload)

    norm_state = {
        "norm_mode": "quantile",
        "reduce_all_but_last": False,
        "shapes": {19: {"state_agent_obj": (2,), "actions": (2, 2)}},
        "norm_stats": {
            19: {
                key: {
                    stat: np.asarray(values, dtype=np.float32)
                    for stat, values in per_stat.items()
                }
                for key, per_stat in norm_stats["19"].items()
            }
        },
    }
    checkpoint = {
        "global_step": 10,
        "epoch": 3,
        "state_dict": {"model.weight": torch.ones(1)},
        "optimizer_states": [{"param_groups": [{"lr": 3.0e-6}]}],
        "lr_schedulers": [{"last_epoch": 10}],
        "hyper_parameters": {
            "norm_stats_state": norm_state,
            "config_tree": {
                "model": OmegaConf.to_container(config.model, resolve=True),
                "run_provenance": provenance,
            },
        },
    }
    arguments = {
        "run_dir": run_dir,
        "expected_max_step": 10,
        "expected_head": HEAD,
        "expected_world_size": 2,
        "resolved_config": config_path,
        "expected_config_sha256": _sha256(config_path),
        "norm_artifact": norm_artifact,
        "expected_norm_sha256": _sha256(norm_artifact),
        "bundle_manifest": bundle_manifest,
        "expected_bundle_sha256": _sha256(bundle_manifest),
        "smoke_result": smoke_result,
        "expected_smoke_result_sha256": _sha256(smoke_result),
    }
    return arguments, checkpoint, checkpoint_path, wandb_stream


def _patch_runtime(monkeypatch, checkpoint, required_metrics):
    monkeypatch.setattr(verifier.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(verifier, "_git_identity", lambda _path: (HEAD, ""))
    monkeypatch.setattr(
        verifier,
        "_strict_load_model_wrapper",
        lambda _path: {
            "status": "passed",
            "map_location": "cpu",
            "strict": True,
            "state_dict_key_count": 1,
        },
    )
    monkeypatch.setattr(
        verifier,
        "_scan_wandb_stream",
        lambda _path, _required: (
            {
                metric: {
                    "count": 1,
                    "first_step": 10,
                    "last_step": 10,
                    "last": 0.25,
                }
                for metric in required_metrics
            },
            [0],
        ),
    )


def test_full_verifier_binds_terminal_checkpoint_and_all_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    arguments, checkpoint, checkpoint_path, wandb_stream = _full_fixture(tmp_path)
    _patch_runtime(monkeypatch, checkpoint, {"Train/MSE", "Valid/MSE"})

    record = verifier.verify_training_full(**arguments)

    assert record["status"] == "passed"
    assert record["repo_head"] == HEAD
    assert record["global_step"] == 10
    assert record["world_size"] == 2
    assert record["terminal_checkpoint"] == str(checkpoint_path)
    assert record["terminal_checkpoint_sha256"] == _sha256(checkpoint_path)
    assert record["resolved_config_sha256"] == arguments["expected_config_sha256"]
    assert record["norm_artifact_sha256"] == arguments["expected_norm_sha256"]
    assert record["bundle"]["sha256"] == arguments["expected_bundle_sha256"]
    assert record["smoke_result"]["sha256"] == arguments["expected_smoke_result_sha256"]
    assert record["optimizer_lrs"] == [3.0e-6]
    assert record["scheduler_last_epochs"] == [10]
    assert record["checkpoint_norm_state"]["stat_array_count"] == 4
    assert record["model_wrapper_load"]["strict"] is True
    assert record["wandb"]["successful_stream"] == str(wandb_stream)
    assert set(record["wandb"]["metric_evidence"]) == {
        "Train/MSE",
        "Valid/MSE",
    }


def test_runtime_config_allows_only_bound_checkpoint_resume_transition(
    tmp_path: Path,
) -> None:
    arguments, _checkpoint, _checkpoint_path, _wandb_stream = _full_fixture(tmp_path)
    run_dir = arguments["run_dir"]
    config_path = arguments["resolved_config"]
    resume = run_dir / "checkpoints" / "last.ckpt"
    resume.parent.mkdir(parents=True, exist_ok=True)
    resume.write_bytes(b"checkpoint")
    config = OmegaConf.load(config_path)
    config.ckpt_path = str(resume)
    config.logger.wandb.resume = "allow"
    bundle = json.loads(arguments["bundle_manifest"].read_text())

    result = verifier._validate_runtime_config_against_bundle(
        config,
        {"configs": bundle["configs"]},
        run_dir,
    )

    assert result["resume_checkpoint"] == str(resume.resolve())

    config.logger.wandb.resume = "never"
    with pytest.raises(AssertionError):
        verifier._validate_runtime_config_against_bundle(
            config,
            {"configs": bundle["configs"]},
            run_dir,
        )


def test_full_verifier_rejects_extra_terminal_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    arguments, checkpoint, checkpoint_path, _ = _full_fixture(tmp_path)
    _patch_runtime(monkeypatch, checkpoint, {"Train/MSE", "Valid/MSE"})
    (checkpoint_path.parent / "step-9.ckpt").write_bytes(b"stale")

    with pytest.raises(AssertionError):
        verifier.verify_training_full(**arguments)


def test_full_verifier_rejects_nonfinite_optimizer_lr(
    tmp_path: Path, monkeypatch
) -> None:
    arguments, checkpoint, _, _ = _full_fixture(tmp_path)
    checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] = float("nan")
    _patch_runtime(monkeypatch, checkpoint, {"Train/MSE", "Valid/MSE"})

    with pytest.raises(AssertionError):
        verifier.verify_training_full(**arguments)


def test_full_verifier_requires_every_configured_wandb_metric(
    tmp_path: Path, monkeypatch
) -> None:
    arguments, checkpoint, _, _ = _full_fixture(tmp_path)
    _patch_runtime(monkeypatch, checkpoint, {"Train/MSE"})

    with pytest.raises(AssertionError, match="missing required W&B metrics"):
        verifier.verify_training_full(**arguments)


def test_checkpoint_model_wrapper_is_loaded_strictly_on_cpu(monkeypatch) -> None:
    captured = {}

    class FakeModelWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)

        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, **kwargs):
            captured["checkpoint_path"] = checkpoint_path
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(verifier, "ModelWrapper", FakeModelWrapper)
    record = verifier._strict_load_model_wrapper(Path("/tmp/final.ckpt"))

    assert captured == {
        "checkpoint_path": "/tmp/final.ckpt",
        "map_location": "cpu",
        "weights_only": False,
        "strict": True,
    }
    assert record["status"] == "passed"
    assert record["strict"] is True
    assert record["map_location"] == "cpu"
    assert record["state_dict_key_count"] == 2
    assert record["parameter_count"] == 9


def test_full_result_write_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = {"status": "passed", "global_step": 10}

    result_path = verifier._write_atomic_result(run_dir, record)

    assert json.loads(result_path.read_text()) == record
    assert list(run_dir.glob(".FULL_RESULT.*.tmp")) == []
    with pytest.raises(AssertionError, match="Refusing to overwrite"):
        verifier._write_atomic_result(run_dir, record)
