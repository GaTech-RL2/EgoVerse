"""Verify a short real-data training smoke and emit a durable result record.

W&B 0.26 offline runs persist history in ``run-*.wandb`` and do not always
materialize ``files/wandb-summary.json``. This verifier reads the native W&B
stream so a smoke cannot fail merely because that compatibility file is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

import egomimic.utils.hydra_resolvers  # noqa: F401
from egomimic.rldb.embodiment.embodiment import get_embodiment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item_key(item: Any) -> str | None:
    if item.key:
        return item.key
    if item.nested_key:
        return ".".join(item.nested_key)
    return None


def _history_row(record: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for item in record.history.item:
        key = _item_key(item)
        if key is None:
            continue
        try:
            row[key] = json.loads(item.value_json)
        except (json.JSONDecodeError, TypeError):
            continue
    return row


def read_wandb_history(
    stream_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return aggregated train/validation rows and the terminal W&B exit code."""

    store = DataStore()
    store.open_for_scan(str(stream_path))
    history_by_step: dict[int, dict[str, float]] = {}
    exit_codes: list[int] = []
    try:
        while True:
            payload = store.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            record_type = record.WhichOneof("record_type")
            if record_type == "exit":
                exit_codes.append(int(record.exit.exit_code))
                continue
            if record_type != "history":
                continue

            row = _history_row(record)
            step = row.get("trainer/global_step")
            if step is None:
                continue
            step = int(step)
            metrics = history_by_step.setdefault(step, {})
            for key, value in row.items():
                if not (
                    key.startswith("Train/")
                    or key.startswith("Timing/")
                    or key.startswith("Optimizer/")
                    or key.startswith("Valid/")
                    or key.startswith("log/")
                ):
                    continue
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
    finally:
        store.close()

    assert exit_codes, f"No terminal W&B exit record in {stream_path}"
    assert exit_codes[-1] == 0, (stream_path, exit_codes)

    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for step, metrics in sorted(history_by_step.items()):
        train_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("Train/") and not key.endswith("_epoch")
        }
        timing_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("Timing/") and not key.endswith("_epoch")
        }
        optimizer_metrics = {
            key: value
            for key, value in metrics.items()
            if key.startswith("Optimizer/") and not key.endswith("_epoch")
        }
        telemetry_metrics = {
            key: value for key, value in metrics.items() if key.startswith("log/")
        }
        if train_metrics or timing_metrics or optimizer_metrics or telemetry_metrics:
            training_rows.append(
                {
                    "trainer_global_step": step,
                    "train_metrics": train_metrics,
                    "timing_metrics": timing_metrics,
                    "optimizer_metrics": optimizer_metrics,
                    "telemetry_metrics": telemetry_metrics,
                }
            )
        validation_metrics = {
            key: value for key, value in metrics.items() if key.startswith("Valid/")
        }
        if validation_metrics:
            validation_rows.append(
                {
                    "trainer_global_step": step,
                    "validation_metrics": validation_metrics,
                }
            )
    return training_rows, validation_rows, exit_codes[-1]


def read_wandb_validation(
    stream_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Backward-compatible validation-only view of the W&B history."""

    _, validation_rows, exit_code = read_wandb_history(stream_path)
    return validation_rows, exit_code


def _has_required_metrics(
    metrics: dict[str, float], required_embodiments: list[int]
) -> bool:
    required_overall = {
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    }
    if not required_overall.issubset(metrics):
        return False
    for embodiment in required_embodiments:
        prefix = f"Valid/emb{embodiment}_"
        if not any(key.startswith(prefix) for key in metrics):
            return False
        name = get_embodiment(embodiment).lower()
        required_domain = {
            f"Valid/MSE/{name}",
            f"Valid/Native_MSE/{name}",
            f"Valid/EnergyScore@32/{name}",
            f"Valid/EnergyScoreAccuracy@32/{name}",
            f"Valid/EnergyScoreDiversity@32/{name}",
        }
        if not required_domain.issubset(metrics):
            return False
    return True


def _validate_energy_score_artifacts(
    output_dir: Path,
    config: Any,
    required_embodiments: list[int],
    expected_world_size: int,
    selected_step: int,
) -> list[dict[str, Any]]:
    energy = OmegaConf.select(config, "evaluator.energy_score", default=None)
    assert energy is not None and energy.enabled is True
    assert int(energy.sample_count) == 32
    seed_bank_path = Path(str(energy.seed_bank_path)).resolve()
    assert seed_bank_path.is_file(), seed_bank_path
    assert _sha256(seed_bank_path) == str(energy.seed_bank_sha256)
    expected_seeds = json.loads(seed_bank_path.read_text())["seeds"]
    assert len(expected_seeds) == 32 and len(set(expected_seeds)) == 32

    root = output_dir / "validation_predictions" / "energy_score"
    candidates = sorted(root.glob(f"epoch-*-step-{selected_step}/rank-*-batch-0.pt"))
    assert len(candidates) == expected_world_size, candidates
    expected_domains = {
        get_embodiment(embodiment).lower() for embodiment in required_embodiments
    }
    records = []
    observed_ranks = set()
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["schema_version"] == 1
        assert payload["metric"] == "EnergyScore@32"
        assert payload["sample_count"] == 32
        assert payload["seed_bank"] == expected_seeds
        assert payload["seed_bank_sha256"] == str(energy.seed_bank_sha256)
        assert payload["global_step"] == selected_step
        assert set(payload["domains"]) == expected_domains
        assert payload["rank"] not in observed_ranks
        observed_ranks.add(payload["rank"])
        for domain, artifact in payload["domains"].items():
            predictions = artifact["predictions"]
            targets = artifact["targets"]
            assert predictions.ndim == 4 and predictions.shape[0] == 32
            assert targets.ndim == 3 and predictions.shape[1:] == targets.shape
            expected_dim = int(energy.action_dims[domain])
            assert predictions.shape[-1] == expected_dim
            for key in (
                "accuracy_by_condition",
                "diversity_by_condition",
                "score_by_condition",
            ):
                values = artifact[key]
                assert values.shape == targets.shape[:1]
                assert bool(torch.isfinite(values).all()), (path, domain, key)
        records.append({"path": str(path), "sha256": _sha256(path)})
    assert observed_ranks == set(range(expected_world_size)), observed_ranks
    return records


def _load_training_config(config_path: Path):
    """Load a Hydra snapshot with the same resolvers used by trainHydra."""
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    return OmegaConf.load(config_path)


def _require_finite(metrics: dict[str, float], keys: set[str]) -> None:
    missing = keys.difference(metrics)
    assert not missing, ("missing metrics", sorted(missing), sorted(metrics))
    invalid = {
        key: metrics[key]
        for key in keys
        if not math.isfinite(float(metrics[key]))
    }
    assert not invalid, ("non-finite metrics", invalid)


def _verify_wandb_visibility(run_path: str, required: set[str]) -> None:
    """Fail closed unless every required metric is visible in the exact run."""

    import wandb

    assert run_path.count("/") == 2, run_path
    missing = set(required)
    last_error = None
    for attempt in range(6):
        try:
            run = wandb.Api(timeout=30).run(run_path)
            assert run.path[-3:] == run_path.split("/"), (run.path, run_path)
            for row in run.scan_history(keys=sorted(required), page_size=1000):
                for key in tuple(missing):
                    value = row.get(key)
                    if value is None:
                        continue
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(numeric):
                        missing.discard(key)
            if not missing:
                return
        except Exception as error:
            last_error = repr(error)
        if attempt < 5:
            time.sleep(5)
    raise AssertionError(
        ("metrics not visible in exact W&B run", run_path, sorted(missing), last_error)
    )


def _verify_released_sweep_smoke(
    output_dir: Path,
    config: Any,
    required_embodiments: list[int],
    expected_head: str,
    expected_strategy: str,
    expected_world_size: int,
    expected_steps: int,
    expected_val_check_interval: int,
    minimum_validation_step: int,
    topology: str,
    sweep_task_id: str,
    latent_dim: int,
    num_latent_tokens: int,
    wandb_run_path: str,
    parameter_manifest: Path,
    split_manifest: Path,
    normalization_artifact: Path,
) -> dict[str, Any]:
    """Strict three-step joint-update gate for one register-sweep row."""

    assert topology in {"shared", "separate"}
    assert latent_dim == 16
    assert num_latent_tokens in {4, 8}
    assert expected_steps == expected_val_check_interval == minimum_validation_step == 3
    assert int(config.trainer.max_steps) == int(config.trainer.limit_train_batches) == 3
    assert int(config.trainer.val_check_interval) == 3
    assert int(config.trainer.limit_val_batches) == 1
    assert int(config.trainer.num_sanity_val_steps) == 0
    assert str(config.trainer.precision) == "bf16"
    assert str(config.trainer.strategy) == expected_strategy
    assert expected_strategy == "ddp_find_unused_parameters_true"
    assert int(config.trainer.devices) == expected_world_size == 2
    assert int(config.trainer.num_nodes) == 1
    assert int(config.trainer.accumulate_grad_batches) == 1
    assert config.model.share_encoder_denoiser is (topology == "shared")
    assert int(config.model.latent_dim) == 16
    assert int(config.model.num_latent_tokens) == num_latent_tokens
    assert int(config.model.unite_flow_updates_per_reconstruction) == 0
    assert int(config.model.unite_gradient_telemetry_every_n_steps) == 3
    assert set(config.data.train_datasets) == {"pushshapes_sim_u_socket"}
    assert set(config.data.valid_datasets) == {"pushshapes_sim_u_socket"}
    energy = config.evaluator.energy_score
    assert energy.enabled is True and int(energy.sample_count) == 32
    assert str(energy.seed_bank_sha256) == (
        "88657b829905d4374823db145ded19b99cec4735f76694734473bcee068bb5b6"
    )
    assert set(energy.action_dims) == {"pushshapes_sim_u_socket"}
    provenance = config.run_provenance
    assert str(provenance.sweep_task_id) == sweep_task_id
    assert str(provenance.topology) == topology
    assert int(provenance.num_latent_tokens) == num_latent_tokens
    assert int(provenance.latent_dim) == 16

    parameter_manifest = parameter_manifest.resolve()
    split_manifest = split_manifest.resolve()
    normalization_artifact = normalization_artifact.resolve()
    for path in (parameter_manifest, split_manifest, normalization_artifact):
        assert path.is_file(), path
    parameter_payload = json.loads(parameter_manifest.read_text())
    assert parameter_payload["topology"] == topology
    assert parameter_payload["num_latent_tokens"] == num_latent_tokens
    assert parameter_payload["latent_dim"] == 16
    assert parameter_payload["action_horizon"] == 16

    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file(), checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(checkpoint["global_step"])
    assert global_step == 3
    optimizer_states = checkpoint.get("optimizer_states", [])
    assert len(optimizer_states) == 1
    composite = optimizer_states[0]
    assert {"adamw", "muon", "group_manifest"}.issubset(composite)
    optimizer_lrs = {
        "AdamW": [
            float(group["lr"]) for group in composite["adamw"]["param_groups"]
        ],
        "Muon": [
            float(group["lr"]) for group in composite["muon"]["param_groups"]
        ],
    }
    assert all(
        values and all(math.isfinite(value) and value > 0 for value in values)
        for values in optimizer_lrs.values()
    )
    schedulers = checkpoint.get("lr_schedulers", [])
    assert len(schedulers) == 1 and int(schedulers[0]["last_epoch"]) == global_step
    ema = checkpoint.get("ema_state_dict")
    assert ema and all(torch.isfinite(value).all() for value in ema.values())
    assert math.isclose(float(checkpoint["ema_decay"]), 0.9978, abs_tol=1.0e-12)
    assert int(checkpoint["ema_num_updates"]) == global_step
    assert checkpoint["ema_validate_with_ema"] is True
    del checkpoint

    streams = [
        *output_dir.glob("wandb/offline-run-*/run-*.wandb"),
        *output_dir.glob("wandb/run-*/run-*.wandb"),
    ]
    assert len(streams) == 1, streams
    training_history, validation_history, wandb_exit_code = read_wandb_history(
        streams[0]
    )
    train_required = {
        "Train/UNITE/TotalLoss",
        "Train/UNITE/ReconstructionLoss",
        "Train/UNITE/FlowLoss",
        "Train/UNITE/ReconstructionL1",
        "Train/MSE",
        "Train/MSE/pushshapes_sim_u_socket",
    }
    optimizer_required = {"Optimizer/LR/AdamW", "Optimizer/LR/Muon"}
    dense = []
    optimizer_rows = []
    for row in training_history:
        if train_required.issubset(row["train_metrics"]):
            _require_finite(row["train_metrics"], train_required)
            dense.append(row)
        if optimizer_required.issubset(row["optimizer_metrics"]):
            _require_finite(row["optimizer_metrics"], optimizer_required)
            assert all(
                row["optimizer_metrics"][key] > 0 for key in optimizer_required
            )
            optimizer_rows.append(row)
    assert len(dense) == 3, training_history
    assert optimizer_rows, training_history

    if topology == "shared":
        telemetry_required = {
            "log/unite_gradient_cosine",
            "log/unite_recon_grad_norm",
            "log/unite_denoise_grad_norm",
        }
        telemetry_forbidden = {
            "log/unite_tokenizer_recon_grad_norm",
            "log/unite_denoiser_flow_grad_norm",
        }
        gradient_cosine_status = "required_finite"
    else:
        telemetry_required = {
            "log/unite_tokenizer_recon_grad_norm",
            "log/unite_denoiser_flow_grad_norm",
        }
        telemetry_forbidden = {
            "log/unite_gradient_cosine",
            "log/unite_recon_grad_norm",
            "log/unite_denoise_grad_norm",
        }
        gradient_cosine_status = "not_applicable_no_shared_parameters"
    telemetry_rows = []
    for row in training_history:
        metrics = row["telemetry_metrics"]
        assert telemetry_forbidden.isdisjoint(metrics), (topology, metrics)
        if telemetry_required.issubset(metrics):
            _require_finite(metrics, telemetry_required)
            norm_keys = {
                key for key in telemetry_required if key.endswith("grad_norm")
            }
            assert all(metrics[key] > 0 for key in norm_keys)
            if topology == "shared":
                assert -1.000001 <= metrics["log/unite_gradient_cosine"] <= 1.000001
            telemetry_rows.append(row)
    assert telemetry_rows, training_history

    valid_required = {
        "Valid/UNITE/TotalLoss",
        "Valid/UNITE/ReconstructionLoss",
        "Valid/UNITE/FlowLoss",
        "Valid/UNITE/ReconstructionL1",
        "Valid/UNITE/ReconstructionNativeMSE",
        "Valid/UNITE/ReconstructionNativeL1",
        "Valid/MSE",
        "Valid/MSE/pushshapes_sim_u_socket",
        "Valid/Native_MSE",
        "Valid/Native_MSE/pushshapes_sim_u_socket",
        "Valid/EnergyScore@32",
        "Valid/EnergyScore@32/pushshapes_sim_u_socket",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreAccuracy@32/pushshapes_sim_u_socket",
        "Valid/EnergyScoreDiversity@32",
        "Valid/EnergyScoreDiversity@32/pushshapes_sim_u_socket",
    }
    candidates = [
        row
        for row in validation_history
        if row["trainer_global_step"] + 1 >= minimum_validation_step
        and valid_required.issubset(row["validation_metrics"])
    ]
    assert candidates, validation_history
    selected = candidates[-1]
    _require_finite(selected["validation_metrics"], valid_required)
    completed_steps = selected["trainer_global_step"] + 1
    energy_artifacts = _validate_energy_score_artifacts(
        output_dir,
        config,
        required_embodiments,
        expected_world_size,
        completed_steps,
    )
    all_required = (
        train_required | optimizer_required | valid_required | telemetry_required
    )
    _verify_wandb_visibility(wandb_run_path, all_required)

    return {
        "status": "passed",
        "repo_head": expected_head,
        "output": str(output_dir),
        "config": str(output_dir / ".hydra" / "config.yaml"),
        "config_sha256": _sha256(output_dir / ".hydra" / "config.yaml"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "global_step": global_step,
        "precision": str(config.trainer.precision),
        "world_size": expected_world_size,
        "trainer_strategy": str(config.trainer.strategy),
        "required_embodiments": required_embodiments,
        "sweep_task_id": sweep_task_id,
        "unite_topology": topology,
        "num_latent_tokens": num_latent_tokens,
        "latent_dim": 16,
        "action_horizon": 16,
        "gradient_cosine_status": gradient_cosine_status,
        "gradient_telemetry": telemetry_rows,
        "optimizer_lrs": optimizer_lrs,
        "validation_metrics": selected["validation_metrics"],
        "energy_score_artifacts": energy_artifacts,
        "parameter_manifest": str(parameter_manifest),
        "parameter_manifest_sha256": _sha256(parameter_manifest),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "normalization_artifact": str(normalization_artifact),
        "normalization_sha256": _sha256(normalization_artifact),
        "wandb_run_path": wandb_run_path,
        "wandb_visibility": "passed",
        "wandb_stream": str(streams[0]),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_exit_code": wandb_exit_code,
        "training_history": training_history,
        "validation_history": validation_history,
    }


def verify_training_smoke(
    output_dir: Path,
    required_embodiments: list[int],
    expected_head: str,
    expected_strategy: str = "ddp",
    expected_world_size: int = 1,
    expected_steps: int = 2,
    expected_val_check_interval: int = 1,
    minimum_validation_step: int = 1,
    released_sweep_topology: str | None = None,
    expected_sweep_task_id: str | None = None,
    expected_latent_dim: int | None = None,
    expected_num_latent_tokens: int | None = None,
    expected_wandb_run_path: str | None = None,
    parameter_manifest: Path | None = None,
    split_manifest: Path | None = None,
    normalization_artifact: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_path = output_dir / ".hydra" / "config.yaml"
    assert config_path.is_file(), config_path
    config = _load_training_config(config_path)

    if released_sweep_topology is not None:
        assert expected_sweep_task_id
        assert expected_latent_dim is not None
        assert expected_num_latent_tokens is not None
        assert expected_wandb_run_path
        assert parameter_manifest is not None
        assert split_manifest is not None
        assert normalization_artifact is not None
        return _verify_released_sweep_smoke(
            output_dir,
            config,
            required_embodiments,
            expected_head,
            expected_strategy,
            expected_world_size,
            expected_steps,
            expected_val_check_interval,
            minimum_validation_step,
            released_sweep_topology,
            expected_sweep_task_id,
            expected_latent_dim,
            expected_num_latent_tokens,
            expected_wandb_run_path,
            parameter_manifest,
            split_manifest,
            normalization_artifact,
        )

    assert expected_steps > 0
    assert expected_val_check_interval > 0
    assert minimum_validation_step >= 0
    assert int(config.trainer.max_steps) == expected_steps
    assert int(config.trainer.limit_train_batches) == expected_steps
    assert int(config.trainer.val_check_interval) == expected_val_check_interval
    assert int(config.trainer.limit_val_batches) == 1
    assert int(config.trainer.num_sanity_val_steps) == 0
    assert int(config.trainer.log_every_n_steps) == 1
    assert str(config.trainer.precision) == "bf16"
    assert str(config.trainer.strategy) == expected_strategy
    assert int(config.launch_params.gpus_per_node) == expected_world_size
    assert int(config.launch_params.nodes) == 1
    assert int(config.trainer.devices) == expected_world_size
    assert int(config.trainer.num_nodes) == 1
    assert config.model.train_metrics_on_step is True
    ema_config = OmegaConf.select(config, "callbacks.ema", default=None)
    assert ema_config is not None
    assert ema_config._target_ == "egomimic.utils.ema_callback.EMACallback"
    assert math.isclose(float(ema_config.decay), 0.9978, abs_tol=1.0e-12)
    assert ema_config.validate_with_ema is True
    normal_policy_stages = [
        stage
        for stage in config.model.robomimic_model.stages
        if str(stage._target_).endswith("UniteLatentPolicy")
    ]
    shared_stages = [
        stage
        for stage in config.model.robomimic_model.stages
        if str(stage._target_).endswith("UniteSharedDenoiser")
    ]
    decoder_stages = [
        stage
        for stage in config.model.robomimic_model.stages
        if str(stage._target_).endswith("UnitePerEmbodimentActionDecoder")
    ]
    assert len(normal_policy_stages) == 1
    assert len(shared_stages) == len(decoder_stages) == 0
    unite_topology = "normal_unite_latent_policy"
    timestep_shift_alpha = float(normal_policy_stages[0].timestep_shift_alpha)
    reconstruction_noise_std = float(
        normal_policy_stages[0].reconstruction_noise_std
    )
    assert math.isclose(timestep_shift_alpha, 0.5, abs_tol=1.0e-12)
    assert math.isclose(reconstruction_noise_std, 0.7, abs_tol=1.0e-12)
    assert (
        config.evaluator._target_
        == "egomimic.eval.human_robot_overlay_eval.HumanRobotOverlayEval"
    )

    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file(), checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(checkpoint["global_step"])
    epoch = int(checkpoint["epoch"])
    assert global_step == expected_steps, global_step
    optimizer_states = checkpoint.get("optimizer_states", [])
    assert optimizer_states, "Smoke checkpoint has no optimizer state"
    optimizer_lrs = [
        float(group["lr"])
        for state in optimizer_states
        for group in state.get("param_groups", [])
    ]
    assert optimizer_lrs and all(math.isfinite(value) for value in optimizer_lrs)
    scheduler_states = checkpoint.get("lr_schedulers", [])
    assert len(scheduler_states) == 1, scheduler_states
    scheduler_last_epoch = int(scheduler_states[0]["last_epoch"])
    assert scheduler_last_epoch == global_step, scheduler_states[0]
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    assert hyper_parameters.get("train_metrics_on_step") is True
    ema_state_dict = checkpoint.get("ema_state_dict")
    assert ema_state_dict, "Smoke checkpoint has no EMA state"
    assert math.isclose(float(checkpoint["ema_decay"]), 0.9978, abs_tol=1.0e-12)
    ema_num_updates = int(checkpoint["ema_num_updates"])
    assert ema_num_updates == global_step
    assert checkpoint.get("ema_validate_with_ema") is True
    ema_tensor_count = len(ema_state_dict)
    ema_parameter_count = sum(value.numel() for value in ema_state_dict.values())
    assert ema_parameter_count > 0
    for name, value in ema_state_dict.items():
        assert torch.is_tensor(value), name
        assert bool(torch.isfinite(value).all()), name
    del checkpoint

    streams = [
        *output_dir.glob("wandb/offline-run-*/run-*.wandb"),
        *output_dir.glob("wandb/run-*/run-*.wandb"),
    ]
    assert len(streams) == 1, streams
    training_history, validation_history, wandb_exit_code = read_wandb_history(
        streams[0]
    )

    required_step_metrics = {
        "train_metrics": {"Train/Loss"},
        "timing_metrics": {
            "Timing/Process_Batch_Sec",
            "Timing/Forward_Pass_Sec",
            "Timing/Compute_Losses_Sec",
        },
        "optimizer_metrics": {"Optimizer/param_group_0_lr"},
    }
    dense_training_history = [
        row
        for row in training_history
        if all(
            required.issubset(row[category])
            for category, required in required_step_metrics.items()
        )
    ]
    assert len(dense_training_history) == expected_steps, training_history
    training_steps = [row["trainer_global_step"] for row in dense_training_history]
    assert training_steps == list(range(expected_steps)), training_steps
    for row in dense_training_history:
        values = [
            value
            for category in required_step_metrics
            for value in row[category].values()
        ]
        assert values and all(math.isfinite(value) for value in values), row

    flow_updates_per_reconstruction = int(
        config.model.unite_flow_updates_per_reconstruction
    )
    telemetry_cadence = int(
        config.model.unite_gradient_telemetry_every_n_steps
    )
    assert flow_updates_per_reconstruction > 0
    assert telemetry_cadence > 0
    assert expected_steps >= flow_updates_per_reconstruction + 1
    required_schedule = {
        "log/unite_update_is_flow",
        "log/unite_update_is_reconstruction",
        "log/unite_update_cycle_position",
    }
    schedule_history = [
        {
            "trainer_global_step": row["trainer_global_step"],
            "telemetry_metrics": row["telemetry_metrics"],
        }
        for row in training_history
        if required_schedule.issubset(row["telemetry_metrics"])
    ]
    assert len(schedule_history) == expected_steps, schedule_history
    for expected_step, row in enumerate(schedule_history):
        assert row["trainer_global_step"] == expected_step, row
        metrics = row["telemetry_metrics"]
        expected_position = expected_step % (flow_updates_per_reconstruction + 1)
        expected_flow = float(expected_position < flow_updates_per_reconstruction)
        expected_reconstruction = 1.0 - expected_flow
        assert metrics["log/unite_update_cycle_position"] == float(
            expected_position
        ), row
        assert metrics["log/unite_update_is_flow"] == expected_flow, row
        assert (
            metrics["log/unite_update_is_reconstruction"]
            == expected_reconstruction
        ), row

    required_telemetry = {
        "log/unite_gradient_cosine",
        "log/unite_recon_grad_norm",
        "log/unite_denoise_grad_norm",
    }
    telemetry_history = [
        {
            "trainer_global_step": row["trainer_global_step"],
            "telemetry_metrics": row["telemetry_metrics"],
        }
        for row in training_history
        if required_telemetry.issubset(row["telemetry_metrics"])
    ]
    assert telemetry_history, training_history
    for row in telemetry_history:
        metrics = row["telemetry_metrics"]
        assert all(math.isfinite(metrics[key]) for key in required_telemetry), row
        assert metrics["log/unite_recon_grad_norm"] > 0.0, row
        assert metrics["log/unite_denoise_grad_norm"] > 0.0, row
        assert -1.000001 <= metrics["log/unite_gradient_cosine"] <= 1.000001, row

    ema_history = [
        {
            "trainer_global_step": row["trainer_global_step"],
            "telemetry_metrics": row["telemetry_metrics"],
        }
        for row in training_history
        if {
            "log/unite_ema_decay",
            "log/unite_ema_num_updates",
        }.issubset(row["telemetry_metrics"])
    ]
    assert ema_history, training_history
    for row in ema_history:
        metrics = row["telemetry_metrics"]
        assert math.isclose(
            metrics["log/unite_ema_decay"], 0.9978, abs_tol=1.0e-8
        ), row
        assert math.isfinite(metrics["log/unite_ema_num_updates"]), row
        assert metrics["log/unite_ema_num_updates"] >= 1.0, row

    # Lightning associates validation metrics with the zero-based training step
    # that triggered validation. The corresponding completed optimizer-step count
    # (and Energy Score artifact suffix) is therefore trainer_global_step + 1.
    scheduled_history = [
        row
        for row in validation_history
        if row["trainer_global_step"] + 1 >= minimum_validation_step
    ]
    assert scheduled_history, validation_history
    qualifying_history = [
        row
        for row in scheduled_history
        if _has_required_metrics(row["validation_metrics"], required_embodiments)
    ]
    assert qualifying_history, (required_embodiments, scheduled_history)
    selected = qualifying_history[-1]
    metrics = selected["validation_metrics"]
    validation_after_optimizer_steps = selected["trainer_global_step"] + 1
    assert validation_after_optimizer_steps <= global_step

    for embodiment in required_embodiments:
        prefix = f"Valid/emb{embodiment}_"
        matches = {
            key: value for key, value in metrics.items() if key.startswith(prefix)
        }
        assert matches, (embodiment, sorted(metrics))
        assert all(math.isfinite(value) for value in matches.values()), matches

    assert all(math.isfinite(value) for value in metrics.values()), metrics
    energy_score_artifacts = _validate_energy_score_artifacts(
        output_dir,
        config,
        required_embodiments,
        expected_world_size,
        validation_after_optimizer_steps,
    )

    return {
        "status": "passed",
        "repo_head": expected_head,
        "output": str(output_dir),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "global_step": global_step,
        "epoch": epoch,
        "precision": str(config.trainer.precision),
        "world_size": expected_world_size,
        "expected_steps": expected_steps,
        "expected_val_check_interval": expected_val_check_interval,
        "minimum_validation_step": minimum_validation_step,
        "optimizer_state_count": len(optimizer_states),
        "optimizer_lrs": optimizer_lrs,
        "scheduler_last_epoch": scheduler_last_epoch,
        "required_embodiments": required_embodiments,
        "trainer_strategy": str(config.trainer.strategy),
        "wandb_stream": str(streams[0]),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_exit_code": wandb_exit_code,
        "validation_trainer_global_step": selected["trainer_global_step"],
        "validation_after_optimizer_steps": validation_after_optimizer_steps,
        "validation_metrics": metrics,
        "energy_score_artifacts": energy_score_artifacts,
        "training_history": training_history,
        "unite_update_schedule": {
            "flow_updates_per_reconstruction": flow_updates_per_reconstruction,
            "cycle_length_optimizer_steps": flow_updates_per_reconstruction + 1,
            "telemetry_cadence_optimizer_steps": telemetry_cadence,
            "history": schedule_history,
        },
        "unite_gradient_telemetry": telemetry_history,
        "ema": {
            "decay": 0.9978,
            "num_updates": ema_num_updates,
            "validation_uses_ema": True,
            "tensor_count": ema_tensor_count,
            "parameter_count": ema_parameter_count,
            "history": ema_history,
        },
        "unite_topology": unite_topology,
        "timestep_shift_alpha": timestep_shift_alpha,
        "reconstruction_noise_std": reconstruction_noise_std,
        "dense_training_steps": training_steps,
        "validation_history": validation_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--required-embodiments", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-strategy", default="ddp")
    parser.add_argument("--expected-world-size", type=int, default=1)
    parser.add_argument("--expected-steps", type=int, default=2)
    parser.add_argument("--expected-val-check-interval", type=int, default=1)
    parser.add_argument("--minimum-validation-step", type=int, default=1)
    parser.add_argument(
        "--released-sweep-topology", choices=("shared", "separate")
    )
    parser.add_argument("--expected-sweep-task-id")
    parser.add_argument("--expected-latent-dim", type=int)
    parser.add_argument("--expected-num-latent-tokens", type=int)
    parser.add_argument("--expected-wandb-run-path")
    parser.add_argument("--parameter-manifest", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--normalization-artifact", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    required_embodiments = [
        int(piece) for piece in args.required_embodiments.split(",") if piece.strip()
    ]
    assert required_embodiments
    record = verify_training_smoke(
        args.output_dir,
        required_embodiments,
        args.expected_head,
        args.expected_strategy,
        args.expected_world_size,
        args.expected_steps,
        args.expected_val_check_interval,
        args.minimum_validation_step,
        args.released_sweep_topology,
        args.expected_sweep_task_id,
        args.expected_latent_dim,
        args.expected_num_latent_tokens,
        args.expected_wandb_run_path,
        args.parameter_manifest,
        args.split_manifest,
        args.normalization_artifact,
    )
    if not args.dry_run:
        result_path = args.output_dir.resolve() / "SMOKE_RESULT.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(result_path)
    label = "VERIFY_PASS" if args.dry_run else "PASS"
    print(f"[smoke] {label} {json.dumps(record, sort_keys=True)}")


if __name__ == "__main__":
    main()
