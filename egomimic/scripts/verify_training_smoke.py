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


def verify_training_smoke(
    output_dir: Path,
    required_embodiments: list[int],
    expected_head: str,
    expected_strategy: str = "ddp",
    expected_world_size: int = 1,
    expected_steps: int = 2,
    expected_val_check_interval: int = 1,
    minimum_validation_step: int = 1,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_path = output_dir / ".hydra" / "config.yaml"
    assert config_path.is_file(), config_path
    config = _load_training_config(config_path)

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

    # num_sanity_val_steps=0 plus the requested minimum persisted trainer step
    # proves this metric came from scheduled validation after optimization began.
    scheduled_history = [
        row
        for row in validation_history
        if row["trainer_global_step"] >= minimum_validation_step
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
        selected["trainer_global_step"],
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
