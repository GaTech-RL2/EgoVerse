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

import egomimic.utils.hydra_resolvers  # noqa: F401 -- project config resolvers
from egomimic.pl_utils.pl_model import ModelWrapper

_HISTORY_METRIC_PREFIXES = ("Train/", "Timing/", "Optimizer/", "Valid/")
_DEFAULT_REQUIRED_STEP_METRICS = {
    "train_metrics": {"Train/Loss"},
    "timing_metrics": {
        "Timing/Process_Batch_Sec",
        "Timing/Forward_Pass_Sec",
        "Timing/Compute_Losses_Sec",
    },
    "optimizer_metrics": {"Optimizer/param_group_0_lr"},
}
_STEP_METRIC_CATEGORIES = {
    "Train/": "train_metrics",
    "Timing/": "timing_metrics",
    "Optimizer/": "optimizer_metrics",
}


def _register_training_config_resolvers() -> None:
    """Mirror the resolvers registered by ``egomimic.trainHydra``."""

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


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


def read_successful_wandb_exit_code(stream_path: Path) -> int:
    """Return a W&B stream's terminal success code without loading history."""

    store = DataStore()
    store.open_for_scan(str(stream_path))
    exit_codes: list[int] = []
    try:
        while True:
            payload = store.scan_data()
            if payload is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            if record.WhichOneof("record_type") == "exit":
                exit_codes.append(int(record.exit.exit_code))
    finally:
        store.close()

    assert exit_codes, f"No terminal W&B exit record in {stream_path}"
    assert exit_codes[-1] == 0, (stream_path, exit_codes)
    return exit_codes[-1]


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
                if not key.startswith(_HISTORY_METRIC_PREFIXES):
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
        if train_metrics or timing_metrics or optimizer_metrics:
            training_rows.append(
                {
                    "trainer_global_step": step,
                    "train_metrics": train_metrics,
                    "timing_metrics": timing_metrics,
                    "optimizer_metrics": optimizer_metrics,
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
    for embodiment in required_embodiments:
        prefix = f"Valid/emb{embodiment}_"
        if not any(
            key.startswith(prefix) and key.endswith("_action_mse") for key in metrics
        ):
            return False
    return True


def _required_metric_contract(
    config: Any,
) -> tuple[list[str] | None, dict[str, set[str]], set[str]]:
    """Resolve configured metric requirements and retain legacy smoke defaults."""

    configured = OmegaConf.select(
        config,
        "run_provenance.required_wandb_metrics",
        default=None,
    )
    required_step_metrics = {
        category: set(metrics)
        for category, metrics in _DEFAULT_REQUIRED_STEP_METRICS.items()
    }
    if configured is None:
        return None, required_step_metrics, set()

    configured_metrics = [str(metric) for metric in configured]
    assert configured_metrics, "required_wandb_metrics must not be empty"
    assert len(configured_metrics) == len(set(configured_metrics)), (
        "required_wandb_metrics contains duplicates",
        configured_metrics,
    )

    required_validation_metrics: set[str] = set()
    for metric in configured_metrics:
        assert metric and metric.startswith(_HISTORY_METRIC_PREFIXES), (
            "Unsupported required W&B metric namespace",
            metric,
        )
        if metric.startswith("Valid/"):
            required_validation_metrics.add(metric)
            continue
        category = next(
            category
            for prefix, category in _STEP_METRIC_CATEGORIES.items()
            if metric.startswith(prefix)
        )
        required_step_metrics[category].add(metric)

    return configured_metrics, required_step_metrics, required_validation_metrics


def _select_dense_training_history(
    training_history: list[dict[str, Any]],
    required_step_metrics: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[int]]:
    dense_training_history = [
        row
        for row in training_history
        if all(
            required.issubset(row[category])
            for category, required in required_step_metrics.items()
        )
    ]
    assert len(dense_training_history) == 2, training_history
    training_steps = [row["trainer_global_step"] for row in dense_training_history]
    assert training_steps == [0, 1], training_steps
    for row in dense_training_history:
        for category, required in required_step_metrics.items():
            required_values = {
                metric: row[category][metric] for metric in sorted(required)
            }
            assert all(math.isfinite(value) for value in required_values.values()), (
                category,
                required_values,
            )
    return dense_training_history, training_steps


def _select_scheduled_validation(
    validation_history: list[dict[str, Any]],
    required_embodiments: list[int],
    required_validation_metrics: set[str],
) -> dict[str, Any]:
    # num_sanity_val_steps=0 plus a persisted trainer step >= 1 proves these
    # metrics came from scheduled validation after optimization had begun.
    scheduled_history = [
        row for row in validation_history if row["trainer_global_step"] >= 1
    ]
    assert scheduled_history, validation_history
    if required_validation_metrics:
        qualifying_history = [
            row
            for row in scheduled_history
            if required_validation_metrics.issubset(row["validation_metrics"])
        ]
        assert qualifying_history, (
            sorted(required_validation_metrics),
            scheduled_history,
        )
        selected = qualifying_history[-1]
        required_values = {
            metric: selected["validation_metrics"][metric]
            for metric in sorted(required_validation_metrics)
        }
        assert all(math.isfinite(value) for value in required_values.values()), (
            required_values
        )
        return selected

    # Legacy configs did not declare required_wandb_metrics. Preserve their
    # per-embodiment ``Valid/emb*_..._action_mse`` contract.
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
            key: value
            for key, value in metrics.items()
            if key.startswith(prefix) and key.endswith("_action_mse")
        }
        assert matches, (embodiment, sorted(metrics))
        assert all(math.isfinite(value) for value in matches.values()), matches
    return selected


def _strict_load_model_wrapper(checkpoint_path: Path) -> dict[str, Any]:
    """Reconstruct the checkpointed wrapper on CPU with an exact state load."""

    assert not torch.cuda.is_available(), (
        "strict CPU verification must run with CUDA hidden"
    )
    wrapper = ModelWrapper.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
        strict=True,
    )
    assert isinstance(wrapper, ModelWrapper), type(wrapper)
    state_dict = wrapper.state_dict()
    non_cpu_tensors = {
        key: str(value.device)
        for key, value in state_dict.items()
        if value.device.type != "cpu"
    }
    assert not non_cpu_tensors, non_cpu_tensors
    record = {
        "status": "passed",
        "model_class": f"{type(wrapper).__module__}.{type(wrapper).__qualname__}",
        "map_location": "cpu",
        "strict": True,
        "state_dict_key_count": len(state_dict),
        "parameter_count": sum(parameter.numel() for parameter in wrapper.parameters()),
        "buffer_count": sum(buffer.numel() for buffer in wrapper.buffers()),
    }
    del wrapper
    return record


def verify_training_smoke(
    output_dir: Path,
    required_embodiments: list[int],
    expected_head: str,
    expected_world_size: int = 1,
) -> dict[str, Any]:
    _register_training_config_resolvers()
    output_dir = output_dir.resolve()
    config_path = output_dir / ".hydra" / "config.yaml"
    assert config_path.is_file(), config_path
    config = OmegaConf.load(config_path)

    assert int(config.trainer.max_steps) == 2
    assert int(config.trainer.limit_train_batches) == 2
    assert int(config.trainer.val_check_interval) == 1
    assert int(config.trainer.limit_val_batches) == 1
    assert int(config.trainer.num_sanity_val_steps) == 0
    assert int(config.trainer.log_every_n_steps) == 1
    assert str(config.trainer.precision) == "bf16"
    assert str(config.trainer.strategy) == "ddp"
    assert int(config.launch_params.gpus_per_node) == expected_world_size
    assert int(config.launch_params.nodes) == 1
    assert int(config.trainer.devices) == expected_world_size
    assert int(config.trainer.num_nodes) == 1
    assert config.model.train_metrics_on_step is True
    assert (
        config.evaluator._target_
        == "egomimic.eval.human_robot_overlay_eval.HumanRobotOverlayEval"
    )
    (
        required_wandb_metrics,
        required_step_metrics,
        required_validation_metrics,
    ) = _required_metric_contract(config)

    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    assert checkpoint_path.is_file(), checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    global_step = int(checkpoint["global_step"])
    epoch = int(checkpoint["epoch"])
    assert global_step == 2, global_step
    optimizer_states = checkpoint.get("optimizer_states", [])
    assert optimizer_states, "Smoke checkpoint has no optimizer state"
    optimizer_state_count = len(optimizer_states)
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
    del checkpoint, hyper_parameters, optimizer_states, scheduler_states
    model_wrapper_load = _strict_load_model_wrapper(checkpoint_path)

    streams = list(output_dir.glob("wandb/offline-run-*/run-*.wandb"))
    assert len(streams) == 1, streams
    training_history, validation_history, wandb_exit_code = read_wandb_history(
        streams[0]
    )

    dense_training_history, training_steps = _select_dense_training_history(
        training_history,
        required_step_metrics,
    )
    selected = _select_scheduled_validation(
        validation_history,
        required_embodiments,
        required_validation_metrics,
    )
    metrics = selected["validation_metrics"]

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
        "optimizer_state_count": optimizer_state_count,
        "optimizer_lrs": optimizer_lrs,
        "scheduler_last_epoch": scheduler_last_epoch,
        "model_wrapper_load": model_wrapper_load,
        "required_embodiments": required_embodiments,
        "required_wandb_metrics": required_wandb_metrics,
        "required_step_metrics": {
            category: sorted(required)
            for category, required in required_step_metrics.items()
        },
        "required_validation_metrics": sorted(required_validation_metrics),
        "wandb_stream": str(streams[0]),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_exit_code": wandb_exit_code,
        "validation_trainer_global_step": selected["trainer_global_step"],
        "validation_metrics": metrics,
        "training_history": training_history,
        "dense_training_steps": training_steps,
        "validation_history": validation_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--required-embodiments", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-world-size", type=int, default=1)
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
        args.expected_world_size,
    )
    if not args.dry_run:
        result_path = args.output_dir.resolve() / "SMOKE_RESULT.json"
        temporary_path = result_path.with_suffix(".json.tmp")
        assert not result_path.exists(), f"Refusing to overwrite {result_path}"
        assert not temporary_path.exists(), f"Stale temporary result: {temporary_path}"
        temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(result_path)
    label = "VERIFY_PASS" if args.dry_run else "PASS"
    print(f"[smoke] {label} {json.dumps(record, sort_keys=True)}")


if __name__ == "__main__":
    main()
