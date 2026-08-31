"""Fail-closed verification for a completed immutable training run.

The verifier intentionally receives every external artifact as an exact
path/SHA pair.  It does not discover a nearby config, normalization cache,
bundle, or smoke result and therefore cannot silently validate the wrong run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

import egomimic.utils.hydra_resolvers  # noqa: F401 -- config resolvers
from egomimic.pl_utils.pl_model import ModelWrapper

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, *, label: str) -> str:
    assert _SHA256_RE.fullmatch(expected), (label, expected)
    assert path.is_file() and path.stat().st_size > 0, (label, path)
    actual = _sha256(path)
    assert actual == expected, (label, path, actual, expected)
    return actual


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    assert isinstance(payload, dict), (label, type(payload))
    return payload


def _git_identity(work_dir: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "-C", str(work_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(work_dir),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, status


def _mapping_value(mapping: Mapping[Any, Any], key: int | str) -> Any:
    if key in mapping:
        return mapping[key]
    alternate = str(key) if isinstance(key, int) else int(key)
    assert alternate in mapping, (key, sorted(str(item) for item in mapping))
    return mapping[alternate]


def _validate_bundle(
    bundle_manifest: Path,
    expected_bundle_sha256: str,
    expected_head: str,
) -> dict[str, Any]:
    sha = _require_sha256(
        bundle_manifest,
        expected_bundle_sha256,
        label="bundle_manifest",
    )
    payload = _load_json_object(bundle_manifest, label="bundle_manifest")
    source = payload.get("source")
    assert isinstance(source, dict), "bundle manifest has no source object"
    assert source.get("head") == expected_head, (source.get("head"), expected_head)
    assert source.get("clean") is True, source
    wandb = payload.get("wandb")
    outputs = payload.get("outputs")
    configs = payload.get("configs")
    assert isinstance(wandb, dict), "bundle manifest has no wandb object"
    assert isinstance(outputs, dict), "bundle manifest has no outputs object"
    assert isinstance(configs, dict), "bundle manifest has no configs object"
    return {
        "path": str(bundle_manifest),
        "sha256": sha,
        "source_head": source["head"],
        "source_clean": True,
        "wandb": wandb,
        "outputs": outputs,
        "configs": configs,
    }


def _validate_runtime_config_against_bundle(
    config: Any,
    bundle_record: Mapping[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    pinned_record = bundle_record["configs"]["full"]
    pinned_path = Path(pinned_record["path"])
    _require_sha256(
        pinned_path,
        str(pinned_record["sha256"]),
        label="bundle_resolved_full_config",
    )
    actual = OmegaConf.to_container(config, resolve=True)
    pinned = OmegaConf.to_container(OmegaConf.load(pinned_path), resolve=True)
    assert isinstance(actual, dict) and isinstance(pinned, dict)
    resume_path = actual.get("ckpt_path")
    if resume_path is not None:
        resolved_resume = Path(str(resume_path)).resolve()
        assert resolved_resume.is_file() and run_dir in resolved_resume.parents
        actual["ckpt_path"] = pinned.get("ckpt_path")
        assert actual["logger"]["wandb"]["resume"] == "allow"
        actual["logger"]["wandb"]["resume"] = pinned["logger"]["wandb"]["resume"]
    assert actual == pinned, "runtime config differs from the immutable full config"
    return {
        "pinned_path": str(pinned_path),
        "pinned_sha256": pinned_record["sha256"],
        "resume_checkpoint": None if resume_path is None else str(resolved_resume),
    }


def _validate_smoke_result(
    smoke_result: Path,
    expected_smoke_result_sha256: str,
    expected_head: str,
    expected_world_size: int,
    required_wandb_metrics: list[str] | None,
) -> dict[str, Any]:
    sha = _require_sha256(
        smoke_result,
        expected_smoke_result_sha256,
        label="smoke_result",
    )
    payload = _load_json_object(smoke_result, label="smoke_result")
    assert payload.get("status") == "passed", payload.get("status")
    assert payload.get("repo_head") == expected_head, (
        payload.get("repo_head"),
        expected_head,
    )
    assert int(payload.get("world_size")) == expected_world_size, payload.get(
        "world_size"
    )
    assert int(payload.get("global_step")) == 2, payload.get("global_step")
    assert int(payload.get("wandb_exit_code")) == 0, payload.get("wandb_exit_code")
    strict_load = payload.get("model_wrapper_load")
    assert isinstance(strict_load, dict), "smoke did not record a strict model load"
    assert strict_load.get("status") == "passed", strict_load
    assert strict_load.get("strict") is True, strict_load
    assert strict_load.get("map_location") == "cpu", strict_load
    if required_wandb_metrics is not None:
        assert payload.get("required_wandb_metrics") == required_wandb_metrics, (
            payload.get("required_wandb_metrics"),
            required_wandb_metrics,
        )
    record = {
        "path": str(smoke_result),
        "sha256": sha,
        "global_step": 2,
        "wandb_exit_code": 0,
        "strict_model_load": True,
    }
    if strict_load.get("ema") is not None:
        record["ema"] = strict_load["ema"]
    return record


def _validate_training_config(
    config: Any,
    *,
    run_dir: Path,
    norm_artifact: Path,
    expected_max_step: int,
    expected_world_size: int,
) -> list[str] | None:
    assert str(config.mode) == "train", config.mode
    assert Path(str(config.paths.output_dir)).resolve() == run_dir, (
        config.paths.output_dir,
        run_dir,
    )
    assert int(config.trainer.max_steps) == expected_max_step

    launch_world_size = int(config.launch_params.gpus_per_node) * int(
        config.launch_params.nodes
    )
    trainer_world_size = int(config.trainer.devices) * int(config.trainer.num_nodes)
    assert launch_world_size == expected_world_size, launch_world_size
    assert trainer_world_size == expected_world_size, trainer_world_size

    assert config.norm_stats.save_cache_dir is None
    assert Path(str(config.norm_stats.precomputed_norm_path)).resolve() == (
        norm_artifact
    )

    provenance = config.get("run_provenance")
    if provenance is not None and provenance.get("training_contract") is not None:
        contract = provenance.training_contract
        assert int(contract.world_size) == expected_world_size
        assert str(contract.precision) == str(config.trainer.precision)
        assert int(contract.max_steps) == expected_max_step
        assert int(contract.validation_interval_steps) == int(
            config.trainer.val_check_interval
        )
        assert math.isclose(
            float(contract.validation_fraction),
            float(config.trainer.limit_val_batches),
        )
        optimizer_target = str(config.model.optimizer._target_)
        assert optimizer_target.rsplit(".", 1)[-1] == str(contract.optimizer)
        peak_lr = float(config.model.optimizer.lr)
        assert math.isclose(float(contract.peak_lr), peak_lr)
        assert int(contract.warmup_steps) == int(config.model.scheduler.warmup_steps)
        warmup_start_lr = peak_lr * float(config.model.scheduler.warmup_start_factor)
        assert math.isclose(float(contract.warmup_start_lr), warmup_start_lr)
        assert math.isclose(
            float(contract.cosine_floor_lr),
            float(config.model.scheduler.eta_min),
        )

    if provenance is not None and provenance.get("norm_contract") is not None:
        contract = provenance.norm_contract
        assert str(contract.norm_mode) == str(config.norm_stats.norm_mode)
        assert bool(contract.reduce_all_but_last) == bool(
            config.norm_stats.reduce_all_but_last
        )
        assert math.isclose(
            float(contract.sample_frac), float(config.norm_stats.sample_frac)
        )

    configured_metrics = OmegaConf.select(
        config,
        "run_provenance.required_wandb_metrics",
        default=None,
    )
    if configured_metrics is None:
        return None
    required = [str(metric) for metric in configured_metrics]
    assert required and len(required) == len(set(required)), required
    assert all(
        metric.startswith(("Train/", "Timing/", "Optimizer/", "Valid/"))
        for metric in required
    ), required
    return required


def _validate_checkpoint_norm_state(
    checkpoint: Mapping[str, Any],
    norm_payload: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    hyper_parameters = checkpoint.get("hyper_parameters")
    assert isinstance(hyper_parameters, Mapping), "checkpoint has no hyper_parameters"
    norm_state = hyper_parameters.get("norm_stats_state")
    assert isinstance(norm_state, Mapping), "checkpoint has no norm_stats_state"
    assert norm_state.get("norm_mode") == norm_payload.get("norm_mode")
    assert bool(norm_state.get("reduce_all_but_last")) == bool(
        norm_payload.get("reduce_all_but_last")
    )

    artifact_stats = norm_payload.get("stats")
    checkpoint_stats = norm_state.get("norm_stats")
    assert isinstance(artifact_stats, Mapping) and artifact_stats
    assert isinstance(checkpoint_stats, Mapping) and checkpoint_stats
    assert {str(key) for key in checkpoint_stats} == set(artifact_stats)

    stat_array_count = 0
    for embodiment, per_key in artifact_stats.items():
        checkpoint_per_key = _mapping_value(checkpoint_stats, embodiment)
        assert set(checkpoint_per_key) == set(per_key), embodiment
        for key, stats in per_key.items():
            checkpoint_key_stats = checkpoint_per_key[key]
            assert set(checkpoint_key_stats) == set(stats), (embodiment, key)
            for stat_name, values in stats.items():
                artifact_array = np.asarray(values, dtype=np.float32)
                checkpoint_array = np.asarray(
                    checkpoint_key_stats[stat_name], dtype=np.float32
                )
                assert artifact_array.shape == checkpoint_array.shape, (
                    embodiment,
                    key,
                    stat_name,
                    artifact_array.shape,
                    checkpoint_array.shape,
                )
                assert np.isfinite(artifact_array).all(), (embodiment, key, stat_name)
                assert np.isfinite(checkpoint_array).all(), (
                    embodiment,
                    key,
                    stat_name,
                )
                assert np.array_equal(artifact_array, checkpoint_array), (
                    embodiment,
                    key,
                    stat_name,
                )
                stat_array_count += 1

    provenance = config.get("run_provenance")
    if provenance is not None and provenance.get("norm_contract") is not None:
        shapes = norm_state.get("shapes")
        assert isinstance(shapes, Mapping), "checkpoint norm state has no shapes"
        for domain, contract in provenance.norm_contract.domains.items():
            embodiment = int(contract.embodiment_id)
            per_key_shapes = _mapping_value(shapes, embodiment)
            assert tuple(per_key_shapes["state_agent_obj"]) == tuple(
                contract.state_agent_obj_shape
            ), domain
            assert tuple(per_key_shapes["actions"]) == tuple(contract.actions_shape), (
                domain
            )

    config_tree = hyper_parameters.get("config_tree")
    assert isinstance(config_tree, Mapping), "checkpoint has no config_tree"
    checkpoint_model = config_tree.get("model")
    assert checkpoint_model is not None, "checkpoint config_tree has no model"
    assert OmegaConf.to_container(
        OmegaConf.create(checkpoint_model), resolve=True
    ) == OmegaConf.to_container(config.model, resolve=True)
    if provenance is not None:
        checkpoint_provenance = config_tree.get("run_provenance")
        assert checkpoint_provenance is not None, (
            "checkpoint config_tree has no run_provenance"
        )
        assert OmegaConf.to_container(
            OmegaConf.create(checkpoint_provenance), resolve=True
        ) == OmegaConf.to_container(provenance, resolve=True)

    return {
        "norm_mode": norm_state["norm_mode"],
        "reduce_all_but_last": bool(norm_state["reduce_all_but_last"]),
        "embodiments": sorted(str(key) for key in artifact_stats),
        "stat_array_count": stat_array_count,
    }


def _strict_load_model_wrapper(checkpoint_path: Path) -> dict[str, Any]:
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
    assert state_dict, "strictly loaded wrapper has an empty state_dict"
    non_cpu = {
        key: str(value.device)
        for key, value in state_dict.items()
        if value.device.type != "cpu"
    }
    assert not non_cpu, non_cpu
    ema_record = None
    if getattr(wrapper, "_ema_config", None) is not None:
        assert hasattr(wrapper, "ema_model")
        optimization_step = int(wrapper.ema_optimization_step.item())
        assert optimization_step > 0, optimization_step
        expected_decay = wrapper._ema_decay_for_step(optimization_step)
        actual_decay = float(wrapper.ema_decay.item())
        assert math.isclose(actual_decay, expected_decay, abs_tol=1.0e-12), (
            actual_decay,
            expected_decay,
        )
        online = dict(wrapper.model.nets.named_parameters())
        averaged = dict(wrapper.ema_model.nets.named_parameters())
        assert online.keys() == averaged.keys()
        assert all(not parameter.requires_grad for parameter in averaged.values())
        assert all(torch.isfinite(parameter).all() for parameter in averaged.values())
        ema_record = {
            "enabled": True,
            "optimization_step": optimization_step,
            "decay": actual_decay,
            "power": float(wrapper._ema_config["power"]),
            "max_value": float(wrapper._ema_config["max_value"]),
            "use_for_validation": bool(
                wrapper._ema_config["use_for_validation"]
            ),
            "parameter_tree_exact": True,
            "parameters_finite": True,
        }
    record = {
        "status": "passed",
        "model_class": f"{type(wrapper).__module__}.{type(wrapper).__qualname__}",
        "map_location": "cpu",
        "strict": True,
        "state_dict_key_count": len(state_dict),
        "parameter_count": sum(parameter.numel() for parameter in wrapper.parameters()),
        "buffer_count": sum(buffer.numel() for buffer in wrapper.buffers()),
    }
    if ema_record is not None:
        record["ema"] = ema_record
    del wrapper
    gc.collect()
    return record


def _item_key(item: Any) -> str | None:
    if item.key:
        return item.key
    if item.nested_key:
        return ".".join(item.nested_key)
    return None


def _scan_wandb_stream(
    stream_path: Path,
    required_metrics: set[str],
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    store = DataStore()
    store.open_for_scan(str(stream_path))
    evidence: dict[str, dict[str, Any]] = {}
    exit_codes: list[int] = []
    try:
        while True:
            raw = store.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            record_type = record.WhichOneof("record_type")
            if record_type == "exit":
                exit_codes.append(int(record.exit.exit_code))
                continue
            if record_type != "history" or not required_metrics:
                continue

            row: dict[str, Any] = {}
            for item in record.history.item:
                key = _item_key(item)
                if key is None:
                    continue
                try:
                    row[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    continue
            step = row.get("trainer/global_step")
            step = int(step) if step is not None else None
            for metric in required_metrics.intersection(row):
                value = float(row[metric])
                assert math.isfinite(value), (stream_path, step, metric, value)
                item = evidence.setdefault(
                    metric,
                    {"count": 0, "first_step": step, "last_step": step, "last": value},
                )
                item["count"] += 1
                if item["first_step"] is None and step is not None:
                    item["first_step"] = step
                if step is not None:
                    item["last_step"] = step
                item["last"] = value
    finally:
        store.close()
    return evidence, exit_codes


def _validate_wandb(
    run_dir: Path,
    required_wandb_metrics: list[str] | None,
) -> dict[str, Any]:
    streams = sorted(
        set(run_dir.glob("wandb/run-*/run-*.wandb"))
        | set(run_dir.glob("wandb/offline-run-*/run-*.wandb")),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    assert streams, f"No native W&B stream under {run_dir / 'wandb'}"

    required = set(required_wandb_metrics or [])
    aggregate: dict[str, dict[str, Any]] = {}
    exits_by_stream: dict[Path, list[int]] = {}
    for stream in streams:
        evidence, exit_codes = _scan_wandb_stream(stream, required)
        exits_by_stream[stream] = exit_codes
        for metric, item in evidence.items():
            if metric not in aggregate:
                aggregate[metric] = dict(item)
                continue
            aggregate_item = aggregate[metric]
            aggregate_item["count"] += item["count"]
            if aggregate_item["first_step"] is None:
                aggregate_item["first_step"] = item["first_step"]
            aggregate_item["last_step"] = item["last_step"]
            aggregate_item["last"] = item["last"]

    missing = required - set(aggregate)
    assert not missing, ("missing required W&B metrics", sorted(missing))

    latest = streams[-1]
    latest_exit_codes = exits_by_stream[latest]
    assert latest_exit_codes, f"No terminal W&B exit record in {latest}"
    assert latest_exit_codes[-1] == 0, (latest, latest_exit_codes)
    return {
        "streams": [{"path": str(path), "sha256": _sha256(path)} for path in streams],
        "successful_stream": str(latest),
        "successful_stream_sha256": _sha256(latest),
        "exit_code": 0,
        "required_metrics": required_wandb_metrics,
        "metric_evidence": {key: aggregate[key] for key in sorted(aggregate)},
    }


def verify_training_full(
    run_dir: Path,
    *,
    expected_max_step: int,
    expected_head: str,
    expected_world_size: int,
    resolved_config: Path,
    expected_config_sha256: str,
    norm_artifact: Path,
    expected_norm_sha256: str,
    bundle_manifest: Path,
    expected_bundle_sha256: str,
    smoke_result: Path,
    expected_smoke_result_sha256: str,
) -> dict[str, Any]:
    assert expected_max_step > 0, expected_max_step
    assert expected_world_size > 0, expected_world_size
    assert _GIT_SHA_RE.fullmatch(expected_head), expected_head

    run_dir = run_dir.resolve()
    assert run_dir.is_dir(), run_dir
    resolved_config = resolved_config.resolve()
    norm_artifact = norm_artifact.resolve()
    bundle_manifest = bundle_manifest.resolve()
    smoke_result = smoke_result.resolve()
    expected_run_config = run_dir / ".hydra" / "config.yaml"
    assert resolved_config == expected_run_config.resolve(), (
        resolved_config,
        expected_run_config,
    )

    config_sha = _require_sha256(
        resolved_config,
        expected_config_sha256,
        label="resolved_config",
    )
    norm_sha = _require_sha256(
        norm_artifact,
        expected_norm_sha256,
        label="norm_artifact",
    )
    bundle_record = _validate_bundle(
        bundle_manifest,
        expected_bundle_sha256,
        expected_head,
    )

    config = OmegaConf.load(resolved_config)
    config_bundle_record = _validate_runtime_config_against_bundle(
        config,
        bundle_record,
        run_dir,
    )
    required_wandb_metrics = _validate_training_config(
        config,
        run_dir=run_dir,
        norm_artifact=norm_artifact,
        expected_max_step=expected_max_step,
        expected_world_size=expected_world_size,
    )
    assert Path(bundle_record["outputs"]["full_run"]).resolve() == run_dir
    assert str(config.logger.wandb.entity) == bundle_record["wandb"]["entity"]
    assert str(config.logger.wandb.project) == bundle_record["wandb"]["project"]
    assert str(config.logger.wandb.group) == bundle_record["wandb"]["group"]
    assert str(config.logger.wandb.id) == bundle_record["wandb"]["full_id"]
    assert str(config.logger.wandb.name) == bundle_record["wandb"]["full_id"]
    smoke_record = _validate_smoke_result(
        smoke_result,
        expected_smoke_result_sha256,
        expected_head,
        expected_world_size,
        required_wandb_metrics,
    )

    work_dir = Path(str(config.paths.work_dir)).resolve()
    actual_head, git_status = _git_identity(work_dir)
    assert actual_head == expected_head, (actual_head, expected_head)
    assert not git_status, (work_dir, git_status)

    terminal_dir = run_dir / "checkpoints" / "final"
    checkpoints = sorted(terminal_dir.glob("*.ckpt"))
    terminal_template = str(
        OmegaConf.select(
            config,
            "callbacks.terminal_checkpoint.filename",
            default="step-{step}",
        )
    )
    if "{epoch}" in terminal_template:
        candidates = sorted(
            terminal_dir.glob(f"epoch-*-step-{expected_max_step}.ckpt")
        )
        assert len(candidates) == 1, candidates
        expected_checkpoint = candidates[0]
    else:
        expected_checkpoint = terminal_dir / f"step-{expected_max_step}.ckpt"
    assert checkpoints == [expected_checkpoint], (checkpoints, expected_checkpoint)
    assert expected_checkpoint.is_file() and expected_checkpoint.stat().st_size > 0

    checkpoint = torch.load(
        expected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert isinstance(checkpoint, Mapping), type(checkpoint)
    global_step = int(checkpoint.get("global_step"))
    epoch = int(checkpoint.get("epoch"))
    assert global_step == expected_max_step, global_step
    if "{epoch}" in terminal_template:
        assert expected_checkpoint.name == (
            f"epoch-{epoch}-step-{global_step}.ckpt"
        ), (expected_checkpoint.name, epoch, global_step)
    state_dict = checkpoint.get("state_dict")
    assert isinstance(state_dict, Mapping) and state_dict, (
        "checkpoint state_dict is empty"
    )

    optimizer_states = checkpoint.get("optimizer_states")
    assert isinstance(optimizer_states, list) and optimizer_states
    optimizer_state_count = len(optimizer_states)
    optimizer_lrs: list[float] = []
    for optimizer_state in optimizer_states:
        assert isinstance(optimizer_state, Mapping)
        param_groups = optimizer_state.get("param_groups")
        assert isinstance(param_groups, list) and param_groups
        for group in param_groups:
            value = float(group["lr"])
            assert math.isfinite(value), value
            optimizer_lrs.append(value)
    assert optimizer_lrs
    expected_final_lr = float(config.model.scheduler.eta_min)
    assert all(
        math.isclose(value, expected_final_lr, rel_tol=1.0e-4, abs_tol=0.0)
        for value in optimizer_lrs
    ), (optimizer_lrs, expected_final_lr)

    scheduler_states = checkpoint.get("lr_schedulers")
    assert isinstance(scheduler_states, list) and scheduler_states
    scheduler_state_count = len(scheduler_states)
    scheduler_last_epochs: list[int] = []
    for scheduler_state in scheduler_states:
        assert isinstance(scheduler_state, Mapping)
        last_epoch = int(scheduler_state["last_epoch"])
        assert last_epoch == expected_max_step, scheduler_state
        scheduler_last_epochs.append(last_epoch)

    norm_payload = _load_json_object(norm_artifact, label="norm_artifact")
    checkpoint_norm_record = _validate_checkpoint_norm_state(
        checkpoint,
        norm_payload,
        config,
    )
    del checkpoint, state_dict, optimizer_states, scheduler_states
    gc.collect()

    model_wrapper_load = _strict_load_model_wrapper(expected_checkpoint)
    ema_config = OmegaConf.select(config, "model.ema", default=None)
    if ema_config is not None and bool(ema_config.enabled):
        for label, ema_load in (
            ("smoke", smoke_record.get("ema")),
            ("full", model_wrapper_load.get("ema")),
        ):
            assert isinstance(ema_load, dict) and ema_load.get("enabled") is True, (
                label,
                ema_load,
            )
            assert math.isclose(float(ema_load["power"]), float(ema_config.power))
            assert math.isclose(
                float(ema_load["max_value"]), float(ema_config.max_value)
            )
            assert ema_load["use_for_validation"] is bool(
                ema_config.use_for_validation
            )
        assert int(model_wrapper_load["ema"]["optimization_step"]) == global_step
    wandb_record = _validate_wandb(run_dir, required_wandb_metrics)

    return {
        "status": "passed",
        "schema_version": 1,
        "repo_head": expected_head,
        "repo_clean": True,
        "run_dir": str(run_dir),
        "global_step": global_step,
        "epoch": epoch,
        "world_size": expected_world_size,
        "precision": str(config.trainer.precision),
        "resolved_config": str(resolved_config),
        "resolved_config_sha256": config_sha,
        "immutable_config": config_bundle_record,
        "norm_artifact": str(norm_artifact),
        "norm_artifact_sha256": norm_sha,
        "bundle": bundle_record,
        "smoke_result": smoke_record,
        "terminal_checkpoint": str(expected_checkpoint),
        "terminal_checkpoint_sha256": _sha256(expected_checkpoint),
        "optimizer_state_count": optimizer_state_count,
        "optimizer_lrs": optimizer_lrs,
        "expected_final_lr": expected_final_lr,
        "scheduler_state_count": scheduler_state_count,
        "scheduler_last_epochs": scheduler_last_epochs,
        "checkpoint_norm_state": checkpoint_norm_record,
        "model_wrapper_load": model_wrapper_load,
        "required_wandb_metrics": required_wandb_metrics,
        "wandb": wandb_record,
    }


def _write_atomic_result(run_dir: Path, record: Mapping[str, Any]) -> Path:
    result_path = run_dir.resolve() / "FULL_RESULT.json"
    assert not result_path.exists(), f"Refusing to overwrite {result_path}"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".FULL_RESULT.", suffix=".tmp", dir=run_dir
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        assert not result_path.exists(), f"Refusing to overwrite {result_path}"
        os.replace(temporary_path, result_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-max-step", type=int, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--norm-artifact", type=Path, required=True)
    parser.add_argument("--expected-norm-sha256", required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--expected-smoke-result-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record = verify_training_full(
        args.run_dir,
        expected_max_step=args.expected_max_step,
        expected_head=args.expected_head,
        expected_world_size=args.expected_world_size,
        resolved_config=args.resolved_config,
        expected_config_sha256=args.expected_config_sha256,
        norm_artifact=args.norm_artifact,
        expected_norm_sha256=args.expected_norm_sha256,
        bundle_manifest=args.bundle_manifest,
        expected_bundle_sha256=args.expected_bundle_sha256,
        smoke_result=args.smoke_result,
        expected_smoke_result_sha256=args.expected_smoke_result_sha256,
    )
    if args.dry_run:
        print(f"[full] VERIFY_PASS {json.dumps(record, sort_keys=True)}")
        return
    result_path = _write_atomic_result(args.run_dir, record)
    print(f"[full] PASS result={result_path} {json.dumps(record, sort_keys=True)}")


if __name__ == "__main__":
    main()
