#!/usr/bin/env python3
"""Strictly validate a reusable full-state Lightning checkpoint.

The validator is intentionally independent of a particular Pipeline model.  It
checks the structural state needed to resume standard Pipeline/Planar BC
training, scans all tensors and numeric scalars for non-finite values, and can
bind a checkpoint to expected provenance identities when those identities are
recorded in the checkpoint.

On success, the last non-empty stdout line is a JSON object containing a
non-negative integer ``global_step``.  This is the contract consumed by the
generic ICE requeue runner and checkpoint mirror.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import numbers
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_MISSING = object()


@dataclass(frozen=True)
class ScanStats:
    tensor_count: int = 0
    tensor_numel: int = 0
    numeric_scalar_count: int = 0

    def __add__(self, other: "ScanStats") -> "ScanStats":
        return ScanStats(
            tensor_count=self.tensor_count + other.tensor_count,
            tensor_numel=self.tensor_numel + other.tensor_numel,
            numeric_scalar_count=(
                self.numeric_scalar_count + other.numeric_scalar_count
            ),
        )


@dataclass(frozen=True)
class IdentitySpec:
    name: str
    output_key: str
    expected_environment: str
    field_names: tuple[str, ...]
    kind: str


_IDENTITY_ROOTS: tuple[tuple[str, ...], ...] = (
    (),
    ("metadata",),
    ("provenance",),
    ("run_provenance",),
    ("hyper_parameters",),
    ("hyper_parameters", "metadata"),
    ("hyper_parameters", "provenance"),
    ("hyper_parameters", "run_provenance"),
    ("hyper_parameters", "config_tree"),
    ("hyper_parameters", "config_tree", "metadata"),
    ("hyper_parameters", "config_tree", "provenance"),
    ("hyper_parameters", "config_tree", "run_provenance"),
    ("hparams",),
    ("hparams", "metadata"),
    ("hparams", "provenance"),
    ("hparams", "run_provenance"),
)

_IDENTITY_SPECS = (
    IdentitySpec(
        name="source",
        output_key="source_commit",
        expected_environment="ICE_EXPECTED_SOURCE_COMMIT",
        field_names=("source_commit", "git_commit", "git_sha", "commit_sha"),
        kind="commit",
    ),
    IdentitySpec(
        name="config",
        output_key="config_sha256",
        expected_environment="ICE_EXPECTED_CONFIG_SHA256",
        field_names=("config_sha256", "resolved_config_sha256"),
        kind="sha256",
    ),
    IdentitySpec(
        name="run",
        output_key="run_id",
        expected_environment="ICE_EXPECTED_RUN_ID",
        field_names=("run_id", "wandb_run_id", "wandb_id"),
        kind="text",
    ),
    IdentitySpec(
        name="split",
        output_key="split_sha256",
        expected_environment="ICE_EXPECTED_SPLIT_SHA256",
        field_names=("split_sha256", "split_manifest_sha256"),
        kind="sha256",
    ),
    IdentitySpec(
        name="normalization",
        output_key="normalization_sha256",
        expected_environment="ICE_EXPECTED_NORMALIZATION_SHA256",
        field_names=(
            "normalization_sha256",
            "normalization_stats_sha256",
            "train_only_normalization_sha256",
            "norm_sha256",
        ),
        kind="sha256",
    ),
)

_SCHEDULER_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("hyper_parameters", "config_tree", "model", "scheduler"),
    ("hyper_parameters", "model", "scheduler"),
    ("config_tree", "model", "scheduler"),
    ("model", "scheduler"),
)

_CALLBACK_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("hyper_parameters", "config_tree", "callbacks"),
    ("hyper_parameters", "callbacks"),
    ("hparams", "config_tree", "callbacks"),
    ("hparams", "callbacks"),
    ("config_tree", "callbacks"),
)
_MODEL_CHECKPOINT_TARGETS = frozenset(
    {
        "lightning.pytorch.callbacks.ModelCheckpoint",
        "pytorch_lightning.callbacks.ModelCheckpoint",
    }
)
_MODEL_CHECKPOINT_STATE_KEY_FIELDS = frozenset(
    {
        "monitor",
        "mode",
        "every_n_train_steps",
        "every_n_epochs",
        "train_time_interval",
    }
)


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _exact_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer: {value!r}")
    return value


def _nested_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _container_children(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, (list, tuple)):
        return value
    return None


def _finite_tensor(
    tensor: torch.Tensor, path: str, *, allow_nonfinite: bool = False
) -> tuple[int, int]:
    try:
        if tensor.device.type == "meta":
            raise RuntimeError(f"checkpoint contains a meta tensor: {path}")
        candidate = tensor.dequantize() if tensor.is_quantized else tensor
        if candidate.layout != torch.strided:
            candidate = candidate.to_dense()
        if (
            candidate.numel()
            and not bool(torch.isfinite(candidate).all())
            and not allow_nonfinite
        ):
            raise RuntimeError(f"checkpoint contains a non-finite tensor: {path}")
        return 1, tensor.numel()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"could not scan checkpoint tensor: {path}") from exc


def _finite_numeric_scalar(value: numbers.Number, path: str) -> None:
    if isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
        finite = math.isfinite(float(value.real)) and math.isfinite(float(value.imag))
    else:
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"could not inspect checkpoint numeric scalar: {path}"
            ) from exc
    if not finite:
        raise RuntimeError(f"checkpoint contains a non-finite scalar: {path}")


def finite_scan(
    value: Any,
    path: str = "root",
    *,
    allowed_nonfinite_paths: frozenset[tuple[Any, ...]] = frozenset(),
) -> ScanStats:
    """Scan every tensor and numeric scalar in an acyclic checkpoint tree."""

    active_containers: set[int] = set()

    def visit(
        current: Any, current_path: str, components: tuple[Any, ...]
    ) -> ScanStats:
        if isinstance(current, torch.Tensor):
            count, numel = _finite_tensor(
                current,
                current_path,
                allow_nonfinite=components in allowed_nonfinite_paths,
            )
            return ScanStats(tensor_count=count, tensor_numel=numel)
        if isinstance(current, bool):
            return ScanStats()
        if isinstance(current, numbers.Number):
            _finite_numeric_scalar(current, current_path)
            return ScanStats(numeric_scalar_count=1)

        children = _container_children(current)
        if children is None:
            return ScanStats()
        identity = id(current)
        if identity in active_containers:
            raise RuntimeError(
                f"checkpoint contains a cyclic container at {current_path}"
            )
        active_containers.add(identity)
        try:
            total = ScanStats()
            if isinstance(current, Mapping):
                for key, child in current.items():
                    total += visit(
                        child,
                        f"{current_path}.{key}",
                        (*components, key),
                    )
            else:
                for index, child in enumerate(children):
                    total += visit(
                        child,
                        f"{current_path}[{index}]",
                        (*components, index),
                    )
            return total
        finally:
            active_containers.remove(identity)

    return visit(value, path, ())


def _model_checkpoint_state_identity(state_key: Any) -> dict[str, Any] | None:
    """Parse Lightning's exact ``ModelCheckpoint.state_key`` representation."""

    prefix = "ModelCheckpoint"
    if not isinstance(state_key, str) or not state_key.startswith(prefix):
        return None
    try:
        identity = ast.literal_eval(state_key[len(prefix) :])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(identity, dict):
        return None
    if frozenset(identity) != _MODEL_CHECKPOINT_STATE_KEY_FIELDS:
        return None
    if identity["monitor"] is not None or identity["mode"] not in {"min", "max"}:
        return None
    for name in ("every_n_train_steps", "every_n_epochs"):
        value = identity[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
    if not (identity["every_n_train_steps"] or identity["every_n_epochs"]):
        return None
    if identity["train_time_interval"] is not None:
        return None
    return identity


def _configured_model_checkpoints(
    payload: Mapping[Any, Any],
) -> tuple[Mapping[Any, Any], ...]:
    configs: list[Mapping[Any, Any]] = []
    for path in _CALLBACK_CONFIG_PATHS:
        callbacks = _nested_value(payload, path)
        if not isinstance(callbacks, Mapping):
            continue
        for config in callbacks.values():
            if (
                isinstance(config, Mapping)
                and config.get("_target_") in _MODEL_CHECKPOINT_TARGETS
            ):
                configs.append(config)
    return tuple(configs)


def _config_matches_unranked_checkpoint(
    config: Mapping[Any, Any], identity: Mapping[str, Any]
) -> bool:
    if config.get("monitor") is not None:
        return False
    if config.get("mode", "min") != identity["mode"]:
        return False
    if config.get("train_time_interval") is not None:
        return False
    if config.get("save_top_k", _MISSING) != -1:
        return False
    for name in ("every_n_train_steps", "every_n_epochs"):
        configured = config.get(name)
        configured = 0 if configured is None else configured
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int)
            or configured != identity[name]
        ):
            return False
    return True


def _is_unranked_model_checkpoint_sentinel(
    state: Any,
    identity: Mapping[str, Any],
) -> bool:
    if not isinstance(state, Mapping):
        return False
    if state.get("monitor", _MISSING) is not None:
        return False
    if state.get("best_model_score", _MISSING) is not None:
        return False
    if state.get("current_score", _MISSING) is not None:
        return False
    if state.get("kth_best_model_path", _MISSING) != "":
        return False
    best_k_models = state.get("best_k_models")
    if not isinstance(best_k_models, Mapping) or best_k_models:
        return False

    value = state.get("kth_value")
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.ndim != 0
        or not value.is_floating_point()
    ):
        return False
    expected = math.inf if identity["mode"] == "min" else -math.inf
    return float(value.item()) == expected


def _allowed_lightning_callback_sentinel_paths(
    payload: Mapping[Any, Any],
) -> frozenset[tuple[Any, ...]]:
    """Allow only Lightning's unused top-k boundary for unranked checkpoints."""

    callbacks = payload.get("callbacks")
    if not isinstance(callbacks, Mapping):
        return frozenset()
    configs = _configured_model_checkpoints(payload)
    allowed: set[tuple[Any, ...]] = set()
    for state_key, state in callbacks.items():
        identity = _model_checkpoint_state_identity(state_key)
        if identity is None:
            continue
        if configs and not any(
            _config_matches_unranked_checkpoint(config, identity) for config in configs
        ):
            continue
        if _is_unranked_model_checkpoint_sentinel(state, identity):
            allowed.add(("callbacks", state_key, "kth_value"))
    return frozenset(allowed)


def _strict_model_reload(
    checkpoint: Path,
    payload: Mapping[Any, Any],
) -> dict[str, int | bool]:
    """Reconstruct the configured ModelWrapper and strictly load its state."""

    hyper_parameters = _require_nonempty_mapping(
        payload.get("hyper_parameters"), "hyper_parameters"
    )
    config_tree = _require_nonempty_mapping(
        hyper_parameters.get("config_tree"),
        "hyper_parameters.config_tree",
    )
    model_config = _require_nonempty_mapping(
        config_tree.get("model"),
        "hyper_parameters.config_tree.model",
    )
    target = model_config.get("_target_")
    if not isinstance(target, str) or not target:
        raise RuntimeError("configured ModelWrapper target is missing")

    try:
        import hydra

        from egomimic.pl_utils.pl_model import ModelWrapper
    except ImportError as exc:
        raise RuntimeError(
            "strict model reload requires the EgoVerse runtime dependencies"
        ) from exc

    wrapper_class = hydra.utils.get_class(target)
    if not isinstance(wrapper_class, type) or not issubclass(
        wrapper_class, ModelWrapper
    ):
        raise RuntimeError(
            "hyper_parameters.config_tree.model._target_ must resolve to "
            f"ModelWrapper, got {target!r}"
        )

    constructor_keys = {
        "scheduler_interval",
        "scheduler_frequency",
        "evaluator",
        "enable_grad_norm",
    }
    constructor_arguments = {
        key: hyper_parameters[key]
        for key in constructor_keys
        if key in hyper_parameters
    }
    constructor_arguments["config_tree"] = config_tree
    try:
        wrapper = wrapper_class(**constructor_arguments)
        incompatible = wrapper.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise RuntimeError(
            f"configured ModelWrapper strict reload failed for {checkpoint}"
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict ModelWrapper reload returned incompatible keys: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    reloaded_scan = finite_scan(wrapper.state_dict(), "strict_model.state_dict")
    if reloaded_scan.tensor_count == 0:
        raise RuntimeError("strictly reloaded ModelWrapper contains no tensors")
    parameter_count = sum(parameter.numel() for parameter in wrapper.parameters())
    return {
        "strict_model_reload": True,
        "strict_model_parameter_count": parameter_count,
        "strict_model_tensor_count": reloaded_scan.tensor_count,
    }


def _require_nonempty_mapping(value: Any, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"missing or empty {label}")
    return value


def _require_nonempty_mapping_list(value: Any, label: str) -> list[Mapping[Any, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"missing or empty {label}")
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping) or not entry:
            raise RuntimeError(f"{label}[{index}] must be a non-empty mapping")
    return value


def _scheduler_configured(payload: Mapping[Any, Any]) -> bool | None:
    observed: list[tuple[str, bool]] = []
    for path in _SCHEDULER_CONFIG_PATHS:
        value = _nested_value(payload, path)
        if value is _MISSING:
            continue
        configured = value is not None and value is not False
        observed.append((".".join(path), configured))
    if not observed:
        return None
    states = {configured for _, configured in observed}
    if len(states) != 1:
        raise RuntimeError(
            "conflicting scheduler configuration metadata: "
            + ", ".join(f"{path}={configured}" for path, configured in observed)
        )
    return observed[0][1]


def _normalize_identity(value: Any, *, kind: str, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} must be a string")
    result = value.strip()
    if result != value or not result or any(ord(character) < 32 for character in result):
        raise RuntimeError(f"{label} is empty or non-canonical")
    if kind == "sha256":
        if re.fullmatch(r"[0-9a-fA-F]{64}", result) is None:
            raise RuntimeError(f"{label} must be a SHA-256 digest")
        return result.lower()
    if kind == "commit":
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", result) is None:
            raise RuntimeError(f"{label} must be a hexadecimal source commit")
        return result.lower()
    return result


def _observed_identity(
    payload: Mapping[Any, Any], spec: IdentitySpec
) -> tuple[str | None, tuple[str, ...]]:
    candidates: list[tuple[str, str]] = []
    for root in _IDENTITY_ROOTS:
        for field_name in spec.field_names:
            path = (*root, field_name)
            value = _nested_value(payload, path)
            if value is _MISSING or value is None:
                continue
            label = ".".join(path)
            normalized = _normalize_identity(value, kind=spec.kind, label=label)
            candidates.append((label, normalized))
    if not candidates:
        return None, ()
    unique = {value for _, value in candidates}
    if len(unique) != 1:
        detail = ", ".join(f"{path}={value!r}" for path, value in candidates)
        raise RuntimeError(f"conflicting {spec.name} checkpoint identities: {detail}")
    return candidates[0][1], tuple(path for path, _ in candidates)


def _expected_value(
    explicit: str | None, spec: IdentitySpec, environment: Mapping[str, str]
) -> str | None:
    raw = explicit
    if raw is None:
        raw = environment.get(spec.expected_environment)
    if raw is None or raw == "":
        return None
    return _normalize_identity(
        raw,
        kind=spec.kind,
        label=f"expected {spec.name} identity",
    )


def _required_scheduler_flag(
    explicit: bool | None, environment: Mapping[str, str]
) -> bool:
    if explicit is not None:
        return explicit
    raw = environment.get("ICE_REQUIRE_LR_SCHEDULERS", "")
    if raw in {"", "0"}:
        return False
    if raw == "1":
        return True
    raise RuntimeError("ICE_REQUIRE_LR_SCHEDULERS must be 0 or 1")


def validate_checkpoint(
    checkpoint: Path,
    *,
    expected_identities: Mapping[str, str | None] | None = None,
    require_lr_schedulers: bool = False,
    strict_model_reload: bool = True,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate ``checkpoint`` and return JSON-serializable metadata."""

    expected_identities = expected_identities or {}
    environment = os.environ if environment is None else environment
    path = checkpoint.expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix != ".ckpt":
        raise RuntimeError(f"not a .ckpt file: {path}")

    before = _stat_signature(path)
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise RuntimeError("checkpoint payload must be a mapping")

    global_step = _exact_nonnegative_integer(
        payload.get("global_step"), "global_step"
    )
    epoch_value = payload.get("epoch", _MISSING)
    epoch = (
        None
        if epoch_value is _MISSING
        else _exact_nonnegative_integer(epoch_value, "epoch")
    )

    state_dict = _require_nonempty_mapping(payload.get("state_dict"), "state_dict")
    optimizer_states = _require_nonempty_mapping_list(
        payload.get("optimizer_states"), "optimizer_states"
    )
    loops = _require_nonempty_mapping(payload.get("loops"), "loops")

    configured_by_metadata = _scheduler_configured(payload)
    scheduler_value = payload.get("lr_schedulers", _MISSING)
    if scheduler_value is _MISSING:
        scheduler_states: list[Mapping[Any, Any]] = []
    elif not isinstance(scheduler_value, list):
        raise RuntimeError("lr_schedulers must be a list when present")
    else:
        scheduler_states = scheduler_value
        for index, entry in enumerate(scheduler_states):
            if not isinstance(entry, Mapping) or not entry:
                raise RuntimeError(
                    f"lr_schedulers[{index}] must be a non-empty mapping"
                )

    scheduler_required = require_lr_schedulers or configured_by_metadata is True
    if scheduler_required and not scheduler_states:
        raise RuntimeError("configured checkpoint is missing lr_schedulers")
    if configured_by_metadata is False and scheduler_states:
        raise RuntimeError(
            "checkpoint carries lr_schedulers while its model config disables them"
        )

    state_scan = finite_scan(state_dict, "root.state_dict")
    if state_scan.tensor_count == 0:
        raise RuntimeError("state_dict contains no tensors")
    callback_sentinel_paths = _allowed_lightning_callback_sentinel_paths(payload)
    full_scan = finite_scan(
        payload,
        allowed_nonfinite_paths=callback_sentinel_paths,
    )
    strict_reload_metadata: dict[str, int | bool]
    if strict_model_reload:
        strict_reload_metadata = _strict_model_reload(path, payload)
    else:
        strict_reload_metadata = {"strict_model_reload": False}

    identities: dict[str, str] = {}
    identity_sources: dict[str, list[str]] = {}
    for spec in _IDENTITY_SPECS:
        observed, sources = _observed_identity(payload, spec)
        expected = _expected_value(
            expected_identities.get(spec.name), spec, environment
        )
        if expected is not None:
            if observed is None:
                raise RuntimeError(
                    f"expected {spec.name} identity is not present in checkpoint"
                )
            if observed != expected:
                raise RuntimeError(
                    f"{spec.name} checkpoint identity mismatch: "
                    f"{observed!r} != {expected!r}"
                )
        if observed is not None:
            identities[spec.output_key] = observed
            identity_sources[spec.output_key] = list(sources)

    after = _stat_signature(path)
    if before != after:
        raise RuntimeError("checkpoint changed while being validated")

    result: dict[str, Any] = {
        "schema_version": 1,
        "valid": True,
        "validation": "FULL_STATE_LIGHTNING_CHECKPOINT",
        "checkpoint_path": str(path),
        "checkpoint_size_bytes": before[2],
        "global_step": global_step,
        "state_dict_entries": len(state_dict),
        "optimizer_state_count": len(optimizer_states),
        "lr_scheduler_state_count": len(scheduler_states),
        "scheduler_configured": bool(scheduler_states) or scheduler_required,
        "loop_state_entries": len(loops),
        "tensor_count": full_scan.tensor_count,
        "tensor_numel": full_scan.tensor_numel,
        "numeric_scalar_count": full_scan.numeric_scalar_count,
        "allowed_callback_sentinel_count": len(callback_sentinel_paths),
        "identity_sources": identity_sources,
    }
    if epoch is not None:
        result["epoch"] = epoch
    result.update(strict_reload_metadata)
    result.update(identities)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one full-state Lightning checkpoint for ICE resume."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--expected-source-commit",
        "--expected-source-identity",
        dest="expected_source",
    )
    parser.add_argument(
        "--expected-config-sha256",
        "--expected-config-identity",
        dest="expected_config",
    )
    parser.add_argument(
        "--expected-run-id",
        "--expected-run-identity",
        dest="expected_run",
    )
    parser.add_argument(
        "--expected-split-sha256",
        "--expected-split-identity",
        dest="expected_split",
    )
    parser.add_argument(
        "--expected-normalization-sha256",
        "--expected-normalization-identity",
        "--expected-norm-sha256",
        dest="expected_normalization",
    )
    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument(
        "--require-lr-schedulers",
        dest="require_lr_schedulers",
        action="store_true",
    )
    scheduler_group.add_argument(
        "--no-require-lr-schedulers",
        dest="require_lr_schedulers",
        action="store_false",
    )
    parser.set_defaults(require_lr_schedulers=None)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help=(
            "skip configured ModelWrapper reconstruction; intended only for "
            "synthetic or non-project checkpoints"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        require_lr_schedulers = _required_scheduler_flag(
            args.require_lr_schedulers, os.environ
        )
        result = validate_checkpoint(
            args.checkpoint,
            expected_identities={
                "source": args.expected_source,
                "config": args.expected_config,
                "run": args.expected_run,
                "split": args.expected_split,
                "normalization": args.expected_normalization,
            },
            require_lr_schedulers=require_lr_schedulers,
            strict_model_reload=not args.structural_only,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"checkpoint validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
