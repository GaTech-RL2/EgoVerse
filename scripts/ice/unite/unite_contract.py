#!/usr/bin/env python3
"""Validate a resolved clean UNITE configuration before ICE execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

EXPECTED_ROWS = {
    (True, 4, 14),
    (True, 8, 14),
    (True, 8, 42),
    (True, 16, 14),
    (False, 4, 14),
    (False, 8, 14),
}


def _select(config: Any, key: str, *, required: bool = True) -> Any:
    value = OmegaConf.select(config, key, default=None)
    if required and value is None:
        raise ValueError(f"missing required UNITE config field: {key}")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not 0.0 < number < float("inf"):
        raise ValueError(f"{label} must be finite and positive")
    return number


def _sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    def identity(item: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
    if identity(before) != identity(after):
        raise RuntimeError(f"config changed while hashing: {path}")
    return digest


def _stage_by_target(config: Any, suffix: str) -> Any:
    stages = _select(config, "model.pipeline.stages")
    matches = [
        stage
        for stage in stages
        if str(stage.get("_target_", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one pipeline stage ending in {suffix!r}")
    return matches[0]


def validate_unite_config(config: Any, *, expected_world_size: int) -> dict[str, Any]:
    """Return a stable contract summary or fail on an incompatible config."""

    world_size = _positive_integer(expected_world_size, "expected_world_size")
    configured_devices = _positive_integer(
        int(_select(config, "trainer.devices")), "trainer.devices"
    )
    if configured_devices != world_size:
        raise ValueError(
            f"trainer.devices/world-size mismatch: {configured_devices} != {world_size}"
        )

    shared = bool(_select(config, "model.share_encoder_denoiser"))
    tokens = _positive_integer(
        int(_select(config, "model.num_latent_tokens")), "model.num_latent_tokens"
    )
    policy = _stage_by_target(config, "ReleasedRecipeUniteLatentPolicy")
    flow_steps = _positive_integer(
        int(policy.flow_steps_per_reconstruction),
        "model.pipeline policy flow_steps_per_reconstruction",
    )
    if (shared, tokens, flow_steps) not in EXPECTED_ROWS:
        raise ValueError(
            "unsupported UNITE topology, register count, or flow ratio"
        )
    latent_dim = _positive_integer(
        int(_select(config, "model.latent_dim")), "model.latent_dim"
    )
    if latent_dim != 16:
        raise ValueError("clean UNITE register sweep requires latent_dim=16")

    if int(policy.dopri5_output_points) != 50:
        raise ValueError("clean UNITE Dopri5 contract requires 50 output points")
    if float(policy.reconstruction_noising_start) != 0.7:
        raise ValueError("clean UNITE reconstruction noising start must be 0.7")
    _finite_positive(policy.dopri5_atol, "dopri5_atol")
    _finite_positive(policy.dopri5_rtol, "dopri5_rtol")

    ema_target = str(_select(config, "callbacks.ema._target_"))
    if not ema_target.endswith("EMACallback"):
        raise ValueError("UNITE continuation requires the repository EMA callback")
    filename = str(_select(config, "callbacks.model_checkpoint.filename"))
    if "{epoch}" not in filename or "{step}" not in filename:
        raise ValueError("checkpoint filename must contain immutable epoch and step")

    view = _select(config, "evaluator.energy_score_validation_view")
    view_world_size = _positive_integer(int(view.world_size), "EnergyScore world_size")
    per_rank = _positive_integer(
        int(view.per_rank_batch_size), "EnergyScore per_rank_batch_size"
    )
    if view_world_size != world_size or per_rank * world_size != 32:
        raise ValueError("EnergyScore validation view must contain exactly 32 conditions")

    diagnostics = _select(config, "evaluator.unite_diagnostics", required=False)
    diagnostics_status = "ABSENT"
    if diagnostics is not None and bool(diagnostics.get("enabled", True)):
        diagnostic_view = diagnostics.get("validation_view")
        if not isinstance(diagnostic_view, Mapping):
            raise ValueError("UNITE diagnostics require a validation_view mapping")
        diagnostic_world = _positive_integer(
            int(diagnostic_view["world_size"]), "diagnostic world_size"
        )
        diagnostic_per_rank = _positive_integer(
            int(diagnostic_view["per_rank_batch_size"]),
            "diagnostic per_rank_batch_size",
        )
        if diagnostic_world != world_size or diagnostic_per_rank * world_size != 32:
            raise ValueError("UNITE diagnostic validation view must total 32 conditions")
        diagnostics_status = "BOUND_TO_WORLD_SIZE"

    return {
        "schema_version": 1,
        "status": "READY",
        "model_family": "unite-register-sweep",
        "row": {
            "share_encoder_denoiser": shared,
            "num_latent_tokens": tokens,
            "latent_dim": latent_dim,
        },
        "world_size": world_size,
        "flow_steps_per_reconstruction": flow_steps,
        "reconstruction_noising_start": 0.7,
        "sampler": {"name": "dopri5", "output_points": 50},
        "energy_score_conditions": 32,
        "diagnostics": diagnostics_status,
        "checkpoint_filename": filename,
        "ema_target": ema_target,
    }


def _publish(path: Path, payload: Mapping[str, Any]) -> str:
    if not path.is_absolute():
        raise ValueError("output must be absolute")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    os.chmod(path, 0o444)
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve(strict=True)
    config = OmegaConf.load(config_path)
    payload = validate_unite_config(config, expected_world_size=args.expected_world_size)
    payload["config"] = str(config_path)
    payload["config_sha256"] = _sha256(config_path)
    if args.output is not None:
        payload["record_sha256"] = _publish(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
