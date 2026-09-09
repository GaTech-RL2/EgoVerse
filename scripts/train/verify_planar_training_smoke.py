#!/usr/bin/env python3
"""Verify a real Planar optimizer/validation smoke and write its gate record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

from egomimic.pl_utils.pl_model import ModelWrapper


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wandb_history(path: Path) -> tuple[dict[int, dict[str, float]], int]:
    store = DataStore()
    store.open_for_scan(str(path))
    rows: dict[int, dict[str, float]] = {}
    exits = []
    try:
        while (payload := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(payload)
            kind = record.WhichOneof("record_type")
            if kind == "exit":
                exits.append(int(record.exit.exit_code))
            if kind != "history":
                continue
            row = {}
            for item in record.history.item:
                key = item.key or ".".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, TypeError):
                    pass
            if "trainer/global_step" not in row:
                continue
            step = int(row["trainer/global_step"])
            for key, value in row.items():
                if key.startswith(("Train/", "Valid/", "Optimizer/", "Timing/")):
                    try:
                        rows.setdefault(step, {})[key] = float(value)
                    except (TypeError, ValueError):
                        pass
    finally:
        store.close()
    assert exits and exits[-1] == 0, exits
    return rows, exits[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--parameter-count", required=True, type=int)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config_path = run_dir / ".hydra/config.yaml"
    checkpoint_path = run_dir / "checkpoints/last.ckpt"
    assert config_path.is_file() and checkpoint_path.is_file()
    cfg = OmegaConf.load(config_path)
    assert cfg.name == "planar_v2_cotrain_clean_dp_standard_h16"
    assert cfg.run_provenance.obstacle_data is False
    assert int(cfg.trainer.max_steps) == 2
    assert int(cfg.trainer.val_check_interval) == 1
    assert int(cfg.trainer.limit_val_batches) == 1
    assert int(cfg.trainer.devices) == args.world_size
    assert int(cfg.trainer.num_nodes) == 1
    assert str(cfg.trainer.precision) == "bf16"
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    assert cfg.callbacks.model_checkpoint.filename == "epoch-{epoch}-step-{step}"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert int(checkpoint["global_step"]) == 2
    assert checkpoint.get("optimizer_states")
    assert len(checkpoint.get("lr_schedulers", [])) == 1
    del checkpoint

    restored = ModelWrapper.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=True, weights_only=False
    )
    restored_count = sum(parameter.numel() for parameter in restored.parameters())
    assert restored_count == args.parameter_count
    del restored

    streams = [
        *run_dir.glob("wandb/run-*/run-*.wandb"),
        *run_dir.glob("wandb/offline-run-*/run-*.wandb"),
    ]
    assert len(streams) == 1, streams
    history, exit_code = _wandb_history(streams[0])
    labels = ("pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper")
    train_required = {"Train/MSE", *(f"Train/MSE/{label}" for label in labels)}
    valid_required = {
        "Valid/MSE",
        "Valid/Native_MSE",
        "Valid/EnergyScore@32",
        "Valid/EnergyScoreAccuracy@32",
        "Valid/EnergyScoreDiversity@32",
    }
    for label in labels:
        valid_required.update(
            {
                f"Valid/MSE/{label}",
                f"Valid/Native_MSE/{label}",
                f"Valid/EnergyScore@32/{label}",
                f"Valid/EnergyScoreAccuracy@32/{label}",
                f"Valid/EnergyScoreDiversity@32/{label}",
            }
        )
    train_rows = [row for row in history.values() if train_required <= row.keys()]
    valid_rows = [
        row
        for step, row in history.items()
        if step >= 1 and valid_required <= row.keys()
    ]
    assert train_rows and valid_rows, history
    checked = {
        key: value
        for row in (train_rows[-1], valid_rows[-1])
        for key, value in row.items()
        if key in train_required or key in valid_required
    }
    assert all(math.isfinite(value) for value in checked.values()), checked

    artifacts = sorted(run_dir.glob("validation_predictions/energy_score/**/*.pt"))
    assert artifacts
    artifact = torch.load(artifacts[-1], map_location="cpu", weights_only=False)
    assert artifact["metric"] == "EnergyScore@32"
    assert len(artifact["seeds"]) == 32 and len(set(artifact["seeds"])) == 32
    assert set(artifact["domains"]) == set(labels)

    result = {
        "status": "passed",
        "repo_head": args.expected_head,
        "run_dir": str(run_dir),
        "world_size": args.world_size,
        "global_step": 2,
        "strict_checkpoint_reload": "passed",
        "parameter_count": restored_count,
        "config_sha256": _sha256(config_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "wandb_stream_sha256": _sha256(streams[0]),
        "wandb_exit_code": exit_code,
        "energy_artifact_count": len(artifacts),
        "energy_artifact_sha256": _sha256(artifacts[-1]),
        "metrics": checked,
    }
    destination = run_dir / "SMOKE_RESULT.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(f"[smoke] PASS {destination}")


if __name__ == "__main__":
    main()
