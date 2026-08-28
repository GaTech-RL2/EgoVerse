#!/usr/bin/env python3
"""Fail closed unless overfit, optimization, validation, and reload artifacts exist."""

import argparse
import json
import math
from pathlib import Path


def require_text(path: Path, marker: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if marker not in text:
        raise RuntimeError(f"Missing {marker!r} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit-log", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--expected-architecture", default="")
    parser.add_argument("--stress-log", type=Path)
    parser.add_argument("--resume-log", type=Path, required=True)
    args = parser.parse_args()

    require_text(args.overfit_log, "OVERFIT_GATE_PASSED")
    require_text(args.smoke_log, "TRAIN_METRICS")
    require_text(args.smoke_log, "VALIDATION_METRICS")
    require_text(args.smoke_log, "SMOKE_GATE_PASSED")
    require_text(args.resume_log, "CHECKPOINT_LOADED step=2")
    require_text(args.resume_log, "VALIDATION_METRICS")
    require_text(args.resume_log, "SMOKE_GATE_PASSED")
    if args.stress_log is not None:
        require_text(args.stress_log, '"unroll_steps": 8.0')
        require_text(args.stress_log, "CHECKPOINT_SAVED step=1")

    required = [
        args.smoke_dir / "checkpoint-last.pth",
        args.smoke_dir / "resolved_config.json",
        args.smoke_dir / "metrics.jsonl",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty smoke artifact: {path}")
    config = json.loads((args.smoke_dir / "resolved_config.json").read_text())
    if args.expected_architecture and config.get("architecture") != args.expected_architecture:
        raise RuntimeError(
            f"Expected architecture {args.expected_architecture}, got "
            f"{config.get('architecture')}"
        )
    if args.expected_architecture == "jit_endpoint":
        expected = {
            "base_lr": 3e-5,
            "min_lr": 3e-6,
            "warmup_steps": 3000,
            "warmup_start_factor": 0.1,
            "lr_total_steps": 240000,
            "lr_schedule": "action_warmup_cosine",
            "weight_decay": 1e-4,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "effective_batch": 1024,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"Action optimizer mismatch for {key}: "
                    f"expected {value!r}, got {config.get(key)!r}"
                )
    if not list(args.smoke_dir.glob("samples-step*.png")):
        raise RuntimeError("Smoke did not produce a fully sampled validation grid")

    train_rows = []
    validation_rows = []
    for line in (args.smoke_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("split") == "train":
            train_rows.append(row)
        if row.get("split") == "validation":
            validation_rows.append(row)
    if not train_rows:
        raise RuntimeError("No optimizer-step training rows were recorded")
    for row in train_rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"Non-finite training metric {key}={value}")
    if args.expected_architecture == "jit_endpoint":
        lrs = [row["lr"] for row in train_rows]
        if min(lrs) < 3e-6 - 1e-12 or max(lrs) > 3e-5 + 1e-12:
            raise RuntimeError(f"JiT endpoint LR escaped action range: {lrs}")
    if not validation_rows:
        raise RuntimeError("No post-training validation rows were recorded")
    if "sample_class_effect_mse" not in validation_rows[-1]:
        raise RuntimeError("Validation did not measure same-noise class effect")
    for row in validation_rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"Non-finite validation metric {key}={value}")
    print(
        "IMAGE_GENERATION_SMOKE_VERIFIED "
        + json.dumps(validation_rows[-1], sort_keys=True)
    )


if __name__ == "__main__":
    main()
