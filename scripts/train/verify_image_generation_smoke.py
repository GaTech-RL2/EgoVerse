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
    args = parser.parse_args()

    require_text(args.overfit_log, "OVERFIT_GATE_PASSED")
    require_text(args.smoke_log, "TRAIN_METRICS")
    require_text(args.smoke_log, "VALIDATION_METRICS")
    require_text(args.smoke_log, "SMOKE_GATE_PASSED")

    required = [
        args.smoke_dir / "checkpoint-last.pth",
        args.smoke_dir / "resolved_config.json",
        args.smoke_dir / "metrics.jsonl",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty smoke artifact: {path}")
    if not list(args.smoke_dir.glob("samples-step*.png")):
        raise RuntimeError("Smoke did not produce a fully sampled validation grid")

    validation_rows = []
    for line in (args.smoke_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("split") == "validation":
            validation_rows.append(row)
    if not validation_rows:
        raise RuntimeError("No post-training validation rows were recorded")
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
