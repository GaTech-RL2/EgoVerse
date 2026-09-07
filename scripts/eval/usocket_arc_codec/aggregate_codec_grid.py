#!/usr/bin/env python3
"""Validate and rank all 896 exhaustive codec-grid shard results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from codec_grid_definition import grid


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(path: Path | None) -> list[dict]:
    if path is None:
        return [
            {
                "candidate": f"D{int(row['distance'])}_M{row['M']}_R{int(row['degrees'])}deg",
                "config": row,
            }
            for row in grid()
        ]
    payload = json.loads(path.read_text())
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate manifest must contain a non-empty candidates list")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--expected-candidates", type=int)
    args = parser.parse_args()
    manifest_rows = load_candidates(args.candidate_manifest)
    expected = args.expected_candidates or len(manifest_rows)
    if len(manifest_rows) != expected:
        raise ValueError(f"expected {expected} candidates, manifest resolved {len(manifest_rows)}")
    candidates = []
    errors = []
    for index, selected in enumerate(manifest_rows):
        config = selected["config"]
        name = selected["candidate"]
        shard = args.shard_root / f"{index:04d}_{name}"
        gate_path = shard / "CODEC_CANDIDATE_GATE.json"
        result_path = shard / "results.json"
        receipt_path = shard / "SHARD_RECEIPT.json"
        if not all(path.is_file() for path in (gate_path, result_path, receipt_path)):
            errors.append({"index": index, "candidate": name, "error": "missing artifact"})
            continue
        try:
            gate = json.loads(gate_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            candidate = gate["candidates"][name]["gate"]
            free = candidate["free_running"]["summary"]
            reset = candidate["state_reset"]["summary"]
            values = [
                free["peak_coverage_mean"], free["delta_vs_raw_mean"], free["success_rate_0.80"],
                free["xy_error_mean_mean"], free["angle_error_mean_deg_mean"],
                reset["object_endpoint_position_error_mean"],
                reset["object_endpoint_position_error_p95"],
                reset["coverage_absolute_delta_mean"],
            ]
            if receipt["index"] != index or receipt["candidate"] != name or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("identity or finite-metric validation failed")
            candidates.append({
                "index": index,
                "candidate": name,
                "D": int(config["distance"]),
                "M": int(config["M"]),
                "R_deg": int(config["degrees"]),
                "status": candidate["status"],
                "free_status": candidate["free_running"]["status"],
                "reset_status": candidate["state_reset"]["status"],
                "peak_coverage_mean": float(free["peak_coverage_mean"]),
                "delta_vs_raw_mean": float(free["delta_vs_raw_mean"]),
                "delta_vs_raw_median": float(free["delta_vs_raw_median"]),
                "success_rate_0.80": float(free["success_rate_0.80"]),
                "xy_error_mean": float(free["xy_error_mean_mean"]),
                "angle_error_mean_deg": float(free["angle_error_mean_deg_mean"]),
                "reset_object_error_mean": float(reset["object_endpoint_position_error_mean"]),
                "reset_object_error_p95": float(reset["object_endpoint_position_error_p95"]),
                "reset_coverage_abs_delta_mean": float(reset["coverage_absolute_delta_mean"]),
                "result_sha256": sha256(result_path),
                "gate_sha256": sha256(gate_path),
            })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"index": index, "candidate": name, "error": str(exc)})
    candidates.sort(
        key=lambda row: (
            row["status"] != "PASS",
            -row["success_rate_0.80"],
            -row["delta_vs_raw_mean"],
            row["reset_object_error_mean"],
        )
    )
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "expected_candidates": expected,
        "complete_candidates": len(candidates),
        "errors": errors,
        "passing_candidates": sum(row["status"] == "PASS" for row in candidates),
        "top_100": candidates[:100],
        "candidate_manifest_sha256": sha256(args.candidate_manifest) if args.candidate_manifest else None,
    }
    (args.output_dir / "GRID_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "GRID_RANKING.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]) if candidates else ["rank"])
        writer.writeheader()
        writer.writerows(candidates)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if len(candidates) == expected and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
