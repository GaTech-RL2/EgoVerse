#!/usr/bin/env python3
"""Run one shard of the exhaustive D/M/R corrected-ARC replay grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import codec_candidate_sweep as sweep
from codec_grid_definition import grid


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    names = [row.get("candidate") for row in rows]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("candidate manifest names must be non-empty and unique")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=29)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    rows = load_candidates(args.candidate_manifest)
    if not 0 <= args.index < len(rows):
        raise ValueError(f"index {args.index} outside [0,{len(rows) - 1}]")
    selected = rows[args.index]
    config = selected["config"]
    name = selected["candidate"]
    sweep.CANDIDATES = {name: config}
    shard_dir = args.output_root / f"{args.index:04d}_{name}"
    output = shard_dir / "results.json"
    receipt_path = shard_dir / "SHARD_RECEIPT.json"
    gate_path = shard_dir / "CODEC_CANDIDATE_GATE.json"
    if not args.preflight_only and receipt_path.is_file() and output.is_file() and gate_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("index") == args.index and receipt.get("candidate") == name:
            print(json.dumps({"status": "SKIP_COMPLETE", "index": args.index, "candidate": name}))
            return 0
    sys.argv = [
        sys.argv[0],
        "--dataset", str(args.dataset),
        "--split-manifest", str(args.split_manifest),
        "--output", str(output),
        "--expected-episodes", str(args.expected_episodes),
    ]
    if args.candidate_manifest is not None:
        sys.argv.extend(["--candidate-manifest-sha256", sha256(args.candidate_manifest)])
    if args.preflight_only:
        sys.argv.append("--preflight-only")
    result = sweep.main()
    if not args.preflight_only:
        receipt = {
            "index": args.index,
            "candidate": name,
            "config": config,
            "result_path": str(output),
            "gate_path": str(gate_path),
            "diagnostic_exit": result,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # Gate failure is a valid measured result. Fatal errors still raise before here.
    return 0 if result in (0, 2) else result


if __name__ == "__main__":
    raise SystemExit(main())
