"""Compare complete-tree measurements with Table 8 without selecting seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .data import write_json

PAPER = "https://arxiv.org/html/2607.21856v1"
# Each pair is (mean, inference-seed standard deviation), not a confidence interval.
TARGETS = {
    "coldstart_thinking": {"single": (1.351, 0.226), "oracle_bon": (0.107, 0.026)},
    "coldstart_bc": {"single": (0.003, 0.001), "oracle_bon": (0.001, 0.0002)},
    "exit": {"single": (0.087, 0.018), "oracle_bon": (0.029, 0.007)},
    "leact": {"single": (0.016, 0.004), "oracle_bon": (0.006, 0.002)},
}


def comparison(report: dict) -> dict:
    scopes = {"Leduc 6r2s independent reproduction",
              "Leduc 6r2s independent reproduction with projection-sampled frontier warmstart"}
    if report.get("scope") not in scopes or report["oracle"]["num_infosets"] != 4032:
        raise ValueError("Expected Leduc 6r2s benchmark measurements, not preflight results")
    pools = {}
    for name in ["coldstart_thinking", "coldstart_bc"]:
        if name + "_all_epochs" in report:
            pools[name] = [(0, epoch) for epoch in report[name + "_all_epochs"]]
        elif name in report:
            pools[name] = [(0, report[name])]
    for method, rounds in report["methods"].items():
        if method not in TARGETS:
            raise ValueError(f"Unknown comparison method: {method}")
        pools[method] = [(row["round"], epoch) for row in rounds for epoch in row["all_epochs"]]
    rows = []
    for method, candidates in pools.items():
        if not candidates:
            continue
        for regime in ["single", "oracle_bon"]:
            # Select the checkpoint's aggregate mean, never the best individual seed.
            round_index, best = min(candidates, key=lambda pair: pair[1][regime]["exploitability"]["mean"])
            metric = best[regime]["exploitability"]
            reference_mean, reference_std = TARGETS[method][regime]
            rows.append({
                "method": method, "regime": regime, "mean": metric["mean"], "std": metric["std"],
                "round": round_index, "checkpoint": best["checkpoint"],
                "inference_seeds": best["inference_seeds"],
                "paper_mean": reference_mean, "paper_std": reference_std,
                "mean_difference_from_paper": metric["mean"] - reference_mean,
            })
    planned = report["protocol"].get("planned_methods", ["leact", "exit"])
    if not planned or any(m not in {"leact", "exit"} for m in planned):
        raise ValueError("Invalid planned comparison methods")
    complete = report["protocol"]["rounds"] > 0 and all(len(report["methods"].get(m, [])) == report["protocol"]["rounds"]
                   for m in planned)
    return {
        "scope": report["scope"], "all_planned_rounds_complete": complete,
        "paper_source": PAPER, "oracle_exploitability": report["oracle"]["exploitability"],
        "oracle_sha256": report["oracle"]["sha256"],
        "selection": "best mean across available epochs/rounds, independently for each metric",
        "oracle_bon_requires_teacher": True, "rows": rows,
        "interpretation": (
            "Independent reconstruction, not an assertion of numerical reproduction. "
            "BC and reasoning coldstarts are separate protocols; reference labels alone do not establish matched initialization. "
            "Best-across-epochs requires explicit all-epoch measurements; legacy single-checkpoint reports remain final-only. "
            "Original prompts and coldstart artifacts were unavailable. "
            "Standard deviations describe inference seeds of one trained model, not training replication."
        ),
    }


def write_comparison(report: dict, output: Path) -> dict:
    result = comparison(report)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", result)
    if result["rows"]:
        path = output / "comparison.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_comparison(json.loads(args.results.read_text()), args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
