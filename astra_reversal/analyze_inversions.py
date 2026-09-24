"""Audit recorded policy-reference inversions without loading the model."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import read_events
from .flow import error_metrics
from .records import digest


def analyze(directory, noise_atol):
    directory = Path(directory).resolve()
    if not np.isfinite(noise_atol) or noise_atol <= 0:
        raise ValueError("Declare a positive finite noise tolerance")

    def array(reference):
        path = (directory / reference["array"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Array path escapes the run directory")
        value = np.load(path, allow_pickle=False)
        if (
            digest(value) != reference["sha256"]
            or list(value.shape) != reference["shape"]
            or str(value.dtype) != reference["dtype"]
        ):
            raise ValueError("Recorded array integrity mismatch")
        return value

    rows, pending = [], None
    for event in read_events(directory):
        if event["kind"] == "inversion":
            result = event["result"]
            reference = result.get("reference")
            if reference is None or "known_noise" not in reference:
                pending = None  # Genuine Astra references have no known noise.
                continue
            known, recovered = array(reference["known_noise"]), array(result["noise"])
            metrics = error_metrics(known, recovered)
            row = {
                "episode_id": event["episode_id"],
                "plan_id": result["plan_id"],
                "observation_step": reference["observation_step"],
                "condition_id": result["inverse_condition_id"],
                "known_noise_recovery": metrics,
                "action_channel_noise_error": error_metrics(
                    known[..., :7], recovered[..., :7]
                ),
                "padding_channel_noise_error": error_metrics(
                    known[..., 7:], recovered[..., 7:]
                ),
                "passed_noise_tolerance": metrics["max_abs"] <= noise_atol,
            }
            rows.append(row)
            pending = (row, result)
        elif event["kind"] == "flow" and event["role"] == "generation" and pending:
            row, result = pending
            if event["condition_id"] == row["condition_id"]:
                np.testing.assert_array_equal(
                    array(event["input"]), array(result["noise"])
                )
                row["same_condition_internal_reconstruction"] = error_metrics(
                    array(result["model_actions"]), array(event["output"])
                )
            pending = None
    if not rows:
        raise ValueError("No recorded inversions with known reference noise")
    return {
        "scope": "Post-hoc audit of actual control-rollout reference inversions; arrays hash-verified",
        "noise_atol": noise_atol,
        "inversions": len(rows),
        "passing_inversions": sum(row["passed_noise_tolerance"] for row in rows),
        "maximum_noise_error": max(
            row["known_noise_recovery"]["max_abs"] for row in rows
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--noise-atol", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.directory, args.noise_atol)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
