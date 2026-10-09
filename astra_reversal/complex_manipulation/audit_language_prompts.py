"""Post-hoc CPU audit of the recorded teacher-review policy text inputs.

Reconstruct the released tokenizer's state formatting at review steps only.
This never modifies a policy, observation, prompt, or experiment outcome.
"""

import argparse
import hashlib
import importlib.metadata
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(runs, assets):
    import sentencepiece

    if importlib.metadata.version("sentencepiece") != "0.2.2":
        raise ValueError("Use the experiment's SentencePiece version, 0.2.2")
    receipt = read(assets / "receipt.json")
    for item in receipt["files"]:
        path = assets / item["name"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("Audit support assets differ from the archived copy")
    stats = read(assets / "norm_stats.json")["norm_stats"]["state"]
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    if mean.shape != (32,) or std.shape != (32,):
        raise ValueError("Expected the released 32-dimensional normalizer")
    tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(assets / "paligemma_tokenizer.model")
    )
    warnings, reviews = [], []
    for run in runs:
        log = run / "evaluation.log"
        lengths = [
            int(value)
            for value in re.findall(
                r"Token length \((\d+)\) exceeds max length \(200\)",
                log.read_text(errors="replace"),
            )
        ]
        warnings.append(
            {
                "run": run.name,
                "log_sha256": sha256(log),
                "warning_count": len(lengths),
                "untruncated_length_counts": dict(Counter(lengths)),
            }
        )
        for directory in sorted((run / "evaluation").iterdir()):
            if not directory.is_dir():
                continue
            requests = sorted((directory / "guidance").glob("request_*.json"))
            if not requests:
                continue
            predictions = [
                json.loads(line)
                for line in (directory / "predictions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            by_step = {row["step"]: row for row in predictions}
            if len(by_step) != len(predictions):
                raise ValueError("Duplicate model prediction step")
            for request_path in requests:
                request = read(request_path)
                step = request["observation_step"]
                proposal_path = request_path.with_name(
                    request_path.name.replace("request_", "proposal_", 1)
                )
                if not proposal_path.exists() or step not in by_step:
                    continue  # No policy input was executed from this request.
                proposal = read(proposal_path)
                prompt = request["context"]["original_instruction"]
                if proposal["method"] == "phase_prompt":
                    prompt += " Current phase: " + proposal["subgoal"].strip()
                if prompt != by_step[step]["model_prompt"]:
                    raise ValueError(
                        "Review reconstruction differs from executed prompt"
                    )
                snapshots = [
                    row for row in request["snapshots"] if row["origin"] == "current"
                ]
                if len(snapshots) != 1 or snapshots[0]["step"] != step:
                    raise ValueError("Review has no unique bound current state")
                state = np.asarray(snapshots[0]["state"])
                if state.shape != (16,) or not np.isfinite(state).all():
                    raise ValueError("Expected the native finite 16D state")
                # Exact released order: pad 16 -> 32, z-score with epsilon,
                # then discretize. The original float64 state is retained.
                normalized = (np.pad(state, (0, 16)) - mean) / (std + 1e-6)
                discrete = (
                    np.digitize(normalized, bins=np.linspace(-1, 1, 257)[:-1]) - 1
                )
                cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
                language = f"Task: {cleaned}"
                state_text = " ".join(map(str, discrete))
                state_prefix = f"{language}, State: {state_text}"
                full = state_prefix + ";\nAction: "
                tokens = tokenizer.encode(full, add_bos=True)
                decoded_full = tokenizer.decode(tokens)
                decoded_retained = tokenizer.decode(tokens[:200])

                def prefix_retained(text):
                    # Token IDs can change at the end of a separately encoded
                    # prefix. Compare decoded normalized text, then verify that
                    # it is actually a prefix of this complete input.
                    prefix = tokenizer.decode(tokenizer.encode(text, add_bos=True))
                    if not decoded_full.startswith(prefix):
                        return None
                    return decoded_retained.startswith(prefix)

                reviews.append(
                    {
                        "run": run.name,
                        "episode": directory.name,
                        "step": step,
                        "request_sha256": sha256(request_path),
                        "proposal_sha256": sha256(proposal_path),
                        "method": proposal["method"],
                        "untruncated_tokens": len(tokens),
                        "dropped_tokens": max(0, len(tokens) - 200),
                        "instruction_prefix_retained": prefix_retained(language),
                        "all_state_values_retained": prefix_retained(state_prefix),
                        "decoded_dropped_tail": tokenizer.decode(tokens[200:]),
                        "decoded_before_truncation": decoded_full,
                        "decoded_after_truncation": decoded_retained,
                    }
                )
    return {
        "kind": "posthoc_language_capacity_audit",
        "policy_or_observation_modified": False,
        "max_token_len": 200,
        "sentencepiece_version": importlib.metadata.version("sentencepiece"),
        "audit_script_sha256": sha256(Path(__file__)),
        "support_assets": receipt,
        "scope": "Every full allocation log plus reconstructed inputs at executed teacher-review steps only. State is not recorded at every intervening policy call, so the decoded truncation tail is not known for all warnings.",
        "logs": warnings,
        "review_count": len(reviews),
        "reviews_over_capacity": sum(row["dropped_tokens"] > 0 for row in reviews),
        "reviews": reviews,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.runs, args.assets)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("review_count", "reviews_over_capacity", "logs")
            }
        )
    )


if __name__ == "__main__":
    main()
