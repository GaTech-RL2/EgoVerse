"""Full frontier reproduction or a fixed-pool self-generated coldstart scaling study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..core import parse_policy
from ..data import file_sha256, fingerprint, load_oracle, read_jsonl, write_json, write_jsonl
from ..experiment import (MODEL_REVISION, epoch_checkpoints, evaluate_checkpoint,
                          parallel_generate, verify_training_update)
from ..pipeline import load_shards
from ..reproduction import phase, persist_selected, scratch_training
from ..training import fits_context, prepare_data, supervised_pairs
from ..compare import write_comparison
from .collect_full import nested_states


def export_corpus(states, metadata, candidates, directory, tokenizer, *, retained_per_state, source):
    """Fixed candidate order, format/context eligibility only, CFR action replacement."""
    spec = {"state_ids": [s["state_id"] for s in states], "retained_per_state": retained_per_state,
            "source": source, "oracle_sha256": metadata["sha256"], "cutoff_len": 2048,
            "prompt_variant": "concise_v1", "candidate_pool_sha256": fingerprint(candidates)}
    if (directory / "manifest.json").exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("corpus_spec") != spec or file_sha256(directory / "data.json") != manifest["data_sha256"]:
            raise ValueError("Previously exported coldstart corpus differs")
        return manifest
    data, audits, token_lengths = [], [], []
    for state in states:
        kept = 0
        audit = {"state_id": state["state_id"], "candidates": []}
        for index, candidate in enumerate(candidates[state["state_id"]]):
            reason = None
            if candidate.get("finish_reason") != "stop" or not candidate.get("reasoning", "").strip():
                reason = "incomplete or empty reasoning"
            if candidate.get("fallback") is not None:
                reason = "invalid generated policy"
            pairs = supervised_pairs(state, candidate, metadata["rules"], prompt_variant="concise_v1")
            if not all(fits_context(p, tokenizer, 2048) for p in pairs):
                reason = "exceeds 2048-token training context"
            eligible = reason is None
            if eligible and kept >= retained_per_state:
                reason = "eligible but exceeds fixed retained-count limit"
            accepted = reason is None
            audit["candidates"].append({"candidate_id": index, "eligible": eligible,
                                        "admitted_to_sft": accepted, "reason": reason})
            if accepted:
                data.extend(pairs)
                kept += 1
                for pair in pairs:
                    messages = [{"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]}
                                for c in pair["conversations"]]
                    token_lengths.append(len(tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=True)))
        audit["retained"] = kept
        audits.append(audit)
    if not data:
        raise ValueError("No complete, valid coldstart examples fit the training context")
    write_json(directory / "data.json", data)
    write_json(directory / "dataset_info.json", {
        "leact_local": {"file_name": "data.json", "formatting": "sharegpt", "columns": {"messages": "conversations"},
                        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}})
    write_jsonl(directory / "coldstart-admission.jsonl", audits)
    manifest = {"method": "coldstart_joint", "corpus_spec": spec, "sft_examples": len(data),
                "selected_traces": len(data)//2, "infosets": len(states),
                "states_with_selected_traces": sum(a["retained"] > 0 for a in audits),
                "retained_count_histogram": {str(k): sum(a["retained"] == k for a in audits) for k in range(retained_per_state+1)},
                "oracle_sha256": metadata["sha256"], "expert_forcing": True,
                "context_length_checked": True, "cutoff_len": 2048,
                "prompt_variant": "concise_v1", "data_sha256": file_sha256(directory / "data.json"),
                "training_tokens_per_epoch_including_masked_prompts": sum(token_lengths),
                "maximum_example_tokens": max(token_lengths),
                "selection": "Fixed first eligible candidates; no CFR-agreement or delta filter"}
    write_json(directory / "manifest.json", manifest)
    return manifest


def train_and_evaluate(args, label, training, base, states, metadata, *, lr, all_epochs):
    manifest = json.loads((training / "manifest.json").read_text())
    phase(args.root, "training", arm=label, learning_rate=lr,
          dataset={k: manifest.get(k) for k in ("sft_examples", "selected_traces", "infosets",
                   "states_with_selected_traces", "cutoff_len", "data_sha256")})
    final = scratch_training(training, args.scratch / label, base, gpus=8, epochs=3, lr=lr)
    verification = training / "training_verification.json"
    if not verification.exists():
        examples = json.loads((training / "manifest.json").read_text())["sft_examples"]
        write_json(verification, verify_training_update(base, final, math.ceil(examples/32)*3))
    checkpoints = epoch_checkpoints(final, expected_epochs=3, require_weights=False) if all_epochs else [final]
    evaluations = []
    # Evaluate the fixed final epoch first for early, selection-free comparisons.
    for checkpoint in reversed(checkpoints):
        phase(args.root, "evaluation", arm=label, checkpoint=str(checkpoint), inference_seeds=[0, 1, 2])
        evaluations.append(evaluate_checkpoint(args.oracle, checkpoint, training.parent / "eval" / checkpoint.name,
                           states=states, metadata=metadata, gpus=8, seeds=[0, 1, 2], prompt_variant="concise_v1"))
    best = min(evaluations, key=lambda v: v["oracle_bon"]["exploitability"]["mean"])
    selected = persist_selected(final, Path(best["checkpoint"]), args.scratch)
    return {"best": best, "final_epoch": evaluations[0], "all_epochs": evaluations,
            "selected_checkpoint": str(selected), "selection": json.loads((training / "manifest.json").read_text()),
            "checkpoint_rule": "lowest oracle-BoN mean across epochs" if all_epochs else "fixed epoch 3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["frontier", "self"], required=True)
    for name in ["root", "oracle", "saved-experiment", "scratch"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frontier-input", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds != 2:
        raise ValueError("The registered frontier reproduction runs two LeAct rounds")
    states, metadata = load_oracle(args.oracle)
    if len(states) != 4032 or not metadata["converged"]:
        raise ValueError("The complete converged 4032-state CFR table is required")
    ordered = nested_states(states)
    base = Path(json.loads((args.saved_experiment / "base_model.json").read_text())["path"])
    if MODEL_REVISION not in str(base) or not list(base.glob("*.safetensors")):
        raise ValueError("Pinned Qwen3-8B base is unavailable")
    args.root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    protocol = {"role": args.role, "base_model": str(base), "oracle_sha256": metadata["sha256"],
                "source_sha256": fingerprint({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}),
                "initialization": "untouched Qwen3-8B for every coldstart; no frontier traces in self arm",
                "coldstart_epochs": 3, "coldstart_lr": 1e-4, "effective_batch": 32,
                "cutoff_len": 2048, "generation_tokens": 4096, "prompt_variant": "concise_v1",
                "evaluation_seeds": [0, 1, 2], "n": 8, "m": 4, "k": 2,
                "rounds": 2 if args.role == "frontier" else 0,
                "planned_methods": ["leact"],
                "scaling_cells": [[128, 1], [4032, 1], [4032, 4]] if args.role == "self" else None,
                "scaling_interpretation": "State coverage and retained trace count increase separately. Fixed three epochs; optimizer steps and token counts increase and are logged.",
                "self_candidate_pool": "8 forward candidates per state from untouched base; same fixed pool in every scaling cell",
                "reconstruction_differences": ["Astra public reasoning replaces unavailable author frontier corpus",
                    "6r2s reasoning coldstart size unspecified; full-coverage joint warmstart is an explicit reconstruction",
                    "Reconstructed concise_v1 prompts and H100 hardware",
                    "Self scaling uses fixed final epoch, not best-across-epochs; frontier reproduction evaluates every epoch"]}
    saved = args.root / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != protocol:
        raise ValueError("Study protocol differs; choose a fresh root")
    write_json(saved, protocol)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    report_path = args.root / "results.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "scope": "Leduc 6r2s independent reproduction" if args.role == "frontier" else "Leduc self-generated coldstart scaling",
        "oracle": metadata, "protocol": protocol, "methods": {"leact": []}, "coldstarts": {}}
    if args.role == "frontier":
        if args.frontier_input is None:
            parser.error("frontier requires --frontier-input")
        completion = json.loads((args.frontier_input / "complete.json").read_text())
        rows = read_jsonl(args.frontier_input / "frontier.jsonl")
        if (completion["oracle_sha256"] != metadata["sha256"] or completion["states"] != len(states)
                or completion["rows_sha256"] != fingerprint(rows)
                or {r["state_id"] for r in rows} != {s["state_id"] for s in states} or len(rows) != len(states)):
            raise ValueError("Frontier corpus failed its complete-coverage provenance check")
        candidates = {r["state_id"]: [{**r, "finish_reason": "stop", "fallback": None}] for r in rows}
        cells = [("frontier_4032_n1", ordered, 1)]
    else:
        phase(args.root, "self_generation", infosets=len(states), n=8, checkpoint=str(base))
        paths = parallel_generate(args.oracle, base, args.root / "self_generation", gpus=8, method="exit",
                                  seed=0, n=8, m=4, prompt_variant="concise_v1")
        raw = load_shards(paths, metadata, method="exit")
        by_id = {s["state_id"]: s for s in states}
        candidates = {r["state_id"]: [{**c, "finish_reason": c["forward"]["finish_reason"],
                         "fallback": parse_policy(c["forward"]["text"], by_id[r["state_id"]]["legal_actions"]).fallback}
                                      for c in r["candidates"]] for r in raw}
        cells = [("self_128_n1", ordered[:128], 1), ("self_4032_n1", ordered, 1), ("self_4032_n4", ordered, 4)]
    for label, subset, count in cells:
        training = args.root / label / "sft"
        export_corpus(subset, metadata, candidates, training, tokenizer, retained_per_state=count, source=args.role)
        if label not in report["coldstarts"]:
            report["coldstarts"][label] = train_and_evaluate(args, label, training, base, states, metadata,
                                                            lr=1e-4, all_epochs=args.role == "frontier")
            write_json(report_path, report)
    if args.role == "frontier":
        anchor = report["coldstarts"]["frontier_4032_n1"]
        current = Path(anchor["selected_checkpoint"])
        report["coldstart_thinking"] = anchor["best"]
        report["coldstart_thinking_all_epochs"] = anchor["all_epochs"]
        write_json(args.root / "anchor-ready.json", {"checkpoint": str(current), "oracle_sha256": metadata["sha256"],
                   "frontier_corpus_sha256": completion["rows_sha256"], "measurement": anchor["best"]})
        write_json(report_path, report)
        for index in range(1, 3):
            previous = next((v for v in report["methods"]["leact"] if v["round"] == index), None)
            if previous:
                current = Path(previous["best"]["checkpoint"])
                continue
            directory = args.root / "leact" / f"round-{index}"
            phase(args.root, "leact_generation", round=index, infosets=len(states), checkpoint=str(current))
            paths = parallel_generate(args.oracle, current, directory / "generation", gpus=8, method="leact",
                                      seed=index, n=8, m=4, prompt_variant="concise_v1")
            raw = load_shards(paths, metadata, method="leact")
            training = directory / "sft"
            if not training.exists():
                prepare_data(states, metadata, training, method="leact", raw=raw, tokenizer=tokenizer,
                             prompt_variant="concise_v1", require_valid_policies=True)
            result = train_and_evaluate(args, f"leact-round-{index}", training, current, states, metadata,
                                        lr=5e-6, all_epochs=True)
            current = Path(result["selected_checkpoint"])
            report["methods"]["leact"].append({"round": index, **result})
            write_json(report_path, report)
            write_comparison(report, args.root)
    phase(args.root, "complete", role=args.role)
    write_json(args.root / "complete.json", {"results_sha256": file_sha256(report_path)})


if __name__ == "__main__":
    main()
