"""Train the admitted pilot, then run full-tree LeAct/ExIt training and evaluation.

Two independent OSMO roles share one explicitly reconstructed reasoning anchor.
Weights for all epochs live on task storage until evaluated; selected weights,
every metric, training record, and raw rollout persist on Lustre.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

from .data import file_sha256, load_oracle, write_json
from .experiment import (MODEL_REVISION, audit_source, check_coldstart_generation,
                         epoch_checkpoints, evaluate_bc_epochs, evaluate_checkpoint,
                         parallel_generate, train, verify_training_update)
from .pipeline import load_shards
from .training import prepare_data
from .compare import write_comparison

PILOT_SHA256 = "01438380affed59a3d174dd50eef291e6934a7af7c9b37f6d0444b27052a91c3"


def phase(root, name, **details):
    value = {"phase": name, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}
    write_json(root / "progress.json", value)
    print("LEACT_PROGRESS " + json.dumps(value), flush=True)


def scratch_training(directory, scratch, checkpoint, *, gpus, epochs, lr):
    directory.mkdir(parents=True, exist_ok=True)
    models = directory / "checkpoints"
    if not models.exists() and not models.is_symlink():
        if scratch.exists():
            raise RuntimeError(f"Unowned scratch training directory exists: {scratch}")
        if shutil.disk_usage(scratch.parent).free < 75 * 1024**3:
            raise RuntimeError("Insufficient task storage for three epoch checkpoints")
        scratch.mkdir()
        models.symlink_to(scratch, target_is_directory=True)
    return train(directory, str(checkpoint), gpus=gpus, epochs=epochs, lr=lr)


def persist_selected(models, selected, scratch_root):
    """Replace our scratch symlink with persistent metadata and selected weights."""
    if not models.is_symlink():
        if not list(selected.glob("*.safetensors")):
            raise ValueError("Previously retained checkpoint is incomplete")
        return selected
    scratch = models.resolve()
    if not scratch.is_relative_to(scratch_root.resolve()):
        raise ValueError("Checkpoint scratch path is outside this task's owned directory")
    relative = selected.relative_to(models)
    temporary = models.with_name("checkpoints-persisting")
    if temporary.exists():
        raise RuntimeError("Incomplete checkpoint copy needs inspection")
    files = [p for p in scratch.rglob("*") if p.is_file() and not p.is_symlink()
             and (p.suffix != ".safetensors" or p.parent == selected.resolve())]
    if shutil.disk_usage(models.parent).free < sum(p.stat().st_size for p in files) + 2*1024**3:
        raise RuntimeError("Insufficient persistent storage for selected checkpoint")
    hashes = {}
    for source in files:
        target = temporary / source.relative_to(scratch)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source.suffix == ".safetensors":
            digest = file_sha256(source)
            if file_sha256(target) != digest:
                raise ValueError("Copied checkpoint checksum mismatch")
            hashes[str(source.relative_to(scratch))] = digest
    if not hashes:
        raise ValueError("No selected checkpoint weights were copied")
    write_json(temporary / "weight_retention.json", {
        "retained_checkpoint": str(models / relative), "weight_sha256": hashes,
        "all_epoch_metrics_retained": True, "scratch_source": str(scratch),
    })
    models.unlink()  # Only the symlink created by scratch_training.
    temporary.rename(models)
    shutil.rmtree(scratch)  # Task-owned transient copies, after verified persistence.
    return models / relative


def combine_completed_roles(root):
    """Publish the benchmark comparison once both independently run roles finish."""
    if not all((root / role / "complete.json").exists() for role in ["leact", "exit"]):
        return
    reports = {role: json.loads((root / role / "results.json").read_text())
               for role in ["leact", "exit"]}
    report = dict(reports["leact"])
    report["protocol"] = {**report["protocol"], "role": "combined"}
    report["methods"] = {role: reports[role]["methods"][role] for role in reports}
    baseline = json.loads((root / "exit/bc/results.json").read_text())
    for key in ["coldstart_bc", "coldstart_bc_all_epochs"]:
        report[key] = baseline[key]
    write_json(root / "results.json", report)
    write_comparison(report, root)


def common_anchor(args, base, tokenizer, states, metadata):
    root = args.root / "anchor"
    ready = root / "ready.json"
    if args.role == "exit":
        deadline = time.monotonic() + 3600
        while not ready.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Common reasoning anchor was not published within one hour")
            time.sleep(10)
        result = json.loads(ready.read_text())
        if result["dataset_sha256"] != PILOT_SHA256:
            raise ValueError("Shared reasoning dataset differs")
        return Path(result["checkpoint"]), result
    if ready.exists():
        result = json.loads(ready.read_text())
        if result["dataset_sha256"] != PILOT_SHA256:
            raise ValueError("Existing anchor differs from admitted pilot")
        return Path(result["checkpoint"]), result
    root.mkdir(parents=True, exist_ok=True)
    source = args.prompt_root / "base/concise_v1/sft-2048"
    if file_sha256(source / "data.json") != PILOT_SHA256:
        raise ValueError("Admitted pilot dataset checksum changed")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["sft_examples"] != 38 or manifest["selected_traces"] != 19 or manifest["cutoff_len"] != 2048:
        raise ValueError("Wrong admitted pilot dataset")
    training = root / "sft"
    if not training.exists():
        shutil.copytree(source, training)
    phase(root, "training_admitted_reasoning", examples=38, epochs=3, learning_rate=5e-6)
    checkpoint = scratch_training(training, args.scratch / "anchor", base,
                                  gpus=args.gpus, epochs=3, lr=5e-6)
    steps = math.ceil(math.ceil(38/args.gpus)/4) * 3
    update = verify_training_update(base, checkpoint, steps)
    write_json(root / "training_verification.json", update)
    checkpoint = persist_selected(checkpoint, checkpoint, args.scratch)
    provenance = {
        "checkpoint": str(checkpoint), "variant": "joint", "origin": "independent_reconstruction",
        "source": str(source), "dataset_sha256": PILOT_SHA256,
        "protocol_note": "Shared anchor trained from the 38 user-approved admitted forward/backward examples generated by base Qwen3-8B with concise_v1. This is a reconstructed pilot anchor, not the unavailable author frontier-generated coldstart corpus.",
        "epochs": 3, "learning_rate": 5e-6, "global_step": update["global_step"],
        "training_parameter_update": update,
    }
    write_json(ready, provenance)
    phase(root, "anchor_ready", checkpoint=str(checkpoint))
    return checkpoint, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=["leact", "exit"], required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--saved-experiment", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.scratch.mkdir(parents=True, exist_ok=True)
    root = args.root / args.role
    root.mkdir(parents=True, exist_ok=True)
    states, metadata = load_oracle(args.oracle)
    if len(states) != 4032 or not metadata["converged"]:
        raise ValueError("Expected the converged full 4032-state Leduc6r2s oracle")
    base = Path(json.loads((args.saved_experiment / "base_model.json").read_text())["path"])
    if MODEL_REVISION not in str(base) or not list(base.glob("*.safetensors")):
        raise ValueError("Pinned base checkpoint is unavailable")
    protocol = {"role": args.role, "rounds": args.rounds, "epochs": args.epochs,
                "gpus": args.gpus, "batch_size": args.batch_size, "n": 8, "m": 4, "k": 2,
                "seed": 0, "evaluation_seeds": [0, 1, 2], "prompt_variant": "concise_v1",
                "learning_rate": 5e-6, "sft_context": 2048, "generation_tokens": 4096,
                "pilot_dataset_sha256": PILOT_SHA256, "require_valid_policies": True,
                "checkpoint_selection": "all three epochs evaluated under three inference seeds; next round uses lowest mean oracle-BoN exploitability",
                "reconstruction_differences": ["38-example self-generated shared reasoning anchor replaces unavailable author coldstart", "reconstructed concise prompts", "strict exclusion of invalid baseline/policy traces", "H100 hardware; effective batch 32", "maximum 3 rounds; no ambiguous Appendix C stopping inequality assumed"]}
    audit_source(root, "full", protocol, metadata["sha256"])
    write_json(root / "base_model.json", {"path": str(base), "revision": MODEL_REVISION})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if args.role == "exit":
        phase(root, "corrected_bc_epoch_evaluation")
        bc = args.saved_experiment / "coldstart/checkpoints"
        evaluate_bc_epochs(args.oracle, bc, root / "bc", states=states, metadata=metadata,
                           gpus=args.gpus, seeds=[0, 1, 2])
    anchor, provenance = common_anchor(args, base, tokenizer, states, metadata)
    write_json(root / "reasoning_anchor.json", provenance)
    report = {"scope": "Leduc 6r2s independent reproduction",
              "initialization_note": "Full-tree training with an explicitly reconstructed shared reasoning coldstart",
              "oracle": metadata, "protocol": protocol, "reasoning_anchor": provenance,
              "methods": {args.role: []}}
    write_json(root / "results.json", report)
    if args.role == "leact":
        phase(root, "post_sft_reasoning_check")
        try:
            report["coldstart_generation_check"] = check_coldstart_generation(
                args.oracle, anchor, root / "coldstart_check", states=states, metadata=metadata,
                gpus=args.gpus, prompt_variant="concise_v1")
        except RuntimeError:
            diagnostic = root / "coldstart_check/result.json"
            if not diagnostic.exists() or json.loads(diagnostic.read_text())["usable_candidates"] != 0:
                raise
            report["coldstart_generation_check"] = json.loads(diagnostic.read_text())
            report["diagnostic_note"] = "A 16-state format diagnostic does not replace full-tree evaluation or generation."
        phase(root, "shared_anchor_full_tree_evaluation")
        report["coldstart_thinking"] = evaluate_checkpoint(
            args.oracle, anchor, root / "eval/coldstart_thinking", states=states, metadata=metadata,
            gpus=args.gpus, seeds=[0, 1, 2], prompt_variant="concise_v1", batch_size=args.batch_size)
        write_json(root / "results.json", report)
        write_comparison(report, root)
    current = anchor
    for round_index in range(1, args.rounds + 1):
        directory = root / args.role / f"round-{round_index}"
        phase(root, "full_tree_generation", method=args.role, round=round_index, infosets=len(states), checkpoint=str(current))
        paths = parallel_generate(args.oracle, current, directory / "generation", gpus=args.gpus,
                                  method=args.role, seed=round_index, n=8, m=4,
                                  prompt_variant="concise_v1", batch_size=args.batch_size)
        raw = load_shards(paths, metadata, method=args.role)
        training = directory / "sft"
        if not training.exists():
            prepare_data(states, metadata, training, method=args.role, raw=raw, tokenizer=tokenizer,
                         prompt_variant="concise_v1", require_valid_policies=True)
        phase(root, "reasoning_sft", method=args.role, round=round_index,
              dataset=json.loads((training / "manifest.json").read_text()))
        final = scratch_training(training, args.scratch / f"{args.role}-round-{round_index}", current,
                                 gpus=args.gpus, epochs=args.epochs, lr=5e-6)
        verification = training / "training_verification.json"
        if not verification.exists():
            examples = json.loads((training / "manifest.json").read_text())["sft_examples"]
            expected_steps = math.ceil(math.ceil(examples / args.gpus) / 4) * args.epochs
            write_json(verification, verify_training_update(current, final, expected_steps))
        epochs = epoch_checkpoints(final, expected_epochs=args.epochs, require_weights=False)
        evaluations = []
        for epoch in epochs:
            phase(root, "full_tree_epoch_evaluation", method=args.role, round=round_index, checkpoint=str(epoch))
            evaluations.append(evaluate_checkpoint(
                args.oracle, epoch, directory / "eval" / epoch.name, states=states, metadata=metadata,
                gpus=args.gpus, seeds=[0, 1, 2], prompt_variant="concise_v1", batch_size=args.batch_size))
        best = min(evaluations, key=lambda r: r["oracle_bon"]["exploitability"]["mean"])
        current = persist_selected(final, Path(best["checkpoint"]), args.scratch)
        report["methods"][args.role].append({"round": round_index, "best": best, "all_epochs": evaluations,
                                              "selection": json.loads((training / "manifest.json").read_text())})
        shared_eval = args.root / "leact/eval/coldstart_thinking/summary.json"
        if shared_eval.exists():
            report["coldstart_thinking"] = json.loads(shared_eval.read_text())
        write_json(root / "results.json", report)
        write_comparison(report, root)
    phase(root, "complete", method=args.role, rounds=args.rounds)
    write_json(root / "complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                         "results_sha256": file_sha256(root / "results.json")})
    combine_completed_roles(args.root)


if __name__ == "__main__":
    main()
