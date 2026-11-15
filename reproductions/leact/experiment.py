"""Bounded single-node experiment driver. Every training run updates all weights."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .data import fingerprint, load_oracle, write_json
from .compare import write_comparison
from .evaluate import aggregate_evaluations, evaluate_records
from .pipeline import load_shards, score_record
from .training import prepare_data, write_recipe

MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def run_logged(command: list[str], path: Path, env: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.with_suffix(".command.json"), command)
    print(f"Running {' '.join(command)}; log={path}", flush=True)
    with path.open("a") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if process.returncode:
        print(path.read_text()[-14000:], flush=True)
        raise RuntimeError(f"Command exited {process.returncode}; see {path}")


def train(directory: Path, checkpoint: str, *, gpus: int, epochs: int, lr: float, max_steps=-1) -> Path:
    marker = directory / "training_complete.json"
    spec = {"checkpoint": checkpoint, "gpus": gpus, "epochs": epochs, "lr": lr,
            "max_steps": max_steps, "data": json.loads((directory / "manifest.json").read_text())["data_sha256"]}
    if marker.exists():
        if json.loads(marker.read_text()) != spec:
            raise ValueError("Completed training phase has different parameters")
        return directory / "checkpoints"
    model_dir = directory / "checkpoints"
    if model_dir.exists() and any(model_dir.iterdir()):
        raise RuntimeError(f"Unfinished training exists in {model_dir}; inspect it before restarting. "
                           "Model-only epoch checkpoints do not promise exact optimizer resume.")
    recipe = write_recipe(directory, checkpoint, gpus=gpus, epochs=epochs, learning_rate=lr, max_steps=max_steps)
    env = dict(os.environ, FORCE_TORCHRUN="1", NPROC_PER_NODE=str(gpus), NNODES="1", NODE_RANK="0",
               MASTER_ADDR="127.0.0.1", MASTER_PORT="29631", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="4", WANDB_DISABLED="true")
    run_logged(["llamafactory-cli", "train", str(recipe)], directory / "train.log", env)
    if not (model_dir / "config.json").exists() or not list(model_dir.glob("*.safetensors")):
        raise RuntimeError("Trainer returned without a saved model")
    import math
    history = json.loads((model_dir / "trainer_state.json").read_text())
    losses = [row["loss"] for row in history["log_history"] if "loss" in row]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Training did not produce finite logged losses")
    write_json(marker, spec)
    return model_dir


def retain_selected_weights(model_dir: Path, selected: Path) -> None:
    """Keep epoch metrics/configuration, but retain only selected model weights."""
    selected = selected.resolve()
    if selected != model_dir.resolve() and selected.parent != model_dir.resolve():
        raise ValueError("Selected checkpoint must belong to this training phase")
    if not list(selected.glob("*.safetensors")):
        raise ValueError("Selected checkpoint has no saved weights")
    removed = []
    for directory in [model_dir, *model_dir.glob("checkpoint-*")]:
        if directory.is_symlink() or (directory.resolve() != model_dir.resolve()
                                     and directory.resolve().parent != model_dir.resolve()):
            raise ValueError("Refusing checkpoint retention through a symlink or external path")
        if directory.resolve() != selected:
            removed.extend(directory.glob("*.safetensors"))
    write_json(model_dir / "weight_retention.json", {
        "retained_checkpoint": str(selected),
        "removed_weight_files": [str(p) for p in removed],
        "all_evaluation_metrics_retained": True,
    })
    for path in removed:
        path.unlink()


def parallel_generate(oracle: Path, checkpoint: Path, directory: Path, *, gpus: int,
                      method: str, seed: int, n=8, m=4, limit=None, thinking=True,
                      eval_seeds: list[int] | None = None, prompt_variant="original",
                      batch_size=32) -> list[Path]:
    if eval_seeds is not None and (method != "eval" or not eval_seeds or len(set(eval_seeds)) != len(eval_seeds)):
        raise ValueError("Multi-seed generation requires evaluation with distinct seeds")
    directory.mkdir(parents=True, exist_ok=True)
    commands, paths = [], []
    for index in range(gpus):
        path = directory / f"shard-{index:02d}.jsonl"
        command = [sys.executable, "-m", "reproductions.leact", "generate", "--oracle", str(oracle),
                   "--checkpoint", str(checkpoint), "--output", str(path), "--method", method,
                   "--seed", str(seed), "--shard", str(index), "--shards", str(gpus),
                   "--n", str(n), "--m", str(m), "--batch-size", str(batch_size),
                   "--prompt-variant", prompt_variant]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        if not thinking:
            command.append("--no-thinking")
        if eval_seeds is not None:
            command.extend(["--eval-seeds", *map(str, eval_seeds)])
        commands.append(command)
        paths.extend([directory / f"seed-{s}" / path.name for s in eval_seeds]
                     if eval_seeds is not None else [path])
    processes, streams = [], []
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(gpus)))).split(",")
        if len(visible) < gpus:
            raise RuntimeError("Requested more GPUs than visible")
        for i, command in enumerate(commands):
            log = directory / f"gpu-{i:02d}.log"
            stream = log.open("a")
            streams.append(stream)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[i], VLLM_WORKER_MULTIPROC_METHOD="spawn",
                       OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
            write_json(log.with_suffix(".command.json"), command)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                               start_new_session=True))
        while any(p.poll() is None for p in processes):
            failed = [i for i, p in enumerate(processes) if p.poll() not in {None, 0}]
            if failed:
                for i in failed:
                    print((directory / f"gpu-{i:02d}.log").read_text()[-10000:], flush=True)
                raise RuntimeError(f"Generation workers failed: {failed}")
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"Generation failed; inspect {directory}")
    finally:
        import signal
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for stream in streams:
            stream.close()
    return paths


def audit_source(root: Path, phase: str, args: dict, oracle_sha256: str) -> None:
    source = {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    spec = {"model": MODEL, "revision": MODEL_REVISION,
            "source_sha256": fingerprint(source), "arguments": args,
            "oracle_sha256": oracle_sha256}
    manifest = root / f"{phase}_protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError("Existing experiment protocol differs; use a fresh run root")
    write_json(manifest, spec)


def verify_training_update(base: Path, checkpoint: Path, expected_steps: int) -> dict:
    """Check a real parameter changed and the trainer produced finite losses."""
    import math
    import torch
    from safetensors import safe_open

    key = "model.layers.0.self_attn.q_proj.weight"

    def weight_slice(directory):
        index = directory / "model.safetensors.index.json"
        filename = json.loads(index.read_text())["weight_map"][key] if index.exists() else "model.safetensors"
        with safe_open(directory / filename, framework="pt", device="cpu") as tensors:
            return tensors.get_slice(key)[:128, :512].float()

    before, after = weight_slice(base), weight_slice(checkpoint)
    if not torch.isfinite(after).all():
        raise RuntimeError("Saved model contains non-finite weights")
    delta = after - before
    if not torch.any(delta != 0):
        raise RuntimeError("Preflight did not measurably change the sampled attention weights")
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    losses = [entry["loss"] for entry in trainer["log_history"] if "loss" in entry]
    if trainer["global_step"] != expected_steps or not losses or not all(math.isfinite(x) for x in losses):
        raise RuntimeError("Unexpected optimizer step count or non-finite training loss")
    return {"global_step": trainer["global_step"], "losses": losses, "sampled_parameter": key,
            "sampled_weight_max_abs_change": float(delta.abs().max()),
            "sampled_weight_changed_fraction": float((delta != 0).float().mean())}


def evaluate_checkpoint(oracle: Path, checkpoint: Path, directory: Path, *, states: list,
                        metadata: dict, gpus: int, seeds: list[int], thinking=True,
                        prompt_variant="original", batch_size=32) -> dict:
    results = []
    pending = [seed for seed in seeds if not (directory / f"seed-{seed}.json").exists()]
    if pending:
        parallel_generate(oracle, checkpoint, directory, gpus=gpus, method="eval", seed=0,
                          thinking=thinking, eval_seeds=pending, prompt_variant=prompt_variant,
                          batch_size=batch_size)
    for seed in seeds:
        result_path = directory / f"seed-{seed}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result["seed"] != seed or result["oracle_sha256"] != metadata["sha256"]
                    or result["infosets"] != len(states) or result["n"] != 8):
                raise ValueError("Cached evaluation does not match the requested oracle or sampling protocol")
        else:
            paths = [directory / f"seed-{seed}" / f"shard-{index:02d}.jsonl" for index in range(gpus)]
            result = evaluate_records(states, metadata, load_shards(paths, metadata, method="eval"))
            write_json(result_path, result)
        results.append(result)
    summary = aggregate_evaluations(results)
    summary["checkpoint"] = str(checkpoint)
    write_json(directory / "summary.json", summary)
    return summary


def check_coldstart_generation(oracle: Path, checkpoint: Path, directory: Path, *,
                              states: list, metadata: dict, gpus: int,
                              prompt_variant="original") -> dict:
    """Bound generation before spending a full round on an unusable trace format."""
    import statistics

    subset = states[:16]
    paths = parallel_generate(oracle, checkpoint, directory / "generation", gpus=gpus,
                              method="leact", seed=0, n=2, m=2, limit=len(subset),
                              prompt_variant=prompt_variant, batch_size=4)
    rows = load_shards(paths, metadata, method="leact")
    by_id = {s["state_id"]: s for s in subset}
    scored = [score_record(by_id[r["state_id"]], r, method="leact", k=2) for r in rows]
    candidates = [candidate for row in scored for candidate in row["candidates"]]
    usable = [c for c in candidates if c["complete"] and c["reasoning"].strip() and c["fallback"] is None]
    result = {
        "states": len(rows), "candidates": len(candidates), "usable_candidates": len(usable),
        "positive_selected": sum(len(r["selected"]) for r in scored),
        "truncated_candidates": sum(not c["complete"] for c in candidates),
        "empty_reasoning_candidates": sum(not c["reasoning"].strip() for c in candidates),
        "forward_policy_fallbacks": sum(c["fallback"] is not None for c in candidates),
        "mean_backward_generated_tokens": statistics.mean(
            c["backward"]["generated_tokens"] for r in rows for c in r["candidates"]),
        "mean_conditioned_generated_tokens": statistics.mean(
            c["forward"]["generated_tokens"] for r in rows for c in r["candidates"]),
        "scope": "format/termination diagnostic; not full-tree benchmark evaluation",
    }
    write_json(directory / "result.json", result)
    print(json.dumps({"coldstart_generation_check": result}, indent=2), flush=True)
    if not usable:
        raise RuntimeError(f"Coldstart produced no complete, nonempty reasoning with a parsed policy; inspect {directory}")
    return result


def epoch_checkpoints(model_dir: Path, *, expected_epochs: int | None = None,
                      require_weights: bool = True) -> list[Path]:
    """Require saved epochs for best-across-epochs; never silently use only final."""
    paths = sorted((p for p in model_dir.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
                   key=lambda p: int(p.name.split("-")[-1]))
    if expected_epochs is not None and len(paths) != expected_epochs:
        raise ValueError(f"Expected {expected_epochs} saved epoch checkpoints, found {len(paths)}")
    if not paths or any(not (p / "config.json").is_file()
                        or (require_weights and not list(p.glob("*.safetensors"))) for p in paths):
        raise ValueError("Best-across-epochs requires all saved epoch weights; final-only evaluation is insufficient")
    return paths


def evaluate_bc_epochs(oracle, checkpoint, root, *, states, metadata, gpus, seeds):
    """BC has no reasoning prerequisite; retain every epoch for separate metrics."""
    evaluations = [evaluate_checkpoint(
        oracle, epoch, root / "eval" / "bc_epochs" / epoch.name,
        states=states, metadata=metadata, gpus=gpus, seeds=seeds, thinking=False,
    ) for epoch in epoch_checkpoints(checkpoint, expected_epochs=3)]
    best = min(evaluations, key=lambda r: r["oracle_bon"]["exploitability"]["mean"])
    report = {
        "scope": "Leduc 6r2s independent reproduction", "oracle": metadata,
        "protocol": {"mode": "bc", "rounds": 0, "seeds": seeds,
                     "checkpoint_selection": "best mean across all saved epochs, separately per metric"},
        "coldstart_bc": best, "coldstart_bc_all_epochs": evaluations, "methods": {},
    }
    write_json(root / "results.json", report)
    write_comparison(report, root)
    return report


def validate_reasoning_anchor(checkpoint: Path | None, provenance: Path | None) -> dict:
    if checkpoint is None or provenance is None:
        raise ValueError("Reasoning mode requires an explicit --reasoning-checkpoint and --reasoning-provenance; BC is not assumed to be a reasoning coldstart")
    record = json.loads(provenance.read_text())
    if record.get("variant") not in {"nocot", "cot", "joint"} or record.get("origin") not in {"author_artifact", "independent_reconstruction"}:
        raise ValueError("Reasoning provenance must identify nocot/cot/joint variant and author_artifact/independent_reconstruction origin")
    if not record.get("source") or not re.fullmatch(r"[a-f0-9]{64}", record.get("dataset_sha256", "")) or not record.get("protocol_note"):
        raise ValueError("Reasoning provenance needs its corpus source, SHA256 checksum and explicit protocol rationale")
    if Path(record.get("checkpoint", "")).resolve() != checkpoint.resolve():
        raise ValueError("Reasoning checkpoint does not match provenance")
    if not (checkpoint / "config.json").is_file() or not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Reasoning checkpoint is missing model configuration or weights")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--phase", choices=["preflight", "full"], default="preflight")
    parser.add_argument("--mode", choices=["bc", "reasoning"], default="bc")
    parser.add_argument("--bc-checkpoint-root", type=Path,
                        help="Evaluate existing BC epoch weights in a fresh report root, without retraining")
    parser.add_argument("--reasoning-checkpoint", type=Path)
    parser.add_argument("--reasoning-provenance", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3, help="M-step epochs in reasoning mode; BC uses Appendix H.3's fixed three epochs")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--oracle", type=Path, default=Path("reproductions/leact/artifacts/leduc6r2s/oracle.jsonl"))
    args = parser.parse_args()
    if args.gpus <= 0 or args.rounds <= 0 or args.epochs <= 0:
        parser.error("GPU, round, and epoch counts must be positive")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error("Evaluation seeds must be distinct")
    reasoning_provenance = None
    if args.phase == "full" and args.mode == "reasoning":
        reasoning_provenance = validate_reasoning_anchor(args.reasoning_checkpoint, args.reasoning_provenance)
        if args.bc_checkpoint_root:
            parser.error("An existing BC checkpoint cannot be used as an implicit reasoning anchor")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    oracle = args.oracle.resolve()
    states, metadata = load_oracle(oracle)
    if not metadata["converged"]:
        raise ValueError("Teacher did not meet the recorded solver tolerance")
    protocol_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if reasoning_provenance is not None:
        protocol_args["reasoning_provenance_sha256"] = fingerprint(reasoning_provenance)
    audit_source(root, args.phase, protocol_args, metadata["sha256"])
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    base = Path(snapshot_download(MODEL, revision=MODEL_REVISION, cache_dir=str(root / "hf-cache")))
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    write_json(root / "base_model.json", {"model": MODEL, "revision": MODEL_REVISION, "path": str(base)})
    if args.phase == "preflight":
        directory = root / "preflight" / "sft"
        subset = states[:16]
        if not directory.exists():
            prepare_data(subset, metadata, directory, method="bc", tokenizer=tokenizer)
        checkpoint = train(directory, str(base), gpus=args.gpus, epochs=1, lr=1e-4, max_steps=2)
        update = verify_training_update(base, checkpoint, 2)
        paths = parallel_generate(oracle, checkpoint, root / "preflight" / "leact", gpus=args.gpus,
                                  method="leact", seed=0, n=2, m=2, limit=len(subset))
        rows = load_shards(paths, metadata, method="leact")
        by_id = {s["state_id"]: s for s in subset}
        scored = [score_record(by_id[r["state_id"]], r, method="leact", k=2) for r in rows]
        result = {"full_parameter_training_steps": 2, "parameter_update": update, "model_reload": True,
                  "states": len(rows), "candidates": sum(len(r["candidates"]) for r in scored),
                  "positive_selected": sum(len(r["selected"]) for r in scored),
                  "truncated_candidates": sum(not c["complete"] for r in scored for c in r["candidates"]),
                  "scope": "infrastructure preflight; no benchmark reproduction claim"}
        write_json(root / "preflight" / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.mode == "bc":
        if args.bc_checkpoint_root:
            cold_checkpoint = args.bc_checkpoint_root.resolve()
            manifest = json.loads((cold_checkpoint.parent / "manifest.json").read_text())
            if manifest.get("method") != "bc" or manifest.get("oracle_sha256") != metadata["sha256"]:
                raise ValueError("Existing BC checkpoint corpus does not match this oracle")
            if not (cold_checkpoint.parent / "training_complete.json").is_file():
                raise ValueError("Existing BC training has no completion record")
        else:
            cold = root / "coldstart"
            if not cold.exists():
                prepare_data(states, metadata, cold, method="bc", tokenizer=tokenizer)
            cold_checkpoint = train(cold, str(base), gpus=args.gpus, epochs=3, lr=1e-4)
        report = evaluate_bc_epochs(oracle, cold_checkpoint, root, states=states,
                                    metadata=metadata, gpus=args.gpus, seeds=args.eval_seeds)
        print(json.dumps(report, indent=2), flush=True)
        return
    cold_checkpoint = args.reasoning_checkpoint.resolve()
    write_json(root / "reasoning_anchor.json", reasoning_provenance)
    diagnostic = check_coldstart_generation(
        oracle, cold_checkpoint, root / "coldstart_check", states=states, metadata=metadata, gpus=args.gpus)
    report = {"scope": "Leduc 6r2s independent reproduction", "oracle": metadata,
              "coldstart_generation_check": diagnostic,
              "reasoning_anchor": reasoning_provenance,
              "protocol": {"mode": "reasoning", "rounds": args.rounds, "epochs": args.epochs, "seeds": args.eval_seeds,
                           "checkpoint_selection": "lowest mean oracle-aware BoN exploitability"},
              "methods": {}}
    write_json(root / "results.json", report)
    report["coldstart_thinking"] = evaluate_checkpoint(
        oracle, cold_checkpoint, root / "eval" / "coldstart_thinking", states=states,
        metadata=metadata, gpus=args.gpus, seeds=args.eval_seeds, thinking=True)
    write_json(root / "results.json", report)
    write_comparison(report, root)
    # The shared reasoning anchor is an explicit external input; keep it intact.
    for method in ["leact", "exit"]:
        current = cold_checkpoint
        report["methods"][method] = []
        for round_index in range(1, args.rounds + 1):
            directory = root / method / f"round-{round_index}"
            paths = parallel_generate(oracle, current, directory / "generation", gpus=args.gpus,
                                      method=method, seed=round_index)
            raw = load_shards(paths, metadata, method=method)
            training = directory / "sft"
            if not training.exists():
                prepare_data(states, metadata, training, method=method, raw=raw, tokenizer=tokenizer)
            final = train(training, str(current), gpus=args.gpus, epochs=args.epochs, lr=5e-6)
            # Resumed rounds may have pruned weights after all per-seed metrics
            # were persisted. evaluate_checkpoint validates and reuses those.
            epochs = epoch_checkpoints(final, expected_epochs=args.epochs, require_weights=False)
            evaluations = [evaluate_checkpoint(oracle, epoch, directory / "eval" / epoch.name,
                                               states=states, metadata=metadata, gpus=args.gpus,
                                               seeds=args.eval_seeds) for epoch in epochs]
            best = min(evaluations, key=lambda r: r["oracle_bon"]["exploitability"]["mean"])
            current = Path(best["checkpoint"])
            report["methods"][method].append({"round": round_index, "best": best, "all_epochs": evaluations,
                                               "selection": json.loads((training / "manifest.json").read_text())})
            write_json(root / "results.json", report)
            write_comparison(report, root)
            retain_selected_weights(final, current)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
