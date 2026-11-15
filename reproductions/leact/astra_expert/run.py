"""Run both queued Astra arms sequentially on one eight-H100 node."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..data import file_sha256, fingerprint, load_oracle, write_json
from ..evaluate import aggregate_evaluations, evaluate_records
from ..experiment import epoch_checkpoints, evaluate_checkpoint, verify_training_update
from ..pipeline import load_shards
from ..reproduction import phase, persist_selected, scratch_training
from .conditioning import ARMS, admitted_pool, prepare_matched
from .data import leduc_gate, solver_gate


def generate(args, checkpoint, directory, arm, seed, *, assisted_eval=False):
    directory.mkdir(parents=True, exist_ok=True)
    processes, streams, paths = [], [], []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.gpus)))).split(",")
    if len(visible) < args.gpus:
        raise RuntimeError("Insufficient visible GPUs")
    try:
        for i in range(args.gpus):
            path = directory / f"shard-{i:02d}.jsonl"
            paths.append(path)
            command = [sys.executable, "-m", "reproductions.leact.astra_expert.worker",
                       "--teacher", str(args.teacher), "--oracle", str(args.oracle),
                       "--checkpoint", str(checkpoint), "--output", str(path), "--arm", arm,
                       "--seed", str(seed), "--shard", str(i), "--shards", str(args.gpus)]
            if assisted_eval:
                command.append("--assisted-eval")
            log = directory / f"gpu-{i:02d}.log"
            write_json(log.with_suffix(".command.json"), command)
            stream = log.open("a")
            streams.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible[i], OMP_NUM_THREADS="2"), start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"Astra generation failed; inspect {directory}")
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"Astra generation failed; inspect {directory}")
    finally:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "teacher", "oracle", "anchor", "scratch"]:
        parser.add_argument("--" + name, type=Path, required=True)
    gate_input = parser.add_mutually_exclusive_group(required=True)
    gate_input.add_argument("--leduc-results", type=Path)
    gate_input.add_argument("--quality-gate", type=Path)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.gpus != 8 or args.rounds not in (1, 2, 3):
        raise ValueError("Expected eight GPUs and one to three registered rounds")
    teacher, teacher_meta = load_oracle(args.teacher)
    cfr, cfr_meta = load_oracle(args.oracle)
    gate = (solver_gate(teacher_meta, json.loads(args.quality_gate.read_text()), cfr_meta["sha256"])
            if args.quality_gate else leduc_gate(json.loads(args.leduc_results.read_text()), cfr_meta["sha256"]))
    if args.anchor.is_file():
        ready = json.loads(args.anchor.read_text())
        if ready["oracle_sha256"] != cfr_meta["sha256"]:
            raise ValueError("Shared coldstart uses a different CFR reference")
        args.anchor = Path(ready["checkpoint"])
    if len(teacher) != 4032 or not teacher_meta["includes_public_summaries"] or teacher_meta["cfr_evaluation_oracle_sha256"] != cfr_meta["sha256"]:
        raise ValueError("Expected the complete, frozen explanation teacher and unchanged CFR evaluator")
    args.root.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    source = {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
    protocol = {"teacher_sha256": teacher_meta["sha256"], "oracle_sha256": cfr_meta["sha256"],
                "source_sha256": fingerprint(source), "gate": gate, "anchor": str(args.anchor),
                "arms": ARMS, "rounds": args.rounds, "epochs": 3, "learning_rate": 5e-6, "effective_batch": 32,
                "context_tokens": 2048, "generation_tokens": 4096, "n": 8, "m": 4, "k": 2,
                "seeds": [0, 1, 2], "compute_matching": "equal retained trace counts and three epochs per arm, per round",
                "r_mixture": "one backward plus one forward example per trace; forward r supplied on a score-independent balanced half",
                "assumption": "Astra is a suitable teacher; its measured CFR exploitability is reported without filtering its labels"}
    protocol_path = args.root / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Queued experiment protocol changed")
    write_json(protocol_path, protocol)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.anchor, local_files_only=True, trust_remote_code=False)
    report_path = args.root / "results.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "scope": "Astra three-task extension; separate from the original reproduction", "protocol": protocol,
        "teacher_quality": teacher_meta["quality"], "arms": {a: [] for a in ARMS}}
    current = {a: args.anchor for a in ARMS}
    for round_index in range(1, args.rounds + 1):
        dirs = {a: args.root / a / f"round-{round_index}" for a in ARMS}
        training = {a: dirs[a] / "sft" for a in ARMS}
        if not all((training[a] / "manifest.json").exists() for a in ARMS):
            if any(training[a].exists() for a in ARMS):
                raise RuntimeError("Partial matched data preparation requires inspection")
            pools, audits = {}, {}
            for arm in ARMS:
                phase(args.root, "astra_generation", arm=arm, round=round_index)
                paths = generate(args, current[arm], dirs[arm] / "generation", arm, round_index)
                raw = load_shards(paths, teacher_meta, method="leact")
                pools[arm], audits[arm] = admitted_pool(teacher, teacher_meta, raw, tokenizer, arm=arm)
            prepare_matched(pools, audits, teacher_meta, training, round_index=round_index)
        for arm in ARMS:
            cached = next((r for r in report["arms"][arm] if r["round"] == round_index), None)
            if cached:
                current[arm] = Path(cached["best"]["checkpoint"])
                if not list(current[arm].glob("*.safetensors")):
                    raise RuntimeError("Retained checkpoint is missing")
                continue
            phase(args.root, "astra_sft", arm=arm, round=round_index)
            final = scratch_training(training[arm], args.scratch / f"{arm}-{round_index}", current[arm],
                                     gpus=8, epochs=3, lr=5e-6)
            manifest = json.loads((training[arm] / "manifest.json").read_text())
            verification = training[arm] / "training_verification.json"
            if not verification.exists():
                write_json(verification, verify_training_update(current[arm], final, math.ceil(manifest["sft_examples"] / 32) * 3))
            evaluations = []
            for epoch in epoch_checkpoints(final, expected_epochs=3, require_weights=False):
                phase(args.root, "astra_standalone_evaluation", arm=arm, round=round_index, checkpoint=str(epoch))
                evaluations.append(evaluate_checkpoint(args.oracle, epoch, dirs[arm] / "eval" / epoch.name,
                    states=cfr, metadata=cfr_meta, gpus=8, seeds=[0, 1, 2], prompt_variant="concise_v1"))
            best = min(evaluations, key=lambda v: v["oracle_bon"]["exploitability"]["mean"])
            current[arm] = persist_selected(final, Path(best["checkpoint"]), args.scratch)
            result = {"round": round_index, "best": best, "all_epochs": evaluations, "selection": manifest}
            if arm == "astra_conditioning_v2":
                phase(args.root, "astra_teacher_assisted_evaluation", arm=arm, round=round_index)
                assisted = []
                for seed in [0, 1, 2]:
                    directory = dirs[arm] / "assisted_eval" / f"seed-{seed}"
                    saved = directory / "result.json"
                    if not saved.exists():
                        paths = generate(args, current[arm], directory, arm, seed, assisted_eval=True)
                        write_json(saved, evaluate_records(cfr, cfr_meta, load_shards(paths, cfr_meta, method="eval")))
                    assisted.append(json.loads(saved.read_text()))
                result["teacher_assisted"] = aggregate_evaluations(assisted)
                write_json(dirs[arm] / "assisted_eval/summary.json", result["teacher_assisted"])
            report["arms"][arm].append(result)
            write_json(report_path, report)
    phase(args.root, "complete", rounds=args.rounds)
    write_json(args.root / "complete.json", {"results_sha256": file_sha256(report_path)})


if __name__ == "__main__":
    main()
