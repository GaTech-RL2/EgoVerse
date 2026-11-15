"""One-GPU, restartable generation for the queued Astra comparison."""
import argparse
import json
import os
from pathlib import Path

from ..core import extract_reasoning
from ..data import load_oracle, read_jsonl, write_json
from ..inference import Request, VLLMBackend, sample_seed
from .conditioning import ARMS, assisted_prompt, generate_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--assisted-eval", action="store_true")
    args = parser.parse_args()
    teacher, teacher_meta = load_oracle(args.teacher)
    cfr, cfr_meta = load_oracle(args.oracle)
    if teacher_meta["cfr_evaluation_oracle_sha256"] != cfr_meta["sha256"]:
        raise ValueError("Teacher and evaluator provenance differ")
    if args.assisted_eval:
        hints = {s["state_id"]: s["teacher_decision_summary"] for s in teacher}
        states = [{**s, "teacher_decision_summary": hints[s["state_id"]]} for s in cfr]
        metadata, method = cfr_meta, "eval"
    else:
        states, metadata, method = teacher, teacher_meta, "leact"
    states = states[args.shard::args.shards]
    spec = {"oracle_sha256": metadata["sha256"], "teacher_sha256": teacher_meta["sha256"],
            "method": method, "astra_arm": args.arm, "objective_version": 2,
            "checkpoint": args.checkpoint, "seed": args.seed, "shard": args.shard,
            "shards": args.shards, "state_ids": [s["state_id"] for s in states],
            "n": 8, "m": 4, "prompt_variant": "concise_v1", "max_tokens": 4096,
            "assisted_eval": args.assisted_eval, "temperature": 1., "top_p": .95}
    manifest = args.output.with_suffix(".meta.json")
    if manifest.exists():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError("Generation resume configuration changed")
    elif args.output.exists():
        raise ValueError("Existing generation lacks its manifest")
    else:
        write_json(manifest, spec)
    previous = read_jsonl(args.output) if args.output.exists() else []
    done = {r["state_id"] for r in previous}
    if len(done) != len(previous) or not done.issubset(spec["state_ids"]):
        raise ValueError("Duplicate or foreign generation rows")
    todo = [s for s in states if s["state_id"] not in done]
    if not todo:
        return
    backend = VLLMBackend(args.checkpoint, max_tokens=4096)
    with args.output.open("a") as stream:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            if args.assisted_eval:
                responses = backend.generate([Request(assisted_prompt(s, metadata["rules"]), 8,
                            sample_seed(args.seed, s["state_id"], "forward")) for s in batch])
                rows = [{"state_id": s["state_id"], "seed": args.seed, "prompt_variant": "concise_v1",
                         "prompts": {"forward": assisted_prompt(s, metadata["rules"])},
                         "baselines": [], "teacher_assisted_evaluation": True,
                         "candidates": [{"reasoning": extract_reasoning(g["text"]), "forward": g} for g in gs]}
                        for s, gs in zip(batch, responses)]
            else:
                rows = generate_batch(batch, metadata["rules"], backend, arm=args.arm, seed=args.seed)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({"done": len(done) + start + len(rows), "total": len(states),
                              "arm": args.arm, "shard": args.shard, "assisted_eval": args.assisted_eval}), flush=True)


if __name__ == "__main__":
    main()
