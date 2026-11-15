"""Run from the repository root: python -m reproductions.leact --help."""

import argparse
import json
from pathlib import Path

from .data import load_oracle, write_json
from .prompts import PROMPT_VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle", help="Build and measure a CPU CFR+ teacher")
    oracle.add_argument("--output", type=Path, required=True)
    oracle.add_argument("--ranks", type=int, default=6)
    oracle.add_argument("--suits", type=int, default=2)
    oracle.add_argument("--iterations", type=int, default=2000)
    oracle.add_argument("--tolerance", type=float, default=0.005)
    oracle.add_argument("--check-every", type=int, default=100)
    generate = commands.add_parser("generate", help="Run one resumable vLLM generation shard")
    generate.add_argument("--oracle", type=Path, required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--revision")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--method", choices=["leact", "exit", "eval"], required=True)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--eval-seeds", type=int, nargs="+",
                          help="Reuse one model for evaluation seeds, saving each under seed-N/output-name")
    generate.add_argument("--n", type=int, default=8)
    generate.add_argument("--m", type=int, default=4)
    generate.add_argument("--shard", type=int, default=0)
    generate.add_argument("--shards", type=int, default=1)
    generate.add_argument("--batch-size", type=int, default=8)
    generate.add_argument("--max-tokens", type=int, default=4096)
    generate.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="original")
    generate.add_argument("--tensor-parallel", type=int, default=1)
    generate.add_argument("--no-thinking", action="store_true")
    generate.add_argument("--limit", type=int)
    prepare = commands.add_parser("prepare", help="Score candidates and export audited SFT data")
    prepare.add_argument("--oracle", type=Path, required=True)
    prepare.add_argument("--method", choices=["bc", "leact", "exit"], required=True)
    prepare.add_argument("--rollouts", type=Path, nargs="+")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument("--gpus", type=int, default=8)
    prepare.add_argument("--epochs", type=int, default=3)
    prepare.add_argument("--lr", type=float, default=5e-6)
    prepare.add_argument("--max-steps", type=int, default=-1)
    prepare.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default="original")
    prepare.add_argument("--cutoff", type=int, default=2048)
    prepare.add_argument("--limit", type=int)
    evaluate = commands.add_parser("evaluate", help="Score a complete game tree from saved decodes")
    evaluate.add_argument("--oracle", type=Path, required=True)
    evaluate.add_argument("--rollouts", type=Path, nargs="+", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "oracle":
        from .oracle import build_oracle
        result = build_oracle(args.output, ranks=args.ranks, suits=args.suits, iterations=args.iterations,
                              tolerance=args.tolerance, check_every=args.check_every)
        print(json.dumps(result, indent=2))
        return
    states, metadata = load_oracle(args.oracle)
    if getattr(args, "limit", None) is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        states = states[:args.limit]
    if args.command == "generate":
        from .inference import VLLMBackend
        from .pipeline import generate_shard
        if args.no_thinking and args.method != "eval":
            parser.error("--no-thinking is only valid for BC evaluation")
        if args.eval_seeds is not None and (args.method != "eval" or len(set(args.eval_seeds)) != len(args.eval_seeds)):
            parser.error("--eval-seeds requires eval and distinct seeds")
        backend = VLLMBackend(args.checkpoint, revision=args.revision,
                              tensor_parallel=args.tensor_parallel, max_tokens=args.max_tokens)
        for seed in args.eval_seeds if args.eval_seeds is not None else [args.seed]:
            output = (args.output.parent / f"seed-{seed}" / args.output.name
                      if args.eval_seeds is not None else args.output)
            generate_shard(states, metadata, backend, output, checkpoint=args.checkpoint,
                           model_revision=args.revision, method=args.method, seed=seed,
                           n=args.n, m=args.m, shard=args.shard, shards=args.shards,
                           batch_size=args.batch_size, thinking=not args.no_thinking,
                           max_tokens=args.max_tokens, prompt_variant=args.prompt_variant)
    elif args.command == "prepare":
        from transformers import AutoTokenizer
        from .pipeline import load_shards
        from .training import prepare_data, write_recipe
        raw = load_shards(args.rollouts or [], metadata, method=args.method) if args.method != "bc" else None
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=False)
        summary = prepare_data(states, metadata, args.output, method=args.method, raw=raw, tokenizer=tokenizer,
                               cutoff=args.cutoff, prompt_variant=args.prompt_variant)
        write_recipe(args.output, args.checkpoint, gpus=args.gpus, epochs=args.epochs,
                     learning_rate=args.lr, max_steps=args.max_steps)
        print(json.dumps(summary, indent=2))
    elif args.command == "evaluate":
        from .evaluate import evaluate_records
        from .pipeline import load_shards
        raw = load_shards(args.rollouts, metadata, method="eval")
        result = evaluate_records(states, metadata, raw)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
