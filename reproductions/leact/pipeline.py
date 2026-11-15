"""E-step execution, auditable scoring, and restartable generation shards."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .core import extract_reasoning, forward_delta, kl, parse_policy, select_candidates
from .data import fingerprint, read_jsonl, write_json
from .inference import Request, sample_seed
from .prompts import backward_prompt, forward_prompt


def score_record(state: dict, raw: dict, *, method: str, k: int = 2) -> dict:
    if raw["state_id"] != state["state_id"]:
        raise ValueError("State ID mismatch")
    actions = state["legal_actions"]
    expert = state["expert_policy"]
    baselines = [parse_policy(g["text"], actions) for g in raw["baselines"]]
    if method == "leact" and not baselines:
        raise ValueError("LeAct requires unconditioned forward baselines")
    candidates = []
    for index, item in enumerate(raw["candidates"]):
        parsed = parse_policy(item["forward"]["text"], actions)
        score = (
            forward_delta(expert, parsed.policy, [p.policy for p in baselines])
            if method == "leact" else -kl(expert, parsed.policy)
        )
        candidates.append({
            "candidate_id": index, "reasoning": item["reasoning"],
            "predicted_policy": parsed.policy, "fallback": parsed.fallback,
            "score": score,
            "complete": item["forward"]["finish_reason"] == "stop"
            and item.get("backward", {"finish_reason": "stop"})["finish_reason"] == "stop",
        })
    selected = select_candidates(candidates, k, method)
    return {
        "state_id": state["state_id"], "candidates": candidates, "selected": selected,
        "baseline_fallbacks": [p.fallback for p in baselines],
        "baseline_truncations": sum(g["finish_reason"] != "stop" for g in raw["baselines"]),
    }


def generate_batch(states: list[dict], rules: str, backend, *, method: str,
                   n: int = 8, m: int = 4, seed: int = 0, thinking=True,
                   prompt_variant="original") -> list[dict]:
    if method not in {"leact", "exit", "eval"} or n <= 0 or m <= 0:
        raise ValueError("Invalid generation method or sample counts")
    if method != "leact":
        outputs = backend.generate([
            Request(forward_prompt(s, rules, thinking=thinking, variant=prompt_variant), n,
                    sample_seed(seed, s["state_id"], "forward"), thinking=thinking)
            for s in states
        ])
        return [
            {"state_id": state["state_id"], "seed": seed, "baselines": [],
             "prompt_variant": prompt_variant,
             "prompts": {"forward": forward_prompt(state, rules, thinking=thinking, variant=prompt_variant)},
             "candidates": [{"reasoning": extract_reasoning(g["text"]), "forward": g} for g in samples]}
            for state, samples in zip(states, outputs)
        ]
    backward = backend.generate([
        Request(backward_prompt(s, rules, variant=prompt_variant), n, sample_seed(seed, s["state_id"], "backward"))
        for s in states
    ])
    # The baseline has no expert hint and no backward-generated trace in context.
    baseline = backend.generate([
        Request(forward_prompt(s, rules, variant=prompt_variant), m, sample_seed(seed, s["state_id"], "baseline"))
        for s in states
    ])
    requests = []
    for state, candidates in zip(states, backward):
        for index, generation in enumerate(candidates):
            requests.append(Request(
                forward_prompt(state, rules, variant=prompt_variant), 1,
                sample_seed(seed, state["state_id"], f"conditioned-{index}"),
                reasoning=extract_reasoning(generation["text"]),
            ))
    conditioned = backend.generate(requests)
    records = []
    for index, (state, candidates, baseline_samples) in enumerate(zip(states, backward, baseline)):
        records.append({
            "state_id": state["state_id"], "seed": seed, "baselines": baseline_samples,
            "prompt_variant": prompt_variant,
            "prompts": {"forward": forward_prompt(state, rules, variant=prompt_variant),
                        "backward": backward_prompt(state, rules, variant=prompt_variant)},
            "candidates": [
                {"reasoning": extract_reasoning(g["text"]), "backward": g,
                 "forward": conditioned[index * n + j][0]}
                for j, g in enumerate(candidates)
            ],
        })
    return records


def generate_shard(states: list[dict], metadata: dict, backend, output: Path, *,
                   checkpoint: str, method: str, seed=0, n=8, m=4,
                   shard=0, shards=1, batch_size=8, thinking=True,
                   max_tokens=4096, model_revision=None, prompt_variant="original") -> None:
    if not 0 <= shard < shards or batch_size <= 0:
        raise ValueError("Invalid shard or batch size")
    selected_states = [s for i, s in enumerate(states) if i % shards == shard]
    source_hash = fingerprint({p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")})
    spec = {
        "schema_version": 1, "source_sha256": source_hash,
        "oracle_sha256": metadata["sha256"], "checkpoint": checkpoint,
        "model_revision": model_revision, "method": method, "seed": seed,
        "n": n, "m": m, "shard": shard, "shards": shards,
        "thinking": thinking, "max_tokens": max_tokens,
        "prompt_variant": prompt_variant,
        "temperature": 1.0, "top_p": 0.95,
        "state_ids": [s["state_id"] for s in selected_states],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".meta.json")
    if manifest.exists():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError("Resume configuration differs from the existing generation shard")
    elif output.exists():
        raise ValueError("Generation shard is missing its manifest")
    else:
        write_json(manifest, spec)
    previous = read_jsonl(output) if output.exists() else []
    done = {row["state_id"] for row in previous}
    if len(done) != len(previous) or not done.issubset(spec["state_ids"]):
        raise ValueError("Duplicate or foreign states in generation shard")
    todo = [s for s in selected_states if s["state_id"] not in done]
    with output.open("a") as stream:
        for start in range(0, len(todo), batch_size):
            records = generate_batch(todo[start:start + batch_size], metadata["rules"], backend,
                                     method=method, n=n, m=m, seed=seed, thinking=thinking,
                                     prompt_variant=prompt_variant)
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(f"{method} seed {seed} shard {shard}: {len(done) + start + len(records)}/{len(selected_states)}", flush=True)


def load_shards(paths: list[Path], metadata: dict, *, method: str) -> list[dict]:
    result, common = [], None
    seen_shards = set()
    for path in paths:
        spec = json.loads(path.with_suffix(".meta.json").read_text())
        if spec["oracle_sha256"] != metadata["sha256"] or spec["method"] != method:
            raise ValueError("Wrong oracle or generation method in shard")
        comparable = {k: v for k, v in spec.items() if k not in {"state_ids", "shard"}}
        if common is not None and common != comparable:
            raise ValueError("Cannot combine generation shards from different runs")
        common = comparable
        if spec["shard"] in seen_shards:
            raise ValueError("Repeated shard")
        seen_shards.add(spec["shard"])
        rows = read_jsonl(path)
        if {r["state_id"] for r in rows} != set(spec["state_ids"]):
            raise ValueError("Incomplete generation shard")
        for row in rows:
            if row.get("prompt_variant", "original") != spec.get("prompt_variant", "original"):
                raise ValueError("Generation prompt variant differs from manifest")
            if row["seed"] != spec["seed"] or len(row["candidates"]) != spec["n"]:
                raise ValueError("Generation sample count or seed mismatch")
            if method == "leact" and len(row["baselines"]) != spec["m"]:
                raise ValueError("Baseline sample count mismatch")
        result.extend(rows)
    if common is None or seen_shards != set(range(common["shards"])):
        raise ValueError("Missing generation shards")
    if len({r["state_id"] for r in result}) != len(result):
        raise ValueError("Duplicate generated states")
    return result
