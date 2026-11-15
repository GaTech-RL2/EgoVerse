"""Full-tree exploitability and explicitly oracle-aware best-of-N evaluation."""

from __future__ import annotations

import statistics

from .core import kl, parse_policy


def evaluate_records(states: list[dict], metadata: dict, raw: list[dict]) -> dict:
    import pyspiel

    by_id = {s["state_id"]: s for s in states}
    if len(raw) != len(states) or {r["state_id"] for r in raw} != set(by_id):
        raise ValueError("Full-tree evaluation requires exactly one record per oracle information state")
    if len(states) != metadata["num_infosets"]:
        raise ValueError("A partial state subset cannot be reported as full-tree exploitability")
    seeds = {r["seed"] for r in raw}
    counts = {len(r["candidates"]) for r in raw}
    if len(seeds) != 1 or len(counts) != 1 or 0 in counts:
        raise ValueError("Mixed inference seeds or candidate counts")
    tables = {"single": {}, "oracle_bon": {}}
    kls = {name: [] for name in tables}
    fallbacks = truncations = 0
    for row in raw:
        state = by_id[row["state_id"]]
        parsed = [parse_policy(c["forward"]["text"], state["legal_actions"]) for c in row["candidates"]]
        losses = [kl(state["expert_policy"], p.policy) for p in parsed]
        fallbacks += sum(p.fallback is not None for p in parsed)
        truncations += sum(c["forward"]["finish_reason"] != "stop" for c in row["candidates"])
        choices = {"single": 0, "oracle_bon": min(range(len(parsed)), key=lambda i: losses[i])}
        for name, index in choices.items():
            tables[name][state["info_state"]] = [(state["action_ids"][a], p) for a, p in parsed[index].policy.items()]
            kls[name].append(losses[index])
    game = pyspiel.load_game(metadata["game_string"])
    result = {
        "seed": next(iter(seeds)), "n": next(iter(counts)), "infosets": len(states),
        "oracle_bon_requires_teacher": True, "oracle_sha256": metadata["sha256"],
        "oracle_exploitability": metadata["exploitability"],
        "parser_fallbacks": fallbacks, "truncated_samples": truncations,
    }
    for name, table in tables.items():
        result[name] = {
            "exploitability": pyspiel.exploitability(game, pyspiel.TabularPolicy(table)),
            "mean_capped_kl": statistics.mean(kls[name]),
            "close_fraction_kl_below_0_1": statistics.mean(v < 0.1 for v in kls[name]),
        }
    return result


def aggregate_evaluations(results: list[dict]) -> dict:
    if not results or len({r["seed"] for r in results}) != len(results):
        raise ValueError("Expected distinct inference seeds")
    if len({(r["oracle_sha256"], r["n"], r["infosets"]) for r in results}) != 1:
        raise ValueError("Cannot aggregate different evaluation protocols")
    summary = {"inference_seeds": [r["seed"] for r in results], "training_replicates": 1,
               "std_ddof": 0, "oracle_bon_requires_teacher": True, "per_seed": results}
    for regime in ["single", "oracle_bon"]:
        summary[regime] = {}
        for metric in ["exploitability", "mean_capped_kl", "close_fraction_kl_below_0_1"]:
            values = [r[regime][metric] for r in results]
            summary[regime][metric] = {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
    return summary
