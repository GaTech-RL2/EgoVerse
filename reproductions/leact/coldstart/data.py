"""Freeze a matched coldstart corpus before changing coverage or LeAct rounds."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from ..astra_expert.client import (ENDPOINT, MODEL, SETTINGS, SYSTEM, call_request,
                                   decode_json, validate_policy)
from ..core import kl
from ..data import file_sha256, fingerprint, load_oracle, read_jsonl, write_json, write_jsonl
from ..reproduction import PILOT_SHA256
from ..training import supervised_pairs

PROMPT_VERSION = "leduc-frontier-coldstart-public-rationale-v1"
ARMS = ("self_38_paper_lr", "astra_38_paper_lr")


def matched_slots(source: Path, states: list[dict], rules: str) -> list[dict]:
    """Preserve the exact state multiplicities and ordering of the 38-example anchor."""
    if file_sha256(source / "data.json") != PILOT_SHA256:
        raise ValueError("Expected the frozen 38-example self-generated anchor")
    by_id = {s["state_id"]: s for s in states}
    pairs, slots = [], []
    for row in read_jsonl(source / "selection.jsonl"):
        for candidate in row["selected"]:
            if candidate.get("admitted_to_sft"):
                state = by_id[row["state_id"]]
                pairs.extend(supervised_pairs(state, candidate, rules, prompt_variant="concise_v1"))
                slots.append({"slot": len(slots), "state_id": state["state_id"],
                              "self_candidate_id": candidate["candidate_id"]})
    if pairs != json.loads((source / "data.json").read_text()) or len(pairs) != 38:
        raise ValueError("Anchor selection does not reconstruct its exact training data")
    return slots


def request_for(state: dict, rules: str) -> dict:
    # No solver distribution, qualitative solver hint or hidden state enters
    # the frontier's forward generation. CFR replaces only the SFT action target.
    context = {"rules": rules, "information_state": state["info_state"],
               "player": state["player"], "legal_actions": state["legal_actions"]}
    instruction = (
        "Choose a mixed strategy and provide a short public strategic justification "
        "written to teach another poker agent. Use 3-5 sentences and at most 200 words. "
        "Connect your visible private card, public card if present, betting history, "
        "uncertainty and legal-action tradeoffs. Refer to the exact visible card tokens "
        "at least once. Do not assert knowledge of the opponent's private card, restate "
        "the rules, or include formatting instructions. Give the justification, not "
        "a transcript of private deliberation. Return only JSON with exactly three "
        'fields: {"reasoning": <public justification>, "action": <most probable legal '
        'action>, "policy": {<every legal action>: <probability>}}. Probabilities must '
        "be finite numbers in [0,1] summing to one. Do not use thinking or role tags."
    )
    return {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context, sort_keys=True) + "\n\n" + instruction}],
            "stream": False, "response_format": {"type": "json_object"},
            "cache": {"no-cache": True}, **SETTINGS}


def validate_candidate(envelope: dict, state: dict) -> dict:
    value = decode_json(envelope)
    if not isinstance(value, dict) or set(value) != {"reasoning", "action", "policy"}:
        raise ValueError("Expected reasoning, action and policy only")
    validate_policy({k: value[k] for k in ("action", "policy")}, state["legal_actions"])
    reasoning = value["reasoning"]
    if not isinstance(reasoning, str) or not 1 <= len(reasoning.split()) <= 200:
        raise ValueError("Public justification must contain 1 to 200 words")
    if any(t in reasoning.lower() for t in ("<think", "</think", "<|", "policy:", "action:")):
        raise ValueError("Public justification contains output-control text")
    for card in re.findall(r"\[(?:Private|Public): ([^\]]+)\]", state["info_state"]):
        if card.strip() and card.strip() not in reasoning:
            raise ValueError("Public justification omitted a visible card token")
    return value


def collection_protocol(slots: list[dict], metadata: dict) -> dict:
    return {
        "study": "Leduc minimal coldstart: matched 38-example comparison",
        "paper": "https://arxiv.org/html/2607.21856v1#A8.SS3",
        "oracle_sha256": metadata["sha256"], "source_dataset_sha256": PILOT_SHA256,
        "model": MODEL, "prompt_version": PROMPT_VERSION, "settings": SETTINGS,
        "arms": list(ARMS), "slots": slots, "examples_per_arm": 2 * len(slots),
        "unique_infosets": len({s["state_id"] for s in slots}),
        "variant": "joint forward and backward, equal example counts",
        "action_target": "Exact CFR distribution in both arms; Astra's proposed policy is audit-only",
        "training": {"epochs": 3, "learning_rate": 1e-4, "effective_batch": 32,
                     "cutoff_len": 2048, "seed": 0, "initialization": "untouched pinned Qwen3-8B"},
        "evaluation": {"infosets": 4032, "inference_seeds": [0, 1, 2], "n": 8,
                       "max_generation_tokens": 4096, "prompt_variant": "concise_v1",
                       "primary": "fixed final epoch; all epochs also reported for paper comparison",
                       "metrics": ["single exploitability", "oracle best-of-8 exploitability"]},
        "interpretation": [
            "Old self_38 at 5e-6 versus self_38 at 1e-4 isolates the learning-rate correction.",
            "New arms match states, multiplicities, student prompts, action targets and optimizer updates.",
            "Astra replaces the reasoning generation procedure; reasoning length and teacher prompt direction are not matched.",
            "Original reasoning came from backward generation and positive-delta selection; frontier coldstart is generated forward without delta filtering.",
            "The 11 pilot infosets are inherited, not a representative coverage sample.",
            "H.3 explicitly specifies 4032 NoCoT examples for 6r2s but does not specify its reasoning-corpus size; this is a reconstruction, not the author's corpus.",
            "Astra substitutes for the paper's Claude/Gemini/GPT-5 generators.",
        ],
        "next_steps": [
            "Compare both coldstarts before another training round.",
            "Then enlarge a fixed nested, stratified frontier pool (128, 512, 4032 states) only as needed.",
            "Run one CFR-guided LeAct round from a promising coldstart before adding rounds or Astra expert targets.",
        ],
        "scope": "Format and initial-policy study; no claim of a minimal recipe or successful reproduction yet",
    }


def collect(source: Path, root: Path, oracle: Path, *, workers=4, limit=None):
    states, metadata = load_oracle(oracle)
    slots = matched_slots(source, states, metadata["rules"])
    protocol = collection_protocol(slots, metadata)
    target = root / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Collection protocol changed; choose a fresh directory")
    write_json(target, protocol)
    by_id = {s["state_id"]: s for s in states}

    def generate(slot):
        state = by_id[slot["state_id"]]
        request = request_for(state, metadata["rules"])
        identity = {**slot, "oracle_sha256": metadata["sha256"], "prompt_version": PROMPT_VERSION,
                    "endpoint": ENDPOINT, "stage": "frontier_coldstart",
                    "request_sha256": fingerprint(request)}
        path = root / "calls" / f"slot-{slot['slot']:03d}.json"
        row = call_request(request, path, identity=identity,
                           validate=lambda envelope: validate_candidate(envelope, state))
        return {**slot, "accepted": row["accepted"], "usage": row["response"].get("usage"),
                "seconds": row["latency_seconds"], "sha256": file_sha256(path)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(generate, slots if limit is None else slots[:limit]):
            print(json.dumps(result), flush=True)
    if limit is not None and limit < len(slots):
        return
    datasets = {ARMS[0]: json.loads((source / "data.json").read_text()), ARMS[1]: []}
    audit = []
    for slot in slots:
        state = by_id[slot["state_id"]]
        path = root / "calls" / f"slot-{slot['slot']:03d}.json"
        row = json.loads(path.read_text())
        candidate = validate_candidate(row["response"], state)
        datasets[ARMS[1]].extend(supervised_pairs(state, candidate, metadata["rules"],
                                                 prompt_variant="concise_v1"))
        audit.append({**slot, "frontier": candidate, "cfr_target": state["expert_policy"],
                      "teacher_policy_kl_to_cfr": kl(state["expert_policy"], candidate["policy"]),
                      "record_sha256": file_sha256(path),
                      "frontier_public_output_only": True, "delta_filter_applied": False})
    for left, right in zip(datasets[ARMS[0]], datasets[ARMS[1]], strict=True):
        if left["conversations"][0] != right["conversations"][0]:
            raise ValueError("Matched student prompts changed")
        if left["conversations"][1]["value"].split("</think>", 1)[1] != right["conversations"][1]["value"].split("</think>", 1)[1]:
            raise ValueError("Matched action targets changed")
    for arm, pairs in datasets.items():
        output = root / "datasets" / arm
        write_json(output / "data.json", pairs)
        write_json(output / "dataset_info.json", {
            "leact_local": {"file_name": "data.json", "formatting": "sharegpt",
                            "columns": {"messages": "conversations"},
                            "tags": {"role_tag": "from", "content_tag": "value",
                                     "user_tag": "human", "assistant_tag": "gpt"}}})
        write_json(output / "manifest.json", {
            "method": "coldstart_joint", "arm": arm, "oracle_sha256": metadata["sha256"],
            "sft_examples": len(pairs), "selected_traces": len(slots),
            "infosets": len({s["state_id"] for s in slots}), "expert_forcing": True,
            "prompt_variant": "concise_v1", "cutoff_len": 2048,
            "context_length_checked": False, "data_sha256": file_sha256(output / "data.json"),
            "source_dataset_sha256": PILOT_SHA256, "protocol_sha256": fingerprint(protocol)})
    write_jsonl(root / "audit.jsonl", audit)
    write_json(root / "collection_complete.json", {"protocol_sha256": fingerprint(protocol),
               "calls": len(slots), "examples_per_arm": 38, "infosets": protocol["unique_infosets"],
               "data_sha256": {arm: file_sha256(root / "datasets" / arm / "data.json") for arm in ARMS},
               "audit_sha256": file_sha256(root / "audit.jsonl")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 19:
        parser.error("limit must be 1 to 19")
    collect(args.source, args.root, args.oracle, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
