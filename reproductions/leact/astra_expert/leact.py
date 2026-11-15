"""The two Astra information conditions, retaining LeAct's student-trace SFT."""

from dataclasses import replace
import json

from ..core import select_candidates
from ..data import file_sha256, fingerprint, write_json, write_jsonl
from ..pipeline import generate_batch as original_generate_batch, score_record
from ..prompts import backward_prompt, forward_prompt
from ..training import fits_context, supervised_pairs
from .client import validate_summary

ARMS = ("astra_actions", "astra_actions_summary")


def hint_prompt(state, rules, arm):
    if arm not in ARMS:
        raise ValueError("Unknown Astra information condition")
    prompt = backward_prompt(state, rules, variant="concise_v1")
    if arm == "astra_actions_summary":
        summary = state["teacher_decision_summary"]
        validate_summary({"decision_summary": summary})
        prompt += ("\n\nAdditional public explanation supplied by the same expert "
                   "after committing its action policy (it may be mistaken):\n" + json.dumps(summary))
    return prompt


def generate_batch(states, rules, backend, *, arm, n=8, m=4, seed=0):
    # Reuse the existing N/M sampling, assistant conditioning and random seeds.
    # Only backward requests change; ordinary forward/evaluation requests have
    # neither the teacher policy nor its explanation in context.
    replacements = {backward_prompt(s, rules, variant="concise_v1"): hint_prompt(s, rules, arm)
                    for s in states}

    class HintBackend:
        def generate(self, requests):
            return backend.generate([replace(r, prompt=replacements.get(r.prompt, r.prompt)) for r in requests])

    raw = original_generate_batch(states, rules, HintBackend(), method="leact", n=n, m=m,
                                  seed=seed, prompt_variant="concise_v1")
    for state, row in zip(states, raw):
        row["prompts"]["backward"] = hint_prompt(state, rules, arm)
        row["astra_arm"] = arm
        row["teacher_policy_sha256"] = fingerprint(state["expert_policy"])
    return raw


def prepare_data(states, metadata, output, *, arm, raw, tokenizer, cutoff=2048, k=2):
    """Joint forward/backward SFT, with the hint included in the length audit."""
    if arm not in ARMS or tokenizer is None:
        raise ValueError("A valid Astra condition and student tokenizer are required")
    if output.exists():
        raise FileExistsError("Choose a fresh training dataset directory")
    by_id = {s["state_id"]: s for s in states}
    if len(by_id) != len(states) or len(raw) != len(states) or {r["state_id"] for r in raw} != set(by_id):
        raise ValueError("Generation must cover the teacher states exactly")
    dataset, audit = [], []
    selected_count = dropped = 0
    for record in raw:
        state = by_id[record["state_id"]]
        expected_prompts = {"forward": forward_prompt(state, metadata["rules"], variant="concise_v1"),
                            "backward": hint_prompt(state, metadata["rules"], arm)}
        if (record.get("astra_arm") != arm or record.get("prompts") != expected_prompts
                or record.get("prompt_variant") != "concise_v1"
                or record.get("teacher_policy_sha256") != fingerprint(state["expert_policy"])):
            raise ValueError("Generation used a different teacher, prompt or information condition")
        scored = score_record(state, record, method="leact", k=k)
        valid_baselines = scored["baseline_truncations"] == 0 and all(f is None for f in scored["baseline_fallbacks"])
        scored.update(valid_baselines_required=True, baseline_valid=valid_baselines)
        scored["selected"] = (select_candidates([c for c in scored["candidates"] if c["fallback"] is None], k, "leact")
                              if valid_baselines else [])
        for candidate in scored["selected"]:
            pairs = supervised_pairs(state, candidate, metadata["rules"], method="leact",
                                     expert_forcing=True, prompt_variant="concise_v1")
            pairs[1]["conversations"][0]["value"] = expected_prompts["backward"]
            fits = all(fits_context(pair, tokenizer, cutoff) for pair in pairs)
            candidate["admitted_to_sft"] = fits
            if fits:
                dataset.extend(pairs)
                selected_count += 1
            else:
                dropped += 1
        audit.append(scored)
    if not dataset:
        raise ValueError("No admitted student traces; cannot run an empty SFT step")
    output.mkdir(parents=True)
    write_json(output / "data.json", dataset)
    write_json(output / "dataset_info.json", {"leact_local": {
        "file_name": "data.json", "formatting": "sharegpt", "columns": {"messages": "conversations"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}})
    write_jsonl(output / "selection.jsonl", audit)
    manifest = {"method": "leact", "astra_arm": arm, "oracle_sha256": metadata["sha256"],
                "teacher_policy_sha256": metadata["sha256"],
                "cfr_evaluation_oracle_sha256": metadata["cfr_evaluation_oracle_sha256"],
                "infosets": len(states), "sft_examples": len(dataset), "selected_traces": selected_count,
                "states_with_selected_traces": sum(any(c["admitted_to_sft"] for c in r["selected"]) for r in audit),
                "overlength_dropped": dropped, "context_length_checked": True, "cutoff_len": cutoff,
                "k": k, "prompt_variant": "concise_v1", "require_valid_policies": True,
                "invalid_baseline_states": sum(not r["baseline_valid"] for r in audit),
                "candidate_count": sum(len(r["candidates"]) for r in audit),
                "candidate_fallbacks": sum(c["fallback"] is not None for r in audit for c in r["candidates"]),
                "truncated_candidates": sum(not c["complete"] for r in audit for c in r["candidates"]),
                "positive_candidates": sum(c["score"] > 0 for r in audit for c in r["candidates"]),
                "expert_forcing": True, "data_sha256": file_sha256(output / "data.json"),
                "supervision": "Admitted student traces plus identical Astra action targets; teacher explanation only in the backward hint"}
    write_json(output / "manifest.json", manifest)
    return manifest
