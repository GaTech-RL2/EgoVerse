"""Three-task Astra extension; admission always uses the standalone delta."""
from copy import deepcopy
import json

from ..core import forward_delta, parse_policy, policy_answer, select_candidates
from ..data import file_sha256, fingerprint, write_json, write_jsonl
from ..inference import Request, sample_seed
from ..pipeline import score_record
from ..prompts import forward_prompt
from ..training import conversation, fits_context
from .leact import generate_batch as generate_hint_batch, hint_prompt

ARMS = ("astra_actions", "astra_conditioning_v2")


def assisted_prompt(state, rules):
    # Validate and quote the teacher text using the same contract as the hint.
    hint_prompt(state, rules, "astra_actions_summary")
    return (forward_prompt(state, rules, variant="concise_v1")
            + "\n\nAdditional public explanation supplied by the expert (it may be mistaken):\n"
            + json.dumps(state["teacher_decision_summary"]))


def backward(state, rules, arm):
    if arm not in ARMS:
        raise ValueError("Unknown conditioning arm")
    return hint_prompt(state, rules, "astra_actions" if arm == "astra_actions" else "astra_actions_summary")


def generate_batch(states, rules, backend, *, arm, n=8, m=4, seed=0):
    if arm not in ARMS:
        raise ValueError("Unknown conditioning arm")
    rows = generate_hint_batch(states, rules, backend, arm=(
        "astra_actions" if arm == "astra_actions" else "astra_actions_summary"), n=n, m=m, seed=seed)
    if arm == "astra_conditioning_v2":
        baselines = backend.generate([Request(assisted_prompt(s, rules), m,
                    sample_seed(seed, s["state_id"], "assisted-baseline")) for s in states])
        conditioned = backend.generate([Request(assisted_prompt(s, rules), 1,
                       sample_seed(seed, s["state_id"], f"assisted-conditioned-{i}"),
                       reasoning=c["reasoning"])
                       for s, row in zip(states, rows) for i, c in enumerate(row["candidates"])])
        for j, (state, row) in enumerate(zip(states, rows)):
            row["assisted_baselines"] = baselines[j]
            row["prompts"]["assisted_forward"] = assisted_prompt(state, rules)
            for i, c in enumerate(row["candidates"]):
                c["assisted_forward"] = conditioned[j*n+i][0]
    for row in rows:
        row["astra_arm"] = arm
        row["objective_version"] = 2
    return rows


def admitted_pool(states, metadata, raw, tokenizer, *, arm, cutoff=2048, k=2):
    by_id = {s["state_id"]: s for s in states}
    if arm not in ARMS or tokenizer is None or len(raw) != len(states) or {r["state_id"] for r in raw} != set(by_id):
        raise ValueError("Expected exact teacher-state coverage and a tokenizer")
    pool, audit = [], []
    for record in raw:
        state = by_id[record["state_id"]]
        prompts = {"forward": forward_prompt(state, metadata["rules"], variant="concise_v1"),
                   "backward": backward(state, metadata["rules"], arm)}
        if arm == "astra_conditioning_v2":
            prompts["assisted_forward"] = assisted_prompt(state, metadata["rules"])
        if (record.get("astra_arm") != arm or record.get("prompts") != prompts
                or record.get("objective_version") != 2
                or record.get("teacher_policy_sha256") != fingerprint(state["expert_policy"])):
            raise ValueError("Generation teacher, prompt, or objective mismatch")
        scored = score_record(state, record, method="leact", k=k)
        valid = scored["baseline_truncations"] == 0 and all(f is None for f in scored["baseline_fallbacks"])
        scored.update(baseline_valid=valid, primary_admission="positive standalone delta only")
        for candidate, generation in zip(scored["candidates"], record["candidates"]):
            candidate["delta_0"] = candidate["score"]
            candidate["delta_r"] = None
            if arm == "astra_conditioning_v2":
                bs = [parse_policy(b["text"], state["legal_actions"]) for b in record["assisted_baselines"]]
                output = generation["assisted_forward"]
                pred = parse_policy(output["text"], state["legal_actions"])
                valid_r = (output["finish_reason"] == "stop" and pred.fallback is None
                           and all(b.fallback is None for b in bs)
                           and all(b["finish_reason"] == "stop" for b in record["assisted_baselines"]))
                candidate["assisted_valid"] = valid_r
                if valid_r:
                    candidate["delta_r"] = forward_delta(state["expert_policy"], pred.policy, [b.policy for b in bs])
        scored["selected"] = select_candidates([c for c in scored["candidates"] if c["fallback"] is None], k, "leact") if valid else []
        for c in scored["selected"]:
            answer = f"<think>\n{c['reasoning']}\n</think>\n\n"
            views = {"backward": conversation(prompts["backward"], answer + "DONE"),
                     "standalone_forward": conversation(prompts["forward"], answer + policy_answer(state["expert_policy"]))}
            if arm == "astra_conditioning_v2":
                views["assisted_forward"] = conversation(prompts["assisted_forward"], answer + policy_answer(state["expert_policy"]))
            c["fits_all_views"] = all(fits_context(pair, tokenizer, cutoff) for pair in views.values())
            c["admitted_to_sft"] = c["fits_all_views"]
            if c["fits_all_views"]:
                pool.append({"state_id": state["state_id"], "candidate_id": c["candidate_id"], "views": views})
        audit.append(scored)
    return pool, audit


def prepare_matched(pools, audits, metadata, directories, *, round_index, cutoff=2048):
    """Keep equal trace counts and updates; use a balanced r-dropout forward mixture.

    Every view is retained for audit. SFT samples one of the two forward contexts
    per retained trace, independently of its score, giving weights 1:.5:.5 in
    expectation. The backward example is always included, as in original LeAct.
    """
    count = min(len(pools[a]) for a in ARMS)
    # An even number permits an exact 50/50 empirical context mixture.
    count -= count % 2
    if count < 2:
        raise ValueError("Fewer than two valid standalone-positive traces in a matched arm")
    manifests = {}
    for arm in ARMS:
        directory = directories[arm]
        if directory.exists():
            raise FileExistsError("Choose fresh SFT directories")
        ordered = sorted(pools[arm], key=lambda row: fingerprint([round_index, row["state_id"], row["candidate_id"], "matched-sampling"]))
        chosen = ordered[:count]
        ids = {(r["state_id"], r["candidate_id"]) for r in chosen}
        audit = deepcopy(audits[arm])
        for row in audit:
            for c in row["selected"]:
                c["admitted_to_sft"] = (row["state_id"], c["candidate_id"]) in ids
                c["rejection_reason"] = (None if c["admitted_to_sft"] else
                                          "context_length" if not c["fits_all_views"] else "matched_compute_subsample")
        dataset, tasks = [], {"backward": 0, "standalone_forward": 0, "assisted_forward": 0}
        for i, row in enumerate(chosen):
            task = "assisted_forward" if arm == "astra_conditioning_v2" and i % 2 == 0 else "standalone_forward"
            dataset.extend([row["views"]["backward"], row["views"][task]])
            row = dict(row)
            tasks["backward"] += 1
            tasks[task] += 1
        directory.mkdir(parents=True)
        write_json(directory / "data.json", dataset)
        write_jsonl(directory / "all-views.jsonl", chosen)
        write_jsonl(directory / "selection.jsonl", audit)
        write_json(directory / "dataset_info.json", {"leact_local": {
            "file_name": "data.json", "formatting": "sharegpt", "columns": {"messages": "conversations"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}}})
        manifest = {"method": "leact", "astra_arm": arm, "objective_version": 2,
                    "oracle_sha256": metadata["sha256"], "cfr_evaluation_oracle_sha256": metadata["cfr_evaluation_oracle_sha256"],
                    "selected_traces": count, "eligible_traces_before_compute_matching": len(pools[arm]),
                    "sft_examples": len(dataset), "task_counts": tasks, "round": round_index,
                    "context_length_checked": True, "cutoff_len": cutoff, "prompt_variant": "concise_v1",
                    "data_sha256": file_sha256(directory / "data.json"),
                    "admission": "top 2 standalone-positive traces per state, all-view length check, then score-independent matched subsample",
                    "forward_context_sampling": "balanced 50/50 r mixture for v2, frozen for this round; all three views retained",
                    "loss": "Serialized-policy token cross-entropy surrogate; backward:assisted:standalone example counts 1:.5:.5",
                    "valid_standalone_baseline_states": sum(r["baseline_valid"] for r in audit),
                    "positive_candidates": sum(c["delta_0"] > 0 for r in audit for c in r["candidates"]),
                    "assisted_positive_only_candidates": sum(c["delta_r"] is not None and c["delta_r"] > 0 and c["delta_0"] <= 0 for r in audit for c in r["candidates"])}
        write_json(directory / "manifest.json", manifest)
        manifests[arm] = manifest
    return manifests
