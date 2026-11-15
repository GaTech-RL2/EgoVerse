"""Export paired ShareGPT data and a full-parameter LLaMA-Factory recipe."""

from __future__ import annotations

import json
from pathlib import Path

from .core import policy_answer, select_candidates
from .data import file_sha256, write_json, write_jsonl
from .pipeline import score_record
from .prompts import backward_prompt, forward_prompt


def conversation(prompt: str, answer: str) -> dict:
    return {"conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]}


def supervised_pairs(state: dict, selected: dict, rules: str, *, method="leact",
                     expert_forcing: bool | None = None, prompt_variant="original") -> list[dict]:
    forcing = method == "leact" if expert_forcing is None else expert_forcing
    target = state["expert_policy"] if forcing else selected["predicted_policy"]
    z = selected["reasoning"]
    pairs = [conversation(forward_prompt(state, rules, variant=prompt_variant), f"<think>\n{z}\n</think>\n\n{policy_answer(target)}")]
    if method == "leact":
        pairs.append(conversation(backward_prompt(state, rules, variant=prompt_variant), f"<think>\n{z}\n</think>\n\nDONE"))
    return pairs


def fits_context(pair: dict, tokenizer, cutoff: int) -> bool:
    messages = [{"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]}
                for c in pair["conversations"]]
    # LLaMA-Factory qwen3 and this chat template use the same ChatML boundaries.
    return len(tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=True)) <= cutoff


def prepare_data(states: list[dict], metadata: dict, output: Path, *, method: str,
                 raw: list[dict] | None = None, k=2, tokenizer=None,
                 cutoff=2048, expert_forcing=None, prompt_variant="original",
                 require_valid_policies=False) -> dict:
    if method not in {"bc", "leact", "exit"}:
        raise ValueError("Unknown training method")
    if output.exists():
        raise FileExistsError(f"Choose a fresh training directory: {output}")
    by_id = {s["state_id"]: s for s in states}
    if method != "bc":
        if raw is None or len(raw) != len(states) or {r["state_id"] for r in raw} != set(by_id):
            raise ValueError("Training generation must cover the requested oracle states exactly")
        if any(r.get("prompt_variant", "original") != prompt_variant for r in raw):
            raise ValueError("Training prompt variant differs from generation")
    dataset, audit = [], []
    dropped = selected_count = 0
    if method == "bc":
        for state in states:
            pair = conversation(forward_prompt(state, metadata["rules"], thinking=False, variant=prompt_variant),
                                "<think>\n\n</think>\n\n" + policy_answer(state["expert_policy"]))
            if tokenizer is None or fits_context(pair, tokenizer, cutoff):
                dataset.append(pair)
            else:
                dropped += 1
    else:
        for record in raw:
            state = by_id[record["state_id"]]
            scored = score_record(state, record, method=method, k=k)
            if require_valid_policies:
                baseline_valid = (method != "leact" or
                                  (scored["baseline_truncations"] == 0
                                   and all(f is None for f in scored["baseline_fallbacks"])))
                scored["valid_baselines_required"] = method == "leact"
                scored["baseline_valid"] = baseline_valid
                scored["selected"] = (select_candidates(
                    [c for c in scored["candidates"] if c["fallback"] is None], k, method)
                    if baseline_valid else [])
            for candidate in scored["selected"]:
                pairs = supervised_pairs(state, candidate, metadata["rules"], method=method,
                                         expert_forcing=expert_forcing, prompt_variant=prompt_variant)
                fits = tokenizer is None or all(fits_context(p, tokenizer, cutoff) for p in pairs)
                candidate["admitted_to_sft"] = fits
                if fits:
                    dataset.extend(pairs)
                    selected_count += 1
                else:
                    dropped += 1
            audit.append(scored)
    if not dataset:
        raise ValueError("No examples survived selection/context filtering; do not run an empty M-step")
    output.mkdir(parents=True)
    write_json(output / "data.json", dataset)
    write_json(output / "dataset_info.json", {
        "leact_local": {"file_name": "data.json", "formatting": "sharegpt",
                        "columns": {"messages": "conversations"},
                        "tags": {"role_tag": "from", "content_tag": "value",
                                 "user_tag": "human", "assistant_tag": "gpt"}},
    })
    if audit:
        write_jsonl(output / "selection.jsonl", audit)
    summary = {
        "method": method, "oracle_sha256": metadata["sha256"], "infosets": len(states),
        "sft_examples": len(dataset), "selected_traces": selected_count,
        "overlength_dropped": dropped, "context_length_checked": tokenizer is not None,
        "cutoff_len": cutoff, "k": k,
        "prompt_variant": prompt_variant,
        "require_valid_policies": require_valid_policies,
        "invalid_baseline_states": sum(not r.get("baseline_valid", True) for r in audit),
        "positive_candidates": sum(c["score"] > 0 for r in audit for c in r["candidates"]) if method == "leact" else None,
        "candidate_count": sum(len(r["candidates"]) for r in audit),
        "candidate_fallbacks": sum(c["fallback"] is not None for r in audit for c in r["candidates"]),
        "truncated_candidates": sum(not c["complete"] for r in audit for c in r["candidates"]),
        "states_with_selected_traces": sum(any(c.get("admitted_to_sft") for c in r["selected"]) for r in audit),
        "expert_forcing": method != "exit" if expert_forcing is None else expert_forcing,
        "data_sha256": file_sha256(output / "data.json"),
    }
    write_json(output / "manifest.json", summary)
    return summary


def write_recipe(directory: Path, checkpoint: str, *, gpus=8, epochs=3,
                 learning_rate=5e-6, seed=0, max_steps=-1) -> Path:
    import yaml

    if gpus <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("Invalid training settings")
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["context_length_checked"]:
        raise ValueError("Token-length audit required before exporting a GPU training recipe")
    if manifest["data_sha256"] != file_sha256(directory / "data.json"):
        raise ValueError("Training dataset checksum mismatch")
    recipe = {
        "model_name_or_path": checkpoint,
        "stage": "sft", "do_train": True, "finetuning_type": "full",
        "dataset": "leact_local", "dataset_dir": str(directory.resolve()),
        "template": "qwen3", "cutoff_len": manifest["cutoff_len"],
        # NoCoT supplies the empty thinking block as a masked prompt prefix.
        # Supervising that block directly teaches an immediate end to reasoning.
        "enable_thinking": manifest["method"] != "bc",
        "train_on_prompt": False, "mask_history": False,
        "output_dir": str((directory / "checkpoints").resolve()),
        "overwrite_output_dir": False, "overwrite_cache": False,
        "save_strategy": "epoch", "save_only_model": True,
        "logging_steps": 1, "report_to": "none", "plot_loss": True,
        "per_device_train_batch_size": 4, "gradient_accumulation_steps": max(1, 32 // (4 * gpus)),
        "learning_rate": learning_rate, "num_train_epochs": epochs,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
        "weight_decay": 0.01, "adam_beta1": 0.9, "adam_beta2": 0.999,
        "bf16": True, "gradient_checkpointing": True,
        "flash_attn": "sdpa", "seed": seed, "data_seed": seed,
        "preprocessing_num_workers": 4, "dataloader_num_workers": 2,
        "ddp_timeout": 1800,
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
            "backward_prefetch": "backward_pre", "forward_prefetch": False,
            "use_orig_params": True, "sync_module_states": True,
            "limit_all_gathers": True, "cpu_ram_efficient_loading": True,
        },
    }
    if max_steps > 0:
        recipe["max_steps"] = max_steps
    path = directory / "train.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return path
