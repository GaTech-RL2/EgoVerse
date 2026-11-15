"""Read-only summary of stored preflight traces and tokenizer/EOS provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def text_summary(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    common = Counter(lines).most_common(3)
    return {
        "characters": len(text), "beginning": text[:420], "ending": text[-420:],
        "open_think_tags": text.count("<think>"), "close_think_tags": text.count("</think>"),
        "most_common_lines": [{"text": line[:180], "count": count} for line, count in common],
    }


def generation_summary(items: list[dict]) -> dict:
    tokens = [item["generated_tokens"] for item in items]
    return {
        "count": len(items), "finish_reasons": dict(Counter(item["finish_reason"] for item in items)),
        "generated_tokens": {"min": min(tokens), "mean": statistics.mean(tokens), "max": max(tokens)},
        "generated_token_histogram": dict(Counter(tokens)),
        "texts_containing_close_think": sum("</think>" in item["text"] for item in items),
    }


def template_snapshot(directory: Path) -> dict:
    direct = directory / "chat_template.jinja"
    if direct.is_file():
        content = direct.read_text()
        source = str(direct)
    else:
        tokenizer = read_json(directory / "tokenizer_config.json") or {}
        content = tokenizer.get("chat_template")
        source = str(directory / "tokenizer_config.json") + "#chat_template"
    encoded = content.encode() if isinstance(content, str) else json.dumps(content, sort_keys=True).encode()
    return {"source": source, "present": content is not None,
            "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}


def checkpoint_snapshot(directory: Path) -> dict:
    fields = ["eos_token_id", "bos_token_id", "pad_token_id", "do_sample", "temperature", "top_p", "top_k"]
    generation = read_json(directory / "generation_config.json")
    model = read_json(directory / "config.json") or {}
    tokenizer = read_json(directory / "tokenizer_config.json") or {}
    return {
        "path": str(directory),
        "generation_config": {k: generation.get(k) for k in fields} if generation else None,
        "model_eos_token_id": model.get("eos_token_id"),
        "tokenizer_eos_token": tokenizer.get("eos_token"),
        "template": template_snapshot(directory),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    shard_paths = sorted((root / "preflight/leact").glob("shard-*.jsonl"))
    rows = [json.loads(line) for path in shard_paths for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No stored preflight generation shards")
    candidates = [(row["state_id"], index, candidate) for row in rows
                  for index, candidate in enumerate(row["candidates"])]
    backward = [c[2]["backward"] for c in candidates]
    conditioned = [c[2]["forward"] for c in candidates]
    baselines = [item for row in rows for item in row["baselines"]]
    samples = []
    for state_id, index, candidate in (candidates[0], candidates[-1]):
        expected_prefix = "<think>\n" + candidate["reasoning"] + "\n</think>\n\n"
        text = candidate["forward"]["text"]
        prefix_matches = text.startswith(expected_prefix)
        continuation = text[len(expected_prefix):] if prefix_matches else text
        samples.append({
            "state_id": state_id, "candidate_index": index,
            "backward": text_summary(candidate["backward"]["text"]),
            "reasoning_characters": len(candidate["reasoning"]),
            "conditioned_prefix_matches": prefix_matches,
            "conditioned_continuation": text_summary(continuation),
        })
    base_info = read_json(root / "base_model.json")
    base = checkpoint_snapshot(Path(base_info["path"]))
    trained = checkpoint_snapshot(root / "preflight/sft/checkpoints")
    result = {
        "scope": "read-only diagnosis of the completed 2-step hardware preflight",
        "root": str(root), "shards": len(shard_paths), "states": len(rows),
        "backward": generation_summary(backward), "conditioned": generation_summary(conditioned),
        "unconditioned_baselines": generation_summary(baselines), "samples": samples,
        "base": base, "trained": trained,
        "templates_identical": (base["template"]["sha256"] == trained["template"]["sha256"]
                                if base["template"]["present"] and trained["template"]["present"] else None),
        "preflight_result": read_json(root / "preflight/result.json"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
