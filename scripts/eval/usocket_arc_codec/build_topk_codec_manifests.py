#!/usr/bin/env python3
"""Build deterministic candidate and episode manifests for codec reranking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_candidate_manifest(ranking_path: Path, top_k: int) -> dict:
    with ranking_path.open(newline="") as stream:
        ranking = list(csv.DictReader(stream))
    if not 1 <= top_k <= len(ranking):
        raise ValueError(f"top-k must be in [1,{len(ranking)}], got {top_k}")
    candidates = []
    for row in ranking[:top_k]:
        candidates.append({
            "source_rank": int(row["rank"]),
            "source_index": int(row["index"]),
            "candidate": row["candidate"],
            "config": {
                "kind": "arc",
                "distance": float(row["D"]),
                "M": int(row["M"]),
                "degrees": float(row["R_deg"]),
            },
        })
    names = [row["candidate"] for row in candidates]
    if len(names) != len(set(names)):
        raise ValueError("selected candidate names are not unique")
    return {
        "schema_version": 1,
        "label": "TOP_K_FROM_VERIFIED_CODEC_RANKING",
        "source_ranking_sha256": sha256(ranking_path),
        "top_k": top_k,
        "candidates": candidates,
    }


def build_episode_manifest(split_path: Path, episode_count: int, salt: str) -> dict:
    split = json.loads(split_path.read_text())
    domains = split.get("domains", {})
    if len(domains) != 1:
        raise ValueError(f"expected one domain, got {list(domains)}")
    domain_name, domain = next(iter(domains.items()))
    original_valid = list(domain["valid_ids"])
    train_ids = list(domain["train_ids"])
    if episode_count < len(original_valid):
        selected = original_valid[:episode_count]
        supplemental = []
    else:
        needed = episode_count - len(original_valid)
        if needed > len(train_ids):
            raise ValueError("requested episode count exceeds split inventory")
        supplemental = sorted(
            train_ids,
            key=lambda episode_id: hashlib.sha256(
                f"{salt}:{episode_id}".encode()
            ).hexdigest(),
        )[:needed]
        selected = original_valid + supplemental
    if len(selected) != episode_count or len(set(selected)) != episode_count:
        raise ValueError("episode selection is not exact and unique")
    return {
        "schema_version": 1,
        "status": "PASS",
        "label": "NON_PROTOCOL_CODEC_DIAGNOSTIC_EPISODES",
        "source_split_sha256": sha256(split_path),
        "selection": {
            "requested_episode_count": episode_count,
            "retained_original_validation_count": min(episode_count, len(original_valid)),
            "supplemental_training_count": len(supplemental),
            "supplemental_method": f"lowest SHA256({salt}:<episode_id>) over original train IDs",
            "warning": "Not a learned-policy validation split when supplemental_training_count is nonzero.",
        },
        "domains": {
            domain_name: {
                "folder_path": domain["folder_path"],
                "valid_count": episode_count,
                "valid_ids": selected,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--episode-count", type=int, default=100)
    parser.add_argument("--selection-salt", default="seed42")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidate_manifest(args.ranking, args.top_k)
    episodes = build_episode_manifest(
        args.split_manifest, args.episode_count, args.selection_salt
    )
    (args.output_dir / "candidates.json").write_text(
        json.dumps(candidates, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "episodes.json").write_text(
        json.dumps(episodes, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
