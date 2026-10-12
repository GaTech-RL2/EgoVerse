"""Fixed-budget operator scaling on clean mono clothing-folding episodes.

Four rungs, 4 then 8 then 12 then 16 operators, one total hour budget. The
operator list is nested: each larger rung keeps the previous operators and
adds more, and takes a subset of the episodes they contributed before.

The op4 rung is a fixed operator set (the four richest folding_clothes
operators). The budget is the most those four can contribute equally, about
38 h, after holding out one episode from each of them. That holdout is the
seen-operator val on every rung, because every larger rung still contains
those four. Operators past the 16 are the unseen-operator val, also shared.
On the 4, 8, and 12 rungs every operator contributes budget/k. The 15th
and 16th operators do not have budget/16 hours, so on the 16 rung they
contribute everything they recorded and the other 14 share the remainder.

    python egomimic/scripts/data_download/fixed_budget_opscale.py plan \\
        --master 'egomimic/hydra_configs/mono_20260929_master (1).json' \\
        --manifest egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json

    python egomimic/scripts/data_download/fixed_budget_opscale.py download \\
        --master 'egomimic/hydra_configs/mono_20260929_master (1).json' \\
        --manifest egomimic/hydra_configs/experiments/fixed_budget_opscale/manifest.json \\
        --dest /workspace/datasets/fixed_budget_opscale
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TIERS = (4, 8, 12, 16)
TASK_ID = "folding_clothes"
FOLD_TASK_IDS = {
    "folding_clothes",
    "fold_clothes",
    "folding_shirt",
    "folding_t-shirt",
    "folding_t_shirt",
    "folding_t_shirts",
    "folding_tshirts",
    "folding_t-shirts",
    "folding_laundry",
    "folding_pants",
    "folding_socks",
    "folding_trousers",
}
PIN_TASK_IDS = {"folding_laundry", "folding_pants", "folding_socks", "folding_trousers"}
# Non-folding actions are contamination even when task_id says folding_clothes.
# An ironing board used as a work surface is not an ironing action. Stacking
# and placing folded items are part of folding.
NON_FOLDING_ACTIONS = {
    "hanging": r"\bhang(?:s|ing|ed|ung)?\b|\bhangers?\b|\bgarment rack\b",
    "ironing": r"\b(?:iron|irons|ironed|ironing)\s+(?:the\s+|a\s+|an\s+|clothes\b|clothing\b|laundry\b|shirt\b|shirts\b|t-?shirts?\b|pants\b|garments?\b)",
    "sorting": r"\bsort(?:s|ing|ed)\s+(?:through|clothes|laundry|garments|items)|\bsorting and folding\b",
    "packing": r"\bpack(?:s|ing|ed)?\b|\bsuitcase\b",
    "organizing": r"\borganiz(?:e|es|ing|ed)\s+(?:the\s+|a\s+|various\s+|clothes|clothing|laundry|items|them|garments)",
    "washing": r"\bwash(?:es|ing|ed)?\b|\blaundry machine\b",
    "buttoning": r"\bbutton(?:s|ing|ed)\b",
    "cleaning": r"\bcleaning\b|\bcleans\b|\bcleaned\s+(?:the|a|an|their|your)\s+(?:clothes|clothing|laundry|garment|shirt|fabric)\b|\bclean\s+(?:the|a|an)\s+(?:clothes|clothing|laundry|garment|shirt|fabric)\b|\blint roller\b",
    "spraying": r"\bspray(?:s|ing|ed)?\b",
    "other_action": r"\b(?:adjust(?:s|ing|ed)?|repair(?:s|ing|ed)?|sew(?:s|ing|ed)?|inspect(?:s|ing|ed)?|prepare(?:s|ing|ed)?)\b",
}
# op4, richest folding_clothes operators in the supplied mono_20260929_master (1).json.
OP4_OPERATORS = (
    "6a6b180cb93db217f5006d09",
    "6a48f40631f5ca4ad4463db4",
    "6a8653df9afd6cdb9f0d0129",
    "6a44e5b5d99fd801592da5a9",
)
# One episode from each op4 operator, held out of every train tier. Those
# four operators are in every rung, so this is the seen-operator val for
# the whole ladder. Operators past the last rung are the unseen val.
SEEN_EPISODES_PER_CORE_OPERATOR = 1
ADDED_SEEN_VAL_TIERS = (8, 12, 16)
ADDED_SEEN_VAL_OPERATORS_PER_TIER = 4
ADDED_SEEN_VAL_EPISODES_PER_OPERATOR = 30
VAL_SAMPLE_SEED = 42026
DEFAULT_DEST = Path("/workspace/datasets/fixed_budget_opscale")
ASSET_NAMES = {
    "video_1": "video.mp4",
    "hand_pose": "hand_pose.json",
    "body_pose": "body_pose.json",
    "pelvis_pose": "pelvis_pose.csv",
    "intrinsics": "intrinsics.json",
}


def to_cs(seconds: float) -> int:
    return int(round(float(seconds) * 100))


def classify_fold_episodes(master: dict) -> tuple[list[dict], list[dict]]:
    """Classify clothing-folding candidates from their descriptions.

    Keep only descriptions that mention folding and contain no explicit
    unrelated action. Stacking/placing folded items remains in scope.
    """
    import re

    rows: list[dict] = []
    audit: list[dict] = []
    for episode in master["episodes"]:
        task_id = str(episode.get("task_id") or "").lower()
        if task_id not in FOLD_TASK_IDS:
            continue
        description = str(episode.get("task_description") or "")
        reasons = [
            label
            for label, pattern in NON_FOLDING_ACTIONS.items()
            if re.search(pattern, description, re.IGNORECASE)
        ]
        if not re.search(r"\bfold\w*\b", description, re.IGNORECASE):
            reasons.append("description_does_not_mention_folding")
        clean = not reasons
        audit.append(
            {
                "episode_id": str(episode["episode_id"]),
                "user_id": str(episode["user_id"]),
                "task_id": task_id,
                "task_description": description,
                "classification": "folding_only" if clean else "mixed_or_unrelated",
                "excluded_for": reasons,
            }
        )
        if clean:
            rows.append(
                {
                    "episode_id": str(episode["episode_id"]),
                    "user_id": str(episode["user_id"]),
                    "duration": float(episode["duration"]),
                    "duration_cs": to_cs(episode["duration"]),
                    "task_id": task_id,
                }
            )
    return rows, audit


def fold_episodes(master: dict) -> list[dict]:
    """Return only episodes classified as folding-only."""
    return classify_fold_episodes(master)[0]


def _fill(
    pool: list[dict], target_cs: int, required_ids: set[str] | None = None
) -> list[dict]:
    """Whole episodes summing to at most ``target_cs``, as close as possible.

    Greedy by duration, then one-for-one swaps that close the remainder.
    The leftover is smaller than any episode that was still available.
    """
    items = sorted(pool, key=lambda e: (-e["duration_cs"], e["episode_id"]))
    required_ids = required_ids or set()
    chosen = [e for e in items if e["episode_id"] in required_ids]
    chosen_ids = {e["episode_id"] for e in chosen}
    remaining = target_cs - sum(e["duration_cs"] for e in chosen)
    if remaining < 0:
        raise ValueError("required task episodes exceed this operator's duration cap")
    for episode in items:
        if episode["episode_id"] in chosen_ids:
            continue
        if episode["duration_cs"] <= remaining:
            chosen.append(episode)
            chosen_ids.add(episode["episode_id"])
            remaining -= episode["duration_cs"]

    unused = [episode for episode in items if episode["episode_id"] not in chosen_ids]
    improved = True
    while improved and unused and chosen:
        improved = False
        best: tuple[int, int, int] | None = None
        for chosen_index, current in enumerate(chosen):
            if current["episode_id"] in required_ids:
                continue
            for unused_index, candidate in enumerate(unused):
                gain = candidate["duration_cs"] - current["duration_cs"]
                if gain > 0 and gain <= remaining:
                    if best is None or gain > best[0]:
                        best = (gain, chosen_index, unused_index)
        if best is None:
            break
        gain, chosen_index, unused_index = best
        swapped = unused[unused_index]
        unused[unused_index] = chosen[chosen_index]
        chosen[chosen_index] = swapped
        remaining -= gain
        improved = True
    chosen.sort(key=lambda e: e["episode_id"])
    return chosen


def _episode_record(episode: dict) -> dict:
    return {
        "episode_id": episode["episode_id"],
        "user_id": episode["user_id"],
        "duration": episode["duration"],
        "duration_cs": episode["duration_cs"],
    }


def plan(
    episodes: list[dict],
    op4_operators: tuple[str, ...] = OP4_OPERATORS,
) -> dict:
    if len(op4_operators) != TIERS[0]:
        raise ValueError(f"op4 needs {TIERS[0]} operators, got {len(op4_operators)}")
    by_operator: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_operator[episode["user_id"]].append(episode)
    missing = [user_id for user_id in op4_operators if user_id not in by_operator]
    if missing:
        raise ValueError(f"op4 operators missing from {TASK_ID}: {missing}")
    rest = sorted(
        (user_id for user_id in by_operator if user_id not in op4_operators),
        key=lambda user_id: (
            -sum(e["duration_cs"] for e in by_operator[user_id]),
            user_id,
        ),
    )
    # op4 is the pinned set; later rungs append the next-richest operators.
    ordered_ids = list(op4_operators) + rest
    if len(ordered_ids) < TIERS[-1]:
        raise ValueError(
            f"{TASK_ID} has {len(ordered_ids)} operators; the ladder needs {TIERS[-1]}"
        )
    ranked = [(user_id, by_operator[user_id]) for user_id in ordered_ids]

    # Hold out a reproducible random trajectory sample for the four operators
    # introduced at each larger rung. These episodes are removed from training
    # at that rung and all later rungs, so the seen-val estimates are episode-
    # disjoint from every training set. Keep the four rare task-ID categories
    # pinned in training by sampling only from the other folding episodes.
    rng = random.Random(VAL_SAMPLE_SEED)
    rung_seen_val: dict[str, dict] = {}
    reserved_by_operator: dict[str, set[str]] = defaultdict(set)
    for tier in ADDED_SEEN_VAL_TIERS:
        start = tier - TIERS[0]
        user_ids = ordered_ids[start : start + ADDED_SEEN_VAL_OPERATORS_PER_TIER]
        if len(user_ids) != ADDED_SEEN_VAL_OPERATORS_PER_TIER:
            raise ValueError(f"op{tier} needs four newly added operators")
        selected = []
        for user_id in user_ids:
            candidates = sorted(
                (
                    e
                    for e in by_operator[user_id]
                    if e.get("task_id") not in PIN_TASK_IDS
                ),
                key=lambda e: e["episode_id"],
            )
            needed = ADDED_SEEN_VAL_EPISODES_PER_OPERATOR
            if len(candidates) < needed:
                raise ValueError(
                    f"op{tier} operator {user_id} has only {len(candidates)} eligible "
                    f"episodes for {needed}-episode seen validation"
                )
            sampled = rng.sample(candidates, needed)
            reserved_by_operator[user_id].update(e["episode_id"] for e in sampled)
            selected.extend(sampled)
        rung_seen_val[f"op{tier}"] = {
            "operators": user_ids,
            "episodes_per_operator": ADDED_SEEN_VAL_EPISODES_PER_OPERATOR,
            "sample_seed": VAL_SAMPLE_SEED,
            "episodes": [
                _episode_record(e)
                for e in sorted(selected, key=lambda e: e["episode_id"])
            ],
            "video_episode": max(
                selected, key=lambda e: (e["duration_cs"], e["episode_id"])
            )["episode_id"],
        }

    core = set(op4_operators)
    seen: list[dict] = []
    train_pool: dict[str, list[dict]] = {}
    for user_id, operator_episodes in ranked:
        ordered = sorted(
            operator_episodes, key=lambda e: (-e["duration_cs"], e["episode_id"])
        )
        held_ids = set(reserved_by_operator.get(user_id, set()))
        if user_id in core:
            held = ordered[:SEEN_EPISODES_PER_CORE_OPERATOR]
            seen.extend(held)
            held_ids.update(e["episode_id"] for e in held)
        train_pool[user_id] = [
            e for e in operator_episodes if e["episode_id"] not in held_ids
        ]

    train_operators = [user_id for user_id, _ in ranked[: TIERS[-1]]]
    available = [
        sum(e["duration_cs"] for e in train_pool[user_id])
        for user_id in train_operators
    ]
    # The op4 rung binds the budget. Later equal rungs divide that same total.
    # The last rung keeps the total by letting operators who cannot reach an
    # equal share contribute everything they have.
    equal_tiers = TIERS[:-1]
    last_tier = TIERS[-1]
    binding_tier = min(equal_tiers, key=lambda k: k * available[k - 1])
    marginal_cs = available[binding_tier - 1]
    budget_cs = binding_tier * marginal_cs
    caps = {k: budget_cs // k for k in equal_tiers}
    equal_share = budget_cs // last_tier
    short = [i for i, hours in enumerate(available) if hours < equal_share]
    if short and short != list(range(short[0], last_tier)):
        raise ValueError(
            f"op{last_tier} short operators must be a suffix of the rung, got {short}"
        )
    n_long = last_tier - len(short)
    last_cap = (budget_cs - sum(available[i] for i in short)) // n_long
    if last_cap > caps[equal_tiers[-1]]:
        raise ValueError(
            f"op{last_tier} cap {last_cap} exceeds the previous rung's cap; "
            "the nested episode subset would miss the budget"
        )
    targets = {
        train_operators[i]: available[i] if i in short else last_cap
        for i in range(last_tier)
    }

    per_tier: dict[int, list[tuple[str, list[dict]]]] = {k: [] for k in TIERS}
    for index, user_id in enumerate(train_operators):
        pool = train_pool[user_id]
        # Smallest k is the largest per-operator cap. Each later cap is a
        # subset of that operator's previous episodes.
        for k in TIERS:
            if index >= k:
                continue
            required = {
                episode["episode_id"]
                for episode in pool
                if episode.get("task_id") in PIN_TASK_IDS
            }
            chosen = _fill(
                pool,
                targets[user_id] if k == last_tier else caps[k],
                required_ids=required,
            )
            per_tier[k].append((user_id, chosen))
            pool = chosen

    def pack(chosen_rows: list[tuple[str, list[dict]]]) -> dict:
        episodes_out = [
            _episode_record(episode) for _, chosen in chosen_rows for episode in chosen
        ]
        achieved = sum(e["duration_cs"] for e in episodes_out)
        return {
            "operators": [user_id for user_id, _ in chosen_rows],
            "episodes": episodes_out,
            "achieved_cs": achieved,
            "video_episode": max(
                episodes_out, key=lambda e: (e["duration_cs"], e["episode_id"])
            )["episode_id"],
        }

    tiers = {}
    short_ids = [train_operators[i] for i in short]
    for k in TIERS:
        packed = pack(per_tier[k])
        if k == last_tier:
            packed["cap_cs"] = last_cap
            packed["target_cs"] = budget_cs
            packed["short_operators"] = short_ids
            packed["short_cs"] = {train_operators[i]: available[i] for i in short}
        else:
            packed["cap_cs"] = caps[k]
            packed["target_cs"] = caps[k] * k
            packed["short_operators"] = []
            packed["short_cs"] = {}
        tiers[f"op{k}"] = packed
    # Train video: longest episode that survives into every rung (the last
    # rung's pack of an operator who is also in op4).
    core = set(tiers[f"op{TIERS[0]}"]["operators"])
    shared_pool = [
        episode
        for episode in tiers[f"op{TIERS[-1]}"]["episodes"]
        if episode["user_id"] in core
    ]
    shared_video = max(shared_pool, key=lambda e: (e["duration_cs"], e["episode_id"]))[
        "episode_id"
    ]
    for tier in tiers.values():
        tier["video_episode"] = shared_video

    unseen_operators = [user_id for user_id, _ in ranked[TIERS[-1] :]]
    unseen_episodes = [
        _episode_record(episode)
        for user_id in unseen_operators
        for episode in by_operator[user_id]
    ]
    seen_records = [_episode_record(episode) for episode in seen]

    def pin(rows: list[dict]) -> str | None:
        if not rows:
            return None
        return max(rows, key=lambda e: (e["duration_cs"], e["episode_id"]))[
            "episode_id"
        ]

    op1_episodes = [_episode_record(e) for e in train_pool[op4_operators[0]]]
    op1 = {
        "operators": [op4_operators[0]],
        "episodes": op1_episodes,
        "achieved_cs": sum(e["duration_cs"] for e in op1_episodes),
        "video_episode": max(
            op1_episodes, key=lambda e: (e["duration_cs"], e["episode_id"])
        )["episode_id"],
    }
    return {
        "task_id": TASK_ID,
        "tiers_k": list(TIERS),
        "op1": op1,
        "source_task_ids": sorted(FOLD_TASK_IDS),
        "pinned_task_ids": sorted(PIN_TASK_IDS),
        "seen_episodes_per_core_operator": SEEN_EPISODES_PER_CORE_OPERATOR,
        "metric_sampling": {
            "method": "random_without_replacement",
            "max_trajectory_starts_per_episode": 128,
            "seed": VAL_SAMPLE_SEED,
            "notes": "Fixed across rungs and passes; full predicted trajectory MSE excludes padding; videos do not contribute.",
        },
        "seen_val_sampling": {
            "episodes_per_added_operator": ADDED_SEEN_VAL_EPISODES_PER_OPERATOR,
            "added_operators_per_rung": ADDED_SEEN_VAL_OPERATORS_PER_TIER,
            "rungs": list(ADDED_SEEN_VAL_TIERS),
            "seed": VAL_SAMPLE_SEED,
            "sampling": "seeded random episodes, reserved from all training rungs",
        },
        "op4_operators": list(op4_operators),
        "binding_tier": binding_tier,
        "budget_cs": budget_cs,
        "budget_hours": budget_cs / 100 / 3600,
        "marginal_operator": train_operators[binding_tier - 1],
        "marginal_cs": marginal_cs,
        "operators_ranked": [
            {
                "user_id": user_id,
                "total_cs": sum(e["duration_cs"] for e in operator_episodes),
                "n_episodes": len(operator_episodes),
                "in_train": index < TIERS[-1],
            }
            for index, (user_id, operator_episodes) in enumerate(ranked)
        ],
        "tiers": tiers,
        "seen_val": {
            "episodes": seen_records,
            "video_episode": pin(seen_records),
        },
        "seen_val_by_rung": rung_seen_val,
        "unseen_val": {
            "operators": unseen_operators,
            "episodes": unseen_episodes,
            "video_episode": pin(unseen_episodes),
        },
    }


def summary_lines(spec: dict) -> list[str]:
    lines = [
        f"budget {spec['budget_hours']:.4f} h  "
        f"(op{spec['binding_tier']} x operator {spec['marginal_operator']} "
        f"{spec['marginal_cs'] / 100 / 3600:.4f} h)"
    ]
    for name, tier in spec["tiers"].items():
        per_op = defaultdict(int)
        for episode in tier["episodes"]:
            per_op[episode["user_id"]] += episode["duration_cs"]
        hours = sorted(v / 100 / 3600 for v in per_op.values())
        lines.append(
            f"{name}: {len(tier['operators'])} ops  {len(tier['episodes'])} eps  "
            f"{tier['achieved_cs'] / 100 / 3600:.4f} h "
            f"(target {tier['target_cs'] / 100 / 3600:.4f} h)  "
            f"per-op {hours[0]:.4f}-{hours[-1]:.4f} h  "
            f"video {tier['video_episode']}"
        )
    seen = spec["seen_val"]["episodes"]
    unseen = spec["unseen_val"]["episodes"]
    lines.append(
        f"seen_val: {len(seen)} eps  "
        f"{sum(e['duration_cs'] for e in seen) / 100 / 3600:.4f} h  "
        f"video {spec['seen_val']['video_episode']}"
    )
    for tier, group in spec.get("seen_val_by_rung", {}).items():
        lines.append(
            f"{tier}_seen_val: {len(group['operators'])} ops  "
            f"{len(group['episodes'])} eps  "
            f"{group['episodes_per_operator']} eps/operator  "
            f"video {group['video_episode']}"
        )
    lines.append(
        f"unseen_val: {len(spec['unseen_val']['operators'])} ops  "
        f"{len(unseen)} eps  "
        f"{sum(e['duration_cs'] for e in unseen) / 100 / 3600:.4f} h  "
        f"video {spec['unseen_val']['video_episode']}"
    )
    return lines


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _index_master(master: dict) -> dict[str, dict]:
    return {str(e["episode_id"]): e for e in master["episodes"]}


def _curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest.with_name(dest.name + ".ok")
    if marker.exists() and dest.exists() and dest.stat().st_size > 0:
        return
    cmd = [
        "curl",
        "-fsSL",
        "-A",
        "Mozilla/5.0",
        "-C",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "-o",
        str(dest),
        url,
    ]
    subprocess.run(cmd, check=True)
    if not dest.exists() or dest.stat().st_size <= 0:
        raise RuntimeError(f"empty download: {dest}")
    marker.write_text("ok\n")


def _link(dest_root: Path, folder: str, episode_id: str) -> None:
    folder_dir = dest_root / folder
    folder_dir.mkdir(parents=True, exist_ok=True)
    link = folder_dir / episode_id
    target = Path("..") / "raw" / episode_id
    if link.is_symlink() and link.readlink() == target:
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def download(
    master: dict, spec: dict, dest: Path, workers: int, force: bool = False
) -> None:
    by_id = _index_master(master)
    folders: dict[str, list[dict]] = {
        name: tier["episodes"] for name, tier in spec["tiers"].items()
    }
    if "op1" in spec:
        folders["op1"] = spec["op1"]["episodes"]
    folders["seen_val"] = spec["seen_val"]["episodes"]
    folders["unseen_val"] = spec["unseen_val"]["episodes"]
    for tier, group in spec.get("seen_val_by_rung", {}).items():
        folders[f"{tier}_seen_val"] = group["episodes"]

    needed: dict[str, dict] = {}
    membership: dict[str, list[str]] = defaultdict(list)
    for folder, episodes in folders.items():
        for episode in episodes:
            if str(episode.get("source", "")).startswith("SQL app.episodes"):
                continue  # already-processed SQL stores have no master raw URLs
            episode_id = episode["episode_id"]
            membership[episode_id].append(folder)
            if episode_id not in needed:
                source = by_id.get(episode_id)
                if source is None:
                    raise KeyError(
                        f"{episode_id} is in the manifest but not the master JSON"
                    )
                needed[episode_id] = source

    raw_root = dest / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path, str]] = []
    for episode_id, source in needed.items():
        episode_dir = raw_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "episode_id": episode_id,
            "user_id": source.get("user_id"),
            "duration": source.get("duration"),
            "task_id": source.get("task_id"),
            "task_description": source.get("task_description"),
            "environment_id": source.get("environment_id"),
            "scene_id": source.get("scene_id"),
        }
        (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        for field, filename in ASSET_NAMES.items():
            url = source.get(field)
            if not url:
                raise KeyError(f"{episode_id} is missing {field}")
            asset_path = episode_dir / filename
            if force:
                asset_path.unlink(missing_ok=True)
                asset_path.with_name(asset_path.name + ".ok").unlink(missing_ok=True)
            jobs.append((episode_id, asset_path, url))

    print(
        f"episodes {len(needed)}  files {len(jobs)}  dest {dest}",
        flush=True,
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_curl, url, path): (episode_id, path)
            for episode_id, path, url in jobs
        }
        done = 0
        for future in as_completed(futures):
            episode_id, path = futures[future]
            done += 1
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{episode_id} {path.name}: {exc}")
            if done % 50 == 0 or done == len(jobs):
                print(f"downloaded {done}/{len(jobs)}", flush=True)
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(f"{len(failures)} download(s) failed:\n{preview}")

    for episode_id, folders_for_episode in membership.items():
        for folder in folders_for_episode:
            _link(dest, folder, episode_id)
    print(f"linked episodes under {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan_cmd = sub.add_parser(
        "plan", help="Write the tier manifest from the master JSON"
    )
    plan_cmd.add_argument("--master", type=Path, required=True)
    plan_cmd.add_argument("--manifest", type=Path, required=True)

    download_cmd = sub.add_parser(
        "download", help="Download tier episodes and symlink them"
    )
    download_cmd.add_argument("--master", type=Path, required=True)
    download_cmd.add_argument("--manifest", type=Path, required=True)
    download_cmd.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    download_cmd.add_argument("--workers", type=int, default=16)
    download_cmd.add_argument(
        "--force",
        action="store_true",
        help="redownload selected source assets even when .ok markers exist",
    )

    args = parser.parse_args()
    if args.cmd == "plan":
        master = _load_json(args.master)
        clean_episodes, audit = classify_fold_episodes(master)
        spec = plan(clean_episodes)
        if args.manifest.exists():
            previous = _load_json(args.manifest)
            for external_key in ("unseen_val", "flagship_unseen_val"):
                external = previous.get(external_key, {})
                if str(external.get("source", "")).startswith("SQL app.episodes"):
                    spec[external_key] = external
        spec["classification_policy"] = (
            "Candidate task IDs include folding_clothes/fold_clothes/folding_shirt/folding_t-shirt "
            "variants plus folding_laundry/pants/socks/trousers; include only descriptions "
            "explicitly mentioning folding and with "
            "no explicit hanging, ironing, sorting, packing, organizing, washing, "
            "buttoning, cleaning, spraying, or unrelated action. Stacking/placing folded "
            "items and using an ironing board as a work surface are allowed."
        )
        spec["classification_audit"] = {
            "candidate_count": len(audit),
            "included_count": len(clean_episodes),
            "excluded_count": len(audit) - len(clean_episodes),
            "report_file": "folding_only_classification.json",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(spec, indent=2) + "\n")
        report = {
            "master_json": str(args.master),
            "task_ids_considered": sorted(FOLD_TASK_IDS),
            "classification_policy": spec["classification_policy"],
            "candidate_count": len(audit),
            "included_count": len(clean_episodes),
            "excluded_count": len(audit) - len(clean_episodes),
            "episodes": audit,
        }
        (args.manifest.parent / "folding_only_classification.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"wrote {args.manifest}")
        print(f"wrote {args.manifest.parent / 'folding_only_classification.json'}")
        print("\n".join(summary_lines(spec)))
        return
    spec = _load_json(args.manifest)
    download(_load_json(args.master), spec, args.dest, args.workers, force=args.force)


if __name__ == "__main__":
    main()
