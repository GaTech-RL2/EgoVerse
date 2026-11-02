#!/usr/bin/env python3
"""Build private, reproducible moving-budget folds from complete episode metadata.

No signed asset URL or private identifier is written into the repository. The
caller supplies the authoritative updated CSV, original/replacement metadata,
and blacklist files. Incomplete CSV coverage fails before any split is written.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

BUDGETS = (24, 32, 48, 64, 72, 96)
SCALES = (4, 8, 16, 32, 64, 128)
TASK = "folding_clothes"
VERSION = "fold_moving_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_ids(path):
    return {
        s.strip()
        for s in Path(path).read_text().splitlines()
        if s.strip() and not s.lstrip().startswith("#")
    }


def read_episodes(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        values = list(csv.DictReader(path.open()))
    else:
        raw = json.loads(path.read_text())
        if isinstance(raw, dict) and "episodes" in raw:
            values = raw["episodes"]
        elif isinstance(raw, dict):
            values = [dict(v, episode_id=k) for k, v in raw.items()]
        else:
            values = raw
    require(isinstance(values, list), f"Invalid episode list: {path}")
    result = []
    for e in values:
        if str(e.get("task_id", "")) != TASK:
            continue
        duration = float(e["duration"] if "duration" in e else e["duration_s"])
        require(math.isfinite(duration) and duration > 0, "Invalid episode duration")
        result.append(
            {
                "episode_id": str(e["episode_id"]),
                "user_id": str(e["user_id"]),
                "duration": duration,
                "task_id": TASK,
            }
        )
    return result


def spaced_indices(length, count):
    require(0 < count <= length, f"Cannot choose {count} distinct items from {length}")
    return (
        [0]
        if count == 1
        else [round(i * (length - 1) / (count - 1)) for i in range(count)]
    )


def holdouts(episodes, extra=False):
    """Stratify on episode ID order (capture chronology), preserving 3 perlevel IDs."""
    ordered = sorted(episodes, key=lambda e: e["episode_id"])
    require(
        len(ordered) >= (14 if extra else 4),
        "Too few episodes for holdouts and training",
    )
    perlevel = [ordered[i]["episode_id"] for i in spaced_indices(len(ordered), 3)]
    if not extra:
        return perlevel, []
    remaining = [e for e in ordered if e["episode_id"] not in perlevel]
    matched = [remaining[i]["episode_id"] for i in spaced_indices(len(remaining), 10)]
    return perlevel, matched


def load_sources(
    main_paths, replacement_paths, csv_path, banned_episode_path, banned_user_path
):
    banned_episodes = read_ids(banned_episode_path)
    banned_users = read_ids(banned_user_path)
    require(
        bool(banned_episodes) and bool(banned_users),
        "Blacklist files must not be empty",
    )
    rows = [r for r in csv.DictReader(Path(csv_path).open()) if r["task_id"] == TASK]
    require(rows, "Authoritative CSV has no folding_clothes operators")
    table = {r["user_id"]: r for r in rows}
    require(len(table) == len(rows), "Duplicate folding operator in authoritative CSV")
    require(
        not (set(table) & banned_users),
        "Authoritative CSV still includes a banned user",
    )
    merged, replacements, source_counts = {}, set(), []
    for kind, paths in (("main", main_paths), ("replacement", replacement_paths)):
        for path in paths:
            rows = read_episodes(path)
            source_counts.append(
                {
                    "kind": kind,
                    "path": str(Path(path).resolve()),
                    "sha256": digest(path),
                    "folding_episodes": len(rows),
                }
            )
            for e in rows:
                eid = e["episode_id"]
                if eid in merged:
                    old = merged[eid]
                    same_identity = all(
                        old[k] == e[k] for k in ("episode_id", "user_id", "task_id")
                    )
                    require(
                        same_identity
                        and abs(old["duration"] - e["duration"]) <= 0.010001,
                        f"Conflicting duplicate episode {eid}",
                    )
                    # Keep precise converted duration when replacement JSON rounds seconds.
                else:
                    merged[eid] = e
                if kind == "replacement":
                    replacements.add(eid)
    excluded = [
        e
        for e in merged.values()
        if e["episode_id"] in banned_episodes or e["user_id"] in banned_users
    ]
    eligible = {
        eid: e
        for eid, e in merged.items()
        if eid not in banned_episodes
        and e["user_id"] not in banned_users
        and e["user_id"] in table
    }
    groups = collections.defaultdict(list)
    for e in eligible.values():
        groups[e["user_id"]].append(e)
    coverage = []
    for uid, row in sorted(table.items()):
        es = groups[uid]
        hours = math.fsum(e["duration"] for e in es) / 3600
        expected = int(row["episodes"])
        expected_hours = float(row["hours"])
        coverage.append(
            {
                "operator": uid,
                "expected_episodes": expected,
                "actual_episodes": len(es),
                "expected_hours": expected_hours,
                "actual_hours": hours,
                "complete": len(es) == expected
                and abs(hours - expected_hours) <= 0.010001,
            }
        )
    # Decimal CSV hours are rounded; permit one hundredth-hour metadata rounding.
    report = {
        "version": VERSION,
        "sources": source_counts,
        "operator_csv_sha256": digest(csv_path),
        "banned_episodes_sha256": digest(banned_episode_path),
        "banned_users_sha256": digest(banned_user_path),
        "expected_operators": len(table),
        "expected_episodes": sum(int(r["episodes"]) for r in table.values()),
        "available_operators": sum(bool(groups[u]) for u in table),
        "available_episodes": len(eligible),
        "actual_hours": math.fsum(e["duration"] for e in eligible.values()) / 3600,
        "replacement_folding_episodes": len(replacements),
        "replacement_eligible_episodes": len(replacements & eligible.keys()),
        "excluded_banned_folding_episodes": len(excluded),
        "non_authoritative_folding_episodes": sum(
            e["user_id"] not in table for e in merged.values()
        ),
        "coverage": coverage,
        "complete": all(r["complete"] for r in coverage),
    }
    return eligible, table, banned_episodes, banned_users, report


def identity(uid, table):
    loc = table[uid]["location_group"].strip()
    return "user:" + uid if not loc or loc.lower() == "unknown" else "location:" + loc


def waterfill(capacities, budget_seconds):
    """Most balanced feasible continuous allocation; saturated small operators stay capped."""
    require(
        math.fsum(capacities) + 1e-7 >= budget_seconds,
        "Total post-holdout capacity cannot meet requested budget",
    )
    low, high = 0.0, max(capacities)
    for _ in range(80):
        mid = (low + high) / 2
        if math.fsum(min(c, mid) for c in capacities) < budget_seconds:
            low = mid
        else:
            high = mid
    return [min(c, high) for c in capacities]


def choose_prefixes(sequences, allocations, budget_seconds, previous):
    """Nearest duration prefixes, then total-error improvement preserving budget nesting."""
    cumulative = [[0.0] for _ in sequences]
    for sums, es in zip(cumulative, sequences):
        for e in es:
            sums.append(sums[-1] + e["duration"])
    counts = []
    for sums, target, old in zip(cumulative, allocations, previous):
        p = bisect.bisect_left(sums, target)
        candidates = {
            max(1, old, min(len(sums) - 1, p)),
            max(1, old, min(len(sums) - 1, p - 1)),
        }
        counts.append(min(candidates, key=lambda i: (abs(sums[i] - target), i)))
    total = math.fsum(sums[n] for sums, n in zip(cumulative, counts))
    # A single neighboring prefix move must improve aggregate error AND stay
    # within one source episode of the operator's waterfilled target.
    while True:
        best = None
        for i, (sums, n, target, old, es) in enumerate(
            zip(cumulative, counts, allocations, previous, sequences)
        ):
            tolerance = max(e["duration"] for e in es)
            for new in (n - 1, n + 1):
                if not max(1, old) <= new < len(sums):
                    continue
                if abs(sums[new] - target) > tolerance + 1e-7:
                    continue
                new_total = total + sums[new] - sums[n]
                if abs(new_total - budget_seconds) + 1e-7 < abs(total - budget_seconds):
                    candidate = (abs(new_total - budget_seconds), i, new, new_total)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            break
        _, i, new, total = best
        counts[i] = new
    return counts, [sums[n] for sums, n in zip(cumulative, counts)]


def create_design(episodes, table, coverage, seed):
    groups = collections.defaultdict(list)
    for e in episodes.values():
        groups[e["user_id"]].append(e)
    identity_groups = collections.defaultdict(list)
    for uid in groups:
        identity_groups[identity(uid, table)].append(uid)
    representatives, decisions = [], []
    for group, users in sorted(identity_groups.items()):
        ranked = sorted(
            users, key=lambda u: (-math.fsum(e["duration"] for e in groups[u]), u)
        )
        representatives.append(ranked[0])
        if len(ranked) > 1:
            decisions.append(
                {
                    "identity_group": group,
                    "representative": ranked[0],
                    "excluded_alternative_ids": ranked[1:],
                    "reason": "Shared location is a conservative identity constraint, not proof of identity.",
                }
            )
    capacities3, capacities13, perlevel = {}, {}, {}
    for uid in representatives:
        if len(groups[uid]) < 4:
            continue
        held, _ = holdouts(groups[uid])
        perlevel[uid] = held
        capacities3[uid] = math.fsum(
            e["duration"] for e in groups[uid] if e["episode_id"] not in held
        )
        if len(groups[uid]) >= 14:
            held, extra = holdouts(groups[uid], True)
            capacities13[uid] = math.fsum(
                e["duration"]
                for e in groups[uid]
                if e["episode_id"] not in held + extra
            )
    core = sorted(capacities13, key=lambda u: (-capacities13[u], u))[:4]
    require(len(core) == 4, "Insufficient core operators after 13 held-out episodes")
    remaining = sorted(
        (u for u in capacities3 if u not in core), key=lambda u: (-capacities3[u], u)
    )
    ordered = core + remaining[:124]
    require(
        len(ordered) == 128, "Fewer than 128 independent eligible training operators"
    )
    matched = {u: holdouts(groups[u], True)[1] for u in core}
    perlevel = {u: perlevel[u] for u in ordered}
    reserved = {
        eid for es in list(matched.values()) + list(perlevel.values()) for eid in es
    }
    train_sequences = {}
    for uid in ordered:
        es = [e for e in groups[uid] if e["episode_id"] not in reserved]
        train_sequences[uid] = sorted(
            es,
            key=lambda e: hashlib.sha256(
                f"{seed}/{uid}/{e['episode_id']}".encode()
            ).hexdigest(),
        )
    unseen_candidates = [
        u for u in representatives if u not in ordered and len(groups[u]) >= 20
    ]
    # Choose unseen independently of training rank, among representatives with enough episodes.
    unseen_candidates.sort(
        key=lambda u: hashlib.sha256(f"{seed}/unseen/{u}".encode()).hexdigest()
    )
    unseen_operators = unseen_candidates[:20]
    if not unseen_operators:
        unseen_operators = sorted(
            (u for u in representatives if u not in ordered),
            key=lambda u: (-len(groups[u]), u),
        )[:20]
    require(unseen_operators, "No independent unseen operators remain")
    unseen = {
        u: [
            sorted(groups[u], key=lambda e: e["episode_id"])[i]["episode_id"]
            for i in spaced_indices(len(groups[u]), min(20, len(groups[u])))
        ]
        for u in unseen_operators
    }
    design = {
        "version": VERSION,
        "seed": seed,
        "coverage": coverage,
        "operator_order": ordered,
        "unseen_operators": unseen_operators,
        "identity_by_operator": {
            u: identity(u, table) for u in ordered + unseen_operators
        },
        "identity_decisions": decisions,
        "seen_matched_by_operator": matched,
        "seen_perlevel_by_operator": perlevel,
        "unseen_by_operator": unseen,
        "episodes": episodes,
        "budgets": {},
        "infeasible": [],
        "selection": "Known location groups retain maximum raw-duration representative; unknown IDs remain independent. Core four ranked by post-13-holdout capacity, next124 by post-3-holdout capacity. Episode holds span ID chronology; training episodes use SHA256(seed/operator/episode) prefixes. Equal allocations use capped waterfill if capacity-limited; whole-episode prefixes are nested across budgets.",
        "unseen_shortfall": {
            "requested_operators": 20,
            "actual_operators": len(unseen_operators),
            "requested_episodes": 400,
            "actual_episodes": sum(map(len, unseen.values())),
        },
    }
    splits = {}
    fixed_matched = [e for u in core for e in matched[u]]
    fixed_unseen = [e for u in unseen_operators for e in unseen[u]]
    for n in SCALES:
        users = ordered[:n]
        sequences = [train_sequences[u] for u in users]
        capacities = [math.fsum(e["duration"] for e in es) for es in sequences]
        previous = [0] * n
        video_users = [users[i] for i in spaced_indices(n, min(5, n))]
        for budget in BUDGETS:
            name = f"fold_moving_b{budget}_op{n}"
            if math.fsum(capacities) < budget * 3600 - 1e-7:
                design["infeasible"].append(
                    {
                        "name": name,
                        "requested_hours": budget,
                        "maximum_hours": math.fsum(capacities) / 3600,
                    }
                )
                continue
            allocations = waterfill(capacities, budget * 3600)
            counts, actual = choose_prefixes(
                sequences, allocations, budget * 3600, previous
            )
            previous = counts
            train = [
                e["episode_id"]
                for es, count in zip(sequences, counts)
                for e in es[:count]
            ]
            splits[name] = {
                "train": train,
                "seen_matched": fixed_matched,
                "seen_perlevel": [e for u in users for e in perlevel[u]],
                "unseen": fixed_unseen,
                "operators": users,
                "video_seen_matched": [matched[u][0] for u in core],
                "video_seen_perlevel": [perlevel[u][0] for u in video_users],
                "video_unseen": [
                    unseen[unseen_operators[i]][0]
                    for i in spaced_indices(
                        len(unseen_operators), min(5, len(unseen_operators))
                    )
                ],
                "budget_hours": budget,
                "operator_count": n,
                "version": VERSION,
            }
            design["budgets"][name] = {
                "target_hours": budget,
                "actual_hours": math.fsum(actual) / 3600,
                "operator_count": n,
                "episodes": len(train),
                "balanced_possible": min(capacities) >= budget * 3600 / n,
                "capacity_hours": dict(zip(users, [c / 3600 for c in capacities])),
                "allocated_target_hours": dict(
                    zip(users, [c / 3600 for c in allocations])
                ),
                "actual_operator_hours": dict(zip(users, [c / 3600 for c in actual])),
                "max_episode_hours": max(e["duration"] for es in sequences for e in es)
                / 3600,
            }
    return design, splits


def validate(design, splits, banned_episodes, banned_users):
    require(design["coverage"]["complete"], "Authoritative CSV coverage is incomplete")
    require(
        not design["infeasible"], f"Infeasible configurations: {design['infeasible']}"
    )
    expected = {f"fold_moving_b{b}_op{n}" for b in BUDGETS for n in SCALES}
    require(set(splits) == expected, "Expected all 36 split manifests")
    order, unseen_users = design["operator_order"], design["unseen_operators"]
    require(
        len(order) == len(set(order)) == 128, "Expected 128 unique ordered operators"
    )
    require(
        not set(order) & (set(unseen_users) | banned_users),
        "Training operators overlap banned/unseen",
    )
    require(not set(unseen_users) & banned_users, "Banned unseen operator")
    identities = design["identity_by_operator"]
    require(
        len({identities[u] for u in order + unseen_users}) == len(order + unseen_users),
        "Identity collision",
    )
    eps = design["episodes"]
    for n in SCALES:
        previous, fixed_level = set(), None
        for budget in BUDGETS:
            name = f"fold_moving_b{budget}_op{n}"
            s = splits[name]
            require(s["operators"] == order[:n], f"Operator identities changed: {name}")
            sets = {
                k: set(s[k])
                for k in ("train", "seen_matched", "seen_perlevel", "unseen")
            }
            for k, values in sets.items():
                require(len(values) == len(s[k]), f"Duplicate episode in {name}/{k}")
                require(not values & banned_episodes, f"Banned episode in {name}/{k}")
                require(
                    all(
                        e in eps
                        and eps[e]["task_id"] == TASK
                        and eps[e]["user_id"] not in banned_users
                        for e in values
                    ),
                    f"Invalid episode in {name}/{k}",
                )
            for a in sets:
                for b in sets:
                    if a != b:
                        require(
                            not sets[a] & sets[b], f"Episode leakage: {name}/{a}/{b}"
                        )
            require(
                {eps[e]["user_id"] for e in sets["train"]} == set(order[:n]),
                "Not every training operator contributes",
            )
            require(len(sets["seen_matched"]) == 40, "seen_matched size must be 40")
            require(
                len(sets["seen_perlevel"]) == 3 * n, "seen_perlevel size must be 3*N"
            )
            require(
                sets["seen_matched"]
                == {
                    e
                    for ids in design["seen_matched_by_operator"].values()
                    for e in ids
                },
                "Matched set changed",
            )
            require(
                sets["unseen"]
                == {e for ids in design["unseen_by_operator"].values() for e in ids},
                "Unseen set changed",
            )
            require(
                sets["seen_perlevel"]
                == {
                    e for u in order[:n] for e in design["seen_perlevel_by_operator"][u]
                },
                "Perlevel set changed",
            )
            require(
                previous <= sets["train"],
                "Training episodes are not nested across budgets",
            )
            previous = sets["train"]
            if fixed_level is not None:
                require(
                    fixed_level == sets["seen_perlevel"],
                    "Perlevel differs across budgets",
                )
            fixed_level = sets["seen_perlevel"]
            actual = math.fsum(eps[e]["duration"] for e in sets["train"]) / 3600
            metadata = design["budgets"][name]
            require(
                abs(actual - metadata["actual_hours"]) < 1e-7,
                "Recorded actual hours mismatch",
            )
            # Only whole-episode granularity is allowed; capacities were checked before sampling.
            require(
                abs(actual - budget) <= metadata["max_episode_hours"] + 1e-7,
                "Budget error exceeds one episode",
            )
            for key, count in (
                ("seen_matched", 4),
                ("seen_perlevel", min(n, 5)),
                ("unseen", min(5, len(unseen_users))),
            ):
                pins = s["video_" + key]
                require(
                    len(pins) == count
                    and len(set(pins)) == count
                    and set(pins) <= sets[key],
                    "Invalid video pins",
                )
                require(
                    len({eps[e]["user_id"] for e in pins}) == count,
                    "Video operators are not distinct",
                )
            for uid in order[:n]:
                h = (
                    math.fsum(
                        eps[e]["duration"]
                        for e in sets["train"]
                        if eps[e]["user_id"] == uid
                    )
                    / 3600
                )
                require(
                    abs(h - metadata["actual_operator_hours"][uid]) < 1e-7,
                    "Per-operator duration mismatch",
                )
                require(
                    h <= metadata["capacity_hours"][uid] + 1e-7,
                    "Operator exceeds available capacity",
                )
    return {
        "configurations": 36,
        "operator_nesting": True,
        "cross_budget_identity": True,
        "budget_episode_nesting": True,
        "validation_disjoint": True,
        "identity_isolation": True,
        "actual_hours": {k: v["actual_hours"] for k, v in design["budgets"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--replacement-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--operator-csv", type=Path)
    parser.add_argument("--banned-episodes", type=Path, required=True)
    parser.add_argument("--banned-users", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.private_dir.resolve()
    repo = Path(__file__).resolve().parents[2]
    require(
        not root.is_relative_to(repo),
        "Private manifests must be outside the repository",
    )
    if args.validate_only:
        design = json.loads((root / "moving_opscale_design.json").read_text())
        splits = {
            p.stem: json.loads(p.read_text())
            for p in (root / "splits").glob("fold_moving_b*_op*.json")
        }
        checks = validate(
            design, splits, read_ids(args.banned_episodes), read_ids(args.banned_users)
        )
    else:
        require(
            bool(args.main_manifest) and args.operator_csv is not None,
            "Supply main manifest and authoritative operator CSV",
        )
        eps, table, banned_eps, banned_users, coverage = load_sources(
            args.main_manifest,
            args.replacement_manifest,
            args.operator_csv,
            args.banned_episodes,
            args.banned_users,
        )
        write_json(root / "moving_opscale_coverage.json", coverage)
        print(
            json.dumps(
                {
                    k: coverage[k]
                    for k in (
                        "complete",
                        "expected_operators",
                        "available_operators",
                        "expected_episodes",
                        "available_episodes",
                        "actual_hours",
                        "replacement_folding_episodes",
                    )
                }
            )
        )
        if args.audit_only:
            return 0 if coverage["complete"] else 2
        require(
            coverage["complete"],
            "Incomplete authoritative CSV coverage; see moving_opscale_coverage.json. No split manifests generated.",
        )
        design, splits = create_design(eps, table, coverage, args.seed)
        if design["infeasible"]:
            write_json(root / "moving_opscale_infeasible.json", design["infeasible"])
        checks = validate(design, splits, banned_eps, banned_users)
        # All checks complete before any executable split manifest is written.
        write_json(root / "moving_opscale_design.json", design)
        for name, split in splits.items():
            write_json(root / "splits" / (name + ".json"), split)
    write_json(root / "moving_opscale_validation.json", checks)
    print(json.dumps(checks, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"Experiment integrity check failed: {exc}") from exc
