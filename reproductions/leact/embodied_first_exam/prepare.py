"""Pin EFE task identities and reserve the entire official core for evaluation.

This prepares a research split; upstream publishes an evaluation core, not a
supervised training split. No simulator, oracle solution, or model is executed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib


REVISION = "b88c450c46de8a96bba83bf9606b66396cccfff5"
HF_REVISION = "3baca489fc4fcba4884c30253cd87b5d73039ac5"
SPLIT_SEED = "efe-multitask-20261025-v1"
MODES = ("standard-realtime", "standard", "privileged")
PILOT_SUITES = ("dexhand", "metaworldplus", "robocasa", "embodied-safety-exam")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def base_name(name, mode):
    if mode not in MODES or not name.endswith("-" + mode):
        raise ValueError(f"Unrecognized task mode/name: {name}, {mode}")
    return name[: -len(mode) - 1]


def inventory(root):
    core_file = root / "docs/core_set.toml"
    declarations = tomllib.loads(core_file.read_text())["task"]
    core = {r["task"]: r["default_mode"] for r in declarations}
    if len(core) != len(declarations):
        raise ValueError("Duplicate official core task")
    rows = []
    for path in sorted((root / "tasks").glob("*/*/task.toml")):
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode())
        meta = data["metadata"]
        name, mode = path.parent.name, meta["mode"]
        base = base_name(name, mode)
        digest = (path.parent / "DIGEST").read_text().strip()
        if not re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", digest):
            raise ValueError(f"Invalid upstream DIGEST: {name}")
        suite = path.parent.parent.name
        gpu = meta["resources"]["gpu_class"]
        # README still lists RoboARC as RTX-only, while current task metadata
        # says 'any'. Do not silently count it as H100-compatible before a
        # rendering smoke test of its pinned Nyx image.
        conflict = suite == "roboarc" and gpu != "rtx"
        rows.append(dict(
            task=name, base_task=base, suite=suite, family=meta["family"],
            path=str(path.parent.relative_to(root)), mode=mode,
            official_core=base in core,
            official_default=base in core and core[base] == mode,
            upstream_digest=digest.removeprefix("sha256:"),
            task_toml_sha256=sha(raw),
            seeds=meta.get("seeds", {}),
            cutoff_seconds=data["agent"]["timeout_sec"],
            reset_budget=meta.get("reset_budget"), slowdown=meta.get("slowdown"),
            network=meta.get("network", "allowlist"),
            protocol=meta.get("protocol"), simulator=meta.get("simulator"),
            oracle=meta.get("oracle"), validated=meta.get("validated", meta.get("oracle") == "full"),
            resources=meta["resources"], declared_gpu_class=gpu,
            hardware_status=("renderer_preflight_required" if conflict else
                             "rtx_required" if gpu == "rtx" else "cpu_or_h100_candidate"),
        ))
    actual = {r["base_task"]: r["mode"] for r in rows if r["official_default"]}
    if actual != core:
        raise ValueError("Checkout does not contain every official core task in its default mode")
    return rows, core, sha(core_file.read_bytes())


def research_split(rows, core):
    """Split whole task identities; never let a mode twin cross the core boundary."""
    variants = defaultdict(list)
    for row in rows:
        variants[row["base_task"]].append(row)
    train_candidates = []
    for base, group in sorted(variants.items()):
        if base in core:
            continue
        # Match the standard observation boundary used by the official core.
        choices = [r for r in group if r["mode"] in {"standard", "standard-realtime"}]
        if not choices:
            raise ValueError(f"Non-core task has no standard mode: {base}")
        train_candidates.append(min(choices, key=lambda r: (r["mode"] != "standard", r["task"])))
    families = defaultdict(set)
    for row in train_candidates:
        families[row["suite"]].add(row["family"])
    development = set()
    for suite, names in sorted(families.items()):
        ordered = sorted(names, key=lambda f: sha(f"{SPLIT_SEED}/{suite}/{f}".encode()))
        # Hold out about 20% of non-core families where two or more exist.
        count = max(1, round(len(ordered) * .2)) if len(ordered) > 1 else 0
        development.update((suite, f) for f in ordered[:count])
    assignments = {r["base_task"]: ("development" if (r["suite"], r["family"]) in development
                                      else "train_candidate") for r in train_candidates}
    selected = {r["task"] for r in train_candidates}
    for row in rows:
        row["partition"] = "evaluation_only" if row["base_task"] in core else assignments[row["base_task"]]
        row["selected_mode"] = row["official_default"] if row["official_core"] else row["task"] in selected
    split = {
        "official_core_evaluation": sorted(r["task"] for r in rows if r["official_default"]),
        "protected_core_all_modes": sorted(r["task"] for r in rows if r["official_core"]),
        "train_candidates": sorted(r["task"] for r in rows if r["selected_mode"] and r["partition"] == "train_candidate"),
        "development": sorted(r["task"] for r in rows if r["selected_mode"] and r["partition"] == "development"),
        "split_seed": SPLIT_SEED,
        "scope": "Research train/development split of non-core tasks; official core is evaluation-only",
        "scene_collision_audit": "Required before corpus admission; task-name separation alone does not prove scene independence",
    }
    check_partitions(rows, split)
    return split


def check_partitions(rows, split):
    by_name = {r["task"]: r for r in rows}
    roles = [split[k] for k in ("official_core_evaluation", "train_candidates", "development")]
    sets = [{by_name[t]["base_task"] for t in role} for role in roles]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("A task or alternate-mode twin crosses a split boundary")
    protected = {by_name[t]["base_task"] for t in split["protected_core_all_modes"]}
    if protected != sets[0] or protected & (sets[1] | sets[2]):
        raise ValueError("Official core mode twins are not all protected")
    train_families = {(by_name[t]["suite"], by_name[t]["family"]) for t in roles[1]}
    dev_families = {(by_name[t]["suite"], by_name[t]["family"]) for t in roles[2]}
    if train_families & dev_families:
        raise ValueError("A family crosses the train/development boundary")


def freeze(root, output):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected pinned EFE revision {REVISION}, got {revision}")
    # Check staged and working metadata, including sparse-checkout files, against
    # the pin. Reading public metadata does not require checking out solutions.
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--",
                                      "docs", "tasks", "README.md"], cwd=root, text=True)
    if changed.strip():
        raise ValueError("Upstream metadata differs from the pinned commit")
    rows, core, core_sha = inventory(root)
    split = research_split(rows, core)
    pilot = {role: [] for role in ("train_candidates", "development")}
    for role in pilot:
        names = set(split[role])
        for suite in PILOT_SUITES:
            candidates = [r for r in rows if r["task"] in names and r["suite"] == suite
                          and r["oracle"] == "full" and r["declared_gpu_class"] == "none"]
            candidates.sort(key=lambda r: sha(f"{SPLIT_SEED}/pilot/{r['task']}".encode()))
            pilot[role].extend(r["task"] for r in candidates[:2])
    split["initial_multitask_pilot"] = pilot
    inventory_bytes, split_bytes = encode(rows), encode(split)
    evaluation = [r for r in rows if r["official_default"]]
    manifest = dict(
        schema_version=1, upstream_revision=revision, hf_assets_revision=HF_REVISION,
        upstream="https://github.com/mll-lab-nu/embodied-first-exam",
        hf_assets="https://huggingface.co/datasets/MLL-Lab/embodied-first-exam-public",
        core_set_sha256=core_sha, inventory_sha256=sha(inventory_bytes), splits_sha256=sha(split_bytes),
        task_directories=len(rows), unique_tasks=len({r["base_task"] for r in rows}),
        suites=len({r["suite"] for r in rows}), official_core_tasks=len(core),
        protected_core_mode_directories=len(split["protected_core_all_modes"]),
        train_candidate_tasks=len(split["train_candidates"]), development_tasks=len(split["development"]),
        core_modes=dict(Counter(r["mode"] for r in evaluation)),
        core_declared_gpu=dict(Counter(r["declared_gpu_class"] for r in evaluation)),
        core_hardware_status=dict(Counter(r["hardware_status"] for r in evaluation)),
        core_single_seed_tasks=sum(isinstance(r["seeds"].get("allowed"), list) and
                                   len(r["seeds"]["allowed"]) == 1 for r in evaluation),
        initial_multitask_pilot=pilot,
        readiness="Metadata and research split frozen; simulator and corpus validation pending",
        notes=[
            "The HF repository contains assets, not demonstrations or an official supervised training split.",
            "No core task, seed, alternate-mode solution, or evaluation feedback may enter training or checkpoint selection.",
            "All core tasks at this revision allow a single environment seed; changing inference seeds is not unseen-scene evaluation.",
            "Non-core candidates are upstream-excluded tasks, not an IID sample. Report suite and family overlap with evaluation.",
            "A core task's model-generated evaluation traces stay evaluation-only, even if the teacher succeeds.",
            "Hardware metadata does not establish successful execution. RoboARC has conflicting README/task GPU declarations.",
            "The declared HF revision is the asset repository snapshot; each task's own image and per-file asset pins take precedence.",
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [("inventory.json", inventory_bytes), ("splits.json", split_bytes),
                       ("manifest.json", encode(manifest))]:
        path = output / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace frozen preparation artifact {path}")
        path.write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.reference.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
