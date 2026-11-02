"""Write the private experiment README and a copy in Downloads from validated splits."""

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private-dir", type=Path, required=True)
    p.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    a = p.parse_args()
    root = a.private_dir.resolve()
    d = json.loads((root / "moving_opscale_design.json").read_text())
    r = json.loads((HERE / "reference_recipe.json").read_text())
    lines = [
        "# Folding clothes moving-budget operator scaling",
        "",
        "Status: data/split/configuration preparation; production launch requires converted-data readiness and the GPU evaluation smoke check. Consult the run-root status.json for live state.",
        "",
        "Compare budgets 24, 32, 48, 64, 72, 96 hours at 4, 8, 16, 32, 64, 128 operators (36 recipes). Use one ordered operator list, identical prefixes across budgets, and nested episode prefixes as each budget grows.",
        "",
        "## Reference training recipe",
        "",
        f"Reference: {r['source_run_url']}",
        "RDT DiT: width 1,280, depth 18, 20 attention heads (~300M DiT; 402.56M trainable total with DINOv3), frozen Qwen, wrist-frame 144-D actions, image history of 2 frames with a 0.0666666667 s gap, proprio history of 4 frames at stride 1, learned history tokens and dropout 0.1. One H100 or H200 per run; batch 64, accumulation 1, bf16, EMA off, compile on. AdamW LR 1e-4, weight decay 0.01, betas (0.9, 0.95), backbone LR scale0.1, gradient clip 1. Seed 42026; 4,000 epochs × 100 steps; WSD: 7,000 warmup / 323,000 stable / 70,000 decay steps.",
        "",
        "Validation metrics every 50 epochs, up to 800 batches per validation set (user override of the reference 500); reference open-loop videos every 200 epochs and at final validation. Checkpoints every 50 epochs, retaining the newest two plus last.ckpt locally. Normalization is training-only quantile, sample_frac=1, max_samples=200000, pool_horizon=false, separately content-keyed per training split.",
        "",
        "## Selection and exclusions",
        "",
        d["selection"],
        "",
        "The supplied updated operator CSV is authoritative for operator identities and location groups. All eight banned users and all supplied banned episode IDs are excluded before selection. Replacement episodes are merged by ID and checked against source metadata. Same-location alternatives are never treated as independent identities; unknown locations retain independent user identities.",
        "",
        f"Full folding coverage: {d['coverage']['available_episodes']} episodes, {d['coverage']['actual_hours']:.6f} hours, {d['coverage']['available_operators']} operators. Source input hashes and exclusion decisions are recorded in moving_opscale_design.json. Two full-manifest episodes were initially missing from the converted R2 index; conversion readiness is checked independently and their IDs remain in their selected manifests.",
        "",
        "Balanced contributions use equal waterfilled targets until an operator reaches its available post-holdout capacity. At 72h/64 operators and 96h/64 operators, some operators cannot reach the mean; remaining hours are allocated equally among unsaturated operators. Operator identities and requested total hours do not change. Whole-episode deterministic prefixes introduce the small deviations below.",
        "",
        "## Exact ordered training operators",
        "",
        "Prefixes of this exact list define 4/8/16/32/64/128 operators:",
        "",
        "```text",
    ]
    lines.extend(f"{i:3d} {u}" for i, u in enumerate(d["operator_order"]))
    lines.extend(
        [
            "```",
            "",
            "## Actual demonstration hours",
            "",
            "| Budget | 4 ops | 8 ops | 16 ops | 32 ops | 64 ops | 128 ops |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for b in (24, 32, 48, 64, 72, 96):
        lines.append(
            "| "
            + str(b)
            + " | "
            + " | ".join(
                f"{d['budgets'][f'fold_moving_b{b}_op{n}']['actual_hours']:.6f}"
                for n in (4, 8, 16, 32, 64, 128)
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Identity decisions",
            "",
            *[
                f"- {x['identity_group']}: retain {x['representative']}; exclude alternative IDs {', '.join(x['excluded_alternative_ids'])}. {x['reason']}"
                for x in d["identity_decisions"]
            ],
            "",
            "Banned user IDs: "
            + ", ".join((root / "inputs/banned_users.txt").read_text().split()),
            "",
            f"Complete banned episode list: `{root/'inputs/banned_episodes.txt'}`.",
            "",
            "Exact per-operator requested/actual hours and capacity are in moving_opscale_design.json → budgets. All 36 passed blacklist, nesting, identity separation, validation disjointness, and hour accounting checks.",
            "",
            "## Validation design",
            "",
            "All validation holds are selected before training sampling. seen_matched=40 fixed episodes (10 each from the core 4); seen_perlevel=3 additional disjoint episodes per training operator (12/24/48/96/192/384); unseen=20 independent operators × 20 episodes = 400, fixed across all recipes. Every validation episode is evaluated using the reference evenly spaced frame selection (up to 800×32 frames per cohort per pass, per the user’s validation-limit override); this evaluates every episode, not every video frame. The coverage callback fails if any episode is omitted. Invalid frame replacement is disabled so a bad selected frame cannot silently change the cohort.",
            "",
            "Metrics: val/seen_matched/Valid/..., val/seen_perlevel/Valid/..., val/unseen/Valid/... using the existing action metrics. Each video event logs 4 matched, 4 or 5 perlevel, and 5 unseen videos (13 at 4 operators, 14 otherwise). Pins are fixed across budgets; perlevel operators are round(i×(N−1)/4), except 4 operators uses all 4. Videos use the archived reference renderer and source-frame timing, labeled videos_open_loop/<cohort>/<operator>/<episode>. A compatibility preparation step creates a canonical local view: legacy left-wrist orientations receive the repository’s established 180-degree local-Z correction exactly once, while already-corrected episodes pass through. Original downloaded episodes and R2 objects are preserved; per-episode provenance records the decision. The reference transform setting remains false because every training input is corrected. Separate per-operator numeric metrics are not added; cohort aggregates use equal frame quotas per episode.",
            "",
            "### Exact holdouts",
            "",
        ]
    )
    for title, key in [
        ("seen_matched", "seen_matched_by_operator"),
        ("seen_perlevel", "seen_perlevel_by_operator"),
        ("unseen", "unseen_by_operator"),
    ]:
        lines.extend([f"#### {title}", "", "```text"])
        for u, ids in d[key].items():
            lines.append(u + ": " + ", ".join(ids))
        lines.extend(["```", ""])
    lines.extend(["### Exact video selections", ""])
    for n in (4, 8, 16, 32, 64, 128):
        s = json.loads((root / "splits" / f"fold_moving_b24_op{n}.json").read_text())
        lines.append(
            f"- op{n}: "
            + "; ".join(
                k + "=" + ",".join(s["video_" + k])
                for k in ("seen_matched", "seen_perlevel", "unseen")
            )
        )
    lines.extend(
        [
            "",
            "## Paths and execution",
            "",
            f"Private root: `{root}`",
            f"YAML directory: `{HERE}`",
            "",
            "The README contains private IDs and is intentionally gitignored; GUIDE.md and the Python/YAML implementation are versioned. The exact manifests stay outside the repository and resolve through ${mecka_private:...}.",
            "",
            "```bash",
            "source /storage/project/r-dxu345-0/acheluva3/EgoVerse-graph/.venv/bin/activate",
            f"cd {HERE.parents[1]}",
            'export PYTHONPATH="$PWD"',
            f"export EGOVERSE_MECKA_PRIVATE_DIR={root}",
            "# Validate all manifests and compose all 36 Hydra configurations:",
            f'python {HERE}/validate_setup.py --private-dir "$EGOVERSE_MECKA_PRIVATE_DIR" --output {root.parent}/gate.json',
            "# After readiness and actual GPU smoke evidence pass, launch through:",
            f"python {HERE}/launch_pace.py --gate {root.parent}/gate.json --readiness {root.parent}/readiness.json --run-root {root.parent} --python /storage/project/r-dxu345-0/acheluva3/EgoVerse-graph/.venv/bin/python --gpu h200 --submit",
            "```",
            "",
            "The launcher refuses missing validation, changed manifests, unverified data, or a dirty/unpinned checkout. It records each submission before sbatch and never silently duplicates jobs. monitor.py records status.json, heartbeat and change-only events.jsonl; Slurm COMPLETED alone does not prove success. Completion requires 400,000 steps, a full checkpoint, all 3 metric cohorts, and 13/14 actual W&B videos.",
            "",
        ]
    )
    out = HERE / "README.md"
    out.write_text("\n".join(lines))
    out.chmod(0o600)
    a.downloads.mkdir(parents=True, exist_ok=True)
    dest = a.downloads / "fold_clothes_moving_opscale_README.md"
    shutil.copyfile(out, dest)
    dest.chmod(0o600)
    print(out)
    print(dest)


if __name__ == "__main__":
    main()
