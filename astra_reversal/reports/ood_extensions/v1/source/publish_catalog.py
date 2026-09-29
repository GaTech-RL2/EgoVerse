"""Publish the validated extension catalog while retaining definition-time bytes."""

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--ood-root", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.validation / "receipt.json"
    receipt = read(receipt_path)
    definition = OUTPUT / "definition_manifest.json"
    if not definition.exists():
        assert sha(OUTPUT / "manifest.json") == receipt["generator_manifest_sha256"]
        shutil.copyfile(OUTPUT / "manifest.json", definition)
    assert sha(definition) == receipt["generator_manifest_sha256"]
    manifest = read(definition)
    assert receipt["status"] == "passed" and receipt["complete_task_coverage"]
    assert receipt["task_count"] == 16 and receipt["reset_checks"] == 32
    assert receipt["validator_sha256"] == sha(
        ROOT / "astra_reversal/validate_ood_extensions.py"
    )
    assert manifest["generator_sha256"] == sha(
        ROOT / "astra_reversal/ood_extensions.py"
    )
    indexed = {(row["task_id"], row["seed"]): row for row in receipt["rows"]}
    assert len(indexed) == 32
    validation = OUTPUT / "validation"
    validation.mkdir(exist_ok=True)
    shutil.copyfile(receipt_path, validation / "receipt.json")
    for task in manifest["tasks"]:
        assert sha(OUTPUT / task["bddl"]) == task["bddl_sha256"]
        for seed in (71, 73):
            row = indexed[task["id"], seed]
            assert row["bddl_sha256"] == task["bddl_sha256"]
            assert not row["initial_success"] and not row["restored_initial_success"]
            assert row["state_roundtrip_exact"] and row["constructed_goal_witness"]
            for artifact in (row["state"], row["preview"]):
                if artifact is None:
                    continue
                path = args.validation / artifact["path"]
                assert sha(path) == artifact["sha256"]
                destination = validation / artifact["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    for name, source in [
        ("LIBERO_LICENSE.txt", args.original_root / "LICENSE"),
        ("PAPER_REPOSITORY_LICENSE.txt", args.ood_root / "LICENSE"),
    ]:
        shutil.copyfile(source, OUTPUT / name)
    for module in ("ood_extensions.py", "validate_ood_extensions.py"):
        shutil.copyfile(
            ROOT / "astra_reversal" / module, OUTPUT / "source" / (module + ".txt")
        )
    manifest.update(
        status="simulator_validated_policy_evaluation_pending",
        definition_manifest_sha256=sha(definition),
        validation_receipt={
            "path": "validation/receipt.json",
            "sha256": sha(receipt_path),
        },
    )
    write(OUTPUT / "manifest.json", manifest)
    lines = [
        "Created **16 additional LIBERO task definitions**, in two separate families: eight object–destination compositions and eight source-spatial-relation × destination compositions. These are separate from the paper’s ten Goal OOD and ten Spatial OOD tasks.",
        "",
        "**Simulator validation passed: 16 tasks × 2 seeds = 32 checks. Policy evaluation is pending; there is no success-rate claim.**",
        "",
        "[Visual task catalog](index.html) · [Task manifest](manifest.json) · [Simulator receipt](validation/receipt.json) · [Existing measured dashboard](../../learned_correction_recipe/dashboard/index.html)",
        "",
        "| Family | Task instruction | Definition |",
        "|---|---|---|",
    ]
    cards = []
    for task in manifest["tasks"]:
        family = (
            "Goal composition"
            if task["family"] == "astra_goal_composition"
            else "Spatial composition"
        )
        lines.append(f"| {family} | {task['instruction']} | [BDDL]({task['bddl']}) |")
        preview = indexed[task["id"], 71]["preview"]
        cards.append(
            f'<article><img loading="lazy" src="validation/{html.escape(preview["path"])}" alt="Initial simulator scene: {html.escape(task["instruction"])}"><div><span>{family}</span><h2>{html.escape(task["instruction"])}</h2><p>{html.escape(task["novelty_axis"])}<br>Two simulator resets checked · policy untested</p><a href="{html.escape(task["bddl"])}">Task definition ↗</a></div></article>'
        )
    lines += [
        "",
        "Novelty was checked against 150 supplied task files: 130 original LIBERO definitions and the paper’s 20 OOD definitions. The comparison uses declared object types, not instance-name aliases; stove base/cook-region aliases are normalized. For Goal additions, the object type × destination goal is absent from all supplied goal atoms. For Spatial additions, the source relation × destination combination is absent; its individual object/destination pair can be familiar. All correction-teacher tasks used in the learned recipe are contained in the compared paper set.",
        "",
        "This establishes novelty relative to those task definitions, not certified novelty relative to the checkpoint’s entire training data. The full training inventory is unknown. Existing object meshes and task templates are reused; these are compositional tasks, not new-object-category tasks or a visual-corruption suite.",
        "",
        "Validation runs the real modified LIBERO parser and simulator (MuJoCo 3.2.3, robosuite 1.4.1), using fixed seeds 71 and 73 on CPU. Each reset is checked before and after ten zero-action stabilization steps; both camera observations must be nonblank. A constructed positive witness moves the target object over the destination and checks the actual success predicate. Restoring the saved initial simulator state must be exact and return success to false. These witnesses are teleports, not controller trajectories; they do not establish robotic reachability or policy solvability. The 32 saved states are validation fixtures, not yet a frozen policy-evaluation split. Preview images show seed 71 after stabilization, with the same display orientation as the policy harness.",
        "",
        "All tasks inherit LIBERO’s existing goal predicates. For example, the bowl-placement tasks use its `On` contact/support predicate, as in the original cream-cheese-in-bowl and paper wine-in-bowl tasks; this is not a new volumetric-containment test. The original predicates and published task files were not modified.",
        "",
        "A future comparison should freeze a new paired reset manifest, run every method on every case with matched noise and a declared action cap, and report these two families separately from the paper benchmark. Keep this task set out of correction training if it is to test transfer. The current recorded-schedule control has no exact-instruction teacher for these tasks and would default to native; any new teacher acquisition or retrieval policy must be declared as a separate method/development split.",
        "",
        "The immutable definition-time input is retained in [definition_manifest.json](definition_manifest.json); the simulator receipt binds its exact hash. The public [manifest](manifest.json) adds the completed validation status and receipt pointer. [publication_manifest.json](publication_manifest.json) binds all delivered definitions, previews, states, source snapshots and licenses.",
        "",
        "Recreate definitions with `python -m astra_reversal.ood_extensions --original-root ORIGINAL_LIBERO --ood-root PAPER_REPOSITORY --output NEW_DIRECTORY`. Validate with `python -m astra_reversal.validate_ood_extensions --tasks NEW_DIRECTORY --libero-root PAPER_REPOSITORY/third_party/modified_libero --output NEW_VALIDATION_DIRECTORY`, in an environment with the pinned simulator dependencies. Source the project environment before Python tooling, as required by AGENTS.md.",
        "",
        "Attribution: Goal templates derive from [QuanyiLi/pi0-text-latent](https://github.com/QuanyiLi/pi0-text-latent), revision `587a6cbf64f16c7b87fa5805dc0ed934192239a4`, including its modified LIBERO. Spatial templates derive from [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), revision `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. Changes replace the named source object for Goal tasks, change destinations and objects of interest, and provide matching task instructions. Original layouts and distractors are retained. [LIBERO MIT license](LIBERO_LICENSE.txt) and [paper repository Apache 2.0 license](PAPER_REPOSITORY_LICENSE.txt) are included.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines))
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Additional LIBERO task catalog</title><style>
body{font:15px/1.65 system-ui,sans-serif;color:#172a43;background:#f5f7fb;max-width:1250px;margin:40px auto;padding:0 22px}h1{font-size:38px;letter-spacing:-1.3px;line-height:1.15;margin:20px 0}h2{font-size:16px;line-height:1.4;margin:10px 0}a{color:#265da5}header{max-width:960px}.badge{font-size:11px;font-weight:700;letter-spacing:1px;color:#15755d}.notice{padding:18px 23px;background:#e8f2ed;border:1px solid #c4ded3;border-radius:10px;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:19px;margin:30px 0}article{background:white;border:1px solid #dce3ec;border-radius:13px;overflow:hidden}article img{width:100%;display:block;aspect-ratio:1}article>div{padding:20px}article span{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#647b95;font-weight:700}article p{font-size:12px;color:#586a7f}article a{font-size:12px}footer{margin:30px 0;font-size:12px;color:#586a7f}@media(max-width:540px){.grid{grid-template-columns:1fr 1fr;gap:10px}article>div{padding:12px}article h2{font-size:13px}article p{font-size:10px}h1{font-size:31px}}</style>
<header><a href="../../learned_correction_recipe/dashboard/index.html">← Methods and measured results</a><p class="badge">SEPARATE TASK EXTENSION · 16 DEFINITIONS</p><h1>Familiar objects.<br>New compositions.</h1><p>Eight new object–destination tasks and eight new source-spatial-relation × destination tasks. Initial simulator previews below are from validation seed 71.</p><div class="notice"><strong>32 simulator checks passed. Policy evaluation is pending.</strong><br>These tasks add no successes or failures to the existing twenty-task paper benchmark.</div><p>Novelty is relative to 150 supplied task definitions. Full checkpoint training coverage is unknown. Constructed positive goal checks are not robot rollouts or reachability proofs.</p><a href="README.md">Methods and limitations</a> · <a href="manifest.json">Exact task manifest</a> · <a href="validation/receipt.json">Simulator evidence</a></header><div class="grid">"""
    page += (
        "\n".join(cards)
        + "</div><footer>Rendered validation scenes, not generated illustrations. Original layouts, assets and success predicates are preserved; new compositions are listed separately.</footer></html>\n"
    )
    (OUTPUT / "index.html").write_text(page)
    files = [
        p
        for p in sorted(OUTPUT.rglob("*"))
        if p.is_file() and p.name != "publication_manifest.json"
    ]
    write(
        OUTPUT / "publication_manifest.json",
        {
            "schema_version": "astra-extension-publication-1.0",
            "source_sha256": sha(Path(__file__)),
            "files": {
                str(p.relative_to(OUTPUT)): {
                    "sha256": sha(p),
                    "bytes": p.stat().st_size,
                }
                for p in files
            },
        },
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "tasks": 16,
                "reset_checks": 32,
                "published_files": len(files),
                "policy_rollouts": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
