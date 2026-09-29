"""Create separate, attributed LIBERO task extensions without editing upstream.

Goal tasks hold the scene template fixed while changing object/destination
composition. Spatial tasks combine an existing source relation with a different
destination. Novelty is relative to supplied task definitions, not model training.
"""

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

ORIGINAL_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
OOD_REVISION = "587a6cbf64f16c7b87fa5805dc0ed934192239a4"
GOALS = [
    ("milk", "in the bowl", "akita_black_bowl_1"),
    ("butter", "on the stove", "flat_stove_1_base_region"),
    ("chocolate_pudding", "in the bowl", "akita_black_bowl_1"),
    ("alphabet_soup", "on the plate", "plate_1"),
    ("tomato_sauce", "on the plate", "plate_1"),
    ("ketchup", "on the stove", "flat_stove_1_base_region"),
    ("orange_juice", "on top of the cabinet", "wooden_cabinet_1_top_side"),
    ("bbq_sauce", "on top of the cabinet", "wooden_cabinet_1_top_side"),
]
RELATIONS = [
    ("next to the cookie box", "next_to_the_cookie_box"),
    ("next to the ramekin", "next_to_the_ramekin"),
    ("between the plate and the ramekin", "between_the_plate_and_the_ramekin"),
    ("on the ramekin", "on_the_ramekin"),
]


def parse(text):
    """Parse balanced BDDL S-expressions without importing the simulator."""
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    roots, stack = [], []
    for token in tokens:
        if token == "(":
            value = []
            (stack[-1] if stack else roots).append(value)
            stack.append(value)
        elif token == ")":
            if not stack:
                raise ValueError("Unmatched closing parenthesis")
            stack.pop()
        elif not stack:
            raise ValueError("Atom outside the problem")
        else:
            stack[-1].append(token)
    if stack or len(roots) != 1 or not roots[0] or roots[0][0] != "define":
        raise ValueError("Expected one balanced BDDL problem")
    return roots[0]


def section(tree, name):
    found = [x for x in tree[1:] if isinstance(x, list) and x[0] == name]
    if len(found) != 1:
        raise ValueError(f"Expected one {name} section")
    return found[0]


def typed_symbols(items):
    result, pending = {}, []
    i = 0
    while i < len(items):
        if items[i] == "-":
            if not pending or i + 1 >= len(items):
                raise ValueError("Invalid typed declaration")
            for symbol in pending:
                if symbol in result:
                    raise ValueError("Duplicate object declaration")
                result[symbol] = items[i + 1]
            pending = []
            i += 2
        else:
            pending.append(items[i])
            i += 1
    if pending:
        raise ValueError("Untyped objects are not supported")
    return result


def symbols(tree):
    objects = typed_symbols(section(tree, ":objects")[1:])
    fixtures = typed_symbols(section(tree, ":fixtures")[1:])
    if set(objects) & set(fixtures):
        raise ValueError("Object and fixture names overlap")
    return {**objects, **fixtures}


def regions(tree):
    return {region[1][1] + "_" + region[0] for region in section(tree, ":regions")[1:]}


def atoms(value):
    if value[0].lower() == "and":
        return [atom for child in value[1:] for atom in atoms(child)]
    return [value]


def reference_key(symbol, types):
    if symbol in types:
        return types[symbol]
    for name in sorted(types, key=len, reverse=True):
        if symbol.startswith(name + "_"):
            suffix = symbol[len(name) + 1 :]
            # Treat these stove regions as the same destination for novelty.
            if types[name] == "flat_stove" and suffix in ("base_region", "cook_region"):
                suffix = "placement_surface"
            return types[name] + "/" + suffix
    raise ValueError(f"Unresolved reference: {symbol}")


def goal_keys(tree):
    types = symbols(tree)
    return [
        tuple([a[0].lower(), *(reference_key(x, types) for x in a[1:])])
        for a in atoms(section(tree, ":goal")[1])
    ]


def spatial_keys(tree):
    types = symbols(tree)
    initial = section(tree, ":init")[1:]
    result = []
    for goal in atoms(section(tree, ":goal")[1]):
        if len(goal) != 3:
            continue
        source = next(
            (
                a
                for a in initial
                if len(a) == 3 and a[1] == goal[1] and a[0].lower() in ("on", "in")
            ),
            None,
        )
        if source:
            result.append(
                (
                    reference_key(goal[1], types),
                    source[0].lower(),
                    reference_key(source[2], types),
                    goal[0].lower(),
                    reference_key(goal[2], types),
                )
            )
    return result


def validate(tree):
    types = symbols(tree)
    declared = set(types) | regions(tree)
    for atom in [*section(tree, ":init")[1:], *atoms(section(tree, ":goal")[1])]:
        if atom[0].lower() not in ("on", "in", "open", "close", "turnon", "turnoff"):
            raise ValueError(f"Unsupported predicate: {atom[0]}")
        if any(symbol not in declared for symbol in atom[1:]):
            raise ValueError(f"Unresolved predicate reference: {atom}")
    if any(name not in declared for name in section(tree, ":obj_of_interest")[1:]):
        raise ValueError("Unresolved object of interest")
    if not section(tree, ":language")[1:]:
        raise ValueError("Missing instruction")
    return True


def rename(value, old, new):
    if isinstance(value, list):
        return [rename(x, old, new) for x in value]
    return value.replace(old, new)


def serialize(value, level=0):
    if not isinstance(value, list):
        return str(value)
    if all(not isinstance(x, list) for x in value):
        return "(" + " ".join(map(str, value)) + ")"
    return (
        "(\n"
        + "\n".join("  " * (level + 1) + serialize(x, level + 1) for x in value)
        + ")"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def generate(original_root, ood_root, output):
    if output.exists():
        raise FileExistsError("Refusing to overwrite a task release")
    if (
        revision(original_root) != ORIGINAL_REVISION
        or revision(ood_root) != OOD_REVISION
    ):
        raise ValueError("Task source revisions differ from the prescribed inputs")
    original = original_root / "libero/libero/bddl_files"
    modified = ood_root / "third_party/modified_libero/libero/libero/bddl_files"
    refs = [
        ("original", p, p.relative_to(original))
        for p in sorted(original.rglob("*.bddl"))
    ]
    refs += [
        ("paper_ood", p, p.relative_to(modified))
        for suite in ("libero_goal_ood", "libero_spatial_ood")
        for p in sorted((modified / suite).glob("*.bddl"))
    ]
    ref_goals, ref_spatial, instructions, inventory = set(), set(), set(), []
    for label, path, relative in refs:
        tree = parse(path.read_text())
        ref_goals.update(goal_keys(tree))
        ref_spatial.update(spatial_keys(tree))
        instructions.add(" ".join(section(tree, ":language")[1:]).lower())
        inventory.append({"source": label, "path": str(relative), "sha256": sha(path)})
    tasks, trees = [], []
    template_path = modified / "libero_goal_ood/put_the_wine_bottle_on_the_plate.bddl"
    template = parse(template_path.read_text())
    for object_type, destination_text, destination in GOALS:
        tree = rename(copy.deepcopy(template), "wine_bottle", object_type)
        name = "put the " + object_type.replace("_", " ") + " " + destination_text
        section(tree, ":language")[:] = [":language", *name.split()]
        section(tree, ":obj_of_interest")[:] = [
            ":obj_of_interest",
            object_type + "_1",
            destination,
        ]
        section(tree, ":goal")[:] = [
            ":goal",
            ["And", ["On", object_type + "_1", destination]],
        ]
        keys = goal_keys(tree)
        if set(keys) & ref_goals or name in instructions:
            raise ValueError(f"Goal composition already exists: {name}")
        tasks.append(
            {
                "family": "astra_goal_composition",
                "instruction": name,
                "source_object": object_type + "_1",
                "destination": destination,
                "template_source": "paper_ood",
                "template": str(template_path.relative_to(modified)),
                "template_sha256": sha(template_path),
                "novelty_key": keys[0],
                "novelty_axis": "object type × destination",
                "novelty_check": "absent from all supplied original and paper OOD goal atoms",
            }
        )
        trees.append(tree)
    for words, file_words in RELATIONS:
        path = (
            original
            / "libero_spatial"
            / f"pick_up_the_black_bowl_{file_words}_and_place_it_on_the_plate.bddl"
        )
        for destination_words, destination in [
            ("on the stove", "flat_stove_1_cook_region"),
            ("on top of the cabinet", "wooden_cabinet_1_top_side"),
        ]:
            tree = parse(path.read_text())
            name = f"pick up the black bowl {words} and place it {destination_words}"
            section(tree, ":language")[:] = [":language", *name.split()]
            section(tree, ":obj_of_interest")[:] = [
                ":obj_of_interest",
                "akita_black_bowl_1",
                destination,
            ]
            section(tree, ":goal")[:] = [
                ":goal",
                ["And", ["On", "akita_black_bowl_1", destination]],
            ]
            keys = spatial_keys(tree)
            if set(keys) & ref_spatial or name in instructions:
                raise ValueError(f"Spatial composition already exists: {name}")
            tasks.append(
                {
                    "family": "astra_spatial_composition",
                    "instruction": name,
                    "source_object": "akita_black_bowl_1",
                    "destination": destination,
                    "source_relation": words,
                    "template_source": "original",
                    "template": str(path.relative_to(original)),
                    "template_sha256": sha(path),
                    "novelty_key": keys[0],
                    "novelty_axis": "source spatial relation × destination",
                    "novelty_check": "absent from supplied source-relation/goal combinations; goal pair alone is familiar",
                }
            )
            trees.append(tree)
    assert len(tasks) == 16 and len({t["instruction"] for t in tasks}) == 16
    output.mkdir(parents=True)
    for i, (task, tree) in enumerate(zip(tasks, trees, strict=True)):
        validate(tree)
        family_dir = output / "bddl" / task["family"]
        family_dir.mkdir(parents=True, exist_ok=True)
        path = family_dir / (task["instruction"].replace(" ", "_") + ".bddl")
        path.write_text(serialize(tree) + "\n")
        assert parse(path.read_text()) == tree
        task.update(
            id=f"{task['family']}:{i if i < 8 else i - 8}",
            bddl=str(path.relative_to(output)),
            bddl_sha256=sha(path),
            goal_atoms=atoms(section(tree, ":goal")[1]),
        )
    manifest = {
        "schema_version": "astra-ood-extension-1.0",
        "status": "definitions_created_simulator_validation_pending",
        "tasks": tasks,
        "task_count": 16,
        "families": {"astra_goal_composition": 8, "astra_spatial_composition": 8},
        "source_revisions": {
            "original_libero": ORIGINAL_REVISION,
            "paper_ood": OOD_REVISION,
        },
        "reference_task_count": len(refs),
        "reference_inventory": inventory,
        "generator_sha256": sha(Path(__file__)),
        "new_policy_rollouts": 0,
        "success_rate": None,
        "claims": [
            "Novelty is relative to the supplied original and paper OOD task definitions, plus the already-used teacher tasks contained in that paper set.",
            "Full checkpoint training coverage is unknown; these are candidate OOD compositions, not certified unseen pretraining data.",
            "These definitions are separate from the paper benchmark and are never pooled into its reported success rate.",
            "Simulator validation and policy solvability are separate checks. Task generation alone is not performance evidence.",
        ],
        "evaluation_contract": {
            "astra_recipe_training_forbidden": True,
            "preserve_paper_benchmark": True,
            "suggested_action_cap": 300,
            "paired_reset_manifest_required": True,
            "new_teacher_acquisition_requires_separate_development_split": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--ood-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(
        args.original_root.resolve(), args.ood_root.resolve(), args.output.resolve()
    )
    print(
        json.dumps(
            {
                "tasks": manifest["task_count"],
                "reference_tasks": manifest["reference_task_count"],
                "status": manifest["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
