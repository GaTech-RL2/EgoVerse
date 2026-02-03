import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict


VALID_DATASET_LINES = [
    "_target_: egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper",
    "",
    "valid_datasets:",
    "  dataset1: ",
    "    _target_: egomimic.rldb.utils.MultiRLDBDataset",
    "    datasets:",
    "      # song record 2",
    "      op1_scene1_ep1:",
    "        _target_: egomimic.rldb.utils.S3RLDBDataset",
    '        bucket_name: "rldb"',
    "        mode: total",
    '        embodiment: "aria_bimanual"',
    "        local_files_only: True",
    '        temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"',
    "        filters: {episode_hash: '2025-11-11-23-06-20-738000'}",
    "",
    '    embodiment: "aria_bimanual"',
    "",
    "",
    "train_datasets:",
    "  dataset1:",
    "    _target_: egomimic.rldb.utils.MultiRLDBDataset",
    "    datasets:",
    "",
]


TRAIN_DATALOADER_LINES = [
    "",
    '    embodiment: "aria_bimanual"',
    "",
    "train_dataloader_params:",
    "  dataset1:",
    "    batch_size: 32",
    "    num_workers: 10",
    "",
    "valid_dataloader_params:",
    "  dataset1:",
    "    batch_size: 32",
    "    num_workers: 10",
]


OPERATOR_NAME_MAP = {
    5: "Ryan",
    6: "Pranav",
    7: "Nadun",
    8: "Yangcen",
    13: "Xinchen",
    14: "Rohan",
    15: "David",
    16: "Vaibhav",
}

NAME_TO_OPERATOR_ID = {value: key for key, value in OPERATOR_NAME_MAP.items()}

RLDB_DATA_ROOT = "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity/S3_rldb_data"


def unique_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def parse_hashes(cell_value):
    if cell_value is None:
        return []
    value = str(cell_value).strip()
    if not value or value.lower() == "nan":
        return []
    parts = [part.strip() for part in value.split(",")]
    hashes = []
    seen = set()
    for part in parts:
        if not part or part.lower() == "nan":
            continue
        if part in seen:
            continue
        seen.add(part)
        hashes.append(part)
    return hashes


def load_hashes(csv_path):
    data = defaultdict(dict)
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        scene_columns = [name for name in fieldnames if name.startswith("Scenario ")]
        operator_column = fieldnames[0] if fieldnames else None
        if not operator_column:
            raise ValueError("CSV missing operator column.")
        for row in reader:
            operator_label = (row.get(operator_column) or "").strip()
            if not operator_label or operator_label == "Total":
                continue
            match = re.search(r"Operator\s+(\d+)", operator_label)
            if not match:
                continue
            operator_id = int(match.group(1))
            for scene_col in scene_columns:
                scene_id = int(scene_col.split()[1])
                data[operator_id][scene_id] = parse_hashes(row.get(scene_col))
    return data


def parse_existing_mixed_diversity(mixed_dir):
    existing = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(mixed_dir, "mixed_diversity_*.yaml")
    for config_path in glob.glob(pattern):
        filename = os.path.basename(config_path)
        match = re.match(r"mixed_diversity_(\d+)_(\d+)_.+\\.yaml", filename)
        if not match:
            continue
        scene_count = int(match.group(1))
        if scene_count in {5, 7}:
            continue
        parsed = parse_existing_config(config_path)
        for operator_id, scenes in parsed.items():
            for scene_id, hashes in scenes.items():
                existing[operator_id][scene_id].extend(hashes)
    for operator_id, scenes in existing.items():
        for scene_id, hashes in scenes.items():
            existing[operator_id][scene_id] = unique_preserve_order(hashes)
    return existing


def parse_existing_config(config_path):
    operator_order = []
    episodes = defaultdict(lambda: defaultdict(list))
    current_operator = None
    current_combo = None
    in_train = False

    with open(config_path) as yaml_file:
        for raw_line in yaml_file:
            line = raw_line.rstrip()
            stripped = line.strip()
            if stripped.startswith("train_datasets:"):
                in_train = True
                continue
            if not in_train:
                continue
            if stripped.startswith("# "):
                candidate = stripped[2:].strip()
                if (
                    candidate.startswith("op")
                    or ":" in candidate
                    or candidate.startswith("_target_")
                    or candidate.startswith("bucket_name")
                    or candidate.startswith("mode")
                    or candidate.startswith("embodiment")
                    or candidate.startswith("local_files_only")
                    or candidate.startswith("temp_root")
                    or candidate.startswith("filters")
                ):
                    continue
                current_operator = candidate
                operator_order.append(current_operator)
                continue
            if stripped.startswith("#"):
                continue
            match = re.match(r"op(\d+)_scene(\d+)_ep(\d+):", stripped)
            if match:
                op_index = int(match.group(1))
                scene_id = int(match.group(2))
                if op_index <= len(operator_order):
                    operator_name = operator_order[op_index - 1]
                else:
                    operator_name = None
                current_combo = (operator_name, scene_id)
                continue
            if "filters:" in stripped and "episode_hash" in stripped:
                hash_match = re.search(r"episode_hash:\s*'([^']+)'", stripped)
                if not hash_match or not current_combo:
                    continue
                operator_name, scene_id = current_combo
                if not operator_name:
                    continue
                operator_id = NAME_TO_OPERATOR_ID.get(operator_name)
                if not operator_id:
                    continue
                episodes[operator_id][scene_id].append(hash_match.group(1))
    return episodes


def merge_preferred_lists(primary_list, secondary_list):
    return unique_preserve_order(list(primary_list) + list(secondary_list))


def filter_existing_hashes(hashes):
    return [
        hash_val
        for hash_val in hashes
        if os.path.isdir(os.path.join(RLDB_DATA_ROOT, hash_val))
    ]


def allocate_counts(combos, available_map, total_episodes, min_per_combo, prefer_combos=None):
    counts = {combo: 0 for combo in combos}
    scene_counts = defaultdict(int)
    operator_counts = defaultdict(int)

    def available_count(combo):
        available = available_map.get(combo, 0)
        return len(available) if isinstance(available, list) else int(available)

    for combo in combos:
        if total_episodes <= 0:
            break
        available = available_count(combo)
        if available <= 0:
            continue
        add_count = min(min_per_combo, available)
        counts[combo] += add_count
        scene_counts[combo[1]] += add_count
        operator_counts[combo[0]] += add_count
        total_episodes -= add_count

    if total_episodes < 0:
        raise ValueError("Minimum allocation exceeds total episode budget.")

    def combo_key(combo):
        prefer_rank = 0 if (prefer_combos and combo in prefer_combos) else 1
        return (
            prefer_rank,
            scene_counts[combo[1]],
            operator_counts[combo[0]],
            counts[combo],
            combo[1],
            combo[0],
        )

    while total_episodes > 0:
        candidates = [combo for combo in combos if counts[combo] < available_count(combo)]
        if not candidates:
            raise ValueError("Not enough available episodes to satisfy budget.")
        best_combo = min(candidates, key=combo_key)
        counts[best_combo] += 1
        scene_counts[best_combo[1]] += 1
        operator_counts[best_combo[0]] += 1
        total_episodes -= 1

    return counts


def choose_hashes(available, count, preferred=None):
    selected = []
    preferred = preferred or []
    available_set = set(available)
    preferred_set = set(preferred)
    for hash_val in available:
        if hash_val in preferred_set and hash_val in available_set and hash_val not in selected:
            selected.append(hash_val)
            if len(selected) == count:
                return selected
    for hash_val in available:
        if hash_val in selected:
            continue
        selected.append(hash_val)
        if len(selected) == count:
            break
    return selected


def build_selection(combos, counts, available_map, preferred_map=None):
    selection = defaultdict(dict)
    for combo in combos:
        operator_id, scene_id = combo
        desired = counts.get(combo, 0)
        if desired <= 0:
            continue
        available = available_map.get(combo, [])
        if desired > len(available):
            raise ValueError(f"Not enough episodes for operator {operator_id}, scene {scene_id}.")
        preferred = preferred_map.get(combo, []) if preferred_map else []
        chosen = choose_hashes(available, desired, preferred=preferred)
        selection[operator_id][scene_id] = chosen
    return selection


def format_yaml(output_path, scenes, operators, selection):
    lines = list(VALID_DATASET_LINES)
    for op_index, operator_id in enumerate(operators, start=1):
        operator_name = OPERATOR_NAME_MAP.get(operator_id, f"Operator {operator_id}")
        lines.append(f"      # {operator_name}")
        for scene_id in scenes:
            hashes = selection.get(operator_id, {}).get(scene_id, [])
            for ep_index, hash_val in enumerate(hashes, start=1):
                dataset_name = f"op{op_index}_scene{scene_id}_ep{ep_index}"
                lines.extend(
                    [
                        f"      {dataset_name}:",
                        "        _target_: egomimic.rldb.utils.S3RLDBDataset",
                        '        bucket_name: "rldb"',
                        "        mode: total",
                        '        embodiment: "aria_bimanual"',
                        "        local_files_only: True",
                        '        temp_root: "/coc/cedarp-dxu345-0/datasets/egoverse/offline_eval_diversity"',
                        f"        filters: {{episode_hash: '{hash_val}'}}",
                    ]
                )
            if hashes:
                lines.append("")
        lines.append("")
    lines.extend(TRAIN_DATALOADER_LINES)
    with open(output_path, "w") as output_file:
        output_file.write("\n".join(lines).rstrip() + "\n")


def summarize_counts(label, scenes, operators, selection):
    scene_counts = defaultdict(int)
    operator_counts = defaultdict(int)
    total = 0
    for operator_id in operators:
        for scene_id in scenes:
            count = len(selection.get(operator_id, {}).get(scene_id, []))
            total += count
            scene_counts[scene_id] += count
            operator_counts[operator_id] += count
    print(f"{label} total episodes: {total}")
    print(f"{label} per scene: {dict(sorted(scene_counts.items()))}")
    print(f"{label} per operator: {dict(sorted(operator_counts.items()))}")
    return total


def format_minutes(value):
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return formatted.replace(".", "_")


def build_preferred_hashes(existing_configs, scene_id, priority):
    preferred = []
    for key in priority:
        preferred.extend(existing_configs.get(key, []))
    return unique_preserve_order(preferred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/nethome/yzhu827/flash/shared/EgoVerse/egomimic/hydra_configs/data/mixed_diversity",
    )
    parser.add_argument(
        "--mixed-diversity-dir",
        default="/nethome/yzhu827/flash/shared/EgoVerse/egomimic/hydra_configs/data/mixed_diversity",
    )
    parser.add_argument(
        "--csv-path",
        default="/nethome/yzhu827/flash/shared/EgoVerse/results/diversity_fold_clothes_hashes.csv",
    )
    parser.add_argument("--budget-hours", type=float, default=4.0)
    parser.add_argument("--episode-minutes", type=float, default=3.75)
    args = parser.parse_args()

    total_minutes = args.budget_hours * 60
    total_episodes = int(round(total_minutes / args.episode_minutes))
    mixed_existing = parse_existing_mixed_diversity(args.mixed_diversity_dir)
    csv_data = load_hashes(args.csv_path)
    mixed_existing_set = set(
        hash_val
        for operator_id in mixed_existing
        for scene_id in mixed_existing[operator_id]
        for hash_val in mixed_existing[operator_id][scene_id]
    )
    mixed_existing_valid_set = set(filter_existing_hashes(mixed_existing_set))

    scenes_5 = [1, 2, 3, 4, 5]
    scenes_7 = [1, 2, 3, 4, 5, 6, 7]
    operators_4 = [5, 6, 7, 8]
    operators_8 = [5, 6, 7, 8, 13, 14, 15, 16]

    existing_a = parse_existing_config(
        "/nethome/yzhu827/flash/shared/EgoVerse/egomimic/hydra_configs/data/mixed_diversity/mixed_diversity_4_4_15.yaml"
    )
    existing_b = parse_existing_config(
        "/nethome/yzhu827/flash/shared/EgoVerse/egomimic/hydra_configs/data/mixed_diversity/mixed_diversity_8_4_7_5.yaml"
    )
    existing_c = parse_existing_config(
        "/nethome/yzhu827/flash/shared/EgoVerse/egomimic/hydra_configs/data/mixed_diversity/mixed_diversity_8_8_3_75.yaml"
    )

    def build_available_and_preferred(combos, priority):
        available_map = {}
        preferred_map = {}
        for operator_id, scene_id in combos:
            primary = filter_existing_hashes(
                mixed_existing.get(operator_id, {}).get(scene_id, [])
            )
            csv_fallback = filter_existing_hashes(
                csv_data.get(operator_id, {}).get(scene_id, [])
            )
            preferred_lists = {
                "a": primary,
                "b": existing_a.get(operator_id, {}).get(scene_id, []),
                "c": existing_b.get(operator_id, {}).get(scene_id, []),
            }
            preferred = build_preferred_hashes(preferred_lists, scene_id, priority)
            preferred = filter_existing_hashes(preferred)
            preferred_map[(operator_id, scene_id)] = preferred
            available_map[(operator_id, scene_id)] = unique_preserve_order(
                preferred + csv_fallback
            )
        return available_map, preferred_map

    configs = [
        {
            "label": "5sc4op",
            "scenes": scenes_5,
            "operators": operators_4,
            "priority": ["a", "b", "c"],
        },
        {
            "label": "5sc8op",
            "scenes": scenes_5,
            "operators": operators_8,
            "priority": ["a", "c", "b"],
        },
        {
            "label": "7sc4op",
            "scenes": scenes_7,
            "operators": operators_4,
            "priority": ["a", "c", "b"],
        },
        {
            "label": "7sc8op",
            "scenes": scenes_7,
            "operators": operators_8,
            "priority": ["a", "c", "b"],
        },
    ]

    selections = {}
    added_from_csv = {}
    for config in configs:
        scenes = config["scenes"]
        operators = config["operators"]
        combos = [(op, scene) for op in operators for scene in scenes]
        available, existing_preferred = build_available_and_preferred(combos, config["priority"])

        min_total = len(combos)
        target_episodes = max(min_total, total_episodes)
        available_counts = {combo: len(available.get(combo, [])) for combo in combos}
        available_total = sum(available_counts.values())
        if available_total < target_episodes:
            missing = [combo for combo, count in available_counts.items() if count == 0]
            missing_preview = ", ".join(
                f"op{combo[0]}_scene{combo[1]}=0" for combo in missing[:6]
            )
            base_count = int(math.floor(target_episodes / len(combos)))
            extra_needed = target_episodes - base_count * len(combos)
            combos_with_extra = sum(
                1 for count in available_counts.values() if count > base_count
            )
            min_available = min(available_counts.values()) if available_counts else 0
            if available_total > 55:
                print(
                    f"Warning: {config['label']} has only {available_total} episodes "
                    f"available (target {target_episodes}). Proceeding with {available_total}. "
                    f"Min per combo={min_available}, Need extra={extra_needed}, "
                    f"Combos with extra={combos_with_extra}. "
                    f"Missing combos: {missing_preview}"
                )
                target_episodes = available_total
            else:
                print(
                    f"Warning: {config['label']} has only {available_total} episodes "
                    f"available (target {target_episodes}). Proceeding with {available_total}. "
                    f"Min per combo={min_available}, Need extra={extra_needed}, "
                    f"Combos with extra={combos_with_extra}. "
                    f"Missing combos: {missing_preview}"
                )
                target_episodes = available_total

        base_count = int(math.floor(target_episodes / len(combos)))
        min_available = min(available_counts.values()) if available_counts else 0
        combos_below_base = [
            combo for combo in combos if len(available.get(combo, [])) < base_count
        ]
        if combos_below_base:
            if available_total > 55 and target_episodes < total_episodes:
                print(
                    f"Warning: {config['label']} cannot keep base={base_count} "
                    f"uniform distribution with {target_episodes} episodes. "
                    f"Using base={min_available}."
                )
                base_count = min_available
            else:
                preview = ", ".join(
                    f"op{combo[0]}_scene{combo[1]}={len(available.get(combo, []))}"
                    for combo in combos_below_base[:6]
                )
                print(
                    f"Warning: {config['label']} cannot keep base={base_count} "
                    f"uniform distribution. Using base={min_available}. "
                    f"Below base={base_count}: {preview}"
                )
                base_count = min_available
        counts = allocate_counts(combos, available, target_episodes, min_per_combo=base_count)
        overlap_preferred = defaultdict(list)
        for prev_selection in selections.values():
            for operator_id in prev_selection:
                for scene_id in prev_selection[operator_id]:
                    overlap_preferred[(operator_id, scene_id)] = prev_selection[operator_id][scene_id]

        preferred_map = {}
        for combo in combos:
            preferred_map[combo] = merge_preferred_lists(
                overlap_preferred.get(combo, []), existing_preferred.get(combo, [])
            )

        selection = build_selection(combos, counts, available, preferred_map=preferred_map)
        selections[config["label"]] = selection
        selected_hashes = [
            hash_val
            for operator_id in selection
            for scene_id in selection[operator_id]
            for hash_val in selection[operator_id][scene_id]
        ]
        added_from_csv[config["label"]] = sum(
            1 for hash_val in selected_hashes if hash_val not in mixed_existing_valid_set
        )

    outputs = {}
    for config in configs:
        scenes = config["scenes"]
        operators = config["operators"]
        minutes_per_combo = total_minutes / (len(scenes) * len(operators))
        minutes_label = format_minutes(minutes_per_combo)
        output_path = f"{args.output_dir}/mixed_diversity_{len(scenes)}_{len(operators)}_{minutes_label}.yaml"
        outputs[config["label"]] = output_path
        format_yaml(output_path, scenes, operators, selections[config["label"]])
        print(f"Wrote {output_path}")

    overlaps = {}
    labels = list(selections.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            left = labels[i]
            right = labels[j]
            left_hashes = set(
                hash_val
                for operator_id in selections[left]
                for scene_id in selections[left][operator_id]
                for hash_val in selections[left][operator_id][scene_id]
            )
            right_hashes = set(
                hash_val
                for operator_id in selections[right]
                for scene_id in selections[right][operator_id]
                for hash_val in selections[right][operator_id][scene_id]
            )
            overlaps[(left, right)] = len(left_hashes & right_hashes)

    for config in configs:
        label = config["label"]
        total = summarize_counts(
            label,
            config["scenes"],
            config["operators"],
            selections[label],
        )
        minutes = total * args.episode_minutes
        print(f"{label} minutes: {minutes:.2f}")
        print(f"{label} episodes added from csv: {added_from_csv.get(label, 0)}")

    for (left, right), count in overlaps.items():
        print(f"Overlap {left} vs {right}: {count}")

    print(f"Target episodes per config: {total_episodes}")


if __name__ == "__main__":
    main()



