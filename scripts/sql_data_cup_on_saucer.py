"""
SQL episode coverage stats for the "cup on saucer" task (scene 1 only).

Notes:
- The SQL `task` column may contain: "cup on saucer", "cup_on_saucer", "cup_in_saucer".
- We count:
  - A: #episodes with non-empty `processed_path`
  - B: total #episodes in SQL for that operator+scene
  - C: optional expected denominator (set to 0 by default; fill if you have image-based expectations)
"""

from __future__ import annotations

import os
import pandas as pd

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


def main() -> None:
    engine = create_default_engine()
    df = episode_table_to_df(engine)

    # Filter for task variants (cup-on-saucer family).
    task_names = ["cup on saucer", "cup_on_saucer", "cup_in_saucer"]
    df_task = df[df["task"].isin(task_names)].copy()

    # Only care about scene 1.
    df_task["scene"] = df_task["scene"].astype(str)
    df_task = df_task[df_task["scene"] == "1"].copy()

    # Operator mapping to "Operator N (Name)" format.
    # Keys must match what appears in the SQL `operator` column.
    operator_mapping = {
        "Jenny": "Operator 1 (Jenny)",
        "Aniketh": "Operator 2 (Aniketh)",
        "Baoyu": "Operator 3 (Baoyu)",
        "Lawrence": "Operator 4 (Lawrence)",
        "RyanC": "Operator 5 (RyanC)",
        "Pranav": "Operator 6 (Pranav)",
        "Elmo": "Operator 7 (Elmo)",
        "Yangcen": "Operator 8 (Yangcen)",
        "Zhenyang": "Operator 9 (Zhenyang)",
        "Mengying": "Operator 10 (Mengying)",
        "Vaibhav": "Operator 11 (Vaibhav)",
        "Shuo": "Operator 12 (Shuo)",
        "Xinchen": "Operator 13 (Xinchen)",
        "David": "Operator 14 (David)",
        "19": "Operator 15 (19)",
        "Woochul": "Operator 16 (Woochul)",
        # Common spelling variant seen elsewhere in this repo
        "Woolchul": "Operator 16 (Woochul)",
    }

    # Optional expected denominators (C). Fill these if you have image-based expectations.
    expected_denominators = {
        # (operator, scene): expected_count
        # ("Jenny", "1"): 0,
    }

    # processed_path is considered "present" if it's not null and not empty after stripping.
    df_task["has_processed_path"] = (
        df_task["processed_path"].notna()
        & (df_task["processed_path"].astype(str).str.strip() != "")
    )

    scene = "1"
    results = []
    for operator in operator_mapping.keys():
        op_scene_df = df_task[(df_task["operator"] == operator) & (df_task["scene"] == scene)]

        # A: Count where processed_path is not empty
        A = int(op_scene_df["has_processed_path"].sum())

        # Episode hashes where processed_path is not empty
        processed_episodes = op_scene_df[op_scene_df["has_processed_path"]]
        episode_hashes = (
            processed_episodes["episode_hash"].tolist() if len(processed_episodes) > 0 else []
        )

        # B: Total count for operator-scene
        B = int(len(op_scene_df))

        # C: Expected denominator (defaults to 0)
        C = int(expected_denominators.get((operator, scene), 0))

        results.append(
            {
                "operator": operator_mapping[operator],
                "scene": scene,
                "A": A,
                "B": B,
                "C": C,
                "episode_hashes": episode_hashes,
            }
        )

    results_df = pd.DataFrame(results)

    # Output tables: one CSV with "A / B / C", one CSV with hashes.
    output_data = {}
    hash_output_data = {}

    operator_order = [operator_mapping[op] for op in operator_mapping.keys()]
    for operator_display in operator_order:
        row = results_df[results_df["operator"] == operator_display]
        if len(row) > 0:
            A_val = int(row["A"].iloc[0])
            B_val = int(row["B"].iloc[0])
            C_val = int(row["C"].iloc[0])
            episode_hashes = row["episode_hashes"].iloc[0]
        else:
            A_val, B_val, C_val, episode_hashes = 0, 0, 0, []

        output_data[operator_display] = {"Scenario 1": f"{A_val} / {B_val} / {C_val}"}
        hash_output_data[operator_display] = {
            "Scenario 1": ", ".join(str(h) for h in episode_hashes) if episode_hashes else ""
        }

    output_df = pd.DataFrame(output_data).T
    output_df.index.name = "Operator"

    hash_output_df = pd.DataFrame(hash_output_data).T
    hash_output_df.index.name = "Operator"

    # Totals
    total_A = int(results_df["A"].sum()) if len(results_df) else 0
    total_B = int(results_df["B"].sum()) if len(results_df) else 0
    total_C = int(results_df["C"].sum()) if len(results_df) else 0
    output_df.loc["Total"] = {"Scenario 1": f"{total_A} / {total_B} / {total_C}"}
    hash_output_df.loc["Total"] = {"Scenario 1": ""}

    os.makedirs("results", exist_ok=True)
    output_df.to_csv("results/diversity_cup_on_saucer_scene1.csv")
    hash_output_df.to_csv("results/diversity_cup_on_saucer_scene1_hashes.csv")

    print("Saved:")
    print(" - results/diversity_cup_on_saucer_scene1.csv")
    print(" - results/diversity_cup_on_saucer_scene1_hashes.csv")
    print("\nPreview (stats):")
    print(output_df)


if __name__ == "__main__":
    main()


