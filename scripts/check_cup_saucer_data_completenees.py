"""
Check SQL episode coverage for the "cup_on_saucer" task across all scenes (1-16).

This script compares:
  A: #episodes with non-empty `processed_path`
  B: total #episodes in SQL for that operator+scene
  C: expected #episodes derived from the data collection plan

Notes:
- 3.75 minutes = 1 episode.
- For "Operator 5 (RyanCo & 19)": RyanC covers scenes 1-8, and operator "19" covers scenes 9-16.
- If you have a real CSV/TSV file for the plan, pass it via --plan-csv.
"""

from __future__ import annotations

import argparse
import io
import os
from typing import Dict, Iterable, List, Optional

import pandas as pd

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df


EPISODE_MINUTES = 3.75
SCENES = [str(i) for i in range(1, 17)]


DEFAULT_PLAN_CSV = """Operator,Scenario 1,Scenario 2,Scenario 3,Scenario 4,Scenario 5,Scenario 6,Scenario 7,Scenario 8,Scenario 9,Scenario 10,Scenario 11,Scenario 12,Scenario 13,Scenario 14,Scenario 15,Scenario 16
Operator 1 (Jenny),120,3.75,3.75,3.75,3.75,3.75,3.75,3.75,/,/,/,15,/,15,15,/
Operator 2 (Aniketh),60,3.75,3.75,3.75,3.75,3.75,3.75,3.75,15,15,/,15,/,15,15,/
Operator 3 (Baoyu),30,3.75,3.75,3.75,3.75,3.75,3.75,3.75,15,15,15,15,15,/,/,/
Operator 4 (Lawrence),30,3.75,3.75,3.75,3.75,3.75,3.75,3.75,15,/,/,15,15,15,15,/
Operator 5 (RyanC),15,15,15,15,15,15,15,15,/,/,/,/,/,/,/,/
Operator 6 (Pranav_2),15,15,15,15,15,15,15,15,/,15,/,/,15,/,/,15
Operator 7 (Elmo),15,15,15,15,15,15,15,15,/,/,15,/,15,/,/,15
Operator 8 (Yangcen),15,15,15,15,15,15,15,15,/,/,15,/,/,15,/,15
Operator 9 (Zhenyang),7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,,,,,,,,
Operator 10 (Mengying),7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,,,,,,,,
Operator 11 (Vaibhav),7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,,,,,,,,
Operator 12 (Shuo),7.5,7.5,7.5,7.5,7.5,7.5,7.5,7.5,,,,,,,,
Operator 13 (Xinchen),7.5,7.5,,,,,,,,,,,,,,
Operator 14 (David),7.5,,7.5,,,,,,,,,,,,,
Operator 15 (Aiden),7.5,,,7.5,,,,,,,,,,,,
Operator 16 (Woochul),7.5,,,,7.5,,,,,,,,,,,
Operator 17 (Lawrance sit (val)),15,,,,,,,,,,,,,,,
"""


def _read_plan_dataframe(plan_csv: Optional[str]) -> pd.DataFrame:
    if plan_csv:
        return pd.read_csv(plan_csv, sep=None, engine="python")
    return pd.read_csv(io.StringIO(DEFAULT_PLAN_CSV))


def _minutes_to_episodes(minutes: float) -> int:
    if pd.isna(minutes):
        return 0
    if str(minutes).strip() == "/":
        return 0
    return int(round(float(minutes) / EPISODE_MINUTES))


def _build_scene_operator_map() -> Dict[str, Dict[str, str]]:
    # Map display row -> scene -> SQL operator name.
    scene_operator_map: Dict[str, Dict[str, str]] = {}

    def assign_all(display: str, sql_operator: str) -> None:
        scene_operator_map[display] = {scene: sql_operator for scene in SCENES}

    assign_all("Operator 1 (Jenny)", "Jenny")
    assign_all("Operator 2 (Aniketh)", "Aniketh")
    assign_all("Operator 3 (Baoyu)", "Baoyu")
    assign_all("Operator 4 (Lawrence)", "Lawrence")
    assign_all("Operator 6 (Pranav_2)", "Pranav_2")
    assign_all("Operator 7 (Elmo)", "Elmo")
    assign_all("Operator 8 (Yangcen)", "Yangcen")
    assign_all("Operator 9 (Zhenyang)", "Zhenyang")
    assign_all("Operator 10 (Mengying)", "Mengying")
    assign_all("Operator 11 (Vaibhav)", "Vaibhav")
    assign_all("Operator 12 (Shuo)", "Shuo")
    assign_all("Operator 13 (Xinchen)", "Xinchen")
    assign_all("Operator 14 (David)", "David")
    assign_all("Operator 15 (Aiden)", "Aiden")
    assign_all("Operator 16 Woochul", "Woochul")
    assign_all("Operator 17 Lawrance sit (val)", "Lawrance_2")

    assign_all("Operator 5 (RyanC)", "RyanC")

    return scene_operator_map


def _normalize_plan_columns(plan_df: pd.DataFrame) -> pd.DataFrame:
    plan_df = plan_df.copy()
    plan_df.columns = [str(col).strip() for col in plan_df.columns]
    if "Operator" not in plan_df.columns:
        first_col = plan_df.columns[0]
        plan_df = plan_df.rename(columns={first_col: "Operator"})
    for scene in SCENES:
        col_name = f"Scenario {scene}"
        if col_name not in plan_df.columns:
            plan_df[col_name] = pd.NA
    # Normalize placeholders like "/" to NA for all scenario columns.
    for scene in SCENES:
        col_name = f"Scenario {scene}"
        plan_df[col_name] = plan_df[col_name].replace("/", pd.NA)
    return plan_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-csv",
        default=None,
        help="Optional path to the data collection plan CSV/TSV.",
    )
    args = parser.parse_args()

    plan_df = _normalize_plan_columns(_read_plan_dataframe(args.plan_csv))

    # Build per-display expected denominators from plan.
    scene_operator_map = _build_scene_operator_map()
    expected_denominators: Dict[tuple, int] = {}

    for _, row in plan_df.iterrows():
        display_name = str(row["Operator"]).strip()
        if display_name not in scene_operator_map:
            continue

        for scene in SCENES:
            col_name = f"Scenario {scene}"
            expected = _minutes_to_episodes(row.get(col_name, pd.NA))
            sql_operator = scene_operator_map[display_name][scene]
            expected_denominators[(sql_operator, scene)] = expected

    engine = create_default_engine()
    df = episode_table_to_df(engine)

    task_names = ["cup_on_saucer", "cup on saucer", "cup_in_saucer"]
    df_task = df[df["task"].isin(task_names)].copy()
    df_task["scene"] = df_task["scene"].astype(str)

    df_task["has_processed_path"] = (
        df_task["processed_path"].notna()
        & (df_task["processed_path"].astype(str).str.strip() != "")
    )

    results = []
    for display_name in scene_operator_map.keys():
        for scene in SCENES:
            sql_operator = scene_operator_map[display_name][scene]
            op_scene_df = df_task[
                (df_task["operator"] == sql_operator) & (df_task["scene"] == scene)
            ]

            A = int(op_scene_df["has_processed_path"].sum())
            B = int(len(op_scene_df))
            C = int(expected_denominators.get((sql_operator, scene), 0))

            episode_hashes = (
                op_scene_df[op_scene_df["has_processed_path"]]["episode_hash"].tolist()
                if len(op_scene_df) > 0
                else []
            )

            results.append(
                {
                    "operator_display": display_name,
                    "scene": scene,
                    "A": A,
                    "B": B,
                    "C": C,
                    "episode_hashes": episode_hashes,
                }
            )

    results_df = pd.DataFrame(results)

    output_data: Dict[str, Dict[str, str]] = {}
    hash_output_data: Dict[str, Dict[str, str]] = {}
    operator_order = list(scene_operator_map.keys())

    for display_name in operator_order:
        output_data[display_name] = {}
        hash_output_data[display_name] = {}
        for scene in SCENES:
            row = results_df[
                (results_df["operator_display"] == display_name)
                & (results_df["scene"] == scene)
            ]
            if len(row) > 0:
                A_val = int(row["A"].iloc[0])
                B_val = int(row["B"].iloc[0])
                C_val = int(row["C"].iloc[0])
                episode_hashes = row["episode_hashes"].iloc[0]
            else:
                A_val, B_val, C_val, episode_hashes = 0, 0, 0, []

            output_data[display_name][f"Scenario {scene}"] = f"{A_val} / {B_val} / {C_val}"
            hash_output_data[display_name][f"Scenario {scene}"] = (
                ", ".join(str(h) for h in episode_hashes) if episode_hashes else ""
            )

    output_df = pd.DataFrame(output_data).T
    output_df = output_df.reindex(columns=[f"Scenario {s}" for s in SCENES])
    output_df.index.name = "Operator"

    # Totals across all operators by scene.
    total_row = {}
    for scene in SCENES:
        total_A = 0
        total_B = 0
        total_C = 0
        for display_name in operator_order:
            cell_value = output_df.loc[display_name, f"Scenario {scene}"]
            parts = cell_value.split(" / ")
            if len(parts) == 3:
                total_A += int(parts[0])
                total_B += int(parts[1])
                total_C += int(parts[2])
        total_row[f"Scenario {scene}"] = f"{total_A} / {total_B} / {total_C}"

    output_df.loc["Total"] = total_row

    hash_output_df = pd.DataFrame(hash_output_data).T
    hash_output_df = hash_output_df.reindex(columns=[f"Scenario {s}" for s in SCENES])
    hash_output_df.index.name = "Operator"
    hash_output_df.loc["Total"] = {f"Scenario {s}": "" for s in SCENES}

    os.makedirs("results", exist_ok=True)
    output_path = "results/cup_on_saucer_data_completeness.csv"
    hashes_path = "results/cup_on_saucer_data_completeness_hashes.csv"
    output_df.to_csv(output_path)
    hash_output_df.to_csv(hashes_path)

    print("Saved:")
    print(f" - {output_path}")
    print(f" - {hashes_path}")
    print("\nPreview (stats):")
    print(output_df.head())


if __name__ == "__main__":
    main()

