from pathlib import Path
import numpy as np
import pandas as pd

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df
from track2.diversity_evaluator import EgoVerseDiversityEvaluator

FPS = 30.0
TARGET_HOURS = 120.0

DATASET_A_LAB = "mecka"
DATASET_B_LAB = "scale"

VISUAL_SAMPLE_SIZE = 200
RANDOM_STATE = 42

RESULTS_DIR = Path("track2/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_reference_dataset():
    print("Connecting to EgoVerse metadata...")

    engine = create_default_engine()
    df = episode_table_to_df(engine)

    print(f"Raw episodes: {len(df):,}")

    core_df = df[
        (df["is_deleted"] == False)
        & df["task"].notna()
        & df["lab"].notna()
        & df["embodiment"].notna()
        & df["num_frames"].notna()
    ].copy()

    for col in ["task", "lab", "embodiment"]:
        core_df = core_df[
            core_df[col].astype(str).str.strip().ne("")
        ]

    core_df = core_df[core_df["num_frames"] > 0].copy()

    core_df["duration_hours"] = (
        core_df["num_frames"] / FPS / 3600.0
    )

    print(f"Usable reference episodes: {len(core_df):,}")
    print(f"Estimated reference hours: {core_df['duration_hours'].sum():,.2f}")

    return core_df


def deterministic_hour_subset(df, lab, target_hours=TARGET_HOURS):
    """
    Non-random, reproducible dataset construction:
    sort by created_at -> episode_hash and take episodes until
    cumulative duration first exceeds target_hours.
    """
    pool = df[df["lab"] == lab].copy()

    if pool.empty:
        raise ValueError(f"No episodes found for lab={lab!r}")

    pool = pool.sort_values(
        ["created_at", "episode_hash"],
        kind="mergesort",
        na_position="last",
    ).copy()

    pool["cumulative_hours"] = pool["duration_hours"].cumsum()

    meets_target = pool["cumulative_hours"] >= target_hours

    if not meets_target.any():
        raise ValueError(
            f"{lab} has only {pool['duration_hours'].sum():.2f} hours."
        )

    cutoff_pos = int(
        np.flatnonzero(meets_target.to_numpy())[0]
    )

    return pool.iloc[: cutoff_pos + 1].copy()


def dataset_summary(name, subset):
    videos = (
        subset["zarr_mp4_path"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    videos = videos[videos != ""]

    return {
        "dataset": name,
        "episodes": len(subset),
        "hours": float(subset["duration_hours"].sum()),
        "raw_unique_tasks": int(subset["task"].nunique()),
        "labs": int(subset["lab"].nunique()),
        "embodiments": int(subset["embodiment"].nunique()),
        "unique_video_paths": int(videos.nunique()),
    }


def main():
    core_df = load_reference_dataset()

    dataset_a = deterministic_hour_subset(
        core_df,
        DATASET_A_LAB,
        TARGET_HOURS,
    )

    dataset_b = deterministic_hour_subset(
        core_df,
        DATASET_B_LAB,
        TARGET_HOURS,
    )

    name_a = f"Dataset A — {DATASET_A_LAB.title()}"
    name_b = f"Dataset B — {DATASET_B_LAB.title()}"

    summary_df = pd.DataFrame([
        dataset_summary(name_a, dataset_a),
        dataset_summary(name_b, dataset_b),
    ])

    print("\nFINAL DATASETS")
    print(summary_df.to_string(index=False))

    if summary_df["hours"].min() <= 100.0:
        raise RuntimeError(
            "Each final dataset must exceed 100 hours."
        )

    if summary_df["unique_video_paths"].min() < VISUAL_SAMPLE_SIZE:
        raise RuntimeError(
            "Not enough MP4 episodes for matched visual evaluation."
        )

    evaluator = EgoVerseDiversityEvaluator(
        reference_df=core_df,
        semantic_model="all-MiniLM-L6-v2",
        semantic_threshold=0.80,
        visual_model="facebook/dinov2-small",
        visual_frames=5,
        reference_visual_distance=0.647402822971344,
    )

    results = evaluator.compare(
        {
            name_a: dataset_a,
            name_b: dataset_b,
        },
        visual_sample_size=VISUAL_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    results_path = RESULTS_DIR / "final_two_dataset_results.csv"
    summary_path = RESULTS_DIR / "final_dataset_summary.csv"

    results.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nFINAL TRACK 2 RESULTS")
    print(
        results[
            [
                "rank",
                "subset",
                "overall_diversity",
                "behavior_diversity",
                "context_visual_diversity",
                "embodiment_diversity",
                "context_relative_spread",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
