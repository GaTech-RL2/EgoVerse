from pathlib import Path

import numpy as np
import pandas as pd

from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)

from track2.diversity_evaluator import (
    EgoVerseDiversityEvaluator,
)


# ============================================================
# CONFIG
# ============================================================

FPS = 30.0

TARGET_HOURS = 120.0

LAB_A = "mecka"
LAB_B = "scale"

# Robustness seeds
RANDOM_SEEDS = [
    0,
    1,
    2,
    3,
    42,
]

# Reduce visual computation cost
VISUAL_SAMPLE_SIZE = 100


RESULT_DIR = Path(
    "track2/results"
)

MANIFEST_DIR = (
    RESULT_DIR /
    "manifests"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_reference_dataset():

    print(
        "Connecting to EgoVerse metadata..."
    )

    engine = create_default_engine()

    df = episode_table_to_df(
        engine
    )

    print(
        f"Raw episodes: {len(df):,}"
    )


    df = df[
        (df["is_deleted"] == False)
        &
        df["task"].notna()
        &
        df["lab"].notna()
        &
        df["embodiment"].notna()
        &
        df["num_frames"].notna()
    ].copy()


    for col in [
        "task",
        "lab",
        "embodiment",
    ]:

        df = df[
            df[col]
            .astype(str)
            .str.strip()
            .ne("")
        ]


    df = df[
        df["num_frames"] > 0
    ].copy()


    df["duration_hours"] = (
        df["num_frames"]
        /
        FPS
        /
        3600
    )


    print(
        f"Usable episodes: {len(df):,}"
    )


    print(
        f"Total hours: "
        f"{df.duration_hours.sum():.2f}"
    )


    return df



# ============================================================
# RANDOM LAB SAMPLING
# ============================================================

def sample_lab_120h(
    df,
    lab,
    seed,
):

    print(
        f"Sampling {lab}, seed={seed}"
    )


    pool = df[
        df["lab"] == lab
    ].copy()


    shuffled = (
        pool
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(drop=True)
    )


    shuffled[
        "cum_hours"
    ] = (
        shuffled[
            "duration_hours"
        ]
        .cumsum()
    )


    idx = np.where(
        shuffled.cum_hours >= TARGET_HOURS
    )[0][0]


    subset = shuffled.iloc[
        :idx+1
    ].copy()


    return subset



# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    subset,
    name,
    seed,
):

    cols = [
        "episode_hash",
        "lab",
        "task",
        "embodiment",
        "duration_hours",
        "zarr_mp4_path",
        "scene",
        "objects",
    ]


    output = subset[
        [
            c for c in cols
            if c in subset.columns
        ]
    ]


    path = (
        MANIFEST_DIR /
        f"{name}_seed_{seed}.csv"
    )


    output.to_csv(
        path,
        index=False,
    )


    print(
        "Saved manifest:",
        path
    )



# ============================================================
# MAIN
# ============================================================

def main():

    df = load_reference_dataset()


    evaluator = EgoVerseDiversityEvaluator(

        reference_df=df,

        semantic_model=
            "all-MiniLM-L6-v2",

        semantic_threshold=
            0.8,

        visual_model=
            "facebook/dinov2-small",

        visual_frames=
            5,

        reference_visual_distance=
            0.647402822971344,
    )


    all_results = []


    for seed in RANDOM_SEEDS:


        print(
            "\n"
            + "="*70
        )

        print(
            f"RUNNING SEED {seed}"
        )

        print(
            "="*70
        )


        mecka = sample_lab_120h(
            df,
            LAB_A,
            seed,
        )


        scale = sample_lab_120h(
            df,
            LAB_B,
            seed,
        )


        print(
            f"Mecka:"
            f" {len(mecka)} episodes "
            f"{mecka.duration_hours.sum():.2f}h"
        )


        print(
            f"Scale:"
            f" {len(scale)} episodes "
            f"{scale.duration_hours.sum():.2f}h"
        )



        # Save exact dataset definition
        save_manifest(
            mecka,
            "mecka",
            seed,
        )


        save_manifest(
            scale,
            "scale",
            seed,
        )



        result = evaluator.compare(

            {
                "Dataset A — Mecka":
                    mecka,

                "Dataset B — Scale":
                    scale,
            },

            visual_sample_size=
                VISUAL_SAMPLE_SIZE,

            random_state=
                seed,
        )


        result["seed"] = seed


        all_results.append(
            result
        )



    final = pd.concat(
        all_results,
        ignore_index=True,
    )


    # ========================================================
    # SAVE ALL RESULTS
    # ========================================================

    final_path = (
        RESULT_DIR /
        "lab_random_120h_all_runs.csv"
    )


    final.to_csv(
        final_path,
        index=False,
    )


    print(
        "Saved:",
        final_path
    )



    # ========================================================
    # SUMMARY
    # ========================================================

    metrics = [
        "overall_diversity",
        "behavior_diversity",
        "context_visual_diversity",
        "embodiment_diversity",
    ]


    summary = (
        final
        .groupby(
            "subset"
        )[metrics]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )


    summary_path = (
        RESULT_DIR /
        "lab_random_120h_summary.csv"
    )


    summary.to_csv(
        summary_path
    )


    print(
        summary
    )



    # ========================================================
    # PAIRWISE WIN
    # ========================================================

    pivot = final.pivot(

        index="seed",

        columns="subset",

        values=
        "overall_diversity"

    )


    pivot[
        "Scale_minus_Mecka"
    ] = (
        pivot[
            "Dataset B — Scale"
        ]
        -
        pivot[
            "Dataset A — Mecka"
        ]
    )


    pivot[
        "Scale_wins"
    ] = (
        pivot[
            "Scale_minus_Mecka"
        ]
        > 0
    )


    pair_path = (
        RESULT_DIR /
        "lab_random_120h_pairwise.csv"
    )


    pivot.to_csv(
        pair_path
    )


    print(
        "\nScale wins:"
        f"{pivot.Scale_wins.sum()}"
        f"/{len(pivot)}"
    )


    print(
        "\nCompleted."
    )



if __name__ == "__main__":
    main()
