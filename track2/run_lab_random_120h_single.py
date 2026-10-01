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


# Compare two labs
LAB_A = "mecka"
LAB_B = "scale"


# Single random seed for testing
RANDOM_SEED = 42


# Reduce visual computation cost
VISUAL_SAMPLE_SIZE = 100


# Output
RESULT_DIR = Path(
    "track2/results/single_random_120h"
)

MANIFEST_DIR = (
    RESULT_DIR /
    "manifests"
)


RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


    # Basic filtering

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



    # convert frames to hours

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
# RANDOM 120H SAMPLING WITHIN LAB
# ============================================================

def sample_lab_120h(
    df,
    lab,
    seed,
):

    print(
        f"\nSampling lab={lab}, seed={seed}"
    )


    pool = df[
        df["lab"] == lab
    ].copy()



    print(
        f"{lab} available hours: "
        f"{pool.duration_hours.sum():.2f}"
    )



    shuffled = (
        pool
        .sample(
            frac=1.0,
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



    index = np.where(
        shuffled["cum_hours"]
        >= TARGET_HOURS
    )[0][0]



    subset = shuffled.iloc[
        :index + 1
    ].copy()



    return subset





# ============================================================
# SAVE DATASET MANIFEST
# ============================================================

def save_manifest(
    subset,
    name,
    seed,
):

    columns = [
        "episode_hash",
        "lab",
        "task",
        "embodiment",
        "duration_hours",
        "zarr_mp4_path",
        "scene",
        "objects",
    ]


    columns = [
        c for c in columns
        if c in subset.columns
    ]



    output = subset[
        columns
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


    # -------------------------
    # Load metadata
    # -------------------------

    df = load_reference_dataset()



    # -------------------------
    # Initialize evaluator
    # -------------------------

    evaluator = EgoVerseDiversityEvaluator(

        reference_df=df,

        semantic_model=
            "all-MiniLM-L6-v2",

        semantic_threshold=
            0.80,


        visual_model=
            "facebook/dinov2-small",


        visual_frames=
            5,


        reference_visual_distance=
            0.647402822971344,
    )



    # -------------------------
    # Sample datasets
    # -------------------------

    mecka_subset = sample_lab_120h(
        df,
        LAB_A,
        RANDOM_SEED,
    )


    scale_subset = sample_lab_120h(
        df,
        LAB_B,
        RANDOM_SEED,
    )



    print("\nDataset sizes:")


    print(
        f"Mecka: "
        f"{len(mecka_subset)} episodes "
        f"{mecka_subset.duration_hours.sum():.3f} hours"
    )


    print(
        f"Scale: "
        f"{len(scale_subset)} episodes "
        f"{scale_subset.duration_hours.sum():.3f} hours"
    )



    # -------------------------
    # Save manifests
    # -------------------------

    save_manifest(
        mecka_subset,
        "mecka",
        RANDOM_SEED,
    )


    save_manifest(
        scale_subset,
        "scale",
        RANDOM_SEED,
    )



    # -------------------------
    # Diversity evaluation
    # -------------------------

    print(
        "\nRunning diversity evaluation..."
    )


    results = evaluator.compare(

        {

            "Dataset A — Mecka":
                mecka_subset,


            "Dataset B — Scale":
                scale_subset,

        },


        visual_sample_size=
            VISUAL_SAMPLE_SIZE,


        random_state=
            RANDOM_SEED,

    )



    # -------------------------
    # Save results
    # -------------------------

    result_path = (
        RESULT_DIR /
        "single_random_120h_results.csv"
    )



    results.to_csv(
        result_path,
        index=False,
    )



    print(
        "\nSaved:"
    )

    print(
        result_path
    )



    print(
        "\nFinal Results:"
    )


    print(
        results
    )



if __name__ == "__main__":
    main()