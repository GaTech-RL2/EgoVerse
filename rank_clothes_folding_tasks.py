from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

engine = create_default_engine()
df = episode_table_to_df(engine)

task = df["task"].fillna("").astype(str).str.lower()
description = df["task_description"].fillna("").astype(str).str.lower()

clothes_words = r"(shirt|t-shirt|tshirt|clothes|clothing|laundry|garment|jeans|pants|trousers|sweater|hoodie|shorts|dress|skirt)"
fold_words = r"\bfold"

pool = df.loc[
    task.str.contains(fold_words, regex=True)
    & (task.str.contains(clothes_words, regex=True) | description.str.contains(clothes_words, regex=True))
    & ~df["is_deleted"].fillna(False).astype(bool)
    & df["zarr_mp4_path"].notna()
    & df["zarr_processed_path"].notna()
].copy()

summary = (
    pool.groupby(["task", "lab", "embodiment"], dropna=False)
    .agg(
        episodes=("episode_hash", "count"),
        median_frames=("num_frames", "median"),
        min_frames=("num_frames", "min"),
        max_frames=("num_frames", "max"),
        operators=("operator", "nunique"),
        scenes=("scene", "nunique"),
    )
    .reset_index()
    .sort_values(["episodes", "operators"], ascending=False)
)

print("\nVALID GARMENT-FOLDING TASK POOLS")
print(summary.head(50).to_string(index=False))

print(f"\nTotal eligible episodes: {len(pool)}")
print(f"Distinct task labels: {pool['task'].nunique()}")
