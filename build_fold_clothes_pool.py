from pathlib import Path

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

TASKS = [
    "fold_clothes",
    "fold_laundry",
    "fold_shirt",
    "fold_black_t-shirt",
    "fold_white_shirt",
    "fold_blue_jeans",
]

MAX_PER_TASK = 350
MIN_FRAMES = 150
MAX_FRAMES = 2_000
OUT = Path("artifacts/foldclothes-v1/manifests/candidate_pool.csv")

engine = create_default_engine()
df = episode_table_to_df(engine)

pool = df.loc[
    (df["lab"] == "microagi")
    & (df["embodiment"] == "human_bimanual")
    & (df["task"].isin(TASKS))
    & (~df["is_deleted"].fillna(False).astype(bool))
    & (df["zarr_mp4_path"].notna())
    & (df["zarr_processed_path"].notna())
    & (df["num_frames"].between(MIN_FRAMES, MAX_FRAMES))
].copy()

parts = []
for task in TASKS:
    task_rows = pool.loc[pool["task"] == task].sample(
        n=min(MAX_PER_TASK, int((pool["task"] == task).sum())),
        random_state=42,
    )
    parts.append(task_rows)

selected = (
    __import__("pandas")
    .concat(parts, ignore_index=True)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

columns = [
    "episode_hash",
    "task",
    "task_description",
    "lab",
    "embodiment",
    "num_frames",
    "objects",
    "rig_name",
    "zarr_mp4_path",
    "zarr_processed_path",
]
selected.loc[:, columns].to_csv(OUT, index=False)

print(f"Wrote {len(selected)} episodes to {OUT}")
print("\nCounts by task:")
print(selected["task"].value_counts().sort_index().to_string())
print("\nFrame summary:")
print(selected["num_frames"].describe().to_string())
