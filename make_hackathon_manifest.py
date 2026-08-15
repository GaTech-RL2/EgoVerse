from pathlib import Path
import pandas as pd

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

SEED = 42
TOP_TASKS = 25
PER_TASK = 16
MIN_FRAMES = 80
MAX_FRAMES = 1200

engine = create_default_engine()
df = episode_table_to_df(engine).copy()

required = [
    "episode_hash", "lab", "task", "num_frames",
    "zarr_processed_path", "zarr_mp4_path",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

eligible = df[
    (df["lab"] == "microagi")
    & (df["embodiment"] == "human_bimanual")
    & (~df["is_deleted"])
    & (df["episode_hash"].notna())
    & (df["task"].notna())
    & (df["num_frames"].between(MIN_FRAMES, MAX_FRAMES))
    & (df["zarr_processed_path"].notna() | df["zarr_mp4_path"].notna())
].copy()

task_sizes = eligible["task"].value_counts()
chosen_tasks = task_sizes.head(TOP_TASKS).index.tolist()

samples = []
for task in chosen_tasks:
    group = eligible[eligible["task"] == task]
    samples.append(
        group.sample(
            n=min(PER_TASK, len(group)),
            random_state=SEED,
        )
    )

manifest = pd.concat(samples, ignore_index=True)

columns = [
    "episode_hash",
    "lab",
    "task",
    "task_description",
    "scene",
    "objects",
    "embodiment",
    "num_frames",
    "is_eval",
    "eval_success",
    "zarr_processed_path",
    "zarr_mp4_path",
]
columns = [c for c in columns if c in manifest.columns]

Path("artifacts").mkdir(exist_ok=True)
manifest[columns].to_csv("artifacts/hackathon_manifest.csv", index=False)
manifest[columns].to_parquet("artifacts/hackathon_manifest.parquet", index=False)

print(f"Eligible episodes: {len(eligible):,}")
print(f"Tasks selected: {len(chosen_tasks)}")
print(f"Manifest episodes: {len(manifest)}")
print("\nEpisodes per task:")
print(manifest["task"].value_counts().sort_index().to_string())
print("\nFrame-count summary:")
print(manifest["num_frames"].describe().to_string())
