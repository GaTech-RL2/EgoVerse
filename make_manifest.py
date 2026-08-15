from pathlib import Path
import pandas as pd

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

N_PER_GROUP = 12
MAX_EPISODES = 360
SEED = 42

engine = create_default_engine()
df = episode_table_to_df(engine).copy()

print("Total catalog rows:", len(df))
print("Columns:", df.columns.tolist())

# Use only rows with an ID; normalize text columns for grouping.
id_col = "episode_hash"
if id_col not in df.columns:
    raise RuntimeError(f"Expected {id_col!r}; available: {df.columns.tolist()}")

df = df[df[id_col].notna()].copy()

for col in ["lab", "task", "scene"]:
    if col not in df.columns:
        df[col] = "unknown"
    df[col] = df[col].fillna("unknown").astype(str)

# Broad, stratified sample: some examples from each lab/task group.
grouped = (
    df.groupby(["lab", "task"], group_keys=False)
      .apply(lambda g: g.sample(n=min(len(g), N_PER_GROUP), random_state=SEED))
      .reset_index(drop=True)
)

# If there are too many groups, take a reproducible final sample.
if len(grouped) > MAX_EPISODES:
    grouped = grouped.sample(n=MAX_EPISODES, random_state=SEED)

out_cols = [c for c in [
    "episode_hash", "lab", "task", "scene", "embodiment",
    "num_frames", "fps", "duration",
] if c in grouped.columns]

Path("artifacts").mkdir(exist_ok=True)
grouped[out_cols].to_csv("artifacts/episode_manifest.csv", index=False)

print(f"\nWrote {len(grouped)} episodes → artifacts/episode_manifest.csv")
print("\nManifest mix by lab:")
print(grouped["lab"].value_counts().to_string())
print("\nManifest mix by task:")
print(grouped["task"].value_counts().head(30).to_string())
