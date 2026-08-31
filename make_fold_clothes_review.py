import json
from pathlib import Path

from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

TASK = "Fold_clothes"
OUT = Path("artifacts/foldclothes-v1/manifests/quicktime-five.json")

engine = create_default_engine()
df = episode_table_to_df(engine)

pool = df.loc[
    (df["task"] == TASK)
    & (~df["is_deleted"].fillna(False).astype(bool))
    & (df["zarr_mp4_path"].notna())
    & (df["zarr_processed_path"].notna())
].copy()

print(f"\nEligible episodes for task={TASK!r}: {len(pool)}")
print("\nBy lab:")
print(pool["lab"].value_counts(dropna=False).head(20).to_string())

print("\nBy embodiment:")
print(pool["embodiment"].value_counts(dropna=False).head(20).to_string())

print("\nFrame-count summary:")
print(pool["num_frames"].describe().to_string())

if len(pool) < 5:
    raise RuntimeError(f"Only found {len(pool)} valid episodes; stop and choose a broader exact task.")

# Mix sources if possible, but keep this first review reproducible.
sample = (
    pool.sample(n=5, random_state=42)
    [["episode_hash", "task", "task_description", "lab", "operator",
      "scene", "objects", "num_frames", "rig_name",
      "zarr_mp4_path", "zarr_processed_path"]]
    .to_dict(orient="records")
)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(sample, indent=2, default=str))

print(f"\nWrote {len(sample)} records to {OUT}")
print("\nReview IDs:")
for row in sample:
    print(
        f"{row['episode_hash']} | "
        f"lab={row['lab']} | frames={row['num_frames']} | "
        f"scene={row['scene']}"
    )
