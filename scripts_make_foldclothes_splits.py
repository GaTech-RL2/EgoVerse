from pathlib import Path
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INFILE = Path("artifacts/foldclothes-v1/manifests/candidate_pool.csv")
OUTDIR = Path("artifacts/foldclothes-v1/manifests")

df = pd.read_csv(INFILE)

required = {"episode_hash", "task", "num_frames", "zarr_mp4_path", "zarr_processed_path"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")

if not df["episode_hash"].is_unique:
    raise RuntimeError("candidate_pool.csv contains duplicate episode_hash values")

df["canonical_task"] = "fold_clothes"

train, holdout = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["task"],
)

val, test = train_test_split(
    holdout,
    test_size=0.50,
    random_state=SEED,
    stratify=holdout["task"],
)

train = train.copy()
val = val.copy()
test = test.copy()
train["split"] = "train"
val["split"] = "val"
test["split"] = "test"

manifest = pd.concat([train, val, test], ignore_index=True)
manifest = manifest.sort_values(["split", "task", "episode_hash"]).reset_index(drop=True)

for name, part in manifest.groupby("split", sort=False):
    part.to_csv(OUTDIR / f"{name}.csv", index=False)
    part.to_parquet(OUTDIR / f"{name}.parquet", index=False)

manifest.to_csv(OUTDIR / "manifest.csv", index=False)
manifest.to_parquet(OUTDIR / "manifest.parquet", index=False)

sha256 = hashlib.sha256((OUTDIR / "manifest.csv").read_bytes()).hexdigest()
(OUTDIR / "manifest.sha256").write_text(f"{sha256}  manifest.csv\n")

assert len(manifest) == len(df)
assert manifest["episode_hash"].is_unique
assert manifest.groupby("episode_hash")["split"].nunique().max() == 1
assert set(manifest["canonical_task"]) == {"fold_clothes"}

print(f"Seed: {SEED}")
print(f"Total episodes: {len(manifest)}")
print(f"Manifest SHA-256: {sha256}")
print("\nBy split:")
print(manifest["split"].value_counts().reindex(["train", "val", "test"]).to_string())
print("\nBy split and original task:")
print(pd.crosstab(manifest["task"], manifest["split"])[["train", "val", "test"]].to_string())
print("\nFrame counts by split:")
print(manifest.groupby("split")["num_frames"].describe()[["count", "mean", "min", "50%", "max"]].to_string())
