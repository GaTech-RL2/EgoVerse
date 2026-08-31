from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
PER_TASK = 129
ROOT = Path("artifacts/foldclothes-v1")
INFILE = ROOT / "train_episode_embeddings.parquet"
OUTDIR = ROOT / "manifests" / "curation_variants"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INFILE).copy()
tasks = sorted(df["task"].unique())

assert len(tasks) == 6
assert all((df["task"] == task).sum() >= PER_TASK for task in tasks)
assert df["episode_hash"].is_unique

base_cols = [c for c in df.columns if c != "visual_embedding"]

def choose_random(group: pd.DataFrame) -> pd.DataFrame:
    return group.sample(n=PER_TASK, random_state=SEED)

def choose_duration_balanced(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values(["num_frames", "episode_hash"]).copy()
    bins = np.array_split(group.index.to_numpy(), 3)

    quotas = [PER_TASK // 3] * 3
    for i in range(PER_TASK % 3):
        quotas[i] += 1

    picks = []
    for i, (idx, quota) in enumerate(zip(bins, quotas)):
        band = group.loc[idx]
        if len(band) < quota:
            raise RuntimeError(f"{group['task'].iloc[0]} band {i} has {len(band)} rows, needs {quota}")
        picks.append(band.sample(n=quota, random_state=SEED + i))

    return pd.concat(picks, ignore_index=False)

def choose_diverse(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("episode_hash").reset_index(drop=True)
    X = np.stack(group["visual_embedding"].map(np.asarray)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    n = len(group)
    chosen = [0]
    min_distance = 1.0 - (X @ X[0])

    for _ in range(1, PER_TASK):
        candidate = int(np.argmax(min_distance))
        chosen.append(candidate)
        distance = 1.0 - (X @ X[candidate])
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen] = -np.inf

    return group.iloc[chosen].copy()

def build_variant(name: str, selector):
    pieces = []
    for task in tasks:
        group = df[df["task"] == task].copy()
        selection = selector(group)
        if len(selection) != PER_TASK:
            raise RuntimeError(f"{name}/{task}: selected {len(selection)}, expected {PER_TASK}")
        selection["curation_variant"] = name
        pieces.append(selection)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["task", "episode_hash"]).reset_index(drop=True)

    assert len(out) == PER_TASK * len(tasks)
    assert out["episode_hash"].is_unique
    assert (out.groupby("task").size() == PER_TASK).all()

    csv_path = OUTDIR / f"{name}.csv"
    parquet_path = OUTDIR / f"{name}.parquet"

    out.drop(columns=["visual_embedding"]).to_csv(csv_path, index=False)
    out.to_parquet(parquet_path, index=False)

    print(f"\n{name}: {len(out)} episodes")
    print(out.groupby("task").size().to_string())
    print("Frame count by task:")
    print(out.groupby("task")["num_frames"].agg(["min", "median", "max"]).to_string())
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {parquet_path}")

build_variant("random-774", choose_random)
build_variant("duration-balanced-774", choose_duration_balanced)
build_variant("diversity-774", choose_diverse)
