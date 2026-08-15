from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

INPUT = Path("artifacts/episode_embeddings.parquet")
OUT_DIR = Path("public/data")

SEED = 42
SUBSET_SIZE = 100
N_SEED_TRIALS = 20


def farthest_first(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Pick a diverse subset by repeatedly adding the least-covered episode."""
    rng = np.random.default_rng(seed)
    n = len(x)
    selected = [int(rng.integers(n))]
    best_similarity = x @ x[selected[0]]

    while len(selected) < min(k, n):
        candidate = int(np.argmin(best_similarity))
        selected.append(candidate)
        best_similarity = np.maximum(best_similarity, x @ x[candidate])

    return np.array(selected, dtype=int)


def subset_metrics(x_all: np.ndarray, indices: np.ndarray) -> dict:
    chosen = x_all[indices]

    # For every corpus episode: similarity to its closest selected representative.
    nearest_similarity = (x_all @ chosen.T).max(axis=1)
    coverage = float(nearest_similarity.mean())

    # Mean similarity among chosen pairs. Lower = fewer lookalike selections.
    sim = chosen @ chosen.T
    upper = sim[np.triu_indices(len(indices), k=1)]
    mean_pairwise_similarity = float(upper.mean())

    # Near-duplicates are defined relative to the selected set's 95th percentile,
    # only for an explanatory count—not the primary score.
    return {
        "coverage": coverage,
        "mean_pairwise_similarity": mean_pairwise_similarity,
        "nearest_similarity_p10": float(np.percentile(nearest_similarity, 10)),
        "nearest_similarity_p50": float(np.percentile(nearest_similarity, 50)),
        "nearest_similarity_p90": float(np.percentile(nearest_similarity, 90)),
        "pairwise_similarity_p95": float(np.percentile(upper, 95)),
    }


def task_counts(df: pd.DataFrame, indices: np.ndarray) -> dict[str, int]:
    return (
        df.iloc[indices]["task"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT).reset_index(drop=True)

    x = np.asarray(df["visual_embedding"].tolist(), dtype=np.float32)
    x = normalize(x)
    n = len(x)

    if n < SUBSET_SIZE:
        raise ValueError(f"Need at least {SUBSET_SIZE} episodes, got {n}.")

    # Baseline: average behavior over many random subsets, but keep seed 42 as
    # the exact baseline shown in the dashboard.
    random_trials = []
    for seed in range(N_SEED_TRIALS):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=SUBSET_SIZE, replace=False))
        random_trials.append(subset_metrics(x, idx))

    baseline_coverage = float(np.mean([m["coverage"] for m in random_trials]))
    baseline_redundancy = float(
        np.mean([m["mean_pairwise_similarity"] for m in random_trials])
    )

    rng = np.random.default_rng(SEED)
    random_idx = np.sort(rng.choice(n, size=SUBSET_SIZE, replace=False))
    optimized_idx = farthest_first(x, SUBSET_SIZE, SEED)

    random_metrics = subset_metrics(x, random_idx)
    optimized_metrics = subset_metrics(x, optimized_idx)

    # Score is indexed to random selection = 50. It rewards broad corpus
    # coverage and lower similarity among selected clips. The normalized
    # deltas are deliberately small because cosine similarities are high.
    def score(metrics: dict) -> float:
        coverage_delta = (
            (metrics["coverage"] - baseline_coverage) / baseline_coverage
        )
        redundancy_delta = (
            (baseline_redundancy - metrics["mean_pairwise_similarity"])
            / baseline_redundancy
        )
        raw = 50.0 + 500.0 * (0.5 * coverage_delta + 0.5 * redundancy_delta)
        return round(float(np.clip(raw, 0, 100)), 1)

    random_metrics["diversity_score"] = score(random_metrics)
    optimized_metrics["diversity_score"] = score(optimized_metrics)

    # Find clear lookalike examples within the random set for the dashboard.
    random_vectors = x[random_idx]
    random_sim = random_vectors @ random_vectors.T
    pairs = []
    for a, b in zip(*np.where(np.triu(random_sim, k=1) > 0)):
        pairs.append(
            {
                "episode_a": df.iloc[random_idx[a]]["episode_hash"],
                "task_a": df.iloc[random_idx[a]]["task"],
                "episode_b": df.iloc[random_idx[b]]["episode_hash"],
                "task_b": df.iloc[random_idx[b]]["task"],
                "similarity": round(float(random_sim[a, b]), 4),
            }
        )
    pairs.sort(key=lambda item: item["similarity"], reverse=True)

    # PCA is used only to draw a 2D map. All metrics above use 512 dimensions.
    coords = PCA(n_components=2, random_state=SEED).fit_transform(x)
    random_set = set(random_idx.tolist())
    optimized_set = set(optimized_idx.tolist())

    episodes = []
    for i, row in df.iterrows():
        episodes.append(
            {
                "episode_id": row["episode_hash"],
                "task": row["task"],
                "task_description": (
                    str(row["task_description"])
                    if pd.notna(row.get("task_description"))
                    else ""
                ),
                "num_frames": int(row["num_frames"]) if pd.notna(row["num_frames"]) else None,
                "x": round(float(coords[i, 0]), 5),
                "y": round(float(coords[i, 1]), 5),
                "in_random": i in random_set,
                "in_optimized": i in optimized_set,
            }
        )

    artifact = {
        "method": {
            "name": "Visual Interaction Diversity Score",
            "plain_english": (
                "A selected training subset scores well when it represents the visual "
                "patterns across the full corpus while avoiding repeated lookalike clips."
            ),
            "corpus_size": n,
            "subset_size": SUBSET_SIZE,
            "tasks_in_corpus": int(df["task"].nunique()),
            "embedding_model": str(df["embedding_model"].iloc[0]),
            "embedding_version": str(df["embedding_version"].iloc[0]),
            "frames_per_episode": int(df["frames_sampled"].iloc[0]),
            "selection_method": "farthest_first over cosine-normalized CLIP embeddings",
            "map_note": "The 2D map is PCA for visualization only; scores use all 512 dimensions.",
            "seed": SEED,
            "random_baseline_trials": N_SEED_TRIALS,
        },
        "baseline": {
            "mean_coverage": round(baseline_coverage, 4),
            "mean_pairwise_similarity": round(baseline_redundancy, 4),
        },
        "scores": {
            "random": {k: round(v, 4) if isinstance(v, float) else v for k, v in random_metrics.items()},
            "optimized": {k: round(v, 4) if isinstance(v, float) else v for k, v in optimized_metrics.items()},
        },
        "task_audit": {
            "corpus": task_counts(df, np.arange(n)),
            "random": task_counts(df, random_idx),
            "optimized": task_counts(df, optimized_idx),
        },
        "lookalike_examples_in_random": pairs[:10],
        "episodes": episodes,
        "selected_episode_ids": {
            "random": df.iloc[random_idx]["episode_hash"].tolist(),
            "optimized": df.iloc[optimized_idx]["episode_hash"].tolist(),
        },
    }

    out = OUT_DIR / "diversity_analysis.json"
    out.write_text(json.dumps(artifact, indent=2))

    print(f"Wrote {out}")
    print("\nRandom subset")
    print(json.dumps(artifact["scores"]["random"], indent=2))
    print("\nOptimized subset")
    print(json.dumps(artifact["scores"]["optimized"], indent=2))
    print("\nRandom-baseline average over 20 seeds")
    print(json.dumps(artifact["baseline"], indent=2))
    print(f"\nTop lookalike similarity in random: {pairs[0]['similarity']:.4f}")
    print(f"Task coverage — random: {len(artifact['task_audit']['random'])}, optimized: {len(artifact['task_audit']['optimized'])}")

if __name__ == "__main__":
    main()
