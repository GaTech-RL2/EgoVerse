from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path("artifacts/foldclothes-v1")
CORPUS = ROOT / "train_episode_embeddings.parquet"
VARIANT_DIR = ROOT / "manifests" / "curation_variants"
OUT = ROOT / "curation_variant_scores.json"

SEED = 42
N_SEED_TRIALS = 20
VARIANT_SIZE = 774
VARIANTS = ("random-774", "duration-balanced-774", "diversity-774")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def subset_metrics(x_all: np.ndarray, indices: np.ndarray) -> dict:
    chosen = x_all[indices]
    nearest_similarity = (x_all @ chosen.T).max(axis=1)
    coverage = float(nearest_similarity.mean())
    sim = chosen @ chosen.T
    upper = sim[np.triu_indices(len(indices), k=1)]
    return {
        "coverage": coverage,
        "mean_pairwise_similarity": float(upper.mean()),
        "nearest_similarity_p10": float(np.percentile(nearest_similarity, 10)),
        "nearest_similarity_p50": float(np.percentile(nearest_similarity, 50)),
        "nearest_similarity_p90": float(np.percentile(nearest_similarity, 90)),
        "pairwise_similarity_p95": float(np.percentile(upper, 95)),
    }


def main() -> None:
    df = pd.read_parquet(CORPUS).reset_index(drop=True)
    if not df["episode_hash"].is_unique:
        raise RuntimeError("train embedding table has duplicate episode_hash values")

    x = l2_normalize(np.stack(df["visual_embedding"].map(np.asarray)).astype(np.float32))
    hash_to_idx = {episode_hash: i for i, episode_hash in enumerate(df["episode_hash"])}
    n = len(df)
    if n < VARIANT_SIZE:
        raise RuntimeError(f"Need at least {VARIANT_SIZE} train embeddings, got {n}")

    random_trials = []
    for seed in range(N_SEED_TRIALS):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=VARIANT_SIZE, replace=False))
        random_trials.append(subset_metrics(x, idx))

    baseline_coverage = float(np.mean([m["coverage"] for m in random_trials]))
    baseline_redundancy = float(
        np.mean([m["mean_pairwise_similarity"] for m in random_trials])
    )

    def score(metrics: dict) -> float:
        coverage_delta = (metrics["coverage"] - baseline_coverage) / baseline_coverage
        redundancy_delta = (
            (baseline_redundancy - metrics["mean_pairwise_similarity"])
            / baseline_redundancy
        )
        raw = 50.0 + 500.0 * (0.5 * coverage_delta + 0.5 * redundancy_delta)
        return round(float(np.clip(raw, 0, 100)), 1)

    runs = {}
    for name in VARIANTS:
        variant = pd.read_csv(VARIANT_DIR / f"{name}.csv")
        if len(variant) != VARIANT_SIZE:
            raise RuntimeError(f"{name}: expected {VARIANT_SIZE} rows, got {len(variant)}")
        missing = [h for h in variant["episode_hash"] if h not in hash_to_idx]
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} hashes missing from train embeddings")
        idx = np.array([hash_to_idx[h] for h in variant["episode_hash"]], dtype=int)
        metrics = subset_metrics(x, idx)
        metrics["diversity_score"] = score(metrics)
        metrics["n_episodes"] = int(len(variant))
        metrics["task_counts"] = (
            variant["task"].value_counts().sort_index().astype(int).to_dict()
        )
        runs[name] = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
        }

    artifact = {
        "method": {
            "name": "Visual Interaction Diversity Score",
            "corpus": "foldclothes-v1 train pool",
            "corpus_size": n,
            "subset_size": VARIANT_SIZE,
            "embedding_model": str(df["embedding_model"].iloc[0]),
            "embedding_version": str(df["embedding_version"].iloc[0]),
            "seed": SEED,
            "random_baseline_trials": N_SEED_TRIALS,
            "plain_english": (
                "A selected training subset scores well when it represents the visual "
                "patterns across the full train pool while avoiding repeated lookalike clips."
            ),
        },
        "baseline": {
            "mean_coverage": round(baseline_coverage, 4),
            "mean_pairwise_similarity": round(baseline_redundancy, 4),
        },
        "scores": runs,
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {OUT}")
    print(json.dumps(artifact["scores"], indent=2))


if __name__ == "__main__":
    main()
