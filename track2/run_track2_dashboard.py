# EgoVerse Track 2 — score + statistical robustness + visual dashboard.
#
# Builds on run_track2.py's deterministic 120h subsets (the "final" method
# per README.md) and adds three things that raw run_track2.py doesn't have:
#   1. Bootstrap 95% CIs + a win-rate / probability-of-superiority estimate,
#      so the ranking isn't just a single point estimate.
#   2. A 2D PCA projection of the DINOv2 visual embeddings, i.e. an actual
#      non-text representation of "how diverse" each dataset looks.
#   3. A single dashboard.png comparing both datasets across all of the above.
#
# Bootstrap resampling reuses cached behavior-cluster lookups and cached
# visual embeddings (see diversity_evaluator.py caches), so after the first
# run — which still has to embed each video with DINOv2 once — reruns and
# the bootstrap itself are fast (pure numpy/pandas, no model inference).
#
# Run from the EgoVerse repo root:
#   conda activate emimic
#   source ~/.egoverse_env
#   python -m track2.run_track2_dashboard

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from track2.diversity_evaluator import (
    EgoVerseDiversityEvaluator,
    categorical_diversity,
    clean_series,
)
from track2.run_track2 import (
    DATASET_A_LAB,
    DATASET_B_LAB,
    TARGET_HOURS,
    deterministic_hour_subset,
    load_reference_dataset,
)

VISUAL_SAMPLE_SIZE = 200
RANDOM_STATE = 42
N_BOOTSTRAP = 1000
CI = 0.95

RESULTS_DIR = Path("track2/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_scores(evaluator, subset_df, visual_embeddings, n_bootstrap, seed):
    """Episode-level bootstrap over behavior/embodiment, and row-level
    bootstrap over the already-embedded visual sample. No re-embedding."""
    rng = np.random.default_rng(seed)
    n_episodes = len(subset_df)
    n_visual = len(visual_embeddings)

    tasks = subset_df["task"].to_numpy()
    embodiments = subset_df["embodiment"].to_numpy()
    ref_embodiment_richness = evaluator.embodiment.reference_embodiment_richness
    ref_visual_distance = evaluator.reference_visual_distance

    out = {k: np.empty(n_bootstrap) for k in ("overall", "behavior", "context", "embodiment")}

    for i in range(n_bootstrap):
        ep_idx = rng.integers(0, n_episodes, n_episodes)
        b = evaluator.behavior.score(pd.Series(tasks[ep_idx]))["diversity"]
        e = categorical_diversity(pd.Series(embodiments[ep_idx]), ref_embodiment_richness)["diversity"]

        vis_idx = rng.integers(0, n_visual, n_visual)
        d = cosine_distances(visual_embeddings[vis_idx])
        upper = d[np.triu_indices_from(d, k=1)]
        raw = float(upper.mean()) if len(upper) else 0.0
        c = float(np.clip(raw / ref_visual_distance, 0.0, 1.0))

        out["behavior"][i] = b
        out["context"][i] = c
        out["embodiment"][i] = e
        out["overall"][i] = (b + c + e) / 3.0

    return out


def ci_table(bootstrap_by_dataset):
    rows = []
    lo_pct, hi_pct = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    for name, comp in bootstrap_by_dataset.items():
        for metric, vals in comp.items():
            lo, hi = np.percentile(vals, [lo_pct, hi_pct])
            rows.append(
                {
                    "subset": name,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    return pd.DataFrame(rows)


def plot_dashboard(results_df, ci_df, embeddings_by_dataset, evaluator, subsets, name_a, name_b, win_rate_b, out_path):
    colors = {name_a: "#4C72B0", name_b: "#DD8452"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    width = 0.35

    # Panel 1: per-dimension scores
    ax = axes[0, 0]
    metrics = ["behavior_diversity", "context_visual_diversity", "embodiment_diversity", "overall_diversity"]
    labels = ["Behavior", "Context", "Embodiment", "Overall"]
    x = np.arange(len(metrics))
    for i, name in enumerate([name_a, name_b]):
        row = results_df[results_df["subset"] == name].iloc[0]
        ax.bar(x + (i - 0.5) * width, [row[m] for m in metrics], width, label=name, color=colors[name])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Diversity scores by dimension")
    ax.legend()

    # Panel 2: bootstrap 95% CI for overall diversity
    ax = axes[0, 1]
    for i, name in enumerate([name_a, name_b]):
        row = ci_df[(ci_df["subset"] == name) & (ci_df["metric"] == "overall")].iloc[0]
        ax.bar(
            i,
            row["mean"],
            yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
            capsize=8,
            color=colors[name],
            width=0.5,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([name_a, name_b])
    ax.set_ylim(0, 1)
    ax.set_title(f"Overall diversity, {int(CI*100)}% bootstrap CI\nP({name_b} > {name_a}) = {win_rate_b:.0%}")

    # Panel 3: PCA of visual embeddings — non-text representation of diversity
    ax = axes[1, 0]
    all_emb = np.concatenate(list(embeddings_by_dataset.values()), axis=0)
    owner = sum([[name] * len(emb) for name, emb in embeddings_by_dataset.items()], [])
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(all_emb)
    owner = np.array(owner)
    for name in [name_a, name_b]:
        mask = owner == name
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.6, label=name, color=colors[name])
    ax.set_title("Visual embedding space (DINOv2, PCA to 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    # Panel 4: top behavior clusters, episode counts
    ax = axes[1, 1]
    top_n = 12
    cluster_counts = {
        name: evaluator.behavior.transform(subset["task"]).astype(str).value_counts()
        for name, subset in subsets.items()
    }
    combined = pd.concat(cluster_counts.values()).groupby(level=0).sum().sort_values(ascending=False)
    top_clusters = combined.head(top_n).index
    xc = np.arange(len(top_clusters))
    for i, name in enumerate([name_a, name_b]):
        vals = [cluster_counts[name].get(c, 0) for c in top_clusters]
        ax.bar(xc + (i - 0.5) * width, vals, width, label=name, color=colors[name])
    ax.set_xticks(xc)
    ax.set_xticklabels([str(c)[:8] for c in top_clusters], rotation=45, ha="right")
    ax.set_title(f"Top {top_n} behavior clusters, episode counts")
    ax.legend()

    fig.suptitle(f"Track 2 Diversity Dashboard — {name_a} vs {name_b}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    core_df = load_reference_dataset()

    dataset_a = deterministic_hour_subset(core_df, DATASET_A_LAB, TARGET_HOURS)
    dataset_b = deterministic_hour_subset(core_df, DATASET_B_LAB, TARGET_HOURS)

    name_a = f"Dataset A — {DATASET_A_LAB.title()}"
    name_b = f"Dataset B — {DATASET_B_LAB.title()}"
    subsets = {name_a: dataset_a, name_b: dataset_b}

    evaluator = EgoVerseDiversityEvaluator(
        reference_df=core_df,
        semantic_model="all-MiniLM-L6-v2",
        semantic_threshold=0.80,
        visual_model="facebook/dinov2-small",
        visual_frames=5,
        reference_visual_distance=0.647402822971344,
    )

    print("\nComputing point-estimate scores...")
    results = evaluator.compare(subsets, visual_sample_size=VISUAL_SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(
        results[
            ["rank", "subset", "overall_diversity", "behavior_diversity", "context_visual_diversity", "embodiment_diversity"]
        ].to_string(index=False)
    )

    available = [clean_series(s["zarr_mp4_path"]).drop_duplicates().shape[0] for s in subsets.values()]
    fair_visual_n = min(int(VISUAL_SAMPLE_SIZE), min(available))
    visual = evaluator._get_visual()

    print(f"\nCollecting matched visual embeddings (n={fair_visual_n} per dataset)...")
    embeddings_by_dataset = {}
    for name, subset in subsets.items():
        paths = clean_series(subset["zarr_mp4_path"]).drop_duplicates().sample(n=fair_visual_n, random_state=RANDOM_STATE)
        embeddings_by_dataset[name] = np.stack([visual.episode_embedding(p) for p in paths])

    print(f"\nRunning bootstrap ({N_BOOTSTRAP} resamples per dataset)...")
    bootstrap_by_dataset = {
        name: bootstrap_scores(evaluator, subsets[name], embeddings_by_dataset[name], N_BOOTSTRAP, seed=RANDOM_STATE + i)
        for i, name in enumerate(subsets)
    }

    win_rate_b = float((bootstrap_by_dataset[name_b]["overall"] > bootstrap_by_dataset[name_a]["overall"]).mean())
    ci_df = ci_table(bootstrap_by_dataset)

    print(f"\nP({name_b} > {name_a}) across independent bootstraps: {win_rate_b:.1%}")
    print(ci_df[ci_df["metric"] == "overall"].to_string(index=False))

    results_path = RESULTS_DIR / "dashboard_scores.csv"
    ci_path = RESULTS_DIR / "dashboard_bootstrap_ci.csv"
    dashboard_path = RESULTS_DIR / "dashboard.png"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    ci_df.to_csv(ci_path, index=False)
    plot_dashboard(results, ci_df, embeddings_by_dataset, evaluator, subsets, name_a, name_b, win_rate_b, dashboard_path)

    print("\nSaved:")
    print(results_path)
    print(ci_path)
    print(dashboard_path)


if __name__ == "__main__":
    main()
