"""
Visualise DemInf curation scores from a previously saved curation run.

Usage:
    python egomimic/scripts/curation/visualize_scores.py \\
        --scores     curation_results/scores.json \\
        --stats      curation_results/curation_stats.json \\
        --kept       curation_results/kept_hashes.json \\
        --output-dir curation_results/

Generates:
    score_distribution.png   — histogram with threshold line
    length_vs_score.png      — scatter: episode length vs MI score
    embodiment_boxplot.png   — box plot grouped by embodiment
    top_bottom.txt           — top-5 and bottom-5 episode hashes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_json(path: Path) -> object:
    with open(path) as f:
        return json.load(f)


def plot_score_distribution(
    scores: dict[str, float],
    threshold: float,
    output_path: Path,
) -> None:
    """Histogram of MI scores with vertical threshold line."""
    import matplotlib.pyplot as plt

    values = list(scores.values())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(
        threshold,
        color="crimson",
        linewidth=2,
        linestyle="--",
        label=f"threshold = {threshold:.4f}",
    )
    ax.set_xlabel("MI Score (nats)", fontsize=13)
    ax.set_ylabel("Number of Episodes", fontsize=13)
    ax.set_title("DemInf: Episode MI Score Distribution", fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_length_vs_score(
    scores: dict[str, float],
    episode_lengths: dict[str, int],
    threshold: float,
    output_path: Path,
) -> None:
    """Scatter plot of episode length vs MI score."""
    import matplotlib.pyplot as plt

    hashes = [h for h in scores if h in episode_lengths]
    lengths = [episode_lengths[h] for h in hashes]
    mi_scores = [scores[h] for h in hashes]
    colours = ["steelblue" if scores[h] >= threshold else "lightcoral" for h in hashes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lengths, mi_scores, c=colours, alpha=0.6, s=20, edgecolors="none")
    ax.axhline(
        threshold, color="crimson", linewidth=1.5, linestyle="--", label="threshold"
    )
    ax.set_xlabel("Episode Length (timesteps)", fontsize=13)
    ax.set_ylabel("MI Score (nats)", fontsize=13)
    ax.set_title("Episode Length vs MI Score", fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_embodiment_boxplot(
    scores: dict[str, float],
    embodiments: dict[str, str],
    output_path: Path,
) -> None:
    """Box plot of MI score grouped by embodiment."""
    import matplotlib.pyplot as plt

    groups: dict[str, list[float]] = {}
    for h, score in scores.items():
        emb = embodiments.get(h, "unknown")
        groups.setdefault(emb, []).append(score)

    if not groups:
        print("No embodiment data available for box plot")
        return

    labels = sorted(groups)
    data = [groups[label] for label in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    colors = ["steelblue", "mediumseagreen", "mediumpurple", "coral", "gold"]
    for patch, color in zip(bp["boxes"], colors * 10):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("MI Score (nats)", fontsize=13)
    ax.set_title("MI Score Distribution by Embodiment", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_top_bottom(
    scores: dict[str, float],
    n: int = 5,
    output_path: Path | None = None,
) -> None:
    """Print (and optionally save) the top-n and bottom-n episode hashes."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:n]
    bottom = ranked[-n:][::-1]

    lines = []
    lines.append(f"\nTop {n} episodes by MI score:")
    for i, (h, s) in enumerate(top, 1):
        lines.append(f"  {i}. {h}  →  {s:.6f}")

    lines.append(f"\nBottom {n} episodes by MI score:")
    for i, (h, s) in enumerate(bottom, 1):
        lines.append(f"  {i}. {h}  →  {s:.6f}")

    text = "\n".join(lines)
    print(text)

    if output_path is not None:
        output_path.write_text(text + "\n")
        print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise DemInf curation scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores", required=True, help="Path to scores.json")
    parser.add_argument("--stats", required=True, help="Path to curation_stats.json")
    parser.add_argument("--kept", required=True, help="Path to kept_hashes.json")
    parser.add_argument(
        "--episode-lengths",
        default=None,
        help="Optional path to JSON {hash: length} (for scatter plot)",
    )
    parser.add_argument(
        "--embodiments",
        default=None,
        help="Optional path to JSON {hash: embodiment} (for box plot)",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    parser.add_argument(
        "--top-n", type=int, default=5, help="Number of top/bottom episodes to print"
    )
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        print("ERROR: matplotlib is required for visualisation.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = _load_json(Path(args.scores))
    stats: dict = _load_json(Path(args.stats))
    kept_hashes: list[str] = _load_json(Path(args.kept))

    threshold = stats.get("threshold_score", float("nan"))
    kept_set = set(kept_hashes)

    print(f"\nLoaded {len(scores)} scored episodes, {len(kept_set)} kept")

    # 1. Score distribution histogram
    plot_score_distribution(
        scores=scores,
        threshold=threshold,
        output_path=output_dir / "score_distribution.png",
    )

    # 2. Length vs score scatter (if lengths available)
    if args.episode_lengths:
        episode_lengths: dict[str, int] = _load_json(Path(args.episode_lengths))
        plot_length_vs_score(
            scores=scores,
            episode_lengths=episode_lengths,
            threshold=threshold,
            output_path=output_dir / "length_vs_score.png",
        )
    else:
        print("Skipping length_vs_score.png — provide --episode-lengths")

    # 3. Embodiment box plot (if embodiments available)
    if args.embodiments:
        embodiments: dict[str, str] = _load_json(Path(args.embodiments))
        plot_embodiment_boxplot(
            scores=scores,
            embodiments=embodiments,
            output_path=output_dir / "embodiment_boxplot.png",
        )
    else:
        print("Skipping embodiment_boxplot.png — provide --embodiments")

    # 4. Top / bottom episodes
    print_top_bottom(
        scores=scores,
        n=args.top_n,
        output_path=output_dir / "top_bottom.txt",
    )


if __name__ == "__main__":
    main()
