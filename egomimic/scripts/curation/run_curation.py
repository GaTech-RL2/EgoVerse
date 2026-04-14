"""
DemInf curation CLI entry point.

Basic usage:
    python egomimic/scripts/curation/run_curation.py \\
        data.folder_path=/path/to/zarr_cache \\
        filter_ratio=0.3 \\
        output_path=./curation_results/

Filter by task before curating:
    python egomimic/scripts/curation/run_curation.py \\
        data.folder_path=/path/to/zarr_cache \\
        "data.filter_lambdas=[\"lambda row: row.get('task')=='fold_clothes'\"]" \\
        filter_ratio=0.3

Dry-run (counts episodes without scoring):
    python egomimic/scripts/curation/run_curation.py \\
        data.folder_path=/path/to/zarr_cache \\
        +dry_run=true

Outputs written to output_path/:
    scores.json              — {episode_hash: mi_score}
    kept_hashes.json         — episode hashes that passed curation
    curation_stats.json      — summary statistics
    score_distribution.png   — histogram with filter threshold line
    deminf_filter.yaml       — Hydra-compatible filter for training
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from egomimic.curation.curator import DemInfCurator
from egomimic.curation.utils import load_episodes_from_dir

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------


def _plot_score_distribution(
    scores: dict[str, float],
    threshold: float,
    output_path: Path,
) -> None:
    """Histogram of MI scores with vertical threshold line."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        values = list(scores.values())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(
            threshold,
            color="crimson",
            linewidth=2,
            linestyle="--",
            label=f"threshold = {threshold:.4f}",
        )
        ax.set_xlabel("MI Score (nats)")
        ax.set_ylabel("Number of Episodes")
        ax.set_title("DemInf: Episode MI Score Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Score distribution plot saved to %s", output_path)
    except ImportError:
        logger.warning("matplotlib not available — skipping score distribution plot")


# ---------------------------------------------------------------------------
# Episode loading (supports both curation and training data configs)
# ---------------------------------------------------------------------------


def _load_episodes(cfg: DictConfig) -> list:
    """
    Load episodes from either a local Zarr directory or a training data config.

    Tries load_episodes_from_config() first (EgoVerse pipeline integration).
    Falls back to load_episodes_from_dir() for environments without the
    EgoVerse pipeline installed.
    """
    data_cfg = cfg.data
    max_episodes = data_cfg.get("max_episodes") or None
    max_timesteps = data_cfg.get("max_timesteps_per_episode") or None

    # Try EgoVerse pipeline integration first
    try:
        from egomimic.curation.utils import load_episodes_from_config

        episodes = load_episodes_from_config(
            data_cfg=data_cfg,
            max_episodes=max_episodes,
            max_timesteps_per_episode=max_timesteps,
        )
        return episodes
    except (ImportError, ValueError):
        pass

    # Fallback: plain directory scan
    folder_path = Path(data_cfg.folder_path)
    if not folder_path.is_dir():
        logger.error("data.folder_path does not exist: %s", folder_path)
        sys.exit(1)

    valid_hashes: set[str] | None = None
    if data_cfg.get("valid_hashes_path") is not None:
        vp = Path(data_cfg.valid_hashes_path)
        with open(vp) as f:
            valid_hashes = set(json.load(f))
        logger.info("Loaded %d valid hashes from %s", len(valid_hashes), vp)

    custom_obs_keys = (
        list(data_cfg.custom_obs_keys) if data_cfg.get("custom_obs_keys") else None
    )
    custom_action_keys = (
        list(data_cfg.custom_action_keys)
        if data_cfg.get("custom_action_keys")
        else None
    )

    max_timesteps = data_cfg.get("max_timesteps_per_episode") or None
    return load_episodes_from_dir(
        folder_path=folder_path,
        valid_hashes=valid_hashes,
        custom_obs_keys=custom_obs_keys,
        custom_action_keys=custom_action_keys,
        max_episodes=max_episodes,
        max_timesteps_per_episode=max_timesteps,
    )


# ---------------------------------------------------------------------------
# Progress bar wrapper
# ---------------------------------------------------------------------------


def _tqdm_available() -> bool:
    try:
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


def _print_curation_summary(
    stats: dict,
    output_dir: Path,
    filter_yaml_path: Path | None,
) -> None:
    """Print the final curation summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("Curation complete.")
    print(sep)
    print(f"  Total episodes: {stats['total_input']:,}")
    print(
        f"  Kept: {stats['kept']:,} ({stats['kept'] / max(stats['total_input'], 1):.0%})"
    )
    print(
        f"  Removed: {stats['low_mi_removed'] + stats['pre_filter_removed']:,} ({(stats['low_mi_removed'] + stats['pre_filter_removed']) / max(stats['total_input'], 1):.0%})"
    )

    if stats.get("per_embodiment"):
        print("\nPer-embodiment breakdown:")
        for emb, emb_stats in stats["per_embodiment"].items():
            kept_n = emb_stats.get("kept", "?")
            total_n = emb_stats.get("total", emb_stats.get("count", "?"))
            pct = (
                f"{kept_n / total_n:.1%}"
                if isinstance(kept_n, int) and isinstance(total_n, int) and total_n > 0
                else "?"
            )
            print(f"  {emb}: {kept_n} kept / {total_n} total  ({pct})")

    print("\nMI score statistics (all scored episodes):")
    print(f"  mean:   {stats.get('mi_mean', float('nan')):.3f}")
    print(f"  median: {stats.get('mi_median', float('nan')):.3f}")
    print(f"  std:    {stats.get('mi_std', float('nan')):.3f}")

    print(f"\nOutput directory: {output_dir.resolve()}")

    if filter_yaml_path is not None:
        print(f"Exported filter:  {filter_yaml_path}")
        # Suggest the training command
        # Find a representative dataset name from the output yaml
        print("\nTo train on curated data (example):")
        lambda_hint = "lambda row: row.get('episode_hash') in KEPT_HASHES"
        print("  python trainHydra.py --config-name=train_zarr \\")
        print("      data=aria model=hpt_bc_flow_aria \\")
        print(
            f'      "+data.train_datasets.aria_bimanual.filters.filter_lambdas=[{json.dumps(lambda_hint)}]"'
        )
        print(f"  # (replace lambda with the one in {filter_yaml_path.name})")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[2] / "hydra_configs" / "curation"),
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dry_run: bool = bool(cfg.get("dry_run", False))

    logger.info("DemInf curation config:\n%s", OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # Load episodes
    # ------------------------------------------------------------------
    logger.info("Loading episodes …")

    if _tqdm_available():
        from tqdm import tqdm

        logger.info("(Progress bars enabled via tqdm)")

    episodes = _load_episodes(cfg)

    if not episodes:
        logger.error("No episodes loaded — check data config / folder_path")
        sys.exit(1)

    logger.info("Loaded %d episodes", len(episodes))

    # ------------------------------------------------------------------
    # Dry-run: just report episode counts
    # ------------------------------------------------------------------
    if dry_run:
        print("\n" + "=" * 60)
        print("DemInf Dry Run")
        print("=" * 60)
        print(f"  Episodes found: {len(episodes)}")

        # Embodiment breakdown
        emb_counts: dict[str, int] = {}
        for ep in episodes:
            emb_counts[ep.embodiment] = emb_counts.get(ep.embodiment, 0) + 1
        if emb_counts:
            print("  Per-embodiment:")
            for emb, count in sorted(emb_counts.items()):
                print(f"    {emb}: {count}")

        print(
            f"  Would keep (filter_ratio={cfg.get('filter_ratio', 0.3):.0%}): "
            f"~{int(len(episodes) * (1 - cfg.get('filter_ratio', 0.3)))}"
        )
        print("=" * 60 + "\n")
        return

    # ------------------------------------------------------------------
    # Curate
    # ------------------------------------------------------------------
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    curator = DemInfCurator.from_config(cfg_container)

    # Wrap the scoring step with tqdm if available
    if _tqdm_available():
        from tqdm import tqdm

        logger.info("Embedding and KSG scoring … (progress below)")
        with tqdm(total=1, desc="DemInf scoring", unit="run", leave=True) as pbar:
            result = curator.curate(episodes)
            pbar.update(1)
    else:
        result = curator.curate(episodes)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    curator.save_result(output_dir)

    # Score distribution plot
    threshold = result.stats.get("threshold_score", float("nan"))
    _plot_score_distribution(
        scores=result.scores,
        threshold=threshold,
        output_path=output_dir / "score_distribution.png",
    )

    # Export Hydra-compatible YAML filter
    filter_yaml_path: Path | None = None
    try:
        filter_yaml_path = curator.export_sql_filter(
            output_path=output_dir / "deminf_filter.yaml",
            format="yaml",
            filter_name="deminf_curated",
        )
    except Exception as exc:
        logger.warning("Failed to export YAML filter: %s", exc)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    _print_curation_summary(
        stats=result.stats,
        output_dir=output_dir,
        filter_yaml_path=filter_yaml_path,
    )


if __name__ == "__main__":
    main()
