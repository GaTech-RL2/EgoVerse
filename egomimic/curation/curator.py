"""
DemInf top-level curation orchestrator.

Ties together loading → filtering → embedding → MI scoring → selection.

Integration with training:
    curator.export_sql_filter() writes a Hydra-compatible YAML that can be
    passed directly to a training run via the data.filters config key:

        python trainHydra.py --config-name=train_zarr \\
            data=aria model=hpt_bc_flow_aria \\
            +data.train_datasets.aria_bimanual.filters.filter_lambdas=[<lambda>]
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from egomimic.curation.embedders import ActionEmbedder, StateEmbedder
from egomimic.curation.filters import (
    ActionClipFilter,
    MinLengthFilter,
    PauseFilter,
    apply_filters,
)
from egomimic.curation.scorer import TrajectoryScorer
from egomimic.curation.utils import Episode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CurationResult:
    """Output of a single curation run."""

    kept_hashes: list[str]
    removed_hashes: list[str]  # filtered out by preprocessing
    low_mi_hashes: list[str]  # discarded for low MI score
    scores: dict[str, float]  # {episode_hash: mi_score} for scored episodes
    stats: dict = field(default_factory=dict)

    @property
    def all_removed_hashes(self) -> list[str]:
        return self.removed_hashes + self.low_mi_hashes


# ---------------------------------------------------------------------------
# DemInfCurator
# ---------------------------------------------------------------------------


class DemInfCurator:
    """
    End-to-end DemInf curation pipeline.

    Args:
        filter_ratio: Fraction of episodes to remove based on MI (e.g. 0.3
            removes the bottom 30% of scored episodes).
        state_mode: "proprioceptive" or "image".
        latent_dim: Embedding latent dimension.
        k_range: (k_min, k_max) for KSG averaging.
        pause_epsilon: Action delta threshold for pause detection.
        action_clip: Whether to apply ActionClipFilter.
        min_trajectory_length: Minimum timesteps after filtering.
        embodiment_overrides: Per-embodiment pause_epsilon overrides.
        cross_embodiment_mode: "independent" (score per embodiment) or "shared"
            (score all embodiments jointly).
        device: Torch device string (for image embedder).
    """

    def __init__(
        self,
        filter_ratio: float = 0.3,
        state_mode: str = "proprioceptive",
        latent_dim: int = 32,
        k_range: tuple[int, int] = (3, 7),
        pause_epsilon: float = 1e-2,
        action_clip: bool = True,
        min_trajectory_length: int = 20,
        embodiment_overrides: dict[str, float] | None = None,
        cross_embodiment_mode: Literal["independent", "shared"] = "independent",
        device: str = "cpu",
    ) -> None:
        self.filter_ratio = filter_ratio
        self.state_mode = state_mode
        self.latent_dim = latent_dim
        self.k_range = k_range
        self.pause_epsilon = pause_epsilon
        self.action_clip = action_clip
        self.min_trajectory_length = min_trajectory_length
        self.embodiment_overrides = embodiment_overrides or {}
        self.cross_embodiment_mode = cross_embodiment_mode
        self.device = device

        self._result: CurationResult | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_filters(self) -> list:
        filters = [
            PauseFilter(
                epsilon=self.pause_epsilon,
                embodiment_overrides=self.embodiment_overrides,
            ),
        ]
        if self.action_clip:
            filters.append(ActionClipFilter())
        filters.append(MinLengthFilter(min_length=self.min_trajectory_length))
        return filters

    def _build_scorer(self) -> TrajectoryScorer:
        state_embedder = StateEmbedder(
            mode=self.state_mode,
            latent_dim=self.latent_dim,
            device=self.device,
            cross_embodiment_mode=self.cross_embodiment_mode,
        )
        action_embedder = ActionEmbedder(
            latent_dim=self.latent_dim,
            device=self.device,
            cross_embodiment_mode=self.cross_embodiment_mode,
        )
        return TrajectoryScorer(
            state_embedder,
            action_embedder,
            k_range=self.k_range,
            cross_embodiment_mode=self.cross_embodiment_mode,
        )

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def curate(self, episodes: list[Episode]) -> CurationResult:
        """
        Full curation pipeline.

        Steps:
            1. Apply preprocessing filters (pauses, clipping, min length).
            2. Fit embedders and compute global MI scores.
            3. Remove the bottom ``filter_ratio`` fraction of episodes.

        Args:
            episodes: Raw episodes to curate.

        Returns:
            CurationResult with kept/removed hashes, scores, and stats.
        """
        logger.info("Starting DemInf curation on %d episodes", len(episodes))

        # Step 1: preprocessing filters
        filters = self._build_filters()
        filtered_episodes, pre_removed = apply_filters(episodes, filters)

        logger.info(
            "After preprocessing: %d / %d episodes retained",
            len(filtered_episodes),
            len(episodes),
        )

        if not filtered_episodes:
            logger.warning("All episodes removed by preprocessing filters!")
            return CurationResult(
                kept_hashes=[],
                removed_hashes=pre_removed,
                low_mi_hashes=[],
                scores={},
                stats={"error": "all_episodes_filtered"},
            )

        # Step 2: embed and score
        scorer = self._build_scorer()
        scorer.fit(filtered_episodes)
        scores = scorer.get_scores()

        # Step 3: remove bottom filter_ratio fraction
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        n_total = len(ranked)
        n_remove = int(np.floor(n_total * self.filter_ratio))
        n_keep = n_total - n_remove

        kept_hashes = [h for h, _ in ranked[:n_keep]]
        low_mi_hashes = [h for h, _ in ranked[n_keep:]]

        threshold_score = ranked[n_keep - 1][1] if n_keep > 0 else float("nan")

        # Build stats
        all_scores = np.array([s for _, s in ranked if np.isfinite(s)])
        embodiment_scores: dict[str, list[float]] = {}
        kept_set = set(kept_hashes)
        ep_by_hash = {ep.episode_hash: ep for ep in filtered_episodes}
        for h, s in scores.items():
            emb = ep_by_hash[h].embodiment if h in ep_by_hash else "unknown"
            embodiment_scores.setdefault(emb, []).append(s)

        embodiment_kept: dict[str, int] = {}
        embodiment_total: dict[str, int] = {}
        for h in scores:
            emb = ep_by_hash[h].embodiment if h in ep_by_hash else "unknown"
            embodiment_total[emb] = embodiment_total.get(emb, 0) + 1
            if h in kept_set:
                embodiment_kept[emb] = embodiment_kept.get(emb, 0) + 1

        per_embodiment_stats = {
            emb: {
                "count": len(vals),
                "kept": embodiment_kept.get(emb, 0),
                "total": embodiment_total.get(emb, 0),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "median": float(np.nanmedian(vals)),
            }
            for emb, vals in embodiment_scores.items()
        }

        stats = {
            "total_input": len(episodes),
            "pre_filter_removed": len(pre_removed),
            "scored": n_total,
            "kept": len(kept_hashes),
            "low_mi_removed": len(low_mi_hashes),
            "filter_ratio": self.filter_ratio,
            "threshold_score": float(threshold_score),
            "mi_mean": float(all_scores.mean()) if len(all_scores) else float("nan"),
            "mi_std": float(all_scores.std()) if len(all_scores) else float("nan"),
            "mi_median": float(np.median(all_scores))
            if len(all_scores)
            else float("nan"),
            "mi_min": float(all_scores.min()) if len(all_scores) else float("nan"),
            "mi_max": float(all_scores.max()) if len(all_scores) else float("nan"),
            "per_embodiment": per_embodiment_stats,
            "cross_embodiment_mode": self.cross_embodiment_mode,
        }

        self._result = CurationResult(
            kept_hashes=kept_hashes,
            removed_hashes=pre_removed,
            low_mi_hashes=low_mi_hashes,
            scores=scores,
            stats=stats,
        )

        logger.info(
            "Curation complete: kept=%d, removed_preprocessing=%d, removed_low_mi=%d",
            len(kept_hashes),
            len(pre_removed),
            len(low_mi_hashes),
        )

        return self._result

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def get_filtered_hashes(self) -> list[str]:
        """Return episode hashes that passed curation."""
        if self._result is None:
            raise RuntimeError("Call curate() first")
        return list(self._result.kept_hashes)

    def save_scores(self, path: str | Path) -> None:
        """Save {episode_hash: mi_score} as JSON."""
        if self._result is None:
            raise RuntimeError("Call curate() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._result.scores, f, indent=2)
        logger.info("Scores saved to %s", path)

    def save_result(self, output_dir: str | Path) -> None:
        """Save scores.json, kept_hashes.json, curation_stats.json to output_dir."""
        if self._result is None:
            raise RuntimeError("Call curate() first")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "scores.json", "w") as f:
            json.dump(self._result.scores, f, indent=2)

        with open(output_dir / "kept_hashes.json", "w") as f:
            json.dump(self._result.kept_hashes, f, indent=2)

        with open(output_dir / "curation_stats.json", "w") as f:
            json.dump(self._result.stats, f, indent=2)

        logger.info("Curation results saved to %s", output_dir)

    def export_sql_filter(
        self,
        output_path: str | Path,
        format: Literal["yaml", "json", "sql"] = "yaml",
        filter_name: str | None = None,
    ) -> Path:
        """
        Export kept episode hashes as a reusable filter artifact.

        Formats
        -------
        yaml (default):
            DatasetFilter-compatible YAML with a ``filter_lambdas`` entry
            containing a set-membership lambda, plus a ``kept_hashes`` list
            and a ``metadata`` block for reproducibility.

        json:
            ``{"kept_hashes": [...], "metadata": {...}}``

        sql:
            Raw ``WHERE episode_hash IN (...)`` clause.

        Returns:
            Resolved Path of the written file.
        """
        if self._result is None:
            raise RuntimeError("Call curate() first")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        name = filter_name or output_path.stem

        meta = self._build_metadata(name)
        kept = self._result.kept_hashes

        if format == "yaml":
            self._export_yaml(output_path, kept, meta)
        elif format == "json":
            self._export_json(output_path, kept, meta)
        elif format == "sql":
            self._export_sql(output_path, kept, meta)
        else:
            raise ValueError(
                f"Unknown format: {format!r}. Choose 'yaml', 'json', or 'sql'."
            )

        logger.info(
            "Exported %s filter (%d hashes) to %s", format, len(kept), output_path
        )
        return output_path.resolve()

    def _build_metadata(self, name: str) -> dict:
        stats = self._result.stats or {}
        return {
            "name": name,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "git_hash": _get_git_hash(),
            "config": {
                "filter_ratio": self.filter_ratio,
                "state_mode": self.state_mode,
                "latent_dim": self.latent_dim,
                "k_range": list(self.k_range),
                "pause_epsilon": self.pause_epsilon,
                "action_clip": self.action_clip,
                "min_trajectory_length": self.min_trajectory_length,
                "cross_embodiment_mode": self.cross_embodiment_mode,
            },
            "total_input": stats.get("total_input", 0),
            "kept": stats.get("kept", len(self._result.kept_hashes)),
            "scored": stats.get("scored", 0),
            "filter_ratio_applied": self.filter_ratio,
            "mi_mean": stats.get("mi_mean"),
            "mi_median": stats.get("mi_median"),
            "mi_std": stats.get("mi_std"),
        }

    def _export_yaml(self, path: Path, kept: list[str], meta: dict) -> None:
        kept_set_repr = repr(set(kept))
        lambda_str = f"lambda row: row.get('episode_hash') in {kept_set_repr}"

        mi_line = (
            f"# MI mean/median/std: {meta['mi_mean']:.4f} / {meta['mi_median']:.4f} / {meta['mi_std']:.4f}"
            if all(
                v is not None
                for v in [meta["mi_mean"], meta["mi_median"], meta["mi_std"]]
            )
            else "# MI stats: unavailable"
        )
        lines = [
            "# DemInf curation filter",
            f"# Name:      {meta['name']}",
            f"# Generated: {meta['timestamp']}",
            f"# Git hash:  {meta['git_hash']}",
            f"# Episodes:  {meta['kept']} kept / {meta['total_input']} total",
            f"# Filter ratio: {meta['filter_ratio_applied']:.0%}",
            mi_line,
            "#",
            "# Usage — add to a training data config's filters section:",
            "#   +data.train_datasets.<name>.filters.filter_lambdas=[<lambda>]",
            "",
            "_target_: egomimic.rldb.filters.DatasetFilter",
            "filter_lambdas:",
            f"  - {json.dumps(lambda_str)}",
            "",
            "kept_hashes:",
        ]
        for h in kept:
            lines.append(f"  - {json.dumps(h)}")
        lines.append("")
        lines.append("metadata:")
        for k, v in meta.items():
            if isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {json.dumps(vv)}")
            else:
                lines.append(f"  {k}: {json.dumps(v)}")
        lines.append("")

        path.write_text("\n".join(lines))

    def _export_json(self, path: Path, kept: list[str], meta: dict) -> None:
        with open(path, "w") as f:
            json.dump({"kept_hashes": kept, "metadata": meta}, f, indent=2)

    def _export_sql(self, path: Path, kept: list[str], meta: dict) -> None:
        hashes_str = ", ".join(f"'{h}'" for h in kept)
        path.write_text(
            f"-- DemInf filter '{meta['name']}' | {meta['timestamp']}\n"
            f"-- {meta['kept']} / {meta['total_input']} episodes kept\n"
            f"WHERE episode_hash IN ({hashes_str})\n"
        )

    @classmethod
    def from_config(cls, cfg) -> "DemInfCurator":
        """
        Construct from a Hydra DictConfig object.

        Expected config structure (see hydra_configs/curation/default.yaml):
            state_mode: proprioceptive
            latent_dim: 32
            k_range: [3, 7]
            filter_ratio: 0.3
            cross_embodiment_mode: independent
            filters:
              pause_epsilon: 0.01
              action_clip: true
              min_trajectory_length: 20
            embodiment_overrides:
              aria_bimanual: {pause_epsilon: 0.005}
        """
        embodiment_overrides: dict[str, float] = {}
        for emb, overrides in (cfg.get("embodiment_overrides") or {}).items():
            if isinstance(overrides, dict) and "pause_epsilon" in overrides:
                embodiment_overrides[emb] = float(overrides["pause_epsilon"])

        filters_cfg = cfg.get("filters", {})
        k_range_raw = cfg.get("k_range", [3, 7])
        k_range = (int(k_range_raw[0]), int(k_range_raw[1]))

        return cls(
            filter_ratio=float(cfg.get("filter_ratio", 0.3)),
            state_mode=str(cfg.get("state_mode", "proprioceptive")),
            latent_dim=int(cfg.get("latent_dim", 32)),
            k_range=k_range,
            pause_epsilon=float(filters_cfg.get("pause_epsilon", 1e-2)),
            action_clip=bool(filters_cfg.get("action_clip", True)),
            min_trajectory_length=int(filters_cfg.get("min_trajectory_length", 20)),
            embodiment_overrides=embodiment_overrides,
            cross_embodiment_mode=str(cfg.get("cross_embodiment_mode", "independent")),
            device=str(cfg.get("device", "cpu")),
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
