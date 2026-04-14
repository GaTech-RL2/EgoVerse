"""
Trajectory-level mutual information scorer.

MI is estimated on the (s, a) distribution across all episodes simultaneously,
then per-timestep contributions are attributed back to source episodes.

``cross_embodiment_mode="independent"`` (default) groups episodes by embodiment
and runs KSG within each group, preventing different action modalities (e.g.
42-D hand keypoints vs. 14-D joint positions) from dominating NN distances.

``fit()`` auto-switches to ``fit_chunked()`` when total timesteps > 100 K,
subsampling a reference set for KSG and scoring all points via approximate
nearest-neighbour lookup against that reference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from egomimic.curation.ksg import ksg_mi_averaged

if TYPE_CHECKING:
    from egomimic.curation.embedders import ActionEmbedder, StateEmbedder
    from egomimic.curation.utils import Episode

logger = logging.getLogger(__name__)

# When total timesteps exceed this threshold, automatically use fit_chunked()
# instead of loading the full dataset into KSG at once.
_AUTO_CHUNKED_THRESHOLD = 100_000
_AUTO_CHUNKED_MAX_POINTS = 100_000


class TrajectoryScorer:
    """
    Score trajectories by their mean per-timestep MI contribution.

    Args:
        state_embedder: Fitted or unfitted StateEmbedder.
        action_embedder: Fitted or unfitted ActionEmbedder.
        k_range: (k_min, k_max) for KSG averaging.
        cross_embodiment_mode: "independent" (score per embodiment group) or
            "shared" (score all episodes jointly).
    """

    def __init__(
        self,
        state_embedder: "StateEmbedder",
        action_embedder: "ActionEmbedder",
        k_range: tuple[int, int] = (3, 7),
        cross_embodiment_mode: Literal["independent", "shared"] = "independent",
    ) -> None:
        self.state_embedder = state_embedder
        self.action_embedder = action_embedder
        self.k_range = k_range
        self.cross_embodiment_mode = cross_embodiment_mode

        self._scores: dict[str, float] = {}
        self._per_timestep_mi: np.ndarray | None = None
        self._episode_hashes: list[str] = []
        self._lengths: list[int] = []

    # ------------------------------------------------------------------
    # Standard fit (single-pass KSG, loads everything into memory)
    # ------------------------------------------------------------------

    def fit(self, episodes: list["Episode"]) -> None:
        """
        Full MI estimation pipeline.

        Steps:
            1. Fit embedders on all episodes (per-embodiment stats if independent).
            2. Embed all (s, a) pairs into a single global matrix.
            3. Run KSG — either globally (shared) or per-embodiment (independent).
            4. Attribute per-timestep MI contributions back to source episodes.
            5. Score each episode as the mean of its MI contributions.
        """
        if not episodes:
            raise ValueError("Cannot fit on empty episode list")

        N_total = sum(len(ep.actions) for ep in episodes)
        if N_total > _AUTO_CHUNKED_THRESHOLD:
            logger.info(
                "Large dataset (%d timesteps > %d threshold) — using fit_chunked() automatically",
                N_total,
                _AUTO_CHUNKED_THRESHOLD,
            )
            self.fit_chunked(episodes, max_points_in_memory=_AUTO_CHUNKED_MAX_POINTS)
            return

        logger.info("Fitting embedders on %d episodes …", len(episodes))
        self.state_embedder.fit(episodes)
        self.action_embedder.fit(episodes)

        if self.cross_embodiment_mode == "independent":
            self._fit_independent(episodes)
        else:
            self._fit_shared(episodes)

    def _fit_shared(self, episodes: list["Episode"]) -> None:
        """Run KSG on all episodes jointly."""
        logger.info("Embedding all (s, a) pairs …")
        s_emb, s_lengths = self.state_embedder.embed_episodes(episodes)
        a_emb, a_lengths = self.action_embedder.embed_episodes(episodes)

        assert s_lengths == a_lengths, "State/action embedding lengths disagree"
        self._lengths = s_lengths
        self._episode_hashes = [ep.episode_hash for ep in episodes]

        N_total = s_emb.shape[0]
        logger.info(
            "Running KSG on %d total (s, a) pairs (k_range=%s) …",
            N_total,
            self.k_range,
        )
        mi_contributions = ksg_mi_averaged(s_emb, a_emb, k_range=self.k_range)
        self._per_timestep_mi = mi_contributions
        self._scores = _aggregate_scores(
            mi_contributions, self._episode_hashes, self._lengths
        )
        _log_score_stats(self._scores)

    def _fit_independent(self, episodes: list["Episode"]) -> None:
        """
        Run KSG separately for each embodiment group, then merge scores.

        Episodes from different embodiments are never compared against each
        other in the KSG nearest-neighbour search, which avoids confounding
        from heterogeneous action modalities.
        """
        # Group episodes by embodiment preserving global index order
        by_embodiment: dict[str, list[tuple[int, "Episode"]]] = {}
        for idx, ep in enumerate(episodes):
            by_embodiment.setdefault(ep.embodiment, []).append((idx, ep))

        # Run KSG per group and collect per-timestep MI arrays (in global order)
        total_timesteps = sum(len(ep.actions) for ep in episodes)
        global_mi = np.full(total_timesteps, float("nan"), dtype=np.float64)
        self._episode_hashes = [ep.episode_hash for ep in episodes]
        self._lengths = [len(ep.actions) for ep in episodes]

        # Build a global timestep offset map
        global_offsets = np.zeros(len(episodes), dtype=int)
        offset = 0
        for i, ep in enumerate(episodes):
            global_offsets[i] = offset
            offset += len(ep.actions)

        for embodiment, indexed_eps in by_embodiment.items():
            eps_group = [ep for _, ep in indexed_eps]
            idxs_group = [idx for idx, _ in indexed_eps]

            logger.info(
                "Embedding + KSG for embodiment '%s' (%d episodes) …",
                embodiment,
                len(eps_group),
            )

            if len(eps_group) < self.k_range[0] + 1:
                # Not enough episodes for KSG — assign NaN scores
                logger.warning(
                    "Embodiment '%s' has too few episodes (%d) for k_range=%s — assigning NaN",
                    embodiment,
                    len(eps_group),
                    self.k_range,
                )
                for idx in idxs_group:
                    n = self._lengths[idx]
                    global_mi[global_offsets[idx] : global_offsets[idx] + n] = float(
                        "nan"
                    )
                continue

            s_emb, s_lengths = self.state_embedder.embed_episodes(eps_group)
            a_emb, a_lengths = self.action_embedder.embed_episodes(eps_group)
            assert (
                s_lengths == a_lengths
            ), f"Length mismatch in embodiment '{embodiment}'"

            # Check we have enough points for k
            N_group = s_emb.shape[0]
            k_max_safe = min(self.k_range[1], N_group - 2)
            k_min_safe = min(self.k_range[0], k_max_safe)
            if k_min_safe < 1:
                logger.warning(
                    "Embodiment '%s': too few timesteps (%d) for KSG — assigning 0.0",
                    embodiment,
                    N_group,
                )
                for idx in idxs_group:
                    n = self._lengths[idx]
                    global_mi[global_offsets[idx] : global_offsets[idx] + n] = 0.0
                continue

            group_mi = ksg_mi_averaged(s_emb, a_emb, k_range=(k_min_safe, k_max_safe))

            # Write back into global array
            local_offset = 0
            for episode_global_idx, length in zip(idxs_group, s_lengths):
                g_start = global_offsets[episode_global_idx]
                global_mi[g_start : g_start + length] = group_mi[
                    local_offset : local_offset + length
                ]
                local_offset += length

        self._per_timestep_mi = global_mi
        self._scores = _aggregate_scores(global_mi, self._episode_hashes, self._lengths)
        _log_score_stats(self._scores)

    # ------------------------------------------------------------------
    # Chunked fit — for large datasets (> 50 K timesteps)
    # ------------------------------------------------------------------

    def fit_chunked(
        self,
        episodes: list["Episode"],
        max_points_in_memory: int = 200_000,
        subsample_strategy: Literal["random", "uniform"] = "random",
    ) -> None:
        """
        MI estimation for large datasets via a subsampled reference set.

        Subsamples ``max_points_in_memory`` timesteps as a KSG reference set,
        builds KD-trees on that set, and scores all timesteps via approximate
        nearest-neighbour lookup:
            ε_i = distance to k-th NN in joint reference space,
            n_s/n_a = count of reference marginals within ε_i,
            mi_i ≈ ψ(k) − ψ(n_s+1) − ψ(n_a+1) + ψ(N_ref).

        Args:
            episodes: Raw (pre-filtered) episodes to score.
            max_points_in_memory: Reference set size cap.
            subsample_strategy: "random" (uniform random) or "uniform"
                (evenly spaced along the concatenated timestep sequence).
        """
        if not episodes:
            raise ValueError("Cannot fit on empty episode list")

        logger.info(
            "fit_chunked: %d episodes, max_points=%d, strategy=%s",
            len(episodes),
            max_points_in_memory,
            subsample_strategy,
        )

        # Step 1: Fit embedders globally
        self.state_embedder.fit(episodes)
        self.action_embedder.fit(episodes)

        # Step 2: Embed all episodes (streaming)
        self._episode_hashes = [ep.episode_hash for ep in episodes]
        self._lengths = [len(ep.actions) for ep in episodes]
        N_total = sum(self._lengths)

        # Probe actual output dims — when obs_dim <= latent_dim the embedder
        # returns the data as-is rather than projecting up to latent_dim.
        _probe = episodes[0]
        s_latent_dim = self.state_embedder.embed(
            _probe.observations[:1], embodiment=_probe.embodiment
        ).shape[1]
        a_latent_dim = self.action_embedder.embed(
            _probe.actions[:1], embodiment=_probe.embodiment
        ).shape[1]

        s_all = np.empty((N_total, s_latent_dim), dtype=np.float32)
        a_all = np.empty((N_total, a_latent_dim), dtype=np.float32)

        offset = 0
        for ep in episodes:
            n = len(ep.actions)
            s_all[offset : offset + n] = self.state_embedder.embed(
                ep.observations, embodiment=ep.embodiment
            )
            a_all[offset : offset + n] = self.action_embedder.embed(
                ep.actions, embodiment=ep.embodiment
            )
            offset += n

        # Step 3: Subsample reference set
        if N_total > max_points_in_memory:
            if subsample_strategy == "random":
                rng = np.random.default_rng(seed=42)
                ref_idx = np.sort(
                    rng.choice(N_total, size=max_points_in_memory, replace=False)
                )
            else:
                ref_idx = np.linspace(0, N_total - 1, max_points_in_memory, dtype=int)
            s_ref = s_all[ref_idx].astype(np.float64)
            a_ref = a_all[ref_idx].astype(np.float64)
            logger.info(
                "Subsampled %d / %d timesteps for KSG reference set",
                max_points_in_memory,
                N_total,
            )
        else:
            # All points fit — reference set = full set
            ref_idx = np.arange(N_total)
            s_ref = s_all.astype(np.float64)
            a_ref = a_all.astype(np.float64)

        N_ref = len(s_ref)

        # Step 4: Build KD-trees on the reference set
        logger.info("Building KD-trees on %d reference points …", N_ref)
        z_ref = np.hstack([s_ref, a_ref])
        tree_z = cKDTree(z_ref)
        tree_s = cKDTree(s_ref)
        tree_a = cKDTree(a_ref)

        # Step 5: Score ALL timesteps against the reference trees
        k_min, k_max = self.k_range
        # Use the middle k for the approximate scoring (averaging over k would
        # require multiple rounds of NN queries — too slow for the chunked path)
        k_approx = (k_min + k_max) // 2

        logger.info(
            "Scoring %d timesteps via approximate KSG (k=%d, N_ref=%d) …",
            N_total,
            k_approx,
            N_ref,
        )

        s_query = s_all.astype(np.float64)
        a_query = a_all.astype(np.float64)
        z_query = np.hstack([s_query, a_query])

        # k-NN distances in joint reference space.
        # For query points IN the reference, self is included at distance 0 →
        # request k+1 and take index k.  For points NOT in reference, self is
        # absent → index k-1 would be the k-th NN.  We always request k+1 and
        # use index k, which slightly over-estimates ε for out-of-reference points
        # (conservative → slightly lower MI estimates, which is acceptable).
        dists, _ = tree_z.query(z_query, k=k_approx + 1, p=np.inf, workers=-1)
        eps = dists[:, k_approx]  # (N_total,)
        eps_strict = eps * (1.0 - 1e-10)

        # Batch marginal counting against reference trees
        batch_sz = 10_000
        n_s = np.empty(N_total, dtype=np.float64)
        n_a = np.empty(N_total, dtype=np.float64)
        for start in range(0, N_total, batch_sz):
            end = min(start + batch_sz, N_total)
            cs = tree_s.query_ball_point(
                s_query[start:end], eps_strict[start:end], p=np.inf, return_length=True
            )
            ca = tree_a.query_ball_point(
                a_query[start:end], eps_strict[start:end], p=np.inf, return_length=True
            )
            # Subtract 1 for self if the query point happens to be in the reference.
            # Using max(0, c-1) as a conservative lower bound avoids negative counts.
            n_s[start:end] = np.maximum(np.asarray(cs, dtype=np.float64) - 1.0, 0.0)
            n_a[start:end] = np.maximum(np.asarray(ca, dtype=np.float64) - 1.0, 0.0)

        mi_contributions = (
            digamma(k_approx) - digamma(n_s + 1) - digamma(n_a + 1) + digamma(N_ref)
        )
        self._per_timestep_mi = mi_contributions
        self._scores = _aggregate_scores(
            mi_contributions, self._episode_hashes, self._lengths
        )
        _log_score_stats(self._scores)

    # ------------------------------------------------------------------
    # Score retrieval
    # ------------------------------------------------------------------

    def get_scores(self) -> dict[str, float]:
        """Return {episode_hash: mi_score} for all scored episodes."""
        if not self._scores:
            raise RuntimeError("Call fit() first")
        return dict(self._scores)

    def get_ranked_episodes(self) -> list[tuple[str, float]]:
        """Return [(episode_hash, score), …] sorted by score descending."""
        return sorted(self._scores.items(), key=lambda kv: kv[1], reverse=True)

    def get_per_timestep_mi(self) -> np.ndarray | None:
        """Return the raw per-timestep MI array (N_total,), or None if not fitted."""
        return self._per_timestep_mi

    def get_episode_lengths(self) -> list[int]:
        return list(self._lengths)

    def get_episode_hashes(self) -> list[str]:
        return list(self._episode_hashes)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_scores(
    per_timestep_mi: np.ndarray,
    episode_hashes: list[str],
    lengths: list[int],
) -> dict[str, float]:
    """Group per-timestep MI contributions back to episodes by mean."""
    scores: dict[str, float] = {}
    start = 0
    for ep_hash, length in zip(episode_hashes, lengths):
        ep_mi = per_timestep_mi[start : start + length]
        # Use nanmean so NaN timesteps (e.g. from empty embodiment groups) degrade
        # gracefully rather than poisoning the whole episode score.
        score = float(np.nanmean(ep_mi)) if length > 0 else float("nan")
        scores[ep_hash] = score
        start += length
    return scores


def _log_score_stats(scores: dict[str, float]) -> None:
    vals = np.array([v for v in scores.values() if np.isfinite(v)])
    if len(vals) == 0:
        logger.warning("No finite MI scores computed")
        return
    logger.info(
        "Scoring complete. MI mean=%.4f std=%.4f min=%.4f max=%.4f",
        vals.mean(),
        vals.std(),
        vals.min(),
        vals.max(),
    )
