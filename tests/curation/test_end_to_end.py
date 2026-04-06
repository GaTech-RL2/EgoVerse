"""
End-to-end tests for the DemInf curation pipeline.

Setup:
    50 "good" episodes: actions = f(state) + small noise (purposeful, correlated)
    50 "bad"  episodes: actions = random noise (uncorrelated with state)

    Run full curation with filter_ratio=0.5.

Assertion:
    >80% of removed episodes are the "bad" ones.

This validates the core DemInf claim: high MI ↔ purposeful (state-predictable)
behaviour, low MI ↔ random noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.curation.curator import CurationResult, DemInfCurator
from egomimic.curation.embedders import ActionEmbedder, StateEmbedder
from egomimic.curation.scorer import TrajectoryScorer
from egomimic.curation.utils import Episode

# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------


def _make_good_episode(
    episode_hash: str,
    T: int = 200,
    state_dim: int = 6,
    action_dim: int = 6,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> Episode:
    """
    'Good' episode: action = linear_transform(state) + small noise.
    High state-action MI because actions are predictable from states.
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T, state_dim)).astype(np.float32)

    # Random linear map: state → action
    W = rng.standard_normal((state_dim, action_dim)).astype(np.float32) * 0.5
    actions = (
        obs @ W + rng.standard_normal((T, action_dim)).astype(np.float32) * noise_scale
    )
    return Episode(
        episode_hash=episode_hash,
        observations=obs,
        actions=actions.astype(np.float32),
        embodiment="eva_bimanual",
        metadata={"total_frames": T},
    )


def _make_bad_episode(
    episode_hash: str,
    T: int = 200,
    state_dim: int = 6,
    action_dim: int = 6,
    seed: int = 0,
) -> Episode:
    """
    'Bad' episode: actions are pure random noise, independent of state.
    Low state-action MI.
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T, state_dim)).astype(np.float32)
    actions = rng.standard_normal((T, action_dim)).astype(np.float32)
    return Episode(
        episode_hash=episode_hash,
        observations=obs,
        actions=actions.astype(np.float32),
        embodiment="eva_bimanual",
        metadata={"total_frames": T},
    )


def _build_dataset(
    n_good: int = 20,
    n_bad: int = 20,
    T: int = 100,
) -> tuple[list[Episode], set[str], set[str]]:
    """Return (episodes, good_hashes, bad_hashes)."""
    episodes = []
    good_hashes: set[str] = set()
    bad_hashes: set[str] = set()

    for i in range(n_good):
        h = f"good_{i:04d}"
        episodes.append(_make_good_episode(h, T=T, seed=i))
        good_hashes.add(h)

    for i in range(n_bad):
        h = f"bad_{i:04d}"
        episodes.append(_make_bad_episode(h, T=T, seed=i + 1000))
        bad_hashes.add(h)

    # Shuffle so order doesn't matter
    rng = np.random.default_rng(42)
    order = rng.permutation(len(episodes))
    episodes = [episodes[i] for i in order]

    return episodes, good_hashes, bad_hashes


# ---------------------------------------------------------------------------
# Curator config
# ---------------------------------------------------------------------------


_CURATOR_KWARGS = dict(
    filter_ratio=0.5,
    state_mode="proprioceptive",
    latent_dim=8,  # small latent keeps KSG fast in tests
    k_range=(3, 4),  # narrow k_range to reduce KSG calls
    pause_epsilon=1e-3,
    action_clip=False,  # disable clip so good/bad signal is preserved
    min_trajectory_length=10,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEndCuration:
    @pytest.fixture(scope="class")
    def curation_result(self) -> tuple[CurationResult, set[str], set[str]]:
        episodes, good_hashes, bad_hashes = _build_dataset(n_good=20, n_bad=20, T=100)
        curator = DemInfCurator(**_CURATOR_KWARGS)
        result = curator.curate(episodes)
        return result, good_hashes, bad_hashes

    def test_correct_number_kept(self, curation_result):
        result, _, _ = curation_result
        # filter_ratio=0.5 → half removed from MI step (pre-filter removes 0)
        assert (
            result.stats["kept"] == 20
        ), f"Expected 20 kept, got {result.stats['kept']}"

    def test_majority_removed_are_bad(self, curation_result):
        """
        >80% of episodes removed by MI scoring should be 'bad' episodes.
        """
        result, good_hashes, bad_hashes = curation_result
        removed_mi = set(result.low_mi_hashes)

        bad_in_removed = len(removed_mi & bad_hashes)
        total_removed = len(removed_mi)

        if total_removed == 0:
            pytest.skip("No episodes removed — check filter_ratio")

        fraction_bad = bad_in_removed / total_removed
        assert fraction_bad > 0.80, (
            f"Expected >80% of removed episodes to be 'bad', "
            f"got {fraction_bad:.1%} ({bad_in_removed}/{total_removed})"
        )

    def test_majority_kept_are_good(self, curation_result):
        """
        >80% of kept episodes should be 'good' episodes.
        """
        result, good_hashes, bad_hashes = curation_result
        kept_set = set(result.kept_hashes)

        good_in_kept = len(kept_set & good_hashes)
        total_kept = len(kept_set)

        fraction_good = good_in_kept / total_kept
        assert fraction_good > 0.80, (
            f"Expected >80% of kept episodes to be 'good', "
            f"got {fraction_good:.1%} ({good_in_kept}/{total_kept})"
        )

    def test_scores_dict_non_empty(self, curation_result):
        result, _, _ = curation_result
        assert len(result.scores) == 40  # all 40 episodes scored

    def test_good_episodes_higher_mean_mi(self, curation_result):
        """Mean MI of good episodes should exceed mean MI of bad episodes."""
        result, good_hashes, bad_hashes = curation_result
        good_mi = np.mean([result.scores[h] for h in good_hashes if h in result.scores])
        bad_mi = np.mean([result.scores[h] for h in bad_hashes if h in result.scores])
        assert (
            good_mi > bad_mi
        ), f"Expected good MI ({good_mi:.4f}) > bad MI ({bad_mi:.4f})"

    def test_get_filtered_hashes_matches_kept(self, curation_result):
        result, _, _ = curation_result
        curator = DemInfCurator(**_CURATOR_KWARGS)
        # Reconstruct with the pre-computed result
        curator._result = result
        assert set(curator.get_filtered_hashes()) == set(result.kept_hashes)

    def test_stats_keys_present(self, curation_result):
        result, _, _ = curation_result
        required_keys = {
            "total_input",
            "pre_filter_removed",
            "scored",
            "kept",
            "low_mi_removed",
            "filter_ratio",
            "threshold_score",
            "mi_mean",
            "mi_std",
            "mi_median",
            "mi_min",
            "mi_max",
            "per_embodiment",
        }
        assert required_keys.issubset(result.stats.keys())


class TestScorerIsolated:
    """Isolated TrajectoryScorer tests without the full curator."""

    def test_scorer_fit_and_rank(self):
        episodes, good_hashes, bad_hashes = _build_dataset(n_good=10, n_bad=10, T=50)
        state_emb = StateEmbedder(mode="proprioceptive", latent_dim=8)
        action_emb = ActionEmbedder(latent_dim=8)
        scorer = TrajectoryScorer(state_emb, action_emb, k_range=(3, 4))
        scorer.fit(episodes)

        scores = scorer.get_scores()
        assert len(scores) == 20

        ranked = scorer.get_ranked_episodes()
        assert ranked[0][1] >= ranked[-1][1], "Ranked list should be descending"

    def test_scorer_per_timestep_mi_shape(self):
        episodes, _, _ = _build_dataset(n_good=5, n_bad=5, T=40)
        state_emb = StateEmbedder(mode="proprioceptive", latent_dim=8)
        action_emb = ActionEmbedder(latent_dim=8)
        scorer = TrajectoryScorer(state_emb, action_emb, k_range=(3, 3))
        scorer.fit(episodes)

        per_ts = scorer.get_per_timestep_mi()
        expected_total = sum(len(ep.actions) for ep in episodes)
        assert per_ts is not None
        assert per_ts.shape == (expected_total,)

    def test_scorer_raises_without_fit(self):
        state_emb = StateEmbedder(mode="proprioceptive", latent_dim=8)
        action_emb = ActionEmbedder(latent_dim=8)
        scorer = TrajectoryScorer(state_emb, action_emb)
        with pytest.raises(RuntimeError, match="fit"):
            scorer.get_scores()


class TestCurationResultIO:
    """Test save/load round-trip for CurationResult."""

    def test_save_result(self, tmp_path):
        episodes, good_hashes, bad_hashes = _build_dataset(n_good=5, n_bad=5, T=40)
        curator = DemInfCurator(**_CURATOR_KWARGS)
        result = curator.curate(episodes)
        curator.save_result(tmp_path)

        import json

        assert (tmp_path / "scores.json").exists()
        assert (tmp_path / "kept_hashes.json").exists()
        assert (tmp_path / "curation_stats.json").exists()

        loaded_scores = json.loads((tmp_path / "scores.json").read_text())
        assert len(loaded_scores) == len(result.scores)

        loaded_kept = json.loads((tmp_path / "kept_hashes.json").read_text())
        assert set(loaded_kept) == set(result.kept_hashes)
