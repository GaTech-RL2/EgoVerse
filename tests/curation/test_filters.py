"""
Tests for DemInf preprocessing filters.

Key scenarios:
- PauseFilter: synthetic trajectory with 50-frame pauses → pauses collapsed
- ActionClipFilter: trajectory with extreme outlier actions → clipped
- MinLengthFilter: short episode → discarded
- Filter pipeline: correct ordering and composition
"""

from __future__ import annotations

import numpy as np

from egomimic.curation.filters import (
    ActionClipFilter,
    EmbodimentAwareFilter,
    MinLengthFilter,
    PauseFilter,
    apply_filters,
)
from egomimic.curation.utils import Episode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    T: int = 100,
    obs_dim: int = 6,
    action_dim: int = 6,
    embodiment: str = "eva_bimanual",
    seed: int = 0,
    episode_hash: str = "test_episode",
) -> Episode:
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((T, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((T, action_dim)).astype(np.float32)
    return Episode(
        episode_hash=episode_hash,
        observations=obs,
        actions=actions,
        embodiment=embodiment,
        metadata={"total_frames": T},
    )


def _make_episode_with_pauses(
    T_active: int = 50,
    T_pause: int = 50,
    action_dim: int = 4,
    epsilon: float = 1e-3,
    seed: int = 0,
) -> Episode:
    """
    Build an episode where the first T_active frames have varying actions
    and the last T_pause frames are paused (repeated last action).
    """
    rng = np.random.default_rng(seed)
    T = T_active + T_pause

    obs = rng.standard_normal((T, 6)).astype(np.float32)

    # Active frames: random actions with large deltas
    active_actions = rng.standard_normal((T_active, action_dim)).astype(np.float32)
    active_actions *= 0.1  # ensure delta > epsilon

    # Pause frames: repeat last active action + tiny noise well below epsilon
    last_action = active_actions[-1]
    pause_noise = rng.standard_normal((T_pause, action_dim)).astype(np.float32) * (
        epsilon * 0.01
    )
    pause_actions = np.tile(last_action, (T_pause, 1)) + pause_noise

    actions = np.concatenate([active_actions, pause_actions], axis=0)

    return Episode(
        episode_hash="paused_episode",
        observations=obs,
        actions=actions,
        embodiment="eva_bimanual",
        metadata={"total_frames": T},
    )


# ---------------------------------------------------------------------------
# PauseFilter
# ---------------------------------------------------------------------------


class TestPauseFilter:
    def test_removes_pause_frames(self):
        T_active, T_pause = 50, 50
        ep = _make_episode_with_pauses(T_active=T_active, T_pause=T_pause, epsilon=1e-3)

        filtered = PauseFilter(epsilon=1e-3)(ep)
        assert filtered is not None, "Episode should survive (has active frames)"

        # Pause block of T_pause identical frames → collapsed to 1 representative
        # So output length ≈ T_active + 1 (≤ T_active + 2 due to boundary effects)
        assert (
            len(filtered.actions) <= T_active + 2
        ), f"Expected ≤ {T_active + 2} frames, got {len(filtered.actions)}"
        assert (
            len(filtered.actions) >= T_active - 1
        ), f"Expected ≥ {T_active - 1} active frames, got {len(filtered.actions)}"

    def test_all_pause_episode_not_none(self):
        """An all-pause episode should collapse to exactly 2 frames.

        Frame 0: always kept.
        Frame 1: first frame of the pause run → kept (in_pause transition).
        Frame 2+: duplicate pause frames → dropped.
        """
        T = 30
        actions = np.zeros((T, 4), dtype=np.float32)
        obs = np.zeros((T, 4), dtype=np.float32)
        ep = Episode("all_pause", obs, actions, "eva_bimanual", {})
        filtered = PauseFilter(epsilon=1e-3)(ep)
        assert filtered is not None
        assert len(filtered.actions) == 2

    def test_no_pauses_unchanged_length(self):
        """Episode with all large action deltas should keep all frames."""
        T = 40
        rng = np.random.default_rng(1)
        # Each action delta clearly above epsilon
        actions = np.cumsum(rng.uniform(0.1, 0.5, (T, 3)).astype(np.float32), axis=0)
        obs = rng.standard_normal((T, 3)).astype(np.float32)
        ep = Episode("no_pause", obs, actions, "eva_bimanual", {})
        filtered = PauseFilter(epsilon=1e-3)(ep)
        assert filtered is not None
        assert len(filtered.actions) == T

    def test_embodiment_override(self):
        """Higher epsilon for aria should collapse more frames."""
        T = 40
        rng = np.random.default_rng(5)
        # Deltas of 0.003: above eva epsilon (0.001), below aria epsilon (0.005)
        base = np.zeros((T, 4), dtype=np.float32)
        noise = rng.standard_normal((T, 4)).astype(np.float32) * 0.003
        actions = base + noise
        actions[0] = 0.0  # ensure first frame is reference
        obs = np.zeros((T, 4), dtype=np.float32)

        ep_aria = Episode("aria_ep", obs, actions, "aria_bimanual", {})
        ep_eva = Episode("eva_ep", obs.copy(), actions.copy(), "eva_bimanual", {})

        f = PauseFilter(
            epsilon=0.001,
            embodiment_overrides={"aria_bimanual": 0.005},
        )
        filtered_aria = f(ep_aria)
        filtered_eva = f(ep_eva)

        # Aria should collapse more (larger epsilon → more pauses detected)
        assert filtered_aria is not None
        assert filtered_eva is not None
        assert len(filtered_aria.actions) < len(filtered_eva.actions)

    def test_preserves_episode_hash_and_embodiment(self):
        ep = _make_episode_with_pauses()
        filtered = PauseFilter()(ep)
        assert filtered is not None
        assert filtered.episode_hash == ep.episode_hash
        assert filtered.embodiment == ep.embodiment


# ---------------------------------------------------------------------------
# ActionClipFilter
# ---------------------------------------------------------------------------


class TestActionClipFilter:
    def test_output_within_clip(self):
        """After filtering, all action values must be within [-clip, clip]."""
        T, action_dim = 80, 6
        rng = np.random.default_rng(10)
        actions = rng.standard_normal((T, action_dim)).astype(np.float32)
        # Insert extreme outliers
        actions[10] = 100.0
        actions[50] = -50.0
        obs = rng.standard_normal((T, 4)).astype(np.float32)
        ep = Episode("clip_test", obs, actions, "eva_bimanual", {})

        filtered = ActionClipFilter(clip=1.0)(ep)
        assert filtered is not None
        assert np.all(filtered.actions >= -1.0), "Actions below -1"
        assert np.all(filtered.actions <= 1.0), "Actions above +1"

    def test_does_not_change_length(self):
        ep = _make_episode(T=60)
        filtered = ActionClipFilter()(ep)
        assert filtered is not None
        assert len(filtered.actions) == 60

    def test_observations_unchanged(self):
        ep = _make_episode(T=50)
        original_obs = ep.observations.copy()
        filtered = ActionClipFilter()(ep)
        assert filtered is not None
        np.testing.assert_array_equal(filtered.observations, original_obs)


# ---------------------------------------------------------------------------
# MinLengthFilter
# ---------------------------------------------------------------------------


class TestMinLengthFilter:
    def test_short_episode_discarded(self):
        ep = _make_episode(T=10)
        result = MinLengthFilter(min_length=20)(ep)
        assert result is None

    def test_exact_length_kept(self):
        ep = _make_episode(T=20)
        result = MinLengthFilter(min_length=20)(ep)
        assert result is not None

    def test_longer_episode_kept(self):
        ep = _make_episode(T=100)
        result = MinLengthFilter(min_length=20)(ep)
        assert result is not None


# ---------------------------------------------------------------------------
# EmbodimentAwareFilter
# ---------------------------------------------------------------------------


class TestEmbodimentAwareFilter:
    def test_passes_through_active_episode(self):
        T = 30
        rng = np.random.default_rng(42)
        actions = np.cumsum(rng.uniform(0.05, 0.2, (T, 4)).astype(np.float32), axis=0)
        obs = rng.standard_normal((T, 4)).astype(np.float32)
        ep = Episode("aria_active", obs, actions, "aria_bimanual", {})
        f = EmbodimentAwareFilter(
            default_epsilon=0.001, embodiment_epsilons={"aria_bimanual": 0.005}
        )
        assert f(ep) is not None


# ---------------------------------------------------------------------------
# apply_filters pipeline
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_pipeline_removes_short_episodes(self):
        episodes = [
            _make_episode(T=5, episode_hash="short"),
            _make_episode(T=100, episode_hash="long"),
        ]
        filters = [MinLengthFilter(min_length=20)]
        kept, removed = apply_filters(episodes, filters)
        assert len(kept) == 1
        assert kept[0].episode_hash == "long"
        assert "short" in removed

    def test_empty_filter_list_keeps_all(self):
        episodes = [_make_episode(T=50, episode_hash=f"ep{i}") for i in range(5)]
        kept, removed = apply_filters(episodes, [])
        assert len(kept) == 5
        assert len(removed) == 0

    def test_full_pipeline_order(self):
        """Pause → clip → min_length pipeline should not crash."""
        episodes = [
            _make_episode_with_pauses(T_active=30, T_pause=30, seed=i) for i in range(5)
        ]
        filters = [
            PauseFilter(epsilon=1e-3),
            ActionClipFilter(),
            MinLengthFilter(min_length=5),
        ]
        kept, removed = apply_filters(episodes, filters)
        assert len(kept) + len(removed) == 5
