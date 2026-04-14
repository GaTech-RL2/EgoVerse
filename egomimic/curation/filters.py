"""
Composable preprocessing filters for DemInf curation.

Each filter is a callable: filter(episode) -> filtered_episode | None.
Returning None signals that the episode should be discarded entirely.

Order of application matters:
    1. PauseFilter   — remove pause frames (inflates MI if left in)
    2. ActionClipFilter — normalise and clip extreme teleop glitches
    3. MinLengthFilter — discard episodes that are too short after filtering
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from egomimic.curation.utils import Episode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol / base
# ---------------------------------------------------------------------------


class EpisodeFilter(Protocol):
    """A callable that maps Episode → Episode | None."""

    def __call__(self, episode: Episode) -> Episode | None: ...


# ---------------------------------------------------------------------------
# PauseFilter
# ---------------------------------------------------------------------------


class PauseFilter:
    """
    Remove timesteps where the action barely changes between consecutive frames.

    Consecutive pause frames are collapsed to a single representative frame
    (the first frame of each pause run) rather than being dropped entirely,
    which preserves context at pause boundaries.

    DemInf (Hejna et al. 2025) explicitly warns that long pauses make the
    action distribution trivially predictable, inflating MI scores despite
    no task progress.

    Args:
        epsilon: L2-norm threshold below which an action delta is a "pause".
        embodiment_overrides: Per-embodiment epsilon overrides, e.g.
            {"aria_bimanual": 0.005} for noisier hand-tracking data.
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        embodiment_overrides: dict[str, float] | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.embodiment_overrides = embodiment_overrides or {}

    def _effective_epsilon(self, embodiment: str) -> float:
        return self.embodiment_overrides.get(embodiment, self.epsilon)

    def __call__(self, episode: Episode) -> Episode | None:
        eps = self._effective_epsilon(episode.embodiment)
        actions = episode.actions  # (T, action_dim)
        T = len(actions)

        if T < 2:
            return episode

        # Compute per-timestep action deltas (||a_t - a_{t-1}||)
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=-1)  # (T-1,)

        # Build keep mask.
        # Always keep the first frame.
        # For t >= 1: keep frame t if delta[t-1] >= eps OR if the previous
        # kept frame was also a pause start (collapse consecutive pauses).
        keep = np.ones(T, dtype=bool)
        in_pause = False
        for t in range(1, T):
            if deltas[t - 1] < eps:
                if in_pause:
                    keep[t] = False  # drop duplicate pause frame
                else:
                    in_pause = True  # first frame of a new pause — keep it
            else:
                in_pause = False

        indices = np.where(keep)[0]
        if len(indices) == 0:
            return None

        return Episode(
            episode_hash=episode.episode_hash,
            observations=episode.observations[indices],
            actions=episode.actions[indices],
            embodiment=episode.embodiment,
            metadata=episode.metadata,
        )


# ---------------------------------------------------------------------------
# ActionClipFilter
# ---------------------------------------------------------------------------


class ActionClipFilter:
    """
    Per-episode Gaussian normalisation + clipping of actions to [-clip, clip].

    Removes teleop glitches (large outlier actions) that would otherwise
    dominate the embedding space and distort MI estimates.

    Args:
        clip: Clip value after normalisation (default 1.0 → unit sphere).
        min_std: Minimum std to avoid division by near-zero.
    """

    def __init__(self, clip: float = 1.0, min_std: float = 1e-6) -> None:
        self.clip = clip
        self.min_std = min_std

    def __call__(self, episode: Episode) -> Episode | None:
        actions = episode.actions.copy()
        mean = actions.mean(axis=0, keepdims=True)
        std = np.maximum(actions.std(axis=0, keepdims=True), self.min_std)
        actions = np.clip((actions - mean) / std, -self.clip, self.clip)

        return Episode(
            episode_hash=episode.episode_hash,
            observations=episode.observations,
            actions=actions,
            embodiment=episode.embodiment,
            metadata=episode.metadata,
        )


# ---------------------------------------------------------------------------
# MinLengthFilter
# ---------------------------------------------------------------------------


class MinLengthFilter:
    """
    Discard episodes shorter than `min_length` timesteps.

    Applied after PauseFilter and ActionClipFilter so the length check
    reflects the actual usable content.

    Args:
        min_length: Minimum number of timesteps required (default 20).
    """

    def __init__(self, min_length: int = 20) -> None:
        self.min_length = min_length

    def __call__(self, episode: Episode) -> Episode | None:
        if len(episode.actions) < self.min_length:
            logger.debug(
                "Episode %s discarded: length %d < %d",
                episode.episode_hash,
                len(episode.actions),
                self.min_length,
            )
            return None
        return episode


# ---------------------------------------------------------------------------
# EmbodimentAwareFilter
# ---------------------------------------------------------------------------


class EmbodimentAwareFilter:
    """
    Wraps PauseFilter with per-embodiment epsilon overrides.

    Convenience wrapper so Hydra configs can specify embodiment-specific
    filtering without wiring overrides manually.

    Args:
        default_epsilon: Fallback epsilon for unlisted embodiments.
        embodiment_epsilons: Dict of embodiment → epsilon.
    """

    def __init__(
        self,
        default_epsilon: float = 1e-3,
        embodiment_epsilons: dict[str, float] | None = None,
    ) -> None:
        self._pause_filter = PauseFilter(
            epsilon=default_epsilon,
            embodiment_overrides=embodiment_epsilons or {},
        )

    def __call__(self, episode: Episode) -> Episode | None:
        return self._pause_filter(episode)


# ---------------------------------------------------------------------------
# Filter pipeline helper
# ---------------------------------------------------------------------------


def apply_filters(
    episodes: list[Episode],
    filters: list[EpisodeFilter],
) -> tuple[list[Episode], list[str]]:
    """
    Apply a list of filters to a list of episodes.

    Args:
        episodes: Input episodes.
        filters: Ordered list of filter callables.

    Returns:
        (kept_episodes, removed_hashes)
    """
    kept: list[Episode] = []
    removed: list[str] = []

    for ep in episodes:
        result: Episode | None = ep
        for f in filters:
            if result is None:
                break
            result = f(result)

        if result is None:
            removed.append(ep.episode_hash)
        else:
            kept.append(result)

    logger.info(
        "Filters: %d kept, %d removed (%.1f%%)",
        len(kept),
        len(removed),
        100.0 * len(removed) / max(len(episodes), 1),
    )
    return kept, removed
