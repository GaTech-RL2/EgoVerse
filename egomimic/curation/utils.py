"""
Data loading utilities for DemInf curation.

Loads episodes from local Zarr stores, extracting raw proprioceptive
observations and actions without applying embodiment transforms.
Transforms are intentionally skipped here — raw coordinates are sufficient
for MI estimation and avoid introducing transform-induced correlations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import zarr

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-embodiment zarr key definitions
# ---------------------------------------------------------------------------

# Maps embodiment string (from zarr metadata) → (obs_keys, action_keys).
# obs_keys: proprioceptive observation arrays (float32, single timestep).
# action_keys: continuous action arrays (float32, single timestep).
# Image keys are excluded — use StateEmbedder(mode="image") if needed.
EMBODIMENT_KEY_MAP: dict[str, dict[str, list[str]]] = {
    # Aria bimanual egocentric hand-tracking
    "aria_bimanual": {
        "obs_keys": ["right.obs_ee_pose", "left.obs_ee_pose", "obs_head_pose"],
        "action_keys": ["right.obs_ee_pose", "left.obs_ee_pose"],
        "image_keys": ["images.front_1"],
    },
    "aria_right_arm": {
        "obs_keys": ["right.obs_ee_pose", "obs_head_pose"],
        "action_keys": ["right.obs_ee_pose"],
        "image_keys": ["images.front_1"],
    },
    "aria_left_arm": {
        "obs_keys": ["left.obs_ee_pose", "obs_head_pose"],
        "action_keys": ["left.obs_ee_pose"],
        "image_keys": ["images.front_1"],
    },
    # Eva single right arm
    "eva_right_arm": {
        "obs_keys": ["right.obs_ee_pose", "right.gripper"],
        "action_keys": ["right.cmd_ee_pose", "right.gripper"],
        "image_keys": ["images.front_1", "images.right_wrist"],
    },
    # Eva bimanual robot arm
    "eva_bimanual": {
        "obs_keys": [
            "right.obs_ee_pose",
            "right.gripper",
            "left.obs_ee_pose",
            "left.gripper",
        ],
        "action_keys": [
            "right.cmd_ee_pose",
            "left.cmd_ee_pose",
            "right.gripper",
            "left.gripper",
        ],
        "image_keys": ["images.front_1", "images.right_wrist", "images.left_wrist"],
    },
    # ALOHA bimanual robot arm (alias)
    "aloha_bimanual": {
        "obs_keys": [
            "right.obs_ee_pose",
            "right.gripper",
            "left.obs_ee_pose",
            "left.gripper",
        ],
        "action_keys": [
            "right.cmd_ee_pose",
            "left.cmd_ee_pose",
            "right.gripper",
            "left.gripper",
        ],
        "image_keys": ["images.front_1", "images.right_wrist", "images.left_wrist"],
    },
    # Scale bimanual hand-tracking (human demonstration, no robot)
    # obs: full hand keypoints (21 joints × 3D = 63D per hand) + wrist + head
    # action: keypoints are the primary motion signal (same as obs in human demos)
    "scale_bimanual": {
        "obs_keys": [
            "right.obs_keypoints",
            "left.obs_keypoints",
            "right.obs_ee_pose",
            "left.obs_ee_pose",
            "obs_head_pose",
        ],
        "action_keys": [
            "right.obs_keypoints",
            "left.obs_keypoints",
        ],
        "image_keys": ["images.front_1"],
    },
}

# Fallback for unknown embodiments: use whatever float32 keys are present.
_FALLBACK_EMBODIMENT = "unknown"


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """Standardized episode representation for curation."""

    episode_hash: str
    observations: np.ndarray  # (T, obs_dim) proprioceptive or (T, C, H, W) images
    actions: np.ndarray  # (T, action_dim) raw actions
    embodiment: str  # e.g. "aria_bimanual", "eva_bimanual"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Key resolution helpers
# ---------------------------------------------------------------------------


def _resolve_keys(
    store: zarr.Group,
    embodiment: str,
) -> tuple[list[str], list[str]]:
    """
    Return (obs_keys, action_keys) present in *store* for *embodiment*.

    Falls back to all float32 arrays when the embodiment is unknown.
    """
    key_spec = EMBODIMENT_KEY_MAP.get(embodiment)
    available = set(store.array_keys())

    if key_spec is not None:
        obs_keys = [k for k in key_spec["obs_keys"] if k in available]
        action_keys = [k for k in key_spec["action_keys"] if k in available]
        if obs_keys and action_keys:
            return obs_keys, action_keys
        logger.warning(
            "Embodiment '%s' matched but some keys missing. "
            "available=%s, wanted obs=%s action=%s",
            embodiment,
            available,
            key_spec["obs_keys"],
            key_spec["action_keys"],
        )

    # Fallback: use all numeric (non-image, non-annotation) keys for both obs and actions.
    features = store.attrs.get("features", {})
    numeric_keys = [
        k
        for k, info in features.items()
        if isinstance(info, dict)
        and info.get("dtype") not in ("jpeg", "json")
        and k in available
    ]
    if not numeric_keys:
        # Last resort: every array key
        numeric_keys = list(available)

    return numeric_keys, numeric_keys


def _read_full_array(store: zarr.Group, key: str) -> np.ndarray | None:
    """Read an entire zarr array as float32, returning None on error."""
    try:
        arr = store[key][:]
        return arr.astype(np.float32)
    except Exception as exc:
        logger.debug("Could not read key '%s': %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------------


def load_episode_from_path(
    path: Path,
    episode_hash: str | None = None,
    custom_obs_keys: list[str] | None = None,
    custom_action_keys: list[str] | None = None,
) -> Episode | None:
    """
    Load a single episode from a local Zarr directory.

    Args:
        path: Path to the .zarr episode directory (or bare hash directory).
        episode_hash: Override the hash; inferred from path.name if None.
        custom_obs_keys: Override observation zarr keys.
        custom_action_keys: Override action zarr keys.

    Returns:
        Episode or None if loading fails.
    """
    path = Path(path)
    if episode_hash is None:
        name = path.name
        episode_hash = name[: -len(".zarr")] if name.endswith(".zarr") else name

    try:
        store = zarr.open_group(str(path), mode="r")
    except Exception as exc:
        logger.warning("Failed to open zarr store at %s: %s", path, exc)
        return None

    metadata = dict(store.attrs)
    embodiment = metadata.get("embodiment", _FALLBACK_EMBODIMENT)

    if custom_obs_keys is not None and custom_action_keys is not None:
        obs_keys, action_keys = custom_obs_keys, custom_action_keys
    else:
        obs_keys, action_keys = _resolve_keys(store, embodiment)

    if not obs_keys or not action_keys:
        logger.warning(
            "No usable keys for episode %s (embodiment=%s)", episode_hash, embodiment
        )
        return None

    # Read and concatenate observations
    obs_arrays = [_read_full_array(store, k) for k in obs_keys]
    obs_arrays = [a for a in obs_arrays if a is not None]
    if not obs_arrays:
        logger.warning("No obs arrays loaded for %s", episode_hash)
        return None

    # Align along time axis (take minimum length)
    T = min(a.shape[0] for a in obs_arrays)
    obs_parts = []
    for a in obs_arrays:
        a = a[:T]
        if a.ndim == 1:
            a = a[:, None]
        obs_parts.append(a)
    observations = np.concatenate(obs_parts, axis=-1)  # (T, obs_dim)

    # Read and concatenate actions
    act_arrays = [_read_full_array(store, k) for k in action_keys]
    act_arrays = [a for a in act_arrays if a is not None]
    if not act_arrays:
        logger.warning("No action arrays loaded for %s", episode_hash)
        return None

    T_act = min(a.shape[0] for a in act_arrays)
    T = min(T, T_act)
    act_parts = []
    for a in act_arrays:
        a = a[:T]
        if a.ndim == 1:
            a = a[:, None]
        act_parts.append(a)
    actions = np.concatenate(act_parts, axis=-1)  # (T, action_dim)
    observations = observations[:T]

    return Episode(
        episode_hash=episode_hash,
        observations=observations,
        actions=actions,
        embodiment=embodiment,
        metadata=metadata,
    )


def load_episodes_from_dir(
    folder_path: Path | str,
    valid_hashes: set[str] | None = None,
    custom_obs_keys: list[str] | None = None,
    custom_action_keys: list[str] | None = None,
    max_episodes: int | None = None,
    max_timesteps_per_episode: int | None = None,
) -> list[Episode]:
    """
    Load all episodes from a local directory of Zarr stores.

    Args:
        folder_path: Root directory containing {episode_hash}/ subdirectories.
        valid_hashes: If provided, only load episodes whose hash is in this set.
        custom_obs_keys: Override observation zarr keys for all episodes.
        custom_action_keys: Override action zarr keys for all episodes.
        max_episodes: Cap on number of episodes to load (for debugging).

    Returns:
        List of successfully loaded Episodes.
    """
    folder_path = Path(folder_path)
    episodes: list[Episode] = []

    all_dirs = sorted(p for p in folder_path.iterdir() if p.is_dir())

    for ep_dir in all_dirs:
        if max_episodes is not None and len(episodes) >= max_episodes:
            break

        name = ep_dir.name
        episode_hash = name[: -len(".zarr")] if name.endswith(".zarr") else name

        if valid_hashes is not None and episode_hash not in valid_hashes:
            continue

        ep = load_episode_from_path(
            ep_dir,
            episode_hash=episode_hash,
            custom_obs_keys=custom_obs_keys,
            custom_action_keys=custom_action_keys,
        )
        if ep is not None:
            if (
                max_timesteps_per_episode is not None
                and len(ep.observations) > max_timesteps_per_episode
            ):
                indices = np.linspace(
                    0, len(ep.observations) - 1, max_timesteps_per_episode, dtype=int
                )
                ep = Episode(
                    episode_hash=ep.episode_hash,
                    observations=ep.observations[indices],
                    actions=ep.actions[indices],
                    embodiment=ep.embodiment,
                    metadata=ep.metadata,
                )
            episodes.append(ep)

    logger.info("Loaded %d episodes from %s", len(episodes), folder_path)
    return episodes


# ---------------------------------------------------------------------------
# EgoVerse pipeline integration
# ---------------------------------------------------------------------------


def load_episodes_from_config(
    data_cfg: "DictConfig",
    max_episodes: Optional[int] = None,
    max_timesteps_per_episode: Optional[int] = None,
) -> list[Episode]:
    """
    Load episodes using EgoVerse's existing data pipeline configuration.

    Supports two DictConfig formats:

    1. **Training data config** (aria.yaml style) — has a ``train_datasets`` key.
       Each sub-config has a ``resolver.folder_path`` and optional ``filters``
       (DatasetFilter with filter_lambdas).  LocalEpisodeResolver is used to
       enumerate episodes matching the filters from the local filesystem.

    2. **Curation config** — has a flat ``folder_path`` key plus an optional
       ``filter_lambdas`` list and/or ``valid_hashes_path``.

    In both cases episodes are streamed one at a time (not pre-loaded into a
    giant tensor), so memory use scales with the largest episode, not the
    full dataset.  For very long Aria sessions you can pass
    ``max_timesteps_per_episode`` to uniformly subsample within each episode.

    Args:
        data_cfg: Hydra DictConfig describing data source(s) and filters.
        max_episodes: Stop after loading this many episodes (debugging / dry-run).
        max_timesteps_per_episode: Uniformly subsample long episodes to at most
            this many timesteps.  ``None`` keeps all timesteps.

    Returns:
        List of Episode objects with observations, actions, embodiment, and metadata.

    Raises:
        ImportError: if EgoVerse pipeline packages are not on ``sys.path``.
        ValueError: if the config format is not recognised.
        FileNotFoundError: if ``folder_path`` does not exist (curation config).
    """
    from omegaconf import OmegaConf

    cfg: dict = OmegaConf.to_container(data_cfg, resolve=True)  # type: ignore[assignment]

    if "train_datasets" in cfg:
        return _load_from_training_config(cfg, max_episodes, max_timesteps_per_episode)
    if "folder_path" in cfg:
        return _load_from_curation_config(cfg, max_episodes, max_timesteps_per_episode)

    raise ValueError(
        "Unrecognised data_cfg format: expected 'train_datasets' (training config) "
        "or 'folder_path' (curation config) at the top level."
    )


# ---------------------------------------------------------------------------
# Internal helpers for EgoVerse data loading
# ---------------------------------------------------------------------------


def _load_and_subsample(
    path: Path,
    episode_hash: str,
    max_timesteps: Optional[int],
) -> Episode | None:
    """Load a single episode and optionally subsample its timesteps."""
    ep = load_episode_from_path(path, episode_hash=episode_hash)
    if ep is None:
        return None
    if max_timesteps is not None and len(ep.observations) > max_timesteps:
        indices = np.linspace(0, len(ep.observations) - 1, max_timesteps, dtype=int)
        ep = Episode(
            episode_hash=ep.episode_hash,
            observations=ep.observations[indices],
            actions=ep.actions[indices],
            embodiment=ep.embodiment,
            metadata=ep.metadata,
        )
    return ep


def _load_from_training_config(
    cfg: dict,
    max_episodes: Optional[int],
    max_timesteps_per_episode: Optional[int],
) -> list[Episode]:
    """
    Load from a training data config (aria.yaml style) with ``train_datasets``.

    Uses LocalEpisodeResolver from the EgoVerse pipeline to enumerate episodes
    that match the configured filter_lambdas without downloading from S3.
    """
    try:
        from egomimic.rldb.filters import DatasetFilter
        from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
    except ImportError as exc:
        raise ImportError(
            "EgoVerse packages not found. Ensure the EgoVerse repo is on sys.path."
        ) from exc

    episodes: list[Episode] = []

    for ds_name, ds_cfg in cfg.get("train_datasets", {}).items():
        if max_episodes is not None and len(episodes) >= max_episodes:
            break

        resolver_cfg = ds_cfg.get("resolver") or {}
        folder_path = Path(resolver_cfg.get("folder_path", ""))
        if not folder_path.is_dir():
            logger.warning(
                "resolver.folder_path does not exist: %s — skipping dataset '%s'",
                folder_path,
                ds_name,
            )
            continue

        # Build DatasetFilter from filter_lambdas list
        filters_cfg = ds_cfg.get("filters") or {}
        filter_lambdas: list[str] = filters_cfg.get("filter_lambdas") or []
        dataset_filter = DatasetFilter(filter_lambdas)

        # Enumerate episodes matching the filter (local filesystem only)
        try:
            filtered_paths = LocalEpisodeResolver._get_local_filtered_paths(
                search_path=folder_path,
                filters=dataset_filter,
            )
        except Exception as exc:
            logger.warning(
                "LocalEpisodeResolver failed for '%s' at %s: %s",
                ds_name,
                folder_path,
                exc,
            )
            continue

        logger.info(
            "Dataset '%s': %d episode(s) matched at %s",
            ds_name,
            len(filtered_paths),
            folder_path,
        )

        for path_str, episode_hash in filtered_paths:
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            ep = _load_and_subsample(
                path=Path(path_str),
                episode_hash=episode_hash,
                max_timesteps=max_timesteps_per_episode,
            )
            if ep is not None:
                episodes.append(ep)

    logger.info("Loaded %d episode(s) from training data config", len(episodes))
    return episodes


def _load_from_curation_config(
    cfg: dict,
    max_episodes: Optional[int],
    max_timesteps_per_episode: Optional[int],
) -> list[Episode]:
    """
    Load from a curation data config with a flat ``folder_path`` key.

    Supports:
    - ``filter_lambdas``: list of lambda strings forwarded to DatasetFilter
    - ``valid_hashes_path``: path to a JSON list of episode hashes (compat shim)
    """
    try:
        from egomimic.rldb.filters import DatasetFilter
        from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
    except ImportError as exc:
        raise ImportError(
            "EgoVerse packages not found. Ensure the EgoVerse repo is on sys.path."
        ) from exc

    folder_path = Path(cfg["folder_path"])
    if not folder_path.is_dir():
        raise FileNotFoundError(f"folder_path does not exist: {folder_path}")

    filter_lambdas: list[str] = list(cfg.get("filter_lambdas") or [])

    # Backward-compat: valid_hashes_path → inject a filter lambda
    valid_hashes_path = cfg.get("valid_hashes_path")
    if valid_hashes_path:
        with open(valid_hashes_path) as fh:
            valid_hashes: frozenset[str] = frozenset(json.load(fh))
        # Embed as a frozenset literal so DatasetFilter can eval() it safely
        filter_lambdas.append(
            f"lambda row: row.get('episode_hash') in {set(valid_hashes)!r}"
        )

    dataset_filter = DatasetFilter(filter_lambdas)
    filtered_paths = LocalEpisodeResolver._get_local_filtered_paths(
        search_path=folder_path,
        filters=dataset_filter,
    )

    episodes: list[Episode] = []
    for path_str, episode_hash in filtered_paths:
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        ep = _load_and_subsample(
            path=Path(path_str),
            episode_hash=episode_hash,
            max_timesteps=max_timesteps_per_episode,
        )
        if ep is not None:
            episodes.append(ep)

    logger.info("Loaded %d episode(s) from curation data config", len(episodes))
    return episodes
