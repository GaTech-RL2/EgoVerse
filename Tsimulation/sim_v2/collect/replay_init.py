"""Load, identify, and restore saved PushShapes episode initial states.

The interactive collector supports two replay sources: existing Zarr episodes
and JSON manifests.  Keeping the storage details here leaves the collector's
event loop focused on collection rather than dataset bookkeeping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import zarr

from Tsimulation.pushshapes.env import PushShapesEnv

REPLAY_SOURCE_KEY = "_replay_source"

EpisodeInit = dict[str, Any]
PoseKey = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]
SeedKey = tuple[int, int]


def _read_episode_init(path: Path) -> EpisodeInit | None:
    """Return a Zarr episode's stored initial state, if it has one."""
    try:
        group = zarr.open_group(str(path), mode="r")
        raw = group.attrs.get("episode_init")
        return None if raw is None else json.loads(raw)
    except Exception:
        # A partially written or unrelated Zarr directory should not prevent
        # resume from inspecting the rest of a dataset.
        return None


def load_replay_inits(source_dir: Path) -> list[EpisodeInit]:
    """Load replayable initial states from Zarr episodes in filename order."""
    inits: list[EpisodeInit] = []
    for entry in sorted(source_dir.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        episode_init = _read_episode_init(entry)
        if episode_init is None:
            continue
        episode_init[REPLAY_SOURCE_KEY] = entry.name
        inits.append(episode_init)
    return inits


def load_replay_manifest(path: Path) -> list[EpisodeInit]:
    """Load a replay manifest and normalize its source marker."""
    manifest = json.loads(path.read_text())
    episodes = manifest["episodes"] if isinstance(manifest, dict) else manifest
    if not isinstance(episodes, list):
        raise ValueError("replay manifest must contain a list of episodes")

    normalized: list[EpisodeInit] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("each replay manifest episode must be an object")
        episode_init = dict(episode)
        episode_init.setdefault(REPLAY_SOURCE_KEY, episode_init.get("source", "?"))
        normalized.append(episode_init)
    return normalized


def init_pose_key(episode_init: EpisodeInit) -> PoseKey:
    """Build a stable resume key from an episode's poses and obstacle level.

    Six-decimal rounding absorbs JSON round-trip jitter without collapsing
    genuinely distinct sampled poses.  Including the obstacle level prevents
    identical poses from different levels being mistaken for each other.
    """

    def rounded(values: Any) -> tuple[float, ...]:
        return tuple(round(float(value), 6) for value in values)

    return (
        rounded(episode_init["agent_pos"]),
        rounded(episode_init["object_pose"]),
        rounded(episode_init["goal_pose"]),
        int(episode_init.get("obstacle_level", -1)),
    )


def collected_resume_keys(output_dir: Path) -> tuple[set[str], set[PoseKey]]:
    """Return source names and legacy pose keys present in ``output_dir``.

    A source-tagged replacement is complete only for that exact source name.
    Different source episodes can legitimately share an initial pose, so
    using their pose as a second resume key would incorrectly skip work. Pose
    matching is retained only for older outputs that have no source marker.
    """
    source_names: set[str] = set()
    pose_keys: set[PoseKey] = set()
    if not output_dir.exists():
        return source_names, pose_keys

    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        episode_init = _read_episode_init(entry)
        if episode_init is None:
            continue
        source_name = episode_init.get(REPLAY_SOURCE_KEY)
        if source_name:
            source_names.add(str(source_name))
            continue
        try:
            pose_keys.add(init_pose_key(episode_init))
        except (KeyError, TypeError, ValueError):
            # Old or external episodes may not carry the full pose schema.
            continue
    return source_names, pose_keys


def collected_seed_keys(output_dir: Path) -> set[SeedKey]:
    """Return ``(obstacle_level, reset_seed)`` pairs already collected."""
    keys: set[SeedKey] = set()
    if not output_dir.exists():
        return keys
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        episode_init = _read_episode_init(entry)
        if episode_init is None or episode_init.get("reset_seed") is None:
            continue
        try:
            keys.add(
                (
                    int(episode_init.get("obstacle_level", -1)),
                    int(episode_init["reset_seed"]),
                )
            )
        except (TypeError, ValueError):
            continue
    return keys


def reset_to_init(env: PushShapesEnv, episode_init: EpisodeInit) -> tuple[dict, dict]:
    """Reset ``env`` and restore an exact saved initial configuration."""
    _observation, info = env.reset(seed=episode_init.get("reset_seed"))
    if "obstacles" in episode_init:
        env.set_obstacles(episode_init["obstacles"])
    env.set_state(
        agent_pos=tuple(episode_init["agent_pos"]),
        agent_angle=float(episode_init.get("agent_angle", 0.0)),
        object_pose=tuple(episode_init["object_pose"]),
        goal_pose=tuple(episode_init["goal_pose"]),
    )
    return env._get_obs(), info
