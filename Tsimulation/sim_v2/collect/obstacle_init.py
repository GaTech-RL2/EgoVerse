"""Curate reproducible, obstacle-relevant reset seeds for manual collection.

The default :class:`PushShapesEnv` reset distribution is intentionally left
unchanged for replay compatibility.  This module builds an explicit per-level
seed bank for collection and verifies every seed again when it is consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from shapely.geometry import LineString, Polygon

from Tsimulation.sim_v2.pushshapes.env import SIM_VERSION, PushShapesEnv
from Tsimulation.sim_v2.pushshapes.obstacles import OBSTACLE_LEVELS, WALL_RADIUS

SCHEMA_VERSION = 1
SAMPLER_REVISION = "chain_obstacle_seed_bank_v1"


@dataclass(frozen=True)
class ObstacleInitCriteria:
    """Acceptance thresholds for one deterministic reset candidate."""

    min_start_goal_distance: float = 180.0
    max_agent_object_distance: float = 220.0
    min_obstacle_clearance: float = 10.0
    min_arena_clearance: float = 8.0
    sweep_samples: int = 31
    min_blocked_samples: int = 2
    endpoint_margin: float = 0.10
    overlap_tolerance: float = 1e-8
    seed_limit: int = 10_000
    pool_multiplier: int = 4


DEFAULT_CRITERIA = ObstacleInitCriteria()


def obstacle_geometry_hash(level: int) -> str:
    """Stable SHA-256 of one level's physical wall geometry."""
    if level not in OBSTACLE_LEVELS:
        raise ValueError(f"unknown obstacle level {level}")
    payload = json.dumps(
        {
            "world_size": PushShapesEnv.WORLD_SIZE,
            "wall_radius": WALL_RADIUS,
            "segments": OBSTACLE_LEVELS[level],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _shortest_angle_delta(start: float, goal: float) -> float:
    return float((goal - start + math.pi) % (2.0 * math.pi) - math.pi)


def _segment_lines(level: int) -> list[LineString]:
    return [LineString([start, end]) for start, end in OBSTACLE_LEVELS[level]]


def _physical_obstacle_polygons(level: int) -> list[Polygon]:
    """Return the actual wall silhouettes, without spawn-only clearance."""
    return [segment.buffer(WALL_RADIUS) for segment in _segment_lines(level)]


def _intersected_segments(
    start: Sequence[float],
    goal: Sequence[float],
    segment_lines: Sequence[LineString],
) -> list[int]:
    path = LineString([start[:2], goal[:2]])
    return [
        index
        for index, segment in enumerate(segment_lines)
        if path.intersects(segment.buffer(WALL_RADIUS))
    ]


def _crossing_direction(
    start: Sequence[float],
    goal: Sequence[float],
    segment: Sequence[Sequence[float]],
) -> int:
    (ax, ay), (bx, by) = segment

    def side(point: Sequence[float]) -> float:
        return (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)

    start_side = side(start)
    goal_side = side(goal)
    # A crossing direction is meaningful only when the endpoints lie on
    # opposite sides of the segment's supporting line.  Comparing the two
    # signed distances directly incorrectly labels almost every same-side
    # pair as a crossing.
    if start_side * goal_side >= 0.0:
        return 0
    return 1 if start_side > goal_side else -1


def _swept_collision_alphas(
    env: PushShapesEnv,
    start: Sequence[float],
    goal: Sequence[float],
    samples: int,
) -> list[float]:
    obstacle_polys = _physical_obstacle_polygons(int(env.obstacle_level))
    angle_delta = _shortest_angle_delta(float(start[2]), float(goal[2]))
    hits: list[float] = []
    for alpha in np.linspace(0.0, 1.0, samples):
        pose = (
            float(start[0] + alpha * (goal[0] - start[0])),
            float(start[1] + alpha * (goal[1] - start[1])),
            float(start[2] + alpha * angle_delta),
        )
        polygon = env._build_object_polygon(pose[:2], pose[2])
        if any(polygon.intersects(obstacle) for obstacle in obstacle_polys):
            hits.append(float(alpha))
    return hits


def _pusher_arena_overflow(env: PushShapesEnv) -> float:
    overflow = 0.0
    for shape in env.agent.physics_shapes(env):
        env._space.reindex_shapes_for_body(shape.body)
        bounds = shape.bb
        overflow = max(
            overflow,
            -float(bounds.left),
            -float(bounds.bottom),
            float(bounds.right) - env.WORLD_SIZE,
            float(bounds.top) - env.WORLD_SIZE,
        )
    return max(0.0, overflow)


def _agent_has_object_line_of_sight(
    level: int,
    agent_pos: Sequence[float],
    object_pose: Sequence[float],
) -> bool:
    path = LineString([agent_pos[:2], object_pose[:2]])
    return not any(
        path.intersects(segment.buffer(WALL_RADIUS))
        for segment in _segment_lines(level)
    )


def _arena_clearance(env: PushShapesEnv, polygon: Polygon) -> float:
    xmin, ymin, xmax, ymax = polygon.bounds
    return float(min(xmin, ymin, env.WORLD_SIZE - xmax, env.WORLD_SIZE - ymax))


def _obstacle_clearance(polygon: Polygon, obstacles: Sequence[Polygon]) -> float:
    return float(
        min((polygon.distance(obstacle) for obstacle in obstacles), default=math.inf)
    )


def evaluate_candidate(
    env: PushShapesEnv,
    seed: int,
    criteria: ObstacleInitCriteria = DEFAULT_CRITERIA,
) -> dict[str, Any] | None:
    """Reset ``env`` with ``seed`` and return an audited candidate or ``None``."""
    level = int(env.obstacle_level)
    if level <= 0:
        raise ValueError("obstacle-aware initialization requires level >= 1")
    if env.object_shape != "T" or env.pusher_shape != "chain_gripper":
        raise ValueError("curation currently requires T + chain_gripper")

    previous_skip_render = env._skip_obs_render
    env._skip_obs_render = True
    try:
        env.reset(seed=int(seed))
    finally:
        env._skip_obs_render = previous_skip_render
    start = tuple(float(value) for value in env.object_pose)
    goal = tuple(float(value) for value in env.goal_pose)
    agent = tuple(float(value) for value in env.agent_pos)
    agent_angle = float(env.pusher_angle)
    chain_joint_angle = float(env.agent.joint_angle)

    start_goal_distance = math.dist(start[:2], goal[:2])
    agent_object_distance = math.dist(agent, start[:2])
    if start_goal_distance < criteria.min_start_goal_distance:
        return None
    if agent_object_distance > criteria.max_agent_object_distance:
        return None
    if not _agent_has_object_line_of_sight(level, agent, start):
        return None

    tolerance = criteria.overlap_tolerance
    physical_obstacles = _physical_obstacle_polygons(level)
    start_polygon = env._build_object_polygon(start[:2], start[2])
    goal_polygon = env._build_object_polygon(goal[:2], goal[2])
    start_arena_clearance = _arena_clearance(env, start_polygon)
    goal_arena_clearance = _arena_clearance(env, goal_polygon)
    start_obstacle_clearance = _obstacle_clearance(start_polygon, physical_obstacles)
    goal_obstacle_clearance = _obstacle_clearance(goal_polygon, physical_obstacles)
    if min(start_arena_clearance, goal_arena_clearance) < (
        criteria.min_arena_clearance - tolerance
    ):
        return None
    if min(start_obstacle_clearance, goal_obstacle_clearance) < (
        criteria.min_obstacle_clearance - tolerance
    ):
        return None

    pusher_shapes = list(env.agent.physics_shapes(env))
    if (
        env._shapes_static_penetration_depth(env._pusher_body, pusher_shapes)
        > tolerance
    ):
        return None
    if env._pusher_object_penetration_depth() > tolerance:
        return None
    if env._object_static_penetration_depth() > tolerance:
        return None
    object_overflow, _ = env._object_arena_metrics()
    if object_overflow > tolerance or _pusher_arena_overflow(env) > tolerance:
        return None

    segment_lines = _segment_lines(level)
    hit_segments = _intersected_segments(start, goal, segment_lines)
    if not hit_segments:
        return None
    collision_alphas = _swept_collision_alphas(env, start, goal, criteria.sweep_samples)
    interior_hits = [
        alpha
        for alpha in collision_alphas
        if criteria.endpoint_margin <= alpha <= 1.0 - criteria.endpoint_margin
    ]
    if len(interior_hits) < criteria.min_blocked_samples:
        return None

    crossing_segments = [
        (segment_index, direction)
        for segment_index in hit_segments
        if (
            direction := _crossing_direction(
                start, goal, OBSTACLE_LEVELS[level][segment_index]
            )
        )
        != 0
    ]
    if not crossing_segments:
        return None
    primary_segment, direction = crossing_segments[0]
    return {
        "level": level,
        "seed": int(seed),
        "geometry_hash": obstacle_geometry_hash(level),
        "agent_pos": list(agent),
        "agent_angle": agent_angle,
        "chain_joint_angle": chain_joint_angle,
        "object_pose": list(start),
        "goal_pose": list(goal),
        "start_goal_distance": float(start_goal_distance),
        "agent_object_distance": float(agent_object_distance),
        "start_arena_clearance": start_arena_clearance,
        "goal_arena_clearance": goal_arena_clearance,
        "start_obstacle_clearance": start_obstacle_clearance,
        "goal_obstacle_clearance": goal_obstacle_clearance,
        "hit_segments": hit_segments,
        "crossing_segments": [index for index, _ in crossing_segments],
        "primary_hit_segment": int(primary_segment),
        "crossing_direction": int(direction),
        "collision_alphas": interior_hits,
        "blocked_fraction": float(len(interior_hits) / criteria.sweep_samples),
    }


def _candidate_features(candidate: dict[str, Any]) -> np.ndarray:
    positions = (
        np.asarray(
            candidate["agent_pos"]
            + candidate["object_pose"][:2]
            + candidate["goal_pose"][:2],
            dtype=np.float64,
        )
        / 512.0
    )
    angles = []
    for key in ("object_pose", "goal_pose"):
        theta = float(candidate[key][2])
        angles.extend((0.25 * math.cos(theta), 0.25 * math.sin(theta)))
    agent_angle = float(candidate["agent_angle"])
    angles.extend((0.25 * math.cos(agent_angle), 0.25 * math.sin(agent_angle)))
    angles.append(0.25 * float(candidate["chain_joint_angle"]) / math.pi)
    return np.concatenate([positions, np.asarray(angles, dtype=np.float64)])


def select_diverse_candidates(
    candidates: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Balance hit side/segment, then apply deterministic farthest-point sampling."""
    if count < 1:
        raise ValueError("count must be >= 1")
    if len(candidates) < count:
        raise ValueError(f"need {count} candidates, found {len(candidates)}")

    ordered = sorted(candidates, key=lambda item: int(item["seed"]))
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for candidate in ordered:
        key = (
            int(candidate["primary_hit_segment"]),
            int(candidate["crossing_direction"]),
        )
        groups.setdefault(key, []).append(candidate)

    features = {int(item["seed"]): _candidate_features(item) for item in ordered}
    selected: list[dict[str, Any]] = []
    selected_per_group = dict.fromkeys(groups, 0)
    used: set[int] = set()
    while len(selected) < count:
        available_groups = {
            key: [item for item in items if int(item["seed"]) not in used]
            for key, items in groups.items()
        }
        available_groups = {
            key: items for key, items in available_groups.items() if items
        }
        if not available_groups:
            raise RuntimeError("candidate selection exhausted unexpectedly")
        lowest_count = min(selected_per_group[key] for key in available_groups)
        least_represented = [
            key
            for key in sorted(available_groups)
            if selected_per_group[key] == lowest_count
        ]

        def diversity_score(item: dict[str, Any]) -> tuple[float, float, float, int]:
            feature = features[int(item["seed"])]
            min_distance = (
                min(
                    float(np.linalg.norm(feature - features[int(chosen["seed"])]))
                    for chosen in selected
                )
                if selected
                else 0.0
            )
            return (
                min_distance,
                float(item["blocked_fraction"]),
                float(item["start_goal_distance"]),
                -int(item["seed"]),
            )

        group_winners = [
            max(available_groups[key], key=diversity_score) for key in least_represented
        ]
        candidate = max(group_winners, key=diversity_score)
        candidate_group = (
            int(candidate["primary_hit_segment"]),
            int(candidate["crossing_direction"]),
        )
        selected.append(candidate)
        selected_per_group[candidate_group] += 1
        used.add(int(candidate["seed"]))
    return selected


def curate_level(
    level: int,
    count: int = 32,
    *,
    criteria: ObstacleInitCriteria = DEFAULT_CRITERIA,
    seed_start: int = 0,
    seed_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select ``count`` diverse, valid seeds for one obstacle level."""
    if level <= 0 or level not in OBSTACLE_LEVELS:
        raise ValueError(f"unknown nonzero obstacle level {level}")
    limit = criteria.seed_limit if seed_limit is None else int(seed_limit)
    pool_target = max(count, count * criteria.pool_multiplier)
    candidates: list[dict[str, Any]] = []
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        render_mode=None,
        image_size=8,
    )
    try:
        for seed in range(int(seed_start), limit):
            candidate = evaluate_candidate(env, seed, criteria)
            if candidate is not None:
                candidates.append(candidate)
                if len(candidates) >= pool_target:
                    break
    finally:
        env.close()
    if len(candidates) < count:
        raise RuntimeError(
            f"level {level}: found {len(candidates)}/{count} valid candidates "
            f"before seed {limit}"
        )
    return select_diverse_candidates(candidates, count)


def curate_manifest(
    levels: Iterable[int] = range(1, 31),
    count: int = 32,
    *,
    criteria: ObstacleInitCriteria = DEFAULT_CRITERIA,
    seed_start: int = 0,
    seed_limit: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for all requested levels."""
    resolved_levels = [int(level) for level in levels]
    if len(resolved_levels) != len(set(resolved_levels)):
        raise ValueError("levels must be unique")
    entries = {
        str(level): curate_level(
            level,
            count,
            criteria=criteria,
            seed_start=seed_start,
            seed_limit=seed_limit,
        )
        for level in resolved_levels
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sampler_revision": SAMPLER_REVISION,
        "sim_version": SIM_VERSION,
        "object_shape": "T",
        "pusher_shape": "chain_gripper",
        "entries_per_level": int(count),
        "criteria": asdict(criteria),
        "levels": entries,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a manifest deterministically and return its resolved path."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a curated-initialization manifest."""
    manifest = json.loads(Path(path).read_text())
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported obstacle-init schema: {manifest.get('schema_version')}"
        )
    if int(manifest.get("sim_version", -1)) != SIM_VERSION:
        raise ValueError(f"manifest sim_version != {SIM_VERSION}")
    if manifest.get("sampler_revision") != SAMPLER_REVISION:
        raise ValueError("manifest sampler revision does not match this code")
    if (
        manifest.get("object_shape") != "T"
        or manifest.get("pusher_shape") != "chain_gripper"
    ):
        raise ValueError("manifest is not for T + chain_gripper")
    if not isinstance(manifest.get("levels"), dict):
        raise ValueError("manifest has no levels mapping")
    return manifest


def level_entries(manifest: dict[str, Any], level: int) -> list[dict[str, Any]]:
    """Return one level's entries after validating geometry and uniqueness."""
    key = str(int(level))
    if key not in manifest.get("levels", {}):
        raise ValueError(f"manifest has no obstacle level {level}")
    entries = manifest["levels"][key]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest level {level} has no entries")
    expected_hash = obstacle_geometry_hash(level)
    seeds = [int(entry["seed"]) for entry in entries]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"manifest level {level} repeats seeds")
    if any(entry.get("geometry_hash") != expected_hash for entry in entries):
        raise ValueError(f"manifest level {level} geometry hash is stale")
    if any(int(entry.get("level", -1)) != int(level) for entry in entries):
        raise ValueError(f"manifest level {level} contains a mislabeled entry")
    return entries


def reset_to_manifest_entry(
    env: PushShapesEnv,
    entry: dict[str, Any],
    *,
    verify: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reset from a manifest seed and fail if its resolved state has drifted."""
    if env.object_shape != "T" or env.pusher_shape != "chain_gripper":
        raise ValueError("manifest entries require T + chain_gripper")
    if int(entry.get("level", -1)) != int(env.obstacle_level):
        raise ValueError(
            f"entry level {entry.get('level')} != env level {env.obstacle_level}"
        )
    if entry.get("geometry_hash") != obstacle_geometry_hash(env.obstacle_level):
        raise ValueError(f"obstacle level {env.obstacle_level} geometry hash is stale")
    obs, info = env.reset(seed=int(entry["seed"]))
    if verify:
        resolved = {
            "agent_pos": env.agent_pos,
            "agent_angle": env.pusher_angle,
            "chain_joint_angle": env.agent.joint_angle,
            "object_pose": env.object_pose,
            "goal_pose": env.goal_pose,
        }
        for key, actual in resolved.items():
            actual_array = np.asarray(actual, dtype=np.float64)
            expected = np.asarray(entry[key], dtype=np.float64)
            if not np.allclose(actual_array, expected, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"seed {entry['seed']} no longer reproduces {key}: "
                    f"expected {expected.tolist()}, got {actual_array.tolist()}"
                )
    return obs, info
