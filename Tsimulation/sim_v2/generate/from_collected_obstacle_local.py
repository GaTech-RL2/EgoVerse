"""Generate validated local MimicGen variants for ChainGripper obstacles.

Ordinary object-frame MimicGen failed across independently curated obstacle
layouts because its warped transport path no longer followed the fixed wall.
This obstacle-conditioned variant instead samples small object/goal changes
around each successful manual route.  It rejects invalid initial states, keeps
only successful solid-physics rollouts, renders each accepted rollout again,
and optionally checks every stored state in a third deterministic replay.

The default 128-episode target includes the 16 manual sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr
from shapely.geometry import box

from Tsimulation.sim_v2.collect.obstacle_init import (
    DEFAULT_CRITERIA,
    _agent_has_object_line_of_sight,
    _arena_clearance,
    _crossing_direction,
    _gate_passage,
    _intersected_segments,
    _obstacle_clearance,
    _physical_obstacle_polygons,
    _pusher_arena_overflow,
    _segment_lines,
    _spawn_exclusion_clearance,
    _swept_collision_alphas,
    level_bank_sha256,
    level_entries,
    level_init_policy,
    load_manifest,
    reset_to_manifest_entry,
)
from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.sim_v2.generate.from_collected_obstacles import (
    LevelResult,
    ManualSource,
    _copy_manual_sources,
    _git_head,
    _read_episode_init,
    _write_rendered_candidate,
    completed_levels,
    episode_dirs,
    load_manual_sources,
    parse_levels,
)
from Tsimulation.sim_v2.generate.mimicgen import _frame_delta, retarget, wrap
from Tsimulation.sim_v2.pushshapes.env import SIM_VERSION, PushShapesEnv
from Tsimulation.sim_v2.pushshapes.obstacles import OBSTACLE_LEVELS

METHOD = "mimicgen_local_obstacle_v1"
SUCCESS_THRESHOLD = 0.95


def transformed_agent_start(
    source: ManualSource, target_object_pose: tuple[float, float, float]
) -> tuple[tuple[float, float], float]:
    transform, rotation = _frame_delta(source.demo.object_pose, target_object_pose)
    x, y = transform(*source.demo.agent_pos)
    return (float(x), float(y)), wrap(float(source.demo.agent_angle) + rotation)


def jittered_state(
    source: ManualSource,
    *,
    level: int,
    variant_index: int,
    retry_index: int,
    generation_seed: int,
    jitter_xy: float,
    jitter_angle_radians: float,
) -> dict:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [generation_seed, level, source.reset_seed, variant_index, retry_index]
        )
    )
    object_delta = np.asarray(
        [
            rng.normal(0.0, jitter_xy),
            rng.normal(0.0, jitter_xy),
            rng.normal(0.0, jitter_angle_radians),
        ]
    )
    goal_delta = np.asarray(
        [
            rng.normal(0.0, jitter_xy),
            rng.normal(0.0, jitter_xy),
            rng.normal(0.0, jitter_angle_radians),
        ]
    )
    object_pose = np.asarray(source.demo.object_pose, dtype=np.float64) + object_delta
    goal_pose = np.asarray(source.demo.goal_pose, dtype=np.float64) + goal_delta
    object_pose[2] = wrap(float(object_pose[2]))
    goal_pose[2] = wrap(float(goal_pose[2]))
    target_object = tuple(float(value) for value in object_pose)
    target_goal = tuple(float(value) for value in goal_pose)
    agent_pos, agent_angle = transformed_agent_start(source, target_object)
    return {
        "object_pose": target_object,
        "goal_pose": target_goal,
        "agent_pos": agent_pos,
        "agent_angle": float(agent_angle),
        "object_delta": object_delta.tolist(),
        "goal_delta": goal_delta.tolist(),
        "variant_index": int(variant_index),
        "retry_index": int(retry_index),
    }


def _apply_state(env: PushShapesEnv, source: ManualSource, state: dict) -> None:
    reset_to_manifest_entry(env, source.entry, verify=True)
    env.set_state(
        agent_pos=state["agent_pos"],
        agent_angle=state["agent_angle"],
        object_pose=state["object_pose"],
        goal_pose=state["goal_pose"],
    )


def valid_jittered_state(env: PushShapesEnv) -> bool:
    criteria = DEFAULT_CRITERIA
    tolerance = criteria.overlap_tolerance
    level = int(env.obstacle_level)
    start = tuple(float(value) for value in env.object_pose)
    goal = tuple(float(value) for value in env.goal_pose)
    agent = tuple(float(value) for value in env.agent_pos)
    if math.dist(start[:2], goal[:2]) < criteria.min_start_goal_distance:
        return False
    if math.dist(agent, start[:2]) > criteria.max_agent_object_distance:
        return False
    if not _agent_has_object_line_of_sight(level, agent, start):
        return False

    obstacles = _physical_obstacle_polygons(level)
    start_polygon = env._build_object_polygon(start[:2], start[2])
    goal_polygon = env._build_object_polygon(goal[:2], goal[2])
    if min(
        _arena_clearance(env, start_polygon),
        _arena_clearance(env, goal_polygon),
    ) < (criteria.min_arena_clearance - tolerance):
        return False
    if min(
        _obstacle_clearance(start_polygon, obstacles),
        _obstacle_clearance(goal_polygon, obstacles),
    ) < (criteria.min_obstacle_clearance - tolerance):
        return False

    policy = level_init_policy(level)
    if (
        policy.spawn_exclusions
        and min(
            _spawn_exclusion_clearance(start_polygon, policy.spawn_exclusions),
            _spawn_exclusion_clearance(goal_polygon, policy.spawn_exclusions),
        )
        < -tolerance
    ):
        return False
    if policy.box_spawn_exclusions:
        forbidden = [box(*item.bounds) for item in policy.box_spawn_exclusions]
        if any(
            polygon.intersects(region)
            for polygon in (start_polygon, goal_polygon)
            for region in forbidden
        ):
            return False

    pusher_shapes = list(env.agent.physics_shapes(env))
    if (
        env._shapes_static_penetration_depth(env._pusher_body, pusher_shapes)
        > tolerance
        or env._pusher_object_penetration_depth() > tolerance
        or env._object_static_penetration_depth() > tolerance
        or env._object_arena_metrics()[0] > tolerance
        or _pusher_arena_overflow(env) > tolerance
    ):
        return False

    if policy.gate_portal is not None:
        return _gate_passage(start, goal, policy.gate_portal) is not None
    hit_segments = _intersected_segments(start, goal, _segment_lines(level))
    if not any(
        _crossing_direction(start, goal, OBSTACLE_LEVELS[level][index]) != 0
        for index in hit_segments
    ):
        return False
    interior_hits = [
        alpha
        for alpha in _swept_collision_alphas(env, start, goal, criteria.sweep_samples)
        if criteria.endpoint_margin <= alpha <= 1.0 - criteria.endpoint_margin
    ]
    return len(interior_hits) >= criteria.min_blocked_samples


def _rollout(
    env: PushShapesEnv,
    source: ManualSource,
    state: dict,
    actions: np.ndarray,
    *,
    extra_steps: int,
) -> tuple[np.ndarray, float] | None:
    _apply_state(env, source, state)
    if not valid_jittered_state(env):
        return None
    played: list[np.ndarray] = []
    best = 0.0
    for action in actions:
        action = np.asarray(action, dtype=np.float64)
        played.append(action.copy())
        _obs, _reward, terminated, _truncated, info = env.step(action)
        best = max(best, float(info["coverage"]))
        if terminated:
            return np.stack(played), best
    hold = np.asarray(actions[-1], dtype=np.float64)
    for _ in range(extra_steps):
        played.append(hold.copy())
        _obs, _reward, terminated, _truncated, info = env.step(hold)
        best = max(best, float(info["coverage"]))
        if terminated:
            return np.stack(played), best
    return None


def _episode_init(
    env: PushShapesEnv,
    *,
    source: ManualSource,
    state: dict,
    canonical_manifest: dict,
    canonical_manifest_path: Path,
    canonical_manifest_sha: str,
) -> dict:
    level = int(source.demo.obstacle_level)
    source_provenance = source.episode_init["obstacle_init"]
    init = env.get_episode_init()
    init["control_gap_mode"] = "ideal"
    init["obstacle_init"] = {
        "schema_version": int(canonical_manifest["schema_version"]),
        "sampler_revision": str(canonical_manifest["sampler_revision"]),
        "manifest_path": str(canonical_manifest_path),
        "manifest_sha256": canonical_manifest_sha,
        "level_bank_sha256": level_bank_sha256(canonical_manifest, level),
        "entry_index": int(source_provenance["entry_index"]),
        "entry_count": len(level_entries(canonical_manifest, level)),
        "geometry_hash": str(source.entry["geometry_hash"]),
        "chain_joint_angle": float(source.entry["chain_joint_angle"]),
        "generated_state_override": True,
    }
    variant = int(state["variant_index"])
    retry = int(state["retry_index"])
    init["generation"] = {
        "method": METHOD,
        "source_episode": source.path.name,
        "source_reset_seed": source.reset_seed,
        "source_entry_index": int(source_provenance["entry_index"]),
        "source_manifest_path": str(canonical_manifest_path),
        "source_manifest_sha256": canonical_manifest_sha,
        "source_level_bank_sha256": str(source_provenance["level_bank_sha256"]),
        "variant_index": variant,
        "retry_index": retry,
        "object_delta": state["object_delta"],
        "goal_delta": state["goal_delta"],
        "target_object_pose": list(state["object_pose"]),
        "target_goal_pose": list(state["goal_pose"]),
        "transformed_agent_pos": list(state["agent_pos"]),
        "transformed_agent_angle": float(state["agent_angle"]),
        "pair_id": (
            f"L{level:02d}:source{source.reset_seed}:variant{variant}:retry{retry}"
        ),
    }
    return init


def validate_episode(
    path: Path,
    *,
    env: PushShapesEnv,
    source: ManualSource,
    state: dict,
    pair_id: str,
) -> None:
    group = zarr.open_group(str(path), mode="r")
    init = _read_episode_init(group, path)
    frames = int(group.attrs["total_frames"])
    actions = np.asarray(group["actions"][:frames], dtype=np.float64)
    states = np.asarray(group["observations.state"][:frames], dtype=np.float64)
    rewards = np.asarray(group["reward"][:frames], dtype=np.float64).reshape(-1)
    if actions.shape != (frames, 4) or states.shape != (frames, 6):
        raise ValueError(f"{path} has invalid generated shapes")
    if init.get("generation", {}).get("pair_id") != pair_id:
        raise ValueError(f"{path} lost generation provenance")
    _apply_state(env, source, state)
    terminated_at = None
    for index, action in enumerate(actions):
        _obs, _reward, terminated, _truncated, _info = env.step(action)
        actual = np.asarray(
            [*env.agent_pos, env.pusher_angle, *env.object_pose], dtype=np.float64
        )
        if not np.allclose(actual, states[index], rtol=0.0, atol=1e-8):
            raise ValueError(f"{path} replay drift at frame {index}")
        if terminated:
            terminated_at = index
            break
    if terminated_at != frames - 1 or float(rewards[-1]) < SUCCESS_THRESHOLD:
        raise ValueError(f"{path} does not replay to its recorded success")


def existing_variants(destination: Path) -> set[tuple[int, int]]:
    variants = set()
    for path in episode_dirs(destination):
        group = zarr.open_group(str(path), mode="r")
        init = _read_episode_init(group, path)
        generation = init.get("generation")
        if isinstance(generation, dict) and generation.get("method") == METHOD:
            variants.add(
                (
                    int(generation["source_reset_seed"]),
                    int(generation["variant_index"]),
                )
            )
    return variants


def generate_level(
    *,
    level: int,
    source_root: Path,
    output_root: Path,
    canonical_manifest: dict,
    canonical_manifest_path: Path,
    canonical_manifest_sha: str,
    manual_count: int,
    target_total: int,
    image_size: int,
    extra_steps: int,
    validate_replay: bool,
    generation_seed: int,
    jitter_xy: float,
    jitter_angle_radians: float,
    max_retries: int,
) -> LevelResult:
    started = time.time()
    source_dir = source_root / f"level_{level:02d}" / "chain_gripper" / "T"
    destination = output_root / f"level_{level:02d}" / "chain_gripper" / "T"
    sources = load_manual_sources(
        source_dir,
        level=level,
        manual_count=manual_count,
        canonical_manifest=canonical_manifest,
    )
    _copy_manual_sources(sources, destination)
    variants = existing_variants(destination)
    current_total = len(episode_dirs(destination))
    if current_total > target_total:
        raise RuntimeError(f"level {level}: output already exceeds {target_total}")
    if current_total == target_total:
        return LevelResult(
            level,
            manual_count,
            target_total - manual_count,
            target_total,
            0,
            0,
        )

    writer = ZarrDemoWriter(
        path=destination,
        env_args={
            "object_shape": "T",
            "pusher_shape": "chain_gripper",
            "obstacle_level": level,
            "control_gap_mode": "ideal",
            "generation_method": METHOD,
        },
        image_size=image_size,
    )
    headless = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=8,
    )
    headless._skip_obs_render = True
    rendered = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=image_size,
    )
    audited = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=8,
    )
    audited._skip_obs_render = True
    attempts = 0
    generated_now = 0
    generated_required = target_total - manual_count
    per_source, remainder = divmod(generated_required, len(sources))
    try:
        for source_index, source in enumerate(sources):
            count_for_source = per_source + int(source_index < remainder)
            for variant_index in range(count_for_source):
                if (source.reset_seed, variant_index) in variants:
                    continue
                accepted = False
                for retry_index in range(max_retries):
                    state = jittered_state(
                        source,
                        level=level,
                        variant_index=variant_index,
                        retry_index=retry_index,
                        generation_seed=generation_seed,
                        jitter_xy=jitter_xy,
                        jitter_angle_radians=jitter_angle_radians,
                    )
                    transformed = retarget(
                        source.demo, state["object_pose"], state["goal_pose"]
                    )
                    attempts += 1
                    rollout = _rollout(
                        headless,
                        source,
                        state,
                        transformed,
                        extra_steps=extra_steps,
                    )
                    if rollout is None:
                        continue
                    actions, _coverage = rollout
                    _apply_state(rendered, source, state)
                    if not valid_jittered_state(rendered):
                        continue
                    init = _episode_init(
                        rendered,
                        source=source,
                        state=state,
                        canonical_manifest=canonical_manifest,
                        canonical_manifest_path=canonical_manifest_path,
                        canonical_manifest_sha=canonical_manifest_sha,
                    )
                    pair_id = str(init["generation"]["pair_id"])
                    written = _write_rendered_candidate(
                        writer,
                        rendered,
                        actions,
                        episode_init=init,
                        level=level,
                    )
                    if written is None:
                        continue
                    path, coverage = written
                    if validate_replay:
                        validate_episode(
                            path,
                            env=audited,
                            source=source,
                            state=state,
                            pair_id=pair_id,
                        )
                    variants.add((source.reset_seed, variant_index))
                    current_total += 1
                    generated_now += 1
                    accepted = True
                    print(
                        f"L{level:02d} total={current_total}/{target_total} "
                        f"generated={generated_now} attempts={attempts} "
                        f"source_seed={source.reset_seed} variant={variant_index} "
                        f"retry={retry_index} frames={len(actions)} "
                        f"coverage={coverage:.6f}",
                        flush=True,
                    )
                    break
                if not accepted:
                    raise RuntimeError(
                        f"L{level:02d} source={source.reset_seed} "
                        f"variant={variant_index} failed {max_retries} retries"
                    )
    finally:
        writer.close()
        headless.close()
        rendered.close()
        audited.close()
    if current_total != target_total:
        raise RuntimeError(f"level {level}: produced {current_total}/{target_total}")
    return LevelResult(
        level,
        manual_count,
        target_total - manual_count,
        current_total,
        attempts,
        time.time() - started,
    )


def write_provenance(
    output_root: Path,
    *,
    levels: Iterable[int],
    source_root: Path,
    canonical_manifest_path: Path,
    canonical_manifest_sha: str,
    args: argparse.Namespace,
) -> None:
    level_list = list(levels)
    payload = {
        "method": METHOD,
        "sim_version": SIM_VERSION,
        "source_root": str(source_root),
        "levels": level_list,
        "manual_sources_per_level": args.manual_count,
        "target_total_per_level": args.target_total,
        "generated_per_level": args.target_total - args.manual_count,
        "generation_seed": args.generation_seed,
        "jitter_xy_std": args.jitter_xy,
        "jitter_angle_degrees_std": args.jitter_angle_deg,
        "max_retries_per_variant": args.max_retries,
        "image_size": args.image_size,
        "extra_hold_steps": args.extra_steps,
        "full_replay_validation": not args.no_replay_validation,
        "canonical_manifest_path": str(canonical_manifest_path),
        "canonical_manifest_sha256": canonical_manifest_sha,
        "source_git_head": _git_head(Path(__file__).resolve().parents[3]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_name = (
        f"generation_provenance_level_{level_list[0]:02d}.json"
        if len(level_list) == 1
        else "generation_provenance.json"
    )
    (output_root / provenance_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--levels", default="completed")
    parser.add_argument("--manual-count", type=int, default=16)
    parser.add_argument("--target-total", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--extra-steps", type=int, default=120)
    parser.add_argument("--generation-seed", type=int, default=260826)
    parser.add_argument("--jitter-xy", type=float, default=1.5)
    parser.add_argument("--jitter-angle-deg", type=float, default=0.75)
    parser.add_argument("--max-retries", type=int, default=32)
    parser.add_argument("--no-replay-validation", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.manual_count < 1 or args.target_total <= args.manual_count:
        parser.error("target total must exceed a positive manual count")
    if args.jitter_xy <= 0 or args.jitter_angle_deg <= 0 or args.max_retries < 1:
        parser.error("jitter scales and max retries must be positive")
    source_root = args.source_root.expanduser().resolve()
    output_root = args.out_root.expanduser().resolve()
    manifest_path = args.canonical_manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if args.levels == "completed":
        levels = completed_levels(source_root, args.manual_count)
    else:
        levels = parse_levels(args.levels)
    if not levels:
        parser.error("no completed levels found")
    write_provenance(
        output_root,
        levels=levels,
        source_root=source_root,
        canonical_manifest_path=manifest_path,
        canonical_manifest_sha=manifest_sha,
        args=args,
    )
    print(
        f"levels={levels} target={args.target_total} manual={args.manual_count} "
        f"generated={args.target_total - args.manual_count} output={output_root}",
        flush=True,
    )
    results = []
    for level in levels:
        results.append(
            generate_level(
                level=level,
                source_root=source_root,
                output_root=output_root,
                canonical_manifest=manifest,
                canonical_manifest_path=manifest_path,
                canonical_manifest_sha=manifest_sha,
                manual_count=args.manual_count,
                target_total=args.target_total,
                image_size=args.image_size,
                extra_steps=args.extra_steps,
                validate_replay=not args.no_replay_validation,
                generation_seed=args.generation_seed,
                jitter_xy=args.jitter_xy,
                jitter_angle_radians=math.radians(args.jitter_angle_deg),
                max_retries=args.max_retries,
            )
        )
    print("COMPLETE", flush=True)
    for result in results:
        print(
            f"L{result.level:02d}: total={result.total} "
            f"attempts={result.attempts} seconds={result.seconds:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
