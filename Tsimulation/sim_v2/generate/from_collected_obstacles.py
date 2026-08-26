"""Expand curated ChainGripper obstacle demonstrations with MimicGen.

This launcher is deliberately separate from ``from_collected``.  The generic
launcher historically wrote its rendered replay with ``obstacle_level=0`` and
sampled unconstrained layouts, so it is not a valid obstacle-data generator.

For every requested level this module:

1. audits the canonical manual demonstrations and copies them into the output;
2. deterministically curates a larger target-layout bank with the same obstacle
   policies used by the reviewed collection manifest;
3. retargets manual action trajectories onto unique target layouts;
4. accepts only candidates that succeed in an exact obstacle-level replay;
5. renders the accepted trajectory in a second exact replay; and
6. replays the committed Zarr once more and checks every recorded state.

The default target of 128 is the *total* per level, including the 16 manual
source demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import zarr

from Tsimulation.sim_v2.collect.obstacle_init import (
    _agent_has_object_line_of_sight,
    _candidate_features,
    _candidate_group,
    _pusher_arena_overflow,
    curate_manifest,
    level_bank_sha256,
    level_entries,
    load_manifest,
    reset_to_manifest_entry,
    write_manifest,
)
from Tsimulation.sim_v2.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.sim_v2.generate.mimicgen import (
    SourceDemo,
    _frame_delta,
    retarget,
    wrap,
)
from Tsimulation.sim_v2.pushshapes.env import SIM_VERSION, PushShapesEnv

METHOD = "mimicgen_curated_obstacle_v1"
SUCCESS_THRESHOLD = 0.95


@dataclass(frozen=True)
class ManualSource:
    """One audited manual source and its canonical manifest entry."""

    path: Path
    demo: SourceDemo
    episode_init: dict
    entry: dict

    @property
    def reset_seed(self) -> int:
        return int(self.episode_init["reset_seed"])


@dataclass(frozen=True)
class LevelResult:
    level: int
    manual: int
    generated: int
    total: int
    attempts: int
    seconds: float


def episode_dirs(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("*.zarr") if item.is_dir())


def parse_levels(raw: str) -> list[int]:
    """Parse comma-separated levels and inclusive ranges."""
    levels: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError(f"descending level range: {token}")
            levels.extend(range(start, end + 1))
        else:
            levels.append(int(token))
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("levels must be a non-empty unique list")
    if any(level < 1 or level > 30 for level in levels):
        raise ValueError("levels must be in [1, 30]")
    return levels


def completed_levels(source_root: Path, manual_count: int) -> list[int]:
    complete: list[int] = []
    for level in range(1, 31):
        source_dir = source_root / f"level_{level:02d}" / "chain_gripper" / "T"
        if len(episode_dirs(source_dir)) >= manual_count:
            complete.append(level)
    return complete


def _read_episode_init(group: zarr.Group, path: Path) -> dict:
    try:
        value = group.attrs["episode_init"]
        return json.loads(str(value))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} has invalid episode_init") from error


def load_manual_sources(
    source_dir: Path,
    *,
    level: int,
    manual_count: int,
    canonical_manifest: dict,
) -> list[ManualSource]:
    """Load and strictly audit the first ``manual_count`` source episodes."""
    paths = episode_dirs(source_dir)
    if len(paths) < manual_count:
        raise ValueError(
            f"level {level}: found {len(paths)}/{manual_count} manual sources"
        )
    paths = paths[:manual_count]
    entries = level_entries(canonical_manifest, level)
    entries_by_seed = {int(entry["seed"]): entry for entry in entries}
    canonical_level_sha = level_bank_sha256(canonical_manifest, level)
    sources: list[ManualSource] = []
    seen_seeds: set[int] = set()
    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        init = _read_episode_init(group, path)
        frames = int(group.attrs["total_frames"])
        actions = np.asarray(group["actions"][:frames], dtype=np.float64)
        rewards = np.asarray(group["reward"][:frames], dtype=np.float64).reshape(-1)
        seed = int(init.get("reset_seed", -1))
        provenance = init.get("obstacle_init")
        if (
            init.get("pusher_shape") != "chain_gripper"
            or init.get("object_shape") != "T"
            or int(init.get("obstacle_level", -1)) != level
            or init.get("control_gap_mode") != "ideal"
        ):
            raise ValueError(f"{path} has the wrong collection signature")
        if not isinstance(provenance, dict):
            raise ValueError(f"{path} has no obstacle-bank provenance")
        if provenance.get("level_bank_sha256") != canonical_level_sha:
            raise ValueError(f"{path} does not match the canonical level bank")
        if seed not in entries_by_seed or seed in seen_seeds:
            raise ValueError(f"{path} has a missing or duplicate canonical seed")
        if actions.shape != (frames, 4) or frames <= 0:
            raise ValueError(f"{path} has invalid actions {actions.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
            raise ValueError(f"{path} contains non-finite data")
        if len(rewards) != frames or float(rewards[-1]) < SUCCESS_THRESHOLD:
            raise ValueError(f"{path} is not a successful demonstration")
        seen_seeds.add(seed)
        sources.append(
            ManualSource(
                path=path,
                demo=SourceDemo(
                    agent="chain_gripper",
                    actions=actions,
                    object_pose=tuple(init["object_pose"]),
                    goal_pose=tuple(init["goal_pose"]),
                    agent_pos=tuple(init["agent_pos"]),
                    agent_angle=float(init.get("agent_angle", 0.0)),
                    object_shape="T",
                    obstacle_level=level,
                ),
                episode_init=init,
                entry=entries_by_seed[seed],
            )
        )
    return sources


def rank_sources(
    target_entry: dict, sources: Sequence[ManualSource]
) -> list[ManualSource]:
    """Prefer matching route topology, then nearest curated-state features."""
    target_group = _candidate_group(target_entry)
    target_features = _candidate_features(target_entry)

    def key(source: ManualSource) -> tuple[int, float, int]:
        source_features = _candidate_features(source.entry)
        return (
            int(_candidate_group(source.entry) != target_group),
            float(np.linalg.norm(source_features - target_features)),
            source.reset_seed,
        )

    return sorted(sources, key=key)


def _copy_manual_sources(sources: Sequence[ManualSource], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = destination / source.path.name
        if target.exists():
            group = zarr.open_group(str(target), mode="r")
            copied_init = _read_episode_init(group, target)
            if int(copied_init.get("reset_seed", -1)) != source.reset_seed:
                raise RuntimeError(f"existing manual copy does not match: {target}")
            continue
        shutil.copytree(source.path, target)


def _target_manifest(
    path: Path, *, level: int, target_bank_size: int
) -> tuple[dict, str]:
    if path.exists():
        manifest = load_manifest(path)
        if (
            list(manifest["levels"]) != [str(level)]
            or int(manifest["entries_per_level"]) != target_bank_size
        ):
            raise ValueError(f"target manifest contract mismatch: {path}")
    else:
        manifest = curate_manifest(levels=[level], count=target_bank_size)
        write_manifest(manifest, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest, digest


def _rollout_headless(
    env: PushShapesEnv,
    entry: dict,
    actions: np.ndarray,
    *,
    agent_pos: tuple[float, float],
    agent_angle: float,
    extra_steps: int,
) -> tuple[np.ndarray, float] | None:
    reset_to_manifest_entry(env, entry, verify=True)
    env.set_state(agent_pos=agent_pos, agent_angle=agent_angle)
    if not _valid_transformed_agent_start(env):
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


def transformed_agent_start(
    source: ManualSource, target_entry: dict
) -> tuple[tuple[float, float], float]:
    """Map the source gripper start through the target object frame."""
    transform, rotation = _frame_delta(
        source.demo.object_pose, tuple(target_entry["object_pose"])
    )
    x, y = transform(*source.demo.agent_pos)
    angle = wrap(float(source.demo.agent_angle) + rotation)
    return (float(x), float(y)), float(angle)


def _valid_transformed_agent_start(env: PushShapesEnv) -> bool:
    """Apply the collection-time physical gates to a transformed gripper start."""
    tolerance = 1e-8
    if not _agent_has_object_line_of_sight(
        env.obstacle_level, env.agent_pos, env.object_pose
    ):
        return False
    if (
        np.linalg.norm(np.asarray(env.agent_pos) - np.asarray(env.object_pose[:2]))
        > 220
    ):
        return False
    pusher_shapes = list(env.agent.physics_shapes(env))
    return not (
        env._shapes_static_penetration_depth(env._pusher_body, pusher_shapes)
        > tolerance
        or env._pusher_object_penetration_depth() > tolerance
        or _pusher_arena_overflow(env) > tolerance
    )


def _generated_episode_init(
    env: PushShapesEnv,
    *,
    source: ManualSource,
    target_entry: dict,
    target_index: int,
    target_manifest: dict,
    target_manifest_path: Path,
    target_manifest_sha: str,
    canonical_manifest_path: Path,
    canonical_manifest_sha: str,
    transformed_agent_pos: tuple[float, float],
    transformed_agent_angle: float,
) -> dict:
    level = int(target_entry["level"])
    init = env.get_episode_init()
    init["control_gap_mode"] = "ideal"
    init["obstacle_init"] = {
        "schema_version": int(target_manifest["schema_version"]),
        "sampler_revision": str(target_manifest["sampler_revision"]),
        "manifest_path": str(target_manifest_path.resolve()),
        "manifest_sha256": target_manifest_sha,
        "level_bank_sha256": level_bank_sha256(target_manifest, level),
        "entry_index": int(target_index),
        "entry_count": len(level_entries(target_manifest, level)),
        "geometry_hash": str(target_entry["geometry_hash"]),
        "chain_joint_angle": float(target_entry["chain_joint_angle"]),
    }
    source_provenance = source.episode_init["obstacle_init"]
    init["generation"] = {
        "method": METHOD,
        "source_episode": source.path.name,
        "source_reset_seed": source.reset_seed,
        "source_entry_index": int(source_provenance["entry_index"]),
        "source_manifest_path": str(canonical_manifest_path.resolve()),
        "source_manifest_sha256": canonical_manifest_sha,
        "source_level_bank_sha256": str(source_provenance["level_bank_sha256"]),
        "target_reset_seed": int(target_entry["seed"]),
        "target_entry_index": int(target_index),
        "target_manifest_sha256": target_manifest_sha,
        "transformed_agent_pos": list(transformed_agent_pos),
        "transformed_agent_angle": float(transformed_agent_angle),
        "pair_id": (
            f"L{level:02d}:target{int(target_entry['seed'])}:source{source.reset_seed}"
        ),
    }
    return init


def _write_rendered_candidate(
    writer: ZarrDemoWriter,
    env: PushShapesEnv,
    actions: np.ndarray,
    *,
    episode_init: dict,
    level: int,
) -> tuple[Path, float] | None:
    writer.start_episode(init_state=episode_init)
    final_coverage = 0.0
    terminated = False
    for action in actions:
        action = np.asarray(action, dtype=np.float64)
        obs, reward, terminated, _truncated, info = env.step(action)
        px, py = env.agent_pos
        ox, oy, object_angle = env.object_pose
        writer.add_step(
            image=obs["image"],
            pusher_obs_pose=np.asarray([px, py, env.pusher_angle]),
            object_obs_pose=np.asarray([ox, oy, object_angle]),
            pusher_cmd_pose=np.asarray(action[:3]),
            action=action,
            reward=reward,
            goal_pose=np.asarray(env.goal_pose),
        )
        final_coverage = float(info["coverage"])
        if terminated:
            break
    if not terminated or final_coverage < SUCCESS_THRESHOLD:
        writer.abort_episode()
        return None
    episode_index = writer.commit_episode()
    path = writer.path / (
        f"episode_T_chain_gripper_obs{level}_{episode_index:06d}.zarr"
    )
    return path, final_coverage


def validate_generated_episode(
    path: Path,
    *,
    env: PushShapesEnv,
    target_entry: dict,
    expected_pair_id: str,
    transformed_agent_pos: tuple[float, float],
    transformed_agent_angle: float,
) -> None:
    """Replay a committed candidate and compare every stored simulator state."""
    group = zarr.open_group(str(path), mode="r")
    init = _read_episode_init(group, path)
    frames = int(group.attrs["total_frames"])
    actions = np.asarray(group["actions"][:frames], dtype=np.float64)
    states = np.asarray(group["observations.state"][:frames], dtype=np.float64)
    rewards = np.asarray(group["reward"][:frames], dtype=np.float64).reshape(-1)
    if actions.shape != (frames, 4) or states.shape != (frames, 6):
        raise ValueError(f"{path} has invalid generated shapes")
    if init.get("generation", {}).get("pair_id") != expected_pair_id:
        raise ValueError(f"{path} lost generation provenance")
    reset_to_manifest_entry(env, target_entry, verify=True)
    env.set_state(
        agent_pos=transformed_agent_pos,
        agent_angle=transformed_agent_angle,
    )
    terminated_at: int | None = None
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


def _existing_generated_target_seeds(destination: Path) -> set[int]:
    seeds: set[int] = set()
    for path in episode_dirs(destination):
        group = zarr.open_group(str(path), mode="r")
        init = _read_episode_init(group, path)
        generation = init.get("generation")
        if isinstance(generation, dict) and generation.get("method") == METHOD:
            seeds.add(int(generation["target_reset_seed"]))
    return seeds


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
    target_bank_size: int,
    image_size: int,
    extra_steps: int,
    validate_replay: bool,
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
    target_manifest_path = (
        output_root
        / "generation_manifests"
        / f"level_{level:02d}_target_bank_{target_bank_size}.json"
    )
    target_manifest, target_manifest_sha = _target_manifest(
        target_manifest_path,
        level=level,
        target_bank_size=target_bank_size,
    )
    target_entries = level_entries(target_manifest, level)
    source_seeds = {source.reset_seed for source in sources}
    used_target_seeds = _existing_generated_target_seeds(destination)
    current_total = len(episode_dirs(destination))
    if current_total > target_total:
        raise RuntimeError(
            f"level {level}: output has {current_total}, above target {target_total}"
        )
    if current_total == target_total:
        return LevelResult(
            level, manual_count, target_total - manual_count, current_total, 0, 0.0
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
    headless_env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=8,
    )
    headless_env._skip_obs_render = True
    render_env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=image_size,
    )
    audit_env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=8,
    )
    audit_env._skip_obs_render = True
    attempts = 0
    generated_this_run = 0
    try:
        for target_index, target_entry in enumerate(target_entries):
            target_seed = int(target_entry["seed"])
            if target_seed in source_seeds or target_seed in used_target_seeds:
                continue
            for source in rank_sources(target_entry, sources):
                attempts += 1
                (
                    transformed_agent_pos,
                    transformed_agent_angle,
                ) = transformed_agent_start(source, target_entry)
                transformed = retarget(
                    source.demo,
                    tuple(target_entry["object_pose"]),
                    tuple(target_entry["goal_pose"]),
                )
                rollout = _rollout_headless(
                    headless_env,
                    target_entry,
                    transformed,
                    agent_pos=transformed_agent_pos,
                    agent_angle=transformed_agent_angle,
                    extra_steps=extra_steps,
                )
                if rollout is None:
                    continue
                actions, _headless_coverage = rollout
                reset_to_manifest_entry(render_env, target_entry, verify=True)
                render_env.set_state(
                    agent_pos=transformed_agent_pos,
                    agent_angle=transformed_agent_angle,
                )
                if not _valid_transformed_agent_start(render_env):
                    continue
                episode_init = _generated_episode_init(
                    render_env,
                    source=source,
                    target_entry=target_entry,
                    target_index=target_index,
                    target_manifest=target_manifest,
                    target_manifest_path=target_manifest_path,
                    target_manifest_sha=target_manifest_sha,
                    canonical_manifest_path=canonical_manifest_path,
                    canonical_manifest_sha=canonical_manifest_sha,
                    transformed_agent_pos=transformed_agent_pos,
                    transformed_agent_angle=transformed_agent_angle,
                )
                pair_id = str(episode_init["generation"]["pair_id"])
                written = _write_rendered_candidate(
                    writer,
                    render_env,
                    actions,
                    episode_init=episode_init,
                    level=level,
                )
                if written is None:
                    continue
                path, coverage = written
                if validate_replay:
                    validate_generated_episode(
                        path,
                        env=audit_env,
                        target_entry=target_entry,
                        expected_pair_id=pair_id,
                        transformed_agent_pos=transformed_agent_pos,
                        transformed_agent_angle=transformed_agent_angle,
                    )
                used_target_seeds.add(target_seed)
                generated_this_run += 1
                current_total += 1
                print(
                    f"L{level:02d} total={current_total}/{target_total} "
                    f"generated={generated_this_run} attempts={attempts} "
                    f"target_seed={target_seed} source_seed={source.reset_seed} "
                    f"frames={len(actions)} coverage={coverage:.6f}",
                    flush=True,
                )
                break
            if current_total >= target_total:
                break
    finally:
        writer.close()
        headless_env.close()
        render_env.close()
        audit_env.close()

    if current_total != target_total:
        raise RuntimeError(
            f"level {level}: generated only {current_total}/{target_total} total "
            f"from {len(target_entries)} curated target layouts and {attempts} attempts"
        )
    return LevelResult(
        level=level,
        manual=manual_count,
        generated=target_total - manual_count,
        total=current_total,
        attempts=attempts,
        seconds=time.time() - started,
    )


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_run_provenance(
    output_root: Path,
    *,
    levels: Iterable[int],
    source_root: Path,
    canonical_manifest_path: Path,
    canonical_manifest_sha: str,
    target_total: int,
    manual_count: int,
    target_bank_size: int,
    image_size: int,
    extra_steps: int,
    validate_replay: bool,
) -> None:
    payload = {
        "method": METHOD,
        "sim_version": SIM_VERSION,
        "source_root": str(source_root.resolve()),
        "levels": list(levels),
        "target_total_per_level": target_total,
        "manual_sources_per_level": manual_count,
        "generated_per_level": target_total - manual_count,
        "target_bank_size": target_bank_size,
        "image_size": image_size,
        "extra_hold_steps": extra_steps,
        "full_replay_validation": validate_replay,
        "canonical_manifest_path": str(canonical_manifest_path.resolve()),
        "canonical_manifest_sha256": canonical_manifest_sha,
        "source_git_head": _git_head(Path(__file__).resolve().parents[3]),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "generation_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--levels",
        default="completed",
        help="'completed', a comma list, or ranges such as 1-7,10",
    )
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--manual-count", type=int, default=16)
    parser.add_argument("--target-total", type=int, default=128)
    parser.add_argument("--target-bank-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--extra-steps", type=int, default=120)
    parser.add_argument("--no-replay-validation", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.manual_count < 1:
        parser.error("--manual-count must be positive")
    if args.target_total <= args.manual_count:
        parser.error("--target-total must exceed --manual-count")
    if args.target_bank_size < args.target_total - args.manual_count:
        parser.error("--target-bank-size is too small for unique generated layouts")
    if args.image_size < 1 or args.extra_steps < 0:
        parser.error("--image-size must be positive and --extra-steps nonnegative")

    canonical_manifest_path = args.canonical_manifest.expanduser().resolve()
    canonical_manifest = load_manifest(canonical_manifest_path)
    canonical_manifest_sha = hashlib.sha256(
        canonical_manifest_path.read_bytes()
    ).hexdigest()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.out_root.expanduser().resolve()
    if args.levels == "completed":
        levels = completed_levels(source_root, args.manual_count)
        if not levels:
            parser.error("no completed manual levels found")
    else:
        try:
            levels = parse_levels(args.levels)
        except ValueError as error:
            parser.error(str(error))

    validate_replay = not args.no_replay_validation
    _write_run_provenance(
        output_root,
        levels=levels,
        source_root=source_root,
        canonical_manifest_path=canonical_manifest_path,
        canonical_manifest_sha=canonical_manifest_sha,
        target_total=args.target_total,
        manual_count=args.manual_count,
        target_bank_size=args.target_bank_size,
        image_size=args.image_size,
        extra_steps=args.extra_steps,
        validate_replay=validate_replay,
    )
    print(
        f"levels={levels} target_total={args.target_total} "
        f"manual={args.manual_count} generated={args.target_total - args.manual_count} "
        f"output={output_root}",
        flush=True,
    )
    results = []
    for level in levels:
        results.append(
            generate_level(
                level=level,
                source_root=source_root,
                output_root=output_root,
                canonical_manifest=canonical_manifest,
                canonical_manifest_path=canonical_manifest_path,
                canonical_manifest_sha=canonical_manifest_sha,
                manual_count=args.manual_count,
                target_total=args.target_total,
                target_bank_size=args.target_bank_size,
                image_size=args.image_size,
                extra_steps=args.extra_steps,
                validate_replay=validate_replay,
            )
        )
    print("COMPLETE", flush=True)
    for result in results:
        print(
            f"L{result.level:02d}: total={result.total} manual={result.manual} "
            f"generated={result.generated} attempts={result.attempts} "
            f"seconds={result.seconds:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
