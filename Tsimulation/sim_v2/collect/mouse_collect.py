"""Mouse-driven demonstration collection for PushShapesEnv.

Each step's action is the mouse cursor's XY in world coordinates. The window
is 2x the 512x512 arena for sub-pixel action resolution. Episodes commit to
per-pusher/per-obstacle-level
subfolders under ``--output``::

    <output>/<pusher>/<obstacles>/episode_000000.zarr

Hotkeys (pygame window must have focus):
    SPACE   start / pause recording in the current episode
    R       abort the current episode (discard buffer) and reset
    A / D   rotate u_socket counterclockwise / clockwise
    Q / X   flush and exit

Usage::

    python -m Tsimulation.collect.mouse_collect \\
        --output data/pushshapes_demos \\
        --object T --pusher circle --obstacles 0 \\
        --num-episodes 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from Tsimulation.collect.balance import (
    N_BUCKETS,
    N_PUSHER_BUCKETS,
    BucketFn,
    BucketTracker,
    bucket_for,
    bucket_pusher_quad,
    count_existing_buckets,
)
from Tsimulation.collect.replay_init import (
    REPLAY_SOURCE_KEY as _REPLAY_SOURCE_KEY,
)
from Tsimulation.collect.replay_init import (
    collected_resume_keys as _collected_resume_keys,
)
from Tsimulation.collect.replay_init import (
    collected_seed_keys as _collected_seed_keys,
)
from Tsimulation.collect.replay_init import (
    init_pose_key as _init_pose_key,
)
from Tsimulation.collect.replay_init import (
    load_replay_inits as _load_replay_inits,
)
from Tsimulation.collect.replay_init import (
    load_replay_manifest,
)
from Tsimulation.collect.replay_init import (
    reset_to_init as _reset_to_init,
)
from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes import get_env, get_module
from Tsimulation.pushshapes.env import SIM_VERSION, PushShapesEnv
from Tsimulation.pushshapes.obstacles import all_levels

_BALANCE_MAX_REDRAWS = 200  # cap rejection-sampling per reset before bailing

_MODE_STANDARD = "standard"
_MODE_ON_TARGET = "on-target"
_DEFAULT_ONTARGET_TAG = "ontarget"

WORLD_SIZE = 512
WINDOW_SCALE = 2      # overridden by --window-scale
WINDOW_SIZE = WORLD_SIZE * WINDOW_SCALE
OVERLAY_COLOR = (20, 20, 20)
RECORDING_COLOR = (210, 60, 60)
PAUSED_COLOR = (180, 140, 0)
OVERLAY_HEIGHT = 92
OVERLAY_BG = (255, 255, 255, 200)
SOCKET_KEY_TURN_SPEED = math.radians(45.0)  # A counterclockwise, D clockwise


def _draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    *,
    saved: int,
    target: int,
    step: int,
    coverage: float,
    recording: bool,
    socket_latched: bool,
    socket_angle: float | None,
    output_path: Path,
    next_idx: int,
) -> None:
    """Translucent stats panel along the top of the window."""
    panel = pygame.Surface((WINDOW_SIZE, OVERLAY_HEIGHT), pygame.SRCALPHA)
    panel.fill(OVERLAY_BG)
    screen.blit(panel, (0, 0))

    # Status badge (REC / PAUSED) in the top-right.
    status, color = ("REC", RECORDING_COLOR) if recording else ("PAUSED", PAUSED_COLOR)
    badge = font.render(status, True, color)
    screen.blit(badge, (WINDOW_SIZE - badge.get_width() - 10, 6))

    socket_status = ""
    controls = "[SPACE] record  [R] abort  [Q] quit"
    if socket_angle is not None:
        socket_status = (
            f"   socket {'LATCHED' if socket_latched else 'open'}"
            f"   angle {math.degrees(socket_angle):6.1f} deg"
        )
        controls += "  [A/D] rotate"
    lines = [
        f"SIM V{SIM_VERSION} POCKET-BOTTOM-ONLY   saved {saved}/{target}  "
        f"next idx={next_idx:06d}",
        f"step {step}   coverage {coverage * 100:5.1f}%{socket_status}",
        f"out: {output_path}",
        controls,
    ]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, OVERLAY_COLOR), (8, 6 + i * 20))


def _episode_output_dir(root: Path, pusher: str, obstacles: int) -> Path:
    """``<root>/<pusher>/<obstacles>/`` — keeps demos partitioned by config."""
    return root / pusher / str(obstacles)


def _apply_on_target(
    env: PushShapesEnv,
    rng: np.random.Generator,
    min_angle_rad: float,
) -> None:
    """Override the object pose so it spawns at the goal's (x, y) with an
    intentionally-bad orientation: angle drawn uniformly in [-pi, pi] but
    rejected if the wrap-aware delta to the goal angle is below
    ``min_angle_rad``. Pusher and goal are left as the env sampled them."""
    init = env.get_episode_init()
    goal_x, goal_y, goal_theta = init["goal_pose"]
    # Rejection sample a "sufficiently wrong" angle.
    for _ in range(64):
        candidate = float(rng.uniform(-math.pi, math.pi))
        delta = abs(((candidate - goal_theta + math.pi) % (2 * math.pi)) - math.pi)
        if delta >= min_angle_rad:
            new_theta = candidate
            break
    else:
        # extremely unlikely; fall back to +180deg from goal
        new_theta = goal_theta + math.pi
    env.set_state(object_pose=(float(goal_x), float(goal_y), float(new_theta)))


def _reset_with_balance(
    env: PushShapesEnv,
    tracker: BucketTracker | None,
    bucket_fn: BucketFn,
    on_target_rng: np.random.Generator | None = None,
    min_angle_rad: float = 0.0,
) -> tuple[dict, dict, int]:
    """Call env.reset() (repeatedly if balanced) until an episode lands in a
    bucket that still has room. ``bucket_fn`` decides how to bucket the
    post-reset state. If ``on_target_rng`` is provided, the object is moved
    onto the goal with a bad orientation after each reset, BEFORE the bucket
    check (so balancing reflects the actual saved state)."""
    obs, info = env.reset()
    if on_target_rng is not None:
        _apply_on_target(env, on_target_rng, min_angle_rad)
        obs = env._get_obs()
    if tracker is None:
        b = bucket_fn(env.get_episode_init(), float(env.WORLD_SIZE))
        return obs, info, b
    for _ in range(_BALANCE_MAX_REDRAWS):
        b = bucket_fn(env.get_episode_init(), float(env.WORLD_SIZE))
        if tracker.has_room(b):
            return obs, info, b
        obs, info = env.reset()
        if on_target_rng is not None:
            _apply_on_target(env, on_target_rng, min_angle_rad)
            obs = env._get_obs()
    # All redraws landed in full buckets — accept whatever we have so the UI
    # doesn't wedge. Caller's bucket counter will overshoot at most by one.
    b = bucket_fn(env.get_episode_init(), float(env.WORLD_SIZE))
    return obs, info, b


class _ConfigError(ValueError):
    """Invalid collector arguments that should produce CLI exit code 2."""


@dataclass(frozen=True)
class _ModeSettings:
    bucket_fn: BucketFn
    num_buckets: int
    writer_tag: str | None
    on_target_rng: np.random.Generator | None
    min_angle_rad: float


@dataclass
class _EpisodeSequence:
    """Track optional replay or curated-seed collection progress."""

    replay_inits: list[dict[str, Any]] | None = None
    seed_list: list[int] | None = None
    index: int = 0

    @property
    def is_controlled(self) -> bool:
        return self.replay_inits is not None or self.seed_list is not None

    @property
    def total(self) -> int | None:
        if self.replay_inits is not None:
            return len(self.replay_inits)
        if self.seed_list is not None:
            return len(self.seed_list)
        return None

    @property
    def exhausted(self) -> bool:
        total = self.total
        return total is not None and self.index >= total

    def advance(self) -> None:
        if self.is_controlled:
            self.index += 1

    def add_source_marker(self, episode_init: dict[str, Any]) -> dict[str, Any]:
        """Attach replay provenance to the init saved by the writer."""
        if self.replay_inits is not None and not self.exhausted:
            episode_init[_REPLAY_SOURCE_KEY] = self.replay_inits[self.index][
                _REPLAY_SOURCE_KEY
            ]
        return episode_init


@dataclass
class _EpisodeInitializer:
    """Apply the active reset strategy without leaking it into the UI loop."""

    env: PushShapesEnv
    sequence: _EpisodeSequence
    tracker: BucketTracker | None
    bucket_fn: BucketFn
    on_target_rng: np.random.Generator | None
    min_angle_rad: float
    fixed_goal: tuple[float, float, float] | None

    def reset(self) -> tuple[dict, dict, int]:
        if self.sequence.seed_list is not None:
            obs, info = self.env.reset(
                seed=self.sequence.seed_list[self.sequence.index]
            )
            return obs, info, -1
        if self.sequence.replay_inits is not None:
            obs, info = _reset_to_init(
                self.env, self.sequence.replay_inits[self.sequence.index]
            )
            return obs, info, -1

        obs, info, bucket = _reset_with_balance(
            self.env,
            self.tracker,
            self.bucket_fn,
            self.on_target_rng,
            self.min_angle_rad,
        )
        if self.fixed_goal is not None:
            self.env.set_state(goal_pose=self.fixed_goal)
            obs = self.env._get_obs()
        return obs, info, bucket

    def episode_init(self) -> dict[str, Any]:
        return self.sequence.add_source_marker(self.env.get_episode_init())


def _load_replay_source(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if args.replay_init_from is not None:
        if args.balance:
            raise _ConfigError(
                "--balance is incompatible with --replay-init-from "
                "(replay uses the exact saved poses)"
            )
        if args.mode != _MODE_STANDARD:
            raise _ConfigError("--replay-init-from only supports --mode standard")
        source = Path(args.replay_init_from)
        if not source.is_dir():
            raise _ConfigError(f"--replay-init-from {source} is not a directory")
        replay_inits = _load_replay_inits(source)
        if not replay_inits:
            raise _ConfigError(
                f"no episodes with stored episode_init found under {source}"
            )
        print(f"replay-init: loaded {len(replay_inits)} inits from {source}")
    else:
        replay_inits = None

    if args.replay_init_file is None:
        return replay_inits
    if args.replay_init_from is not None:
        raise _ConfigError("--replay-init-file and --replay-init-from are exclusive")
    if args.balance:
        raise _ConfigError("--balance is incompatible with --replay-init-file")

    source = Path(args.replay_init_file)
    if not source.is_file():
        raise _ConfigError(f"--replay-init-file {source} not found")
    replay_inits = load_replay_manifest(source)
    print(f"replay-init-file: loaded {len(replay_inits)} inits from {source}")
    return replay_inits


def _load_seed_list(args: argparse.Namespace) -> list[int] | None:
    if args.seeds_file is None:
        return None
    if args.replay_init_from is not None:
        raise _ConfigError("--seeds-file and --replay-init-from are mutually exclusive")
    if args.balance:
        raise _ConfigError(
            "--balance is incompatible with --seeds-file "
            "(the curated seeds fix the start/goal placement)"
        )
    if args.mode != _MODE_STANDARD:
        raise _ConfigError("--seeds-file only supports --mode standard")

    source = Path(args.seeds_file)
    if not source.is_file():
        raise _ConfigError(f"--seeds-file {source} not found")
    levels = json.loads(source.read_text())["levels"]
    level_key = str(args.obstacles)
    if level_key not in levels:
        available_levels = sorted(int(key) for key in levels)
        raise _ConfigError(
            f"{source} has no seeds for obstacle level {args.obstacles} "
            f"(has {available_levels})"
        )
    seed_list = [int(entry["seed"]) for entry in levels[level_key]]
    print(
        f"seeds-file: {len(seed_list)} curated seeds for level "
        f"{args.obstacles} from {source}"
    )
    return seed_list


def _mode_settings(args: argparse.Namespace, env_args: dict[str, Any]) -> _ModeSettings:
    if args.mode == _MODE_ON_TARGET:
        env_args["min_angle_deg"] = args.min_angle_deg
        return _ModeSettings(
            bucket_fn=bucket_pusher_quad,
            num_buckets=N_PUSHER_BUCKETS,
            writer_tag=(args.tag if args.tag is not None else _DEFAULT_ONTARGET_TAG),
            on_target_rng=np.random.default_rng(args.seed),
            min_angle_rad=math.radians(args.min_angle_deg),
        )
    return _ModeSettings(
        bucket_fn=bucket_for,
        num_buckets=N_BUCKETS,
        writer_tag=args.tag,
        on_target_rng=None,
        min_angle_rad=0.0,
    )


def _parse_fixed_goal(
    raw_fixed_goal: str | None,
) -> tuple[float, float, float] | None:
    if raw_fixed_goal is None:
        return None
    try:
        values = [float(value) for value in raw_fixed_goal.split(",")]
    except ValueError as exc:
        raise _ConfigError(
            "--fixed-goal must be three comma-separated floats "
            f"X,Y,THETA, got {raw_fixed_goal!r}"
        ) from exc
    if len(values) != 3:
        raise _ConfigError(
            f"--fixed-goal must be exactly X,Y,THETA, got {len(values)} values"
        )
    return values[0], values[1], values[2]


def _filter_replay_inits(
    replay_inits: list[dict[str, Any]],
    *,
    obstacle_level: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Keep matching, not-yet-collected replay initial states."""
    matching = [
        episode_init
        for episode_init in replay_inits
        if int(episode_init.get("obstacle_level", -1)) == obstacle_level
    ]
    wrong_level = len(replay_inits) - len(matching)
    if wrong_level:
        print(
            f"replay-init: dropped {wrong_level} source inits whose "
            f"obstacle_level != {obstacle_level} (env's --obstacles)"
        )
    if not matching:
        return []

    saved_names, saved_pose_keys = _collected_resume_keys(output_dir)
    if not saved_names and not saved_pose_keys:
        return matching

    pending = [
        episode_init
        for episode_init in matching
        if episode_init[_REPLAY_SOURCE_KEY] not in saved_names
        and _init_pose_key(episode_init) not in saved_pose_keys
    ]
    print(
        f"replay-init resume: skipping {len(matching) - len(pending)} inits "
        f"already saved in {output_dir}; {len(pending)} remaining"
    )
    return pending


def run(args: argparse.Namespace) -> int:
    if not args.output and not args.output_dir:
        print("error: provide either --output or --output-dir", file=sys.stderr)
        return 2

    try:
        replay_inits = _load_replay_source(args)
        seed_list = _load_seed_list(args)
        fixed_goal = _parse_fixed_goal(args.fixed_goal)
    except _ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _episode_output_dir(Path(args.output), args.pusher, args.obstacles)
    )

    if replay_inits is not None:
        replay_inits = _filter_replay_inits(
            replay_inits,
            obstacle_level=int(args.obstacles),
            output_dir=output_dir,
        )
        if not replay_inits:
            print("nothing left to collect after filtering and resume — exiting.")
            return 0

    if seed_list is not None:
        saved_seed_keys = _collected_seed_keys(output_dir)
        level = int(args.obstacles)
        pending_seeds = [
            seed for seed in seed_list if (level, int(seed)) not in saved_seed_keys
        ]
        print(
            f"seeds-file resume: skipping {len(seed_list) - len(pending_seeds)} "
            f"seeds already saved in {output_dir}; {len(pending_seeds)} remaining"
        )
        seed_list = pending_seeds
        if not seed_list:
            print("nothing left to collect after seed resume — exiting.")
            return 0

    env_args: dict[str, Any] = {
        "object_shape": args.object,
        "pusher_shape": args.pusher,
        "obstacle_level": args.obstacles,
        "image_size": args.image_size,
        "fps": args.fps,
        "collector": "mouse",
        "mode": args.mode,
        "solid_pusher": True,
        "solid_contact_guard": True,
    }
    if args.pusher == "u_socket":
        env_args["action_mode"] = "mouse_xy_keyboard_theta"
        env_args["socket_inside_friction_only"] = True
    mode = _mode_settings(args, env_args)
    if fixed_goal is not None:
        env_args["fixed_goal"] = list(fixed_goal)
        if args.balance and args.mode == _MODE_STANDARD:
            print(
                "warning: --fixed-goal + --balance --mode standard collapses the "
                "goal_quadrant axis of the 16 bucket grid (all goals -> one "
                "quadrant); effective balancing reduces to 4 object-quadrant "
                "buckets.",
                file=sys.stderr,
            )

    pygame.init()
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption(
        f"PushShapes Sim V{SIM_VERSION} "
        f"[{args.object}/{args.pusher}/obs={args.obstacles}]"
    )
    global WINDOW_SCALE, WINDOW_SIZE
    WINDOW_SCALE = float(args.window_scale)
    WINDOW_SIZE = int(WORLD_SIZE * WINDOW_SCALE)
    print(f"[teleop] window {WINDOW_SIZE}x{WINDOW_SIZE} px "
          f"(scale {WINDOW_SCALE}; 1 screen px = {1.0/WINDOW_SCALE:.3f} world units)")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    env = get_env(args.sim_version)(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
        render_mode=None,  # we manage the window so we can overlay
        image_size=args.image_size,
        seed=args.seed,
    )
    # A higher collection threshold can provide replay margin for physics
    # paths (notably U-socket contact) that vary slightly across processes.
    env.SUCCESS_THRESHOLD = float(args.success_threshold)

    writer = ZarrDemoWriter(
        path=output_dir,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
        tag=mode.writer_tag,
    )

    tracker: BucketTracker | None = None
    if args.balance:
        per_bucket = args.per_bucket
        if per_bucket is None:
            per_bucket = max(1, -(-args.num_episodes // mode.num_buckets))  # ceil div
        initial_counts = count_existing_buckets(
            output_dir,
            float(env.WORLD_SIZE),
            bucket_fn=mode.bucket_fn,
            num_buckets=mode.num_buckets,
            entry_filter=lambda entry: writer.matches_episode_name(entry.name),
        )
        tracker = BucketTracker(
            target_per_bucket=per_bucket,
            initial_counts=initial_counts,
            num_buckets=mode.num_buckets,
        )
        target_total = tracker.goal_total
        existing = sum(initial_counts)
        tag_msg = f" [tag={mode.writer_tag}]" if mode.writer_tag else ""
        print(
            f"balance{tag_msg}: target {per_bucket}/bucket x "
            f"{mode.num_buckets} buckets "
            f"= {target_total} episodes (found {existing} pre-existing in {output_dir})"
        )
        if existing > 0:
            print(tracker.histogram())

    sequence = _EpisodeSequence(
        replay_inits=replay_inits,
        seed_list=seed_list,
    )
    if tracker is None:
        target_total = (
            sequence.total if sequence.total is not None else args.num_episodes
        )

    if sequence.exhausted:
        writer.close()
        env.close()
        pygame.display.quit()
        pygame.quit()
        print("nothing to collect: the selected replay/seed source is empty")
        return 0

    existing_matching = 0
    if tracker is None and not sequence.is_controlled:
        existing_matching = writer.existing_episode_count()
        if existing_matching > 0:
            print(
                f"resume: found {existing_matching} pre-existing matching episodes "
                f"in {output_dir}"
            )
        if existing_matching >= target_total:
            writer.close()
            env.close()
            pygame.display.quit()
            pygame.quit()
            print(
                f"nothing left to collect: found {existing_matching} matching "
                f"episodes in {output_dir} (target={target_total})"
            )
            return 0

    initializer = _EpisodeInitializer(
        env=env,
        sequence=sequence,
        tracker=tracker,
        bucket_fn=mode.bucket_fn,
        on_target_rng=mode.on_target_rng,
        min_angle_rad=mode.min_angle_rad,
        fixed_goal=fixed_goal,
    )
    obs, info, current_bucket = initializer.reset()
    coverage = info.get("coverage", 0.0)
    socket_angle_target = env.pusher_angle

    # Auto-start recording so a successful push is never lost because the
    # user forgot to press SPACE before moving the shape.
    writer.start_episode(init_state=initializer.episode_init())
    recording = True
    saved = existing_matching
    running = True

    while running:
        if tracker is not None and tracker.filled:
            break
        if sequence.exhausted:
            break
        if tracker is None and not sequence.is_controlled and saved >= target_total:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_x:
                    running = False
                elif event.key == pygame.K_SPACE:
                    recording = not recording
                elif event.key == pygame.K_s:
                    # FORCE-SAVE (added 2026-08-05, user request): commit the
                    # current episode even though coverage < SUCCESS_THRESHOLD.
                    # Used to record configurations that are IMPOSSIBLE for this
                    # pusher -- the episode is real teleop data, it just never
                    # reaches 0.95. Logged to forced_saves.jsonl so these are
                    # never mistaken for successes downstream.
                    if writer.steps_in_episode > 0:
                        _cov = float(coverage)
                        _idx = writer.commit_episode()
                        if _idx >= 0:
                            saved += 1
                            if tracker is not None:
                                tracker.increment(current_bucket)
                            sequence.advance()
                            try:
                                import json as _json
                                import os as _os
                                import time as _time
                                with open(_os.path.join(str(output_dir), "forced_saves.jsonl"), "a") as _fh:
                                    _fh.write(_json.dumps({
                                        "episode_idx": int(_idx),
                                        "coverage": _cov,
                                        "success_threshold": float(args.success_threshold),
                                        "pusher": args.pusher,
                                        "obstacle_level": int(args.obstacles),
                                        "reason": "impossible_or_substandard",
                                        "ts": _time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    }) + "\n")
                            except Exception as _e:
                                print(f"WARNING: could not log forced save: {_e}")
                            print(
                                f"FORCE-SAVED episode {_idx:06d}  coverage={_cov:.4f} "
                                f"(< {float(args.success_threshold):.2f})  ({saved}/{target_total})"
                            )
                            obs, info, current_bucket = initializer.reset()
                            coverage = info.get("coverage", 0.0)
                            socket_angle_target = env.pusher_angle
                            writer.start_episode(init_state=initializer.episode_init())
                            recording = True
                    else:
                        print("nothing recorded yet -- press SPACE to start")
                elif event.key == pygame.K_r:
                    writer.abort_episode()
                    # Replay mode: stay on the same init so the user can retry.
                    obs, info, current_bucket = initializer.reset()
                    coverage = info.get("coverage", 0.0)
                    socket_angle_target = env.pusher_angle
                    writer.start_episode(init_state=initializer.episode_init())
                    recording = True
        # Action = mouse pos in world coords. Window is scaled up from the
        # arena so we get sub-pixel resolution (0.5 world units at 2x scale).
        mx, my = pygame.mouse.get_pos()
        wx = mx / WINDOW_SCALE
        wy = my / WINDOW_SCALE
        action_values = [
            np.clip(wx, 0.0, float(WORLD_SIZE)),
            np.clip(wy, 0.0, float(WORLD_SIZE)),
        ]
        if args.pusher == "u_socket":
            keys = pygame.key.get_pressed()
            turn_step = SOCKET_KEY_TURN_SPEED * env.DT
            if keys[pygame.K_a]:
                socket_angle_target -= turn_step
            if keys[pygame.K_d]:
                socket_angle_target += turn_step
            socket_angle_target = (socket_angle_target + math.pi) % (
                2 * math.pi
            ) - math.pi
            action_values.append(socket_angle_target)
        action = np.asarray(action_values, dtype=np.float64)

        # Store pre-step obs so (state[t], action[t]) pairs are aligned:
        # state[t] is the state BEFORE action[t] is applied.
        pre_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        coverage = info.get("coverage", 0.0)

        if recording:
            pusher_obs_pose = pre_obs["agent_pos"]
            if args.pusher == "u_socket":
                pusher_obs_pose = np.concatenate(
                    [pre_obs["agent_pos"], pre_obs["agent_angle"]]
                )
            writer.add_step(
                image=pre_obs["image"],
                pusher_obs_pose=pusher_obs_pose,
                object_obs_pose=pre_obs["object_pose"],
                pusher_cmd_pose=action,
                action=action,
                reward=reward,
                goal_pose=pre_obs["goal_pose"],
            )

        world_surf = env.world_surface()
        if WINDOW_SCALE != 1:
            world_surf = pygame.transform.scale(world_surf, (WINDOW_SIZE, WINDOW_SIZE))
        screen.blit(world_surf, (0, 0))
        _draw_overlay(
            screen,
            font,
            saved=saved,
            target=target_total,
            step=env.step_count,
            coverage=coverage,
            recording=recording,
            socket_latched=bool(info.get("socket_latched", False)),
            socket_angle=(
                float(obs["agent_angle"][0]) if args.pusher == "u_socket" else None
            ),
            output_path=output_dir,
            next_idx=writer.next_episode_index,
        )
        pygame.display.flip()
        clock.tick(args.fps)

        if terminated or truncated:
            if writer.steps_in_episode > 0 and terminated:
                idx = writer.commit_episode()
                if idx >= 0:
                    saved += 1
                    if tracker is not None:
                        tracker.increment(current_bucket)
                    sequence.advance()
                    print(
                        f"auto-saved episode {idx:06d}  bucket={current_bucket:>2}  "
                        f"({saved}/{target_total})"
                    )
                    if tracker is not None:
                        print(tracker.histogram())
            else:
                writer.abort_episode()
            if tracker is not None:
                more_to_collect = not tracker.filled
            elif sequence.is_controlled:
                more_to_collect = not sequence.exhausted
            else:
                more_to_collect = saved < target_total
            if more_to_collect:
                obs, info, current_bucket = initializer.reset()
                coverage = info.get("coverage", 0.0)
                socket_angle_target = env.pusher_angle
                writer.start_episode(init_state=initializer.episode_init())
                recording = True

    writer.close()
    env.close()
    pygame.display.quit()
    pygame.quit()
    print(f"done. saved {saved} episodes to {output_dir}")
    if tracker is not None:
        print(tracker.histogram())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default=None,
        help="dataset root; demos are stored under <output>/<pusher>/<obstacles>/."
        " Ignored if --output-dir is set.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="write episodes directly into this exact directory, bypassing the"
        " <pusher>/<obstacles> subpath. Use this to keep custom folder names.",
    )
    p.add_argument("--object", default="T", choices=["T", "U", "Z"])
    p.add_argument(
        "--sim-version",
        default=None,
        help="sim version to COLLECT under (v1/v2/v3); default = current. "
             "Recorded into episode_init as sim_version.",
    )
    p.add_argument(
        "--pusher",
        default="circle",
        choices=["circle", "circle_small", "stick", "L", "u_socket"],
    )
    p.add_argument("--obstacles", type=int, default=0, choices=list(all_levels()))
    p.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="target total for this exact object/pusher/obstacle setup inside "
        "the output dir; existing matching episodes count toward the target",
    )
    p.add_argument(
        "--window-scale", type=float, default=2.0,
        help="teleop window = 512 * this (default 2.0 -> 1024px). Lower it if the\n"
             "window does not fit the screen. ACTION RESOLUTION SCALES WITH IT:\n"
             "one screen pixel = 1/scale world units, so 2.0 gives 0.5-unit\n"
             "sub-pixel control and 1.0 gives only 1.0-unit steps.",
    )
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--success-threshold",
        type=float,
        default=0.95,
        help="coverage required before an episode is auto-saved (default: 0.95)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--balance",
        action="store_true",
        help="balance saved episodes across the 16 (object-quadrant, goal-quadrant)"
        " buckets via rejection sampling at each reset",
    )
    p.add_argument(
        "--per-bucket",
        type=int,
        default=None,
        help="with --balance, episodes per bucket. Default is"
        " ceil(num_episodes/N) where N is 16 in standard mode and 4 in on-target mode",
    )
    p.add_argument(
        "--mode",
        choices=[_MODE_STANDARD, _MODE_ON_TARGET],
        default=_MODE_STANDARD,
        help="standard: random object + goal placements (4x4=16 buckets)."
        " on-target: object spawns AT goal xy with a bad orientation"
        " (4 pusher-quadrant buckets); useful for collecting recovery-rotation demos",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="alphanumeric tag inserted into saved filenames. on-target mode"
        f" defaults to '{_DEFAULT_ONTARGET_TAG}' so it stays distinct from"
        " standard episodes in the same folder. Tagged + untagged sequences"
        " are numbered independently",
    )
    p.add_argument(
        "--fixed-goal",
        default=None,
        help="X,Y,THETA (radians) — override the env's randomized goal_pose"
        " with this fixed pose after each reset. Useful for matching"
        " benchmarks that use a single goal configuration"
        " (e.g. Diffusion Policy PushT: 256,256,0.7853981633974483).",
    )
    p.add_argument(
        "--replay-init-file",
        default=None,
        help="JSON manifest from scripts/make_recollect_manifest.py — recollect"
        " exactly the listed episodes from their saved start poses. Use this"
        " when the episodes have no reset_seed (so --seeds-file cannot work)"
        " and you do not want to copy the source zarrs around.",
    )
    p.add_argument(
        "--seeds-file",
        default=None,
        help="hard_seeds.json from scripts/find_hard_seeds.py. Collects exactly"
        " that level's curated seeds, one episode each, in order, then stops."
        " Each seed is re-derived via env.reset(seed=...), so the start/goal"
        " match what the seed search scored. Disables --balance.",
    )
    p.add_argument(
        "--replay-init-from",
        default=None,
        help="path to an existing dataset directory. Iterates its .zarr"
        " episodes in filename order; for each one, calls env.reset() then"
        " env.set_state() to force the saved (agent_pos, object_pose,"
        " goal_pose). Disables --balance and on-target mode. Stops once"
        " every source init has been recorded once.",
    )
    p.add_argument(
        "--min-angle-deg",
        type=float,
        default=30.0,
        help="on-target mode: minimum |delta| (degrees) between object and"
        " goal orientation. Below this, the spawn is redrawn so the rotation"
        " recovery task is non-trivial",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
