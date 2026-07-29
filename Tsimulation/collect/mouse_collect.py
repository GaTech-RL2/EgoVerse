"""Mouse-driven demonstration collection for PushShapesEnv.

Each step's action is the mouse cursor's XY in world coordinates. The window
is 2x the 512x512 arena for sub-pixel action resolution. Episodes commit to
per-pusher/per-obstacle-level
subfolders under ``--output``::

    <output>/<pusher>/<obstacles>/episode_000000.zarr

Hotkeys (pygame window must have focus):
    SPACE   start / pause recording in the current episode
    S       commit the current episode and reset for the next
    R       abort the current episode (discard buffer) and reset
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
from pathlib import Path

import numpy as np
import pygame
import zarr

from Tsimulation.collect.balance import (
    N_BUCKETS,
    N_PUSHER_BUCKETS,
    BucketTracker,
    bucket_for,
    bucket_pusher_quad,
    count_existing_buckets,
)
from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv

_BALANCE_MAX_REDRAWS = 200  # cap rejection-sampling per reset before bailing

_MODE_STANDARD = "standard"
_MODE_ON_TARGET = "on-target"
_DEFAULT_ONTARGET_TAG = "ontarget"

WORLD_SIZE = 512
WINDOW_SCALE = 2
WINDOW_SIZE = WORLD_SIZE * WINDOW_SCALE
OVERLAY_COLOR = (20, 20, 20)
RECORDING_COLOR = (210, 60, 60)
PAUSED_COLOR = (180, 140, 0)
OVERLAY_HEIGHT = 92
OVERLAY_BG = (255, 255, 255, 200)


def _draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    *,
    saved: int,
    target: int,
    step: int,
    coverage: float,
    recording: bool,
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

    lines = [
        f"saved {saved}/{target}  next idx={next_idx:06d}",
        f"step {step}   coverage {coverage * 100:5.1f}%",
        f"out: {output_path}",
        "[SPACE] record  [S] save  [R] abort  [Q] quit",
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
    env.set_state(
        object_pose=(float(goal_x), float(goal_y), float(new_theta))
    )


_REPLAY_SOURCE_KEY = "_replay_source"


def _load_replay_inits(source_dir: Path) -> list[dict]:
    """Read ``episode_init`` from every ``*.zarr`` under ``source_dir`` (sorted
    by filename). Each returned init has ``_replay_source`` injected with the
    source episode's filename so resume can skip already-recorded ones."""
    inits: list[dict] = []
    for entry in sorted(source_dir.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        try:
            g = zarr.open_group(str(entry), mode="r")
            raw = g.attrs.get("episode_init")
            if raw is None:
                continue
            init = json.loads(raw)
            init[_REPLAY_SOURCE_KEY] = entry.name
            inits.append(init)
        except Exception:
            continue
    return inits


def _init_pose_key(init: dict) -> tuple:
    """Resume key for output episodes without a ``_replay_source`` marker.

    Includes ``obstacle_level`` so an obs0 output episode never spuriously
    matches a non-obs0 source episode that happens to share the same pose
    (the source dataset contains several such cross-level pose duplicates).
    6-decimal rounding absorbs JSON round-trip jitter without collapsing
    genuinely distinct sampled poses."""
    def _r(xs):
        return tuple(round(float(v), 6) for v in xs)
    return (
        _r(init["agent_pos"]),
        _r(init["object_pose"]),
        _r(init["goal_pose"]),
        int(init.get("obstacle_level", -1)),
    )


def _collected_resume_keys(output_dir: Path) -> tuple[set[str], set[tuple]]:
    """Scan ``output_dir`` for already-collected replay episodes.

    Returns ``(source_names, pose_keys)`` — an episode is considered already
    done if either set matches a source init. Pose-key fallback covers
    output episodes written before the ``_replay_source`` marker existed."""
    names: set[str] = set()
    pose_keys: set[tuple] = set()
    if not output_dir.exists():
        return names, pose_keys
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        try:
            g = zarr.open_group(str(entry), mode="r")
            raw = g.attrs.get("episode_init")
            if raw is None:
                continue
            init = json.loads(raw)
            saved = init.get(_REPLAY_SOURCE_KEY)
            if saved:
                names.add(saved)
            pose_keys.add(_init_pose_key(init))
        except Exception:
            continue
    return names, pose_keys


def _reset_to_init(env: PushShapesEnv, init: dict) -> tuple[dict, dict]:
    """Reset, then override the env's sampled poses with the saved init so the
    episode starts from the exact (agent, object, goal) configuration."""
    _obs, info = env.reset()
    env.set_state(
        agent_pos=tuple(init["agent_pos"]),
        object_pose=tuple(init["object_pose"]),
        goal_pose=tuple(init["goal_pose"]),
    )
    return env._get_obs(), info


def _reset_with_balance(
    env: PushShapesEnv,
    tracker: BucketTracker | None,
    bucket_fn,
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


def run(args: argparse.Namespace) -> int:
    if not args.output and not args.output_dir:
        print("error: provide either --output or --output-dir", file=sys.stderr)
        return 2

    replay_inits: list[dict] | None = None
    if args.replay_init_from is not None:
        if args.balance:
            print(
                "error: --balance is incompatible with --replay-init-from "
                "(replay uses the exact saved poses)",
                file=sys.stderr,
            )
            return 2
        if args.mode != _MODE_STANDARD:
            print(
                "error: --replay-init-from only supports --mode standard",
                file=sys.stderr,
            )
            return 2
        src = Path(args.replay_init_from)
        if not src.is_dir():
            print(f"error: --replay-init-from {src} is not a directory", file=sys.stderr)
            return 2
        replay_inits = _load_replay_inits(src)
        if not replay_inits:
            print(
                f"error: no episodes with stored episode_init found under {src}",
                file=sys.stderr,
            )
            return 2
        print(f"replay-init: loaded {len(replay_inits)} inits from {src}")

    pygame.init()
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption(
        f"PushShapes mouse collect [{args.object}/{args.pusher}/obs={args.obstacles}]"
    )
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    env = PushShapesEnv(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
        render_mode=None,  # we manage the window so we can overlay
        image_size=args.image_size,
        seed=args.seed,
    )

    env_args = {
        "object_shape": args.object,
        "pusher_shape": args.pusher,
        "obstacle_level": args.obstacles,
        "image_size": args.image_size,
        "fps": args.fps,
        "collector": "mouse",
        "mode": args.mode,
    }
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _episode_output_dir(Path(args.output), args.pusher, args.obstacles)
    )

    # Replay-init filter + resume.
    if replay_inits is not None:
        # 1. Drop source inits whose obstacle_level doesn't match the live env.
        # This is the safety net for the "leaked non-obs0 inits got replayed in
        # an obs0 env" failure mode — without it, iterating an unfiltered source
        # dataset against a fixed --obstacles will silently mis-record poses.
        before = len(replay_inits)
        replay_inits = [
            init
            for init in replay_inits
            if int(init.get("obstacle_level", -1)) == int(args.obstacles)
        ]
        wrong_level = before - len(replay_inits)
        if wrong_level:
            print(
                f"replay-init: dropped {wrong_level} source inits whose "
                f"obstacle_level != {args.obstacles} (env's --obstacles)"
            )
        if not replay_inits:
            print("nothing to collect after obstacle-level filter — exiting.")
            return 0

        # 2. Resume: skip inits already covered in output_dir, by source-name
        # marker OR by pose-key fallback (output episodes written before the
        # _replay_source marker existed are matched by pose alone).
        saved_names, saved_pose_keys = _collected_resume_keys(output_dir)
        if saved_names or saved_pose_keys:
            before = len(replay_inits)
            replay_inits = [
                init
                for init in replay_inits
                if init[_REPLAY_SOURCE_KEY] not in saved_names
                and _init_pose_key(init) not in saved_pose_keys
            ]
            skipped = before - len(replay_inits)
            print(
                f"replay-init resume: skipping {skipped} inits already saved "
                f"in {output_dir}; {len(replay_inits)} remaining"
            )
            if not replay_inits:
                print("nothing left to collect — exiting.")
                return 0

    # Mode-specific knobs: bucketing function, writer tag, on-target rng/angle.
    if args.mode == _MODE_ON_TARGET:
        bucket_fn = bucket_pusher_quad
        num_buckets = N_PUSHER_BUCKETS
        writer_tag = args.tag if args.tag is not None else _DEFAULT_ONTARGET_TAG
        on_target_rng = np.random.default_rng(args.seed)
        min_angle_rad = math.radians(args.min_angle_deg)
        env_args["min_angle_deg"] = args.min_angle_deg
    else:
        bucket_fn = bucket_for
        num_buckets = N_BUCKETS
        writer_tag = args.tag  # may be None
        on_target_rng = None
        min_angle_rad = 0.0

    # --fixed-goal: parse X,Y,THETA and apply to every reset so the goal pose
    # is constant across the dataset (matches single-goal benchmarks like
    # Diffusion Policy's PushT). Random agent/object positions are preserved.
    fixed_goal: tuple[float, float, float] | None = None
    if args.fixed_goal is not None:
        try:
            parts = [float(x) for x in args.fixed_goal.split(",")]
        except ValueError:
            print(
                f"error: --fixed-goal must be three comma-separated floats X,Y,THETA, got {args.fixed_goal!r}",
                file=sys.stderr,
            )
            return 2
        if len(parts) != 3:
            print(
                f"error: --fixed-goal must be exactly X,Y,THETA, got {len(parts)} values",
                file=sys.stderr,
            )
            return 2
        fixed_goal = (parts[0], parts[1], parts[2])
        env_args["fixed_goal"] = list(fixed_goal)
        if args.balance and args.mode == _MODE_STANDARD:
            print(
                "warning: --fixed-goal + --balance --mode standard collapses the "
                "goal_quadrant axis of the 16 bucket grid (all goals → one quadrant); "
                "effective balancing reduces to 4 object-quadrant buckets.",
                file=sys.stderr,
            )

    writer = ZarrDemoWriter(
        path=output_dir,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
        tag=writer_tag,
    )

    init_idx = 0  # only used in replay mode; advances on commit, stays on abort.

    def _do_reset() -> tuple[dict, dict, int]:
        """Reset the env using whichever mode is active.

        Returns ``(obs, info, current_bucket)``. ``current_bucket`` is -1 in
        replay mode (no bucketing) — the caller's tracker is also None then.

        If ``--fixed-goal`` is set, the env's randomized goal is overridden
        with the fixed pose AFTER the (possibly bucket-rejection-sampled)
        reset, so agent + object stays random but goal is constant."""
        if replay_inits is not None:
            obs, info = _reset_to_init(env, replay_inits[init_idx])
            return obs, info, -1
        obs, info, bucket = _reset_with_balance(
            env, tracker, bucket_fn, on_target_rng, min_angle_rad
        )
        if fixed_goal is not None:
            env.set_state(goal_pose=fixed_goal)
            obs = env._get_obs()
        return obs, info, bucket

    def _make_init_state() -> dict:
        """Init dict to hand to the writer at start_episode. In replay mode,
        stamps the source episode filename onto the saved init so resume can
        skip it on a later run."""
        init = env.get_episode_init()
        if replay_inits is not None and init_idx < len(replay_inits):
            init[_REPLAY_SOURCE_KEY] = replay_inits[init_idx][_REPLAY_SOURCE_KEY]
        return init

    tracker: BucketTracker | None = None
    if args.balance:
        per_bucket = args.per_bucket
        if per_bucket is None:
            per_bucket = max(1, -(-args.num_episodes // num_buckets))  # ceil div
        # Count only same-tag files so on-target and standard episodes have
        # independent bucket budgets in a mixed folder.
        filename_filter = writer_tag if writer_tag is not None else None
        initial_counts = count_existing_buckets(
            output_dir,
            float(env.WORLD_SIZE),
            bucket_fn=bucket_fn,
            num_buckets=num_buckets,
            filename_contains=filename_filter,
        )
        tracker = BucketTracker(
            target_per_bucket=per_bucket,
            initial_counts=initial_counts,
            num_buckets=num_buckets,
        )
        target_total = tracker.goal_total
        existing = sum(initial_counts)
        tag_msg = f" [tag={writer_tag}]" if writer_tag else ""
        print(
            f"balance{tag_msg}: target {per_bucket}/bucket x {num_buckets} buckets "
            f"= {target_total} episodes (found {existing} pre-existing in {output_dir})"
        )
        if existing > 0:
            print(tracker.histogram())
    elif replay_inits is not None:
        target_total = len(replay_inits)
    else:
        target_total = args.num_episodes

    obs, info, current_bucket = _do_reset()
    coverage = info.get("coverage", 0.0)

    # Auto-start recording so a successful push is never lost because the
    # user forgot to press SPACE before moving the shape.
    writer.start_episode(init_state=_make_init_state())
    recording = True
    saved = 0
    running = True

    while running:
        if tracker is not None and tracker.filled:
            break
        if replay_inits is not None and init_idx >= len(replay_inits):
            break
        if tracker is None and replay_inits is None and saved >= target_total:
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_x:
                    running = False
                elif event.key == pygame.K_SPACE:
                    recording = not recording
                elif event.key == pygame.K_r:
                    writer.abort_episode()
                    # Replay mode: stay on the same init so the user can retry.
                    obs, info, current_bucket = _do_reset()
                    coverage = info.get("coverage", 0.0)
                    writer.start_episode(init_state=_make_init_state())
                    recording = True
                elif event.key == pygame.K_s:
                    if writer.steps_in_episode > 0:
                        idx = writer.commit_episode()
                        if idx >= 0:
                            saved += 1
                            if tracker is not None:
                                tracker.increment(current_bucket)
                            if replay_inits is not None:
                                init_idx += 1
                            print(
                                f"saved episode {idx:06d}  bucket={current_bucket:>2}  "
                                f"({saved}/{target_total})"
                            )
                            if tracker is not None:
                                print(tracker.histogram())
                    if replay_inits is not None and init_idx >= len(replay_inits):
                        running = False
                        break
                    obs, info, current_bucket = _do_reset()
                    coverage = info.get("coverage", 0.0)
                    writer.start_episode(init_state=_make_init_state())
                    recording = True

        # Action = mouse pos in world coords. Window is scaled up from the
        # arena so we get sub-pixel resolution (0.5 world units at 2x scale).
        mx, my = pygame.mouse.get_pos()
        wx = mx / WINDOW_SCALE
        wy = my / WINDOW_SCALE
        action = np.array(
            [np.clip(wx, 0.0, float(WORLD_SIZE)), np.clip(wy, 0.0, float(WORLD_SIZE))],
            dtype=np.float64,
        )

        # Store pre-step obs so (state[t], action[t]) pairs are aligned:
        # state[t] is the state BEFORE action[t] is applied.
        pre_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        coverage = info.get("coverage", 0.0)

        if recording:
            writer.add_step(
                image=pre_obs["image"],
                pusher_obs_pose=pre_obs["agent_pos"],
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
            target=args.num_episodes,
            step=env.step_count,
            coverage=coverage,
            recording=recording,
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
                    if replay_inits is not None:
                        init_idx += 1
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
            elif replay_inits is not None:
                more_to_collect = init_idx < len(replay_inits)
            else:
                more_to_collect = saved < target_total
            if more_to_collect:
                obs, info, current_bucket = _do_reset()
                coverage = info.get("coverage", 0.0)
                writer.start_episode(init_state=_make_init_state())
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
        "--pusher",
        default="circle",
        choices=["circle", "circle_small", "stick"],
    )
    p.add_argument("--obstacles", type=int, default=0, choices=list(range(30)))
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--fps", type=int, default=30)
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
