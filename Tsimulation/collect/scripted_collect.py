"""Headless scripted demonstration collection for PushShapesEnv.

Uses a simple two-phase heuristic so we can bootstrap a demo set without a
mouse — handy while the physics is being tuned and human mouse demos are
finicky to record.

Heuristic per step:

1. Compute the unit vector from the object's center toward the goal center.
2. Pick an "approach point" a fixed offset behind the object along the
   anti-push direction.
3. If the pusher is far from that approach point, the action target is the
   approach point (move into contact).
4. Otherwise the action target is past the goal along the push direction
   (push the object toward the goal).

Episodes that don't reach a configurable coverage threshold are discarded
so we don't pollute the dataset with failed attempts. Obstacle-heavy
levels will see a lower success rate.

Usage::

    python -m Tsimulation.collect.scripted_collect \\
        --output data/pushshapes_scripted \\
        --object T --pusher circle --obstacles 0 \\
        --num-episodes 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from Tsimulation.collect.balance import (
    N_BUCKETS,
    BucketTracker,
    bucket_for,
    count_existing_buckets,
)
from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv

APPROACH_OFFSET = 60.0
PUSH_LOOKAHEAD = 80.0
CONTACT_RADIUS = 25.0

# Default per-frame drift tolerance (world units, L-inf over the 5-vec
# [pusher_x, pusher_y, obj_x, obj_y, obj_theta]) when replaying a recorded
# episode through the live env. Episodes that exceed this are rejected.
DEFAULT_REPLAY_DRIFT_THRESHOLD = 0.5


def _replay_validate(
    env: PushShapesEnv,
    ep_init: dict,
    actions: np.ndarray,
    recorded_states: np.ndarray,
    drift_threshold: float,
) -> tuple[bool, float, int]:
    """Replay ``actions`` through ``env`` from ``ep_init`` and reject if
    any post-step drift between env and ``recorded_states[t+1]`` exceeds
    ``drift_threshold``. Returns ``(ok, max_drift, frame)``.

    Delegates the step-loop to ``replay_zarr._replay_step_loop`` so there
    is a single source of truth for the replay logic shared with
    ``Tsimulation.examples.replay_zarr.replay_one`` and the dataloader's
    coverage filter.
    """
    from Tsimulation.examples.replay_zarr import _replay_step_loop
    env.reset(seed=ep_init.get("reset_seed"))
    env.set_state(
        agent_pos=tuple(ep_init["agent_pos"]),
        object_pose=tuple(ep_init["object_pose"]),
        goal_pose=tuple(ep_init["goal_pose"]),
    )
    metrics = _replay_step_loop(env, actions, recorded_states,
                                early_stop_drift=drift_threshold)
    if metrics["early_stop_frame"] is not None:
        return False, metrics["drift_max"], int(metrics["early_stop_frame"])
    return True, metrics["drift_max"], len(actions)


def scripted_action(
    *,
    agent_xy: np.ndarray,
    object_xy: np.ndarray,
    goal_xy: np.ndarray,
    world_size: float,
) -> np.ndarray:
    """Return a (2,) float64 action target in world coords."""
    push_vec = goal_xy - object_xy
    push_dist = float(np.linalg.norm(push_vec))
    if push_dist < 1e-3:
        return object_xy.astype(np.float64)
    push_dir = push_vec / push_dist
    approach_point = object_xy - push_dir * APPROACH_OFFSET
    if float(np.linalg.norm(agent_xy - approach_point)) > CONTACT_RADIUS:
        target = approach_point
    else:
        target = object_xy + push_dir * PUSH_LOOKAHEAD
    return np.clip(target, 0.0, world_size).astype(np.float64)


def run_episode(
    env: PushShapesEnv,
    writer: ZarrDemoWriter,
    *,
    max_steps: int,
    success_threshold: float,
    rng: np.random.Generator,
    jitter: float,
    seed: int | None,
    tracker: BucketTracker | None = None,
    replay_validate: bool = True,
    replay_drift_threshold: float = DEFAULT_REPLAY_DRIFT_THRESHOLD,
) -> tuple[bool, float, int, int, str]:
    """Roll out one scripted episode.

    Returns ``(committed, final_coverage, steps, bucket, reject_reason)``.
    ``reject_reason`` is ``""`` on commit; ``"bucket_full"`` /
    ``"low_coverage"`` / ``"replay_drift"`` when the episode is rejected.

    With ``replay_validate=True`` (default), the recorded actions are
    replayed through a fresh env from the captured ``episode_init`` after
    the rollout completes; if any frame's L-inf drift in the
    ``[pusher_xy, obj_xy, obj_theta]`` 5-vec exceeds
    ``replay_drift_threshold``, the episode is aborted instead of committed.
    """
    obs, info = env.reset(seed=seed)
    ep_init = env.get_episode_init()
    bucket = bucket_for(ep_init, float(env.WORLD_SIZE))
    if tracker is not None and not tracker.has_room(bucket):
        return False, 0.0, 0, bucket, "bucket_full"
    writer.start_episode(init_state=ep_init)
    coverage = float(info.get("coverage", 0.0))
    final_coverage = coverage
    steps = 0
    recorded_actions: list[np.ndarray] = []
    recorded_states: list[np.ndarray] = []
    for _ in range(max_steps):
        agent_xy = np.asarray(obs["agent_pos"], dtype=np.float64)
        object_xy = np.asarray(obs["object_pose"][:2], dtype=np.float64)
        goal_xy = np.asarray(obs["goal_pose"][:2], dtype=np.float64)
        action = scripted_action(
            agent_xy=agent_xy,
            object_xy=object_xy,
            goal_xy=goal_xy,
            world_size=float(env.WORLD_SIZE),
        )
        if jitter > 0.0:
            action = action + rng.normal(0.0, jitter, size=(2,))
            action = np.clip(action, 0.0, float(env.WORLD_SIZE))

        next_obs, reward, terminated, truncated, info = env.step(action)
        coverage = float(info.get("coverage", 0.0))

        writer.add_step(
            image=obs["image"],
            pusher_obs_pose=obs["agent_pos"],
            object_obs_pose=obs["object_pose"],
            pusher_cmd_pose=action,
            action=action,
            reward=float(reward),
            goal_pose=obs["goal_pose"],
        )
        recorded_actions.append(np.asarray(action, dtype=np.float64))
        recorded_states.append(
            np.concatenate([
                np.asarray(obs["agent_pos"], dtype=np.float64),
                np.asarray(obs["object_pose"], dtype=np.float64),
            ])
        )

        obs = next_obs
        steps += 1
        final_coverage = coverage
        if terminated or truncated:
            break

    if final_coverage < success_threshold or writer.steps_in_episode == 0:
        writer.abort_episode()
        return False, final_coverage, steps, bucket, "low_coverage"

    if replay_validate and len(recorded_actions) > 0:
        actions_arr = np.stack(recorded_actions, axis=0)
        states_arr = np.stack(recorded_states, axis=0)
        ok, drift, frame = _replay_validate(
            env, ep_init, actions_arr, states_arr, replay_drift_threshold,
        )
        if not ok:
            writer.abort_episode()
            return False, final_coverage, steps, bucket, f"replay_drift={drift:.3f}@f{frame}"

    idx = writer.commit_episode()
    if idx < 0:
        return False, final_coverage, steps, bucket, "commit_failed"
    return True, final_coverage, steps, bucket, ""


def run(args: argparse.Namespace) -> int:
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = PushShapesEnv(
        object_shape=args.object,
        pusher_shape=args.pusher,
        obstacle_level=args.obstacles,
        image_size=args.image_size,
        seed=args.seed,
    )
    # --max-steps stays on the CLI / collector loop only — it's not passed
    # to the env (which no longer truncates by step count).
    env_args = {
        "object_shape": args.object,
        "pusher_shape": args.pusher,
        "obstacle_level": args.obstacles,
        "image_size": args.image_size,
        "fps": args.fps,
        "max_episode_steps": args.max_steps,
        "collector": "scripted",
    }
    writer = ZarrDemoWriter(
        path=args.output,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
    )
    rng = np.random.default_rng(args.seed)

    tracker: BucketTracker | None = None
    if args.balance:
        per_bucket = args.per_bucket
        if per_bucket is None:
            per_bucket = max(1, -(-args.num_episodes // N_BUCKETS))  # ceil div
        initial_counts = count_existing_buckets(
            Path(args.output), float(env.WORLD_SIZE)
        )
        tracker = BucketTracker(
            target_per_bucket=per_bucket, initial_counts=initial_counts
        )
        target_total = tracker.goal_total
        existing = sum(initial_counts)
        print(
            f"balance: target {per_bucket}/bucket x {N_BUCKETS} buckets "
            f"= {target_total} episodes (found {existing} pre-existing in {args.output})"
        )
        if existing > 0:
            print(tracker.histogram())
    else:
        target_total = args.num_episodes

    saved = 0
    attempts = 0
    drift_rejects = 0
    max_attempts = (
        (target_total - (tracker.total if tracker is not None else 0))
        * args.max_tries
    )
    max_attempts = max(max_attempts, args.max_tries)
    while True:
        if tracker is not None and tracker.filled:
            break
        if tracker is None and saved >= target_total:
            break
        if attempts >= max_attempts:
            break
        attempts += 1
        ep_seed = (args.seed + attempts) if args.seed is not None else None
        committed, coverage, steps, bucket, reject_reason = run_episode(
            env,
            writer,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            rng=rng,
            jitter=args.jitter,
            seed=ep_seed,
            tracker=tracker,
            replay_validate=args.replay_validate,
            replay_drift_threshold=args.replay_drift_threshold,
        )
        if committed:
            saved += 1
            if tracker is not None:
                tracker.increment(bucket)
            print(
                f"saved ep {saved:03d}/{target_total}  bucket={bucket:>2}  "
                f"steps={steps:3d}  coverage={coverage:.3f}  (attempt {attempts})"
            )
            if tracker is not None and tracker.filled:
                break
        else:
            if reject_reason.startswith("replay_drift"):
                drift_rejects += 1
            if args.verbose or reject_reason.startswith("replay_drift"):
                print(
                    f"discard       attempt {attempts:3d}  bucket={bucket:>2}  "
                    f"steps={steps:3d}  coverage={coverage:.3f}  reason={reject_reason}"
                )

    writer.close()
    env.close()
    print(
        f"done. saved {saved}/{target_total} in {attempts} attempts "
        f"(replay_drift_rejected={drift_rejects}) -> {args.output}"
    )
    if tracker is not None:
        print(tracker.histogram())
    return 0 if saved == target_total else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="directory for episode_*.zarr stores"
    )
    parser.add_argument("--object", default="T", choices=["T", "U", "Z"])
    parser.add_argument(
        "--pusher",
        default="circle",
        choices=["circle", "circle_small", "stick"],
    )
    parser.add_argument("--obstacles", type=int, default=0, choices=list(range(30)))
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.6,
        help="minimum final coverage to count an episode as a success",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=4,
        help="give up after target_total * max_tries attempts (target_total is"
        " num_episodes, or per_bucket*16 when --balance is set)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="balance saved episodes across the 16 (object-quadrant, goal-quadrant)"
        " buckets via rejection sampling",
    )
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=None,
        help="with --balance, episodes per bucket (default: ceil(num_episodes/16))",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=4.0,
        help="zero-mean Gaussian noise (stddev, in world coords) added to actions",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="force SDL dummy driver so no display is required (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        dest="headless",
        action="store_false",
        help="render to a real display (requires a windowed environment)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--replay-validate",
        dest="replay_validate",
        action="store_true",
        default=True,
        help="replay recorded actions through the env after each rollout and"
        " reject episodes that drift from the recorded states (default: True)",
    )
    parser.add_argument(
        "--no-replay-validate",
        dest="replay_validate",
        action="store_false",
        help="skip replay-drift validation (faster but allows non-deterministic"
        " episodes to land on disk)",
    )
    parser.add_argument(
        "--replay-drift-threshold",
        type=float,
        default=DEFAULT_REPLAY_DRIFT_THRESHOLD,
        help="max per-frame L-inf drift (world units) in [pusher_xy, obj_xy,"
        f" obj_theta] tolerated by replay validation (default:"
        f" {DEFAULT_REPLAY_DRIFT_THRESHOLD})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
