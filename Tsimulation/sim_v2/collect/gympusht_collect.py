"""Mouse-driven demonstration collection on the upstream huggingface/gym-pusht
``PushT-v0`` environment, saving to our zarr schema.

This script does NOT modify or reimplement gym-pusht — it drives the env via
the public ``gymnasium`` API (``gym.make`` + ``reset`` + ``step`` + ``render``).
gym-pusht requires pymunk <7, so run this from the isolated venv that has
that constraint::

    PYTHONPATH=. .venv-gympusht/bin/python -m Tsimulation.collect.gympusht_collect \\
        --output-dir data/pushshapes_demos/gympusht_circle \\
        --num-episodes 50

Hotkeys (pygame window must have focus):
    SPACE   start / pause recording in the current episode
    S       commit the current episode and reset for the next
    R       abort the current episode (discard buffer) and reset
    Q / X   flush and exit

Zarr schema written matches existing PushShapesEnv collections so downstream
training / replay code works unchanged:

    observations.state              (T, 5)  [agent_x, agent_y, obj_x, obj_y, obj_theta]
    observations.images.front_img_1 (T,)    JPEG-encoded 96x96x3
    observations.pusher_cmd_pose    (T, 2)  commanded mouse target
    actions                         (T, 2)  same as cmd (gym-pusht action == target)
    reward                          (T, 1)  env reward
    goal_pose                       (T, 3)  constant per-episode, broadcast
    annotations                     (0,)    empty (parity with mouse_collect)

env_args saved in ``task_description`` mark these episodes as ``collector="gympusht"``
so a downstream consumer can route them differently from PushShapesEnv data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gym_pusht  # noqa: F401  — side-effect registers gym_pusht/PushT-v0
import gymnasium as gym
import numpy as np
import pygame

from Tsimulation.collect.balance import (
    N_BUCKETS,
    BucketTracker,
    bucket_for,
    count_existing_buckets,
)
from Tsimulation.collect.zarr_writer import ZarrDemoWriter

ENV_ID = "gym_pusht/PushT-v0"
WORLD_SIZE = 512  # gym-pusht's action range and image-space arena

WINDOW_SCALE = 2  # display window = WORLD_SIZE * WINDOW_SCALE
WINDOW_SIZE = WORLD_SIZE * WINDOW_SCALE

OVERLAY_COLOR = (20, 20, 20)
RECORDING_COLOR = (210, 60, 60)
PAUSED_COLOR = (180, 140, 0)
OVERLAY_HEIGHT = 92
OVERLAY_BG = (255, 255, 255, 200)

_BALANCE_MAX_REDRAWS = 200


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
    panel = pygame.Surface((WINDOW_SIZE, OVERLAY_HEIGHT), pygame.SRCALPHA)
    panel.fill(OVERLAY_BG)
    screen.blit(panel, (0, 0))

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


def _state_from(obs: dict, info: dict) -> np.ndarray:
    """Pack agent_pos (from obs) + block_pose (from info) into our 5-vec
    `observations.state` layout `[agent_x, agent_y, obj_x, obj_y, obj_theta]`."""
    agent = np.asarray(obs["agent_pos"], dtype=np.float64).reshape(-1)
    block = np.asarray(info["block_pose"], dtype=np.float64).reshape(-1)
    return np.concatenate([agent[:2], block[:3]], axis=0)


def _goal_pose_from(info: dict) -> np.ndarray:
    return np.asarray(info["goal_pose"], dtype=np.float64).reshape(-1)


def _episode_init_from(obs: dict, info: dict) -> dict:
    """Stable JSON-serializable dict mirroring PushShapesEnv.get_episode_init().

    Useful for replay-init resume parity (matching new_circle_2's episode_init
    layout) and for documenting what env produced this episode."""
    agent = np.asarray(obs["agent_pos"], dtype=np.float64).tolist()
    block = np.asarray(info["block_pose"], dtype=np.float64).tolist()
    goal = np.asarray(info["goal_pose"], dtype=np.float64).tolist()
    return {
        "agent_pos": list(agent[:2]),
        "object_pose": list(block[:3]),
        "goal_pose": list(goal[:3]),
        "object_shape": "T",  # gym-pusht is T-only
        "pusher_shape": "circle",  # gym-pusht's pusher is a fixed circle
        "obstacle_level": 0,  # gym-pusht has no obstacles
        "obstacles": [],
        "reset_seed": int(info["reset_seed"]) if "reset_seed" in info else None,
    }


def _resize_render(rgb: np.ndarray, target: int) -> pygame.Surface:
    """Render frame -> pygame Surface scaled to ``target`` square."""
    # rgb is (H, W, 3) uint8. pygame wants (W, H, 3) for make_surface.
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    if surf.get_width() != target:
        surf = pygame.transform.scale(surf, (target, target))
    return surf


def _reset_with_balance(
    env: gym.Env,
    tracker: BucketTracker | None,
    seed: int,
) -> tuple[dict, dict, int]:
    """Reset (possibly multiple times if balanced) until landing in a bucket
    with room. Returns (obs, info, bucket_id)."""
    obs, info = env.reset(seed=seed)
    if tracker is None:
        return obs, info, -1
    for _ in range(_BALANCE_MAX_REDRAWS):
        ep_init = _episode_init_from(obs, info)
        b = bucket_for(ep_init, float(WORLD_SIZE))
        if tracker.has_room(b):
            return obs, info, b
        seed += 1
        obs, info = env.reset(seed=seed)
    ep_init = _episode_init_from(obs, info)
    return obs, info, bucket_for(ep_init, float(WORLD_SIZE))


def run(args: argparse.Namespace) -> int:
    if not args.output_dir:
        print("error: --output-dir is required", file=sys.stderr)
        return 2

    pygame.init()
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_caption(f"gym-pusht mouse collect [{ENV_ID}]")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    # gym-pusht: pixels_agent_pos gives 96x96 image + agent_pos in one obs dict;
    # info carries block_pose / goal_pose / coverage that our schema needs.
    env = gym.make(
        ENV_ID,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )

    env_args = {
        "env_id": ENV_ID,
        "object_shape": "T",
        "pusher_shape": "circle",
        "obstacle_level": 0,
        "image_size": args.image_size,
        "fps": args.fps,
        "collector": "gympusht",
        "mode": "standard",
    }
    output_dir = Path(args.output_dir)

    writer = ZarrDemoWriter(
        path=output_dir,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
        tag=args.tag,
    )

    tracker: BucketTracker | None = None
    if args.balance:
        per_bucket = args.per_bucket
        if per_bucket is None:
            per_bucket = max(1, -(-args.num_episodes // N_BUCKETS))  # ceil div
        initial_counts = count_existing_buckets(
            output_dir,
            float(WORLD_SIZE),
            bucket_fn=bucket_for,
            num_buckets=N_BUCKETS,
            entry_filter=lambda entry: writer.matches_episode_name(entry.name),
        )
        tracker = BucketTracker(
            target_per_bucket=per_bucket,
            initial_counts=initial_counts,
            num_buckets=N_BUCKETS,
        )
        target_total = tracker.goal_total
        print(
            f"balance: target {per_bucket}/bucket x {N_BUCKETS} buckets "
            f"= {target_total} episodes (found {sum(initial_counts)} pre-existing "
            f"in {output_dir})"
        )
        if sum(initial_counts):
            print(tracker.histogram())
    else:
        target_total = args.num_episodes

    existing_matching = 0
    if tracker is None:
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

    seed_counter = int(args.seed)
    obs, info, current_bucket = _reset_with_balance(env, tracker, seed_counter)
    seed_counter += 1
    coverage = info.get("coverage", 0.0)
    writer.start_episode(init_state=_episode_init_from(obs, info))
    recording = True
    saved = existing_matching
    running = True

    while running:
        if tracker is not None and tracker.filled:
            break
        if tracker is None and saved >= target_total:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_x):
                    running = False
                elif event.key == pygame.K_SPACE:
                    recording = not recording
                elif event.key == pygame.K_r:
                    writer.abort_episode()
                    obs, info, current_bucket = _reset_with_balance(
                        env, tracker, seed_counter
                    )
                    seed_counter += 1
                    coverage = info.get("coverage", 0.0)
                    writer.start_episode(init_state=_episode_init_from(obs, info))
                    recording = True
                elif event.key == pygame.K_s:
                    if writer.steps_in_episode > 0:
                        idx = writer.commit_episode()
                        if idx >= 0:
                            saved += 1
                            if tracker is not None:
                                tracker.increment(current_bucket)
                            print(
                                f"saved episode {idx:06d}  bucket={current_bucket:>2}  "
                                f"({saved}/{target_total})"
                            )
                            if tracker is not None:
                                print(tracker.histogram())
                    obs, info, current_bucket = _reset_with_balance(
                        env, tracker, seed_counter
                    )
                    seed_counter += 1
                    coverage = info.get("coverage", 0.0)
                    writer.start_episode(init_state=_episode_init_from(obs, info))
                    recording = True

        # Mouse XY -> action in env coords [0, WORLD_SIZE].
        mx, my = pygame.mouse.get_pos()
        action = np.array(
            [
                float(np.clip(mx / WINDOW_SCALE, 0.0, float(WORLD_SIZE))),
                float(np.clip(my / WINDOW_SCALE, 0.0, float(WORLD_SIZE))),
            ],
            dtype=np.float32,
        )

        # Pre-step image (96x96 from obs) and state, so (state[t], action[t])
        # are aligned: state is the state BEFORE action[t] is applied.
        pre_pixels = np.asarray(obs["pixels"], dtype=np.uint8)
        pre_state = _state_from(obs, info)
        pre_goal = _goal_pose_from(info)

        obs, reward, terminated, truncated, info = env.step(action)
        coverage = info.get("coverage", 0.0)

        if recording:
            writer.add_step(
                image=pre_pixels,
                pusher_obs_pose=pre_state[:2],
                object_obs_pose=pre_state[2:5],
                pusher_cmd_pose=action.astype(np.float64),
                action=action.astype(np.float64),
                reward=float(reward),
                goal_pose=pre_goal,
            )

        # Display: render the env at its native 680x680, scale to window.
        render_rgb = env.render()
        if render_rgb is not None:
            surf = _resize_render(render_rgb, WINDOW_SIZE)
            screen.blit(surf, (0, 0))
        _draw_overlay(
            screen,
            font,
            saved=saved,
            target=target_total,
            step=writer.steps_in_episode,
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
                    print(
                        f"auto-saved episode {idx:06d}  bucket={current_bucket:>2}  "
                        f"({saved}/{target_total})"
                    )
                    if tracker is not None:
                        print(tracker.histogram())
            else:
                writer.abort_episode()
            more_to_collect = (
                not tracker.filled if tracker is not None else saved < target_total
            )
            if more_to_collect:
                obs, info, current_bucket = _reset_with_balance(
                    env, tracker, seed_counter
                )
                seed_counter += 1
                coverage = info.get("coverage", 0.0)
                writer.start_episode(init_state=_episode_init_from(obs, info))
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
        "--output-dir",
        required=True,
        help="directory to write episode_*.zarr stores into",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="target total for this exact collector config inside the output "
        "dir; existing matching episodes count toward the target",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=96,
        help="must match gym-pusht's pixels obs (96 in upstream). Stored as-is.",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--balance",
        action="store_true",
        help="balance saved episodes across the 16 (object_quadrant, goal_quadrant)"
        " buckets via rejection-sampling at each reset",
    )
    p.add_argument(
        "--per-bucket",
        type=int,
        default=None,
        help="with --balance: episodes per bucket. Default ceil(num_episodes/16)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="alphanumeric tag inserted into saved filenames (e.g. 'gympusht')",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
