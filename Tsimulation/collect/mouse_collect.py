"""Mouse-driven demonstration collection for PushShapesEnv.

Usage::

    python -m Tsimulation.collect.mouse_collect \\
        --output data/pushshapes_demos \\
        --object T \\
        --pusher circle \\
        --obstacles 0 \\
        --num-episodes 50 \\
        --image-size 96

Hotkeys (pygame window must have focus):
    SPACE  start / pause recording in the current episode
    S      commit the current episode and reset for the next
    R      abort the current episode (discard buffer) and reset
    Q / X  flush and exit

Each step's action is the mouse cursor's XY in world coordinates (the window
is 1:1 with the 512x512 arena).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv

WINDOW_SIZE = 512
OVERLAY_COLOR = (20, 20, 20)
RECORDING_COLOR = (210, 60, 60)
PAUSED_COLOR = (180, 140, 0)


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
    lines = [
        f"saved {saved}/{target}  next idx={next_idx:06d}",
        f"step {step}   coverage {coverage * 100:5.1f}%",
        f"out: {output_path}",
        "[SPACE] record  [S] save  [R] abort  [Q] quit",
    ]
    status = "REC" if recording else "PAUSED"
    color = RECORDING_COLOR if recording else PAUSED_COLOR

    panel = pygame.Surface((WINDOW_SIZE, 92), pygame.SRCALPHA)
    panel.fill((255, 255, 255, 200))
    screen.blit(panel, (0, 0))

    status_surface = font.render(status, True, color)
    screen.blit(status_surface, (WINDOW_SIZE - status_surface.get_width() - 10, 6))

    for i, line in enumerate(lines):
        text = font.render(line, True, OVERLAY_COLOR)
        screen.blit(text, (8, 6 + i * 20))


def run(args: argparse.Namespace) -> int:
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
        render_mode=None,  # we manage the window ourselves so we can overlay
        image_size=args.image_size,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )

    env_args = {
        "object_shape": args.object,
        "pusher_shape": args.pusher,
        "obstacle_level": args.obstacles,
        "image_size": args.image_size,
        "fps": args.fps,
        "max_episode_steps": args.max_steps,
    }
    writer = ZarrDemoWriter(
        path=args.output,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
    )

    obs, info = env.reset()
    coverage = info.get("coverage", 0.0)

    # Auto-start recording so a successful push is never lost because the
    # user forgot to press SPACE before moving the shape.
    writer.start_episode()
    recording = True
    saved = 0
    running = True
    while running and saved < args.num_episodes:
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
                    obs, info = env.reset()
                    coverage = info.get("coverage", 0.0)
                    writer.start_episode()
                    recording = True
                elif event.key == pygame.K_s:
                    if writer.steps_in_episode > 0:
                        idx = writer.commit_episode()
                        if idx >= 0:
                            saved += 1
                            print(
                                f"saved episode {idx:06d}  ({saved}/{args.num_episodes})"
                            )
                    obs, info = env.reset()
                    coverage = info.get("coverage", 0.0)
                    writer.start_episode()
                    recording = True

        # Action = mouse position in world coords (window is 1:1 with arena).
        mx, my = pygame.mouse.get_pos()
        action = np.array(
            [np.clip(mx, 0, WINDOW_SIZE - 1), np.clip(my, 0, WINDOW_SIZE - 1)],
            dtype=np.float32,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        coverage = info.get("coverage", 0.0)

        if recording:
            writer.add_step(
                image=obs["image"],
                pusher_obs_pose=obs["agent_pos"],
                object_obs_pose=obs["object_pose"],
                pusher_cmd_pose=action,
                action=action,
                reward=reward,
                goal_pose=obs["goal_pose"],
            )

        # Render world + overlay.
        world = env.world_surface()
        screen.blit(world, (0, 0))
        _draw_overlay(
            screen,
            font,
            saved=saved,
            target=args.num_episodes,
            step=env._step_count,
            coverage=coverage,
            recording=recording,
            output_path=Path(args.output),
            next_idx=writer.next_episode_index,
        )
        pygame.display.flip()
        clock.tick(args.fps)

        if terminated or truncated:
            if writer.steps_in_episode > 0 and terminated:
                idx = writer.commit_episode()
                if idx >= 0:
                    saved += 1
                    print(
                        f"auto-saved episode {idx:06d}  ({saved}/{args.num_episodes})"
                    )
            else:
                writer.abort_episode()
            if saved < args.num_episodes:
                obs, info = env.reset()
                coverage = info.get("coverage", 0.0)
                writer.start_episode()
                recording = True

    writer.close()
    env.close()
    pygame.display.quit()
    pygame.quit()
    print(f"done. saved {saved} episodes to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="directory for episode_*.zarr stores"
    )
    parser.add_argument("--object", default="T", choices=["T", "U", "Z"])
    parser.add_argument("--pusher", default="circle", choices=["circle", "stick"])
    parser.add_argument("--obstacles", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
