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
import sys
from pathlib import Path

import numpy as np
import pygame

from Tsimulation.collect.zarr_writer import ZarrDemoWriter
from Tsimulation.pushshapes.env import PushShapesEnv

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
    }
    output_dir = _episode_output_dir(Path(args.output), args.pusher, args.obstacles)
    writer = ZarrDemoWriter(
        path=output_dir,
        env_args=env_args,
        image_size=args.image_size,
        fps=args.fps,
    )

    obs, info = env.reset()
    coverage = info.get("coverage", 0.0)

    # Auto-start recording so a successful push is never lost because the
    # user forgot to press SPACE before moving the shape.
    writer.start_episode(init_state=env.get_episode_init())
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
                    writer.start_episode(init_state=env.get_episode_init())
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
                    writer.start_episode(init_state=env.get_episode_init())
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
                    print(
                        f"auto-saved episode {idx:06d}  ({saved}/{args.num_episodes})"
                    )
            else:
                writer.abort_episode()
            if saved < args.num_episodes:
                obs, info = env.reset()
                coverage = info.get("coverage", 0.0)
                writer.start_episode(init_state=env.get_episode_init())
                recording = True

    writer.close()
    env.close()
    pygame.display.quit()
    pygame.quit()
    print(f"done. saved {saved} episodes to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        required=True,
        help="dataset root; demos are stored under <output>/<pusher>/<obstacles>/",
    )
    p.add_argument("--object", default="T", choices=["T", "U", "Z"])
    p.add_argument("--pusher", default="circle", choices=["circle", "stick"])
    p.add_argument("--obstacles", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
