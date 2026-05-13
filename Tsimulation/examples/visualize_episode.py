"""Visualize a recorded PushShapes episode (observations + actions).

Renders, frame-by-frame, the recorded observation image alongside a panel
showing the proprioceptive state, the action target, current reward, and a
short trace of the last few action targets overlaid on the scene.

Usage::

    # Interactive (window pops up)
    python -m Tsimulation.examples.visualize_episode \\
        --dataset data/pushshapes_demos --episode 0

    # Headless: dump an MP4 of the episode (no display needed)
    python -m Tsimulation.examples.visualize_episode \\
        --dataset data/pushshapes_demos --episode 0 --save out.mp4

Window hotkeys: SPACE play/pause, LEFT/RIGHT step, UP/DOWN speed,
R restart, Q quit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pygame
import simplejpeg
import zarr

from Tsimulation.collect.zarr_writer import (
    ACTION_KEY,
    GOAL_KEY,
    IMAGE_KEY,
    REWARD_KEY,
    STATE_KEY,
)

WORLD_SIZE = 512
IMAGE_PANEL = 480
RIGHT_PANEL_W = 320
WINDOW_W = IMAGE_PANEL + RIGHT_PANEL_W
WINDOW_H = IMAGE_PANEL
TRACE_LEN = 30

BG = (245, 245, 245)
PANEL_BG = (255, 255, 255)
TEXT = (25, 25, 25)
DIM = (110, 110, 110)
ACTION_MARK = (220, 40, 40)
TRACE_LINE = (220, 40, 40)
GOAL_MARK = (40, 160, 80)
AGENT_MARK = (200, 60, 60)
OBJECT_MARK = (60, 100, 200)


def _resolve_episode_path(args: argparse.Namespace) -> Path:
    if args.path is not None:
        return Path(args.path)
    if args.dataset is None:
        raise SystemExit("must provide either --dataset/--episode or --path")
    return Path(args.dataset) / f"episode_{args.episode:06d}.zarr"


def _load_episode(ep_path: Path) -> dict:
    if not ep_path.exists():
        raise FileNotFoundError(f"no such episode store: {ep_path}")
    store = zarr.open_group(str(ep_path), mode="r")
    attrs = dict(store.attrs)
    total = int(attrs["total_frames"])

    images_raw = store[IMAGE_KEY][:total]
    images = np.stack(
        [simplejpeg.decode_jpeg(bytes(b), colorspace="RGB") for b in images_raw], axis=0
    )
    return {
        "images": images,
        "states": np.asarray(store[STATE_KEY][:total]),
        "actions": np.asarray(store[ACTION_KEY][:total]),
        "rewards": np.asarray(store[REWARD_KEY][:total]).reshape(-1),
        "goal_pose": np.asarray(store[GOAL_KEY][0]),
        "metadata": attrs,
        "total": total,
        "path": ep_path,
    }


def _world_to_image(xy: np.ndarray, panel: int = IMAGE_PANEL) -> tuple[int, int]:
    """Map (x, y) in 512-px world coords to pixel coords inside the image panel."""
    s = panel / WORLD_SIZE
    return int(round(xy[0] * s)), int(round(xy[1] * s))


def _render_frame(
    *,
    image_rgb: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    reward: float,
    goal_pose: np.ndarray,
    frame_idx: int,
    total_frames: int,
    metadata: dict,
    recent_actions: list[np.ndarray],
    font: pygame.font.Font,
    font_small: pygame.font.Font,
    surface: pygame.Surface,
) -> pygame.Surface:
    surface.fill(BG)

    # --- Left: upscaled observation image, with overlays. ---
    h, w = image_rgb.shape[:2]
    img_surface = pygame.image.frombuffer(
        np.ascontiguousarray(image_rgb).tobytes(), (w, h), "RGB"
    )
    img_surface = pygame.transform.smoothscale(img_surface, (IMAGE_PANEL, IMAGE_PANEL))
    surface.blit(img_surface, (0, 0))

    # Trace of recent action targets.
    if len(recent_actions) >= 2:
        pts = [_world_to_image(a) for a in recent_actions]
        pygame.draw.lines(surface, TRACE_LINE, False, pts, 2)

    # Current action target marker (red X).
    ax, ay = _world_to_image(action)
    pygame.draw.line(surface, ACTION_MARK, (ax - 6, ay - 6), (ax + 6, ay + 6), 2)
    pygame.draw.line(surface, ACTION_MARK, (ax - 6, ay + 6), (ax + 6, ay - 6), 2)

    # Agent (pusher) marker — small circle.
    pgx, pgy = _world_to_image(state[0:2])
    pygame.draw.circle(surface, AGENT_MARK, (pgx, pgy), 4, 1)

    # Object center marker — small square.
    ox, oy = _world_to_image(state[2:4])
    pygame.draw.rect(surface, OBJECT_MARK, (ox - 4, oy - 4, 8, 8), 1)

    # Goal marker — green diamond.
    gx, gy = _world_to_image(goal_pose[0:2])
    pygame.draw.polygon(
        surface,
        GOAL_MARK,
        [(gx, gy - 6), (gx + 6, gy), (gx, gy + 6), (gx - 6, gy)],
        1,
    )

    # --- Right: text panel. ---
    panel_x = IMAGE_PANEL
    panel_rect = pygame.Rect(panel_x, 0, RIGHT_PANEL_W, WINDOW_H)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)

    task_name = metadata.get("task_name", "?")
    fps = metadata.get("fps", "?")
    try:
        env_args = json.loads(metadata.get("task_description", "{}")).get(
            "env_args", {}
        )
    except json.JSONDecodeError:
        env_args = {}
    header_lines = [
        f"{task_name}  frame {frame_idx + 1:>4}/{total_frames}",
        f"fps={fps}  embodiment={metadata.get('embodiment', '?')}",
        f"object={env_args.get('object_shape', '?')}  "
        f"pusher={env_args.get('pusher_shape', '?')}  "
        f"obs={env_args.get('obstacle_level', '?')}",
    ]
    y = 12
    for line in header_lines:
        surface.blit(font_small.render(line, True, DIM), (panel_x + 12, y))
        y += 18
    y += 6
    pygame.draw.line(
        surface, DIM, (panel_x + 12, y), (panel_x + RIGHT_PANEL_W - 12, y), 1
    )
    y += 10

    def _draw_section(title: str, rows: list[tuple[str, str]]) -> None:
        nonlocal y
        surface.blit(font.render(title, True, TEXT), (panel_x + 12, y))
        y += 22
        for label, value in rows:
            surface.blit(font_small.render(label, True, DIM), (panel_x + 18, y))
            surface.blit(font_small.render(value, True, TEXT), (panel_x + 130, y))
            y += 17
        y += 8

    obj_theta_deg = float(np.rad2deg(state[4]))
    goal_theta_deg = float(np.rad2deg(goal_pose[2]))
    _draw_section(
        "observations.state",
        [
            ("agent xy", f"{state[0]:7.1f}, {state[1]:7.1f}"),
            ("object xy", f"{state[2]:7.1f}, {state[3]:7.1f}"),
            ("object θ", f"{obj_theta_deg:+7.1f}°"),
        ],
    )
    _draw_section(
        "actions  (target xy)",
        [
            ("target xy", f"{action[0]:7.1f}, {action[1]:7.1f}"),
            (
                "Δ to agent",
                f"{action[0] - state[0]:+7.1f}, {action[1] - state[1]:+7.1f}",
            ),
        ],
    )
    _draw_section(
        "reward / goal",
        [
            ("reward (IoU)", f"{reward:6.3f}"),
            ("goal xy", f"{goal_pose[0]:7.1f}, {goal_pose[1]:7.1f}"),
            ("goal θ", f"{goal_theta_deg:+7.1f}°"),
        ],
    )

    legend = [
        ("red x", "action target"),
        ("red trail", "recent actions"),
        ("red ○", "agent"),
        ("blue □", "object"),
        ("green ◆", "goal"),
    ]
    surface.blit(font.render("legend", True, TEXT), (panel_x + 12, y))
    y += 22
    for label, desc in legend:
        surface.blit(font_small.render(label, True, DIM), (panel_x + 18, y))
        surface.blit(font_small.render(desc, True, TEXT), (panel_x + 110, y))
        y += 17

    return surface


def _save_mp4(episode: dict, out_path: Path, fps: int) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((WINDOW_W, WINDOW_H))
    font = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (WINDOW_W, WINDOW_H))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")

    recent: list[np.ndarray] = []
    try:
        for i in range(episode["total"]):
            recent.append(episode["actions"][i].copy())
            if len(recent) > TRACE_LEN:
                recent.pop(0)
            _render_frame(
                image_rgb=episode["images"][i],
                state=episode["states"][i],
                action=episode["actions"][i],
                reward=float(episode["rewards"][i]),
                goal_pose=episode["goal_pose"],
                frame_idx=i,
                total_frames=episode["total"],
                metadata=episode["metadata"],
                recent_actions=recent,
                font=font,
                font_small=font_small,
                surface=surface,
            )
            arr = pygame.surfarray.array3d(surface)
            # pygame: (W, H, 3) -> cv2 expects (H, W, 3) BGR
            frame = np.transpose(arr, (1, 0, 2))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()
        pygame.quit()
    print(f"wrote {episode['total']} frames -> {out_path}")


def _run_interactive(episode: dict, fps: int) -> None:
    pygame.init()
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"PushShapes viz — {episode['path'].name}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)

    playing = True
    speed = 1.0
    idx = 0
    last_advance_ms = pygame.time.get_ticks()

    recent: list[np.ndarray] = []

    def _rebuild_trace(up_to: int) -> None:
        recent.clear()
        start = max(0, up_to - TRACE_LEN + 1)
        for k in range(start, up_to + 1):
            recent.append(episode["actions"][k].copy())

    _rebuild_trace(idx)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_x, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 1)
                    playing = False
                    _rebuild_trace(idx)
                elif event.key == pygame.K_RIGHT:
                    idx = min(episode["total"] - 1, idx + 1)
                    playing = False
                    _rebuild_trace(idx)
                elif event.key == pygame.K_UP:
                    speed = min(8.0, speed * 1.5)
                elif event.key == pygame.K_DOWN:
                    speed = max(0.1, speed / 1.5)
                elif event.key == pygame.K_r:
                    idx = 0
                    _rebuild_trace(idx)
                    playing = True

        # Advance index based on playback speed.
        if playing and episode["total"] > 1:
            now = pygame.time.get_ticks()
            step_ms = max(1, int(1000.0 / (fps * speed)))
            if now - last_advance_ms >= step_ms:
                last_advance_ms = now
                if idx + 1 < episode["total"]:
                    idx += 1
                    recent.append(episode["actions"][idx].copy())
                    if len(recent) > TRACE_LEN:
                        recent.pop(0)
                else:
                    playing = False  # hold on last frame

        _render_frame(
            image_rgb=episode["images"][idx],
            state=episode["states"][idx],
            action=episode["actions"][idx],
            reward=float(episode["rewards"][idx]),
            goal_pose=episode["goal_pose"],
            frame_idx=idx,
            total_frames=episode["total"],
            metadata=episode["metadata"],
            recent_actions=recent,
            font=font,
            font_small=font_small,
            surface=screen,
        )
        # Footer with controls + speed.
        status = "PLAY" if playing else "PAUSE"
        footer = font_small.render(
            f"{status}  speed x{speed:.2f}   "
            "[SPACE]play  [←/→]step  [↑/↓]speed  [R]reset  [Q]quit",
            True,
            DIM,
        )
        screen.blit(footer, (8, WINDOW_H - 20))
        pygame.display.flip()
        clock.tick(60)

    pygame.display.quit()
    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_argument_group("episode source")
    src.add_argument("--dataset", help="directory containing episode_*.zarr stores")
    src.add_argument(
        "--episode", type=int, default=0, help="episode index inside --dataset"
    )
    src.add_argument(
        "--path",
        help="path to a single episode_*.zarr store (overrides --dataset/--episode)",
    )
    parser.add_argument(
        "--save",
        help="if set, render headlessly and save an MP4 to this path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="override playback fps (default: stored fps)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ep_path = _resolve_episode_path(args)
    episode = _load_episode(ep_path)
    fps = args.fps or int(episode["metadata"].get("fps", 30))

    if args.save:
        _save_mp4(episode, Path(args.save), fps=fps)
        return 0

    _run_interactive(episode, fps=fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
