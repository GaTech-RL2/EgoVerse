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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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

# Layout
WORLD_SIZE = 512
IMAGE_PANEL = 480
RIGHT_PANEL_W = 320
WINDOW_W = IMAGE_PANEL + RIGHT_PANEL_W
WINDOW_H = IMAGE_PANEL
TRACE_LEN = 30

# Palette
BG = (245, 245, 245)
PANEL_BG = (255, 255, 255)
TEXT = (25, 25, 25)
DIM = (110, 110, 110)
ACTION_MARK = (220, 40, 40)
TRACE_LINE = (220, 40, 40)
GOAL_MARK = (40, 160, 80)
AGENT_MARK = (200, 60, 60)
OBJECT_MARK = (60, 100, 200)


# ---------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------- #


@dataclass
class Episode:
    """Decoded episode buffers + metadata."""

    images: np.ndarray  # (T, H, W, 3) uint8
    states: np.ndarray  # (T, 5)
    actions: np.ndarray  # (T, 2)
    rewards: np.ndarray  # (T,)
    goal_pose: np.ndarray  # (3,) — constant across the episode
    metadata: dict
    path: Path

    @property
    def total(self) -> int:
        return int(self.images.shape[0])


@dataclass
class RenderResources:
    """pygame Surface + fonts; shared across interactive and headless paths."""

    surface: pygame.Surface
    font: pygame.font.Font
    font_small: pygame.font.Font


def _load_episode(ep_path: Path) -> Episode:
    if not ep_path.exists():
        raise FileNotFoundError(f"no such episode store: {ep_path}")
    store = zarr.open_group(str(ep_path), mode="r")
    attrs = dict(store.attrs)
    total = int(attrs["total_frames"])
    images_raw = store[IMAGE_KEY][:total]
    return Episode(
        images=np.stack(
            [simplejpeg.decode_jpeg(bytes(b), colorspace="RGB") for b in images_raw],
            axis=0,
        ),
        states=np.asarray(store[STATE_KEY][:total]),
        actions=np.asarray(store[ACTION_KEY][:total]),
        rewards=np.asarray(store[REWARD_KEY][:total]).reshape(-1),
        goal_pose=np.asarray(store[GOAL_KEY][0]),
        metadata=attrs,
        path=ep_path,
    )


# ---------------------------------------------------------------------- #
# Rendering
# ---------------------------------------------------------------------- #


def _world_to_image(xy: Sequence[float], panel: int = IMAGE_PANEL) -> tuple[int, int]:
    """Map (x, y) in 512-px world coords to pixel coords inside the image panel."""
    s = panel / WORLD_SIZE
    return int(round(xy[0] * s)), int(round(xy[1] * s))


def _make_render_resources(headless: bool = False) -> RenderResources:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    return RenderResources(
        surface=pygame.Surface((WINDOW_W, WINDOW_H)),
        font=pygame.font.Font(None, 22),
        font_small=pygame.font.Font(None, 18),
    )


def _blit_observation(surface: pygame.Surface, image_rgb: np.ndarray) -> None:
    """Upscale the (low-res) recorded observation into the left panel."""
    h, w = image_rgb.shape[:2]
    img = pygame.image.frombuffer(
        np.ascontiguousarray(image_rgb).tobytes(), (w, h), "RGB"
    )
    surface.blit(pygame.transform.smoothscale(img, (IMAGE_PANEL, IMAGE_PANEL)), (0, 0))


def _draw_overlay_markers(
    surface: pygame.Surface,
    *,
    state: np.ndarray,
    action: np.ndarray,
    goal_pose: np.ndarray,
    recent_actions: Sequence[np.ndarray],
) -> None:
    """Trace of recent actions + current-action marker + agent/object/goal pins."""
    if len(recent_actions) >= 2:
        pygame.draw.lines(
            surface, TRACE_LINE, False, [_world_to_image(a) for a in recent_actions], 2
        )
    # Action target — red X.
    ax, ay = _world_to_image(action)
    pygame.draw.line(surface, ACTION_MARK, (ax - 6, ay - 6), (ax + 6, ay + 6), 2)
    pygame.draw.line(surface, ACTION_MARK, (ax - 6, ay + 6), (ax + 6, ay - 6), 2)
    # Agent — red circle outline.
    pgx, pgy = _world_to_image(state[0:2])
    pygame.draw.circle(surface, AGENT_MARK, (pgx, pgy), 4, 1)
    # Object center — blue square outline.
    ox, oy = _world_to_image(state[2:4])
    pygame.draw.rect(surface, OBJECT_MARK, (ox - 4, oy - 4, 8, 8), 1)
    # Goal — green diamond outline.
    gx, gy = _world_to_image(goal_pose[0:2])
    pygame.draw.polygon(
        surface,
        GOAL_MARK,
        [(gx, gy - 6), (gx + 6, gy), (gx, gy + 6), (gx - 6, gy)],
        1,
    )


class _StatsPanel:
    """Small helper for laying out a column of label/value rows on the right
    panel. Internal y-cursor avoids the `nonlocal y` closure dance."""

    PANEL_X = IMAGE_PANEL

    def __init__(self, res: RenderResources):
        self.surface = res.surface
        self.font = res.font
        self.font_small = res.font_small
        self.y = 12

    def background(self) -> None:
        pygame.draw.rect(
            self.surface,
            PANEL_BG,
            pygame.Rect(self.PANEL_X, 0, RIGHT_PANEL_W, WINDOW_H),
        )
        self.y = 12

    def header(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.surface.blit(
                self.font_small.render(line, True, DIM), (self.PANEL_X + 12, self.y)
            )
            self.y += 18
        self.y += 6
        pygame.draw.line(
            self.surface,
            DIM,
            (self.PANEL_X + 12, self.y),
            (self.PANEL_X + RIGHT_PANEL_W - 12, self.y),
            1,
        )
        self.y += 10

    def section(self, title: str, rows: Sequence[tuple[str, str]]) -> None:
        self.surface.blit(
            self.font.render(title, True, TEXT), (self.PANEL_X + 12, self.y)
        )
        self.y += 22
        for label, value in rows:
            self.surface.blit(
                self.font_small.render(label, True, DIM), (self.PANEL_X + 18, self.y)
            )
            self.surface.blit(
                self.font_small.render(value, True, TEXT), (self.PANEL_X + 130, self.y)
            )
            self.y += 17
        self.y += 8

    def legend(self, items: Sequence[tuple[str, str]]) -> None:
        self.surface.blit(
            self.font.render("legend", True, TEXT), (self.PANEL_X + 12, self.y)
        )
        self.y += 22
        for label, desc in items:
            self.surface.blit(
                self.font_small.render(label, True, DIM), (self.PANEL_X + 18, self.y)
            )
            self.surface.blit(
                self.font_small.render(desc, True, TEXT), (self.PANEL_X + 110, self.y)
            )
            self.y += 17


def _env_args_from_metadata(metadata: dict) -> dict:
    try:
        return json.loads(metadata.get("task_description", "{}")).get("env_args", {})
    except json.JSONDecodeError:
        return {}


def _render_frame(
    episode: Episode,
    frame_idx: int,
    recent_actions: Sequence[np.ndarray],
    res: RenderResources,
) -> None:
    """Composite one frame onto `res.surface` in place."""
    state = episode.states[frame_idx]
    action = episode.actions[frame_idx]
    reward = float(episode.rewards[frame_idx])

    res.surface.fill(BG)
    _blit_observation(res.surface, episode.images[frame_idx])
    _draw_overlay_markers(
        res.surface,
        state=state,
        action=action,
        goal_pose=episode.goal_pose,
        recent_actions=recent_actions,
    )

    panel = _StatsPanel(res)
    panel.background()
    env_args = _env_args_from_metadata(episode.metadata)
    panel.header(
        [
            f"{episode.metadata.get('task_name', '?')}  "
            f"frame {frame_idx + 1:>4}/{episode.total}",
            f"fps={episode.metadata.get('fps', '?')}  "
            f"embodiment={episode.metadata.get('embodiment', '?')}",
            f"object={env_args.get('object_shape', '?')}  "
            f"pusher={env_args.get('pusher_shape', '?')}  "
            f"obs={env_args.get('obstacle_level', '?')}",
        ]
    )
    panel.section(
        "observations.state",
        [
            ("agent xy", f"{state[0]:7.1f}, {state[1]:7.1f}"),
            ("object xy", f"{state[2]:7.1f}, {state[3]:7.1f}"),
            ("object θ", f"{float(np.rad2deg(state[4])):+7.1f}°"),
        ],
    )
    panel.section(
        "actions  (target xy)",
        [
            ("target xy", f"{action[0]:7.1f}, {action[1]:7.1f}"),
            (
                "Δ to agent",
                f"{action[0] - state[0]:+7.1f}, {action[1] - state[1]:+7.1f}",
            ),
        ],
    )
    panel.section(
        "reward / goal",
        [
            ("reward (IoU)", f"{reward:6.3f}"),
            ("goal xy", f"{episode.goal_pose[0]:7.1f}, {episode.goal_pose[1]:7.1f}"),
            ("goal θ", f"{float(np.rad2deg(episode.goal_pose[2])):+7.1f}°"),
        ],
    )
    panel.legend(
        [
            ("red x", "action target"),
            ("red trail", "recent actions"),
            ("red ○", "agent"),
            ("blue □", "object"),
            ("green ◆", "goal"),
        ]
    )


# ---------------------------------------------------------------------- #
# Trace rolling buffer
# ---------------------------------------------------------------------- #


@dataclass
class TraceBuffer:
    """Fixed-length tail of recent action targets — used for the red trail."""

    maxlen: int = TRACE_LEN
    points: list[np.ndarray] = field(default_factory=list)

    def push(self, action: np.ndarray) -> None:
        self.points.append(action.copy())
        if len(self.points) > self.maxlen:
            self.points.pop(0)

    def rebuild_for(self, episode: Episode, up_to: int) -> None:
        start = max(0, up_to - self.maxlen + 1)
        self.points = [episode.actions[k].copy() for k in range(start, up_to + 1)]


# ---------------------------------------------------------------------- #
# Output paths: MP4 dump + interactive playback
# ---------------------------------------------------------------------- #


def _save_mp4(episode: Episode, out_path: Path, fps: int) -> None:
    res = _make_render_resources(headless=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WINDOW_W, WINDOW_H)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")

    trace = TraceBuffer()
    try:
        for i in range(episode.total):
            trace.push(episode.actions[i])
            _render_frame(episode, i, trace.points, res)
            # pygame surfarray gives (W, H, 3); cv2 wants (H, W, 3) BGR.
            frame = np.transpose(pygame.surfarray.array3d(res.surface), (1, 0, 2))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        pygame.quit()
    print(f"wrote {episode.total} frames -> {out_path}")


def _run_interactive(episode: Episode, fps: int) -> None:
    pygame.init()
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"PushShapes viz — {episode.path.name}")
    clock = pygame.time.Clock()
    res = RenderResources(
        surface=screen,
        font=pygame.font.Font(None, 22),
        font_small=pygame.font.Font(None, 18),
    )

    trace = TraceBuffer()
    trace.rebuild_for(episode, 0)
    idx = 0
    playing = True
    speed = 1.0
    last_advance_ms = pygame.time.get_ticks()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (pygame.K_q, pygame.K_x, pygame.K_ESCAPE):
                running = False
            elif event.key == pygame.K_SPACE:
                playing = not playing
            elif event.key == pygame.K_LEFT:
                idx = max(0, idx - 1)
                playing = False
                trace.rebuild_for(episode, idx)
            elif event.key == pygame.K_RIGHT:
                idx = min(episode.total - 1, idx + 1)
                playing = False
                trace.rebuild_for(episode, idx)
            elif event.key == pygame.K_UP:
                speed = min(8.0, speed * 1.5)
            elif event.key == pygame.K_DOWN:
                speed = max(0.1, speed / 1.5)
            elif event.key == pygame.K_r:
                idx = 0
                trace.rebuild_for(episode, idx)
                playing = True

        # Advance index based on playback speed.
        if playing and episode.total > 1:
            now = pygame.time.get_ticks()
            step_ms = max(1, int(1000.0 / (fps * speed)))
            if now - last_advance_ms >= step_ms:
                last_advance_ms = now
                if idx + 1 < episode.total:
                    idx += 1
                    trace.push(episode.actions[idx])
                else:
                    playing = False  # hold on the final frame

        _render_frame(episode, idx, trace.points, res)
        status = "PLAY" if playing else "PAUSE"
        footer = res.font_small.render(
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


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #


def _resolve_episode_path(args: argparse.Namespace) -> Path:
    if args.path is not None:
        return Path(args.path)
    if args.dataset is None:
        raise SystemExit("must provide either --dataset/--episode or --path")
    return Path(args.dataset) / f"episode_{args.episode:06d}.zarr"


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
        "--save", help="if set, render headlessly and save an MP4 to this path"
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
    episode = _load_episode(_resolve_episode_path(args))
    fps = args.fps or int(episode.metadata.get("fps", 30))
    if args.save:
        _save_mp4(episode, Path(args.save), fps=fps)
    else:
        _run_interactive(episode, fps=fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
