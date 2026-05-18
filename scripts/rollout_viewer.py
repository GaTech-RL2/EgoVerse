"""Local live viewer for `eval_rollout.py` output — no network, no tunnel.

Watches an eval_rollout output dir for finished episodes (signaled by a
per-episode `.npz` sidecar landing next to its `.mp4`) and plays them back
in a pygame window styled like `Tsimulation/visualize_episode.py`: a left
image panel with the env render + an action-target overlay, and a right
HUD panel with per-step state, reward / coverage, and episode summary.

Auto-advances to the next episode as new ones appear, so you can watch a
rollout unfold live while the eval is still running. Episodes are played
in `episode_idx` order even if workers finish out of order.

Usage::

    python scripts/rollout_viewer.py --output-dir ./eval_runs/<ts>

Hotkeys: SPACE play/pause, ←/→ step frame, ↑/↓ speed, R restart episode,
N/P next/prev episode, A toggle auto-advance, Q quit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pygame

# Layout (mirrors Tsimulation/visualize_episode.py so the two viewers feel
# identical — same dimensions, fonts, palette).
WORLD_SIZE = 512
IMAGE_PANEL = 480
RIGHT_PANEL_W = 320
WINDOW_W = IMAGE_PANEL + RIGHT_PANEL_W
WINDOW_H = IMAGE_PANEL
TRACE_LEN = 30

# Palette — same hex values as visualize_episode.py.
BG = (245, 245, 245)
PANEL_BG = (255, 255, 255)
TEXT = (25, 25, 25)
DIM = (110, 110, 110)
ACTION_MARK = (220, 40, 40)
TRACE_LINE = (220, 40, 40)
OK_GREEN = (40, 160, 80)

# How often to rescan the output dir for newly-completed episodes (s).
# Cheap glob — sub-ms for hundreds of files — so 1 Hz is plenty.
_RESCAN_INTERVAL_S = 1.0


# ---------------------------------------------------------------------- #
# Episode loading
# ---------------------------------------------------------------------- #


@dataclass
class Episode:
    """One episode's pre-decoded frames + per-step state + metadata.

    Pre-decoding the MP4 keeps the playback loop trivial — backward step,
    restart, jump-to-frame are all O(1) array indexing. At 512×512 RGB
    that's ~235 MB peak for a 300-frame episode; we hold one episode at a
    time so this is fine for any reasonable horizon."""

    idx: int
    video_path: Path
    state_path: Path
    images: np.ndarray  # (T, H, W, 3) uint8 — env-rendered frames
    pusher_obs: np.ndarray  # (T, 2)
    object_obs: np.ndarray  # (T, 3) — x, y, theta
    actions: np.ndarray  # (T, 2)
    rewards: np.ndarray  # (T,)
    coverages: np.ndarray  # (T,)
    goal_pose: np.ndarray  # (3,)
    metadata: dict

    @property
    def total(self) -> int:
        return int(self.images.shape[0])


def _decode_mp4(path: Path) -> np.ndarray:
    """Read all frames as a (T, H, W, 3) uint8 RGB array."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open MP4: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"MP4 had no frames: {path}")
    return np.stack(frames, axis=0)


def _load_episode(npz_path: Path) -> Episode:
    """Pair the `.npz` sidecar with its sibling `.mp4` and decode both.

    If the MP4 has slightly more or fewer frames than the sidecar's state
    arrays (can happen if the episode was interrupted), truncate to the
    shorter so all per-frame indexing is safe."""
    mp4_path = npz_path.with_suffix(".mp4")
    if not mp4_path.exists():
        raise FileNotFoundError(f"no MP4 next to sidecar: {mp4_path}")
    images = _decode_mp4(mp4_path)
    npz = np.load(npz_path, allow_pickle=True)
    pusher = np.asarray(npz["pusher_obs"])
    obj = np.asarray(npz["object_obs"])
    actions = np.asarray(npz["actions"])
    rewards = np.asarray(npz["rewards"])
    coverages = np.asarray(npz["coverages"])
    goal_pose = np.asarray(npz["goal_pose"])
    md_blob = npz["metadata_json"] if "metadata_json" in npz.files else "{}"
    metadata = json.loads(str(md_blob)) if md_blob is not None else {}

    n = min(images.shape[0], pusher.shape[0])
    return Episode(
        idx=int(metadata.get("episode_idx", -1)),
        video_path=mp4_path,
        state_path=npz_path,
        images=images[:n],
        pusher_obs=pusher[:n],
        object_obs=obj[:n],
        actions=actions[:n],
        rewards=rewards[:n],
        coverages=coverages[:n],
        goal_pose=goal_pose,
        metadata=metadata,
    )


# ---------------------------------------------------------------------- #
# Episode discovery — sidecar presence is "ready to play"
# ---------------------------------------------------------------------- #


def _scan_ready_episodes(videos_dir: Path) -> list[Path]:
    """Sorted list of `.npz` sidecars whose sibling `.mp4` also exists.
    Sorting by filename = sorting by `episode_idx` (zero-padded names)."""
    out: list[Path] = []
    if not videos_dir.exists():
        return out
    for npz in sorted(videos_dir.glob("episode_*.npz")):
        if npz.with_suffix(".mp4").exists():
            out.append(npz)
    return out


class EpisodeQueue:
    """Tracks the sorted `.npz` paths that have appeared so far. The
    viewer identifies its current episode by *path*, not by list index,
    so an earlier-numbered episode finishing late doesn't yank the
    cursor back."""

    def __init__(self, videos_dir: Path):
        self.videos_dir = videos_dir
        self.paths: list[Path] = []
        self._last_scan = 0.0

    def refresh(self, now: float) -> None:
        if now - self._last_scan < _RESCAN_INTERVAL_S:
            return
        self._last_scan = now
        found = _scan_ready_episodes(self.videos_dir)
        if len(found) != len(self.paths):
            self.paths = found

    def index_of(self, path: Path) -> int:
        """Position of `path` in the (sorted) queue, or -1 if not present."""
        try:
            return self.paths.index(path)
        except ValueError:
            return -1


# ---------------------------------------------------------------------- #
# Rendering — same primitives as Tsimulation/visualize_episode.py
# ---------------------------------------------------------------------- #


def _world_to_image(xy: Sequence[float], panel: int = IMAGE_PANEL) -> tuple[int, int]:
    s = panel / WORLD_SIZE
    return int(round(xy[0] * s)), int(round(xy[1] * s))


def _blit_frame(surface: pygame.Surface, frame_rgb: np.ndarray) -> None:
    h, w = frame_rgb.shape[:2]
    surf = pygame.image.frombuffer(
        np.ascontiguousarray(frame_rgb).tobytes(), (w, h), "RGB"
    )
    surface.blit(pygame.transform.smoothscale(surf, (IMAGE_PANEL, IMAGE_PANEL)), (0, 0))


def _draw_action_overlay(
    surface: pygame.Surface,
    *,
    action: np.ndarray,
    recent_actions: Sequence[np.ndarray],
) -> None:
    """Draw only what the env render doesn't already show: the action
    target (red X) and the recent-action trail. Agent/object/goal are
    already drawn by the env in their canonical colors."""
    if len(recent_actions) >= 2:
        pygame.draw.lines(
            surface,
            TRACE_LINE,
            False,
            [_world_to_image(a) for a in recent_actions],
            2,
        )
    ax, ay = _world_to_image(action)
    pygame.draw.line(surface, ACTION_MARK, (ax - 6, ay - 6), (ax + 6, ay + 6), 2)
    pygame.draw.line(surface, ACTION_MARK, (ax - 6, ay + 6), (ax + 6, ay - 6), 2)


class _StatsPanel:
    """Right-side label/value column. Same layout as visualize_episode.py.

    Each method appends one block (header, section) using an internal
    y-cursor — the caller doesn't track vertical offsets."""

    PANEL_X = IMAGE_PANEL

    def __init__(self, surface: pygame.Surface, font, font_small) -> None:
        self.surface = surface
        self.font = font
        self.font_small = font_small
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
                self.font_small.render(line, True, DIM),
                (self.PANEL_X + 12, self.y),
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

    def section(
        self,
        title: str,
        rows: Sequence[tuple[str, str]],
        *,
        title_color: tuple[int, int, int] = TEXT,
    ) -> None:
        self.surface.blit(
            self.font.render(title, True, title_color), (self.PANEL_X + 12, self.y)
        )
        self.y += 22
        for label, value in rows:
            self.surface.blit(
                self.font_small.render(label, True, DIM),
                (self.PANEL_X + 18, self.y),
            )
            self.surface.blit(
                self.font_small.render(value, True, TEXT),
                (self.PANEL_X + 130, self.y),
            )
            self.y += 17
        self.y += 8


# ---------------------------------------------------------------------- #
# Trace buffer (red action trail)
# ---------------------------------------------------------------------- #


@dataclass
class TraceBuffer:
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
# Frame composition
# ---------------------------------------------------------------------- #


def _render_frame(
    screen: pygame.Surface,
    episode: Episode,
    frame_idx: int,
    recent_actions: Sequence[np.ndarray],
    font,
    font_small,
    *,
    queue_size: int,
    queue_pos: int,
    status: str,
    speed: float,
    auto_advance: bool,
) -> None:
    pusher = episode.pusher_obs[frame_idx]
    obj = episode.object_obs[frame_idx]
    action = episode.actions[frame_idx]
    reward = float(episode.rewards[frame_idx])
    coverage = float(episode.coverages[frame_idx])

    screen.fill(BG)
    _blit_frame(screen, episode.images[frame_idx])
    _draw_action_overlay(screen, action=action, recent_actions=recent_actions)

    md = episode.metadata
    scenario = md.get("scenario", {}) or {}
    panel = _StatsPanel(screen, font, font_small)
    panel.background()
    panel.header(
        [
            f"episode {episode.idx:>4}   " f"frame {frame_idx + 1:>4}/{episode.total}",
            f"queue {queue_pos + 1}/{queue_size}   "
            f"seed={md.get('seed', '?')}   worker={md.get('worker_id', '?')}",
            f"object={scenario.get('object_shape', '?')}  "
            f"pusher={scenario.get('pusher_shape', '?')}  "
            f"obs={scenario.get('obstacle_level', '?')}",
        ]
    )
    panel.section(
        "observations.state",
        [
            ("pusher xy", f"{pusher[0]:7.1f}, {pusher[1]:7.1f}"),
            ("object xy", f"{obj[0]:7.1f}, {obj[1]:7.1f}"),
            ("object θ", f"{float(np.rad2deg(obj[2])):+7.1f}°"),
        ],
    )
    panel.section(
        "action  (target xy)",
        [
            ("action xy", f"{action[0]:7.1f}, {action[1]:7.1f}"),
            (
                "Δ to pusher",
                f"{action[0] - pusher[0]:+7.1f}, {action[1] - pusher[1]:+7.1f}",
            ),
        ],
    )
    panel.section(
        "reward / goal",
        [
            ("reward", f"{reward:6.3f}"),
            ("coverage (IoU)", f"{coverage:6.3f}"),
            ("goal xy", f"{episode.goal_pose[0]:7.1f}, {episode.goal_pose[1]:7.1f}"),
            ("goal θ", f"{float(np.rad2deg(episode.goal_pose[2])):+7.1f}°"),
        ],
    )
    success = bool(md.get("success", False))
    panel.section(
        "episode summary",
        [
            ("max coverage", f"{md.get('max_coverage', 0.0):6.3f}"),
            ("episode return", f"{md.get('episode_return', 0.0):8.2f}"),
            ("success", "YES" if success else "no"),
        ],
        title_color=OK_GREEN if success else TEXT,
    )

    auto = "ON" if auto_advance else "OFF"
    footer = font_small.render(
        f"{status}  x{speed:.2f}  auto-adv:{auto}   "
        "[SPACE]play  [←/→]step  [↑/↓]speed  [R]reset  "
        "[N/P]ep  [A]auto  [Q]quit",
        True,
        DIM,
    )
    screen.blit(footer, (8, WINDOW_H - 20))


def _draw_idle(screen: pygame.Surface, font, message: str) -> None:
    """Idle-state screen shown before the first episode lands."""
    screen.fill(BG)
    pygame.draw.rect(
        screen, PANEL_BG, pygame.Rect(IMAGE_PANEL, 0, RIGHT_PANEL_W, WINDOW_H)
    )
    text = font.render(message, True, DIM)
    rect = text.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))
    screen.blit(text, rect)


# ---------------------------------------------------------------------- #
# Main loop
# ---------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        required=True,
        help="eval_runs/<ts>/ directory (the same path passed to "
        "eval_rollout.py's --output-dir)",
    )
    p.add_argument("--fps", type=int, default=30, help="playback fps")
    p.add_argument(
        "--no-auto-advance",
        action="store_true",
        help="don't auto-advance to the next episode when one ends",
    )
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    videos_dir = out / "videos"
    queue = EpisodeQueue(videos_dir)

    pygame.init()
    pygame.display.set_caption(f"rollout viewer — {out.name}")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    font = pygame.font.Font(None, 22)
    font_small = pygame.font.Font(None, 18)
    clock = pygame.time.Clock()

    auto_advance = not args.no_auto_advance
    episode: Episode | None = None
    current_path: Path | None = None
    frame_idx = 0
    trace = TraceBuffer()
    playing = True
    speed = 1.0
    last_advance_ms = pygame.time.get_ticks()
    running = True

    def _try_load(path: Path) -> bool:
        """Replace the current episode with one loaded from `path`. Keeps
        the previous episode on failure so a corrupt file doesn't kill
        the viewer."""
        nonlocal episode, current_path, frame_idx, last_advance_ms
        try:
            ep = _load_episode(path)
        except Exception:
            sys.stderr.write(f"viewer: failed to load {path.name}:\n")
            sys.stderr.write(traceback.format_exc())
            return False
        episode = ep
        current_path = path
        frame_idx = 0
        trace.rebuild_for(episode, 0)
        last_advance_ms = pygame.time.get_ticks()
        return True

    try:
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                    continue
                if ev.type != pygame.KEYDOWN:
                    continue
                if ev.key in (pygame.K_q, pygame.K_x, pygame.K_ESCAPE):
                    running = False
                    continue
                # All other keys only do something once an episode is loaded.
                if episode is None:
                    continue
                if ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_LEFT:
                    frame_idx = max(0, frame_idx - 1)
                    trace.rebuild_for(episode, frame_idx)
                    playing = False
                elif ev.key == pygame.K_RIGHT:
                    frame_idx = min(episode.total - 1, frame_idx + 1)
                    trace.rebuild_for(episode, frame_idx)
                    playing = False
                elif ev.key == pygame.K_UP:
                    speed = min(8.0, speed * 1.5)
                elif ev.key == pygame.K_DOWN:
                    speed = max(0.1, speed / 1.5)
                elif ev.key == pygame.K_r:
                    frame_idx = 0
                    trace.rebuild_for(episode, 0)
                    playing = True
                elif ev.key == pygame.K_n:
                    pos = queue.index_of(current_path) if current_path else -1
                    if pos >= 0 and pos + 1 < len(queue.paths):
                        if _try_load(queue.paths[pos + 1]):
                            playing = True
                elif ev.key == pygame.K_p:
                    pos = queue.index_of(current_path) if current_path else -1
                    if pos > 0:
                        if _try_load(queue.paths[pos - 1]):
                            playing = True
                elif ev.key == pygame.K_a:
                    auto_advance = not auto_advance

            # Pick up any new episodes that finished since last frame.
            queue.refresh(time.monotonic())

            # Lazy-load the first episode the moment one appears.
            if episode is None:
                if queue.paths:
                    _try_load(queue.paths[0])
                else:
                    _draw_idle(screen, font, f"waiting for episodes in {videos_dir}…")
                    pygame.display.flip()
                    clock.tick(8)
                    continue

            # Frame advancement (time-based — speed-scaled like
            # visualize_episode.py).
            if playing and episode.total > 1:
                now = pygame.time.get_ticks()
                step_ms = max(1, int(1000.0 / (args.fps * speed)))
                if now - last_advance_ms >= step_ms:
                    last_advance_ms = now
                    if frame_idx + 1 < episode.total:
                        frame_idx += 1
                        trace.push(episode.actions[frame_idx])
                    else:
                        # End of episode — auto-advance if the next one
                        # is ready, else hold on the final frame.
                        pos = queue.index_of(current_path) if current_path else -1
                        if auto_advance and pos >= 0 and pos + 1 < len(queue.paths):
                            _try_load(queue.paths[pos + 1])
                        else:
                            playing = False

            pos = queue.index_of(current_path) if current_path else 0
            _render_frame(
                screen,
                episode,
                frame_idx,
                trace.points,
                font,
                font_small,
                queue_size=len(queue.paths),
                queue_pos=max(0, pos),
                status="PLAY" if playing else "PAUSE",
                speed=speed,
                auto_advance=auto_advance,
            )
            pygame.display.flip()
            clock.tick(60)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
