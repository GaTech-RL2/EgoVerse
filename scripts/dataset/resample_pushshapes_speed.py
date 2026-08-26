#!/usr/bin/env python3
"""Create speed-adjusted, replay-validated PushShapes datasets.

The action timeline is resampled by ``--speed`` while keeping the dataset FPS
unchanged (1.5 means 1.5x faster and 0.5 means half speed). Each transformed
sequence is replayed from its recorded ``episode_init``. Observations, rewards,
and images are regenerated from that replay so they stay aligned with actions.

Examples:
    python scripts/dataset/resample_pushshapes_speed.py INPUT OUTPUT \
        --speed 1.5 --pusher-color red --variant-name red_circle \
        --min-replay-coverage 0.95 --workers 6

    python scripts/dataset/resample_pushshapes_speed.py INPUT OUTPUT \
        --speed 0.5 --pusher-color blue --variant-name blue_circle \
        --min-replay-coverage 0.95 --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from Tsimulation.collect.replay_init import reset_to_init
from Tsimulation.pushshapes import PushShapesEnv

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

IMAGE_KEY = "observations.images.front_img_1"
STATE_KEY = "observations.state"
COMMAND_KEY = "observations.pusher_cmd_pose"
ACTION_KEY = "actions"
REWARD_KEY = "reward"
GOAL_KEY = "goal_pose"


@dataclass(frozen=True)
class EpisodeResult:
    episode: str
    source_frames: int
    output_frames: int
    speed: float
    coverage: float | None
    status: str
    error: str = ""


def output_frame_count(source_frames: int, speed: float) -> int:
    """Return a length that keeps both timeline endpoints."""
    if source_frames < 1:
        raise ValueError(f"source_frames must be positive, got {source_frames}")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"speed must be a finite positive number, got {speed}")
    if source_frames == 1:
        return 1
    return max(2, int(round((source_frames - 1) / speed)) + 1)


def resample_actions(actions: np.ndarray, speed: float) -> np.ndarray:
    """Linearly resample absolute action targets along the time axis."""
    actions = np.asarray(actions)
    if actions.ndim < 1 or len(actions) < 1:
        raise ValueError("actions must contain at least one frame")
    output_frames = output_frame_count(len(actions), speed)
    if output_frames == len(actions):
        return actions.copy()
    source_positions = np.linspace(0.0, len(actions) - 1, output_frames)
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, len(actions) - 1)
    weights = source_positions - left
    weight_shape = (output_frames,) + (1,) * (actions.ndim - 1)
    weights = weights.reshape(weight_shape)
    result = actions[left] * (1.0 - weights) + actions[right] * weights
    return result.astype(actions.dtype, copy=False)


def recolor_red_pusher_blue(image: np.ndarray) -> np.ndarray:
    """Turn rendered red pusher pixels blue without touching the blue object."""
    result = np.asarray(image, dtype=np.uint8).copy()
    rgb = result.astype(np.int16)
    mask = (
        (rgb[..., 0] > 120)
        & (rgb[..., 0] > rgb[..., 1] + 35)
        & (rgb[..., 0] > rgb[..., 2] + 35)
    )
    red = result[..., 0][mask].copy()
    result[..., 0][mask] = result[..., 2][mask]
    result[..., 2][mask] = red
    return result


def _decode_episode_init(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = attrs.get("episode_init")
    if raw is None:
        raise ValueError("episode is missing attrs['episode_init']")
    init = json.loads(raw) if isinstance(raw, str) else dict(raw)
    required = {"agent_pos", "object_pose", "goal_pose"}
    missing = sorted(required - init.keys())
    if missing:
        raise ValueError(f"episode_init is missing {missing}")
    return init


def _read_annotations(group: zarr.Group, speed: float, frames: int) -> list[tuple[str, int, int]]:
    if "annotations" not in group:
        return []
    annotations: list[tuple[str, int, int]] = []
    for raw in group["annotations"][:]:
        payload = json.loads(bytes(raw).decode("utf-8"))
        start = int(round(int(payload["start_idx"]) / speed))
        end = int(round(int(payload["end_idx"]) / speed))
        start = min(max(start, 0), frames - 1)
        end = min(max(end, start), frames - 1)
        annotations.append((str(payload["text"]), start, end))
    return annotations


def _image_size(attrs: dict[str, Any]) -> int:
    try:
        shape = attrs["features"][IMAGE_KEY]["shape"]
        if len(shape) >= 2 and int(shape[0]) == int(shape[1]):
            return int(shape[0])
    except (KeyError, TypeError, ValueError):
        pass
    return 96


def _writer_metadata(
    attrs: dict[str, Any], *, source_episode: str, speed: float,
    pusher_color: str, variant_name: str, coverage: float,
) -> dict[str, Any]:
    # ZarrWriter must own these schema fields for the transformed arrays.
    excluded = {"total_frames", "features", "embodiment", "fps", "task_name", "task_description"}
    metadata = {key: value for key, value in attrs.items() if key not in excluded}
    metadata.update(
        {
            "source_episode": source_episode,
            "speed_factor": float(speed),
            "pusher_color": pusher_color,
            "embodiment_variant": variant_name,
            "replay_final_coverage": float(coverage),
        }
    )
    return metadata


def _process_episode(
    source_path: str,
    output_dir: str,
    speed: float,
    pusher_color: str,
    variant_name: str,
    min_coverage: float | None,
    resume: bool,
) -> EpisodeResult:
    source = Path(source_path)
    destination = Path(output_dir) / source.name
    if resume and destination.exists():
        return EpisodeResult(source.name, 0, 0, speed, None, "skipped_existing")

    source_frames = 0
    output_frames = 0
    env: PushShapesEnv | None = None
    temporary = destination.parent / f".{destination.name}.partial-{os.getpid()}"
    try:
        group = zarr.open_group(str(source), mode="r")
        attrs = dict(group.attrs)
        source_frames = int(attrs.get("total_frames", group[ACTION_KEY].shape[0]))
        if source_frames < 1 or source_frames > group[ACTION_KEY].shape[0]:
            raise ValueError(
                f"invalid total_frames={source_frames} for actions shape {group[ACTION_KEY].shape}"
            )
        source_actions = np.asarray(group[ACTION_KEY][:source_frames])
        actions = resample_actions(source_actions, speed)
        output_frames = len(actions)
        episode_init = _decode_episode_init(attrs)
        annotations = _read_annotations(group, speed, output_frames)

        env = PushShapesEnv(
            object_shape=str(episode_init.get("object_shape", "T")),
            pusher_shape=str(episode_init.get("pusher_shape", "circle")),
            obstacle_level=int(episode_init.get("obstacle_level", 0)),
            image_size=_image_size(attrs),
        )
        # Timeline compression/expansion and the physical embodiment limit must
        # move together. Otherwise a faster timeline merely asks an unchanged
        # pusher to jump farther per step and destroys otherwise valid demos.
        env.PUSHER_SPEED = type(env).PUSHER_SPEED * speed
        env.STICK_TURN_RATE = type(env).STICK_TURN_RATE * speed
        # The validation pass does not need pixels. Rejected episodes therefore
        # avoid the most expensive part of conversion entirely.
        env._skip_obs_render = True
        obs, _ = reset_to_init(env, episode_init)

        states: list[np.ndarray] = []
        commands: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        goals: list[np.ndarray] = []
        coverage = 0.0
        for action in actions:
            states.append(
                np.concatenate(
                    [
                        np.asarray(obs["agent_pos"], dtype=np.float64).reshape(-1)[:2],
                        np.asarray(obs["object_pose"], dtype=np.float64).reshape(-1)[:3],
                    ]
                )
            )
            command = np.asarray(action, dtype=np.float64).reshape(-1)
            commands.append(command)
            goals.append(np.asarray(obs["goal_pose"], dtype=np.float64).reshape(-1)[:3])
            obs, reward, _terminated, _truncated, info = env.step(action)
            coverage = float(info.get("coverage", reward))
            rewards.append(np.asarray([reward], dtype=np.float64))

        if min_coverage is not None and not coverage > min_coverage:
            return EpisodeResult(
                source.name, source_frames, output_frames, speed, coverage, "rejected_coverage"
            )

        # Replay accepted actions once more with rendering enabled. PushShapes
        # replay is deterministic from episode_init, so these pixels align with
        # the numeric observations collected above.
        env._skip_obs_render = False
        obs, _ = reset_to_init(env, episode_init)
        images: list[np.ndarray] = []
        for index, action in enumerate(actions):
            image = np.asarray(obs["image"], dtype=np.uint8)
            if pusher_color == "blue":
                image = recolor_red_pusher_blue(image)
            images.append(image)
            if index == output_frames - 1:
                env._skip_obs_render = True
            obs, _reward, _terminated, _truncated, _info = env.step(action)

        numeric_data = {
            STATE_KEY: np.stack(states),
            COMMAND_KEY: np.stack(commands),
            ACTION_KEY: np.asarray(actions),
            REWARD_KEY: np.stack(rewards),
            GOAL_KEY: np.stack(goals),
        }
        image_data = {IMAGE_KEY: np.stack(images)}
        if temporary.exists():
            shutil.rmtree(temporary)
        writer = ZarrWriter(
            episode_path=temporary,
            embodiment=str(attrs.get("embodiment", "pushshapes_sim")),
            fps=int(attrs.get("fps", 30)),
            task_name=str(attrs.get("task_name", "pushshapes")),
            task_description=str(attrs.get("task_description", "")),
            annotations=annotations,
        )
        writer.write(
            numeric_data=numeric_data,
            image_data=image_data,
            metadata_override=_writer_metadata(
                attrs,
                source_episode=source.name,
                speed=speed,
                pusher_color=pusher_color,
                variant_name=variant_name,
                coverage=coverage,
            ),
        )
        temporary.replace(destination)
        return EpisodeResult(source.name, source_frames, output_frames, speed, coverage, "written")
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        return EpisodeResult(
            source.name,
            source_frames,
            output_frames,
            speed,
            None,
            "error",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
    finally:
        if env is not None:
            env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--pusher-color", choices=("red", "blue"), default="red")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument(
        "--min-replay-coverage",
        type=float,
        default=None,
        help="Write only episodes with final replay coverage strictly above this value.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=0, help="0 processes all episodes")
    parser.add_argument("--resume", action="store_true", help="Skip output episodes that already exist")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if not args.input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input_dir}")
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise SystemExit("input and output directories must be different")
    output_frame_count(2, args.speed)  # validates speed
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.min_replay_coverage is not None and not 0.0 <= args.min_replay_coverage <= 1.0:
        raise SystemExit("--min-replay-coverage must be between 0 and 1")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.variant_name) is None:
        raise SystemExit("--variant-name may contain only letters, numbers, _, ., and -")

    episodes = sorted(path for path in args.input_dir.glob("*.zarr") if path.is_dir())
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.glob("*.zarr"))
    if existing and not args.resume:
        raise SystemExit(
            f"output already contains {len(existing)} episodes; use a new directory or --resume"
        )
    print(
        f"Processing {len(episodes)} episodes: speed={args.speed:g}x, "
        f"color={args.pusher_color}, coverage>{args.min_replay_coverage}, workers={args.workers}",
        flush=True,
    )

    manifest_path = args.output_dir / "speed_transform_manifest.csv"
    fields = list(EpisodeResult.__dataclass_fields__)
    counts: dict[str, int] = {}
    with manifest_path.open("a" if args.resume else "w", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fields)
        if not args.resume or manifest_path.stat().st_size == 0:
            writer.writeheader()

        def record(result: EpisodeResult, finished: int) -> None:
            writer.writerow(asdict(result))
            manifest.flush()
            counts[result.status] = counts.get(result.status, 0) + 1
            if finished <= 5 or finished % 20 == 0 or result.status == "error":
                cov = "n/a" if result.coverage is None else f"{result.coverage:.4f}"
                print(
                    f"[{finished:5d}/{len(episodes)}] {result.episode}: "
                    f"{result.status} T={result.source_frames}->{result.output_frames} cov={cov}",
                    flush=True,
                )
                if result.error:
                    print(result.error, file=sys.stderr, flush=True)

        jobs = [
            (
                str(episode), str(args.output_dir), args.speed, args.pusher_color,
                args.variant_name, args.min_replay_coverage, args.resume,
            )
            for episode in episodes
        ]
        if args.workers == 1:
            for finished, job in enumerate(jobs, 1):
                record(_process_episode(*job), finished)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_process_episode, *job) for job in jobs]
                for finished, future in enumerate(as_completed(futures), 1):
                    record(future.result(), finished)

    print(f"Finished. Counts: {json.dumps(counts, sort_keys=True)}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
