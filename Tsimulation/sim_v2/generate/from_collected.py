"""Expand collected PushShapes Zarr demonstrations to a fixed total.

Unlike :mod:`run_batch`, this launcher uses the operator-collected episodes as
MimicGen sources.  It is resumable: source episodes are copied once, generated
successes are appended contiguously, and every candidate is re-rendered and
revalidated before its Zarr is committed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import zarr

from Tsimulation.sim_v2.generate.mimicgen import SourceDemo, generate
from Tsimulation.sim_v2.generate.run_batch import write


def episode_dirs(path: Path) -> list[Path]:
    return sorted(item for item in path.glob("*.zarr") if item.is_dir())


def load_sources(path: Path) -> list[SourceDemo]:
    demos: list[SourceDemo] = []
    for episode in episode_dirs(path):
        group = zarr.open_group(str(episode), mode="r")
        init = json.loads(str(group.attrs["episode_init"]))
        frames = int(group.attrs["total_frames"])
        actions = np.asarray(group["actions"][:frames], dtype=np.float64)
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError(f"{episode} has invalid actions shape {actions.shape}")
        demos.append(
            SourceDemo(
                agent=str(init["pusher_shape"]),
                actions=actions,
                object_pose=tuple(init["object_pose"]),
                goal_pose=tuple(init["goal_pose"]),
                agent_pos=tuple(init["agent_pos"]),
                agent_angle=float(init.get("agent_angle", 0.0)),
                object_shape=str(init["object_shape"]),
                obstacle_level=int(init["obstacle_level"]),
            )
        )
    if not demos:
        raise ValueError(f"no collected .zarr episodes found under {path}")
    signatures = {
        (demo.agent, demo.object_shape, demo.obstacle_level, demo.actions.shape[1])
        for demo in demos
    }
    if len(signatures) != 1:
        raise ValueError(f"source episodes do not share one signature: {signatures}")
    return demos


def prepare_output(source: Path, destination: Path) -> int:
    sources = episode_dirs(source)
    existing = episode_dirs(destination)
    if existing:
        if len(existing) < len(sources):
            raise RuntimeError(
                f"destination has {len(existing)} episodes but source has "
                f"{len(sources)}; refusing an ambiguous partial seed copy"
            )
        return len(existing)
    destination.mkdir(parents=True, exist_ok=True)
    for episode in sources:
        shutil.copytree(episode, destination / episode.name)
    return len(sources)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="output root; episodes go under <out>/<agent>/<object>")
    parser.add_argument("--target-total", type=int, default=3000)
    parser.add_argument("--attempt-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=250825)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()

    if args.target_total <= 0 or args.attempt_batch <= 0:
        parser.error("--target-total and --attempt-batch must be positive")
    sources = load_sources(args.source)
    first = sources[0]
    destination = args.out / first.agent / first.object_shape
    total = prepare_output(args.source, destination)
    if total > args.target_total:
        raise RuntimeError(
            f"destination already has {total} episodes, above target "
            f"{args.target_total}"
        )

    print(
        f"sources={len(sources)} destination={destination} "
        f"existing={total} target={args.target_total}",
        flush=True,
    )
    batch = 0
    attempts_total = 0
    started = time.time()
    while total < args.target_total:
        remaining = args.target_total - total
        result = generate(
            sources,
            n_attempts=args.attempt_batch,
            seed=args.seed + batch,
        )
        attempts_total += result.attempts
        candidates = result.demos[:remaining]
        written = write(candidates, args.out, first.agent, args.image_size)
        total += written
        print(
            f"batch={batch:04d} attempts={result.attempts} "
            f"candidates={len(result.demos)} written={written} "
            f"total={total}/{args.target_total} "
            f"rate={total - len(sources):d}/{attempts_total} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        batch += 1

    print(
        f"COMPLETE total={total} attempts={attempts_total} "
        f"elapsed={time.time() - started:.1f}s destination={destination}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
