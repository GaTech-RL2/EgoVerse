"""Coverage-balanced collection helpers.

Buckets each episode by (object-start quadrant, goal quadrant), giving 16
cells (4 x 4). Used by the mouse / scripted collectors to reject episodes
whose bucket is already at quota, so the saved dataset has roughly equal
coverage across all (start, goal) combinations.

Convention: pygame / pymunk image coords, origin at top-left, +y down.
So "top-right" on screen is high-x, low-y.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import zarr

N_QUADRANTS = 4
N_BUCKETS = N_QUADRANTS * N_QUADRANTS  # 16 (object-quad x goal-quad)
N_PUSHER_BUCKETS = N_QUADRANTS         # 4 (pusher quadrant only)

# Visual labels for buckets when printing the histogram.
_QUADRANT_LABELS = ("TL", "TR", "BL", "BR")

BucketFn = Callable[[dict, float], int]


def quadrant(x: float, y: float, world_size: float) -> int:
    """0=TL, 1=TR, 2=BL, 3=BR for a point in a world_size x world_size arena."""
    mid = world_size / 2.0
    qx = 1 if x >= mid else 0
    qy = 1 if y >= mid else 0
    return qy * 2 + qx


def bucket_for(episode_init: dict, world_size: float) -> int:
    """Bucket id 0..15 = object_quadrant * 4 + goal_quadrant."""
    obj_x, obj_y = episode_init["object_pose"][:2]
    goal_x, goal_y = episode_init["goal_pose"][:2]
    obj_q = quadrant(float(obj_x), float(obj_y), world_size)
    goal_q = quadrant(float(goal_x), float(goal_y), world_size)
    return obj_q * N_QUADRANTS + goal_q


def bucket_pusher_quad(episode_init: dict, world_size: float) -> int:
    """Bucket id 0..3 = pusher quadrant only.

    Intended for scenarios where the object/goal positions are constrained
    (e.g. object starts on the goal) and the only meaningful spatial signal
    is where the pusher starts."""
    ax, ay = episode_init["agent_pos"][:2]
    return quadrant(float(ax), float(ay), world_size)


def count_existing_buckets(
    folder: Path,
    world_size: float,
    bucket_fn: BucketFn = bucket_for,
    num_buckets: int = N_BUCKETS,
    filename_contains: str | None = None,
) -> list[int]:
    """Scan ``folder`` for ``*.zarr`` episode groups and tally per-bucket counts
    from each one's stored ``episode_init`` attribute.

    ``bucket_fn`` decides which bucket an episode falls in.
    ``num_buckets`` sizes the returned list (must agree with bucket_fn's range).
    ``filename_contains``, if set, restricts the scan to entries whose name
    contains that substring (e.g. ``"ontarget"`` to count only tagged files)."""
    counts = [0] * num_buckets
    folder = Path(folder)
    if not folder.exists():
        return counts
    for entry in sorted(folder.iterdir()):
        if not entry.is_dir() or not entry.name.endswith(".zarr"):
            continue
        if filename_contains is not None and filename_contains not in entry.name:
            continue
        try:
            group = zarr.open_group(str(entry), mode="r")
            raw = group.attrs.get("episode_init")
            if raw is None:
                continue
            ep_init = json.loads(raw)
            counts[bucket_fn(ep_init, world_size)] += 1
        except Exception:
            continue
    return counts


class BucketTracker:
    """Per-bucket counter with an acceptance test and a printable histogram.

    Defaults to 16 buckets laid out as a 4x4 (object-quadrant x goal-quadrant)
    grid for the standard collect scenario. Pass ``num_buckets=N_PUSHER_BUCKETS``
    for the 4-bucket pusher-only scheme used by on-target collection."""

    def __init__(
        self,
        target_per_bucket: int,
        initial_counts: list[int] | None = None,
        num_buckets: int = N_BUCKETS,
    ):
        if target_per_bucket < 1:
            raise ValueError("target_per_bucket must be >= 1")
        if num_buckets not in (N_BUCKETS, N_PUSHER_BUCKETS):
            raise ValueError(
                f"num_buckets must be {N_BUCKETS} or {N_PUSHER_BUCKETS}, got {num_buckets}"
            )
        self.target = int(target_per_bucket)
        self.num_buckets = int(num_buckets)
        if initial_counts is None:
            self.counts = [0] * num_buckets
        else:
            if len(initial_counts) != num_buckets:
                raise ValueError(
                    f"initial_counts must have {num_buckets} entries, got {len(initial_counts)}"
                )
            self.counts = [int(c) for c in initial_counts]

    def has_room(self, b: int) -> bool:
        return self.counts[b] < self.target

    def increment(self, b: int) -> None:
        self.counts[b] += 1

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def goal_total(self) -> int:
        return self.target * self.num_buckets

    @property
    def filled(self) -> bool:
        return all(c >= self.target for c in self.counts)

    def histogram(self) -> str:
        if self.num_buckets == N_BUCKETS:
            return self._histogram_4x4()
        return self._histogram_pusher_row()

    def _histogram_4x4(self) -> str:
        """4x4 grid: rows = object-start quadrant, cols = goal quadrant."""
        header = "          goal:  " + "  ".join(f"{l:>3}" for l in _QUADRANT_LABELS)
        rows = [header]
        for i, obj_lbl in enumerate(_QUADRANT_LABELS):
            cells = []
            for j in range(N_QUADRANTS):
                c = self.counts[i * N_QUADRANTS + j]
                marker = "*" if c >= self.target else " "
                cells.append(f"{c:>2}{marker}")
            rows.append(f"object {obj_lbl}:      " + "  ".join(cells))
        rows.append(f"  total: {self.total}/{self.goal_total} (target {self.target}/bucket)")
        return "\n".join(rows)

    def _histogram_pusher_row(self) -> str:
        """Single-row 4-bucket layout for pusher-quadrant balancing."""
        header = "pusher quad:  " + "  ".join(f"{l:>4}" for l in _QUADRANT_LABELS)
        cells = []
        for i in range(N_PUSHER_BUCKETS):
            c = self.counts[i]
            marker = "*" if c >= self.target else " "
            cells.append(f"{c:>3}{marker}")
        body = "             " + "  ".join(cells)
        footer = f"  total: {self.total}/{self.goal_total} (target {self.target}/bucket)"
        return "\n".join([header, body, footer])
