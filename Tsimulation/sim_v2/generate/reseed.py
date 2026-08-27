"""Reseed MimicGen from the DISTINCT subset of an existing dataset.

An existing 1000-episode cell is not 1000 behaviours: measured on
control_gap_all6x13_1000_per_cell_simv2_20260826, ~70% of every cell has a
near-duplicate within 20 units of path shape, and the cell collapses to
228-339 distinct behaviours at a 60-100 unit threshold.

Those few hundred distinct episodes are, however, a far better seed set than a
scripted planner produces: they are real successes, they already span the
layout distribution, and crucially they carry VARIED initial effector poses,
which the planner-seeded runs did not (854 episodes with agent_angle == 0.000).

So: dedupe the cell down to its distinct core, use THAT as the MimicGen source
set, and regenerate. The point is not more episodes -- it is that a source set
of a few hundred genuinely different manoeuvres spans a much larger space
under retargeting than a dozen copies of one manoeuvre.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import zarr

from Tsimulation.sim_v2.generate.diversity import trajectory_signature
from Tsimulation.sim_v2.generate.mimicgen import SourceDemo


def load_cell(cell_dir: str, limit: int | None = None) -> list[dict]:
    """Read episodes from a dataset cell into raw records."""
    out = []
    for p in sorted(glob.glob(os.path.join(cell_dir, "*.zarr")))[:limit]:
        try:
            g = zarr.open(p, mode="r")
            n = int(g.attrs["total_frames"])
            if n < 5:
                continue
            ini = json.loads(g.attrs["episode_init"])
            st = np.asarray(g["observations.state"])[:n]
            out.append({
                "path": p,
                "actions": np.asarray(g["actions"])[:n],
                "object_xy": st[:, 3:5],
                "init": ini,
            })
        except Exception:
            continue
    return out


def distinct_subset(records: list[dict], min_distance: float = 60.0) -> list[dict]:
    """Greedy dedupe on object-path shape.

    Greedy rather than clustering: it needs no cluster count chosen in
    advance, and the threshold is in world units so it can be reasoned about
    against the arena size instead of tuned blindly.
    """
    kept: list[dict] = []
    sigs: list[np.ndarray] = []
    for r in records:
        s = trajectory_signature(r["object_xy"])
        if sigs and np.linalg.norm(np.asarray(sigs) - s[None, :], axis=1).min() < min_distance:
            continue
        kept.append(r)
        sigs.append(s)
    return kept


def to_sources(records: list[dict]) -> list[SourceDemo]:
    """Raw records -> MimicGen sources, preserving each episode's own scene."""
    out = []
    for r in records:
        ini = r["init"]
        out.append(SourceDemo(
            agent=ini["pusher_shape"],
            actions=np.asarray(r["actions"], dtype=np.float64),
            object_pose=tuple(ini["object_pose"]),
            goal_pose=tuple(ini["goal_pose"]),
            agent_pos=tuple(ini["agent_pos"]),
            object_shape=ini["object_shape"],
            obstacle_level=int(ini.get("obstacle_level", 0)),
        ))
    return out


def sources_from_cell(cell_dir: str, min_distance: float = 60.0,
                      limit: int | None = None):
    """Convenience: load a cell, dedupe it, return (sources, n_read)."""
    recs = load_cell(cell_dir, limit=limit)
    kept = distinct_subset(recs, min_distance)
    return to_sources(kept), len(recs)
