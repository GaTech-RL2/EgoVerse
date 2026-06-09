"""Segment-level train/valid split definitions for the pick-left RICL run."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

SEGMENTS_CSV = Path(__file__).with_name("episode_lists") / "action_segments.csv"
BY = ("verb", "object", "hand")

TRAIN_GROUPS = (
    ("pick", "blue stuffed toy", "left"),
    ("pick", "eggplant", "left"),
    ("pick", "blue marker", "left"),
    ("pick", "pink stuffed toy", "left"),
)

VALID_GROUPS = (
    ("pick", "brown stuffed toy", "left"),
    ("pick", "pink stuffed toy", "right"),
    ("pick", "white cup", "left"),
)

# OOD / novel-task eval: the models are trained on PICK only, so PLACE is a verb
# the parametric baseline never saw. RICL can retrieve PLACE demos as context.
# These segments are never in parametric training (training is filtered to the pick
# groups via SegmentFilteredDataset); they appear only at eval (as targets for both
# models, and as RICL's retrieval support set, leave-one-segment-out).
PLACE_EVAL_GROUPS = (
    ("place", "white coffee cup", "right"),
    ("place", "blue stuffed toy", "left"),
)


def _episode_hashes(groups: tuple[tuple[str, str, str], ...]) -> frozenset[str]:
    allowed = set(groups)
    eps: set[str] = set()
    with SEGMENTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            key = tuple(row[col] for col in BY)
            if key in allowed:
                eps.add(row["episode"])
    return frozenset(eps)


@lru_cache(maxsize=1)
def train_episode_hashes() -> frozenset[str]:
    return _episode_hashes(TRAIN_GROUPS)


@lru_cache(maxsize=1)
def valid_episode_hashes() -> frozenset[str]:
    return _episode_hashes(VALID_GROUPS)


@lru_cache(maxsize=1)
def place_eval_episode_hashes() -> frozenset[str]:
    return _episode_hashes(PLACE_EVAL_GROUPS)


@lru_cache(maxsize=1)
def all_episode_hashes() -> frozenset[str]:
    return train_episode_hashes() | valid_episode_hashes()
