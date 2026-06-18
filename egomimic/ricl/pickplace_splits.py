"""Segment-level splits for the pick/place RICL experiment (eva_bimanual).

Train/val and eval are disjoint *(verb, object, hand)* segment groups, all inside
``pick_place`` episodes. Filtering is frame-level (``SegmentFilteredDataset``):
one episode can contain both a train and an eval segment, so we keep only the
frames whose segment matches the split's groups — episode-hash overlap between
splits never leaks frames.

Train/VAL : pick blue stuffed toy (L), pick white cup (R),
            place eggplant (L), place green bowl (R)
EVAL      : pick croissant (L), pick doritos (R),
            place juice pouch (L), place pink mug (R)

Train and VAL share the four train groups; the episodes carrying those groups are
split ~90/10 into held-out valid episodes (deterministic by md5 of the hash), so
validation loss is measured on NEW episodes, never on a trained frame.
"""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

SEGMENTS_CSV = Path(__file__).with_name("episode_lists") / "action_segments.csv"
BY = ("verb", "object", "hand")

TRAIN_GROUPS = (
    ("pick", "blue stuffed toy", "left"),
    ("pick", "white cup", "right"),
    ("place", "eggplant", "left"),
    ("place", "green bowl", "right"),
)

EVAL_GROUPS = (
    ("pick", "croissant", "left"),
    ("pick", "doritos", "right"),
    ("place", "juice pouch", "left"),
    ("place", "pink mug", "right"),
)

# Hold out ~1/VALID_FRACTION_DENOM of the train-group episodes for validation.
VALID_FRACTION_DENOM = 10


def _episode_hashes(groups: tuple[tuple[str, str, str], ...]) -> frozenset[str]:
    allowed = set(groups)
    eps: set[str] = set()
    with SEGMENTS_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if tuple(row[col] for col in BY) in allowed:
                eps.add(row["episode"])
    return frozenset(eps)


def _is_valid_episode(episode_hash: str) -> bool:
    """Deterministic held-out flag, stable across runs (no RNG seed plumbing)."""
    digest = hashlib.md5(episode_hash.encode()).hexdigest()
    return int(digest, 16) % VALID_FRACTION_DENOM == 0


@lru_cache(maxsize=1)
def _train_group_episodes() -> frozenset[str]:
    return _episode_hashes(TRAIN_GROUPS)


@lru_cache(maxsize=1)
def train_episode_hashes() -> frozenset[str]:
    """Train-group episodes minus the held-out validation episodes."""
    return frozenset(h for h in _train_group_episodes() if not _is_valid_episode(h))


@lru_cache(maxsize=1)
def valid_episode_hashes() -> frozenset[str]:
    """Held-out train-group episodes (same groups, NEW episodes)."""
    return frozenset(h for h in _train_group_episodes() if _is_valid_episode(h))


@lru_cache(maxsize=1)
def eval_episode_hashes() -> frozenset[str]:
    """Episodes carrying the held-out eval groups."""
    return _episode_hashes(EVAL_GROUPS)


@lru_cache(maxsize=1)
def all_pickplace_episode_hashes() -> frozenset[str]:
    """Every episode touched by this experiment (for the embedding sweep)."""
    return train_episode_hashes() | valid_episode_hashes() | eval_episode_hashes()
