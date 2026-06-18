"""Episode splits for the (fold_clothes + bag_groceries) -> cup_on_saucer RICL experiment.

Train (both models, parametric): task in {fold_clothes, bag_groceries}, eva_bimanual.
Eval (held out): task=cup_on_saucer, eva_bimanual. The baseline never trains on
cup_on_saucer; RICL retrieves cup_on_saucer demos at eval (bank == cup, LOSO episode).

This is the cross-task generalization setting RICL needs (genuinely novel eval task
where the parametric baseline is weak), distinct from the within-group pickplace LOSO
that could not separate retrieval from the generic-interp confound.

Episode-hash subsets (100 each, is_eval=False / eval_success=True, zarr present) are
frozen in episode_lists/{fold_train,bag_train,cup_eval}_episodes.json. fold_train and
cup_eval reuse the lists from the earlier fold-only experiment (fold_cup_splits).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_LISTS = Path(__file__).with_name("episode_lists")


@lru_cache(maxsize=1)
def fold_train_episode_hashes() -> frozenset[str]:
    return frozenset(json.loads((_LISTS / "fold_train_episodes.json").read_text()))


@lru_cache(maxsize=1)
def bag_train_episode_hashes() -> frozenset[str]:
    return frozenset(json.loads((_LISTS / "bag_train_episodes.json").read_text()))


@lru_cache(maxsize=1)
def foldbag_train_episode_hashes() -> frozenset[str]:
    """Combined train pool: fold_clothes + bag_groceries episodes."""
    return fold_train_episode_hashes() | bag_train_episode_hashes()


@lru_cache(maxsize=1)
def cup_eval_episode_hashes() -> frozenset[str]:
    return frozenset(json.loads((_LISTS / "cup_eval_episodes.json").read_text()))


@lru_cache(maxsize=1)
def all_foldbag_cup_episode_hashes() -> frozenset[str]:
    """Every episode touched by this experiment (for the embedding sweep)."""
    return foldbag_train_episode_hashes() | cup_eval_episode_hashes()
