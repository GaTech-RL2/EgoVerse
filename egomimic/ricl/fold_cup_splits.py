"""Episode splits for the fold_clothes -> cup_on_saucer OOD RICL experiment.

Train (both models, parametric): task=fold_clothes, eva_bimanual.
Eval (held out): task=cup_on_saucer, eva_bimanual. The baseline never trains on
cup_on_saucer; RICL retrieves cup_on_saucer demos at eval (bank == cup, LOSO episode).
Episode-hash subsets (100 each, eval_success only) are frozen in episode_lists/*.json.
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
def cup_eval_episode_hashes() -> frozenset[str]:
    return frozenset(json.loads((_LISTS / "cup_eval_episodes.json").read_text()))


@lru_cache(maxsize=1)
def all_fold_cup_episode_hashes() -> frozenset[str]:
    return fold_train_episode_hashes() | cup_eval_episode_hashes()
