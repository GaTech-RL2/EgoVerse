from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from egomimic.rldb.resolve_memo import memoized

logger = logging.getLogger(__name__)

BLOCKLIST_DIR = Path(__file__).resolve().parents[1] / "resources" / "blocklists"


def load_hash_list(source: str | Sequence[str] | None) -> frozenset[str]:
    """Episode hashes from a list, a text file, or a blocklist name.

    A string is a path if one exists there, else ``BLOCKLIST_DIR/<name>.txt``.
    Files hold one hash per line; ``#`` starts a comment.
    """
    if source is None:
        return frozenset()
    if not isinstance(source, str):
        return frozenset(str(h) for h in source)
    path = Path(source)
    if not path.is_file():
        path = BLOCKLIST_DIR / f"{source}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"exclude_hashes {source!r}: no such file, and no blocklist "
            f"{path.name} in {BLOCKLIST_DIR}"
        )
    hashes = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return frozenset(h for h in hashes if h)


class DatasetFilter:
    def __init__(
        self,
        filter_lambdas: Sequence[str] | None = None,
        episode_hashes: Sequence[str] | None = None,
        exclude_hashes: str | Sequence[str] | None = None,
    ) -> None:
        self.filter_lambdas = list(filter_lambdas or [])
        if isinstance(episode_hashes, str):  # `filters.episode_hashes=abc` override
            episode_hashes = [episode_hashes]
        pins = frozenset(str(h) for h in (episode_hashes or []))
        # Episodes that never match, e.g. known-bad data (see load_hash_list).
        self.exclude_hashes = load_hash_list(exclude_hashes)
        # An excluded pin is dropped rather than raised on, so one blocklist can
        # apply under hand-pinned splits; it is logged, and a pin list excluded
        # entirely is an error because an empty pin set means "no pin".
        dropped = pins & self.exclude_hashes
        if dropped:
            if dropped == pins:
                raise ValueError(
                    f"all {len(pins)} pinned episode(s) are in exclude_hashes"
                )
            logger.warning(
                "Dropping %d of %d pinned episode(s) listed in exclude_hashes: %s",
                len(dropped),
                len(pins),
                sorted(dropped)[:10] + (["..."] if len(dropped) > 10 else []),
            )
        # Pinned episode hashes. Empty = no pin. Validated at resolve time by the
        # resolvers (missing / deleted / wrong embodiment is an error there).
        self.episode_hashes: frozenset[str] = pins - dropped
        self.filters = []
        for expr in self.filter_lambdas:
            try:
                predicate = eval(expr)
            except Exception as exc:
                print(f"Invalid filter: {expr}", file=sys.stderr)
                raise ValueError(f"Invalid filter: {expr}") from exc
            if not callable(predicate):
                print(f"Invalid filter: {expr}", file=sys.stderr)
                raise ValueError(f"Invalid filter: {expr}")
            self.filters.append(predicate)

    def __repr__(self) -> str:
        return (
            f"DatasetFilter(filter_lambdas={self.filter_lambdas!r}, "
            f"episode_hashes={sorted(self.episode_hashes)!r}, "
            f"exclude_hashes=<{len(self.exclude_hashes)} hashes>)"
        )

    def cache_key(self) -> tuple | None:
        """Hashable identity of this filter's contents (resolve-once memo key).

        None means "don't memoize": a subclass that doesn't define its own
        cache_key may carry state that affects matches(), so it must not share
        its parent's key.
        """
        if "cache_key" not in type(self).__dict__:
            return None
        return (
            type(self),
            tuple(self.filter_lambdas),
            self.episode_hashes,
            self.exclude_hashes,
        )

    def matches(self, row: Mapping[str, Any]) -> bool:
        row = dict(row)
        if row.get("is_deleted", False):
            return False
        if self.episode_hashes and row.get("episode_hash") not in self.episode_hashes:
            return False
        if row.get("episode_hash") in self.exclude_hashes:
            return False
        for expr, predicate in zip(self.filter_lambdas, self.filters, strict=True):
            result = predicate(row)
            if not isinstance(result, bool):
                raise TypeError(f"Filter must return bool: {expr}")
            if not result:
                return False
        return True


class ScaleAnnotationDatasetFilter(DatasetFilter):
    def __init__(
        self,
        project_name: str,
        filter_lambdas: Sequence[str] | None = None,
        episode_hashes: Sequence[str] | None = None,
        exclude_hashes: str | Sequence[str] | None = None,
    ) -> None:
        from egomimic.utils.scale_utils import build_df_from_tasks, get_completed_tasks

        self.project_name = project_name
        self.api_key = os.environ["SCALE_API_KEY"]
        # Hydra builds this filter once per dataset instantiation; share the
        # Scale API pull across them inside resolve_once().
        self.tasks = memoized(
            ("scale_completed_tasks", project_name),
            lambda: get_completed_tasks(project_name, self.api_key),
        )
        self.df = build_df_from_tasks(self.tasks)
        self.completed_episode_hashes = frozenset(
            self.df["SEQUENCE_ID"].unique().tolist()
        )
        super().__init__(filter_lambdas, episode_hashes, exclude_hashes)

    def cache_key(self) -> tuple | None:
        base = super().cache_key()
        if base is None:
            return None
        return base + (self.project_name, self.completed_episode_hashes)

    def matches(self, row: Mapping[str, Any]) -> bool:
        if row.get("episode_hash") not in self.completed_episode_hashes:
            return False
        return super().matches(row)
