from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from egomimic.utils.scale_utils import build_df_from_tasks, get_completed_tasks


class DatasetFilter:
    def __init__(
        self,
        filter_lambdas: Sequence[str] | None = None,
        episode_hashes: Sequence[str] | None = None,
    ) -> None:
        self.filter_lambdas = list(filter_lambdas or [])
        # Pinned episode hashes. Empty = no pin. Validated at resolve time by the
        # resolvers (missing / deleted / wrong embodiment is an error there).
        if isinstance(episode_hashes, str):  # `filters.episode_hashes=abc` override
            episode_hashes = [episode_hashes]
        self.episode_hashes: frozenset[str] = frozenset(
            str(h) for h in (episode_hashes or [])
        )
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
            f"episode_hashes={sorted(self.episode_hashes)!r})"
        )

    def matches(self, row: Mapping[str, Any]) -> bool:
        row = dict(row)
        if row.get("is_deleted", False):
            return False
        if self.episode_hashes and row.get("episode_hash") not in self.episode_hashes:
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
    ) -> None:
        self.project_name = project_name
        self.api_key = os.environ["SCALE_API_KEY"]
        self.tasks = get_completed_tasks(self.project_name, self.api_key)
        self.df = build_df_from_tasks(self.tasks)
        self.completed_episode_hashes = set(self.df["SEQUENCE_ID"].unique().tolist())
        super().__init__(filter_lambdas, episode_hashes)

    def matches(self, row: Mapping[str, Any]) -> bool:
        if row.get("episode_hash") not in self.completed_episode_hashes:
            return False
        return super().matches(row)
