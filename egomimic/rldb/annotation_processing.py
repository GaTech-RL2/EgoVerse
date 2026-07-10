"""Modular annotation processing.

Annotation ROLE is encoded in the zarr key NAME (e.g. ``annotations_task``,
``annotations_subtask``) instead of a ``level`` field inside each entry —
entries are plain ``{"text", "start_idx", "end_idx"}`` spans. ``ZarrDataset``
fetches every ``annotations*`` key into the batch (``list[list[str]]`` per
key after ``annotation_collate``); a processor then turns those raw lists
into per-role sampled strings inside ``process_batch_for_training``.

Processors are hydra-instantiable and return ``{role: list[str | None]}``:
``None`` marks "no annotation for this item" so the algo can apply its own
fallback (``default_prompt`` for prompts, no-loss for prediction targets).

- :class:`AnnotationProcessor` — default: samples ONE annotation per item
  from a single key with a ``first`` / ``random`` strategy (role ``task``).
- :class:`SubtaskAnnotationProcessor` — hierarchical: samples the
  conditioning instruction from ``task_key`` and the prediction target from
  ``subtask_key`` (roles ``task`` + ``subtask``). No cross-role fallback:
  a missing task never leaks the subtask text into the prompt.
"""

from __future__ import annotations

import random
from typing import Literal


class AnnotationProcessor:
    """Sample one annotation string per item from ``batch[key]``.

    ``batch[key]`` is the raw ``list[list[str]]`` left by
    ``annotation_collate`` (outer = batch items, inner = all annotation texts
    active at that item's frame). Items with no active annotation — or a
    missing/``None`` key — yield ``None``.
    """

    #: roles this processor emits (subclasses extend)
    roles: tuple[str, ...] = ("task",)

    def __init__(
        self,
        key: str | None = "annotations",
        strategy: Literal["first", "random"] = "random",
    ):
        self.key = key
        self.strategy = strategy

    def _sample_one(self, texts) -> str | None:
        if not texts:
            return None
        if self.strategy == "random":
            return texts[random.randint(0, len(texts) - 1)]
        return texts[0]  # "first"

    def _sample_key(self, batch, key, batch_size) -> list[str | None]:
        if key is None or key not in batch:
            return [None] * batch_size
        return [self._sample_one(texts) for texts in batch[key]]

    def __call__(self, batch, batch_size: int) -> dict[str, list[str | None]]:
        return {"task": self._sample_key(batch, self.key, batch_size)}


class SubtaskAnnotationProcessor(AnnotationProcessor):
    """Sample the (task instruction, subtask target) pair from role-named keys.

    - ``task`` (from ``task_key``): the conditioning instruction — populated
      for ALL annotated episodes (pick_place instruction / sort goal).
    - ``subtask`` (from ``subtask_key``): the prediction target — only
      populated where a decomposition exists (sort); ``None`` elsewhere, so
      those items contribute no LM loss and nothing leaks into the prompt.

    ``tie_identical`` keeps the two roles textually IDENTICAL whenever an
    item's two candidate lists are equal (the eva regime, where the same
    instruction set is injected under both keys): one sample is drawn and
    reused, instead of two independent draws picking different paraphrases.
    """

    roles = ("task", "subtask")

    def __init__(
        self,
        task_key: str = "annotations_task",
        subtask_key: str = "annotations_subtask",
        strategy: Literal["first", "random"] = "random",
        tie_identical: bool = True,
    ):
        super().__init__(key=task_key, strategy=strategy)
        self.task_key = task_key
        self.subtask_key = subtask_key
        self.tie_identical = tie_identical

    def __call__(self, batch, batch_size: int) -> dict[str, list[str | None]]:
        task = self._sample_key(batch, self.task_key, batch_size)
        subtask = self._sample_key(batch, self.subtask_key, batch_size)
        if self.tie_identical and self.task_key in batch and self.subtask_key in batch:
            for i in range(batch_size):
                t_list = batch[self.task_key][i]
                s_list = batch[self.subtask_key][i]
                if t_list and t_list == s_list:
                    subtask[i] = task[i]
        return {"task": task, "subtask": subtask}
