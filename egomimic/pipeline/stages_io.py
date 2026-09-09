"""Small data-boundary stages for configured pipelines."""

from __future__ import annotations

from egomimic.pipeline.core import Stage


class ActionTargetBuilder(Stage):
    """Move normalized loader actions into the pipeline target namespace.

    ``action_key`` is the batch key the dataset's transform list actually
    writes. It defaults to ``actions`` (Planar), but embodiment transform lists
    name it differently -- Eva/Human/Yam cartesian pipelines emit
    ``actions_cartesian`` -- and the graph must read the key that exists rather
    than force every embodiment to rename its output.
    """

    train_only = True
    writes = ("target",)

    def __init__(self, action_key: str = "actions"):
        super().__init__()
        self.action_key = str(action_key)
        if not self.action_key:
            raise ValueError("action_key must be non-empty")
        self.reads = (self.action_key,)

    def forward(self, batch: dict) -> dict:
        if self.action_key not in batch:
            raise ValueError(
                f"ActionTargetBuilder requires normalized actions at "
                f"{self.action_key!r}"
            )
        batch["target"] = batch.pop(self.action_key)
        return batch
