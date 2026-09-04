"""Drop the action dimensions an end-effector structurally does not own.

A shared policy head has one width, so a three-finger hand and a parallel jaw
write the same keypoint block a five-finger hand does. The slots they do not
own carry no signal, and training a head to reproduce them teaches it a fiction.
``egomimic.rldb.embodiment.action_layout`` says which dimensions those are; this
module carries the answer onto a head and applies it to a loss.

The unmasked path is preserved exactly: when nothing is masked, the loss call is
the original one, so an existing run's numbers do not move.
"""

from __future__ import annotations

import torch

from egomimic.rldb.embodiment.action_layout import action_masks
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def masked_loss(loss_fn, pred, target, mask):
    """Average ``loss_fn`` over the unmasked action dimensions.

    Args:
        loss_fn: A loss taking ``(pred, target)`` and a ``reduction`` keyword.
        pred: Predicted actions.
        target: Target actions, the same shape as ``pred``.
        mask: A tensor broadcastable to ``pred`` that is nonzero on the
            dimensions to keep, or ``None``.

    Returns:
        The scalar loss. With ``mask`` of ``None`` this is ``loss_fn(pred,
        target)`` itself, not an equivalent rewriting of it.
    """
    if mask is None:
        return loss_fn(pred, target)
    per_element = loss_fn(pred, target, reduction="none")
    weights = mask.to(per_element.dtype).expand_as(per_element)
    total = weights.sum()
    if total == 0:
        # Every dimension is masked. Return a zero that still carries a
        # gradient path, so a misconfigured mask fails loudly in the metrics
        # rather than detaching part of the graph.
        return (per_element * weights).sum()
    return (per_element * weights).sum() / total


class ActionMaskMixin:
    """Give an ``nn.Module`` head one keypoint validity mask per embodiment."""

    def init_action_masks(self, infer_ac_dims) -> None:
        """Register a mask for each embodiment in ``infer_ac_dims``.

        Masks are non-persistent buffers, so they follow the module across
        devices without entering a checkpoint or changing its state dict.

        Args:
            infer_ac_dims: The head's ``{embodiment name: action width}``
                mapping, or ``None``.
        """
        self._action_mask_ids: tuple[int, ...] = ()
        ids = []
        for name, mask in action_masks(infer_ac_dims).items():
            try:
                embodiment_id = get_embodiment_id(name)
            except KeyError:
                continue
            self.register_buffer(
                f"_action_mask_{embodiment_id}",
                torch.from_numpy(mask).to(torch.bool),
                persistent=False,
            )
            ids.append(embodiment_id)
        self._action_mask_ids = tuple(sorted(ids))

    def _mask_for_id(self, embodiment_id: int, width: int):
        mask = getattr(self, f"_action_mask_{int(embodiment_id)}", None)
        if mask is None or mask.shape[-1] < width:
            return None
        # A head may compare only the leading dimensions the two tensors share,
        # so trim from the front the way the tensors themselves were trimmed.
        return mask[:width]

    def action_mask(self, data, target):
        """Return the mask for this batch, shaped to broadcast over ``target``.

        Args:
            data: The batch. ``data["embodiment"]`` holds one integer ID per
                sample; without it no mask applies.
            target: The target action tensor, whose last axis is the width.

        Returns:
            A boolean tensor broadcastable to ``target``, or ``None`` when
            nothing in this batch masks anything.
        """
        if not getattr(self, "_action_mask_ids", ()):
            return None
        ids = data.get("embodiment") if hasattr(data, "get") else None
        if ids is None:
            return None
        ids = torch.as_tensor(ids).reshape(-1)
        width = target.shape[-1]
        unique = [int(i) for i in torch.unique(ids).tolist()]
        if not any(i in self._action_mask_ids for i in unique):
            return None

        if len(unique) == 1:
            mask = self._mask_for_id(unique[0], width)
            return None if mask is None else mask.view(*([1] * (target.ndim - 1)), width)

        ones = torch.ones(width, dtype=torch.bool, device=target.device)
        rows = torch.stack(
            [
                m if (m := self._mask_for_id(int(i), width)) is not None else ones
                for i in ids
            ]
        )
        return rows.view(rows.shape[0], *([1] * (target.ndim - 2)), width)


__all__ = ["ActionMaskMixin", "masked_loss"]
