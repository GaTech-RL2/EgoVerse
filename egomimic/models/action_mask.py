"""Exclude structurally absent keypoint coordinates from action losses.

Fixed-width keypoint actions retain three coordinates for every topology slot,
including slots that an end-effector does not have. Registry-derived boolean
masks mark those coordinates as false while keeping wrist-pose coordinates and
present keypoint slots true. ``ActionMaskMixin`` registers the masks by
embodiment ID, and ``masked_loss`` removes false elements from the loss
reduction without changing tensor widths.

When no mask applies, ``masked_loss`` calls the supplied loss function without
a ``reduction`` argument and returns that result directly.
"""

from __future__ import annotations

import torch

from egomimic.rldb.embodiment.action_layout import action_masks
from egomimic.rldb.embodiment.embodiment import get_embodiment_id


def masked_loss(loss_fn, pred, target, mask):
    """Compute the weighted mean of an elementwise loss.

    Args:
        loss_fn: A loss callable accepting ``(pred, target)`` and, when a mask
            is present, ``reduction="none"``.
        pred: Predicted actions with the same shape as ``target``.
        target: Target actions, the same shape as ``pred``.
        mask: Boolean or numeric weights broadcastable to ``pred``. Registry
            masks are boolean; false elements receive zero weight. ``None``
            disables masking.

    Returns:
        With ``mask=None``, the direct result of ``loss_fn(pred, target)``.
        Otherwise, the weighted sum divided by the sum of expanded weights;
        an all-zero mask returns a differentiable scalar zero.
    """
    if mask is None:
        return loss_fn(pred, target)
    per_element = loss_fn(pred, target, reduction="none")
    weights = mask.to(per_element.dtype).expand_as(per_element)
    total = weights.sum()
    if total == 0:
        # Preserve a gradient path even when the expanded mask selects no
        # elements.
        return (per_element * weights).sum()
    return (per_element * weights).sum() / total


class ActionMaskMixin:
    """Register and select static action masks by integer embodiment ID."""

    def init_action_masks(self, infer_ac_dims) -> None:
        """Register masks derived from ``infer_ac_dims`` and the registry.

        Only entries that resolve to an incomplete keypoint topology and a
        recognized keypoint action width produce a mask. The masks are
        non-persistent buffers: module device moves include them, while
        ``state_dict`` and checkpoints do not.

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
        # Loss callers retain the leading common action columns, so discard any
        # trailing mask columns beyond that common width.
        return mask[:width]

    def action_mask(self, data, target):
        """Return an embodiment-selected mask broadcastable over ``target``.

        Args:
            data: A mapping whose optional ``embodiment`` value contains one
                integer ID per batch sample.
            target: The target action tensor, whose last axis is the width.

        Returns:
            For one unique masked ID, a boolean tensor with singleton leading
            axes. For mixed IDs, a tensor with one mask row per sample and
            singleton intermediate axes. Returns ``None`` if no registered mask
            applies or ``data`` has no ``embodiment`` value.
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
