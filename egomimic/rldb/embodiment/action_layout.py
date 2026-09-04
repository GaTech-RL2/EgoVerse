"""Locate each end-effector's keypoint slots inside a canonical action tensor.

The ``keypoints`` action space lays out one fixed-width block per side:
``[xyz(3), rot(3|4), keypoints(3 * n_slots)]``. Every end-effector writes the
full block, because a shared head has one width. A hand that has no pinky, or a
parallel jaw that has three meaningful slots at all, therefore writes slots it
does not own.

`DEXTEROUS_EMBODIMENT_DESIGN_V2.md` §2.3 calls those slots masked, and they are
not the same thing as the ``1e9`` sentinel the corpus already uses. A sentinel
says one frame's estimate failed; a mask says this hand has no pinky on any
frame. The loss must ignore the second kind, which is what this module's mask
feeds.
"""

from __future__ import annotations

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment, ResolvedEmbodiment

#: Wrist block widths a side may carry before its keypoints: none, xyz + ypr,
#: or xyz + quaternion. See ``_split_keypoints``.
WRIST_WIDTHS = (0, 6, 7)


def keypoint_layout(action_dim: int, n_slots: int, n_sides: int = 2):
    """Split a canonical action width into its per-side blocks.

    Args:
        action_dim: The width of one action vector.
        n_slots: Slots in the keypoint topology, the same for every side.
        n_sides: Sides the tensor concatenates.

    Returns:
        ``(per_side, wrist)``: the width of one side's block and the width of
        the pose that precedes its keypoints. ``None`` when ``action_dim`` is
        not a keypoint layout, which is the ordinary answer for a ``cartesian``
        tensor.
    """
    if n_sides <= 0 or action_dim % n_sides:
        return None
    per_side = action_dim // n_sides
    wrist = per_side - 3 * n_slots
    if wrist not in WRIST_WIDTHS:
        return None
    return per_side, wrist


def keypoint_action_mask(spec, action_dim: int) -> np.ndarray | None:
    """Return which dimensions of an action tensor the end-effectors populate.

    Args:
        spec: An embodiment name, a morphology mapping, or a
            ``ResolvedEmbodiment``.
        action_dim: The width of one action vector.

    Returns:
        A ``(action_dim,)`` boolean array, ``True`` where the dimension carries
        a slot the end-effector owns. ``None`` when nothing is masked, so a
        caller can take its existing unmasked path unchanged. That is the
        answer for every embodiment in the corpus today.

    Raises:
        ValueError: If the sides disagree on keypoint topology, because one
            action tensor cannot hold two layouts.
    """
    resolved = spec if isinstance(spec, ResolvedEmbodiment) else Embodiment.resolve(spec)
    sides = [side for side in ("left", "right") if side in resolved.end_effectors]
    if not sides:
        return None
    topologies = {resolved.keypoints(side).topology for side in sides}
    if len(topologies) != 1:
        raise ValueError(
            f"{resolved.describe()}: sides declare keypoint topologies "
            f"{sorted(topologies)}; one action tensor holds one layout"
        )
    n_slots = resolved.keypoints(sides[0]).n_slots
    if all(resolved.keypoints(side).is_complete for side in sides):
        return None

    layout = keypoint_layout(action_dim, n_slots, len(sides))
    if layout is None:
        return None
    per_side, wrist = layout

    mask = np.ones(action_dim, dtype=bool)
    for index, side in enumerate(sides):
        keypoints = resolved.keypoints(side)
        base = index * per_side + wrist
        for slot in range(n_slots):
            if slot not in keypoints.valid:
                mask[base + 3 * slot : base + 3 * slot + 3] = False
    return mask


def action_masks(infer_ac_dims) -> dict[str, np.ndarray]:
    """Build one mask per embodiment a head serves.

    Args:
        infer_ac_dims: The head's ``{embodiment name: action width}`` mapping.

    Returns:
        A mapping from embodiment name to its mask, holding only the
        embodiments that actually mask something. An embodiment the registry
        does not know is skipped rather than failing head construction, which
        keeps an unrelated checkpoint loadable.
    """
    masks = {}
    for name, action_dim in (infer_ac_dims or {}).items():
        try:
            mask = keypoint_action_mask(name, int(action_dim))
        except (TypeError, ValueError):
            continue
        if mask is not None:
            masks[name] = mask
    return masks


__all__ = ["WRIST_WIDTHS", "action_masks", "keypoint_action_mask", "keypoint_layout"]
