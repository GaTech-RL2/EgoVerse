"""Locate static keypoint validity slots in fixed-width action tensors.

A recognized keypoint layout concatenates one block per resolved side. Each
block contains zero or one wrist pose followed by three coordinates for every
slot in a shared topology. ``keypoint_action_mask`` marks wrist coordinates and
registry-valid slots true and structurally absent slots false. The validity
list applies to every frame; it does not encode a failed estimate in an
individual frame.
"""

from __future__ import annotations

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment, ResolvedEmbodiment

#: Accepted per-side prefixes: no wrist, XYZ plus YPR, or XYZ plus quaternion.
WRIST_WIDTHS = (0, 6, 7)


def keypoint_layout(action_dim: int, n_slots: int, n_sides: int = 2):
    """Infer equal per-side keypoint blocks from an action width.

    Args:
        action_dim: The width of one action vector.
        n_slots: Number of keypoint slots in each side's topology.
        n_sides: Number of side blocks concatenated in the action.

    Returns:
        ``(per_side_width, wrist_width)`` when the action divides evenly across
        the sides and each block has a 0-, 6-, or 7-column wrist prefix before
        ``3 * n_slots`` coordinates. Otherwise, ``None``.
    """
    if n_sides <= 0 or action_dim % n_sides:
        return None
    per_side = action_dim // n_sides
    wrist = per_side - 3 * n_slots
    if wrist not in WRIST_WIDTHS:
        return None
    return per_side, wrist


def keypoint_action_mask(spec, action_dim: int) -> np.ndarray | None:
    """Build a static validity mask for a resolved keypoint action layout.

    Args:
        spec: An embodiment name, a morphology mapping, or a
            ``ResolvedEmbodiment``.
        action_dim: The width of one action vector.

    Returns:
        A boolean array of shape ``(action_dim,)``. Wrist coordinates and
        coordinates for registry-valid slots are true; coordinates for absent
        slots are false. Sides are ordered left, then right. Returns ``None``
        when every selected end-effector has a complete topology or when
        ``action_dim`` is not a recognized keypoint layout.

    Raises:
        TypeError: If ``spec`` is not a supported resolver input.
        ValueError: If ``spec`` cannot be resolved or its sides declare
            different keypoint topologies.
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
    """Build masks for resolvable entries with structurally absent slots.

    Args:
        infer_ac_dims: The head's ``{embodiment name: action width}`` mapping.

    Returns:
        A mapping from embodiment name to boolean mask. Entries are omitted if
        their name or width cannot be resolved, their width is not a recognized
        keypoint layout, or all selected keypoint slots are valid.
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
