"""Thin re-export shim for the pushshapes_sim keymap.

The canonical embodiment helper (`get_keymap` + `viz_gt_preds`) lives in
:mod:`egomimic.rldb.embodiment.pushshapes` because the training pipeline
imports it from there. This module just re-exports `get_keymap` so local
sim-only scripts can use the shorter import path.
"""

from __future__ import annotations

from egomimic.rldb.embodiment.pushshapes import get_keymap

__all__ = ["get_keymap"]
