"""Backward-compat import shim (PHASE 1, structural fixes).

The composable input modules moved from ``egomimic/algo/input_modules.py`` to
their role home ``egomimic/models/stems/input_modules.py`` (DESIGN.md §2
``models/stems/`` — the input-side stem pieces of the hourglass). The classes
are unchanged; this module re-exports the full public surface so the legacy
import path ``egomimic.algo.input_modules`` (and any
``_target_: egomimic.algo.input_modules.ObsToken`` config) keeps resolving to
the exact same objects.

Prefer importing from ``egomimic.models.stems.input_modules`` directly in new
code. This shim is removed at the DESIGN.md step-13 final flip.
"""

from egomimic.models.stems.input_modules import (  # noqa: F401
    ActionInToken,
    InputModule,
    ObsToken,
)

__all__ = ["ActionInToken", "InputModule", "ObsToken"]
