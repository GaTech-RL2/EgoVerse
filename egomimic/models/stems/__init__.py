"""Per-embodiment STEM modules (DESIGN.md §2 ``models/stems/``).

Role home for the input-side pieces of the hourglass: the per-frame obs fuse
(:class:`ObsEncoder`) and the image codec (:class:`VisualCore` /
:class:`SpatialSoftmax`). Relocated here from ``models/bc_rnn_nets`` in
DESIGN.md step 6 via ``git mv`` (no behaviour change); the old
``egomimic.models.bc_rnn_nets`` import path stays alive through the
``bc_rnn_nets`` facade shim until DESIGN.md step 13.
"""

from egomimic.models.stems.obs_encoder import ObsEncoder
from egomimic.models.stems.visual_core import SpatialSoftmax, VisualCore

__all__ = ["ObsEncoder", "VisualCore", "SpatialSoftmax"]
