"""Per-embodiment STEM modules (DESIGN.md §2 ``models/stems/``).

Role home for the input-side pieces of the hourglass: the per-frame obs fuse
(:class:`ObsEncoder`), the image codec (:class:`VisualCore` /
:class:`SpatialSoftmax`), the per-frame conv image encoder
(:class:`SimpleConv`), and the obs→AdaLN-cond fuser (:class:`CondEncoderModule`).

``ObsEncoder`` / ``VisualCore`` were relocated here from ``models/bc_rnn_nets``
in DESIGN.md step 6 (no behaviour change). ``SimpleConv`` (image_encoders.py)
and ``CondEncoderModule`` (cond_encoders.py) were relocated here from
``models/hnet`` in dedup collapse 2 via ``git mv`` so that ``models/hnet`` is
pure chunking machinery — a verbatim move (proven state_dict-identical), no
logic change.
"""

from egomimic.models.stems.cond_encoders import CondEncoderModule
from egomimic.models.stems.image_encoders import SimpleConv
from egomimic.models.stems.obs_encoder import ObsEncoder
from egomimic.models.stems.visual_core import SpatialSoftmax, VisualCore

__all__ = [
    "ObsEncoder",
    "VisualCore",
    "SpatialSoftmax",
    "SimpleConv",
    "CondEncoderModule",
]
