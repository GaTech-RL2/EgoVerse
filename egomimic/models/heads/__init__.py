"""Per-embodiment HEAD / readout modules (DESIGN.md §2 ``models/heads/``).

Role home for the output-side readouts of the hourglass:

  * :class:`GMMActionHead`      — diagonal Gaussian-mixture action head (default
                                  flat-linear chunk readout).
  * :class:`QueryActionDecoder` — ACT/HPT-style learnable-query chunk readout
                                  (selected via ``chunk_head: queries``).

Relocated here from ``models/bc_rnn_nets`` in DESIGN.md step 6 via ``git mv``
(no behaviour change); the old ``egomimic.models.bc_rnn_nets`` import path stays
alive through the ``bc_rnn_nets`` facade shim until DESIGN.md step 13.
"""

from egomimic.models.heads.gmm_head import GMMActionHead
from egomimic.models.heads.query_decoder import QueryActionDecoder

__all__ = ["GMMActionHead", "QueryActionDecoder"]
