"""Per-embodiment HEAD / readout modules (DESIGN.md §2 ``models/heads/``).

Role home for the output-side readouts of the hourglass:

  * :class:`GMMActionHead`      — diagonal Gaussian-mixture action head (default
                                  flat-linear chunk readout).

Relocated here from ``models/bc_rnn_nets`` in DESIGN.md step 6 via ``git mv``
(no behaviour change); the ``bc_rnn_nets`` facade shim was deleted at the
DESIGN.md step-13 final flip.
"""

from egomimic.models.heads.gmm_head import GMMActionHead

__all__ = ["GMMActionHead"]
