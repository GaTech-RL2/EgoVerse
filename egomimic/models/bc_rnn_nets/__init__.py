"""BC-RNN network components (robomimic recipe) — FACADE shim.

DESIGN.md step 6 relocated every member of this package to its ROLE home under
``egomimic.models.{stems,cores,heads}`` via ``git mv`` (no behaviour change):

  * ``ObsEncoder``        -> ``egomimic.models.stems.obs_encoder``
  * ``VisualCore`` / ``SpatialSoftmax`` -> ``egomimic.models.stems.visual_core``
  * ``LSTMCore``          -> ``egomimic.models.cores.lstm_core``
  * ``TransformerCore``   -> ``egomimic.models.cores.transformer_core``
  * ``HNetCore``          -> ``egomimic.models.cores.hnet_core``
  * ``GMMActionHead``     -> ``egomimic.models.heads.gmm_head``
  * ``QueryActionDecoder``-> ``egomimic.models.heads.query_decoder``

The 7 BC-RNN model configs' ``_target_``s were flipped to the role paths in the
same step. This package is kept ALIVE as a thin FACADE so every legacy import
path the repo / external scratch / error-message hints use keeps working
unchanged until the final flip (DESIGN.md step 13):

  * ``from egomimic.models.bc_rnn_nets import GMMActionHead, HNetCore, ...``   (names)
  * ``from egomimic.models.bc_rnn_nets.<sub> import ...``                      (submodules)
  * ``_target_: egomimic.models.bc_rnn_nets.<sub>.<Class>``                    (yaml _target_)
      <sub> in {obs_encoder, visual_core, lstm_core, transformer_core,
                hnet_core, gmm_head, query_decoder}

Mechanism (identical to the ``hnet_nets`` collapse shim from DESIGN.md step 3):
each member module now lives under a role package; we import the real module and
register it in ``sys.modules`` under the legacy ``egomimic.models.bc_rnn_nets.<sub>``
key, so ``from egomimic.models.bc_rnn_nets.<sub> import X`` and a yaml
``_target_: egomimic.models.bc_rnn_nets.<sub>.X`` both resolve to the SAME
module object (identical symbols, including private ones the tests may import).
No code is duplicated; this file forwards to the live role tree.

ROLE-HOME ROUTING: the legacy single-file names map to their NEW role package,
so e.g. ``bc_rnn_nets.visual_core`` aliases ``models.stems.visual_core``.

DO NOT add logic here — it is a pure forwarder. When DESIGN.md step 13 flips
every consumer to the role paths directly, this whole package is deleted.
"""

import importlib as _importlib
import sys as _sys

# Legacy ``bc_rnn_nets.<name>`` submodule  ->  its real role-package module.
_SUBMODULE_HOMES = {
    "obs_encoder": "egomimic.models.stems.obs_encoder",
    "visual_core": "egomimic.models.stems.visual_core",
    "lstm_core": "egomimic.models.cores.lstm_core",
    "transformer_core": "egomimic.models.cores.transformer_core",
    "hnet_core": "egomimic.models.cores.hnet_core",
    "gmm_head": "egomimic.models.heads.gmm_head",
    "query_decoder": "egomimic.models.heads.query_decoder",
}

for _legacy, _home in _SUBMODULE_HOMES.items():
    _mod = _importlib.import_module(_home)
    # Register the real module under the legacy dotted path so both
    # ``import egomimic.models.bc_rnn_nets.<legacy>`` and a yaml
    # ``_target_: egomimic.models.bc_rnn_nets.<legacy>.<Class>`` hit it.
    _sys.modules[f"{__name__}.{_legacy}"] = _mod

# Top-level name re-exports (mirror the pre-move ``bc_rnn_nets.__init__``).
from egomimic.models.cores.hnet_core import HNetCore
from egomimic.models.cores.lstm_core import LSTMCore
from egomimic.models.cores.transformer_core import TransformerCore
from egomimic.models.heads.gmm_head import GMMActionHead
from egomimic.models.heads.query_decoder import QueryActionDecoder
from egomimic.models.stems.obs_encoder import ObsEncoder
from egomimic.models.stems.visual_core import SpatialSoftmax, VisualCore

__all__ = [
    "ObsEncoder",
    "LSTMCore",
    "TransformerCore",
    "HNetCore",
    "GMMActionHead",
    "QueryActionDecoder",
    "VisualCore",
    "SpatialSoftmax",
]

del _sys, _importlib, _legacy, _home, _mod, _SUBMODULE_HOMES
