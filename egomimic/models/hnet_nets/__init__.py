"""Backward-compat shim — ``egomimic.models.hnet_nets`` moved to
``egomimic.models.hnet`` (DESIGN.md step 3, the H-Net collapse).

The canonical H-Net stage tree now lives at ``egomimic.models.hnet`` (the
pact SUPERSET: cross-attn + ``residual_scale`` + ``causal_conv1d`` +
``adaln_per_token``; the inferior ``bc_rnn_nets/_hnet_vendored`` dup was
deleted in the same step). This package is a thin facade kept ALIVE so the
old import paths used across the repo keep working unchanged until the final
flip (DESIGN.md step 13):

  * ``from egomimic.models.hnet_nets import HNet, ChunkerStage, ...``         (names)
  * ``from egomimic.models.hnet_nets.<sub> import ...``                       (submodules)
      <sub> in {blocks, cond_encoders, config, context, hnet, image_encoders,
                isotropic_builder, routing, stages, _smoke_stages}

Mechanism: we import each real ``egomimic.models.hnet.<sub>`` module and
register it in ``sys.modules`` under the legacy ``egomimic.models.hnet_nets.<sub>``
key, so ``from egomimic.models.hnet_nets.stages import _scatter_state`` and the
like resolve to the SAME module object (identical symbols, including private
ones such as ``_scatter_state`` / ``_adaln`` that the unit tests import). No
code is duplicated; this file forwards to the live tree.

DO NOT add logic here — it is a pure forwarder. When DESIGN.md step 13 flips
every consumer to ``egomimic.models.hnet`` directly, this whole package is
deleted.
"""

import sys as _sys
import importlib as _importlib

# Submodules to alias under the legacy package name. Order doesn't matter for
# aliasing (importlib resolves intra-tree deps), but listing the leaves the repo
# imports by path makes the contract explicit.
_SUBMODULES = (
    "blocks",
    "cond_encoders",
    "config",
    "context",
    "hnet",
    "image_encoders",
    "isotropic_builder",
    "routing",
    "stages",
    "_smoke_stages",
)

for _name in _SUBMODULES:
    _mod = _importlib.import_module(f"egomimic.models.hnet.{_name}")
    # Register under the legacy dotted path so
    # ``import egomimic.models.hnet_nets.<name>`` and
    # ``from egomimic.models.hnet_nets.<name> import X`` both hit the real module.
    _sys.modules[f"{__name__}.{_name}"] = _mod

# Top-level name re-exports (mirror the canonical ``models.hnet.__init__``).
from egomimic.models.hnet.cond_encoders import CondEncoderModule
from egomimic.models.hnet.context import HNetContext
from egomimic.models.hnet.hnet import HNet, ratio_loss_from_aux
from egomimic.models.hnet.stages import (
    ChunkerStage,
    ComputeStage,
    EncoderDecoderStage,
)

__all__ = [
    "CondEncoderModule",
    "HNetContext",
    "HNet",
    "ratio_loss_from_aux",
    "ChunkerStage",
    "ComputeStage",
    "EncoderDecoderStage",
]

del _sys, _importlib, _name, _mod, _SUBMODULES
