"""Vendored H-Net machinery for ``HNetCore`` (EgoVerse-pact-2).

A VERBATIM copy of the H-Net stage-tree modules from EgoVerse2's
``egomimic.models.hnet_nets`` (branch ``hpt-hnet-pusher-nc3``), pinned here so
``egomimic.models.bc_rnn_nets.hnet_core.HNetCore`` runs against the exact code
it was written and verified against -- DECOUPLED from pact-2's own
``egomimic.models.hnet_nets``, which diverged on the PACT branch
(``stages.py`` gained a ``residual_scale`` skip gate; ``blocks.py`` diverged
bidirectionally -- per-frame cross-attn / sliding-window attn in EV2 vs
``adaln_per_token`` / causal_conv1d fixes in pact). HNetCore's obs-only,
AdaLN-off, window-off, cross-attn-off config never touches the diverged code
paths, so the numerics match either tree; vendoring is the conservative,
instruction-mandated choice (do NOT modify pact-2's hnet_nets -- pact's own
code uses it). Mirrors the ``visual_core.py`` precedent from the original port.

Import closure (7 modules, fully self-contained within this package):
  context -> (torch only)
  config  -> (torch only)
  routing -> (torch only)
  blocks  -> config
  isotropic_builder -> blocks, config
  stages  -> blocks, context, isotropic_builder, routing
  hnet    -> context, stages

The ONLY edit vs the EV2 originals: intra-package imports rewritten from
``egomimic.models.hnet_nets.X`` -> ``egomimic.models.bc_rnn_nets._hnet_vendored.X``.
Do NOT edit these files to track pact's hnet_nets -- update from EV2 if the
H-Net core itself changes. See PORT_NOTES.md delta section.
"""
