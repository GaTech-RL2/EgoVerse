"""Backward-compat FACADE — ``egomimic/eval`` was CURATED into role buckets in
DESIGN.md step 8 (the ``zoo`` bucket later dissolved into per-algo eval
 folders ``eval/{hpt,pi,act}/`` — see DESIGN.md): ``eval/{core,tf,dfot,probes,hpt,pi,act}/``.

Every evaluator module moved (via ``git mv``, no behaviour change) from the flat
``egomimic/eval/eval_*.py`` layout into role buckets:

  * core/   — eval, eval_video, eval_composite, eval_sim, eval_hnet, eval_vae_recon
  * tf/     — eval_dfot_val, eval_dfot_controller_tf
  * dfot/   — the DFoT self/video/policy rollout evaluators
  * probes/ — eval_boundary_strip, eval_pca_tokens
  * hpt/    — eval_hpt   (peer per-algo eval folder; formerly zoo/)
  * pi/     — eval_pi    (peer per-algo eval folder; formerly zoo/)
  * act/    — eval_act   (peer per-algo eval folder; formerly zoo/)

The EVALUATOR YAML ``_target_``s were edited to the new bucketed paths directly
in the same step (DESIGN.md notes ``_target_`` resolution is weaker through
``__init__`` shims, so the yamls do NOT rely on this facade). This facade keeps
the legacy ``egomimic.eval.eval_<name>`` *python import* paths alive for code
consumers (``egomimic.trainHydra``, the ``scripts/`` smoke/verify helpers, any
external scratch) until the final flip (DESIGN.md step 13):

  * ``from egomimic.eval.eval import Eval``                  (-> eval.core.eval)
  * ``from egomimic.eval.eval_sim import PackedSimEval``     (-> eval.core.eval_sim)
  * ``from egomimic.eval.eval_hnet import HNetEvalVideo``    (-> eval.core.eval_hnet)
  * ... and every other ``egomimic.eval.eval_<name>`` legacy path.

Mechanism (identical to the ``hnet_nets`` / ``bc_rnn_nets`` collapse shims):
each legacy ``egomimic.eval.eval_<name>`` dotted path is registered in
``sys.modules`` pointing at the REAL bucketed module, so the legacy import
resolves to the SAME module object (identical symbols, incl. private ones the
scripts import such as ``_env_to_zarr_pushshapes`` / ``_state_to_init``).

DO NOT add logic here — pure forwarder; deleted at DESIGN.md step 13.
"""

import importlib as _importlib
import sys as _sys

# Legacy flat module basename  ->  its real bucketed module path.
_MODULE_HOMES = {
    # core/
    "eval": "egomimic.eval.core.eval",
    "eval_video": "egomimic.eval.core.eval_video",
    "eval_composite": "egomimic.eval.core.eval_composite",
    "eval_sim": "egomimic.eval.core.eval_sim",
    "eval_hnet": "egomimic.eval.core.eval_hnet",
    "eval_vae_recon": "egomimic.eval.core.eval_vae_recon",
    # tf/
    "eval_dfot_val": "egomimic.eval.tf.eval_dfot_val",
    "eval_dfot_controller_tf": "egomimic.eval.tf.eval_dfot_controller_tf",
    # dfot/
    "eval_dfot_self_rollout": "egomimic.eval.dfot.eval_dfot_self_rollout",
    "eval_dfot_video_rollout": "egomimic.eval.dfot.eval_dfot_video_rollout",
    # COMBINE A: the pixel/spatial video-rollout modules were collapsed into
    # the family-agnostic eval_dfot_video_rollout (which exports the compat
    # aliases). Keep the legacy import names pointing at the unified module.
    "eval_dfot_pixel_video_rollout": "egomimic.eval.dfot.eval_dfot_video_rollout",  # noqa: E501
    "eval_dfot_spatial_video_rollout": "egomimic.eval.dfot.eval_dfot_video_rollout",  # noqa: E501
    # COMBINE B: the policy-action + receding-horizon modules were merged into
    # eval_dfot_policy (they share _rollout / _ddim_from_v). Keep the legacy
    # import names pointing at the merged module.
    "eval_dfot_policy": "egomimic.eval.dfot.eval_dfot_policy",
    "eval_dfot_policy_action": "egomimic.eval.dfot.eval_dfot_policy",
    "eval_dfot_policy_receding_horizon": "egomimic.eval.dfot.eval_dfot_policy",  # noqa: E501
    "eval_dfot_bundle_anchored": "egomimic.eval.dfot.eval_dfot_bundle_anchored",
    # probes/
    "eval_boundary_strip": "egomimic.eval.probes.eval_boundary_strip",
    "eval_pca_tokens": "egomimic.eval.probes.eval_pca_tokens",
    # per-algo eval folders (hpt/ pi/ act/)
    "eval_act": "egomimic.eval.act.eval_act",
    "eval_hpt": "egomimic.eval.hpt.eval_hpt",
    "eval_pi": "egomimic.eval.pi.eval_pi",
}

for _legacy, _home in _MODULE_HOMES.items():
    _mod = _importlib.import_module(_home)
    _sys.modules[f"{__name__}.{_legacy}"] = _mod

del _sys, _importlib, _legacy, _home, _mod, _MODULE_HOMES
