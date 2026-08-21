"""TEACHER-FORCED evaluators (DESIGN.md §2 ``egomimic/eval/{tf}``).

The teacher-forced (GT-conditioned) prediction/overlay evaluators — the cheap,
dense, per-step signal you render + iterate on BEFORE spending on closed-loop
sim. Curated here in DESIGN.md step 8 (``git mv``, no behaviour change):

  * :class:`DFoTValEval`        — TF action-prediction val with viz overlay.
  * :class:`DFoTControllerTFEval` — TF sanity check for the spatial_rh
    closed-loop controller (drives the inference step on GT history).
"""

from egomimic.eval.tf.eval_dfot_val import DFoTValEval
from egomimic.eval.tf.eval_dfot_controller_tf import DFoTControllerTFEval

__all__ = ["DFoTValEval", "DFoTControllerTFEval"]
