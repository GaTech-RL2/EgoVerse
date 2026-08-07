"""Diagnostic PROBE evaluators (DESIGN.md §2 ``egomimic/eval/{probes}``).

Visualisation probes into the chunker internals — curated here in DESIGN.md
step 8 (``git mv``, no behaviour change):

  * :class:`BoundaryStripEval` — renders the chunker's boundary probability +
    its committed boundary decisions over time.
  * :class:`PCATokenEval`      — PCA scatter of the post-compression (chunker
    output) high-level tokens.
"""

from egomimic.eval.probes.eval_boundary_strip import BoundaryStripEval
from egomimic.eval.probes.eval_pca_tokens import PCATokenEval

__all__ = ["BoundaryStripEval", "PCATokenEval"]
