"""Baseline-policy ZOO (DESIGN.md §2 ``egomimic/algo/zoo/``).

Role home for the third-party / comparison policies that share the
``egomimic.algo.algo.Algo`` spine but are NOT part of the hourglass line:

  * :class:`ACT` — action-chunking transformer baseline.
  * :class:`HPT` — heterogeneous pre-trained transformer baseline.
  * :class:`PI`  — pi-0 / pi-0.5 baseline (requires openpi).

Relocated here from flat ``egomimic/algo/{act,hpt,pi}.py`` in DESIGN.md step 8
via ``git mv`` (no behaviour change). The legacy ``egomimic.algo.{act,hpt,pi}``
import paths + yaml ``_target_``s stay ALIVE through the thin re-export shim
files kept at ``egomimic/algo/{act,hpt,pi}.py`` until the final flip
(DESIGN.md step 13).

``PI`` is imported lazily (it pulls in ``openpi``, an optional dep) so importing
this package does not hard-require openpi.
"""

from egomimic.algo.zoo.act import ACT
from egomimic.algo.zoo.hpt import HPT

__all__ = ["ACT", "HPT", "PI"]


def __getattr__(name):  # PEP 562 lazy import for the optional-openpi PI algo.
    if name == "PI":
        from egomimic.algo.zoo.pi import PI

        return PI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
