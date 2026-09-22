"""torch.compile wiring shared by the algos.

``nn.Module.compile`` (not ``torch.compile(module)``) is what the algos call:
it swaps the module's call implementation IN PLACE, so the module keeps its
identity and -- the part that matters here -- its ``state_dict`` keys. Wrapping
a module in ``torch.compile`` instead returns an ``OptimizedModule`` whose
parameters are all prefixed ``_orig_mod.``, which silently breaks every
checkpoint written before the flag was turned on.

The other half of the contract: ``Module.compile`` only routes ``__call__``.
A submodule the algo invokes through a named method (``stem.compute_latent``,
``head.compute_loss``) runs uncompiled no matter what, so only modules that are
actually *called* are worth handing over.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def compile_modules(
    modules: Iterable[tuple[str, Optional[nn.Module]]],
    mode: Optional[str] = None,
    dynamic: bool = False,
) -> List[str]:
    """``module.compile(...)`` for each named module; returns the names done."""
    kwargs = {"dynamic": dynamic}
    if mode:
        kwargs["mode"] = mode
    done = []
    for name, module in modules:
        if not isinstance(module, nn.Module):
            continue  # a head with no sub-network, or a None slot
        module.compile(**kwargs)
        done.append(name)
    if done:
        logger.info(
            "torch.compile (mode=%s, dynamic=%s): %s",
            mode or "default",
            dynamic,
            ", ".join(done),
        )
    return done
