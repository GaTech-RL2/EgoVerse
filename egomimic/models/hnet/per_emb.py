"""PerEmb — the single, reusable per-embodiment container for the dual-stream H-Net.

Instead of every class re-implementing "shared module vs. one-module-per-embodiment"
with ad-hoc ``nn.ModuleDict`` + dispatch code, this is the ONE place that logic lives:

  * ``per_emb(factory, embodiments)`` builds either a ``PerEmb`` (one copy per
    embodiment, when ``embodiments`` is given) or a single SHARED module/param
    (bare ``factory()`` when ``embodiments`` is falsy). The shared branch returns
    the bare object so param NAMES are byte-identical to the non-per-emb layout
    (BC for single-embodiment / legacy checkpoints).
  * ``pick(x, emb_id)`` resolves a (possibly-``PerEmb``) attribute to the ACTIVE
    embodiment's module/param. Plain (shared) objects pass through unchanged.

A module is "specific" (per-embodiment) by wrapping it in ``per_emb(...)``; it is
"agnostic" (shared) by passing ``embodiments=None``. Works for ``nn.Module``
factories (→ ``nn.ModuleDict``) and ``nn.Parameter`` factories (→ ``nn.ParameterDict``).
"""

from typing import Optional, Sequence

import torch.nn as nn


class PerEmb(nn.Module):
    """Holds one item per embodiment, keyed by the domain name (== ctx.embodiment_id).

    ``get(emb_id)`` / calling the module returns the ACTIVE embodiment's copy; the
    other embodiments' copies are never touched on that forward (so an inactive
    embodiment's params get no gradient — the intended per-emb behavior).
    """

    def __init__(self, factory, embodiments: Sequence[str]):
        super().__init__()
        if not embodiments:
            raise ValueError(
                "PerEmb needs a non-empty embodiments list; use "
                "per_emb(factory, embodiments) for the shared fallback."
            )
        self.embodiments = list(embodiments)
        probe = factory()
        Dict = nn.ParameterDict if isinstance(probe, nn.Parameter) else nn.ModuleDict
        table = {self.embodiments[0]: probe}
        for e in self.embodiments[1:]:
            table[e] = factory()
        self.table = Dict(table)

    def get(self, emb_id: Optional[str]):
        if emb_id is None:
            if len(self.table) == 1:
                return next(iter(self.table.values()))
            raise RuntimeError(
                "PerEmb.get requires ctx.embodiment_id (multi-embodiment)."
            )
        if emb_id not in self.table:
            raise KeyError(
                f"PerEmb has no entry for embodiment {emb_id!r}; "
                f"available: {list(self.table.keys())}."
            )
        return self.table[emb_id]

    def forward(self, *args, emb_id: Optional[str] = None, **kwargs):
        return self.get(emb_id)(*args, **kwargs)


def per_emb(factory, embodiments: Optional[Sequence[str]]):
    """One copy per embodiment (``PerEmb``) when ``embodiments`` is given, else a
    single SHARED ``factory()`` (bare → byte-identical naming / BC)."""
    return PerEmb(factory, embodiments) if embodiments else factory()


def pick(x, emb_id: Optional[str]):
    """Resolve a possibly-per-emb attribute to the active module/param for
    ``emb_id``; plain (shared) objects pass through unchanged."""
    return x.get(emb_id) if isinstance(x, PerEmb) else x
