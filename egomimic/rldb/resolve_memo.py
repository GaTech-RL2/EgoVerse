"""Resolve-once memo scope.

``train()`` instantiates each dataset config up to three times (train, valid,
norm-stat copy), and each instantiation re-pulls the SQL episode table, the
Scale task list and the path resolution. Inside ``resolve_once()`` those are
computed once per key. Outside it nothing is cached, so long-lived processes
(notebooks, eval scripts) always see the current episode set.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Hashable
from typing import Any, TypeVar

_T = TypeVar("_T")

_MEMO: dict[Hashable, Any] | None = None


@contextlib.contextmanager
def resolve_once() -> Generator[None, None, None]:
    """Memoize resolution for the duration of the block. Nested use joins the
    outermost scope; the memo is dropped when that scope exits."""
    global _MEMO
    if _MEMO is not None:
        yield
        return
    _MEMO = {}
    try:
        yield
    finally:
        _MEMO = None


def memoized(key: Hashable | None, compute: Callable[[], _T]) -> _T:
    """``compute()`` once per ``key`` inside ``resolve_once()``. Uncached outside
    it, or when ``key`` is None. Callers must not mutate the returned value."""
    if _MEMO is None or key is None:
        return compute()
    if key not in _MEMO:
        _MEMO[key] = compute()
    return _MEMO[key]
