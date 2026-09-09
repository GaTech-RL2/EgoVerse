#!/usr/bin/env python3
"""Dependency-free definition of the exhaustive corrected-ARC codec grid."""

from __future__ import annotations


def grid() -> list[dict]:
    d_values = list(range(16, 80, 10)) + [80]
    m_values = list(range(16, 160, 10)) + [160]
    r_values = list(range(6, 31, 4))
    rows = [
        {"distance": float(d), "M": int(m), "degrees": float(r), "kind": "arc"}
        for d in d_values
        for m in m_values
        for r in r_values
    ]
    identities = {(row["distance"], row["M"], row["degrees"]) for row in rows}
    if len(rows) != 896 or len(identities) != 896:
        raise RuntimeError("grid construction did not produce 896 unique rows")
    return rows
