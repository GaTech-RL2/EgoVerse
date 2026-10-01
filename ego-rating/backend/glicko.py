"""Glicko-2 ratings derived from the pairwise comparison log (via the `glicko2` lib).

Every configured axis has its OWN independent rating system: :func:`compute`
replays one axis's outcomes; :func:`compute_all` does it for every axis.

Each span carries (rating, rd, vol) per axis:
  * rating — the quality estimate (Glicko-2 scale, starts 1500)
  * rd     — rating deviation = 1σ uncertainty (starts 350; shrinks with games)
  * vol    — volatility (starts 0.06)

We replay one axis's outcomes ('a'/'b' decisive, 'equal' = a draw, score 0.5
for both) in timestamp order, updating both players **per game** — one
comparison = a one-game rating period. This is the standard online Glicko-2
mode (e.g. Lichess) and is valid for sparse/interactive data (Glickman).
Ratings are never stored; recomputing from the log keeps the comparison log the
single source of truth.

τ (system constant) defaults to 0.3 — low, because clip quality is *stable* (each
comparison is a noisy measurement of a fixed quality, not genuine volatility).

The active-learning pairer uses `rd` (uncertainty) and `expected_score` (outcome
probability, for the entropy term). See literature notes in pairing.py.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Optional

import glicko2

from backend import db

DEFAULT_INITIAL = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
DEFAULT_TAU = 0.3

_initial = DEFAULT_INITIAL
_rd = DEFAULT_RD
_vol = DEFAULT_VOL
_tau = DEFAULT_TAU


def set_params(initial=None, rd=None, vol=None, tau=None) -> None:
    global _initial, _rd, _vol, _tau
    if initial is not None:
        _initial = float(initial)
    if rd is not None:
        _rd = float(rd)
    if vol is not None:
        _vol = float(vol)
    if tau is not None:
        _tau = float(tau)


def params() -> dict:
    return {"initial": _initial, "rd": _rd, "vol": _vol, "tau": _tau}


def _new_player() -> "glicko2.Player":
    p = glicko2.Player(rating=_initial, rd=_rd, vol=_vol)
    p._tau = _tau  # the lib stores tau per-player; default 0.5, we want lower
    return p


# Glicko expected score (used by the active-learning entropy term), with the
# g(φ) deflation so combined uncertainty widens the outcome toward 0.5.
_Q = math.log(10) / 400.0


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * _Q**2 * rd**2 / math.pi**2)


def expected_score(r_a: float, rd_a: float, r_b: float, rd_b: float) -> float:
    """Glicko P(a beats b), deflated by the pair's combined rating deviation."""
    g = _g(math.sqrt(rd_a**2 + rd_b**2))
    return 1.0 / (1.0 + 10.0 ** (-g * (r_a - r_b) / 400.0))


def compute(
    conn: sqlite3.Connection,
    axis: str,
    scene: Optional[str] = None,
    operator: Optional[str] = None,
) -> dict[str, dict]:
    """Replay one axis's outcomes into per-span Glicko-2 state + W/L/D stats.

    Returns {span_id: {"rating", "rd", "vol", "games", "wins", "losses",
    "draws"}} for every span in the (optionally filtered) pool. Only
    comparisons between two in-pool spans count.
    """
    where, p = db.span_filter(scene, operator)
    pool = {
        r["span_id"]
        for r in conn.execute(f"SELECT span_id FROM spans{where}", p).fetchall()
    }
    players = {s: _new_player() for s in pool}
    stats = {s: {"games": 0, "wins": 0, "losses": 0, "draws": 0} for s in pool}

    rows = conn.execute(
        """
        SELECT c.span_a, c.span_b, ar.outcome
        FROM comparisons c
        JOIN axis_ratings ar ON ar.comparison_id = c.comparison_id
        WHERE c.skipped = 0 AND ar.axis = ?
        ORDER BY c.ts, c.comparison_id, ar.axis_rating_id
        """,
        (axis,),
    ).fetchall()

    for r in rows:
        a, b, outcome = r["span_a"], r["span_b"], r["outcome"]
        if a not in players or b not in players:
            continue
        pa, pb = players[a], players[b]
        # Capture pre-game ratings so both updates use the same period snapshot.
        ra, rda = pa.rating, pa.rd
        rb, rdb = pb.rating, pb.rd
        score_a = {"a": 1.0, "b": 0.0, "equal": 0.5}[outcome]
        pa.update_player([rb], [rdb], [score_a])
        pb.update_player([ra], [rda], [1.0 - score_a])
        stats[a]["games"] += 1
        stats[b]["games"] += 1
        if outcome == "a":
            stats[a]["wins"] += 1
            stats[b]["losses"] += 1
        elif outcome == "b":
            stats[b]["wins"] += 1
            stats[a]["losses"] += 1
        else:
            stats[a]["draws"] += 1
            stats[b]["draws"] += 1

    out = {}
    for s in pool:
        pl = players[s]
        out[s] = {
            "rating": pl.rating,
            "rd": pl.rd,
            "vol": pl.vol,
            **stats[s],
        }
    return out


def compute_all(
    conn: sqlite3.Connection,
    axes: list[str],
    scene: Optional[str] = None,
    operator: Optional[str] = None,
) -> dict[str, dict[str, dict]]:
    """Per-axis :func:`compute` over the same pool: {axis: {span_id: stats}}."""
    return {ax: compute(conn, ax, scene, operator) for ax in axes}
