"""Comparability grouping (semantic similarity) + balanced-adaptive pairing.

Two spans are eligible to be compared when the cosine similarity of their
annotation (`label` = the episode's task_description) embeddings is >= a
threshold (default 0.8). That defines an adjacency graph over the spans; its
connected components are the comparability "groups" (Elo is only meaningful
within a group, since comparisons never cross the 0.8 boundary).

Pairing is **balanced adaptive**: among eligible, not-yet-seen-by-this-rater
pairs in the current filter pool, serve
  1. pairs never globally compared before (round-robin coverage), then
  2. the least-compared spans (balance), then
  3. the closest current rating (most informative), averaged across the fixed
     axes (each axis is an independent Glicko-2 system fed by the same pair).

Embeddings use sentence-transformers (all-MiniLM-L6-v2) with a TF-IDF fallback.
"""

from __future__ import annotations

import random
import sqlite3
import threading
from typing import Optional

import numpy as np

from backend import db, glicko

DEFAULT_THRESHOLD = 0.8
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODAL_APP = "ego-rating-embed"

# Active-learning pairing (validated against the literature — ASAP / BALD /
# TrueSkill match-quality / Chatbot-Arena CI-reduction):
#   warm-up: the rater's first `_warmup` comparisons use most-similar-first
#            pairing (build intuition + seed the ratings);
#   active:  thereafter pick the pair maximising expected information gain,
#            approximated by score = (rd_a + rd_b) · p·(1−p) — combines high
#            uncertainty (rating deviation) with maximal outcome entropy
#            (p≈0.5 ⇔ closest rating ⇔ peak Fisher information). ε-greedy adds a
#            little random exploration so no span is starved.
DEFAULT_WARMUP = 10
DEFAULT_EPSILON = 0.1

_threshold = DEFAULT_THRESHOLD
_model_name = DEFAULT_MODEL
# Embedding backend: "modal" (remote, default — local torch is too heavy here),
# "local" (in-process sentence-transformers), "tfidf" (lexical fallback),
# or "auto" (modal, falling back to tfidf).
_backend = "modal"
_warmup = DEFAULT_WARMUP
_epsilon = DEFAULT_EPSILON
# Prefer pairs from different operators/scenes — same-operator pairs tend to be
# near-identical on every axis (rater feedback), so they're served last.
_prefer_diverse = True
_model = None
_lock = threading.Lock()
# Cache the adjacency/groups keyed on the span set + labels + threshold + backend.
_cache: dict = {"key": None}


def set_params(
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    warmup: Optional[int] = None,
    epsilon: Optional[float] = None,
    prefer_diverse: Optional[bool] = None,
) -> None:
    global _threshold, _model_name, _backend, _warmup, _epsilon, _prefer_diverse
    if threshold is not None:
        _threshold = float(threshold)
    if model:
        _model_name = model
    if backend:
        _backend = backend
    if warmup is not None:
        _warmup = int(warmup)
    if epsilon is not None:
        _epsilon = float(epsilon)
    if prefer_diverse is not None:
        _prefer_diverse = bool(prefer_diverse)
    _cache["key"] = None  # invalidate


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings (n, d) using the configured backend.

    Default is Modal (this box can't load torch in-process); on any failure we
    fall back to a local TF-IDF embedding so grouping still works (lexically).
    """
    if _backend in ("modal", "auto"):
        try:
            return _embed_modal(texts)
        except Exception as exc:
            print(f"[ego-rating] Modal embed unavailable ({exc}); using TF-IDF.")
            return _embed_tfidf(texts)
    if _backend == "local":
        try:
            return _embed_local(texts)
        except Exception as exc:
            print(f"[ego-rating] local embed unavailable ({exc}); using TF-IDF.")
            return _embed_tfidf(texts)
    return _embed_tfidf(texts)


def _embed_modal(texts: list[str]) -> np.ndarray:
    import modal

    embedder = modal.Cls.from_name(MODAL_APP, "Embedder")
    vecs = embedder().embed.remote(list(texts))
    return np.asarray(vecs, dtype=float)


def _embed_local(texts: list[str]) -> np.ndarray:
    global _model
    from sentence_transformers import SentenceTransformer

    if _model is None:
        _model = SentenceTransformer(_model_name)
    return np.asarray(
        _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
        dtype=float,
    )


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        mat = TfidfVectorizer().fit_transform([t or "" for t in texts]).toarray()
    except ValueError:
        # All-empty / stopword-only pool -> no shared vocabulary. Treat each label
        # as unique (identity) so grouping degrades to "no eligible pairs" rather
        # than raising (keeps the "always falls back" guarantee).
        return np.eye(len(texts), dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ---------------------------------------------------------------------------
# Adjacency + connected-component groups (cached)
# ---------------------------------------------------------------------------
def _build(conn: sqlite3.Connection) -> dict:
    """Return a consistent snapshot {adj, sims, groups, sizes}, rebuilding the
    cache under `_lock` on a miss. Returning a fresh dict (not the shared
    `_cache`) means a concurrent rebuild can't tear a caller's view, and holding
    the lock across the embed serialises concurrent cold builds (no double-embed).
    """
    rows = conn.execute("SELECT span_id, label FROM spans ORDER BY span_id").fetchall()
    ids = [r["span_id"] for r in rows]
    labels = [r["label"] or "" for r in rows]
    key = (tuple(ids), tuple(labels), _threshold, _model_name, _backend)

    with _lock:
        if _cache.get("key") != key:
            adj: dict[str, set] = {i: set() for i in ids}
            sims: dict[frozenset, float] = {}
            if len(ids) >= 2:
                emb = _embed(labels)
                sim = emb @ emb.T
                n = len(ids)
                for i in range(n):
                    for j in range(i + 1, n):
                        if sim[i, j] >= _threshold:
                            adj[ids[i]].add(ids[j])
                            adj[ids[j]].add(ids[i])
                            sims[frozenset((ids[i], ids[j]))] = float(sim[i, j])
            groups, sizes = _components(ids, adj)
            _cache["adj"] = adj
            _cache["sims"] = sims
            _cache["groups"] = groups
            _cache["sizes"] = sizes
            _cache["key"] = key  # set last: a reader seeing this key sees full data
        return {
            "adj": _cache["adj"],
            "sims": _cache["sims"],
            "groups": _cache["groups"],
            "sizes": _cache["sizes"],
        }


def _components(ids: list[str], adj: dict[str, set]) -> tuple[dict, dict]:
    """Union-find over the adjacency -> {span: group_id}, {group_id: size}."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in ids:
        for b in adj[a]:
            union(a, b)

    roots: dict[str, int] = {}
    groups: dict[str, int] = {}
    for i in ids:
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        groups[i] = roots[r]
    sizes: dict[int, int] = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    return groups, sizes


def get_groups(conn: sqlite3.Connection) -> dict[str, int]:
    return dict(_build(conn)["groups"])


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def _pool_edges(conn, scene, operator, adj) -> tuple[list[str], set]:
    """Eligible (similar) pairs whose both spans are in the filter pool.

    `adj` is passed in (from one `_build` snapshot) so a caller's edges and sims
    always come from the same adjacency build.
    """
    where, p = db.span_filter(scene, operator)
    pool = [
        r["span_id"]
        for r in conn.execute(f"SELECT span_id FROM spans{where}", p).fetchall()
    ]
    poolset = set(pool)
    edges = set()
    for a in pool:
        for b in adj.get(a, ()):
            if b in poolset:
                edges.add(frozenset((a, b)))
    return pool, edges


def _rater_pairs(conn, rater_id) -> tuple[set, set]:
    """This rater's (rated_pairs, skipped_only_pairs). Rated pairs are never
    re-served; pairs the rater only ever skipped come back as a last-resort
    tier (skipping is 'not now', not 'never')."""
    rated, skipped = set(), set()
    for r in conn.execute(
        "SELECT span_a, span_b, skipped FROM comparisons WHERE rater_id = ?",
        (rater_id,),
    ).fetchall():
        pr = frozenset((r["span_a"], r["span_b"]))
        (skipped if r["skipped"] else rated).add(pr)
    return rated, skipped - rated


def _rater_decided_count(conn, rater_id) -> int:
    return conn.execute(
        "SELECT count(*) FROM comparisons WHERE rater_id = ? AND skipped = 0",
        (rater_id,),
    ).fetchone()[0]


def next_pair(
    conn: sqlite3.Connection,
    rater_id: int,
    scene: Optional[str] = None,
    operator: Optional[str] = None,
) -> Optional[tuple[str, str, float, str]]:
    """Pick the next (span_a, span_b, similarity, phase) for this rater, or None.

    Two phases (see module docstring):
      * "warmup"  — the rater's first `_warmup` comparisons: most-similar-first.
      * "active"  — afterwards: maximise the mean over axes of
                    (rd_a+rd_b)·p·(1−p) over eligible pairs, with ε-greedy
                    exploration.
    Always restricted to similarity-eligible pairs the rater hasn't rated.
    Pairs the rater skipped return only after the unseen ones run out; with
    `prefer_diverse`, different-operator/scene pairs are served first.
    """
    cache = _build(conn)
    sims = cache["sims"]
    pool, edges = _pool_edges(conn, scene, operator, cache["adj"])
    rated, skipped = _rater_pairs(conn, rater_id)
    candidates = [e for e in edges if e not in rated and e not in skipped]
    if not candidates:
        # Everything unseen is exhausted — resurface pairs the rater skipped.
        candidates = [e for e in edges if e not in rated]
    if not candidates:
        return None
    candidates = _prefer_diverse_tier(conn, candidates)

    axes = db.axis_names()
    stats_by_axis = glicko.compute_all(conn, axes, scene, operator)
    phase = "active" if _rater_decided_count(conn, rater_id) >= _warmup else "warmup"

    def total_games(s: str) -> int:
        return sum(stats_by_axis[ax].get(s, {}).get("games", 0) for ax in axes)

    if phase == "warmup":
        pair_decided: dict = {}
        for r in conn.execute(
            "SELECT span_a, span_b FROM comparisons WHERE skipped = 0"
        ).fetchall():
            pr = frozenset((r["span_a"], r["span_b"]))
            pair_decided[pr] = pair_decided.get(pr, 0) + 1

        def warmup_key(e):
            a, b = tuple(e)
            return (
                1 if pair_decided.get(e, 0) > 0 else 0,  # never-compared pairs first
                -round(sims.get(e, 0.0), 2),  # most-similar first
                total_games(a) + total_games(b),  # then least-compared spans
            )

        best = min(candidates, key=warmup_key)
    elif random.random() < _epsilon:
        best = random.choice(candidates)  # exploration
    else:

        def info_score(e):
            a, b = tuple(e)
            # Mean expected information gain across the fixed axes (each axis is
            # an independent rating system fed by the same comparison).
            eig = 0.0
            for ax in axes:
                sa, sb = stats_by_axis[ax][a], stats_by_axis[ax][b]
                p = glicko.expected_score(
                    sa["rating"], sa["rd"], sb["rating"], sb["rd"]
                )
                eig += (sa["rd"] + sb["rd"]) * p * (1.0 - p)
            eig /= max(len(axes), 1)
            # Tie-break (matters only at the all-equal cold start): cover the
            # least-seen spans first.
            return (eig, -(total_games(a) + total_games(b)))

        best = max(candidates, key=info_score)

    sim = sims.get(best, 0.0)
    a, b = tuple(best)
    if random.random() < 0.5:  # de-bias which side is "left"
        a, b = b, a
    return a, b, sim, phase


def _prefer_diverse_tier(conn, candidates: list) -> list:
    """Rater feedback: same-operator pairs are near-identical on every axis.
    Serve the most diverse non-empty tier: different operator AND scene, then
    different operator, then everything."""
    if not _prefer_diverse:
        return candidates
    meta = {
        r["span_id"]: (r["operator"], r["scene"])
        for r in conn.execute("SELECT span_id, operator, scene FROM spans").fetchall()
    }

    def tier(e):
        a, b = tuple(e)
        (op_a, sc_a), (op_b, sc_b) = meta[a], meta[b]
        if op_a != op_b and sc_a != sc_b:
            return 0
        if op_a != op_b:
            return 1
        return 2

    best = min(tier(e) for e in candidates)
    return [e for e in candidates if tier(e) == best]


def progress(conn, rater_id, scene=None, operator=None) -> dict:
    """How many eligible pairs this rater has RATED, of the pool's total
    (skips don't count — skipped pairs come back)."""
    _, edges = _pool_edges(conn, scene, operator, _build(conn)["adj"])
    rated, _ = _rater_pairs(conn, rater_id)
    done = sum(1 for e in edges if e in rated)
    return {"done": done, "total": len(edges)}


def is_eligible(conn, a: str, b: str) -> bool:
    """Whether (a, b) is a comparable (similarity ≥ threshold) pair."""
    return b in _build(conn)["adj"].get(a, set())


def pair_similarity(conn, a: str, b: str) -> float:
    """Cosine similarity of an eligible pair's labels (0.0 if not eligible)."""
    return float(_build(conn)["sims"].get(frozenset((a, b)), 0.0))


# ---------------------------------------------------------------------------
# Leaderboard (per-axis Glicko ranking within comparability groups) + the
# reward-model dataset export
# ---------------------------------------------------------------------------
def _group_scores(
    ids: list[str], groups: dict[str, int], stats: dict[str, dict]
) -> dict[str, float]:
    """Normalized score per span for ONE axis: the Glicko expected win
    probability against that span's comparability-group average (mean rating /
    mean RD over the group members in `ids`). Bounded (0,1), monotone in
    rating, and only ever group-relative — matching the invariant that
    comparisons never cross a group boundary. Callers rescale to 1-10."""
    gp = glicko.params()
    by_group: dict[int, list[str]] = {}
    for s in ids:
        by_group.setdefault(groups.get(s, -1), []).append(s)
    means: dict[int, tuple[float, float]] = {}
    for g, members in by_group.items():
        rs = [stats.get(s, {}).get("rating", gp["initial"]) for s in members]
        rds = [stats.get(s, {}).get("rd", gp["rd"]) for s in members]
        means[g] = (sum(rs) / len(rs), sum(rds) / len(rds))
    out = {}
    for s in ids:
        st = stats.get(s, {})
        r, rd = st.get("rating", gp["initial"]), st.get("rd", gp["rd"])
        mr, mrd = means[groups.get(s, -1)]
        out[s] = glicko.expected_score(r, rd, mr, mrd)
    return out


def _axis_rows(conn, scene=None, operator=None) -> list[dict]:
    """Shared span rows for leaderboard/export: per-axis stats + scores."""
    cache = _build(conn)
    groups, sizes = cache["groups"], cache["sizes"]
    axes = db.axis_names()
    weights = db.get_weights(conn)
    stats_by_axis = glicko.compute_all(conn, axes, scene, operator)
    gp = glicko.params()
    where, p = db.span_filter(scene, operator)
    rows = conn.execute(
        f"SELECT span_id, video_uri, start, end, scene, operator, label FROM spans{where}",
        p,
    ).fetchall()
    ids = [r["span_id"] for r in rows]
    scores_by_axis = {ax: _group_scores(ids, groups, stats_by_axis[ax]) for ax in axes}

    default = {
        "rating": gp["initial"],
        "rd": gp["rd"],
        "vol": gp["vol"],
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }
    out = []
    for r in rows:
        sid = r["span_id"]
        ax_stats = {}
        for ax in axes:
            st = stats_by_axis[ax].get(sid, default)
            g = st["games"]
            ax_stats[ax] = {
                "rating": round(st["rating"], 1),
                "rd": round(st["rd"], 1),
                "vol": round(st["vol"], 4),
                # Conservative lower-bound estimate (rating − 2·RD), à la
                # TrueSkill ordinal — useful when RDs differ a lot.
                "conservative": round(st["rating"] - 2.0 * st["rd"], 1),
                "games": g,
                "wins": st["wins"],
                "losses": st["losses"],
                "draws": st["draws"],
                "win_rate": (st["wins"] / g) if g else None,
                # 1-10 scale: 1 + 9 * P(beat the group average); 5.5 = average.
                "score": round(1.0 + 9.0 * scores_by_axis[ax][sid], 2),
            }
        grp = groups.get(sid, -1)
        out.append(
            {
                "span_id": sid,
                "video_uri": r["video_uri"],
                "start": r["start"],
                "end": r["end"],
                "scene": r["scene"],
                "operator": r["operator"],
                "label": r["label"],
                "group": grp,
                "group_size": sizes.get(grp, 1),
                "axes": ax_stats,
                "total_score": weighted_total(ax_stats, weights),
                "games": sum(a["games"] for a in ax_stats.values()),
            }
        )
    return out


def weighted_total(ax_stats: dict[str, dict], weights: dict[str, float]) -> float:
    """Weighted mean of the 1-10 axis scores. All-zero weights degrade to the
    plain mean (a total of 0 would be meaningless — scores start at 1)."""
    w = {ax: weights.get(ax, 1.0) for ax in ax_stats}
    total_w = sum(w.values())
    if total_w <= 0:
        w = {ax: 1.0 for ax in ax_stats}
        total_w = float(len(ax_stats)) or 1.0
    return round(sum(st["score"] * w[ax] for ax, st in ax_stats.items()) / total_w, 2)


def leaderboard(conn, scene=None, operator=None) -> list[dict]:
    out = _axis_rows(conn, scene, operator)
    # Rank by weighted total score desc; ties → more games first (more evidence).
    out.sort(key=lambda x: (x["total_score"], x["games"]), reverse=True)
    return out


def reward_dataset(conn) -> list[dict]:
    """One record per episode, over the FULL span pool (no filter): the
    reward-model training targets. Keys the trainer needs: episode/video ref,
    task text, per-axis normalized `score` on 1-10 (+ rating/rd/games for
    filtering or uncertainty weighting), and the comparability group the
    scores are relative to."""
    weights = db.get_weights(conn)
    records = []
    for r in _axis_rows(conn):
        records.append(
            {
                "episode_hash": r["span_id"],
                "video": r["video_uri"],
                "start": r["start"],
                "end": r["end"],
                "task_description": r["label"],
                "scene": r["scene"],
                "operator": r["operator"],
                "group": r["group"],
                "group_size": r["group_size"],
                "axes": {
                    ax: {
                        "score": st["score"],
                        "rating": st["rating"],
                        "rd": st["rd"],
                        "games": st["games"],
                        "wins": st["wins"],
                        "losses": st["losses"],
                        "draws": st["draws"],
                    }
                    for ax, st in r["axes"].items()
                },
                "total_score": r["total_score"],
                "weights": weights,
                "num_comparisons": r["games"] // max(len(r["axes"]), 1),
            }
        )
    return records


def total_ranking(conn) -> list[dict]:
    """The cumulative episode ranking by weighted total score, over the FULL
    span pool: one record per episode, best first, with the weights it was
    computed under and the per-axis 1-10 scores the total combines."""
    weights = db.get_weights(conn)
    rows = _axis_rows(conn)
    rows.sort(key=lambda x: (x["total_score"], x["games"]), reverse=True)
    return [
        {
            "rank": i,
            "episode_hash": r["span_id"],
            "video": r["video_uri"],
            "task_description": r["label"],
            "scene": r["scene"],
            "operator": r["operator"],
            "group": r["group"],
            "group_size": r["group_size"],
            "total_score": r["total_score"],
            "weights": weights,
            "axis_scores": {ax: st["score"] for ax, st in r["axes"].items()},
            "num_comparisons": r["games"] // max(len(r["axes"]), 1),
        }
        for i, r in enumerate(rows, 1)
    ]
