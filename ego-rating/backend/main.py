"""FastAPI app for the ego-rating tool — pairwise multi-axis Glicko-2 ranking.

Raters are shown two spans whose annotations are similar (>= a cosine-similarity
threshold) and rate them A / Equal / B on a FIXED set of axes (from config.yaml);
each axis is an independent Glicko-2 ranking derived from the comparison log.
Pairs are served by a balanced-adaptive scheduler. The rankings are exported as
a reward-model training dataset (``/export/reward_dataset.jsonl``).

Run from the ego-rating/ directory:

    uvicorn backend.main:app --reload

Serves the SPA at ``/`` and (if present) video clips under ``/videos``.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import db, glicko, pairing, s3_resolver

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
VIDEOS_DIR = BASE_DIR / "videos"


def apply_config_params() -> None:
    """Push elo/similarity settings from config.yaml into the engines."""
    try:
        cfg = db.read_config()
    except Exception:
        return
    sim = cfg.get("similarity") or {}
    pr = cfg.get("pairing") or {}
    pairing.set_params(
        threshold=sim.get("threshold"),
        model=sim.get("model"),
        backend=sim.get("backend"),
        warmup=pr.get("warmup_comparisons"),
        epsilon=pr.get("epsilon"),
        prefer_diverse=pr.get("prefer_diverse"),
    )
    g = cfg.get("glicko") or {}
    glicko.set_params(
        initial=g.get("initial"), rd=g.get("rd"), vol=g.get("vol"), tau=g.get("tau")
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_app()  # create tables, read config.yaml, upsert spans
    apply_config_params()
    _warm_embeddings_in_background()
    yield


def _warm_embeddings_in_background() -> None:
    """Build the comparability graph (Modal embed) off the request path so the
    first comparison isn't a ~30 s wait on the Modal cold start."""
    import threading

    def _warm():
        try:
            conn = db.connect()
            try:
                print("[ego-rating] warming embedding/adjacency cache…", flush=True)
                pairing._build(conn)
                print("[ego-rating] embedding/adjacency cache warmed.", flush=True)
            finally:
                conn.close()
        except Exception as e:  # never block startup on warm-up
            print(f"[ego-rating] embedding warm-up skipped: {e}", flush=True)

    threading.Thread(target=_warm, daemon=True).start()


app = FastAPI(title="ego-rating", lifespan=lifespan)

# Permissive CORS — single-user dev tool; the SPA is same-origin anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CompareBody(BaseModel):
    span_a: str
    span_b: str
    rater_id: int
    # Either a full rating over every configured axis...
    ratings: dict[str, str] = {}  # {axis_name: "a" | "equal" | "b"}
    # ...or a skip (rater saw the pair but declined to judge).
    skip: bool = False


class WeightsBody(BaseModel):
    weights: dict[str, float]  # {axis_name: weight >= 0}; partial updates OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _span_payload(conn: sqlite3.Connection, span_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT span_id, video_uri, start, end, scene, operator, label "
        "FROM spans WHERE span_id = ?",
        (span_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "span_id": row["span_id"],
        "video_uri": row["video_uri"],
        "start": row["start"],
        "end": row["end"],
        "scene": row["scene"],
        "operator": row["operator"],
        "label": row["label"],
    }


# ---------------------------------------------------------------------------
# Config / metadata
# ---------------------------------------------------------------------------
@app.get("/config")
def get_config(conn: sqlite3.Connection = Depends(db.get_db)):
    scenes = [
        r["scene"]
        for r in conn.execute(
            "SELECT DISTINCT scene FROM spans ORDER BY scene"
        ).fetchall()
    ]
    operators = [
        r["operator"]
        for r in conn.execute(
            "SELECT DISTINCT operator FROM spans ORDER BY operator"
        ).fetchall()
    ]
    return {
        "annotation": db.get_annotation(),
        "axes": db.get_axes(),
        "scenes": scenes,
        "operators": operators,
        "similarity_threshold": pairing._threshold,
    }


# ---------------------------------------------------------------------------
# Comparison queue
# ---------------------------------------------------------------------------
@app.get("/next-pair")
def get_next_pair(
    rater_id: int,
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    """Next pair of similar spans for this rater to compare, or null when none
    remain (every eligible pair in the filter has been seen by this rater)."""
    pair = pairing.next_pair(conn, rater_id, scene, operator)
    if pair is None:
        return None
    a, b, sim, phase = pair
    return {
        "a": _span_payload(conn, a),
        "b": _span_payload(conn, b),
        "similarity": round(sim, 3),
        "phase": phase,
    }


@app.post("/compare")
def post_compare(body: CompareBody, conn: sqlite3.Connection = Depends(db.get_db)):
    if body.span_a == body.span_b:
        raise HTTPException(status_code=400, detail="span_a and span_b must differ")
    for sid in (body.span_a, body.span_b):
        if (
            conn.execute("SELECT 1 FROM spans WHERE span_id = ?", (sid,)).fetchone()
            is None
        ):
            raise HTTPException(status_code=404, detail=f"unknown span_id: {sid}")
    if not body.skip:
        # A rating must cover the configured axis set exactly.
        expected = set(db.axis_names())
        got = set(body.ratings)
        if got != expected:
            raise HTTPException(
                status_code=400,
                detail=f"ratings must cover exactly the configured axes "
                f"{sorted(expected)}; got {sorted(got)}",
            )
        bad = {a: v for a, v in body.ratings.items() if v not in ("a", "b", "equal")}
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"axis outcomes must be 'a', 'b', or 'equal'; got {bad}",
            )
        # Rated comparisons must be a comparable (similar) pair, so Glicko never
        # mixes across the similarity boundary. Skips are exempt (ignored by
        # Glicko).
        if not pairing.is_eligible(conn, body.span_a, body.span_b):
            raise HTTPException(
                status_code=400,
                detail="span_a and span_b are not a comparable (similar) pair",
            )
    db.ensure_rater(conn, body.rater_id)
    cur = conn.execute(
        "INSERT INTO comparisons (span_a, span_b, rater_id, skipped) VALUES (?, ?, ?, ?)",
        (body.span_a, body.span_b, body.rater_id, 1 if body.skip else 0),
    )
    if not body.skip:
        conn.executemany(
            "INSERT INTO axis_ratings (comparison_id, axis, outcome) VALUES (?, ?, ?)",
            [(cur.lastrowid, ax, out) for ax, out in body.ratings.items()],
        )
    conn.commit()
    return {"ok": True}


@app.delete("/comparisons/last")
def undo_last_comparison(rater_id: int, conn: sqlite3.Connection = Depends(db.get_db)):
    """Undo this rater's most recent submission (rated or skipped): delete it
    and return the pair so the UI can re-serve it, with the prior per-axis
    choices for prefill."""
    row = conn.execute(
        "SELECT comparison_id, span_a, span_b, skipped FROM comparisons "
        "WHERE rater_id = ? ORDER BY ts DESC, comparison_id DESC LIMIT 1",
        (rater_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="nothing to undo for this rater")
    prior = {
        r["axis"]: r["outcome"]
        for r in conn.execute(
            "SELECT axis, outcome FROM axis_ratings WHERE comparison_id = ?",
            (row["comparison_id"],),
        ).fetchall()
    }
    conn.execute(
        "DELETE FROM axis_ratings WHERE comparison_id = ?", (row["comparison_id"],)
    )
    conn.execute(
        "DELETE FROM comparisons WHERE comparison_id = ?", (row["comparison_id"],)
    )
    conn.commit()
    return {
        "a": _span_payload(conn, row["span_a"]),
        "b": _span_payload(conn, row["span_b"]),
        "similarity": round(
            pairing.pair_similarity(conn, row["span_a"], row["span_b"]), 3
        ),
        "phase": "undo",
        "was_skip": bool(row["skipped"]),
        "prior_ratings": prior,
    }


# In-process cache: episode annotation context is immutable per episode_hash.
_annotation_cache: dict[str, dict] = {}


@app.get("/annotation/{span_id}")
def get_annotation_context(span_id: str, conn: sqlite3.Connection = Depends(db.get_db)):
    """Everything we have to judge `annotation_accuracy` against: the episode's
    description (the span label), its **subtask annotations** (`segments`, from
    the legacy DB, cached on the span at resolve time), plus objects / scene /
    environment from the episode's MongoDB doc (best-effort — local spans just
    get the label)."""
    row = conn.execute(
        "SELECT label, video_uri, segments FROM spans WHERE span_id = ?", (span_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown span_id: {span_id}")
    try:
        segments = json.loads(row["segments"] or "[]")
    except (TypeError, ValueError):
        segments = []
    if span_id in _annotation_cache:
        # segments come from the (possibly re-resolved) DB row, not the cache.
        return {**_annotation_cache[span_id], "segments": segments}
    out = {
        "label": row["label"],
        "objects": [],
        "tools": [],
        "scene_desc": "",
        "environment_desc": "",
    }
    ref = row["video_uri"]
    if len(ref) == 24 and all(c in "0123456789abcdef" for c in ref.lower()):
        try:
            from bson import ObjectId

            doc = (
                s3_resolver._get_mongo_db()["episodes"].find_one(
                    {"_id": ObjectId(ref)},
                    ["objects", "scene_desc", "environment_desc", "tools"],
                )
                or {}
            )
            out["objects"] = [str(o) for o in (doc.get("objects") or [])]
            out["tools"] = [str(t) for t in (doc.get("tools") or [])]
            out["scene_desc"] = str(doc.get("scene_desc") or "")
            out["environment_desc"] = str(doc.get("environment_desc") or "")
        except Exception as e:  # annotation context is best-effort
            print(f"[ego-rating] annotation lookup failed for {span_id}: {e}")
    _annotation_cache[span_id] = out
    return {**out, "segments": segments}


@app.get("/progress")
def get_progress(
    rater_id: int,
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    return pairing.progress(conn, rater_id, scene, operator)


# ---------------------------------------------------------------------------
# Leaderboard (per-axis Glicko-2 ranking)
# ---------------------------------------------------------------------------
@app.get("/leaderboard")
def get_leaderboard(
    scene: Optional[str] = None,
    operator: Optional[str] = None,
    conn: sqlite3.Connection = Depends(db.get_db),
):
    return {
        "rows": pairing.leaderboard(conn, scene, operator),
        "axes": db.axis_names(),
        "weights": db.get_weights(conn),
        "glicko": glicko.params(),
    }


@app.put("/weights")
def put_weights(body: WeightsBody, conn: sqlite3.Connection = Depends(db.get_db)):
    """Adjust how much each axis contributes to the total score (persisted, so
    the leaderboard and the total-ranking export always agree)."""
    known = set(db.axis_names())
    unknown = set(body.weights) - known
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown axes {sorted(unknown)}; configured: {sorted(known)}",
        )
    bad = {
        a: w
        for a, w in body.weights.items()
        if not (w >= 0 and w == w and w != float("inf"))
    }
    if bad:
        raise HTTPException(
            status_code=400, detail=f"weights must be finite and >= 0: {bad}"
        )
    return {"weights": db.set_weights(conn, body.weights)}


def _jsonl_response(records: list[dict], filename: str) -> Response:
    content = "".join(json.dumps(r) + "\n" for r in records)
    return Response(
        content=content,
        media_type="application/jsonl",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/reward_dataset.jsonl")
def export_reward_dataset(conn: sqlite3.Connection = Depends(db.get_db)):
    """The reward-model training dataset: one JSON line per episode with a
    normalized per-axis `score` in (0,1) (Glicko expected win probability vs
    the episode's comparability-group average), plus rating/rd/games so the
    trainer can filter or uncertainty-weight. Always the FULL pool (no
    scene/operator filter) — scores are group-relative, not filter-relative."""
    return _jsonl_response(pairing.reward_dataset(conn), "reward_dataset.jsonl")


@app.get("/export/total_ranking.jsonl")
def export_total_ranking(conn: sqlite3.Connection = Depends(db.get_db)):
    """The cumulative episode ranking by weighted total score (1-10): one JSON
    line per episode, best first, computed under the current axis weights.
    Full pool, no filter."""
    return _jsonl_response(pairing.total_ranking(conn), "total_ranking.jsonl")


@app.get("/export/preferences.jsonl")
def export_preferences(conn: sqlite3.Connection = Depends(db.get_db)):
    """The raw pairwise preference log: one JSON line per rated comparison with
    the per-axis labels — the input format for Bradley–Terry-style preference
    training (and the source data the scores derive from). Skips excluded."""
    rows = conn.execute(
        """
        SELECT c.comparison_id, c.span_a, c.span_b, c.rater_id, c.ts,
               sa.video_uri AS video_a, sb.video_uri AS video_b,
               sa.label AS task_a, sb.label AS task_b
        FROM comparisons c
        LEFT JOIN spans sa ON sa.span_id = c.span_a
        LEFT JOIN spans sb ON sb.span_id = c.span_b
        WHERE c.skipped = 0
        ORDER BY c.ts, c.comparison_id
        """
    ).fetchall()
    labels = _axis_labels_by_comparison(conn)
    return _jsonl_response(
        [
            {
                "comparison_id": r["comparison_id"],
                "episode_a": r["span_a"],
                "episode_b": r["span_b"],
                "video_a": r["video_a"],
                "video_b": r["video_b"],
                "task_a": r["task_a"],
                "task_b": r["task_b"],
                "labels": labels.get(r["comparison_id"], {}),
                "rater_id": r["rater_id"],
                "ts": r["ts"],
            }
            for r in rows
        ],
        "preferences.jsonl",
    )


def _axis_labels_by_comparison(conn: sqlite3.Connection) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for r in conn.execute(
        "SELECT comparison_id, axis, outcome FROM axis_ratings ORDER BY axis_rating_id"
    ).fetchall():
        out.setdefault(r["comparison_id"], {})[r["axis"]] = r["outcome"]
    return out


# ---------------------------------------------------------------------------
# Video streaming (resolve a span's ref -> redirect to a streamable URL)
# ---------------------------------------------------------------------------
@app.get("/video/{span_id}")
async def get_video(
    span_id: str, request: Request, conn: sqlite3.Connection = Depends(db.get_db)
):
    """Stream the span's video through this server (same-origin).

    Remote refs (episode_hash / r2:// key / https) resolve to a presigned R2 URL;
    we proxy it, forwarding the browser's Range header so only the watched bytes
    are transferred (no full download, server- or client-side). Streaming through
    our origin avoids the cross-origin redirect that some browsers refuse to play.
    Local refs redirect to the range-capable /videos mount.
    """
    row = conn.execute(
        "SELECT video_uri FROM spans WHERE span_id = ?", (span_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown span_id: {span_id}")
    ref = row["video_uri"]
    try:
        url = s3_resolver.resolve_to_url(ref)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"could not resolve video for {span_id}: {e}"
        )
    if url is None:
        # Local file — the static /videos mount already supports Range.
        return RedirectResponse("/" + ref.lstrip("/"), status_code=307)

    fwd = {}
    if request.headers.get("range"):
        fwd["Range"] = request.headers["range"]
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, read=None), follow_redirects=True
    )
    try:
        upstream = await client.send(
            client.build_request("GET", url, headers=fwd), stream=True
        )
    except Exception as e:
        await client.aclose()  # don't leak the client if the upstream fetch fails
        raise HTTPException(status_code=502, detail=f"video upstream fetch failed: {e}")
    keep = (
        "content-type",
        "content-length",
        "content-range",
        "accept-ranges",
        "etag",
        "last-modified",
        "cache-control",
    )
    headers = {k: upstream.headers[k] for k in keep if k in upstream.headers}

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code, headers=headers)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.get("/raw-comparisons")
def get_raw_comparisons(conn: sqlite3.Connection = Depends(db.get_db)):
    """All comparison rows for the admin screen (most recent first), with each
    comparison's per-axis outcomes."""
    rows = conn.execute(
        """
        SELECT c.comparison_id, c.span_a, c.span_b, c.rater_id, c.skipped, c.ts,
               sa.label AS label_a, sb.label AS label_b
        FROM comparisons c
        LEFT JOIN spans sa ON sa.span_id = c.span_a
        LEFT JOIN spans sb ON sb.span_id = c.span_b
        ORDER BY c.ts DESC, c.comparison_id DESC
        """
    ).fetchall()
    labels = _axis_labels_by_comparison(conn)
    return {
        "rows": [
            {**dict(r), "ratings": labels.get(r["comparison_id"], {})} for r in rows
        ]
    }


@app.post("/reload-config")
def post_reload_config(
    force: bool = False, conn: sqlite3.Connection = Depends(db.get_db)
):
    """Re-read config.yaml and re-apply annotation/axes/params. The span pool
    is only re-resolved if the config's selection block changed — pass
    ``?force=true`` to re-sample explicitly (with shuffle/limit this draws a
    NEW random pool and drops comparisons of episodes that fall out of it)."""
    try:
        n = db.load_and_upsert_spans(conn, force=force)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    apply_config_params()
    return {"ok": True, "span_count": n}


# ---------------------------------------------------------------------------
# Static SPA + videos (registered last so API routes take precedence)
# ---------------------------------------------------------------------------
@app.get("/favicon.ico")
def favicon():
    # The page sets an inline SVG favicon; answer the browser's probe with 204
    # so it never 404s in the logs.
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if VIDEOS_DIR.exists():
    app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")
