"""Uploads: a vendor sends a delivery's files straight to the bucket, in parts, through
presigned links the dashboard hands out for that vendor's own folder only. No bucket
credentials leave the lab, and the file bytes never pass through the dashboard (except
on the local-directory store, which has no presigning: there parts come through here).

The JSON API, shared by the upload page (upload.js) and the CLI (egomimic.ingest.upload);
every POST carries the session's CSRF token in X-CSRF-Token:

    POST /api/login                      {user, password} -> {csrf}
    GET  /api/session                    -> {user, role, vendor, csrf}
    GET  /api/deliveries/<id>            -> {state, files, uploads}
    POST /api/deliveries/<id>/uploads    {name, size} -> {upload_id, part_size, parts}
    POST /api/deliveries/<id>/urls       {name, upload_id, parts: [n]} -> {urls: {n: url}}
    PUT  /api/deliveries/<id>/part       ?name&upload_id&part (local store only) -> {etag}
    POST /api/deliveries/<id>/finish     {name, upload_id, parts: [[n, etag]]}
    POST /api/deliveries/<id>/abort      {name, upload_id}
    POST /api/deliveries/<id>/clear      {record}: an abandoned upload
    POST /api/deliveries/<id>/complete   {files: [names]}: the delivery is exactly these

A delivery is processed only once marked complete; marking it complete with a file list
deletes any other file in it, so a correction is a full re-upload of the folder.
"""

from __future__ import annotations

import math

from flask import Blueprint, abort, current_app, g, jsonify, request, session

from egomimic.ingest import formats
from egomimic.ingest.dashboard import views
from egomimic.ingest.store import (
    Ref,
    Store,
    UploadRejected,
    UploadsInProgress,
    _file_name,
    part_length,
)

MIN_PART = (
    16 * 1024**2
)  # R2: parts of one upload are equal-sized, >= 5 MiB but the last
MAX_PARTS = 10_000
URLS_PER_CALL = 100
URL_EXPIRES_S = 3600
MAX_FILES = 1000
JSON_LIMIT = 2 * 1024**2  # a finish call lists up to 10k (part, etag) pairs

bp = Blueprint("uploads", __name__)


def _store() -> Store:
    return current_app.extensions["ingest_store"]


def _fail(status: int, message: str, **extra):
    resp = jsonify({"error": message, **extra})
    resp.status_code = status
    abort(resp)


def _body() -> dict:
    request.max_content_length = JSON_LIMIT
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        _fail(400, "expected a JSON object")
    return body


def _ref(delivery_id: str) -> Ref:
    """The signed-in vendor's own delivery; staff do not upload."""
    if g.user is None:
        _fail(401, "log in first")
    if g.account["role"] != "vendor":
        _fail(403, "only vendor accounts upload")
    try:
        return Ref(g.account["vendor"], delivery_id)
    except ValueError as e:
        _fail(400, str(e))


def _name(body: dict) -> str:
    try:
        return _file_name(str(body.get("name", "")))
    except ValueError as e:
        _fail(400, str(e))


def _upload(ref: Ref, body: dict) -> tuple[str, str, dict]:
    """(name, upload id, record) of an unfinished upload to this delivery, as recorded
    when it started: an id from another delivery or file is refused. Every upload
    started here is size-bound (size, part size, part count in the record)."""
    name, upload_id = _name(body), str(body.get("upload_id", ""))
    rec = _store().upload_record(ref, upload_id) if upload_id else None
    if not rec or rec.get("name") != name or rec.get("size") is None:
        _fail(404, "no such upload in progress")
    return name, upload_id, rec


def _length(rec: dict, n) -> int:
    if not isinstance(n, int):
        _fail(400, "part numbers are integers")
    try:
        return part_length(rec, n)
    except UploadRejected as e:
        _fail(400, str(e))


def accepted() -> tuple[frozenset[str], int]:
    names = frozenset().union(*(f.allowed for f in formats.formats()))
    return names, max(f.max_file_bytes for f in formats.formats())


def part_size(size: int) -> int:
    """Equal parts (R2 requires it), at most MAX_PARTS of them, in whole MiB."""
    need = math.ceil(size / MAX_PARTS)
    if need <= MIN_PART:
        return MIN_PART
    return math.ceil(need / 1024**2) * 1024**2


@bp.post("/api/login")
def api_login():
    """Log in for the CLI: the same checks and throttling as the login form."""
    if not request.is_json:  # a cross-site form cannot send JSON without a preflight
        _fail(415, "send JSON")
    body = _body()
    ok, error, status = current_app.extensions["ingest_login"](
        str(body.get("user", "")), str(body.get("password", ""))
    )
    if not ok:
        _fail(status, error)
    return jsonify({"csrf": session["csrf"]})


@bp.get("/api/session")
def api_session():
    if g.user is None:
        _fail(401, "log in first")
    a = g.account
    return jsonify(
        {
            "user": g.user,
            "role": a["role"],
            "vendor": a["vendor"],
            "csrf": session["csrf"],
        }
    )


@bp.get("/api/deliveries/<delivery_id>")
def api_delivery(delivery_id):
    ref = _ref(delivery_id)
    store = _store()
    r = views.row(store, ref)
    return jsonify(
        {
            "vendor": ref.vendor,
            "state": r.state if r else "new",
            "files": {n: size for n, (size, _) in (r.files if r else {}).items()},
            "uploads": [
                {k: u.get(k) for k in ("record", "name", "state", "started_at")}
                for u in store.uploads_in_progress(ref)
            ],
        }
    )


@bp.post("/api/deliveries/<delivery_id>/uploads")
def api_start(delivery_id):
    ref = _ref(delivery_id)
    body = _body()
    name = _name(body)
    size = body.get("size")
    if not isinstance(size, int) or size < 0:
        _fail(400, "size must be the file's size in bytes")
    allowed, limit = accepted()
    if name not in allowed:
        _fail(422, f"{name}: no accepted format has this file; expected one of "
                   f"{', '.join(sorted(allowed))}")  # fmt: skip
    if size > limit:
        _fail(422, f"{name}: {size} bytes is over the {limit / 1024**3:.0f} GiB limit")
    ps = part_size(size)
    store = _store()
    upload_id = store.start_upload(ref, name, size=size, part_size=ps)
    rec = store.upload_record(ref, upload_id)
    return jsonify(
        {"upload_id": upload_id, "part_size": rec["part_size"], "parts": rec["parts"]}
    )


@bp.post("/api/deliveries/<delivery_id>/urls")
def api_urls(delivery_id):
    ref = _ref(delivery_id)
    body = _body()
    name, upload_id, rec = _upload(ref, body)
    parts = body.get("parts")
    if not isinstance(parts, list) or not 0 < len(parts) <= URLS_PER_CALL:
        _fail(400, f"parts: 1-{URLS_PER_CALL} part numbers")
    lengths = {n: _length(rec, n) for n in parts}  # only 1..the declared part count
    store = _store()
    urls = {
        str(n): store.part_url(ref, name, upload_id, n, URL_EXPIRES_S, length)
        for n, length in lengths.items()
    }  # each signed for its exact length
    return jsonify({"urls": urls})  # null: PUT the part to /part (local store)


@bp.put("/api/deliveries/<delivery_id>/part")
def api_part(delivery_id):
    """Parts through the dashboard: only for the local store, which cannot presign."""
    ref = _ref(delivery_id)
    store = _store()
    args = {
        "name": request.args.get("name", ""),
        "upload_id": request.args.get("upload_id", ""),
    }
    name, upload_id, rec = _upload(ref, args)
    if store.part_url(ref, name, upload_id, 1) is not None:
        _fail(404, "parts go to the presigned links")
    try:
        part = int(request.args.get("part", ""))
    except ValueError:
        _fail(400, "part must be a number")
    length = _length(rec, part)
    request.max_content_length = rec["part_size"]
    data = request.get_data()
    if len(data) != length:
        _fail(400, f"part {part} must be {length} bytes, got {len(data)}")
    etag = store.upload_part(ref, name, upload_id, part, data)
    return jsonify({"etag": etag})


@bp.post("/api/deliveries/<delivery_id>/finish")
def api_finish(delivery_id):
    ref = _ref(delivery_id)
    body = _body()
    name, upload_id, rec = _upload(ref, body)
    parts = body.get("parts")
    try:
        parts = [(int(n), str(etag)) for n, etag in parts]
    except (TypeError, ValueError):
        _fail(400, "parts: [[part number, etag], ...]")
    try:
        _store().finish_upload(ref, name, upload_id, sorted(parts))
    except UploadRejected as e:  # wrong parts, or the joined file is not the size
        _fail(422, str(e))
    except Exception as e:  # a missing or mismatched part: the client retries or aborts
        _fail(409, f"could not join the parts of {name}: {type(e).__name__}")
    return jsonify({"name": name})


@bp.post("/api/deliveries/<delivery_id>/abort")
def api_abort(delivery_id):
    ref = _ref(delivery_id)
    name, upload_id, _ = _upload(ref, _body())
    _store().abort_upload(ref, name, upload_id)
    return jsonify({"name": name})


@bp.post("/api/deliveries/<delivery_id>/clear")
def api_clear(delivery_id):
    """Drop an abandoned upload (a closed tab, a crashed CLI) so the delivery can be
    completed again."""
    ref = _ref(delivery_id)
    record = str(_body().get("record", ""))
    store = _store()
    if record not in {u["record"] for u in store.uploads_in_progress(ref)}:
        _fail(404, "no such upload in progress")
    store.clear_upload(ref, record)
    return jsonify({"cleared": record})


@bp.post("/api/deliveries/<delivery_id>/complete")
def api_complete(delivery_id):
    ref = _ref(delivery_id)
    files = _body().get("files")
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        _fail(400, f"files: the delivery's file names (1-{MAX_FILES})")
    try:
        names = {_file_name(str(f)) for f in files}
    except ValueError as e:
        _fail(400, str(e))
    try:
        fmt = formats.detect(names)
    except formats.AmbiguousFormat as e:
        _fail(422, str(e))
    if fmt is None or fmt.unexpected(names):
        _fail(422, "these files are not a delivery in any accepted format",
              formats=[{"name": f.name, "required": sorted(f.required),
                        "optional": sorted(f.optional)} for f in formats.formats()])  # fmt: skip
    store = _store()
    try:
        done = store.mark_complete(ref, {"by": g.user}, sorted(names))
    except UploadsInProgress as e:
        _fail(409, "uploads to this delivery are still in progress",
              uploads=[u["record"] for u in e.records])  # fmt: skip
    except ValueError as e:  # a listed file was never uploaded
        _fail(422, str(e))
    return jsonify({"files": {n: size for n, (size, _) in done.items()}})
