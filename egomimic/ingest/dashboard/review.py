"""Staff review: approve or decline converted deliveries, one at a time or in bulk.

A decision names the stage (the exact converted result) the reviewer saw, so it only
ever applies to that result: the watcher commits an approved stage on its next pass, and
ignores decisions for a stage that has since been replaced. A decision is refused here
when the delivery changed after the page was loaded (re-uploaded, a correction under
way, re-converted), so staff never approve something they did not see.

    POST /d/<vendor>/<delivery_id>/decide   decision, stage, comment
    POST /decide                            decision, comment, item = vendor/id/stage ...

Each bulk item is checked on its own; one that changed is skipped and reported. The
newest decision for a stage wins, so a reviewer can change their mind until the watcher
acts. A decline needs a comment: the vendor sees it (not who wrote it).
"""

from __future__ import annotations

import re

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    redirect,
    request,
    url_for,
)

from egomimic.ingest.store import Ref, Store

MAX_COMMENT = 1000
MAX_BULK = 500
_ITEM = re.compile(r"([^/]+)/([^/]+)/([0-9a-f]{64})")

bp = Blueprint("review", __name__)


def _store() -> Store:
    return current_app.extensions["ingest_store"]


def _staff() -> None:
    if g.user is None:
        abort(401)
    if g.account["role"] != "staff":
        abort(404)  # vendors do not review; the page does not exist for them


def decide(ref: Ref, decision: str, stage: str, comment: str) -> str | None:
    """Record a decision on the stage the reviewer saw; the reason it was refused, or
    None. Checked against the bucket as it is now."""
    store = _store()
    status = store.get_status(ref) or {}
    current = store.get_delivery(ref)
    if status.get("state") != "ready" or status.get("stage") != stage:
        return "it is no longer waiting for review with the result you saw"
    if current is None or not current.complete:
        return "an upload to it is under way"
    if current.fingerprint != status.get("fingerprint"):
        return "it was re-uploaded since it was converted"
    store.add_decision(ref, decision, g.user, comment, stage)
    return None


def _form() -> tuple[str, str]:
    decision = request.form.get("decision", "")
    comment = request.form.get("comment", "").strip()
    if decision not in ("approve", "decline"):
        abort(400, "decision must be approve or decline")
    if len(comment) > MAX_COMMENT:
        abort(400, f"comments are at most {MAX_COMMENT} characters")
    return decision, comment


@bp.post("/d/<vendor>/<delivery_id>/decide")
def decide_one(vendor, delivery_id):
    _staff()
    try:
        ref = Ref(vendor, delivery_id)
    except ValueError:
        abort(404)
    decision, comment = _form()
    page = url_for("delivery", vendor=vendor, delivery_id=delivery_id)
    if decision == "decline" and not comment:
        flash("A decline needs a comment: the vendor sees it.", "error")
        return redirect(page)
    stage = request.form.get("stage", "")
    if not re.fullmatch(r"[0-9a-f]{64}", stage):
        abort(400, "stage missing")
    refused = decide(ref, decision, stage, comment)
    if refused:
        flash(f"Not recorded: {refused}. Reload and review it again.", "error")
    else:
        done = "Approved" if decision == "approve" else "Declined"
        flash(f"{done}; the next pipeline pass acts on it.", "ok")
    return redirect(page)


@bp.post("/decide")
def decide_many():
    _staff()
    decision, comment = _form()
    back = request.form.get("back", "")
    if not back.startswith("/") or back.startswith("//"):
        back = url_for("index")
    items = request.form.getlist("item")
    if not items:
        flash("Tick the deliveries to decide on first.", "error")
        return redirect(back)
    if len(items) > MAX_BULK:
        abort(400, f"at most {MAX_BULK} at once")
    if decision == "decline" and not comment:
        flash("A decline needs a comment: the vendors see it.", "error")
        return redirect(back)
    done, skipped = 0, []
    for item in items:
        m = _ITEM.fullmatch(item)
        try:
            ref = Ref(m.group(1), m.group(2)) if m else None
        except ValueError:
            ref = None
        if ref is None:
            abort(400, "malformed item")
        refused = decide(ref, decision, m.group(3), comment)
        if refused:
            skipped.append(f"{ref.vendor}/{ref.delivery_id} ({refused})")
        else:
            done += 1
    verb = "Approved" if decision == "approve" else "Declined"
    flash(f"{verb} {done}; the next pipeline pass acts on them.", "ok")
    if skipped:
        flash(f"Skipped {len(skipped)}, changed since the page loaded: "
              + "; ".join(skipped), "error")  # fmt: skip
    return redirect(back)
