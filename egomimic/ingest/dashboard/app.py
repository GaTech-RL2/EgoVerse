"""The ingestion dashboard: vendors follow their own deliveries, staff see every vendor.

    python -m egomimic.ingest.dashboard [--local-store DIR | --bucket rldb --root PREFIX]

Configuration comes from the environment, so it runs the same locally and on AWS:
EGOVERSE_DASHBOARD_SECRET (required: signs session cookies), R2_* credentials (read
from ~/.egoverse_env when present), EGOVERSE_DASHBOARD_INSECURE=1 to allow cookies
over plain http when running on localhost, and EGOVERSE_DASHBOARD_PROXIES=<n> when n
trusted proxies (the AWS load balancer) forward requests, so login throttling sees the
client's address rather than the proxy's.

The dashboard only reads and writes the bucket (accounts, statuses, later uploads and
decisions); the watcher on SkyNet does the work.
"""

from __future__ import annotations

import hmac
import os
import secrets
import time
from datetime import timedelta
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from jinja2 import DictLoader
from werkzeug.exceptions import HTTPException

from egomimic.ingest import accounts, formats
from egomimic.ingest.dashboard import review, templates, uploads, views
from egomimic.ingest.dashboard.throttle import Throttle
from egomimic.ingest.store import Ref, Store

ACCOUNTS_TTL = 10.0  # seconds an accounts read is reused across requests


def create_app(
    store: Store,
    secret_key: str,
    *,
    secure_cookies: bool = True,
    proxies: int = 0,
    throttle: Throttle | None = None,
) -> Flask:
    """`proxies`: how many trusted proxies (e.g. the AWS load balancer) sit in front,
    so the client address is taken from their X-Forwarded-For; 0 trusts none."""
    if not secret_key or len(secret_key) < 32:
        raise ValueError("the session secret needs at least 32 characters")
    app = Flask(__name__)
    app.jinja_loader = DictLoader(templates.TEMPLATES)  # *.html: autoescaped
    app.jinja_env.filters.update(templates.FILTERS)
    app.config.update(
        SECRET_KEY=secret_key,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=secure_cookies,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        MAX_CONTENT_LENGTH=64 * 1024,  # forms only; files go straight to the bucket
    )
    if proxies:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxies, x_proto=proxies)
    app.extensions["ingest_store"] = store
    throttle = throttle or Throttle()
    # The upload page sends parts straight to the bucket's presigned links.
    connect_src = " ".join(["'self'", *store.upload_origins])
    cache: dict = {"at": -1e9, "accounts": {}}

    def load_accounts() -> dict:
        if time.monotonic() - cache["at"] > ACCOUNTS_TTL:
            cache.update(at=time.monotonic(), accounts=store.get_accounts())
        return cache["accounts"]

    def password_mark(account: dict) -> str:
        """Changes with the password, so a reset ends the account's sessions."""
        return account["password"]["scrypt"]["hash"][:16]

    @app.before_request
    def who():
        g.user = g.account = None
        user = session.get("user")
        if user:
            account = load_accounts().get(user)
            if account and hmac.compare_digest(
                session.get("mark", ""), password_mark(account)
            ):
                g.user, g.account = user, account
            else:  # removed, or the password changed since login
                session.clear()
        if "csrf" not in session:
            session["csrf"] = secrets.token_urlsafe(32)
        if request.method in ("POST", "PUT", "DELETE") and request.path != "/api/login":
            sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf", "")
            if not hmac.compare_digest(sent, session["csrf"]):
                abort(400, "form expired; reload the page")

    @app.after_request
    def headers(resp):
        if getattr(g, "retry_after", None):
            resp.headers.setdefault("Retry-After", str(g.retry_after))
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; frame-ancestors 'none'; form-action 'self'; "
            f"connect-src {connect_src}"
        )
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def common():
        return {
            "user": g.get("user"),
            "account": g.get("account"),
            "csrf": session.get("csrf", ""),
            "STATES": views.STATES,
            "staff": is_staff(),
        }

    def is_staff() -> bool:
        return bool(g.get("account")) and g.account["role"] == "staff"

    def login_required(f):
        @wraps(f)
        def inner(*a, **kw):
            if g.user is None:
                return redirect(url_for("login", next=request.path))
            return f(*a, **kw)

        return inner

    def visible(vendor: str) -> bool:
        return is_staff() or g.account.get("vendor") == vendor

    @app.get("/login")
    def login():
        return render_template("login.html", error="")

    def log_in(user: str, password: str) -> tuple[bool, str, int]:
        """Check and start a session: (ok, error, http status). The form and the CLI's
        /api/login share it, throttling included."""
        user = user.strip().lower()
        ip = request.remote_addr or "?"
        wait = throttle.retry_after(ip, user)  # before scrypt: cheap to refuse
        if wait:
            g.retry_after = wait
            minutes = (wait + 59) // 60
            return False, f"Too many attempts. Try again in {minutes} minute(s).", 429
        account = load_accounts().get(user)
        if not accounts.check_password(account, password):
            throttle.failed(ip, user)
            return False, "Wrong user or password.", 401
        throttle.succeeded(ip, user)
        session.clear()  # a fresh session on login, with a fresh CSRF token
        session.permanent = True
        session.update(
            user=user, mark=password_mark(account), csrf=secrets.token_urlsafe(32)
        )
        return True, "", 200

    app.extensions["ingest_login"] = log_in

    @app.post("/login")
    def login_post():
        ok, error, status = log_in(
            request.form.get("user", ""), request.form.get("password", "")
        )
        if not ok:
            extra = {"Retry-After": str(g.retry_after)} if status == 429 else {}
            return render_template("login.html", error=error), status, extra
        target = request.args.get("next", "")
        if not target.startswith("/") or target.startswith("//"):
            target = url_for("index")  # only our own pages
        return redirect(target)

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/style.css")
    def style():
        return templates.CSS, 200, {"Content-Type": "text/css; charset=utf-8"}

    @app.get("/review.js")
    def review_js():
        return templates.REVIEW_JS, 200, {"Content-Type": "text/javascript"}

    @app.get("/upload.js")
    def upload_js():
        return templates.UPLOAD_JS, 200, {"Content-Type": "text/javascript"}

    @app.get("/upload.py")
    def upload_cli():
        """The command-line uploader, one standard-library file vendors can run as is."""
        from egomimic.ingest import upload

        source = Path(upload.__file__).read_text()
        return (
            source,
            200,
            {
                "Content-Type": "text/x-python; charset=utf-8",
                "Content-Disposition": "attachment; filename=egoverse_upload.py",
            },
        )

    @app.get("/new")
    @login_required
    def new():
        if is_staff():  # vendors upload; staff review
            flash("Uploads are made from vendor accounts.", "error")
            return redirect(url_for("index"))
        delivery_id = request.args.get("delivery", "")
        _, limit = uploads.accepted()
        return render_template(
            "new.html",
            delivery_id=delivery_id,
            formats=formats.formats(),
            limit_gib=limit // 1024**3,
        )

    @app.get("/")
    @login_required
    def index():
        if is_staff():
            vendor = request.args.get("vendor") or None
            if vendor is not None and not _slug(vendor):
                abort(404)
        else:
            vendor = g.account["vendor"]
        listing = views.rows(store, vendor)
        summary = [
            ("Total deliveries", len(listing)),
            ("Awaiting review", sum(r.state == "ready" for r in listing)),
            ("Committed", sum(r.state == "committed" for r in listing)),
            (
                "Needs attention",
                sum(r.state in {"failed", "rejected", "unsupported"} for r in listing),
            ),
        ]
        state = request.args.get("state")
        if state:
            listing = [r for r in listing if r.state == state]
        vendors = (
            sorted({r.ref.vendor for r in views.rows(store)}) if is_staff() else []
        )
        return render_template(
            "index.html",
            rows=listing,
            vendor=vendor,
            state=state,
            vendors=vendors,
            summary=summary,
        )

    @app.get("/d/<vendor>/<delivery_id>")
    @login_required
    def delivery(vendor, delivery_id):
        try:
            ref = Ref(vendor, delivery_id)
        except ValueError:
            abort(404)
        if not visible(vendor):
            abort(404)  # the same answer as a delivery that does not exist
        r = views.row(store, ref)
        if r is None:
            abort(404)
        st = r.status
        # A decision on the current stage the watcher has not acted on yet, else the one
        # the watcher recorded with its outcome.
        decision = r.decision or st.get("decision") or {}
        return render_template(
            "delivery.html",
            r=r,
            error=st.get("error", "") if is_staff() else views.vendor_error(st),
            decision=decision if is_staff() else views.vendor_decision(decision),
            decisions=store.decisions(ref) if is_staff() else [],
        )

    @app.errorhandler(HTTPException)
    def error_page(e: HTTPException):
        """Errors in the dashboard's own layout; the API keeps its JSON answers."""
        if e.response is not None:  # a full response was given (the API's JSON errors)
            return e.response
        if request.path.startswith("/api/"):
            return jsonify({"error": e.description or e.name}), e.code
        messages = {
            404: "There is no such page, or it is not one your account can see.",
            400: e.description or "That request was not understood.",
        }
        text = messages.get(e.code, e.description or e.name)
        return render_template(
            "error.html", code=e.code, name=e.name, text=text
        ), e.code

    app.register_blueprint(uploads.bp)
    app.register_blueprint(review.bp)
    return app


def _slug(s: str) -> bool:
    try:
        Ref(s, "x")
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse

    from egomimic.utils.env import load_env

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    accounts.add_store_args(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    if not args.local_store:
        load_env()
    secret = os.environ.get("EGOVERSE_DASHBOARD_SECRET", "")
    if not secret:
        raise SystemExit("set EGOVERSE_DASHBOARD_SECRET (e.g. python -c "
                         "'import secrets; print(secrets.token_urlsafe(48))')")  # fmt: skip
    insecure = os.environ.get("EGOVERSE_DASHBOARD_INSECURE") == "1"
    proxies = int(os.environ.get("EGOVERSE_DASHBOARD_PROXIES", "0"))
    app = create_app(
        accounts.store_from_args(args),
        secret,
        secure_cookies=not insecure,
        proxies=proxies,
    )
    app.run(host=args.host, port=args.port)
    return 0
