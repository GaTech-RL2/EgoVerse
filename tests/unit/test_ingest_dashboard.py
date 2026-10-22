"""The dashboard (egomimic/ingest/dashboard): accounts, login, and who sees what."""

import re

import pytest

pytest.importorskip("flask")

from egomimic.ingest import accounts  # noqa: E402
from egomimic.ingest.dashboard import views  # noqa: E402
from egomimic.ingest.dashboard.app import create_app  # noqa: E402
from egomimic.ingest.store import LocalStore, Ref  # noqa: E402

SECRET = "s" * 48
PW = "correct horse battery"
A1 = Ref("acme", "episode_0001")
B1 = Ref("bolt", "episode_0001")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "SCRYPT", {"n": 2**4, "r": 8, "p": 1})  # fast tests
    s = LocalStore(tmp_path / "bucket")
    accounts.add(s, "acme-ops", PW, role="vendor", vendor="acme")
    accounts.add(s, "bolt-ops", PW, role="vendor", vendor="bolt")
    accounts.add(s, "simar", PW, role="staff")
    for ref in (A1, B1):
        folder = tmp_path / f"src_{ref.vendor}"
        folder.mkdir()
        (folder / "episode.mcap").write_bytes(b"x" * 2048)
        s.upload_delivery(ref, folder, {"by": "test"})
    return s


@pytest.fixture
def app(store):
    return create_app(store, SECRET, secure_cookies=False)


def csrf(client, path="/login"):
    page = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf" value="([^"]+)"', page).group(1)


def login(app, user, password=PW):
    c = app.test_client()
    r = c.post("/login", data={"user": user, "password": password, "csrf": csrf(c)})
    return c, r


def ready(store, ref, **extra):
    d = store.get_delivery(ref)
    store.put_status(
        ref,
        {
            "state": "ready",
            "fingerprint": d.fingerprint,
            "stage": "ab" * 32,
            "report": {"num_frames": 3661, "valid_frac": {"left": 0.42}},
            "problems": [],
            "history": [{"state": "ready", "at": "20261021T140312.000000Z"}],
            **extra,
        },
    )


# ---------------------------------------------------------------- accounts


def test_passwords_are_stored_hashed_and_checked(store):
    acct = store.get_accounts()["acme-ops"]
    assert PW not in str(acct) and acct["role"] == "vendor" and acct["vendor"] == "acme"
    assert accounts.check_password(acct, PW)
    assert not accounts.check_password(acct, PW + "!")
    assert not accounts.check_password(None, PW)  # unknown user


def test_accounts_are_validated(store):
    with pytest.raises(ValueError, match="already exists"):
        accounts.add(store, "simar", PW, role="staff")
    with pytest.raises(ValueError, match="at least"):
        accounts.add(store, "new", "short", role="staff")
    with pytest.raises(ValueError, match="vendor slug"):
        accounts.add(store, "new", PW, role="vendor")
    with pytest.raises(ValueError, match="no vendor"):
        accounts.add(store, "new", PW, role="staff", vendor="acme")
    with pytest.raises(ValueError):
        accounts.add(store, "Bad.Name", PW, role="staff")


def test_the_cli_lists_accounts_without_hashes(store, capsys):
    root = str(store.root)
    assert accounts.main(["--local-store", root, "list"]) == 0
    out = capsys.readouterr().out
    assert '"acme-ops"' in out and "scrypt" not in out
    assert accounts.main(["--local-store", root, "remove", "bolt-ops"]) == 0
    assert "bolt-ops" not in store.get_accounts()
    assert accounts.main(["--local-store", root, "remove", "nobody"]) == 1


# ---------------------------------------------------------------- login


def test_pages_need_a_login(app):
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert c.get("/d/acme/episode_0001").status_code == 302


def test_a_wrong_password_or_unknown_user_is_refused(app):
    for user, pw in (("acme-ops", "wrong password!"), ("nobody", PW)):
        c, r = login(app, user, pw)
        assert r.status_code == 401 and "Wrong user or password" in r.get_data(True)
        assert c.get("/").status_code == 302


def test_a_post_without_the_form_token_is_refused(app):
    c = app.test_client()
    c.get("/login")
    r = c.post("/login", data={"user": "simar", "password": PW})
    assert r.status_code == 400


def test_login_only_redirects_to_our_own_pages(app):
    c = app.test_client()
    for target, expect in (("/d/acme/episode_0001", "/d/acme/episode_0001"),
                           ("//evil.example", "/"), ("https://evil.example", "/")):  # fmt: skip
        r = c.post(
            f"/login?next={target}",
            data={"user": "simar", "password": PW, "csrf": csrf(c)},  # new per login
        )
        assert r.headers["Location"] == expect


def test_logout_ends_the_session(app):
    c, _ = login(app, "simar")
    assert c.get("/").status_code == 200
    c.post("/logout", data={"csrf": csrf(c, "/")})
    assert c.get("/").status_code == 302


def test_a_password_change_or_removal_ends_existing_sessions(app, store, monkeypatch):
    from egomimic.ingest.dashboard import app as app_module

    monkeypatch.setattr(app_module, "ACCOUNTS_TTL", -1)  # no caching
    c, _ = login(app, "acme-ops")
    assert c.get("/").status_code == 200
    accounts.set_password(store, "acme-ops", PW + " new")
    assert c.get("/").status_code == 302
    c, _ = login(app, "bolt-ops")
    accounts.remove(store, "bolt-ops")
    assert c.get("/").status_code == 302


def test_cookies_and_headers(store):
    app = create_app(store, SECRET)  # secure cookies by default
    c, r = login(app, "simar")
    cookie = r.headers["Set-Cookie"]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie
    r = c.get("/login")
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    with pytest.raises(ValueError):
        create_app(store, "too short")


# ---------------------------------------------------------------- who sees what


def test_a_vendor_sees_only_its_own_deliveries(app, store):
    ready(store, A1)
    ready(store, B1)
    c, _ = login(app, "acme-ops")
    page = c.get("/").get_data(as_text=True)
    assert "/d/acme/episode_0001" in page and "/d/bolt/" not in page
    assert c.get("/?vendor=bolt").get_data(as_text=True).count("/d/bolt/") == 0
    assert c.get("/d/acme/episode_0001").status_code == 200
    assert c.get("/d/bolt/episode_0001").status_code == 404  # as if it did not exist
    assert c.get("/d/acme/no_such").status_code == 404


def test_staff_see_every_vendor_and_can_filter(app, store):
    ready(store, A1)
    c, _ = login(app, "simar")
    page = c.get("/").get_data(as_text=True)
    assert "/d/acme/episode_0001" in page and "/d/bolt/episode_0001" in page
    page = c.get("/?vendor=bolt").get_data(as_text=True)
    assert "/d/bolt/" in page and "/d/acme/" not in page
    page = c.get("/?state=ready").get_data(as_text=True)
    assert "/d/acme/" in page and "/d/bolt/" not in page  # bolt: no status yet
    assert c.get("/d/bolt/episode_0001").status_code == 200
    assert c.get("/?vendor=../x").status_code == 404


def test_architecture_is_not_part_of_dashboard(app):
    staff, _ = login(app, "simar")
    for path in ("/architecture", "/architecture/download"):
        assert staff.get(path).status_code == 404
    assert 'href="/architecture"' not in staff.get("/").get_data(as_text=True)


def test_summary_respects_vendor_scope_but_not_state_filter(app, store):
    ready(store, A1)
    staff, _ = login(app, "simar")
    vendor, _ = login(app, "bolt-ops")

    def metrics(client, path):
        page = client.get(path).get_data(as_text=True)
        return re.findall(r"<strong>(\d+)</strong>\s*<span>([^<]+)</span>", page)

    assert ("2", "Total deliveries") in metrics(staff, "/?state=ready")
    assert ("1", "Awaiting review") in metrics(staff, "/")
    assert ("1", "Total deliveries") in metrics(staff, "/?vendor=bolt")
    assert ("0", "Awaiting review") in metrics(vendor, "/?vendor=acme")


def test_logins_are_throttled_before_any_password_work(store, monkeypatch):
    from egomimic.ingest.dashboard import throttle as th

    now = [0.0]
    app = create_app(
        store, SECRET, secure_cookies=False, throttle=th.Throttle(lambda: now[0])
    )
    checks = []
    real = accounts.check_password
    monkeypatch.setattr(
        accounts, "check_password", lambda a, p: checks.append(1) or real(a, p)
    )
    for _ in range(th.PAIR_FAILURES):
        assert login(app, "acme-ops", "wrong password!")[1].status_code == 401
    c, r = login(app, "acme-ops")  # even the right password waits now
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    assert "Too many attempts" in r.get_data(True)
    assert len(checks) == th.PAIR_FAILURES  # refused without hashing
    assert login(app, "bolt-ops")[1].status_code == 302  # other accounts unaffected
    now[0] += th.WINDOW  # temporary: the window passes, no lasting lockout
    assert login(app, "acme-ops")[1].status_code == 302


def test_one_address_trying_many_accounts_is_throttled():
    from egomimic.ingest.dashboard import throttle as th

    t = th.Throttle(lambda: 0.0)
    for i in range(th.IP_ATTEMPTS):
        assert t.retry_after("1.2.3.4", f"user{i}") == 0
    assert t.retry_after("1.2.3.4", "another") > 0
    assert t.retry_after("5.6.7.8", "another") == 0


def test_one_account_guessed_from_many_addresses_is_capped():
    from egomimic.ingest.dashboard import throttle as th

    t = th.Throttle(lambda: 0.0)
    for i in range(th.USER_FAILURES):
        assert t.retry_after(f"10.0.0.{i}", "acme-ops") == 0
        t.failed(f"10.0.0.{i}", "acme-ops")
    assert t.retry_after("10.0.1.1", "acme-ops") > 0


def test_behind_a_trusted_proxy_the_client_address_is_used(store):
    from egomimic.ingest.dashboard import throttle as th

    app = create_app(
        store, SECRET, secure_cookies=False, proxies=1,
        throttle=th.Throttle(lambda: 0.0),
    )  # fmt: skip
    c = app.test_client()
    for i in range(th.PAIR_FAILURES):
        r = c.post(
            "/login",
            data={"user": "acme-ops", "password": "nope nope nope", "csrf": csrf(c)},
            headers={"X-Forwarded-For": "9.9.9.9"},
        )
    r = c.post(
        "/login",
        data={"user": "acme-ops", "password": PW, "csrf": csrf(c)},
        headers={"X-Forwarded-For": "8.8.8.8"},  # another client: not throttled
    )
    assert r.status_code == 302


def test_updated_is_the_last_status_write(store):
    ready(store, A1)  # history says 2026-10-21 14:03; put_status stamps updated_at
    r = views.row(store, A1)
    assert r.updated == r.status["updated_at"] != "20261021T140312.000000Z"


def test_the_detail_page_shows_the_conversion(app, store):
    ready(store, A1, diff={"action": "new", "frames": 3661})
    c, _ = login(app, "acme-ops")
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "Converted; waiting for EgoVerse review" in page
    assert "3661" in page and "left: 0.42" in page and "episode.mcap" in page
    assert "2.0 KB" in page and "2026-10-21 14:03" in page  # the history entry


def test_vendor_text_is_escaped(app, store):
    ready(store, A1, state="rejected", problems=["<script>alert(1)</script>"])
    c, _ = login(app, "acme-ops")
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_vendors_do_not_see_our_internals(app, store):
    trace = "RuntimeError: disk full\nTraceback (most recent call last):\n  File ..."
    ready(store, A1, state="failed", error=trace)
    ready(
        store, B1, state="failed",
        error="ValueError: unsupported lens in /coc/flash7/x/work/b/episode.mcap",
        decision={"decision": "decline", "who": "simar", "comment": "blurry"},
    )  # fmt: skip
    c, _ = login(app, "acme-ops")
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "disk full" not in page and "Traceback" not in page
    assert "Failed on our side" in page
    c, _ = login(app, "bolt-ops")
    page = c.get("/d/bolt/episode_0001").get_data(as_text=True)
    assert "unsupported lens in &lt;path&gt;" in page and "/coc/" not in page
    assert "blurry" in page and "simar" not in page  # the decision, not who made it
    c, _ = login(app, "simar")
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "Traceback" in page  # staff see everything


def test_states_follow_the_bucket_when_it_moves_on(store, tmp_path):
    ready(store, A1)
    assert views.row(store, A1).state == "ready"
    store.start_upload(A1, "episode.mcap")  # a correction begins
    assert views.row(store, A1).state == "uploading"
    assert views.state_of(None, None) == "queued"
    listed = store.get_delivery(B1)
    assert views.state_of(listed, {"state": "ready", "fingerprint": "old"}) == "queued"


def test_staff_opening_the_upload_page_are_sent_back_with_a_reason(app):
    c, _ = login(app, "simar")
    r = c.get("/new")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert "Uploads are made from vendor accounts." in c.get("/").get_data(True)


def test_errors_use_the_dashboard_layout_and_the_api_keeps_json(app):
    c, _ = login(app, "acme-ops")
    r = c.get("/d/bolt/episode_0001")
    page = r.get_data(as_text=True)
    assert r.status_code == 404 and "Not Found" in page and "Back to" in page
    assert 'href="/style.css"' in page
    r = c.get("/api/deliveries/x/nonexistent")
    assert r.status_code == 404 and "error" in r.get_json()
    r = c.post("/api/deliveries/x/uploads", json={}, headers={"X-CSRF-Token": "bad"})
    assert r.status_code == 400 and r.get_json()["error"]


def test_nested_values_read_as_text():
    from egomimic.ingest.dashboard.templates import fmt

    assert fmt({"head": 0.9918055, "left": 0.4176}) == "head: 0.992, left: 0.418"
    assert fmt({"new": {"head": 0.5}}) == "new: head: 0.5"
    assert fmt([]) == "none" and fmt(None) == "none" and fmt(True) == "true"
