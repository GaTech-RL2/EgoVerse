"""Dashboard uploads (egomimic/ingest/dashboard/uploads.py) and the command-line
uploader (egomimic/ingest/upload.py), on the local-directory store."""

import io
import json
import types
import urllib.error

import pytest

pytest.importorskip("flask")

from egomimic.ingest import accounts, upload  # noqa: E402
from egomimic.ingest.dashboard import uploads  # noqa: E402
from egomimic.ingest.dashboard.app import create_app  # noqa: E402
from egomimic.ingest.store import LocalStore, R2Store, Ref  # noqa: E402

SECRET = "s" * 48
PW = "correct horse battery"
A = Ref("acme", "episode_0001")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "SCRYPT", {"n": 2**4, "r": 8, "p": 1})
    s = LocalStore(tmp_path / "bucket")
    accounts.add(s, "acme-ops", PW, role="vendor", vendor="acme")
    accounts.add(s, "bolt-ops", PW, role="vendor", vendor="bolt")
    accounts.add(s, "simar", PW, role="staff")
    return s


@pytest.fixture
def app(store):
    return create_app(store, SECRET, secure_cookies=False)


class Api:
    """A logged-in JSON client over Flask's test client."""

    def __init__(self, app, user):
        self.c = app.test_client()
        r = self.c.post("/api/login", json={"user": user, "password": PW})
        assert r.status_code == 200, r.get_json()
        self.csrf = r.get_json()["csrf"]

    def __call__(self, method, path, body=None, **kw):
        headers = {"X-CSRF-Token": self.csrf}
        r = self.c.open(path, method=method, json=body, headers=headers, **kw)
        return r.status_code, r.get_json()


def send(api, delivery, name, data: bytes):
    """Upload one file through the API (local store: parts via /part)."""
    base = f"/api/deliveries/{delivery}"
    code, start = api("POST", f"{base}/uploads", {"name": name, "size": len(data)})
    assert code == 200, start
    ref = {"name": name, "upload_id": start["upload_id"]}
    ps = start["part_size"]
    parts = list(range(1, start["parts"] + 1))
    code, out = api("POST", f"{base}/urls", {**ref, "parts": parts})
    assert code == 200 and all(u is None for u in out["urls"].values())
    etags = []
    for n in parts:
        r = api.c.put(
            f"{base}/part",
            query_string={**ref, "part": n},
            data=data[(n - 1) * ps : n * ps],
            headers={"X-CSRF-Token": api.csrf},
        )
        assert r.status_code == 200, r.get_json()
        etags.append([n, r.get_json()["etag"]])
    return api("POST", f"{base}/finish", {**ref, "parts": etags})


def test_a_vendor_uploads_a_delivery_through_the_api(app, store, monkeypatch):
    monkeypatch.setattr(uploads, "MIN_PART", 4)  # several parts from a small file
    api = Api(app, "acme-ops")
    assert send(api, "episode_0001", "episode.mcap", b"0123456789")[0] == 200
    assert send(api, "episode_0001", "annotation.mcap", b"labels")[0] == 200
    d = store.get_delivery(A)
    assert not d.complete  # not until the vendor says the delivery is all there
    code, out = api(
        "POST", "/api/deliveries/episode_0001/complete",
        {"files": ["episode.mcap", "annotation.mcap"]},
    )  # fmt: skip
    assert code == 200 and out["files"] == {"annotation.mcap": 6, "episode.mcap": 10}
    d = store.get_delivery(A)
    assert d.complete and store.completion(A)["by"] == "acme-ops"
    path = store.root / "ingest-acme/episode_0001/episode.mcap"
    assert path.read_bytes() == b"0123456789"
    code, out = api("GET", "/api/deliveries/episode_0001")
    assert out["state"] == "queued" and out["uploads"] == []


def test_uploads_go_only_to_the_vendors_own_folder(app, store):
    api = Api(app, "acme-ops")
    send(api, "episode_0001", "episode.mcap", b"x")
    assert store.get_delivery(Ref("acme", "episode_0001"))
    assert store.get_delivery(Ref("bolt", "episode_0001")) is None
    # Another vendor cannot use acme's upload id, even for the same delivery name.
    code, start = api(
        "POST", "/api/deliveries/episode_0002/uploads",
        {"name": "episode.mcap", "size": 1},
    )  # fmt: skip
    bolt = Api(app, "bolt-ops")
    ref = {"name": "episode.mcap", "upload_id": start["upload_id"]}
    assert (
        bolt("POST", "/api/deliveries/episode_0002/urls", {**ref, "parts": [1]})[0]
        == 404
    )
    assert bolt("POST", "/api/deliveries/episode_0002/abort", ref)[0] == 404
    # ... nor can acme use it for another delivery or file.
    assert api("POST", "/api/deliveries/other/urls", {**ref, "parts": [1]})[0] == 404
    wrong = {**ref, "name": "annotation.mcap"}
    assert (
        api("POST", "/api/deliveries/episode_0002/finish", {**wrong, "parts": []})[0]
        == 404
    )


def test_staff_and_strangers_do_not_upload(app):
    staff = Api(app, "simar")
    body = {"name": "episode.mcap", "size": 1}
    assert staff("POST", "/api/deliveries/x/uploads", body)[0] == 403
    assert app.test_client().get("/api/deliveries/x").status_code == 401
    assert staff.c.get("/new").status_code == 302  # back to the list, with a reason


def test_api_writes_need_the_csrf_token(app):
    api = Api(app, "acme-ops")
    r = api.c.post(
        "/api/deliveries/x/uploads", json={"name": "episode.mcap", "size": 1}
    )
    assert r.status_code == 400
    r = api.c.post("/api/login", data={"user": "acme-ops", "password": PW})
    assert r.status_code == 415  # the API login takes JSON only (no cross-site forms)


def test_files_no_format_accepts_are_refused_before_uploading(app, monkeypatch):
    api = Api(app, "acme-ops")
    start = "/api/deliveries/episode_0001/uploads"
    code, out = api("POST", start, {"name": "recording.vrs", "size": 1})
    assert code == 422 and "episode.mcap" in out["error"]
    code, out = api("POST", start, {"name": "episode.mcap", "size": 300 * 1024**3})
    assert code == 422 and "GiB limit" in out["error"]
    for bad in ("../x/episode.mcap", "/etc/episode.mcap", "_complete.json"):
        assert api("POST", start, {"name": bad, "size": 1})[0] == 400
    assert api("POST", start, {"name": "episode.mcap", "size": -1})[0] == 400
    assert api("POST", "/api/deliveries/..%2Fbolt/uploads", {})[0] in (400, 404)


def test_completion_waits_for_uploads_and_needs_a_whole_delivery(app, store):
    api = Api(app, "acme-ops")
    base = "/api/deliveries/episode_0001"
    send(api, "episode_0001", "annotation.mcap", b"labels")
    code, out = api("POST", f"{base}/complete", {"files": ["annotation.mcap"]})
    assert code == 422 and out["formats"][0]["required"] == ["episode.mcap"]
    code, out = api("POST", f"{base}/complete", {"files": ["episode.mcap"]})
    assert code == 422  # listed but never uploaded
    code, start = api("POST", f"{base}/uploads", {"name": "episode.mcap", "size": 1})
    code, out = api("POST", f"{base}/complete", {"files": ["annotation.mcap"]})
    assert code in (409, 422)
    code, out = api(
        "POST", f"{base}/complete", {"files": ["episode.mcap", "annotation.mcap"]}
    )
    assert code == 409 and len(out["uploads"]) == 1  # the tab that was closed
    assert api("POST", f"{base}/clear", {"record": "f" * 64})[0] == 404
    assert api("POST", f"{base}/clear", {"record": out["uploads"][0]})[0] == 200
    assert api("GET", base)[1]["uploads"] == []


def test_part_sizes_stay_within_r2_limits():
    assert uploads.part_size(0) == uploads.MIN_PART
    assert uploads.part_size(10 * 1024**3) == uploads.MIN_PART
    big = 200 * 1024**3
    ps = uploads.part_size(big)
    assert ps % 1024**2 == 0 and -(-big // ps) <= uploads.MAX_PARTS


def test_presigned_stores_send_parts_to_the_bucket(store, monkeypatch):
    class FakeMeta:
        endpoint_url = "https://acct.r2.cloudflarestorage.com"

    fake = types.SimpleNamespace(meta=FakeMeta())
    r2 = R2Store("rldb", client=fake)
    assert r2.upload_origins == [
        "https://acct.r2.cloudflarestorage.com",
        "https://*.acct.r2.cloudflarestorage.com",
    ]
    monkeypatch.setattr(LocalStore, "upload_origins", r2.upload_origins)
    monkeypatch.setattr(LocalStore, "part_url", lambda self, *a, **k: "https://p")
    app = create_app(store, SECRET, secure_cookies=False)
    api = Api(app, "acme-ops")
    csp = api.c.get("/new").headers["Content-Security-Policy"]
    assert "connect-src 'self' https://acct.r2.cloudflarestorage.com" in csp
    base = "/api/deliveries/episode_0001"
    code, start = api("POST", f"{base}/uploads", {"name": "episode.mcap", "size": 1})
    ref = {"name": "episode.mcap", "upload_id": start["upload_id"]}
    assert api("POST", f"{base}/urls", {**ref, "parts": [1]})[1]["urls"] == {
        "1": "https://p"
    }
    r = api.c.put(
        f"{base}/part", query_string={**ref, "part": 1}, data=b"x",
        headers={"X-CSRF-Token": api.csrf},
    )  # fmt: skip
    assert r.status_code == 404  # bytes never go through the dashboard then


def test_the_upload_page_and_uploader_are_served(app):
    api = Api(app, "acme-ops")
    page = api.c.get("/new?delivery=episode_0001").get_data(as_text=True)
    assert "Correct episode_0001" in page and "episode.mcap" in page
    assert 'src="/upload.js"' in page
    assert "presigned" in api.c.get("/upload.js").get_data(as_text=True)
    r = api.c.get("/upload.py")
    assert r.status_code == 200 and b"def upload_delivery" in r.data
    assert "attachment" in r.headers["Content-Disposition"]
    assert "New delivery" in api.c.get("/").get_data(as_text=True)


# ---------------------------------------------------------------- the CLI


class FakeOpener:
    """urllib's opener, answered by Flask's test client (cookies kept by the client)."""

    def __init__(self, app):
        self.c = app.test_client()

    def open(self, req, timeout=None):
        url = req.full_url.removeprefix("http://dash")
        r = self.c.open(
            url,
            method=req.get_method(),
            data=req.data,
            headers=dict(req.header_items()),
        )
        if r.status_code >= 400:
            raise urllib.error.HTTPError(
                url, r.status_code, r.status, {}, io.BytesIO(r.data)
            )
        return _Resp(r.data)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def cli(app):
    c = upload.Client("http://dash", opener=FakeOpener(app))
    c.login("acme-ops", PW)
    return c


def test_the_cli_uploads_a_folder_and_a_correction_replaces_it(
    app, store, tmp_path, monkeypatch
):
    monkeypatch.setattr(uploads, "MIN_PART", 7)
    folder = tmp_path / "episode_0001"
    folder.mkdir()
    (folder / "episode.mcap").write_bytes(bytes(range(256)) * 3)
    (folder / "annotation.mcap").write_bytes(b"labels v1")
    out = upload.upload_delivery(
        cli(app), folder, "episode_0001", log=lambda *a, **k: 0
    )
    assert out["files"] == {"annotation.mcap": 9, "episode.mcap": 768}
    d = store.get_delivery(A)
    assert d.complete
    bucket = store.root / "ingest-acme/episode_0001"
    assert (bucket / "episode.mcap").read_bytes() == bytes(range(256)) * 3

    (folder / "annotation.mcap").unlink()  # the correction drops the labels
    (folder / "episode.mcap").write_bytes(b"v2")
    upload.upload_delivery(cli(app), folder, "episode_0001", log=lambda *a, **k: 0)
    assert sorted(p.name for p in bucket.iterdir() if p.name != "_complete.json") == [
        "episode.mcap"
    ]
    assert store.get_delivery(A).complete


def test_the_cli_stops_on_unfinished_uploads_unless_told(app, store, tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "episode.mcap").write_bytes(b"x")
    store.start_upload(A, "episode.mcap")  # a crashed earlier attempt
    with pytest.raises(upload.UploadError, match="clear-stale"):
        upload.upload_delivery(cli(app), folder, "episode_0001", log=lambda *a, **k: 0)
    upload.upload_delivery(
        cli(app), folder, "episode_0001", clear_stale=True, log=lambda *a, **k: 0
    )
    assert store.get_delivery(A).complete


def test_the_cli_aborts_a_file_that_fails(app, store, tmp_path, monkeypatch):
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "episode.mcap").write_bytes(b"x" * 100)
    monkeypatch.setattr(upload, "RETRIES", 1)
    c = cli(app)
    real = c.call

    def flaky(method, path, *a, **kw):
        if path.endswith("/part"):
            raise upload.UploadError("network down")
        return real(method, path, *a, **kw)

    c.call = flaky
    with pytest.raises(upload.UploadError, match="network down"):
        upload.upload_delivery(c, folder, "episode_0001", log=lambda *a, **k: 0)
    assert store.uploads_in_progress(A) == []  # aborted, nothing left behind
    assert not (store.root / "ingest-acme/episode_0001/_complete.json").exists()


def test_the_cli_refuses_plain_http_to_a_remote_dashboard(capsys):
    assert upload.main(["x", "--url", "http://dash.example", "--user", "u"]) == 2
    assert "https" in capsys.readouterr().err


def test_the_api_login_reports_errors_as_json(app):
    r = app.test_client().post(
        "/api/login", json={"user": "acme-ops", "password": "no"}
    )
    assert r.status_code == 401 and json.loads(r.data)["error"].startswith("Wrong")


# ---------------------------------------------------------------- a modified client


def started(api, data_len, monkeypatch, min_part=4):
    monkeypatch.setattr(uploads, "MIN_PART", min_part)
    base = "/api/deliveries/episode_0001"
    code, start = api(
        "POST", f"{base}/uploads", {"name": "episode.mcap", "size": data_len}
    )
    assert code == 200 and (start["part_size"], start["parts"]) == (4, 3)
    return base, {"name": "episode.mcap", "upload_id": start["upload_id"]}


def put(api, base, ref, n, data):
    r = api.c.put(
        f"{base}/part", query_string={**ref, "part": n}, data=data,
        headers={"X-CSRF-Token": api.csrf},
    )  # fmt: skip
    return r.status_code, r.get_json()


def test_links_only_for_the_declared_parts(app, monkeypatch):
    api = Api(app, "acme-ops")
    base, ref = started(api, 10, monkeypatch)
    assert api("POST", f"{base}/urls", {**ref, "parts": [1, 2, 3]})[0] == 200
    for extra in ([4], [0], [10_000], ["1"]):
        assert api("POST", f"{base}/urls", {**ref, "parts": extra})[0] == 400


def test_parts_must_have_their_exact_length(app, store, monkeypatch):
    api = Api(app, "acme-ops")
    base, ref = started(api, 10, monkeypatch)
    assert put(api, base, ref, 1, b"0123" * 50)[0] == 413  # oversized: not even read
    assert put(api, base, ref, 3, b"89xx")[0] == 400  # the last part is 2 bytes
    assert put(api, base, ref, 4, b"0123")[0] == 400  # an extra part
    assert put(api, base, ref, 1, b"0123")[0] == 200


def test_finishing_needs_exactly_the_declared_parts(app, store, monkeypatch):
    api = Api(app, "acme-ops")
    base, ref = started(api, 10, monkeypatch)
    etags = [
        [n, put(api, base, ref, n, chunk)[1]["etag"]]
        for n, chunk in ((1, b"0123"), (2, b"4567"), (3, b"89"))
    ]
    code, out = api("POST", f"{base}/finish", {**ref, "parts": etags[:2]})  # missing
    assert code == 422 and "exactly parts 1..3" in out["error"]
    code, out = api("POST", f"{base}/finish", {**ref, "parts": etags + [[4, "x"]]})
    assert code == 422  # extra
    assert api("POST", f"{base}/finish", {**ref, "parts": etags})[0] == 200


def test_a_file_bigger_than_declared_is_removed(app, store, monkeypatch):
    """A client that gets bytes past the part checks (on R2: a bucket not enforcing the
    signed length) still cannot complete a file other than the size it declared."""
    api = Api(app, "acme-ops")
    base, ref = started(api, 10, monkeypatch)
    uid = ref["upload_id"]
    etags = [
        [n, store.upload_part(A, "episode.mcap", uid, n, chunk)]  # behind the API
        for n, chunk in ((1, b"0123"), (2, b"4567" * 10_000), (3, b"89"))
    ]
    code, out = api("POST", f"{base}/finish", {**ref, "parts": etags})
    assert code == 422 and "removed" in out["error"]
    assert not (store.root / "ingest-acme/episode_0001/episode.mcap").exists()
    assert store.uploads_in_progress(A) == []


def test_internal_uploads_cannot_be_driven_through_the_api(app, store):
    """An upload started without a declared size (internal tools) is not the API's."""
    api = Api(app, "acme-ops")
    uid = store.start_upload(A, "episode.mcap")
    ref = {"name": "episode.mcap", "upload_id": uid}
    assert (
        api("POST", "/api/deliveries/episode_0001/urls", {**ref, "parts": [1]})[0]
        == 404
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost.attacker.example",
        "http://127.0.0.1.attacker.example",
        "http://localhost@attacker.example",
        "https://user:pw@dash.example",
        "http://dash.example",
        "ftp://dash.example",
        "http://10.0.0.1",
        "https://",
        "http://localhost:notaport",
    ],
)
def test_the_password_only_goes_to_https_or_this_machine(url, monkeypatch, capsys):
    monkeypatch.setattr(
        upload, "Client", lambda *a: pytest.fail(f"would have logged in to {url}")
    )
    assert upload.url_problem(url)
    assert upload.main(["x", "--url", url, "--user", "u"]) == 2


@pytest.mark.parametrize(
    "url",
    ["https://dash.example", "https://dash.example:8443/", "http://localhost:8770",
     "http://127.0.0.1:8765", "http://[::1]:8000"],
)  # fmt: skip
def test_https_and_loopback_urls_are_allowed(url):
    assert upload.url_problem(url) is None
