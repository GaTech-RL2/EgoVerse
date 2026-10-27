"""Staff review in the dashboard (egomimic/ingest/dashboard/review.py): approve and
decline, one at a time or in bulk, only ever on the result the reviewer saw."""

import re
import shutil
import subprocess

import pytest

pytest.importorskip("flask")

from egomimic.ingest import accounts  # noqa: E402
from egomimic.ingest.dashboard import views  # noqa: E402
from egomimic.ingest.dashboard.app import create_app  # noqa: E402
from egomimic.ingest.store import LocalStore, Ref  # noqa: E402

SECRET = "s" * 48
PW = "correct horse battery"
A = Ref("acme", "episode_0001")
B = Ref("acme", "episode_0002")
C = Ref("bolt", "episode_0001")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "SCRYPT", {"n": 2**4, "r": 8, "p": 1})
    s = LocalStore(tmp_path / "bucket")
    accounts.add(s, "acme-ops", PW, role="vendor", vendor="acme")
    accounts.add(s, "simar", PW, role="staff")
    accounts.add(s, "ryan-co", PW, role="staff")
    return s


@pytest.fixture
def app(store):
    return create_app(store, SECRET, secure_cookies=False)


def login(app, user):
    c = app.test_client()
    page = c.get("/login").get_data(as_text=True)
    token = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    c.post("/login", data={"user": user, "password": PW, "csrf": token})
    page = c.get("/").get_data(as_text=True)
    c.csrf = re.search(r'name="csrf" value="([^"]+)"', page).group(1)
    return c


def converted(store, tmp_path, ref, stage="ab" * 32, data=b"x"):
    """A delivery the watcher has staged: complete, with a ready status."""
    folder = tmp_path / f"src_{ref.vendor}_{ref.delivery_id}_{data.hex()[:8]}"
    folder.mkdir()
    (folder / "episode.mcap").write_bytes(data)
    store.upload_delivery(ref, folder, {"by": "test"})
    store.put_status(
        ref,
        {
            "state": "ready",
            "stage": stage,
            "fingerprint": store.get_delivery(ref).fingerprint,
            "report": {"num_frames": 60},
        },
    )


def decide(c, ref, decision, stage="ab" * 32, comment=""):
    return c.post(
        f"/d/{ref.vendor}/{ref.delivery_id}/decide",
        data={"decision": decision, "stage": stage, "comment": comment, "csrf": c.csrf},
        follow_redirects=True,
    )


def test_staff_approve_the_result_they_saw(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "simar")
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "Approve" in page and "Result abababababab" in page
    page = decide(c, A, "approve").get_data(as_text=True)
    assert "Approved; the next pipeline pass acts on it." in page
    (d,) = store.decisions(A)
    assert (d["decision"], d["who"], d["stage"]) == ("approve", "simar", "ab" * 32)
    assert views.row(store, A).state == "approved"
    assert "currently\n<strong>approved</strong> by simar" in page
    vendor = login(app, "acme-ops")
    page = vendor.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "Approved; added to the dataset at the next pass" in page
    assert "simar" not in page and "<h2>Review</h2>" not in page


def test_a_decline_needs_a_comment_and_the_vendor_sees_it(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "simar")
    page = decide(c, A, "decline").get_data(as_text=True)
    assert "A decline needs a comment" in page and store.decisions(A) == []
    decide(c, A, "decline", comment="Camera faces the ceiling")
    assert views.row(store, A).state == "declined"
    page = login(app, "acme-ops").get("/d/acme/episode_0001").get_data(as_text=True)
    assert "Declined by EgoVerse" in page and "Camera faces the ceiling" in page
    assert "simar" not in page


def test_a_reviewer_can_change_their_mind_until_the_watcher_acts(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "simar")
    decide(c, A, "approve")
    decide(c, A, "decline", comment="wrong task")
    assert views.row(store, A).state == "declined"
    assert store.latest_decisions()[A]["decision"] == "decline"


def test_nothing_is_recorded_for_a_result_the_reviewer_did_not_see(
    app, store, tmp_path
):
    converted(store, tmp_path, A, stage="cd" * 32)  # re-converted since the page loaded
    c = login(app, "simar")
    page = decide(c, A, "approve", stage="ab" * 32).get_data(as_text=True)
    assert "Not recorded" in page and store.decisions(A) == []

    converted(store, tmp_path, B)
    folder = tmp_path / "v2"
    folder.mkdir()
    (folder / "episode.mcap").write_bytes(b"v2")
    store.upload_delivery(B, folder, {})  # re-uploaded after conversion
    page = decide(c, B, "approve").get_data(as_text=True)
    assert "re-uploaded" in page and store.decisions(B) == []

    store.start_upload(B, "episode.mcap")  # a correction under way
    page = decide(c, B, "approve").get_data(as_text=True)
    assert "upload to it is under way" in page and store.decisions(B) == []


def test_only_ready_deliveries_can_be_decided(app, store, tmp_path):
    converted(store, tmp_path, A)
    store.put_status(A, {**store.get_status(A), "state": "committed"})
    c = login(app, "simar")
    assert "<h2>Review</h2>" not in c.get("/d/acme/episode_0001").get_data(True)
    assert "Not recorded" in decide(c, A, "approve").get_data(True)


def test_vendors_cannot_decide(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "acme-ops")
    r = c.post(
        "/d/acme/episode_0001/decide",
        data={"decision": "approve", "stage": "ab" * 32, "csrf": c.csrf},
    )
    assert r.status_code == 404
    r = c.post(
        "/decide",
        data={"decision": "approve", "item": f"acme/episode_0001/{'ab' * 32}",
              "csrf": c.csrf},
    )  # fmt: skip
    assert r.status_code == 404 and store.decisions(A) == []


def test_decisions_need_the_form_token(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "simar")
    c.csrf = "forged"
    assert decide(c, A, "approve").status_code == 400
    assert store.decisions(A) == []


def test_bulk_approval_decides_each_and_skips_what_changed(app, store, tmp_path):
    converted(store, tmp_path, A)
    converted(store, tmp_path, B)
    converted(store, tmp_path, C, stage="ef" * 32)
    store.put_status(B, {**store.get_status(B), "stage": "99" * 32})  # re-converted
    c = login(app, "simar")
    page = c.get("/").get_data(as_text=True)
    assert page.count('class="pick"') == 3 and 'id="all"' in page
    r = c.post(
        "/decide",
        data={
            "decision": "approve",
            "item": [f"acme/episode_0001/{'ab' * 32}", f"acme/episode_0002/{'ab' * 32}",
                     f"bolt/episode_0001/{'ef' * 32}"],
            "back": "/?state=ready",
            "csrf": c.csrf,
        },
    )  # fmt: skip
    assert r.headers["Location"] == "/?state=ready"
    page = c.get("/").get_data(as_text=True)
    assert "Approved 2; the next pipeline pass acts on them." in page
    assert "Skipped 1" in page and "acme/episode_0002" in page
    assert [len(store.decisions(r)) for r in (A, B, C)] == [1, 0, 1]
    assert store.decisions(C)[0]["stage"] == "ef" * 32  # each its own stage


def test_bulk_decisions_are_checked(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "simar")
    item = f"acme/episode_0001/{'ab' * 32}"
    post = lambda **d: c.post("/decide", data={"csrf": c.csrf, **d})  # noqa: E731
    assert post(decision="approve", item=["acme/../x/" + "ab" * 32]).status_code == 400
    assert post(decision="approve", item=[item] * 501).status_code == 400
    assert post(decision="maybe", item=[item]).status_code == 400
    post(decision="decline", item=[item])  # no comment
    post(decision="approve")  # nothing ticked
    page = c.get("/").get_data(as_text=True)
    assert "A decline needs a comment" in page and "Tick the deliveries" in page
    assert store.decisions(A) == []
    r = post(decision="approve", item=[item], back="//evil.example")
    assert r.headers["Location"] == "/"


def test_latest_decisions_come_from_one_listing(store):
    store.add_decision(A, "approve", "ryan-co", stage="ab" * 32)
    store.add_decision(A, "decline", "simar", "blurry", stage="ab" * 32)
    store.add_decision(C, "approve", "simar", stage="cd" * 32)
    store._put("pipeline/decisions/acme/episode_0001/garbage.json", b"{}")
    latest = store.latest_decisions()
    assert latest[A]["decision"] == "decline" and latest[A]["who"] == "simar"
    assert latest[C] == {**latest[C], "who": "simar", "stage": "cd" * 32}
    assert list(store.latest_decisions("bolt")) == [C]
    assert store.decisions(A)[-1]["comment"] == "blurry"  # the record keeps it


# ---------------------------------------------------------------- the whole loop

needs_ffmpeg = pytest.mark.skipif(
    not (
        shutil.which("ffmpeg")
        and "libx264"
        in subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True
        ).stdout.decode()
    ),
    reason="needs ffmpeg with libx264",
)


@needs_ffmpeg
def test_upload_convert_approve_commit(app, store, tmp_path):
    """A vendor's delivery, staged by the watcher, approved in the dashboard, committed
    by the next pass."""
    lance = pytest.importorskip("lance")
    from fixtures.mcap_episodes import write_mcap_episode

    from egomimic.ingest.launch import LocalLauncher
    from egomimic.ingest.run import commit_round
    from egomimic.ingest.watch import run_pass

    root, work = tmp_path / "lance", tmp_path / "work"
    staging = str(tmp_path / "lance_staging")
    launcher = LocalLauncher(
        store, str(root), tmp_path / "jobs", codec="libx264", mano=False
    )

    def watch():
        return run_pass(
            store, launcher, work, raw_root=str(tmp_path / "raw"),
            lance_staging=staging, revision="r1",
        )  # fmt: skip

    store.upload_delivery(A, write_mcap_episode(tmp_path / "ep"), {"by": "acme-ops"})
    assert watch().staged == ["acme/episode_0001"]
    c = login(app, "simar")
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    stage = re.search(r'name="stage" value="([0-9a-f]{64})"', page).group(1)
    decide(c, A, "approve", stage=stage)
    assert watch().approved == ["acme/episode_0001"]
    commit_round(staging, str(root), work)
    assert watch().committed == ["acme/episode_0001"]
    assert lance.dataset(root / "recordings.lance").count_rows() == 1
    page = login(app, "acme-ops").get("/d/acme/episode_0001").get_data(as_text=True)
    assert "In the dataset" in page


def test_a_decline_can_become_an_approval_before_the_watcher_acts(app, store, tmp_path):
    converted(store, tmp_path, A)
    c = login(app, "simar")
    decide(c, A, "decline", comment="looked blurry")
    assert views.row(store, A).state == "declined"
    page = c.get("/d/acme/episode_0001").get_data(as_text=True)
    assert "<h2>Review</h2>" in page  # still reviewable
    assert 'class="pick"' in c.get("/").get_data(as_text=True)  # and in bulk
    decide(c, A, "approve")
    assert views.row(store, A).state == "approved"
    assert store.latest_decisions()[A]["decision"] == "approve"
    # Once the watcher has acted, it is no longer up for review.
    store.put_status(A, {**store.get_status(A), "state": "declined"})
    assert "<h2>Review</h2>" not in c.get("/d/acme/episode_0001").get_data(True)


def test_a_re_uploaded_delivery_is_not_offered_for_review(app, store, tmp_path):
    converted(store, tmp_path, A)
    folder = tmp_path / "v2"
    folder.mkdir()
    (folder / "episode.mcap").write_bytes(b"v2")
    store.upload_delivery(A, folder, {})
    c = login(app, "simar")
    assert views.row(store, A).state == "queued"
    assert "<h2>Review</h2>" not in c.get("/d/acme/episode_0001").get_data(True)
    assert 'class="pick"' not in c.get("/").get_data(as_text=True)
