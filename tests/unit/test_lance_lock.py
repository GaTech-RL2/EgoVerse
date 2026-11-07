"""The writer lock (egomimic/rldb/lance/lock.py): one writer per Lance root.

The R2 backend runs against a fake client that enforces If-None-Match / If-Match the way
R2 does; the two committer races run in real processes on a local root."""

import io
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path

import pytest

lance = pytest.importorskip("lance")
pytest.importorskip("av")

from fixtures.lance_tables import EVA_DELIVERY_FRAMES, eva_delivery  # noqa: E402

from egomimic.rldb.lance import commit, lock  # noqa: E402
from egomimic.rldb.lance.lock import LockHeld, LockLost, writer_lock  # noqa: E402
from egomimic.rldb.lance.schema import recording_id, steps_path  # noqa: E402
from egomimic.scripts.lance_commit import commit_staged  # noqa: E402

EVA_STEPS = steps_path("rl2", "eva_bimanual", "parallel_jaw_gripper")


# --- local backend ---------------------------------------------------------------------


def _hold(root, seconds, ready, wait=0.0):
    with writer_lock(root, wait=wait):
        ready.set()
        time.sleep(seconds)


def test_a_second_process_waits_for_the_lock(tmp_path):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), 2.0, ready))
    p.start()
    assert ready.wait(60)
    t = time.monotonic()
    with writer_lock(tmp_path, wait=30):
        waited = time.monotonic() - t
    p.join()
    assert waited > 0.5


def test_giving_up_on_a_held_lock_names_the_holder(tmp_path):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), 5.0, ready))
    p.start()
    assert ready.wait(60)
    with pytest.raises(LockHeld, match=f"pid {p.pid}"):
        with writer_lock(tmp_path, wait=0.3):
            pass
    p.join()


def test_a_killed_holder_frees_the_local_lock(tmp_path):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), 600.0, ready))
    p.start()
    assert ready.wait(60)
    p.kill()
    p.join()
    with writer_lock(tmp_path, wait=5):
        pass


def test_the_lock_is_reentrant_within_a_process(tmp_path):
    with writer_lock(tmp_path, wait=0) as outer:
        with writer_lock(str(tmp_path) + "/", wait=0) as inner:
            assert inner is outer and outer.depth == 2
        assert outer.depth == 1
    # Released: a fresh hold works at once.
    with writer_lock(tmp_path, wait=0):
        pass


# --- R2 backend, on a fake client ------------------------------------------------------


class _Err(Exception):
    def __init__(self, status):
        self.response = {"ResponseMetadata": {"HTTPStatusCode": status}}


class FakeS3:
    """put_object / get_object with R2's conditional-PUT semantics."""

    def __init__(self):
        self.objs, self.n = {}, 0
        self.mu = threading.Lock()

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None):
        with self.mu:
            cur = self.objs.get((Bucket, Key))
            if IfNoneMatch == "*" and cur is not None:
                raise _Err(412)
            if IfMatch is not None and (cur is None or cur[0] != IfMatch):
                raise _Err(412)
            self.n += 1
            etag = f'"{self.n}"'
            self.objs[(Bucket, Key)] = (etag, Body)
            return {"ETag": etag}

    def get_object(self, Bucket, Key):
        with self.mu:
            cur = self.objs.get((Bucket, Key))
        if cur is None:
            raise _Err(404)
        return {"ETag": cur[0], "Body": io.BytesIO(cur[1])}

    def body(self, key="r/_writer.lock"):
        return json.loads(self.objs[("b", key)][1])


ROOT = "s3://b/r"


def _s3lock(fake, ttl=60.0):
    return lock._S3Lock(ROOT, ttl, "tester", client=fake)


def test_s3_acquire_creates_then_others_wait_until_released():
    fake = FakeS3()
    a = _s3lock(fake)
    a.acquire(wait=0)
    assert fake.body()["token"] == a.token and not fake.body()["released"]
    b = _s3lock(fake)
    with pytest.raises(LockHeld, match="tester"):
        b.acquire(wait=0)
    a.release()
    assert fake.body()["released"]
    b.acquire(wait=0)
    assert fake.body()["token"] == b.token
    b.release()


def test_s3_an_expired_holder_is_taken_over_and_then_finds_it_lost():
    fake = FakeS3()
    a = _s3lock(fake, ttl=60.0)
    a.acquire(wait=0)
    a._stop.set()  # a stalls: no more heartbeats
    # ...until its expiry passes.
    etag, _ = fake.objs[("b", "r/_writer.lock")]
    stale = {**fake.body(), "expires_at": time.time() - 1}
    fake.objs[("b", "r/_writer.lock")] = (etag, json.dumps(stale).encode())
    b = _s3lock(fake)
    b.acquire(wait=0)
    assert fake.body()["token"] == b.token
    a.renewed = 0.0  # its last renewal is old, so check() renews and finds the takeover
    with pytest.raises(LockLost):
        a.check()
    a.release()  # releasing a lost lock leaves the new holder alone
    assert fake.body()["token"] == b.token and not fake.body()["released"]
    b.release()


def test_s3_the_heartbeat_pushes_expiry_forward():
    fake = FakeS3()
    a = _s3lock(fake, ttl=0.4)
    a.acquire(wait=0)
    first = fake.body()["expires_at"]
    time.sleep(0.35)
    assert fake.body()["expires_at"] > first and not a.lost
    a.release()


def test_s3_two_racing_acquirers_get_it_one_at_a_time():
    fake = FakeS3()
    inside, overlap = [], []

    def worker():
        lk = _s3lock(fake)
        lk.acquire(wait=10)
        inside.append(1)
        overlap.append(len(inside))
        time.sleep(0.05)
        inside.pop()
        lk.release()

    lock.POLL_S, poll = 0.01, lock.POLL_S
    try:
        ts = [threading.Thread(target=worker) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        lock.POLL_S = poll
    assert overlap == [1] * 6


# --- the committer races the R2 test found -----------------------------------------------


def _gate_committed_ids(barrier):
    """Make each committed_ids() read wait (up to 3 s) for the other writer's, so two
    unlocked writers both read before either writes."""
    real = commit.committed_ids

    def gated(root):
        out = real(root)
        try:
            barrier.wait(timeout=3)
        except threading.BrokenBarrierError:
            pass
        return out

    commit.committed_ids = gated


def _commit_child(staging, root, work, barrier, out):
    _gate_committed_ids(barrier)
    r = commit_staged(staging, "eva", root, Path(work))
    out.put(
        {
            "added": [a for b in r["batches"] for a in b["added"]],
            "rejected": r["rejected"],
        }
    )


def test_two_committers_with_the_same_recording_store_it_once(tmp_path):
    vid = "100"
    for name in ("st1", "st2"):
        src = eva_delivery(tmp_path / f"built_{name}", vid)
        commit.stage(src, commit.join(tmp_path / name, "eva", vid))
    root = tmp_path / "lance"
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(2), ctx.Queue()
    ps = [
        ctx.Process(
            target=_commit_child,
            args=(str(tmp_path / n), str(root), str(tmp_path / f"w{n}"), barrier, out),
        )
        for n in ("st1", "st2")
    ]
    for p in ps:
        p.start()
    reports = [out.get(timeout=300) for _ in ps]
    for p in ps:
        p.join()
    rid = recording_id("eva", vid)
    recs = (
        lance.dataset(root / "recordings.lance")
        .to_table()
        .column("recording_id")
        .to_pylist()
    )
    steps = (
        lance.dataset(root / EVA_STEPS)
        .to_table(columns=["recording_id"])
        .column(0)
        .to_pylist()
    )
    assert recs == [rid]
    assert steps.count(rid) == EVA_DELIVERY_FRAMES
    adds = [
        c["added"] for c in lance.dataset(root / "changes.lance").to_table().to_pylist()
    ]
    assert adds == [[rid]]
    # One writer added it; the other, which waited, found it committed.
    assert sorted(len(r["added"]) for r in reports) == [0, 1]


def _cleanup_child(root, staging, ready, go, stop, rounds):
    import datetime

    ready.set()
    go.wait(60)
    while not stop.is_set():
        try:
            commit.cleanup(
                root,
                staging,
                older_than=datetime.timedelta(0),
                upload_grace=datetime.timedelta(0),
            )
            rounds.value += 1
        except Exception:  # the root may not exist yet
            time.sleep(0.05)


def test_cleanup_during_a_commit_leaves_its_videos(tmp_path, monkeypatch):
    staging, root = tmp_path / "staging", tmp_path / "lance"
    for vid in ("100", "200", "300"):
        commit.stage(
            eva_delivery(tmp_path / f"b{vid}", vid), commit.join(staging, "eva", vid)
        )
    ctx = mp.get_context("spawn")
    ready, go, stop, rounds = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Value("i", 0)
    p = ctx.Process(
        target=_cleanup_child, args=(str(root), str(staging), ready, go, stop, rounds)
    )
    p.start()
    assert ready.wait(120)
    # Start the cleanup loop between placing the videos and publishing the rows: without
    # the lock it would see them unreferenced and delete them.
    real_stage = commit._stage

    def stage_then_let_cleanup_run(root_, d):
        out = real_stage(root_, d)
        go.set()
        time.sleep(1.0)
        return out

    monkeypatch.setattr(commit, "_stage", stage_then_let_cleanup_run)
    commit_staged(str(staging), "eva", str(root), tmp_path / "work")
    time.sleep(0.5)
    stop.set()
    p.join(60)
    assert rounds.value > 0  # cleanup did run alongside
    recs = (
        lance.dataset(root / "recordings.lance")
        .to_table(columns=["video_path"])
        .column(0)
        .to_pylist()
    )
    paths = [p_ for m in recs for _, p_ in m]
    assert len(recs) == 3 and paths
    assert all((root / p_).exists() for p_ in paths), [
        p_ for p_ in paths if not (root / p_).exists()
    ]


def test_an_ingest_round_reports_busy_while_another_writer_holds_the_lock(
    tmp_path, monkeypatch
):
    from egomimic.ingest import run

    root, staging = tmp_path / "lance", tmp_path / "staging"
    (staging / "mcap-x").mkdir(parents=True)
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_hold, args=(str(root), 5.0, ready))
    p.start()
    assert ready.wait(60)
    monkeypatch.setattr(run, "LOCK_WAIT_S", 0.3)
    out = run.commit_round(str(staging), str(root), tmp_path / "work")
    p.join()
    assert set(out) == {"busy"} and f"pid {p.pid}" in out["busy"]
