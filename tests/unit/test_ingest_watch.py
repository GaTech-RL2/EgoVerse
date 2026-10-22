"""The watcher pass (egomimic/ingest/watch.py) on a local-directory store."""

import shutil
import subprocess

import pytest
from fixtures.mcap_episodes import write_mcap_episode

lance = pytest.importorskip("lance")
if not (
    shutil.which("ffmpeg")
    and "libx264"
    in subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True
    ).stdout.decode()
):
    pytest.skip("needs ffmpeg with libx264", allow_module_level=True)

from egomimic.ingest import formats, watch  # noqa: E402
from egomimic.ingest.store import LocalStore, Ref  # noqa: E402
from egomimic.ingest.watch import Busy, run_pass  # noqa: E402

SETTINGS = dict(codec="libx264", mano=False)


@pytest.fixture
def env(tmp_path):
    return LocalStore(tmp_path / "bucket"), tmp_path / "lance", tmp_path / "work"


def deliver(store, tmp_path, ref, name="v1", **kw):
    """Upload a fixture episode as a lab machine would, and mark it complete."""
    ep = write_mcap_episode(tmp_path / f"src_{ref.delivery_id}_{name}", **kw)
    store.upload_delivery(ref, ep, {"by": "test"})


def run(env, **kw):
    store, root, work = env
    return run_pass(store, root, work, **SETTINGS, **kw)


def status(env, ref):
    return env[0].get_status(ref)


def lance_rows(root, table):
    return lance.dataset(root / table).to_table().to_pylist()


A = Ref("acme", "episode_0001")


def test_a_new_delivery_is_staged_then_committed_on_approval(env, tmp_path):
    store, root, _ = env
    deliver(store, tmp_path, A)
    s = run(env)
    assert s.staged == ["acme/episode_0001"] and not s.committed
    st = status(env, A)
    assert st["state"] == "ready" and st["format"] == "mcap"
    assert st["diff"]["action"] == "new" and st["report"]["num_frames"] == 60
    assert store.has_staged(A, st["stage"])
    assert not (root / "recordings.lance").exists()  # nothing before approval

    store.add_decision(A, "approve", "simar", stage=st["stage"])
    s = run(env)
    assert s.committed == ["acme/episode_0001"]
    st = status(env, A)
    assert st["state"] == "committed" and st["committed"]["status"] == "committed"
    prov = st["provenance"]  # kept after the stage itself is deleted
    assert prov["settings"] == {"mano": False, "mano_iters": 2400, "codec": "libx264"}
    assert prov["converter"]["revision"] not in ("", "unknown")
    assert len(prov["content_digest"]) == 64
    assert st["decision"]["who"] == "simar"
    assert len(lance_rows(root, "recordings.lance")) == 1
    assert not store.has_staged(A, st["stage"])  # the stage is cleaned up

    s = run(env)  # nothing new: a no-op
    assert s.staged == s.committed == []
    assert [h["state"] for h in status(env, A)["history"]] == [
        "processing", "ready", "committing", "committed",
    ]  # fmt: skip


def test_incomplete_deliveries_are_left_alone(env, tmp_path):
    store, _, _ = env
    uid = store.start_upload(A, "episode.mcap")
    store.upload_part(A, "episode.mcap", uid, 1, b"half an upload")
    assert run(env).staged == [] and status(env, A) is None


def test_rejected_and_unsupported_deliveries_say_why(env, tmp_path):
    store, _, _ = env
    deliver(store, tmp_path, A, imu_hz=100)
    folder = tmp_path / "vrs"
    folder.mkdir()
    (folder / "recording.vrs").write_bytes(b"raw")
    b = Ref("acme", "raw_vrs")
    store.upload_delivery(b, folder, {})
    s = run(env)
    assert sorted(s.rejected) == ["acme/episode_0001", "acme/raw_vrs"]
    assert status(env, A)["state"] == "rejected"
    assert status(env, A)["problems"] == ["[imu] /imu: 100 Hz, need >= 200"]
    assert status(env, b)["state"] == "unsupported"
    assert "recording.vrs" in status(env, b)["problems"][0]


def test_oversized_deliveries_are_rejected_without_downloading(
    env, tmp_path, monkeypatch
):
    store, _, _ = env
    deliver(store, tmp_path, A)
    small = formats.Format(**{**formats.by_name("mcap").__dict__, "max_file_bytes": 10})
    monkeypatch.setattr(formats, "formats", lambda: [small])
    monkeypatch.setattr(
        store, "download", lambda *a: pytest.fail("downloaded an oversized delivery")
    )
    assert run(env).rejected == ["acme/episode_0001"]
    assert "over the mcap limit" in status(env, A)["problems"][0]


def test_a_decline_drops_the_stage_and_never_commits(env, tmp_path):
    store, root, _ = env
    deliver(store, tmp_path, A)
    run(env)
    stage = status(env, A)["stage"]
    store.add_decision(A, "decline", "simar", "wrong task", stage=stage)
    assert run(env).declined == ["acme/episode_0001"]
    assert status(env, A)["state"] == "declined"
    assert not store.has_staged(A, stage)
    assert not (root / "recordings.lance").exists()


def test_an_approval_only_applies_to_the_stage_it_was_given_for(env, tmp_path):
    store, root, _ = env
    deliver(store, tmp_path, A)
    run(env)
    first = status(env, A)["stage"]
    deliver(store, tmp_path, A, name="v2", metadata_override={"scene": "kitchen_b"})
    store.add_decision(A, "approve", "simar", stage=first)  # approves the old stage
    s = run(env)
    assert s.staged == ["acme/episode_0001"] and not s.committed
    second = status(env, A)["stage"]
    assert second != first and not store.has_staged(A, first)  # superseded, gone
    assert run(env).committed == []  # the old approval does not carry over
    store.add_decision(A, "approve", "simar", stage=second)
    assert run(env).committed == ["acme/episode_0001"]
    assert lance_rows(root, "recordings.lance")[0]["scene"] == "kitchen_b"


def test_a_re_upload_after_commit_is_staged_as_a_replacement(env, tmp_path):
    store, root, _ = env
    deliver(store, tmp_path, A)
    run(env)
    store.add_decision(A, "approve", "simar", stage=status(env, A)["stage"])
    run(env)
    deliver(store, tmp_path, A, name="v2", labels_in=None)  # labels withdrawn
    run(env)
    st = status(env, A)
    assert st["state"] == "ready" and st["diff"]["action"] == "replace"
    assert st["diff"]["labels"]["removed"] == [
        "drop it in the basket",
        "pick up the shirt",
    ]
    store.add_decision(A, "approve", "simar", stage=st["stage"])
    run(env)
    assert status(env, A)["committed"]["status"] == "replaced"
    assert lance_rows(root, "annotations.lance") == []


def test_a_stage_that_went_stale_is_staged_again(env, tmp_path):
    store, root, _ = env
    deliver(store, tmp_path, A)
    run(env)
    stale = status(env, A)["stage"]
    # Lance moves on behind the watcher's back (e.g. a direct commit of the recording).
    from egomimic.scripts.mcap_process.mcap_to_lance import convert_and_commit

    ep = write_mcap_episode(
        tmp_path / "direct", metadata_override={"scene": "elsewhere"}
    )
    assert convert_and_commit(ep, root, **SETTINGS).status == "committed"
    store.add_decision(A, "approve", "simar", stage=stale)
    assert run(env).stale == ["acme/episode_0001"]
    assert status(env, A)["state"] == "stale"
    s = run(env)  # the next pass re-stages against the new Lance
    assert s.staged == ["acme/episode_0001"]
    st = status(env, A)
    assert st["stage"] != stale and st["diff"]["action"] == "replace"


def test_one_failing_delivery_does_not_stop_the_pass(env, tmp_path, monkeypatch):
    store, _, _ = env
    b = Ref("acme", "episode_0002")
    deliver(store, tmp_path, A)
    deliver(store, tmp_path, b)
    real = formats.prepare_delivery

    def flaky(delivery, staged, *a, **kw):
        if "episode_0001" in str(staged):
            raise RuntimeError("disk full")
        return real(delivery, staged, *a, **kw)

    monkeypatch.setattr(formats, "prepare_delivery", flaky)
    s = run(env)
    assert s.failed == ["acme/episode_0001"] and s.staged == ["acme/episode_0002"]
    assert "disk full" in status(env, A)["error"]


def test_a_second_pass_is_turned_away_while_one_runs(env):
    store, root, work = env
    with watch._lock(work, root):
        with pytest.raises(Busy):
            run_pass(store, root, work, **SETTINGS)


def test_two_work_folders_cannot_write_one_lance_root(env, tmp_path):
    store, root, work = env
    with watch._lock(work, root):
        with pytest.raises(Busy):
            run_pass(store, root, tmp_path / "other_work", **SETTINGS)


def _approved_then_changed_mid_pass(env, tmp_path, monkeypatch, change):
    """A is staged and approved; during the next pass, while the watcher stages B, the
    vendor changes A (`change`). The old listing still shows A as approved and current."""
    store, root, _ = env
    b = Ref("acme", "episode_0002")
    deliver(store, tmp_path, A)
    run(env)
    store.add_decision(A, "approve", "simar", stage=status(env, A)["stage"])
    deliver(store, tmp_path, b)
    real = formats.prepare_delivery

    def staging_b(delivery, staged, *a, **kw):
        if "episode_0002" in str(staged):
            change(store)
        return real(delivery, staged, *a, **kw)

    monkeypatch.setattr(formats, "prepare_delivery", staging_b)
    return run(env)


def test_a_correction_started_mid_pass_blocks_the_old_approval(
    env, tmp_path, monkeypatch
):
    s = _approved_then_changed_mid_pass(
        env, tmp_path, monkeypatch, lambda store: store.start_upload(A, "episode.mcap")
    )
    assert s.committed == []  # A is no longer complete: its old stage must not land
    assert status(env, A)["state"] == "ready"
    assert not (env[1] / "recordings.lance").exists()


def test_a_re_upload_mid_pass_blocks_the_old_approval(env, tmp_path, monkeypatch):
    def reupload(store):
        deliver(store, tmp_path, A, name="v2", metadata_override={"scene": "kitchen_b"})

    s = _approved_then_changed_mid_pass(env, tmp_path, monkeypatch, reupload)
    assert s.committed == []
    assert not (env[1] / "recordings.lance").exists()
    s = run(env)  # the next pass stages the correction for a fresh review
    assert s.staged == ["acme/episode_0001"] and s.committed == []
    assert status(env, A)["diff"]["action"] == "new"


def test_mano_that_cannot_run_stops_the_job_before_any_delivery(
    env, tmp_path, monkeypatch
):
    store, root, work = env
    deliver(store, tmp_path, A)
    monkeypatch.setattr(watch, "mano_problem", lambda d: "MANO is not installed")
    args = ["--lance", str(root), "--work", str(work), "--local-store", str(store.root)]
    assert watch.main([*args, "--codec", "libx264", "--mano"]) == 2
    assert status(env, A) is None  # nothing touched, nothing marked failed
    assert watch.main([*args, "--codec", "libx264", "--no-mano"]) == 0
    assert status(env, A)["state"] == "ready"


def test_mano_problem_names_the_missing_model_files(tmp_path, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "mano", types.ModuleType("mano"))  # installed
    (tmp_path / "MANO_LEFT.pkl").write_bytes(b"left")
    assert "MANO_RIGHT.pkl" in watch.mano_problem(str(tmp_path))
    (tmp_path / "MANO_RIGHT.pkl").write_bytes(b"right")
    assert watch.mano_problem(str(tmp_path)) is None


class Killed(BaseException):
    """A kill (timeout, preemption): not an Exception, so the pass cannot catch it."""


def test_a_pass_killed_while_staging_is_restarted(env, tmp_path, monkeypatch):
    store, _, _ = env
    deliver(store, tmp_path, A)

    def killed(*a, **kw):
        raise Killed

    with monkeypatch.context() as m:
        m.setattr(formats, "prepare_delivery", killed)
        with pytest.raises(Killed):
            run(env)
    assert status(env, A)["state"] == "processing"  # left behind by the kill
    s = run(env)
    assert s.staged == ["acme/episode_0001"] and status(env, A)["state"] == "ready"


@pytest.mark.parametrize("writes", [1, 2, 4])
def test_a_pass_killed_while_committing_resumes_it(env, tmp_path, monkeypatch, writes):
    """Killed after the n-th of the commit's Lance table writes; the next pass finishes
    the approved stage, without duplicating rows."""
    from egomimic.rldb.lance import commit

    store, root, _ = env
    deliver(store, tmp_path, A)
    run(env)
    stage = status(env, A)["stage"]
    store.add_decision(A, "approve", "simar", stage=stage)
    count = {"n": 0}
    with monkeypatch.context() as m:
        for name in ("_append", "_replace"):
            real = getattr(commit, name)

            def wrapped(*a, _real=real, **kw):
                _real(*a, **kw)
                count["n"] += 1
                if count["n"] == writes:
                    raise Killed

            m.setattr(commit, name, wrapped)
        with pytest.raises(Killed):
            run(env)
    assert status(env, A)["state"] == "committing"  # left behind by the kill
    assert store.has_staged(A, stage)  # the approved stage is still there
    s = run(env)
    assert s.committed == ["acme/episode_0001"]
    assert status(env, A)["state"] == "committed"
    assert status(env, A)["committed"]["status"] in (
        "committed",
        "resumed",
        "unchanged",
    )
    steps = lance.dataset(root / "steps/aria.lance").to_table().num_rows
    assert steps == 60 and len(lance_rows(root, "recordings.lance")) == 1
    assert not store.has_staged(A, stage)
