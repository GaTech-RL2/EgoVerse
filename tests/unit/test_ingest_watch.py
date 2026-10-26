"""The watcher pass (egomimic/ingest/watch.py) on a local-directory store, with conversion
jobs run in-process and the committer (lance_commit) run after each pass."""

import json
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
from egomimic.ingest.launch import ENDED, RUNNING, Launcher, LocalLauncher  # noqa: E402
from egomimic.ingest.run import commit_round  # noqa: E402
from egomimic.ingest.store import LocalStore, Ref  # noqa: E402
from egomimic.ingest.watch import Busy, run_pass  # noqa: E402

SETTINGS = dict(codec="libx264", mano=False)
A = Ref("acme", "episode_0001")
B = Ref("acme", "episode_0002")


class Env:
    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.store = LocalStore(tmp_path / "buckets")
        self.root = tmp_path / "lance"
        self.work = tmp_path / "work"
        self.raw = tmp_path / "raw"
        self.staging = tmp_path / "lance_staging"
        self.launcher = LocalLauncher(
            self.store, str(self.root), tmp_path / "jobs", **SETTINGS
        )

    def run(self, launcher=None, **kw):
        kw = {"revision": "r1", **kw}
        return run_pass(
            self.store, launcher or self.launcher, self.work,
            raw_root=str(self.raw), lance_staging=str(self.staging), **kw,
        )  # fmt: skip

    def commit(self):
        return commit_round(str(self.staging), str(self.root), self.work)

    def status(self, ref):
        return self.store.get_status(ref)

    def deliver(self, ref, name="v1", **kw):
        """Upload a fixture episode as a lab machine would, and mark it complete."""
        ep = write_mcap_episode(self.tmp / f"src_{ref.delivery_id}_{name}", **kw)
        self.store.upload_delivery(ref, ep, {"by": "test"})

    def approve(self, ref, who="simar"):
        self.store.add_decision(ref, "approve", who, stage=self.status(ref)["stage"])

    def rows(self, table):
        return lance.dataset(self.root / table).to_table().to_pylist()


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def test_a_delivery_is_staged_then_handed_to_the_committer_on_approval(env):
    env.deliver(A)
    s = env.run()
    assert s.submitted == s.staged == ["acme/episode_0001"]
    st = env.status(A)
    assert st["state"] == "ready" and st["revision"] == "r1"
    assert st["diff"]["action"] == "new" and st["report"]["num_frames"] == 60
    assert env.store.has_staged(A, st["stage"])
    assert not env.staging.exists() and not env.root.exists()  # nothing before approval

    env.approve(A)
    s = env.run()
    assert s.approved == ["acme/episode_0001"]
    st = env.status(A)
    assert st["state"] == "approved"
    assert env.store.get_delivery(A) is None and not env.store.has_staged(
        A, st["stage"]
    )
    raw = env.raw / "acme" / "episode_0001" / st["stage"]
    assert (raw / "episode.mcap").is_file()
    meta = json.loads((env.staging / "mcap" / "acme-rec-0" / "meta.json").read_text())
    assert meta["provenance"]["stage"] == st["stage"]
    assert meta["provenance"]["raw"] == str(raw)

    env.commit()
    assert env.run().committed == ["acme/episode_0001"]
    assert env.status(A)["state"] == "committed"
    (rec,) = env.rows("recordings.lance")
    assert rec["recording_id"] == st["recording_id"]
    assert len(env.rows("annotations.lance")) == 2
    (change,) = [c for c in env.rows("changes.lance") if c["source"] == "mcap"]
    prov = json.loads(change["provenance"])[st["recording_id"]]
    assert prov["stage"] == st["stage"] and prov["settings"]["codec"] == "libx264"
    assert prov["converter"]["revision"] not in ("", "unknown")
    assert [h["state"] for h in env.status(A)["history"]] == [
        "converting", "ready", "approved", "committed",
    ]  # fmt: skip
    s = env.run()  # nothing new: a no-op
    assert s.submitted == s.committed == []


def test_incomplete_deliveries_are_left_alone(env):
    uid = env.store.start_upload(A, "episode.mcap")
    env.store.upload_part(A, "episode.mcap", uid, 1, b"half an upload")
    assert env.run().submitted == [] and env.status(A) is None


def test_a_vendors_own_upload_counts_once_done_is_written(env):
    ep = write_mcap_episode(env.tmp / "own")
    env.store._upload_file(A.file_key("episode.mcap"), ep / "episode.mcap")
    assert env.run().submitted == []
    env.store._put(A.prefix + "_done.json", b"{}")
    assert env.run().staged == ["acme/episode_0001"]


def test_rejected_and_unsupported_deliveries_say_why(env):
    env.deliver(A, imu_hz=100)
    folder = env.tmp / "vrs"
    folder.mkdir()
    (folder / "recording.vrs").write_bytes(b"raw")
    b = Ref("acme", "raw_vrs")
    env.store.upload_delivery(b, folder, {})
    s = env.run()
    assert sorted(s.rejected) == ["acme/episode_0001", "acme/raw_vrs"]
    assert env.status(A)["state"] == "rejected"
    assert env.status(A)["problems"] == ["[imu] /imu: 100 Hz, need >= 200"]
    assert env.status(b)["state"] == "unsupported"
    assert "recording.vrs" in env.status(b)["problems"][0]


def test_oversized_deliveries_are_rejected_without_converting(env, monkeypatch):
    env.deliver(A)
    small = formats.Format(**{**formats.by_name("mcap").__dict__, "max_file_bytes": 10})
    monkeypatch.setattr(formats, "formats", lambda: [small])
    monkeypatch.setattr(
        env.launcher,
        "submit",
        lambda *a: pytest.fail("converted an oversized delivery"),
    )
    assert env.run().rejected == ["acme/episode_0001"]
    assert "over the mcap limit" in env.status(A)["problems"][0]


def test_a_decline_drops_the_stage_and_never_commits(env):
    env.deliver(A)
    env.run()
    stage = env.status(A)["stage"]
    env.store.add_decision(A, "decline", "simar", "wrong task", stage=stage)
    assert env.run().declined == ["acme/episode_0001"]
    assert env.status(A)["state"] == "declined"
    assert not env.store.has_staged(A, stage)
    assert not env.staging.exists()


def test_an_approval_only_applies_to_the_stage_it_was_given_for(env):
    env.deliver(A)
    env.run()
    first = env.status(A)["stage"]
    env.deliver(A, name="v2", metadata_override={"scene": "kitchen_b"})
    env.store.add_decision(A, "approve", "simar", stage=first)  # the old stage
    s = env.run()
    assert s.staged == ["acme/episode_0001"] and not s.approved
    second = env.status(A)["stage"]
    assert second != first and not env.store.has_staged(A, first)  # superseded, gone
    assert env.run().approved == []  # the old approval does not carry over
    env.approve(A)
    assert env.run().approved == ["acme/episode_0001"]
    env.commit()
    assert env.rows("recordings.lance")[0]["scene"] == "kitchen_b"


def test_restaging_a_committed_recording_waits_for_replace(env):
    env.deliver(A)
    env.run()
    env.approve(A)
    env.run()
    env.commit()
    env.run()
    env.deliver(A, name="v2", labels_in=None)  # labels withdrawn
    env.run()
    st = env.status(A)
    assert st["state"] == "ready" and st["diff"]["action"] == "replace"
    # A delivery speaks for its lab's labels and the sources it names: the label acme
    # wrote ("pick up the shirt") is not withdrawn by a delivery that names no acme labels.
    assert st["diff"]["labels"]["removed"] == ["drop it in the basket"]
    env.approve(A)
    env.run()
    env.commit()
    assert env.run().commit_rejected == ["acme/episode_0001"]
    assert "already committed" in env.status(A)["problems"][0]
    assert len(env.rows("annotations.lance")) == 2  # untouched until someone replaces

    from egomimic.scripts.lance_commit import commit_staged

    commit_staged(str(env.staging), "mcap", str(env.root), env.work, replace=True)
    assert [a["text"] for a in env.rows("annotations.lance")] == ["pick up the shirt"]


def test_one_failing_delivery_does_not_stop_the_pass(env, monkeypatch):
    env.deliver(A)
    env.deliver(B)
    real = formats.prepare_delivery

    def flaky(delivery, staged, *a, **kw):
        if "episode_0001" in str(staged):
            raise RuntimeError("disk full")
        return real(delivery, staged, *a, **kw)

    monkeypatch.setattr(formats, "prepare_delivery", flaky)
    s = env.run()
    assert s.failed == ["acme/episode_0001"] and s.staged == ["acme/episode_0002"]
    assert "disk full" in env.status(A)["error"]


class Held(Launcher):
    """Submits, but nothing runs until `finish` (a Batch queue)."""

    def __init__(self, env, ended=False):
        self.env, self.jobs, self.ended = env, {}, ended

    def submit(self, ref, fingerprint):
        self.jobs[f"j{len(self.jobs)}"] = (ref, fingerprint)
        return f"j{len(self.jobs) - 1}"

    def state(self, job_id):
        return ENDED if self.ended else RUNNING

    def finish(self):
        for ref, fp in self.jobs.values():
            self.env.launcher.submit(ref, fp)


def test_jobs_in_flight_are_capped_and_the_rest_wait(env):
    for i in range(4):
        env.deliver(Ref("acme", f"e{i}"))
    held = Held(env)
    s = env.run(held, max_in_flight=2, canary=10)
    assert len(s.submitted) == 2 and len(s.queued) == 2
    assert {env.status(Ref("acme", f"e{i}"))["state"] for i in range(4)} == {
        "converting",
        "queued",
    }
    s = env.run(held, max_in_flight=2, canary=10)  # still running: nothing new starts
    assert s.submitted == [] and len(s.queued) == 2
    held.finish()
    s = env.run(held, max_in_flight=2, canary=10)
    assert len(s.staged) == 2 and len(s.submitted) == 2


def test_a_vendor_converts_only_a_canary_until_one_is_approved(env):
    for i in range(5):
        env.deliver(Ref("acme", f"e{i}"))
    env.deliver(Ref("beta", "e0"))
    s = env.run(canary=2)
    assert sorted(s.staged) == ["acme/e0", "acme/e1", "beta/e0"]
    assert len(s.queued) == 3
    assert env.run(canary=2).submitted == []  # the canary waits for review
    env.approve(Ref("acme", "e0"))
    assert env.run(canary=2).approved == ["acme/e0"]
    s = env.run(canary=2)
    assert len(s.staged) == 3  # approved at r1: the rest of acme goes
    for i in range(5, 8):
        env.deliver(Ref("acme", f"e{i}"))
    s = env.run(canary=2, revision="r2")  # a new converter starts a new canary
    assert len(s.submitted) == 2 and len(s.queued) == 1


def test_a_job_that_ends_without_an_outcome_fails_the_delivery(env):
    env.deliver(A)
    s = env.run(Held(env, ended=True))
    assert s.submitted == s.failed == ["acme/episode_0001"]
    assert "without an outcome" in env.status(A)["error"]


def test_a_re_upload_while_converting_is_converted_again(env):
    env.deliver(A)
    held = Held(env)
    env.run(held)
    env.deliver(A, name="v2", metadata_override={"scene": "kitchen_b"})
    held.finish()  # the job sees the new version: superseded
    s = env.run(held)
    assert env.status(A)["state"] == "converting" and s.submitted == [
        "acme/episode_0001"
    ]


def test_a_correction_started_after_approval_blocks_the_hand_off(env):
    env.deliver(A)
    env.run()
    env.approve(A)
    env.store.start_upload(A, "episode.mcap")
    assert env.run().approved == []
    assert env.status(A)["state"] == "ready" and not env.staging.exists()


def test_a_second_pass_is_turned_away_while_one_runs(env):
    with watch._flock(env.work / ".watch.lock"):
        with pytest.raises(Busy):
            env.run()


def test_mano_that_cannot_run_stops_a_local_pass_before_any_delivery(env, monkeypatch):
    env.deliver(A)
    monkeypatch.setattr(watch, "mano_problem", lambda d: "MANO is not installed")
    args = [
        "--lance", str(env.root), "--work", str(env.work),
        "--local-store", str(env.store.root), "--launcher", "local",
        "--raw-root", str(env.raw), "--lance-staging", str(env.staging),
        "--codec", "libx264",
    ]  # fmt: skip
    assert watch.main([*args, "--mano"]) == 2
    assert env.status(A) is None  # nothing touched, nothing marked failed
    assert watch.main([*args, "--no-mano"]) == 0
    assert env.status(A)["state"] == "ready"


def test_mano_problem_names_the_missing_model_files(tmp_path, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "mano", types.ModuleType("mano"))  # installed
    (tmp_path / "MANO_LEFT.pkl").write_bytes(b"left")
    assert "MANO_RIGHT.pkl" in watch.mano_problem(str(tmp_path))
    (tmp_path / "MANO_RIGHT.pkl").write_bytes(b"right")
    assert watch.mano_problem(str(tmp_path)) is None


def test_batch_launcher_submits_the_job_command():
    from egomimic.ingest.launch import BatchLauncher

    class FakeBatch:
        def __init__(self):
            self.submitted = []

        def submit_job(self, **kw):
            self.submitted.append(kw)
            return {"jobId": "abc"}

        def describe_jobs(self, jobs):
            return {"jobs": [{"status": "RUNNABLE"}]}

    fake = FakeBatch()
    launcher = BatchLauncher("q", "d", ["--lance", "s3://rldb/lance"], client=fake)
    assert launcher.submit(Ref("acme", "ep.1"), "f" * 64) == "abc"
    (job,) = fake.submitted
    assert job["jobQueue"] == "q" and job["jobName"] == "convert-acme-ep-1-ffffffffffff"
    cmd = job["containerOverrides"]["command"]
    assert cmd[:3] == ["python", "-m", "egomimic.ingest.convert_job"]
    assert cmd[cmd.index("--fingerprint") + 1] == "f" * 64
    assert launcher.state("abc") == RUNNING
