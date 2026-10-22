"""The ingest box's tools: onboarding a vendor, the review CLI, the Batch setup and the
loop, without Cloudflare or AWS."""

import hashlib
import json
import shutil
import subprocess

import pytest

from egomimic.ingest import onboard, review
from egomimic.ingest.aws import setup_batch
from egomimic.ingest.store import LocalStore, Ref


def test_onboard_makes_one_bucket_and_a_key_for_only_that_bucket():
    calls = []

    def api(method, path, body):
        calls.append((method, path, body))
        if path.endswith("permission_groups"):
            return [
                {"id": "g-read", "name": "Workers R2 Storage Bucket Item Read"},
                {"id": "g-write", "name": onboard.WRITE_GROUP},
            ]
        if path.endswith("/tokens"):
            return {"id": "tok-id", "value": "tok-value"}
        return {}

    out = onboard.onboard(
        api, "acct", "scale", dashboard_origins=["https://ingest.example.org"]
    )
    assert out == {
        "bucket": "ingest-scale",
        "access_key_id": "tok-id",
        "secret_access_key": hashlib.sha256(b"tok-value").hexdigest(),
        "endpoint": "https://acct.r2.cloudflarestorage.com",
    }
    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/accounts/acct/r2/buckets"),
        ("PUT", "/accounts/acct/r2/buckets/ingest-scale/cors"),
        ("PUT", "/accounts/acct/r2/buckets/ingest-scale/lifecycle"),
        ("GET", "/accounts/acct/tokens/permission_groups"),
        ("POST", "/accounts/acct/tokens"),
    ]
    cors = calls[1][2]["rules"][0]
    assert cors["allowed"]["methods"] == ["PUT"] and cors["exposeHeaders"] == ["ETag"]
    (policy,) = calls[4][2]["policies"]
    assert policy["resources"] == {
        "com.cloudflare.edge.r2.bucket.acct_default_ingest-scale": "*"
    }
    assert policy["permission_groups"] == [{"id": "g-write"}]
    with pytest.raises(ValueError):
        onboard.onboard(api, "acct", "Scale Inc", dashboard_origins=[])


def test_onboard_dry_run_sends_nothing(capsys):
    assert onboard.main(["scale", "--account", "acct", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "POST /accounts/acct/r2/buckets" in out and "ingest-scale" in out


def _ready(store, ref, stage):
    store.put_status(ref, {"state": "ready", "stage": stage, "report": {"labels": 2}})


def test_review_decides_on_the_stage_in_the_status(tmp_path, capsys):
    store = LocalStore(tmp_path / "buckets")
    a, b, c = Ref("acme", "a"), Ref("acme", "b"), Ref("beta", "c")
    _ready(store, a, "1" * 64)
    _ready(store, b, "2" * 64)
    store.put_status(c, {"state": "converting"})
    args = ["--local-store", str(tmp_path / "buckets")]
    assert review.main([*args, "list", "--state", "ready"]) == 0
    assert capsys.readouterr().out.count("ready") == 2
    review.main([*args, "approve", "--all-ready", "--vendor", "acme", "--who", "simar"])
    assert [d["stage"] for d in store.decisions(a)] == ["1" * 64]
    assert [d["stage"] for d in store.decisions(b)] == ["2" * 64]
    out = review.decide(store, [c], "approve", "simar")
    assert out == {"beta/c": "skipped: converting, nothing to review"}
    with pytest.raises(SystemExit):  # a decline needs a reason
        review.main([*args, "decline", "acme/a", "--who", "simar"])


def test_batch_setup_asks_for_spot_gpus_that_scale_to_zero(capsys):
    args = [
        "--image", "ecr/egoverse-ingest:abc", "--subnets", "s1", "s2",
        "--security-group", "sg", "--instance-role", "ir", "--job-role", "jr",
        "--env-secret", "arn:secret",
    ]  # fmt: skip

    class FakeBatch:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def call(**kw):
                self.calls.append((name, kw))
                return {}

            return call

    fake = FakeBatch()
    assert setup_batch.main(args, client=fake) == 0
    (env_call, env), (_, queue), (_, definition) = fake.calls
    assert env_call == "create_compute_environment"
    res = env["computeResources"]
    assert res["type"] == "SPOT" and res["minvCpus"] == 0 and res["maxvCpus"] == 448
    assert all(t.startswith(("g5", "g6")) for t in res["instanceTypes"])
    needs = {
        r["type"]: r["value"]
        for r in definition["containerProperties"]["resourceRequirements"]
    }
    assert needs["GPU"] == "1" and needs["VCPU"] == "8"
    assert (
        queue["computeEnvironmentOrder"][0]["computeEnvironment"]
        == env["computeEnvironmentName"]
    )
    secrets = definition["containerProperties"]["secrets"]
    assert {s["name"] for s in secrets} == {
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
    }


def test_the_loop_runs_a_pass_then_the_committer(tmp_path, capsys):
    pytest.importorskip("lance")
    ffmpeg = shutil.which("ffmpeg")
    out = subprocess.run(
        [ffmpeg or "ffmpeg", "-hide_banner", "-encoders"], capture_output=True
    )
    if not ffmpeg or "libx264" not in out.stdout.decode():
        pytest.skip("needs ffmpeg with libx264")
    from fixtures.mcap_episodes import write_mcap_episode

    from egomimic.ingest import run

    store = LocalStore(tmp_path / "buckets")
    a = Ref("acme", "episode_0001")
    store.upload_delivery(a, write_mcap_episode(tmp_path / "src"), {})
    args = [
        "--local-store", str(tmp_path / "buckets"), "--launcher", "local",
        "--lance", str(tmp_path / "lance"), "--work", str(tmp_path / "work"),
        "--raw-root", str(tmp_path / "raw"),
        "--lance-staging", str(tmp_path / "lance_staging"),
        "--codec", "libx264", "--no-mano", "--once",
    ]  # fmt: skip
    assert run.main(args) == 0
    assert store.get_status(a)["state"] == "ready"
    store.add_decision(a, "approve", "simar", stage=store.get_status(a)["stage"])
    run.main(args)  # hands off, then commits
    rounds = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rounds[-1]["pass"]["approved"] == ["acme/episode_0001"]
    assert rounds[-1]["commit"]["aria"]["tags"][0].startswith("aria-")
    run.main(args)
    assert store.get_status(a)["state"] == "committed"
