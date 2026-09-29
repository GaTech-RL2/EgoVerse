"""Corruption and scope checks for the independent archive-byte audit."""

import gzip
import hashlib
import io
import json
import tarfile
import urllib.error

import pytest

from astra_reversal import recipe_archive_audit as audit


def encoded(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def record(value):
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def packed(files, *, root="results", extra=None):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz") as stream:
        for name, value in files.items():
            item = tarfile.TarInfo(root + "/" + name)
            item.size = len(value)
            stream.addfile(item, io.BytesIO(value))
        if extra:
            item, value = extra
            stream.addfile(item, io.BytesIO(value) if value is not None else None)
    return target.getvalue()


def artifact(phase="evaluation"):
    workflow = "test-recipe-" + phase
    training = {
        "schema_version": "recipe-training-1.0",
        "status": "complete",
        "base_weights_unchanged": True,
        "provider_calls": 0,
        "provider_tokens": 0,
        "workflow": workflow,
    }
    files = {
        "runtime.json": encoded(
            {
                "phase": phase,
                "worker": 0,
                "workflow": workflow,
                "provider_credential_attached": False,
                "tf32": False,
            }
        ),
        "progress.json": encoded({"status": "complete", "phase": phase}),
        "protocol.json": encoded({"frozen": True}),
        "training_receipt.json": encoded(training),
        "rollouts/task_0/native/summary.json": encoded({"status": "complete"}),
        "rollouts/task_0/native/events.jsonl": encoded({"sequence": 0}),
        "rollouts/task_0/native/arrays/000000.npy": b"not decoded by this byte-only auditor",
        "rollouts/task_0/native/rollout.mp4": b"recorded-video-fixture",
    }
    if phase == "training":
        bundle = {
            "training_receipt.json": encoded(training),
            "protocol.json": files["protocol.json"],
            "head/state.pt": b"frozen-checkpoint-bytes",
            "selector/normalization.npz": b"frozen-normalization-bytes",
        }
        bundle["seal.json"] = encoded(
            {name: record(value) for name, value in bundle.items()}
        )
        files.update({"bundle/" + name: value for name, value in bundle.items()})
        files["learned_bundle.tar.gz"] = packed(bundle, root="bundle")
        files["learned_bundle.json"] = encoded(
            {
                "schema_version": "recipe-bundle-1.0",
                **record(files["learned_bundle.tar.gz"]),
            }
        )
    else:
        files["completion_receipt.json"] = encoded(
            {
                "schema_version": "recipe-evaluation-worker-1.0",
                "status": "complete",
                "worker": 0,
                "provider_calls": 0,
                "provider_tokens": 0,
                "files": {name: record(value) for name, value in files.items()},
            }
        )
    return files


def run(tmp_path, files, *, phase="evaluation", data=None, **kwargs):
    archive = tmp_path / "input.tar.gz"
    archive.write_bytes(data if data is not None else packed(files))
    return audit.audit_archive(
        archive,
        tmp_path / "verified",
        phase=phase,
        worker=0,
        workflow="test-recipe-" + phase,
        minimum_free_bytes=0,
        **kwargs,
    )


def test_eval_hashes_every_member_but_keeps_no_array_or_whole_archive(tmp_path):
    files = artifact()
    video = "rollouts/task_0/native/rollout.mp4"
    receipt = run(tmp_path, files, retain_videos=[video])
    output = tmp_path / "verified"
    inventory = json.loads((output / "member_inventory.json").read_text())
    assert inventory == {name: record(value) for name, value in files.items()}
    assert (
        receipt["archive"]["sha256"]
        == record((tmp_path / "input.tar.gz").read_bytes())["sha256"]
    )
    assert receipt["archive"]["gzip_crc_verified"]
    assert (
        receipt["member_inventory"]["sha256"]
        == record((output / "member_inventory.json").read_bytes())["sha256"]
    )
    assert receipt["checks"]["completion_seal_verified"]
    assert receipt["checks"]["array_descriptors_verified"] is False
    assert (output / "results" / video).read_bytes() == files[video]
    assert not list(output.rglob("*.npy"))
    assert not list(output.rglob("*.tar.gz"))
    assert json.loads((output / "audit.json").read_text()) == receipt


def test_training_seals_and_embedded_bundle_are_distinct_from_worker_seal(tmp_path):
    receipt = run(tmp_path, artifact("training"), phase="training")
    assert receipt["checks"]["training_bundle_seal_verified"]
    assert receipt["checks"]["embedded_bundle_archive_verified"]
    assert receipt["checks"]["completion_seal_verified"] is False
    assert (
        tmp_path / "verified/results/bundle/head/state.pt"
    ).read_bytes() == b"frozen-checkpoint-bytes"


@pytest.mark.parametrize("corruption", ["crc", "truncated", "hidden_trailer"])
def test_tar_end_does_not_bypass_gzip_integrity(tmp_path, corruption):
    files = artifact()
    data = packed(files)
    if corruption == "crc":
        data = data[:-8] + bytes([data[-8] ^ 1]) + data[-7:]
    elif corruption == "truncated":
        data = data[:-3]
    else:
        data = gzip.compress(gzip.decompress(data) + b"unlisted-payload")
    with pytest.raises(audit.AuditError):
        run(tmp_path, files, data=data)
    assert not (tmp_path / "verified").exists()
    assert not list(tmp_path.glob(".recipe-audit-*"))


@pytest.mark.parametrize(
    "change", ["alter_array", "omit_array", "extra_member", "seal_self", "wrong_bytes"]
)
def test_completion_seal_requires_exact_inventory(tmp_path, change):
    files = artifact()
    name = "rollouts/task_0/native/arrays/000000.npy"
    if change == "alter_array":
        files[name] += b"corruption"
    elif change == "omit_array":
        files.pop(name)
    elif change == "extra_member":
        files["unsealed.txt"] = b"extra"
    else:
        seal = json.loads(files["completion_receipt.json"])
        if change == "seal_self":
            seal["files"]["completion_receipt.json"] = record(b"fake")
        else:
            seal["files"][name]["bytes"] += 1
        files["completion_receipt.json"] = encoded(seal)
    with pytest.raises(audit.AuditError, match="completion_seal_mismatch"):
        run(tmp_path, files)


def test_nested_checkpoint_cannot_differ_even_with_valid_own_hash(tmp_path):
    files = artifact("training")
    nested = {
        name[len("bundle/") :]: value
        for name, value in files.items()
        if name.startswith("bundle/")
    }
    nested["head/state.pt"] += b"different-model"
    files["learned_bundle.tar.gz"] = packed(nested, root="bundle")
    files["learned_bundle.json"] = encoded(
        {
            "schema_version": "recipe-bundle-1.0",
            **record(files["learned_bundle.tar.gz"]),
        }
    )
    with pytest.raises(audit.AuditError, match="embedded_bundle_mismatch"):
        run(tmp_path, files, phase="training")


@pytest.mark.parametrize("kind", ["duplicate", "traversal", "symlink", "absolute"])
def test_unsafe_tar_members_rejected_without_extraction(tmp_path, kind):
    name = {
        "duplicate": "results/protocol.json",
        "traversal": "results/../escape",
        "symlink": "results/link",
        "absolute": "/escape",
    }[kind]
    member = tarfile.TarInfo(name)
    value = b"x"
    if kind == "symlink":
        member.type, member.linkname, value = tarfile.SYMTYPE, "/etc/passwd", None
    else:
        member.size = 1
    files = artifact()
    with pytest.raises(audit.AuditError):
        run(tmp_path, files, data=packed(files, extra=(member, value)))
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("field", ["expected_archive", "expected_files"])
def test_external_published_binding_mismatch_fails(tmp_path, field):
    wrong = record(b"wrong")
    options = {
        field: wrong if field == "expected_archive" else {"protocol.json": wrong}
    }
    with pytest.raises(audit.AuditError, match="mismatch"):
        run(tmp_path, artifact(), **options)


def test_retention_budget_aborts_atomically(tmp_path):
    with pytest.raises(audit.AuditError, match="retained_byte_limit_exceeded"):
        run(tmp_path, artifact(), maximum_retained_bytes=10)
    assert not (tmp_path / "verified").exists()
    assert not list(tmp_path.glob(".recipe-audit-*"))


def test_existing_audit_is_never_overwritten(tmp_path):
    files = artifact()
    run(tmp_path, files)
    before = (tmp_path / "verified/audit.json").read_bytes()
    with pytest.raises(audit.AuditError, match="audit_destination_exists"):
        run(tmp_path, files)
    assert (tmp_path / "verified/audit.json").read_bytes() == before


def test_remote_http_errors_hide_private_signed_url(monkeypatch):
    url = "https://private.example/object?secret=never-print"

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(url, 403, "private-message", {}, None)

    monkeypatch.setattr(audit.urllib.request, "urlopen", fail)
    with pytest.raises(audit.AuditError) as exc:
        audit._remote_open(url)
    assert str(exc.value) == "archive_http_403"


def test_remote_object_version_must_remain_unchanged(tmp_path, monkeypatch):
    data = packed(artifact())

    class Response(io.BytesIO):
        def __init__(self, *, probe):
            super().__init__(data[:1] if probe else data)
            self.status = 206 if probe else 200
            self.headers = {
                "Content-Length": str(1 if probe else len(data)),
                "ETag": '"changed"' if probe else '"original"',
            }
            if probe:
                self.headers["Content-Range"] = f"bytes 0-0/{len(data)}"

    monkeypatch.setattr(
        audit, "_remote_open", lambda _, probe=False: Response(probe=probe)
    )
    with pytest.raises(audit.AuditError, match="archive_changed_during_read"):
        audit.audit_archive(
            "https://private.example/signed",
            tmp_path / "verified",
            phase="evaluation",
            worker=0,
            workflow="test-recipe-evaluation",
            minimum_free_bytes=0,
        )
    assert not (tmp_path / "verified").exists()


def test_required_full_completion_cannot_be_replaced_with_progress_only(tmp_path):
    files = artifact()
    files.pop("completion_receipt.json")
    with pytest.raises(audit.AuditError, match="missing_required_structured_file"):
        run(tmp_path, files)
