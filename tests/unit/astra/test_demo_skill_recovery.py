"""Recovery preserves evidence, call accounting and original prompt inputs."""

import base64
import copy
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from astra_reversal.demo_segments import write_json
from astra_reversal.demo_skill_agent import (
    PROMPT_TEMPLATE_VERSION,
    SYSTEM_PROMPT,
    build_request,
    parse_proposal,
)
from astra_reversal.demo_skill_experiment import (
    NATIVE,
    DemoSkillExperiment,
    load_protocol,
)
from astra_reversal.demo_skill_recovery import RecoveryArchive, RecoveryClient, confined
from astra_reversal.osmo.demo_skill_recovery import unpack_archive, validate_manifest
from astra_reversal.osmo.relay_health import ForwardHealth
from astra_reversal.records import Recorder, digest, file_sha256


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "results"
    bank_dir = root / "demo_bank"
    bank_dir.mkdir(parents=True)
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    Image.fromarray(image).save(bank_dir / "overview.png")
    bank = SimpleNamespace(
        bank_id="b" * 64,
        directory=bank_dir,
        catalog=lambda: [
            {
                "source_id": "10",
                "prompt": "put the bowl on the plate",
                "frame_count": 30,
            }
        ],
    )
    entry = {
        "episode_id": "episode",
        "suite": "libero_goal_ood",
        "task_id": 0,
        "instruction": "put cream cheese in the basket",
        "seed": 97,
        "initial_state_id": 0,
    }
    snapshot = {
        "step": 0,
        "observation": {
            "observation/image": image,
            "observation/wrist_image": image.copy(),
            "observation/state": np.zeros(8, np.float32),
        },
    }
    attempt = "native_dev0"
    result = {
        "attempt_id": attempt,
        "episode_id": "episode",
        "program_sha256": digest(NATIVE),
        "split": "development",
        "success": False,
        "reset_audit": {"initial_state_id": 0},
        "snapshots": [snapshot],
    }
    path = root / "experiment/rollouts" / attempt
    recorder = Recorder(path)
    recorder.event(
        "program_start",
        entry=entry,
        program=NATIVE,
        arm="action_composition",
        split="development",
    )
    recorder.event("rollout_result", result=result)
    files = {
        str(p.relative_to(path)): file_sha256(p) for p in path.rglob("*") if p.is_file()
    }
    write_json(path / "trace_manifest.json", {"files": files, "sha256": digest(files)})
    summary = {k: v for k, v in result.items() if k != "snapshots"}
    summary["trace_sha256"] = digest(files)
    write_json(path / "summary.json", summary)
    request = build_request(
        role="select_sources",
        arm="action_composition",
        task=entry["instruction"],
        episode_id="episode",
        attempt_id="request1",
        request_index=1,
        bank=bank,
        initial_snapshot=snapshot,
        history=[],
        library=[],
    )
    decision = parse_proposal(
        {
            "selected_sources": ["10"],
            "program": NATIVE,
            "failure_hypothesis": "A test trace failed",
            "expected_effect": "Select a source",
        },
        request,
    )
    row = {
        "accepted": True,
        "request_fingerprint": request["request_fingerprint"],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "configured_model": "gpt-6-astra",
        "reasoning_effort": "medium",
        "token_usage": {"total_tokens": 123},
    }
    unknown = {
        "accepted": False,
        "invocation_id": "timed-out",
        "token_usage": None,
        "job_start_unknown": True,
        "token_usage_is_lower_bound": True,
    }
    protocol = load_protocol()
    report = {
        "arm": "action_composition",
        "bank_id": bank.bank_id,
        "protocol": protocol,
        "status": "provider_or_contract_error",
        "physical_rollouts": [summary],
        "provider_records": [row, unknown],
    }
    write_json(root / "experiment/summary.json", report)
    write_json(root / "experiment/requests/00001.json", request)
    write_json(root / "experiment/decisions/00001.json", decision)
    write_json(
        root / "runtime.json", {"workflow": "old-run", "payload_sha256": "c" * 64}
    )
    archive = RecoveryArchive(
        root,
        arm=report["arm"],
        bank_id=bank.bank_id,
        protocol=protocol,
        namespace="new-run",
    )
    return SimpleNamespace(
        root=root,
        bank=bank,
        entry=entry,
        snapshot=snapshot,
        request=request,
        decision=decision,
        result=result,
        summary=summary,
        archive=archive,
    )


def test_restores_real_arrays_without_executing_or_double_counting(evidence, tmp_path):
    e = evidence

    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery called the simulator")

    client = SimpleNamespace(records=[], response_log=tmp_path / "provider.jsonl")
    experiment = DemoSkillExperiment(
        None,
        forbidden,
        None,
        e.bank,
        load_protocol(),
        tmp_path / "continued",
        "action_composition",
        client,
        recovery=e.archive,
    )
    result = experiment.run_program(e.entry, NATIVE, "native_dev0", "development")
    assert experiment.report["physical_rollouts"] == [e.summary]
    np.testing.assert_array_equal(
        result["snapshots"][0]["observation"]["observation/image"],
        e.snapshot["observation"]["observation/image"],
    )
    assert experiment.reset_audits["episode"] == e.result["reset_audit"]
    with pytest.raises(FileExistsError):
        experiment.run_program(e.entry, NATIVE, "native_dev0", "development")


def test_cached_decisions_keep_unknown_cost_and_new_calls_have_new_identity(
    evidence, tmp_path
):
    e = evidence
    e.archive.rollout(
        e.entry, NATIVE, "native_dev0", "development", tmp_path / "replayed"
    )
    fresh = []
    client = SimpleNamespace(
        records=[],
        response_log=tmp_path / "provider.jsonl",
        propose=lambda r: fresh.append(r) or "new-response",
    )
    recovered = RecoveryClient(client, e.archive)
    assert recovered.propose(e.request) == e.decision
    assert not fresh
    assert len(client.records) == 1
    assert e.archive.attempt_name(1, "ignored") == "request1"
    assert e.archive.attempt_name(2, "request2") == "request2:continuation:new-run"
    assert recovered.propose({"request_index": 2}) == "new-response"
    assert len(fresh) == 1
    assert client.records == e.archive.records
    assert json.loads(client.response_log.read_text().splitlines()[-1])[
        "job_start_unknown"
    ]
    e.archive.exhausted()


@pytest.mark.parametrize("change", ["entry", "program", "split", "bytes"])
def test_refuses_changed_episode_program_or_trace(evidence, tmp_path, change):
    e = evidence
    entry, program, split = copy.deepcopy(e.entry), copy.deepcopy(NATIVE), "development"
    if change == "entry":
        entry["initial_state_id"] = 1
    if change == "program":
        program["native"] = False
    if change == "split":
        split = "evaluation"
    if change == "bytes":
        array = next((e.root / "experiment/rollouts/native_dev0/arrays").glob("*.npy"))
        array.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differ|change"):
        e.archive.rollout(entry, program, "native_dev0", split, tmp_path / "replayed")


@pytest.mark.parametrize(
    "field,value",
    [
        ("configured_model", "another-model"),
        ("reasoning_effort", "low"),
        ("system_prompt_sha256", "bad"),
        ("request_fingerprint", "bad"),
    ],
)
def test_changed_provider_configuration_is_not_reused(evidence, tmp_path, field, value):
    evidence.archive.records[0][field] = value
    client = SimpleNamespace(records=[], response_log=tmp_path / "provider.jsonl")
    with pytest.raises(ValueError, match="prompt or identity"):
        RecoveryClient(client, evidence.archive).propose(evidence.request)
    assert client.records == []


def test_png_recompression_preserves_original_wire_but_changed_pixels_fail(evidence):
    e = evidence
    request = copy.deepcopy(e.request)
    wire = request["initial_snapshot"]["images"][0]
    stream = io.BytesIO()
    Image.fromarray(e.snapshot["observation"]["observation/image"]).save(
        stream, format="PNG", compress_level=0
    )
    wire["data"] = base64.b64encode(stream.getvalue()).decode()
    request.pop("request_fingerprint")
    request["request_fingerprint"] = digest(request)
    assert e.archive.bind_request(request) == e.request
    changed = e.snapshot["observation"]["observation/image"].copy()
    changed[0, 0, 0] += 1
    stream = io.BytesIO()
    Image.fromarray(changed).save(stream, format="PNG")
    wire["data"] = base64.b64encode(stream.getvalue()).decode()
    request.pop("request_fingerprint")
    request["request_fingerprint"] = digest(request)
    with pytest.raises(ValueError, match="complete recorded inputs"):
        e.archive.bind_request(request)


def test_cannot_skip_unconsumed_evidence(evidence, tmp_path):
    with pytest.raises(ValueError, match="skipped"):
        evidence.archive.rollout(
            evidence.entry, NATIVE, "new-rollout", "development", tmp_path / "new"
        )
    with pytest.raises(ValueError, match="consume"):
        evidence.archive.exhausted()


@pytest.mark.parametrize("relative", ["../escape", "/absolute", None, 123])
def test_confines_artifacts(tmp_path, relative):
    with pytest.raises(ValueError):
        confined(tmp_path, relative)


@pytest.mark.parametrize(
    "member", ["results/../../outside", "/outside", "another-root/a"]
)
def test_archive_rejects_escaping_members(tmp_path, member):
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.size = 1
        archive.addfile(info, io.BytesIO(b"a"))
    with pytest.raises(ValueError, match="Unsafe"):
        unpack_archive(
            path,
            tmp_path / "unpacked",
            {"bytes": path.stat().st_size, "sha256": file_sha256(path)},
        )


def test_recovery_manifest_is_bound_to_worker_and_owned_prefix():
    workflow = "astra-pi05-input-skill-library-20260930-full-2"
    value = {
        "schema_version": "demo-skill-recovery-1",
        "source_workflow": workflow,
        "workers": [
            {
                "worker": i,
                "arm": arm,
                "bytes": 1,
                "sha256": "a" * 64,
                "key": f"experiments/astra-reversal-20260924/{workflow}/worker_{i}/artifacts.tar.gz",
            }
            for i, arm in enumerate(load_protocol()["arms"])
        ],
    }
    assert validate_manifest(value) == value
    value["workers"][0]["key"] = value["workers"][1]["key"]
    with pytest.raises(ValueError, match="worker"):
        validate_manifest(value)


def test_forward_health_detects_hung_process_and_periodically_renews():
    monitor = ForwardHealth(0)
    # A worker can be bootstrapping without a relay server for several minutes.
    for now in (20, 40, 60):
        assert monitor.observe(False, now) is None
    assert monitor.observe(True, 70) is None
    assert monitor.observe(False, 80) is None
    assert monitor.observe(True, 90) is None
    assert monitor.observe(False, 100) is None
    assert monitor.observe(False, 110) is None
    assert monitor.observe(False, 120) == "three_failed_health_checks"
    assert ForwardHealth(0).observe(True, 900) == "periodic_transport_renewal"
