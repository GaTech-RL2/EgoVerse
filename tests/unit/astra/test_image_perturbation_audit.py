import copy
import io
import json
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.image_perturbation_audit import (
    ArtifactStore,
    audit_case,
    audit_worker,
    replay_image_application,
)
from astra_reversal.image_perturbations import CAMERAS, apply_image_perturbations
from astra_reversal.records import Recorder, digest, file_sha256

from .conftest import SyntheticPolicy
from .test_image_perturbation_agent import proposal_for, snapshot
from .test_image_perturbation_search import SyntheticLibrary


def descriptor_case(tmp_path):
    recorder = Recorder(tmp_path / "case")
    recorder.event(
        "image_generation", pixels=np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    )
    store = ArtifactStore(recorder.directory)
    event = store.lines("events.jsonl")[0]
    store.register(event)
    return store, event


def test_array_inventory_proves_descriptor_hash_file_hash_and_coverage(tmp_path):
    store, event = descriptor_case(tmp_path)
    pixels = store.array(event["pixels"])
    assert digest(pixels) == event["pixels"]["sha256"]
    counts = store.finish()
    assert counts["array_files"] == counts["array_references"] == 1
    assert counts["array_bytes"] > pixels.nbytes


@pytest.mark.parametrize(
    "mutation", ["array", "extra", "missing", "descriptor", "late"]
)
def test_array_inventory_rejects_tampering_missing_and_unreferenced_files(
    tmp_path, mutation
):
    store, event = descriptor_case(tmp_path)
    path = store.directory / event["pixels"]["array"]
    if mutation == "array":
        np.save(path, np.zeros((2, 4, 3), np.uint8), allow_pickle=False)
    elif mutation == "extra":
        np.save(path.with_name("extra.npy"), np.zeros(1), allow_pickle=False)
    elif mutation == "missing":
        path.unlink()
    elif mutation == "descriptor":
        changed = copy.deepcopy(event)
        changed["pixels"]["sha256"] = "a" * 64
        with pytest.raises(ValueError, match="Conflicting"):
            store.register(changed)
        return
    else:
        store.array(event["pixels"])
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, FileNotFoundError)):
        store.finish()


def test_exact_pixel_replay_rejects_one_byte_or_state_change(tmp_path):
    raw = {
        CAMERAS[0]: np.zeros((224, 224, 3), np.uint8),
        CAMERAS[1]: np.ones((224, 224, 3), np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
    }
    library = SimpleNamespace(resolve=lambda *_: None)
    operations = [
        {
            "kind": "occlusion",
            "camera": CAMERAS[0],
            "box_xyxy": [0, 0, 112, 224],
            "fill_rgb": [127, 127, 127],
            "strength": 0.5,
        }
    ]
    modified, audit = apply_image_perturbations(raw, operations)
    assert replay_image_application(raw, modified, operations, library, audit) == audit
    for field in (CAMERAS[0], CAMERAS[1], "observation/state"):
        changed = copy.deepcopy(modified)
        changed[field].flat[0] += 1
        with pytest.raises(ValueError, match="exact pinned pixel replay"):
            replay_image_application(raw, changed, operations, library, audit)
    changed_audit = copy.deepcopy(audit)
    changed_audit["cameras"][CAMERAS[0]]["mask_fraction"] = 0.1
    with pytest.raises(ValueError, match="audit differs"):
        replay_image_application(raw, modified, operations, library, changed_audit)


class NativeDecodeSyntheticPolicy(SyntheticPolicy):
    """CPU constant field with the frozen native decoding formula, never a model."""

    observation_image_size = 224

    @staticmethod
    def output_transform(data):
        return {"actions": (data["actions"][:, :7] + 1.0) / 2.0 * np.ones(7)}

    def prepare_intervened(self, observation, observation_id, prompt, *, text):
        assert text.alpha == 0
        condition = self.prepare(observation, observation_id, prompt)
        return condition, {
            "condition_id": condition.condition_id,
            "original_prompt": prompt,
            "has_effect": False,
        }


def make_complete_case(
    root, *, baseline_success=False, development=False, preflight=False
):
    """Use the actual search/recording/client code; only physics, policy and HTTP are synthetic."""
    from astra_reversal import image_perturbation_agent, intervention_rollout
    from astra_reversal.image_perturbation_search import (
        ImagePerturbationSearch,
        load_protocol,
    )

    root.mkdir()
    assets = root / "assets"
    assets.mkdir()
    (assets / "norm_stats.json").write_text(
        json.dumps(
            {"norm_stats": {"actions": {"q01": [0.0] * 7, "q99": [0.999999] * 7}}}
        )
    )
    policy = NativeDecodeSyntheticPolicy()
    policy.metadata = {
        "input_profile": "openpi_libero",
        "frozen": True,
        "horizon": 10,
        "model_action_dim": 32,
        "input_profile_assets": {
            "norm_stats": {"sha256": file_sha256(assets / "norm_stats.json")}
        },
        "synthetic_test_only": True,
    }
    protocol = load_protocol()
    if development:
        protocol["seed"] = protocol["development_seed"]
    entry = {
        "episode_id": f"libero_goal_ood:seed{protocol['seed']}:task0:state0",
        "suite": "libero_goal_ood",
        "task_id": 0,
        "initial_state_id": 0,
        "seed": protocol["seed"],
        "instruction": "put object in bowl",
        "reset_state_sha256": "b" * 64,
        "reset_model_sha256": "c" * 64,
        "bddl_sha256": "d" * 64,
        "initially_successful": False,
    }
    payload = {
        "horizon": 10,
        "model_action_dim": 32,
        "timestep_seconds": 0.05,
        "lower": (-1.0,) * 7,
        "upper": (1.0,) * 7,
        "semantics": {"synthetic_test_only": True},
    }
    spec = ActionSpec(action_spec_id=digest(payload), **payload)
    library = SyntheticLibrary()
    search = ImagePerturbationSearch(
        policy,
        lambda *_: (SimpleNamespace(), None, None),
        SimpleNamespace(suite=entry["suite"]),
        entry,
        protocol,
        root / "case",
        library=library,
        development=development,
    )

    def raw(step):
        return snapshot(step, value=step % 200)["observation"]

    reset = {
        key: entry[key]
        for key in (
            "episode_id",
            "seed",
            "reset_state_sha256",
            "reset_model_sha256",
            "bddl_sha256",
        )
    }
    reset.update(
        post_stabilization_observation_sha256=digest(raw(0)),
        initial_success=False,
        initial_terminated=False,
    )
    reset["sha256"] = digest(reset)

    def run_rollout(env, entry, benchmark, callback, **kwargs):
        name = Path(kwargs["video_path"]).stem
        success = name == "recovered_noise_1" and baseline_success
        count = 10 if success else 30
        assert kwargs["action_budget"] == 300 and kwargs["execute_steps"] == 5
        assert kwargs["expected_reset"] in (None, reset)
        for step in range(0, count, 5):
            assert callback(raw(step), step).shape == (10, 7)
        return {
            "episode_id": entry["episode_id"],
            "success": success,
            "actions_executed": count,
            "policy_replans": count // 5,
            "wall_seconds": 2.0,
            "initial_success": False,
            "zero_action_success": False,
            "terminated": not success,
            "reset_audit": copy.deepcopy(reset),
            "snapshots": [
                snapshot(step, label=f"step_{step}", value=step % 200)
                for step in sorted({0, count // 3, 2 * count // 3, count})
            ],
        }

    class Opener:
        def open(self, request, timeout):
            context = json.loads(
                json.loads(request.data)["messages"][1]["content"][0]["text"]
            )["request"]
            proposal = proposal_for(context)
            if context["decision_index"] == 2:
                proposal["request_id"] = "0" * 16
            envelope = {
                "model": protocol["astra"]["model"],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "total_tokens": 13,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(proposal),
                        },
                    }
                ],
            }
            response = io.BytesIO(json.dumps(envelope).encode())
            response.status = 200
            return response

    class Client(image_perturbation_agent.ImagePerturbationClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.opener = Opener()

        def propose(self, request):
            with pytest.MonkeyPatch.context() as patch:
                if preflight and request["decision_index"] == 2:
                    patch.delenv("NVIDIA_INFERENCE_API_KEY")
                return super().propose(request)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NVIDIA_INFERENCE_API_KEY", "synthetic-offline-audit-test-key")
        patch.setattr(
            "astra_reversal.image_perturbation_search.ActionSpec.from_environment",
            lambda *_: spec,
        )
        patch.setattr(intervention_rollout, "run_rollout", run_rollout)
        patch.setattr(image_perturbation_agent, "ImagePerturbationClient", Client)
        search.run()
    metadata = {
        "checkpoint.json": policy.metadata,
        "protocol.json": protocol,
        "runtime.json": {"phase": "development" if development else "evaluation"},
        "image_library.json": library.metadata(),
        "reset_manifest.json": {"episodes": [entry]},
        "frozen_plan.json": {"assigned_episodes": [entry["episode_id"]]},
    }
    return SimpleNamespace(
        directory=root / "case", library=library, assets=assets, metadata=metadata
    )


@pytest.fixture(scope="module")
def complete_case(tmp_path_factory):
    return make_complete_case(tmp_path_factory.mktemp("image-audit") / "complete")


def audit(fixture):
    return audit_case(
        fixture.directory,
        library=fixture.library,
        metadata=fixture.metadata,
        assets=fixture.assets,
    )


def test_complete_native_schema_trace_pixel_feedback_expiry_noise_and_cost(
    complete_case,
):
    result = audit(complete_case)
    counts = result["counts"]
    assert (
        result["status"] == "passed"
        and result["complete"]
        and result["all_arrays_verified"]
    )
    assert counts["physical_rollouts"] == 13 and counts["actions"] == 390
    assert counts["generations"] == 78 and counts["velocity_evaluations"] == 2070
    assert counts["provider_calls"] == 8 and counts["preflight_failures"] == 0
    assert counts["accepted_proposals"] == 4 and counts["executed_decisions"] == 12
    assert counts["native_condition_fallback_actions"] == 20
    assert result["provider"]["tokens"]["total_tokens"]["sum"] == 104
    assert result["provider"]["failed_calls"] == 4
    assert result["numerical_maxima"] == {
        "noise_reconstruction_max_abs": 0.0,
        "controller_decode_max_abs": 0.0,
    }


@pytest.mark.parametrize("role", ["baseline", "known_noise", "policy_fresh"])
def test_native_controls_cannot_be_relabelled_as_another_noise_arm(complete_case, role):
    from astra_reversal.image_perturbation_audit import _physical

    summary = json.loads((complete_case.directory / "summary.json").read_text())
    row = summary["baseline"] if role == "baseline" else summary["controls"][role]
    row["mode"] = "random_noise"
    with pytest.raises(ValueError, match="baseline/control role"):
        _physical(summary)


@pytest.mark.parametrize("development,preflight", [(False, False), (True, True)])
def test_baseline_success_zero_calls_and_forced_dev_extra_cost(
    tmp_path, development, preflight
):
    fixture = make_complete_case(
        tmp_path / "complete",
        baseline_success=True,
        development=development,
        preflight=preflight,
    )
    result = audit(fixture)
    assert result["counts"]["physical_rollouts"] == 3 + 5 * int(development)
    assert result["counts"]["provider_calls"] == 2 * int(development)
    assert result["counts"]["preflight_failures"] == 2 * int(development)
    assert result["provider"]["tokens"]["total_tokens"]["sum"] == 26 * int(development)


@contextmanager
def changed_jsonl(directory, mutate):
    path = directory / "events.jsonl"
    original = path.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    try:
        yield
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("condition", "condition ID"),
        ("stale", "held beyond expiry"),
        ("instruction", "task changed"),
        ("noise", "noise"),
        ("history", "feedback"),
        ("cost", "summary differs"),
        ("pixel_audit", "audit differs"),
        ("time_order", "outside its physical attempt"),
    ],
)
def test_complete_trace_rejects_scientifically_material_tampering(
    complete_case, mutation, match
):
    def change(rows):
        generations = [row for row in rows if row["kind"] == "image_generation"]
        row = next(
            row
            for row in generations
            if row["attempt_id"] == "astra_occlusion_2"
            and row["observation_step"] == 25
        )
        if mutation == "condition":
            row["condition_id"] = "f" * 64
        elif mutation == "stale":
            row["image_perturbations"] = next(
                item
                for item in generations
                if item["attempt_id"] == "astra_occlusion_2"
                and item["observation_step"] == 20
            )["image_perturbations"]
        elif mutation == "instruction":
            row["instruction"] = "another task"
        elif mutation == "noise":
            row["latent"] = next(
                item
                for item in generations
                if item["mode"] == "policy_fresh" and item["observation_step"] == 5
            )["latent"]
        elif mutation == "history":
            request = next(
                item["request"] for item in rows if item["kind"] == "image_request"
            )
            request["completed_rollout_feedback"][0]["success"] = True
        elif mutation == "cost":
            next(item["attempt"] for item in rows if item["kind"] == "image_attempt")[
                "velocity_evaluations"
            ] += 10
        elif mutation == "pixel_audit":
            row["image_audit"]["cameras"][CAMERAS[0]]["changed_pixels"] = 1
        else:
            row = next(
                item
                for item in generations
                if item["attempt_id"] == "astra_occlusion_2"
                and item["observation_step"] == 0
            )
            index = rows.index(row)
            rows.insert(1, rows.pop(index))
            for index, item in enumerate(rows):
                item["sequence"] = index

    with (
        changed_jsonl(complete_case.directory, change),
        pytest.raises(ValueError, match=match),
    ):
        audit(complete_case)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("usage", "normalized usage"),
        ("payload", "payload/model/cache"),
        ("response", "provider body"),
        ("model", "provider body"),
    ],
)
def test_provider_audit_rejects_unbound_content_and_invented_usage(
    complete_case, mutation, match
):
    from astra_reversal.image_perturbation_audit import _CaseAudit

    checker = _CaseAudit(
        complete_case.directory,
        library=complete_case.library,
        metadata=complete_case.metadata,
        assets=complete_case.assets,
    )
    request = checker.groups["image_request"][0]["request"]
    attempt = checker.summary["arms"]["astra_occlusion"]["attempts"][1]
    record, decision = (
        copy.deepcopy(attempt["provider_records"][0]),
        attempt["decisions"][0],
    )
    if mutation == "usage":
        record["token_usage"]["total_tokens"] += 1
    elif mutation == "payload":
        record["payload_sha256"] = "f" * 64
    elif mutation == "model":
        record["response"]["model"] = "a-different-provider-model"
    else:
        message = record["response"]["choices"][0]["message"]
        proposal = json.loads(message["content"])
        proposal["decision_id"] = "different-executed-decision"
        message["content"] = json.dumps(proposal)
    with pytest.raises(ValueError, match=match):
        checker.provider(request, record, decision)


def test_archive_wrapper_uses_extracted_results_directory_and_binds_crc(
    tmp_path, monkeypatch
):
    import astra_reversal.image_perturbation_audit as module

    archive = tmp_path / "results.tar.gz"
    payload = b"synthetic worker metadata"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("results/runtime.json")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    seen = []

    def worker(directory, **kwargs):
        assert directory.name == "results"
        assert (directory / "runtime.json").read_bytes() == payload
        seen.append(directory)
        return {"status": "passed"}

    monkeypatch.setattr(module, "_audit_worker", worker)
    receipt = audit_worker(archive, library=SimpleNamespace(), assets=tmp_path)
    assert receipt["archive"] == {
        "sha256": file_sha256(archive),
        "bytes": archive.stat().st_size,
        "uncompressed_bytes": len(payload),
        "gzip_crc_verified": True,
    }
    assert len(seen) == 1 and not seen[0].exists()
    damaged = bytearray(archive.read_bytes())
    damaged[-8] ^= 1
    archive.write_bytes(damaged)
    with pytest.raises((OSError, tarfile.TarError)):
        audit_worker(archive, library=SimpleNamespace(), assets=tmp_path)
