"""Small synthetic corruption checks; no policy, simulator or real provider."""

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from astra_reversal.action_adapter import ActionSpec
from astra_reversal.flow import error_metrics, noise_statistics
from astra_reversal.frs_agent import prompt_manifest
from astra_reversal.frs_audit import (
    _array,
    _rng,
    _TaskAudit,
    _worker_inputs,
    normalization_adapter,
    replay_generation,
    replay_guide,
    verify_provider_binding,
    verify_reset_entry,
)
from astra_reversal.frs_guide import project_policy_pixels
from astra_reversal.frs_noise_policy import AuxiliaryNoisePolicy
from astra_reversal.frs_operators import (
    directional_reference,
    edit_native,
    repeated_gaussian_noise,
    resample_padding_noise,
)
from astra_reversal.image_perturbation_audit import ArtifactStore
from astra_reversal.records import Recorder, digest, file_sha256

from .test_frs_agent import (
    MODEL,
    client_factory,
    envelope_for,
    request_for,
)

# Import the existing hermetic fake HTTP fixture into this module.
assert client_factory


def observation():
    return {
        "observation/image": np.full((224, 224, 3), 93, np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 127, np.uint8),
        "observation/state": np.array(
            [0.2, 0.1, 0.6, 0, 0, 0, 0.02, -0.02], np.float32
        ),
    }


def guide_for(raw):
    # Explicitly synthetic orthographic calibration with right=-worldY.
    calibration = np.array(
        [[0, 100, 0, 112], [100, 0, -100, 112], [0, 0, 0, 1], [0, 0, 0, 1]], np.float64
    )
    eef = raw["observation/state"][:3].astype(np.float64)
    table = eef.copy()
    table[2] = 0.1
    points = np.stack((eef, table))
    xy = project_policy_pixels(points, calibration)
    probes = project_policy_pixels(
        np.stack(
            (table, table + [0.01, 0, 0], table + [0, 0.01, 0], table + [0, 0, 0.01])
        ),
        calibration,
    )
    base = Image.fromarray(raw["observation/image"]).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line([tuple(p) for p in xy], fill=(35, 100, 255, 155), width=2)
    x, y = xy[0]
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(35, 100, 255, 200))
    guide = np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))
    return guide, {
        "camera": "agentview",
        "image_convention": "opengl_then_flip_both_axes",
        "policy_receives_guide": False,
        "table_height": 0.1,
        "world_points": points.tolist(),
        "camera_transform": calibration.tolist(),
        "policy_pixels_xy": xy.tolist(),
        "axis_probe_delta_pixels": (probes[1:] - probes[0]).tolist(),
        "camera_to_controller_signs": [1, -1, 1],
        "raw_sha256": digest(raw["observation/image"]),
        "guide_sha256": digest(guide),
    }


def generation(method="native_euler10", *, edit=False):
    raw = observation()
    entry = {
        "episode_id": "synthetic:state0",
        "seed": 19,
        "instruction": "Move the cup.",
    }
    spec = ActionSpec(
        "synthetic-controller",
        10,
        32,
        0.05,
        (-1.0,) * 7,
        (1.0,) * 7,
        {"synthetic": True},
    )
    adapter = normalization_adapter(spec, np.full(7, -0.8), np.full(7, 0.9))
    loop = method.startswith("critique_frs_")
    repeated = loop or method in ("native_repeated_noise", "learned_noise")
    rng = _rng(entry, 0, 0)
    noise = (
        repeated_gaussian_noise(rng)["noise"]
        if repeated
        else rng.standard_normal((1, 10, 32)).astype(np.float32)
    )
    native = (noise * 0.1).astype(np.float32)
    native_actions, clipping = adapter.decode(native, np.zeros((1, 32), np.float32))
    event = {
        "method": method,
        "step": 0,
        "observation": raw,
        "observation_sha256": digest(raw),
        "condition_id": digest({**raw, "prompt": entry["instruction"]}),
        "base_noise": noise.copy(),
        "predicted_noise": noise.copy(),
        "actor_receipt": None,
        "native_model_actions": native,
        "native_actions": native_actions,
        "native_clipping": clipping,
        "proposal": None,
        "guide": None,
        "guide_receipt": None,
        "reference": None,
        "reversed_noise": None,
        "noise_transform": None,
        "executed_noise": noise.copy(),
        "generation_kind": "native_defer",
        "reconstruction": None,
        "generated_model_actions": native.copy(),
        "actions": native_actions.copy(),
        "clipping": dict(clipping),
        "noise_statistics": noise_statistics(noise),
    }
    paper = method in ("astra_direction_direct", "astra_frs")
    if paper:
        event["guide"], event["guide_receipt"] = guide_for(raw)
    if edit:
        event["proposal"] = (
            {"fine": False, "coords": [0, 1, 1], "motion_amount": "less"}
            if paper
            else {
                "mode": "edit",
                "delta_xyz": [0.3, -0.2, 0.1],
                "apply_steps": 3,
                "gripper": "close",
            }
        )
        reference = (
            directional_reference(adapter, raw, coords=[0, -1, 1], motion_amount="less")
            if paper
            else edit_native(
                adapter,
                raw,
                native_actions,
                delta_xyz=[0.3, -0.2, 0.1],
                apply_steps=3,
                gripper="close",
            )
        )
        event["reference"] = reference
        if method == "astra_direction_direct":
            generated = reference["target_model_actions"]
            event.update(
                executed_noise=None,
                noise_statistics=None,
                generation_kind="direct_reference",
            )
        else:
            reverse = np.arange(320, dtype=np.float32).reshape(1, 10, 32) / 100
            transformed = resample_padding_noise(
                reverse, _rng(entry, 0, 2), repeat_mean=loop
            )
            generated = (reference["target_model_actions"] + 0.03).astype(np.float32)
            event.update(
                reversed_noise=reverse,
                noise_transform=transformed,
                executed_noise=transformed["noise"],
                generation_kind="frs_edit",
                reconstruction=error_metrics(
                    reference["target_model_actions"], generated
                ),
                noise_statistics=noise_statistics(transformed["noise"]),
            )
        actions, clipping = adapter.decode(generated, np.zeros((1, 32), np.float32))
        event.update(
            generated_model_actions=generated, actions=actions, clipping=clipping
        )
    return event, entry, adapter


@pytest.mark.parametrize(
    "method,edit,expected_vf",
    [
        ("native_euler10", False, 10),
        ("native_repeated_noise", False, 10),
        ("critique_frs_learning", True, 30),
        ("astra_frs", True, 30),
        ("astra_direction_direct", True, 10),
    ],
)
def test_exact_action_and_noise_boundary_replays_without_model(
    method, edit, expected_vf
):
    event, entry, adapter = generation(method, edit=edit)
    counts = replay_generation(event, entry=entry, adapter=adapter)
    assert counts["velocity_evaluations"] == expected_vf
    assert counts["interventions"] == int(edit)
    if method == "astra_frs":
        assert (
            event["reconstruction"]["max_abs"] > 0.02
        )  # Valid Euler deviation is not a failed audit.


@pytest.mark.parametrize(
    "tamper",
    ["base_noise", "generated_model_actions", "actions", "condition_id", "raw_pixel"],
)
def test_native_boundary_detects_changed_noise_actions_and_policy_input(tamper):
    event, entry, adapter = generation()
    if tamper == "condition_id":
        event[tamper] = "f" * 64
    elif tamper == "raw_pixel":
        event["observation"]["observation/image"][0, 0, 0] += 1
    else:
        event[tamper].flat[0] += 0.1
    with pytest.raises(ValueError):
        replay_generation(event, entry=entry, adapter=adapter)


@pytest.mark.parametrize(
    "tamper",
    ["wrong_mean", "wrong_padding", "unprojected_noise", "rotation", "wrong_role"],
)
def test_learning_edit_detects_mismatched_execution_and_reference(tamper):
    event, entry, adapter = generation("critique_frs_learning", edit=True)
    if tamper == "wrong_mean":
        event["executed_noise"][0, 0, 0] += 0.1
    elif tamper == "wrong_padding":
        event["executed_noise"][0, 0, 7] += 0.1
    elif tamper == "unprojected_noise":
        event["executed_noise"][..., :7] = event["reversed_noise"][..., :7]
    elif tamper == "rotation":
        event["reference"]["target_actions"][0, 3] += 0.1
    else:
        event["generation_kind"] = "native_defer"
    with pytest.raises(ValueError):
        replay_generation(event, entry=entry, adapter=adapter)


@pytest.mark.parametrize("tamper", ["sign", "pixels", "raw_hash", "policy_flag"])
def test_guide_requires_signed_calibration_and_byte_identical_raster(tamper):
    raw = observation()
    guide, receipt = guide_for(raw)
    assert replay_guide(raw, guide, receipt) == [1, -1, 1]
    if tamper == "sign":
        receipt["camera_to_controller_signs"][1] = 1
    elif tamper == "pixels":
        guide = guide.copy()
        guide[0, 0, 0] += 1
    elif tamper == "raw_hash":
        receipt["raw_sha256"] = "a" * 64
    else:
        receipt["policy_receives_guide"] = True
    with pytest.raises(ValueError):
        replay_guide(raw, guide, receipt)


def provider_case(client_factory):
    request = request_for("action_edit")
    client = client_factory(envelope_for(request))
    response = client.propose(request)
    decision = {
        "request_fingerprint": request["request_fingerprint"],
        "response": response,
        "error": None,
    }
    settings = {
        "model": MODEL,
        "reasoning_effort": "medium",
        "max_completion_tokens": 8192,
    }
    return request, decision, client.records[0], settings


def test_provider_receipt_binds_exact_observable_request_and_actual_response(
    client_factory,
):
    assert verify_provider_binding(*provider_case(client_factory))


@pytest.mark.parametrize(
    "tamper", ["payload", "model", "response", "sampling", "fingerprint"]
)
def test_provider_join_rejects_wrong_input_response_and_settings(
    client_factory, tamper
):
    request, decision, row, settings = provider_case(client_factory)
    if tamper == "payload":
        row["payload_sha256"] = "0" * 64
    elif tamper == "model":
        row["response"]["model"] = "another/model"
    elif tamper == "response":
        decision["response"] = {**decision["response"], "delta_xyz": [0, 0, 0]}
    elif tamper == "sampling":
        row["sampling_settings"]["reasoning_effort"] = "low"
    else:
        row["request_fingerprint"] = "a" * 64
    with pytest.raises(ValueError):
        verify_provider_binding(request, decision, row, settings)


def checkpoint_audit(tmp_path):
    audit = _TaskAudit.__new__(_TaskAudit)
    audit.summary = {"suite": "synthetic-suite", "task_id": 0}
    audit.entry0 = {"episode_id": "synthetic:state0"}
    audit.store = ArtifactStore(tmp_path)
    audit.learning_rounds, audit.actor_uses, audit.samples = [], [], {}
    audit.starts = {
        "baseline": {
            "attempt_id": "baseline",
            "sequence": 2,
            "method": "native_repeated_noise",
        }
    }
    audit.generation_groups = {"baseline": []}
    audit.counts = Counter()
    seed = int(digest(audit.entry0["episode_id"])[:8], 16)
    actor = AuxiliaryNoisePolicy("synthetic-suite:0", seed, device="cpu")
    initial = actor.save(tmp_path / "noise_policy_initial")
    audit.summary["initial_noise_policy_checkpoint"] = initial
    audit.groups = {"noise_policy_initial": [{"sequence": 1, "checkpoint": initial}]}
    for index in range(1, 4):
        candidate = f"candidate-{index}"
        checkpoint = actor.save(tmp_path / f"noise_policy_round{index}")
        audit.learning_rounds.append(
            {
                "round_index": index,
                "candidate_attempt_id": candidate,
                "checkpoint": checkpoint,
                "promoted": False,
                "update": None,
            }
        )
        audit.samples[candidate] = []
    return audit


def test_rejected_rounds_preserve_exact_untrained_weights_optimizer_and_replay(
    tmp_path,
):
    audit = checkpoint_audit(tmp_path)
    receipts = audit.checkpoints()
    assert len(receipts) == 3 and len({row["state_id"] for row in receipts}) == 1
    assert all(
        row["replay_samples"] == row["accepted_updates"] == 0 for row in receipts
    )


def test_checkpoint_byte_tamper_is_not_treated_as_a_valid_rejected_round(tmp_path):
    audit = checkpoint_audit(tmp_path)
    with (tmp_path / "noise_policy_round2/state.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="state bytes"):
        audit.checkpoints()


def test_claimed_promotion_without_labels_cannot_claim_training(tmp_path):
    audit = checkpoint_audit(tmp_path)
    audit.learning_rounds[0]["promoted"] = True
    with pytest.raises(ValueError, match="Empty accepted"):
        audit.checkpoints()


def test_array_inventory_rejects_hidden_or_changed_training_artifacts(tmp_path):
    recorder = Recorder(tmp_path / "case")
    recorder.event("generation", noise=np.zeros((1, 10, 32), np.float32))
    store = ArtifactStore(recorder.directory)
    event = store.lines("events.jsonl")[0]
    store.register(event)
    store.resolve(event)
    np.save(
        recorder.directory / "arrays/unreferenced.npy",
        np.ones(7, np.float32),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="unreferenced"):
        store.finish()


def test_empty_executed_prefix_is_still_a_valid_array_boundary():
    assert (
        _array(np.empty((0, 7), np.float32), np.empty((0, 7), np.float32), "empty") == 0
    )


@pytest.mark.parametrize("tamper", [None, "seed", "state", "model", "state_hash"])
def test_reset_provenance_binds_actual_state_model_and_seed(tamper):
    state = np.arange(12, dtype=np.float64)
    model = {
        "body_pos": np.zeros((2, 3), np.float64),
        "body_quat": np.ones((2, 4), np.float64),
    }
    entry = {
        "suite": "synthetic",
        "seed": 43,
        "task_id": 2,
        "initial_state_id": 1,
        "episode_id": "synthetic:seed43:task2:state1",
        "reset_state": state.tolist(),
        "reset_state_sha256": digest(state),
        "reset_model": {k: v.tolist() for k, v in model.items()},
        "reset_model_sha256": digest(model),
    }
    if tamper == "seed":
        entry["seed"] = 19
    elif tamper == "state":
        entry["reset_state"][0] += 1
    elif tamper == "model":
        entry["reset_model"]["body_pos"][0][0] += 0.1
    elif tamper == "state_hash":
        entry["reset_state_sha256"] = "0" * 64
    if tamper is None:
        recovered, _ = verify_reset_entry(entry, "synthetic")
        np.testing.assert_array_equal(recovered, state)
    else:
        with pytest.raises(ValueError):
            verify_reset_entry(entry, "synthetic")


def worker_proof_fixture(tmp_path, *, sealed=True, final_after=False):
    """Synthetic metadata only; this exercises the native-weight proof boundary."""
    worker = tmp_path / "worker"
    task = worker / "task_2"
    task.mkdir(parents=True)
    summary = {"status": "complete", "task_id": 2, "development": False}
    protocol, checkpoint = {"seed": 43}, {"synthetic_checkpoint": True}
    entry = {"task_id": 2, "episode_id": "synthetic:seed43:task2:state0"}
    manifest = {"episodes": [entry]}
    manifest["sha256"] = digest(manifest)
    runtime = {
        "workflow": "synthetic-seal-worker",
        "worker": 2,
        "phase": "evaluation",
        "gpu": "NVIDIA L40S",
        "tf32": False,
    }
    tensors = {
        "synthetic.weight": {"shape": [1], "dtype": "torch.float32", "sha256": "1" * 64}
    }
    before = {"tensors": tensors, "sha256": digest(tensors)}
    values = {
        "runtime.json": runtime,
        "protocol.json": protocol,
        "checkpoint.json": checkpoint,
        "prompts.json": prompt_manifest(),
        "reset_manifest.json": manifest,
        "frozen_weights_before.json": before,
        "frozen_plan.json": {
            "runtime": runtime,
            "protocol_sha256": digest(protocol),
            "manifest_sha256": manifest["sha256"],
            "assigned_tasks": [2],
            "assigned_episodes": [entry["episode_id"]],
        },
    }
    for name, value in values.items():
        (worker / name).write_text(json.dumps(value, sort_keys=True))
    (task / "summary.json").write_text(json.dumps(summary))
    (task / "events.jsonl").write_text('{"kind":"synthetic"}\n')
    (task / "provider.jsonl").write_text('{"request_index":1}\n')
    if final_after:
        (worker / "frozen_weights_after.json").write_text(json.dumps(before))
    if sealed:
        (task / "frozen_weights_after.json").write_text(json.dumps(before))
        seal = {
            "schema_version": "frs-completed-task-1.0",
            "task_id": 2,
            "workflow": runtime["workflow"],
            "worker": runtime["worker"],
            "native_tensor_sha256": before["sha256"],
            "task_files_sha256": {
                name: file_sha256(task / name)
                for name in (
                    "summary.json",
                    "events.jsonl",
                    "provider.jsonl",
                    "frozen_weights_after.json",
                )
            },
            "worker_metadata_sha256": {
                name: file_sha256(worker / name) for name in values
            },
        }
        (task / "completion_receipt.json").write_text(json.dumps(seal))
    audit = SimpleNamespace(
        store=ArtifactStore(task),
        summary=summary,
        protocol=protocol,
        checkpoint=checkpoint,
        groups={"task": [{"entries": [entry]}]},
    )
    # Bind the bytes already consumed by the main task audit.
    for name in ("summary.json", "events.jsonl", "provider.jsonl"):
        audit.store.remember(name)
    return worker, audit


@pytest.mark.parametrize(
    "sealed,final_after,scope",
    [
        (True, False, "completed_task"),
        (True, True, "worker_complete"),
        (False, True, "worker_complete"),
    ],
)
def test_completed_task_proof_survives_later_worker_interruption(
    tmp_path, sealed, final_after, scope
):
    worker, audit = worker_proof_fixture(
        tmp_path, sealed=sealed, final_after=final_after
    )
    receipt = _worker_inputs(worker, audit)
    assert receipt["weight_check_scope"] == scope
    assert ("frozen_weights_after.json" in receipt["input_file_sha256"]) == final_after
    assert receipt["input_file_sha256"]["prompts.json"] == file_sha256(
        worker / "prompts.json"
    )
    for field, name in (
        ("completion_receipt_sha256", "completion_receipt.json"),
        ("task_frozen_weights_after_sha256", "frozen_weights_after.json"),
    ):
        expected = file_sha256(audit.store.path(name)) if sealed else None
        assert receipt[field] == expected
        assert audit.store.files.get(name) == expected


def test_old_unsealed_task_still_requires_final_worker_weights(tmp_path):
    worker, audit = worker_proof_fixture(tmp_path, sealed=False)
    with pytest.raises(ValueError, match="Missing final worker weight receipt"):
        _worker_inputs(worker, audit)


@pytest.mark.parametrize(
    "tamper",
    [
        "task_id",
        "worker",
        "workflow",
        "native_digest",
        "omitted_recording",
        "extra_recording",
        "missing_metadata",
        "worker_hash",
        "provider_bytes",
        "summary_bytes",
    ],
)
def test_completion_seal_rejects_wrong_scope_or_changed_recording(tmp_path, tamper):
    worker, audit = worker_proof_fixture(tmp_path)
    path = audit.store.path("completion_receipt.json")
    seal = json.loads(path.read_text())
    if tamper in ("task_id", "worker"):
        seal[tamper] += 1
    elif tamper == "workflow":
        seal[tamper] = "another-worker"
    elif tamper == "native_digest":
        seal["native_tensor_sha256"] = "0" * 64
    elif tamper == "omitted_recording":
        del seal["task_files_sha256"]["provider.jsonl"]
    elif tamper == "extra_recording":
        seal["task_files_sha256"]["../other.json"] = "0" * 64
    elif tamper == "missing_metadata":
        del seal["worker_metadata_sha256"]["prompts.json"]
    elif tamper == "worker_hash":
        seal["worker_metadata_sha256"]["runtime.json"] = "0" * 64
    else:
        name = "provider.jsonl" if tamper == "provider_bytes" else "summary.json"
        with audit.store.path(name).open("a") as stream:
            stream.write("\n")
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError):
        _worker_inputs(worker, audit)


@pytest.mark.parametrize("scope", ["task", "worker"])
def test_changed_native_weights_never_pass_with_an_otherwise_valid_seal(
    tmp_path, scope
):
    worker, audit = worker_proof_fixture(tmp_path, final_after=True)
    root = audit.store.directory if scope == "task" else worker
    path = root / "frozen_weights_after.json"
    after = json.loads(path.read_text())
    after["tensors"]["synthetic.weight"]["sha256"] = "2" * 64
    after["sha256"] = digest(after["tensors"])
    path.write_text(json.dumps(after))
    with pytest.raises(ValueError, match="Native pi05 tensor bytes changed"):
        _worker_inputs(worker, audit)


def test_task_after_receipt_cannot_substitute_for_missing_completion_seal(tmp_path):
    worker, audit = worker_proof_fixture(tmp_path, final_after=True)
    audit.store.path("completion_receipt.json").unlink()
    with pytest.raises(ValueError, match="lack a completion seal"):
        _worker_inputs(worker, audit)
