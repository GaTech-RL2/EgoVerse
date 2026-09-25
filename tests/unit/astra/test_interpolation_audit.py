"""Offline integrity tests using tiny synthetic arrays, never policy inference."""

import copy
import io
import json
import math
import tarfile

import numpy as np
import pytest

from astra_reversal.interpolation_audit import (
    ALIGNMENT,
    CAPTURE_BOUNDARY,
    FLOW_SHAPE,
    GATE_CHECKS,
    ArrayStore,
    AuditError,
    CaseAudit,
    Context,
    _extract_archive,
    audit_worker,
    render_linux,
    verify_noise,
    verify_reset,
)
from astra_reversal.interpolation_catalog import donor_catalog, oracle_for
from astra_reversal.interpolation_search import ARMS, load_protocol
from astra_reversal.interventions import noise_basis, perturb_noise
from astra_reversal.records import Recorder, digest


def rows(directory):
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]


def rewrite(directory, values):
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in values)
    )


@pytest.fixture
def saved_arrays(tmp_path):
    directory = tmp_path / "case"
    recorder = Recorder(directory)
    recorder.event(
        "fixture", noise=np.zeros(FLOW_SHAPE, np.float32), state=np.ones(8, np.float32)
    )
    return directory


@pytest.mark.parametrize(
    "corruption", ["bytes", "dtype", "shape", "missing", "extra", "path"]
)
def test_every_npy_byte_shape_dtype_and_reference_must_match(saved_arrays, corruption):
    recorded = rows(saved_arrays)
    ref = recorded[0]["noise"]
    path = saved_arrays / ref["array"]
    if corruption == "bytes":
        np.save(path, np.ones(FLOW_SHAPE, np.float32), allow_pickle=False)
    elif corruption == "dtype":
        np.save(path, np.zeros(FLOW_SHAPE, np.float64), allow_pickle=False)
    elif corruption == "shape":
        np.save(path, np.zeros((1, 32, 10), np.float32), allow_pickle=False)
    elif corruption == "missing":
        path.unlink()
    elif corruption == "extra":
        np.save(saved_arrays / "arrays/extra.npy", np.zeros(1), allow_pickle=False)
    else:
        ref["array"] = "arrays/../../outside.npy"
    with pytest.raises(AuditError):
        ArrayStore(saved_arrays, recorded).finish()


def test_array_file_cannot_change_after_first_verification(saved_arrays):
    recorded = rows(saved_arrays)
    store = ArrayStore(saved_arrays, recorded)
    store.read(recorded[0]["noise"])
    np.save(
        saved_arrays / recorded[0]["noise"]["array"], np.ones(FLOW_SHAPE, np.float32)
    )
    with pytest.raises(AuditError, match="changed"):
        store.finish()


def test_held_known_random_and_first_fresh_noise_have_distinct_bindings():
    known = np.random.default_rng(4).standard_normal(FLOW_SHAPE).astype(np.float32)
    recovered = known + np.float32(0.05)
    basis, basis_id = noise_basis(FLOW_SHAPE, 7)
    proposal = {
        "basis_id": basis_id,
        "coefficients": [0.5, -0.25] + [0.0] * 6,
        "perturbation_scale": 0.3,
    }
    for mode, latent in (
        ("known_noise", known),
        ("recovered_noise", recovered),
        ("oracle_tli", recovered),
    ):
        assert (
            verify_noise(
                latent, mode=mode, step=0, known=known, recovered=recovered, basis=basis
            )
            == 0
        )
    wrong = recovered.copy()
    wrong.flat[0] = np.nextafter(wrong.flat[0], np.float32(math.inf))
    with pytest.raises(AuditError, match="noise construction"):
        verify_noise(
            wrong,
            mode="oracle_tli",
            step=5,
            known=known,
            recovered=recovered,
            basis=basis,
        )
    changed = perturb_noise(recovered, basis, proposal)
    assert (
        verify_noise(
            changed,
            mode="random_noise",
            step=5,
            known=known,
            recovered=recovered,
            basis=basis,
            proposal=proposal,
        )
        == 0
    )
    with pytest.raises(AuditError, match="leaked"):
        verify_noise(
            changed,
            mode="oracle_tli",
            step=0,
            known=known,
            recovered=recovered,
            basis=basis,
            proposal=proposal,
        )
    fresh = np.random.default_rng(11)
    verify_noise(
        known,
        mode="policy_fresh",
        step=0,
        known=known,
        recovered=recovered,
        basis=basis,
        fresh_rng=fresh,
    )
    expected = np.random.default_rng(11).standard_normal(FLOW_SHAPE).astype(np.float32)
    assert (
        verify_noise(
            expected,
            mode="policy_fresh",
            step=5,
            known=known,
            recovered=recovered,
            basis=basis,
            fresh_rng=fresh,
        )
        == 0
    )


def observation():
    return {
        "observation/image": np.full((224, 224, 3), 100, np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 30, np.uint8),
        "observation/state": np.zeros(8, np.float32),
    }


def test_linux_redraw_has_strict_integer_pixels_and_preserves_raw_other_camera():
    raw = observation()
    marks = [
        {
            "camera": "observation/image",
            "kind": "point",
            "coordinates": [8, 9],
            "gain": 0.85,
        }
    ]
    edited = render_linux(raw, marks)
    original = np.array([100, 100, 100], np.float32)
    expected = np.add(
        original,
        np.multiply(
            np.array([255, 0, 255], np.float32) - original,
            np.float32(0.85),
            dtype=np.float32,
        ),
        dtype=np.float32,
    ).astype(np.uint8)
    np.testing.assert_array_equal(edited["observation/image"][9, 8], expected)
    assert np.all(raw["observation/image"] == 100)
    np.testing.assert_array_equal(
        raw["observation/wrist_image"], edited["observation/wrist_image"]
    )
    changed = copy.deepcopy(edited)
    changed["observation/image"][9, 8, 1] += 1
    assert digest(changed) != digest(edited)


class Tokenizer:
    """Synthetic two-token instructions; this is not the checkpoint tokenizer."""

    def bos_id(self):
        return 2

    def encode(self, prompt, add_bos=False):
        if prompt == "\n":
            return [10]
        values = [20 + int(digest(prompt)[:3], 16), 30 + int(digest(prompt)[3:6], 16)]
        return ([2] if add_bos else []) + values


def context():
    result = object.__new__(Context)
    result.metadata, result.checkpoint = {}, {"synthetic": True}
    result.tokenizer, result.token_cache = Tokenizer(), {}
    result.operator_sha, result.compatibility = "a" * 64, {"synthetic": True}
    result.q01, result.q99 = -np.ones(7), np.ones(7)
    result.banks = {}
    for donor in donor_catalog():
        metadata = {
            "provenance": {
                "source_prompt": donor["prompt"],
                "compatibility": result.compatibility,
            },
            "arrays": {
                "states": {
                    "shape": [18, 1, 200, 2048],
                    "dtype": "float32",
                    "sha256": digest(donor["source_id"]),
                }
            },
            "captured_layer_indices": list(range(18)),
            "effective_layer_indices": list(range(17)),
        }
        metadata["bank_id"] = digest(metadata)
        result.banks[donor["source_id"]] = metadata
    return result


def metric(ctx, target, effect, *, sources=None, factor=None, label="text"):
    delta = float(effect)
    row = {
        "text_before_sha256": digest(label),
        "text_after_sha256": digest(label + " changed") if effect else digest(label),
        "delta_frobenius": delta,
        "base_frobenius": 2.0,
        "delta_rms": delta / math.sqrt(int(ctx.tokens(target)[2].sum()) * 2048),
        "relative_rms": delta / 2,
        "has_effect": effect,
        "protected_text_unchanged": True,
    }
    if sources:
        row.update(
            mapping_a=ctx.mapping(sources[0], target),
            mapping_b=ctx.mapping(sources[1], target),
            factor=factor,
        )
    return row


def provenance(ctx, raw, target, sources, operator, alpha, source_ids=None):
    ids, mask, instruction = ctx.tokens(target)
    tei = operator in ("tei", "tei_tli")
    tli = operator in ("tli", "tei_tli") and alpha != 0.5
    tei_effect = tei and sources != (target, target)
    padding = np.concatenate(
        (np.ones((1, 512), bool), np.zeros((1, 256), bool), mask), axis=1
    )
    value = {
        "operator": operator,
        "operator_version": 1,
        "operator_source_sha256": ctx.operator_sha,
        "alpha": alpha,
        "tei_orientation": "A_at_0_B_at_1",
        "tei_formula": "(1-alpha)*E_A+alpha*E_B",
        "tli_formula": "(1-2*alpha)*(T_A-T_B)",
        "alignment": ALIGNMENT,
        "tli_boundary": CAPTURE_BOUNDARY,
        "target_prompt": target,
        "source_prompts": list(sources),
        "raw_condition_id": digest({**raw, "prompt": target}),
        "observation_id": digest(raw),
        "compatibility": ctx.compatibility,
        "target_token_sha256": digest(ids),
        "target_token_mask_sha256": digest(mask),
        "target_instruction_mask_sha256": digest(instruction),
        "target_instruction_positions": [1, 2],
        "source_tokens": [
            {
                "token_sha256": digest(parts[0]),
                "token_mask_sha256": digest(parts[1]),
                "instruction_mask_sha256": digest(parts[2]),
            }
            for parts in map(ctx.tokens, sources)
        ],
        "banks": {
            label: ctx.banks[source_id]
            for label, source_id in zip(("a", "b"), source_ids, strict=True)
        }
        if tli
        else None,
        "tli_active": tli,
        "tli_layer_indices": list(range(17)) if tli else [],
        "tli_layers": [
            {
                **metric(
                    ctx,
                    target,
                    True,
                    sources=sources,
                    factor=1 - 2 * alpha,
                    label=f"layer{index}",
                ),
                "layer_index": index,
                "hidden_dtype": "torch.float32",
                "direct_vision_write_unchanged": True,
            }
            for index in range(17)
        ]
        if tli
        else [],
        "tei": metric(ctx, target, tei_effect, sources=sources if tei else None),
        "text_start": 768,
        "text_slots": 200,
        "embedding_dtype": "torch.float32",
        "prefix_before_sha256": digest("prefix"),
        "prefix_after_sha256": digest("changed") if tei_effect else digest("prefix"),
        "prefix_padding_sha256": digest(padding),
        "prefix_attention_sha256": digest(np.zeros_like(padding)),
        "prefix_positions_sha256": digest(np.cumsum(padding, axis=1) - 1),
        "has_effect": tei_effect or tli,
        "native_target_equivalent": not (tei_effect or tli),
    }
    for key in (
        "text_mask_fixed",
        "direct_writes_instruction_only",
        "later_vision_hidden_states_may_change",
        "cache_rebuilt",
        "vision_prefix_unchanged",
        "protected_embedding_slots_unchanged",
        "hooks_removed",
    ):
        value[key] = True
    if tei:
        value["source_scaled_embedding_sha256"] = [digest(prompt) for prompt in sources]
    bind_condition(value)
    return value


def bind_condition(value):
    value.pop("condition_id", None)
    value["condition_id"] = (
        digest({"raw_condition_id": value["raw_condition_id"], "interpolation": value})
        if value["has_effect"]
        else value["raw_condition_id"]
    )


@pytest.mark.parametrize("corruption", ["layer", "mask", "protected", "bank"])
def test_tli_provenance_rejects_rehashed_wrong_layers_masks_boundaries_or_banks(
    corruption,
):
    ctx, raw, target = context(), observation(), "target"
    source_ids = tuple(list(ctx.banks)[:2])
    sources = tuple(ctx.banks[key]["provenance"]["source_prompt"] for key in source_ids)
    value = provenance(ctx, raw, target, sources, "tli", 0.0, source_ids)
    expected, _ = ctx.conditioning(value, raw, target, sources, "tli", 0.0, source_ids)
    assert expected == value["condition_id"]
    value = copy.deepcopy(value)
    if corruption == "layer":
        value["tli_layers"][-1]["layer_index"] = 17
    elif corruption == "mask":
        value["target_instruction_mask_sha256"] = "0" * 64
    elif corruption == "protected":
        value["tli_layers"][4]["protected_text_unchanged"] = False
    else:
        value["banks"]["a"]["bank_id"] = "0" * 64
    bind_condition(value)
    with pytest.raises(AuditError):
        ctx.conditioning(value, raw, target, sources, "tli", 0.0, source_ids)


def test_paired_reset_rejects_internally_rehashed_different_start():
    entry = {
        "episode_id": "fixture",
        "seed": 29,
        "reset_state_sha256": "a" * 64,
        "reset_model_sha256": "b" * 64,
        "bddl_sha256": "c" * 64,
    }
    reset = {
        **entry,
        "post_stabilization_observation_sha256": "d" * 64,
        "initial_success": False,
        "initial_terminated": False,
    }
    reset["sha256"] = digest(reset)
    attempt = {
        "reset_audit": reset,
        "initial_success": False,
        "zero_action_success": False,
    }
    verify_reset(attempt, entry, "d" * 64)
    changed = copy.deepcopy(attempt)
    changed["reset_audit"]["post_stabilization_observation_sha256"] = "e" * 64
    changed["reset_audit"]["sha256"] = digest(
        {key: value for key, value in changed["reset_audit"].items() if key != "sha256"}
    )
    with pytest.raises(AuditError, match="paired baseline"):
        verify_reset(changed, entry, "e" * 64, reset)


def complete_case(directory):
    """Small complete synthetic event graph; no model accuracy claim is made."""
    ctx, raw, protocol = context(), observation(), load_protocol()
    target = "unit target"
    entry = {
        "episode_id": "libero_goal_ood:seed29:task0:state0",
        "suite": "libero_goal_ood",
        "task_id": 0,
        "seed": 29,
        "instruction": target,
        "reset_state_sha256": "a" * 64,
        "reset_model_sha256": "b" * 64,
        "bddl_sha256": "c" * 64,
    }
    oracle = oracle_for(entry["suite"], 0).metadata()
    seed_words = [29, int(digest(entry["episode_id"])[:8], 16)]
    known = (
        np.random.default_rng(np.random.SeedSequence(seed_words))
        .standard_normal(FLOW_SHAPE)
        .astype(np.float32)
    )
    basis, basis_id = noise_basis(FLOW_SHAPE, np.random.SeedSequence(seed_words + [1]))
    ctx.metadata = {
        "protocol.json": protocol,
        "reset_manifest.json": {"episodes": [entry]},
        "frozen_plan.json": {"assigned_episodes": [entry["episode_id"]]},
    }
    recorder = Recorder(directory)
    recorder.event(
        "case", entry=entry, protocol=protocol, basis=basis, basis_id=basis_id
    )
    raw_id, observation_id = digest({**raw, "prompt": target}), digest(raw)
    zeros = np.zeros(FLOW_SHAPE, np.float32)
    decoded = ctx.decode(zeros)
    initial = {
        "passed": True,
        "condition_id": raw_id,
        "known_noise_sha256": digest(known),
        "recovered_noise_sha256": digest(known),
        "errors": {
            key: {"max_abs": 0.0, "rmse": 0.0}
            for key in (
                "noise",
                "actions",
                "native_parity",
                "zero_embedding_hook_parity",
            )
        },
        "velocity_evaluations": 1230,
        "development_embedding_probe": None,
    }
    zero = {
        "condition_id": raw_id,
        "raw_condition_id": raw_id,
        "observation_id": observation_id,
        "enabled": False,
        "has_effect": False,
        "delta_frobenius": 0,
        "prefix_before_sha256": "a" * 64,
        "prefix_after_sha256": "a" * 64,
        **{
            key: True
            for key in (
                "text_mask_fixed",
                "padding_unchanged",
                "vision_prefix_unchanged",
                "attention_unchanged",
                "positions_unchanged",
            )
        },
    }
    spec = {
        "horizon": 10,
        "model_action_dim": 32,
        "lower": [-1.0] * 7,
        "upper": [1.0] * 7,
    }
    recorder.event(
        "inversion_initialization",
        observation=raw,
        known_noise=known,
        recovered_noise=known,
        reference=zeros,
        roundtrip=zeros,
        native_actions=decoded,
        adapter_actions=decoded,
        zero_embedding_hook=zero,
        action_spec=spec,
        **initial,
    )
    recorder.event(
        "interpolation_gate_reference",
        observation=raw,
        observation_id=observation_id,
        condition_id=raw_id,
        latent=known,
        generated_actions=zeros,
        decoded_actions=decoded,
        solver=protocol["execution_solver"],
        velocity_evaluations=10,
    )
    gate = {
        "passed": True,
        "complete": True,
        "status": "passed",
        "velocity_evaluations_complete": True,
        "velocity_evaluations": 50,
        "reference_velocity_evaluations": 10,
        "solver": protocol["execution_solver"],
        "known_noise_sha256": digest(known),
        "observation_id": observation_id,
        "native_condition_id": raw_id,
        "checks": {},
    }
    source_ids = (oracle["source_a_id"], oracle["source_b_id"])
    from astra_reversal.flow import error_metrics

    for label, (operator, alpha, nonzero) in GATE_CHECKS.items():
        sources = (
            tuple(ctx.banks[key]["provenance"]["source_prompt"] for key in source_ids)
            if nonzero
            else (target, target)
        )
        prov = provenance(ctx, raw, target, sources, operator, alpha, source_ids)
        endpoint = zeros.copy()
        if nonzero:
            endpoint[:, :, :7] = 0.1
        action = ctx.decode(endpoint)
        row = {
            "passed": True,
            "status": "passed",
            "operator": operator,
            "alpha": alpha,
            "expected_effect": "nonzero" if nonzero else "identity",
            "source_ids": list(source_ids) if nonzero else None,
            "condition_id": prov["condition_id"],
            "provenance": prov,
            "errors": error_metrics(zeros, endpoint),
            "controlled_channel_errors": error_metrics(
                zeros[:, :, :7], endpoint[:, :, :7]
            ),
            "decoded_action_errors": error_metrics(decoded, action),
            "velocity_evaluations": 10,
        }
        gate["checks"][label] = row
        recorder.event(
            "interpolation_gate_check",
            label=label,
            observation_id=observation_id,
            latent_sha256=digest(known),
            generated_actions=endpoint,
            decoded_actions=action,
            **row,
        )
    recorder.event("interpolation_gate", **gate)
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
        post_stabilization_observation_sha256=observation_id,
        initial_success=False,
        initial_terminated=False,
    )
    reset["sha256"] = digest(reset)
    attempts = []
    for mode in ("recovered_noise", "known_noise", "policy_fresh"):
        attempt_id = mode + "_1"
        recorder.event(
            "phase_generation",
            mode=mode,
            iteration=1,
            attempt_id=attempt_id,
            observation_step=0,
            observation=raw,
            modified_observation_sha256=observation_id,
            vision=[],
            active_interpolation=None,
            conditioning=None,
            condition_id=raw_id,
            applied_accepted_decision_id=None,
            text_has_effect=False,
            vision_has_effect=False,
            held_text_after_failed_call=False,
            native_condition_fallback=False,
            latent=known,
            generated_actions=zeros,
            controller_actions=decoded.astype(np.float32),
            clipping={"count": 0, "max_abs": 0.0},
            velocity_evaluations=10,
        )
        attempt = {
            "attempt_id": attempt_id,
            "episode_id": entry["episode_id"],
            "mode": mode,
            "iteration": 1,
            "success": True,
            "actions_executed": 1,
            "execute_steps": 5,
            "action_budget": 300,
            "policy_replans": 1,
            "initial_success": False,
            "zero_action_success": False,
            "reset_audit": reset,
            "noise_proposal": None,
            "velocity_evaluations": 10,
        }
        for key in (
            "clipped_values",
            "vision_active_policy_calls",
            "vision_changed_policy_calls",
            "text_nonzero_policy_calls",
            "accepted_decision_policy_calls",
            "native_condition_fallback_policy_calls",
            "actions_with_nonzero_text",
            "actions_with_changed_vision",
            "actions_with_held_text_after_failed_call",
            "native_condition_fallback_actions",
            "actions_with_accepted_decision",
        ):
            attempt[key] = 0
        recorder.event(
            "phase_attempt",
            attempt=attempt,
            snapshots=[
                {"step": 0, "observation": raw},
                {"step": 1, "observation": raw},
            ],
        )
        attempts.append(attempt)
    summary = {
        "status": "complete",
        "episode_id": entry["episode_id"],
        "protocol_sha256": digest(protocol),
        "reset_entry_sha256": digest(entry),
        "checkpoint": ctx.checkpoint,
        "oracle": oracle,
        "source_catalog": donor_catalog(),
        "initialization": initial,
        "interpolation_gate": gate,
        "baseline": attempts[0],
        "controls": {row["mode"]: row for row in attempts[1:]},
        "arms": {arm: {"attempts": [attempts[0]]} for arm in ARMS},
        "physical_cost": {
            "rollouts": 3,
            "simulated_actions": 3,
            "velocity_evaluations": 1310,
            "rollout_velocity_evaluations": 30,
            "initialization_velocity_evaluations": 1230,
            "interpolation_gate_velocity_evaluations": 50,
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    return ctx


def test_full_synthetic_case_reconciles_arrays_unique_work_and_five_probe_solves(
    tmp_path,
):
    directory = tmp_path / "case"
    ctx = complete_case(directory)
    result = CaseAudit(directory, ctx).run()
    assert result["status"] == "passed"
    assert result["counts"]["physical_attempts"] == 3
    assert result["counts"]["physical_actions"] == 3
    assert result["counts"]["velocity_evaluations"] == 1310
    assert result["counts"]["array_files"] == len(
        list((directory / "arrays").glob("*.npy"))
    )


def test_valid_new_array_hash_does_not_authorize_wrong_known_latent(tmp_path):
    directory = tmp_path / "case"
    ctx = complete_case(directory)
    events = rows(directory)
    generation = next(
        row
        for row in events
        if row["kind"] == "phase_generation" and row["mode"] == "known_noise"
    )
    ref = generation["latent"]
    value = np.load(directory / ref["array"], allow_pickle=False)
    value[0, 0, 0] += 0.125
    np.save(directory / ref["array"], value, allow_pickle=False)
    ref["sha256"] = digest(value)
    rewrite(directory, events)
    with pytest.raises(AuditError, match="noise construction"):
        CaseAudit(directory, ctx).run()


def test_rehashed_wrong_internal_endpoint_dtype_fails_native_shape_contract(tmp_path):
    directory = tmp_path / "case"
    ctx = complete_case(directory)
    events = rows(directory)
    initial = next(row for row in events if row["kind"] == "inversion_initialization")
    ref = initial["reference"]
    value = np.load(directory / ref["array"], allow_pickle=False).astype(np.float64)
    np.save(directory / ref["array"], value, allow_pickle=False)
    ref.update(dtype="float64", sha256=digest(value))
    rewrite(directory, events)
    with pytest.raises(AuditError, match="full float32"):
        CaseAudit(directory, ctx).run()


def test_declared_nonzero_gate_fails_when_saved_controlled_endpoint_is_identical(
    tmp_path,
):
    directory = tmp_path / "case"
    ctx = complete_case(directory)
    events = rows(directory)
    label = "tei_nonzero_oracle_sources"
    probe = next(
        row
        for row in events
        if row["kind"] == "interpolation_gate_check" and row["label"] == label
    )
    for name, value in (
        ("generated_actions", np.zeros(FLOW_SHAPE, np.float32)),
        ("decoded_actions", ctx.decode(np.zeros(FLOW_SHAPE, np.float32))),
    ):
        ref = probe[name]
        np.save(directory / ref["array"], value, allow_pickle=False)
        ref["sha256"] = digest(value)
    gate = next(row for row in events if row["kind"] == "interpolation_gate")
    summary = json.loads((directory / "summary.json").read_text())
    for name in ("errors", "controlled_channel_errors", "decoded_action_errors"):
        metric = {"max_abs": 0.0, "rmse": 0.0}
        probe[name] = metric
        gate["checks"][label][name] = metric
        summary["interpolation_gate"]["checks"][label][name] = metric
    rewrite(directory, events)
    (directory / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(AuditError, match="Recomputed weighted probe gate failed"):
        CaseAudit(directory, ctx).run()


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate"])
def test_archive_rejects_unsafe_members(tmp_path, kind):
    path = tmp_path / "archive.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(
            "results/../../escape" if kind == "traversal" else "results/value"
        )
        if kind == "symlink":
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(member)
        else:
            member.size = 1
            archive.addfile(member, io.BytesIO(b"x"))
            if kind == "duplicate":
                archive.addfile(member, io.BytesIO(b"y"))
    with pytest.raises(AuditError):
        _extract_archive(path, tmp_path / "out")


def test_missing_worker_metadata_writes_failed_receipt(tmp_path):
    output = tmp_path / "audit.json"
    with pytest.raises(AuditError, match="metadata"):
        audit_worker(tmp_path, assets=tmp_path, output=output)
    assert json.loads(output.read_text())["status"] == "failed"
