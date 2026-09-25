"""Small recording-corruption cases for the frozen CPU audit snapshot."""

import copy
import importlib.util
import io
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "astra_reversal/reports/iterative_interventions/audit_source"
FIXTURE = ROOT / "tests/fixtures/astra/intervention_audit/arrays.npz"
EPISODE = "fixture:seed19:task0:state0"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    # These checks exercise stored arrays/reset records, not tokenization. The
    # frozen CLI imports this optional package eagerly, but must never use it here.
    optional = types.ModuleType("sentencepiece")

    def no_tokenizer(*args, **kwargs):
        raise AssertionError("A saved-array check attempted tokenization")

    optional.SentencePieceProcessor = no_tokenizer
    monkeypatch.setitem(sys.modules, "sentencepiece", optional)
    result = []
    for name in ("audit_worker", "verify_final", "linux_vision"):
        spec = importlib.util.spec_from_file_location(
            "test_snapshot_" + name, SOURCE / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def npy_bytes(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def array_reference(module, name, value):
    return {
        "array": "arrays/" + name + ".npy",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": module.digest(value),
    }


def empty_reader(module):
    audit = module.CaseAudit.__new__(module.CaseAudit)
    audit.refs, audit.inventory, audit.values = {}, {}, {}
    audit.consumers, audit.groups = {}, {}
    return audit


def add_array(module, audit, name, value):
    ref = array_reference(module, name, value)
    audit.refs[ref["array"]] = ref
    audit.array(ref["array"], npy_bytes(value))
    return ref


def small_case(module):
    audit = empty_reader(module)
    with np.load(FIXTURE, allow_pickle=False) as saved:
        refs = {
            name: add_array(module, audit, name, saved[name]) for name in saved.files
        }
    audit.name = "case_fixture"
    audit.data = {"events.jsonl": b"synthetic fixture", "summary.json": b"{}"}
    audit.entry = {"episode_id": EPISODE}
    audit.protocol = {"seed": 19}
    audit.metadata = {"runtime.json": {"phase": "evaluation"}}
    audit.initial = {
        "known_noise": refs["known"],
        "recovered_noise": refs["known"],
        "reference": refs["full_actions"],
        "roundtrip": refs["full_actions"],
        "native_actions": refs["controller_actions"],
        "adapter_actions": refs["controller_actions"],
        "errors": {
            key: {"max_abs": 0.0, "rmse": 0.0}
            for key in (
                "noise",
                "actions",
                "native_parity",
                "zero_embedding_hook_parity",
            )
        },
        "passed": True,
        "velocity_evaluations": 1230,
        "action_spec": {"lower": [-2.0] * 7, "upper": [2.0] * 7},
    }
    audit.case = {"basis": refs["basis"], "basis_id": refs["basis"]["sha256"]}
    audit.q01, audit.q99 = np.zeros(7), np.full(7, 2.0 - 1e-6)
    audit.proposals = {}
    audit.generations, audit.rows = [], []
    for mode in ("reversal_identity", "known_noise"):
        audit.generations.append(
            {
                "mode": mode,
                "candidate_id": mode,
                "iteration": 1,
                "observation_step": 0,
                "latent": refs["known"],
                "generated_actions": refs["full_actions"],
                "controller_actions": refs["controller_actions"],
                "velocity_evaluations": 10,
            }
        )
        audit.rows.append(
            {
                "kind": "attempt",
                "attempt": {
                    "candidate_id": mode,
                    "iteration": 1,
                    "proposal": None,
                    "actions_executed": 5,
                    "policy_replans": 1,
                    "velocity_evaluations": 10,
                    "reset_audit": {"identity": "same-scene"},
                    "success": False,
                },
            }
        )
    audit.verified_conditions = len(audit.generations)
    audit.conditioning_counts = Counter()
    audit.max_text_relative = audit.vision_effects = 0
    audit.summary = {
        "baseline": audit.rows[0]["attempt"],
        "controls": {"known_noise": audit.rows[1]["attempt"]},
        "arms": {},
        "physical_cost": {"velocity_evaluations": 1250},
    }
    return audit


def reset_rows(module):
    state = np.asarray([0.0, 0.5, -0.5])
    model = {"body_pos": np.asarray([[0.0, 0.0, 1.0]])}
    entry = {
        "episode_id": EPISODE,
        "seed": 19,
        "reset_state": state.tolist(),
        "reset_state_sha256": module.digest(state),
        "reset_model": {key: value.tolist() for key, value in model.items()},
        "reset_model_sha256": module.digest(model),
        "bddl_sha256": "b" * 64,
    }
    with np.load(FIXTURE, allow_pickle=False) as saved:
        noise = array_reference(module, "known", saved["known"])
    observation = {"state": array_reference(module, "state", state)}
    initial = {
        "kind": "inversion_initialization",
        "observation": observation,
        "known_noise": noise,
        "recovered_noise": noise,
        "known_noise_sha256": noise["sha256"],
        "recovered_noise_sha256": noise["sha256"],
        "zero_embedding_hook": {"observation_id": "o" * 64},
    }
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
        post_stabilization_model_sha256=entry["reset_model_sha256"],
        post_stabilization_state_sha256="s" * 64,
        post_stabilization_observation_sha256="o" * 64,
        stabilization_steps=10,
        initial_success=False,
        initial_terminated=False,
    )
    reset["sha256"] = module.digest(reset)
    rows = [{"kind": "case", "entry": entry}, initial]
    for _ in range(2):
        rows.extend(
            [
                {"kind": "attempt", "attempt": {"reset_audit": copy.deepcopy(reset)}},
                {
                    "kind": "candidate_generation",
                    "observation_step": 0,
                    "observation": copy.deepcopy(observation),
                },
            ]
        )
    return rows


def test_snapshot_bytes_are_frozen(modules):
    assert len(modules[1].verify_snapshot(SOURCE)) == 64


def test_corrupted_array_and_duplicate_member_are_rejected(modules):
    module = modules[0]
    audit = empty_reader(module)
    with np.load(FIXTURE, allow_pickle=False) as saved:
        original = saved["known"].copy()
    ref = array_reference(module, "noise", original)
    audit.refs[ref["array"]] = ref
    corrupt = original.copy()
    corrupt[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="content hash"):
        audit.array(ref["array"], npy_bytes(corrupt))
    audit.array(ref["array"], npy_bytes(original))
    with pytest.raises(ValueError, match="duplicate"):
        audit.array(ref["array"], npy_bytes(original))


def test_valid_noise_and_control_fixture_passes(modules):
    result = small_case(modules[0]).finish({})
    assert result["rollouts"] == result["generations_verified"] == 2
    assert result["maximum_noise_expression_error"] == 0


def test_self_consistent_array_cannot_replace_the_recovered_latent(modules):
    module = modules[0]
    audit = small_case(module)
    changed = audit.value(audit.initial["recovered_noise"]).copy()
    changed[0, 0, 0] += 0.01
    audit.generations[0]["latent"] = add_array(module, audit, "wrong_latent", changed)
    with pytest.raises(ValueError, match="latent differs"):
        audit.finish({})


def test_valid_reset_records_bind_to_manifest_and_initial_arrays(modules):
    module = modules[1]
    result = module.verify_reset_records(reset_rows(module))
    assert result["rollout_resets"] == result["initial_observations"] == 2


def test_uniform_wrong_reset_is_rejected_even_with_valid_digests(modules):
    module = modules[1]
    rows = reset_rows(module)
    for row in rows:
        if row["kind"] == "attempt":
            reset = row["attempt"]["reset_audit"]
            reset.pop("sha256")
            reset["episode_id"] = "another-scene"
            reset["sha256"] = module.digest(reset)
    with pytest.raises(ValueError, match="manifest entry"):
        module.verify_reset_records(rows)


def test_first_generation_from_another_observation_is_rejected(modules):
    module = modules[1]
    rows = reset_rows(module)
    rows[-1]["observation"]["state"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="initial arrays"):
        module.verify_reset_records(rows)


def test_renderer_matches_saved_linux_kernel_bytes(modules):
    import json

    reference = json.loads(FIXTURE.with_name("linux_blend_reference.json").read_text())
    for case in reference["cases"]:
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        layer = np.full_like(image, case["target"])
        result = modules[2].blend_linux(image, layer, case["gain"])
        assert modules[0].digest(result) == case["output_sha256"]


def test_renderer_does_not_admit_a_one_byte_pixel_error(modules):
    import json

    reference = json.loads(FIXTURE.with_name("linux_blend_reference.json").read_text())
    case = next(
        row for row in reference["cases"] if row["gain"] == 0.85 and row["target"] == 0
    )
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = modules[2].blend_linux(image, np.zeros_like(image), case["gain"])
    assert result[6, 4] == 15  # input100; a fused float32 result can truncate to14.
    result[6, 4] -= 1
    assert modules[0].digest(result) != case["output_sha256"]
