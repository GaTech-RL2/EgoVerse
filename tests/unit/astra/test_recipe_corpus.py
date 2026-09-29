"""Executed-prefix and provenance regression tests; no policy or network calls."""

import io
import json
import tarfile

import numpy as np
import pytest

from astra_reversal.interpolation_catalog import donor_catalog
from astra_reversal.recipe_corpus import (
    CorpusSource,
    build_corpus,
    iter_samples,
    load_corpus,
    plan_corpus,
)
from astra_reversal.records import digest, file_sha256


@pytest.fixture
def recorded_case(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    arrays = {}

    def array(value):
        name = f"arrays/{len(arrays):08d}.npy"
        arrays[name] = value.copy()
        return {
            "array": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": digest(value),
        }

    entry = {
        "episode_id": "task:seed29:state0",
        "instruction": "put the milk on the plate",
    }
    spec = {
        "action_spec_id": "spec",
        "horizon": 10,
        "model_action_dim": 32,
        "lower": [-1.0] * 7,
        "upper": [1.0] * 7,
    }
    baseline = {
        "attempt_id": "recovered_noise_1",
        "mode": "recovered_noise",
        "success": False,
    }
    attempt = {
        "attempt_id": "astra_tli_2",
        "mode": "astra_tli",
        "iteration": 2,
        "success": True,
        "initial_success": False,
        "actions_executed": 7,
        "execute_steps": 5,
        "reset_audit": {"sha256": "reset"},
    }
    proposal = {
        "decision_id": "accepted-0",
        "attempt_id": attempt["attempt_id"],
        "observation_step": 0,
        "decision_index": 1,
        "source_a_id": "14",
        "source_b_id": "18",
        "alpha": 0.25,
    }
    active = {k: proposal[k] for k in ("source_a_id", "source_b_id", "alpha")}
    prompts = {r["source_id"]: r["prompt"] for r in donor_catalog()}
    events = [
        {"kind": "case", "entry": entry},
        {"kind": "inversion_initialization", "action_spec": spec},
        {
            "kind": "interpolation_decision",
            "attempt_id": attempt["attempt_id"],
            "decision": {"accepted": True, "proposal": proposal},
        },
    ]
    chunks = []
    for index, step in enumerate((0, 5)):
        chunk = np.full((10, 7), 0.99, dtype=np.float32)
        chunk[:5] = np.arange(35, dtype=np.float32).reshape(5, 7) / 50 + index / 10
        chunks.append(chunk)
        events.append(
            {
                "kind": "phase_generation",
                "attempt_id": attempt["attempt_id"],
                "observation_step": step,
                "observation": {
                    "observation/image": array(np.full((224, 224, 3), index, np.uint8)),
                    "observation/wrist_image": array(
                        np.full((224, 224, 3), index + 20, np.uint8)
                    ),
                    "observation/state": array(np.arange(8, dtype=np.float32)),
                },
                "controller_actions": array(chunk),
                "generated_actions": {"sha256": "unexecuted-model-is-not-a-label"},
                "latent": {"sha256": "recorded-noise-is-provenance-only"},
                "condition_id": "condition",
                "clipping": {"count": 0, "max_abs": 0.0},
                "active_interpolation": active.copy(),
                "applied_accepted_decision_id": proposal["decision_id"],
                "conditioning": {
                    "operator": "tli",
                    "alpha": 0.25,
                    "target_prompt": entry["instruction"],
                    "source_prompts": [prompts["14"], prompts["18"]],
                },
                "vision": [],
                "vision_has_effect": False,
            }
        )
    events.append({"kind": "phase_attempt", "attempt": attempt})
    summary = {
        "episode_id": entry["episode_id"],
        "protocol_sha256": "protocol",
        "baseline": baseline,
        "controls": {},
        "arms": {"astra_tli": {"attempts": [baseline, attempt]}},
    }
    archive = tmp_path / "worker.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            member = tarfile.TarInfo("results/case/" + name)
            member.size = len(buffer.getvalue())
            stream.addfile(member, io.BytesIO(buffer.getvalue()))

    def seal():
        (case / "summary.json").write_text(json.dumps(summary))
        (case / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
        hashes = {
            name: file_sha256(case / name) for name in ("summary.json", "events.jsonl")
        }
        audit = {
            "status": "passed",
            "complete": True,
            "archive": {
                "sha256": file_sha256(archive),
                "bytes": archive.stat().st_size,
            },
            "cases": [
                {
                    "status": "passed",
                    "episode_id": entry["episode_id"],
                    "protocol_sha256": "protocol",
                    "summary_sha256": hashes["summary.json"],
                    "events_sha256": hashes["events.jsonl"],
                }
            ],
        }
        feedback = {
            "status": "passed",
            "episode_id": entry["episode_id"],
            "protocol_sha256": "protocol",
            "input_file_sha256": hashes,
        }
        (tmp_path / "arrays_audit.json").write_text(json.dumps(audit))
        (tmp_path / "feedback.json").write_text(json.dumps(feedback))

    seal()
    source = CorpusSource(
        "source",
        "phase_interpolation",
        case,
        archive,
        "results/case",
        tmp_path / "arrays_audit.json",
        tmp_path / "feedback.json",
        ("astra_tli",),
    )
    return source, events, summary, chunks, seal


def test_executed_prefix_and_terminal_mask(recorded_case, tmp_path):
    source, _, _, chunks, _ = recorded_case
    result = build_corpus([source], tmp_path / "corpus")
    assert (
        result["trajectory_count"],
        result["window_count"],
        result["executed_action_count"],
    ) == (1, 2, 7)
    _, arrays = load_corpus(tmp_path / "corpus")
    np.testing.assert_array_equal(arrays["controller_actions"][0], chunks[0][:5])
    np.testing.assert_array_equal(arrays["controller_actions"][1, :2], chunks[1][:2])
    np.testing.assert_array_equal(
        arrays["executed_mask"], [[True] * 5, [True, True, False, False, False]]
    )
    assert not arrays["controller_actions"][1, 2:].any()
    assert arrays["controller_actions"].shape == (2, 5, 7)
    samples = list(iter_samples(tmp_path / "corpus"))
    assert samples[1]["executed_actions"].shape == (2, 7)
    assert samples[1]["original_prompt"] == "put the milk on the plate"
    assert samples[1]["choice"] == {
        "operator": "tli",
        "source_a_id": "14",
        "source_b_id": "18",
        "alpha": 0.25,
    }
    assert samples[1]["kind"] == "correction"


def test_deterministic_archive_output(recorded_case, tmp_path):
    source = recorded_case[0]
    a = build_corpus([source], tmp_path / "a")
    b = build_corpus([source], tmp_path / "b")
    assert a["corpus_id"] == b["corpus_id"]
    assert (tmp_path / "a/arrays.npz").read_bytes() == (
        tmp_path / "b/arrays.npz"
    ).read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_generation",
        "stale_choice",
        "wrong_prompt",
        "vision",
        "missing_completion",
    ],
)
def test_invalid_recording_rejected_even_with_updated_receipt(recorded_case, mutation):
    source, events, _, _, seal = recorded_case
    generation = next(e for e in events if e["kind"] == "phase_generation")
    if mutation == "missing_generation":
        events.remove(generation)
    elif mutation == "stale_choice":
        generation["active_interpolation"]["alpha"] = 0.1
    elif mutation == "wrong_prompt":
        generation["conditioning"]["source_prompts"][0] = "unrelated prompt"
    elif mutation == "vision":
        generation["vision_has_effect"] = True
    else:
        events.pop()
    seal()
    with pytest.raises(ValueError):
        plan_corpus([source])


def test_hash_mismatch_rejected(recorded_case):
    source = recorded_case[0]
    path = source.case_dir / "events.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="case bytes"):
        plan_corpus([source])


def test_array_content_revalidated(recorded_case, tmp_path):
    source, events, _, _, seal = recorded_case
    generation = next(e for e in events if e["kind"] == "phase_generation")
    generation["controller_actions"]["sha256"] = "0" * 64
    seal()
    with pytest.raises(ValueError, match="Selected NPY"):
        build_corpus([source], tmp_path / "invalid")
    assert not (tmp_path / "invalid/metadata.json").exists()


def test_failed_attempts_are_not_labels(recorded_case):
    source, events, summary, _, seal = recorded_case
    summary["arms"]["astra_tli"]["attempts"][-1]["success"] = False
    events[-1]["attempt"]["success"] = False
    seal()
    assert plan_corpus([source])["window_count"] == 0


def test_holding_a_valid_prior_choice_is_explicit(recorded_case):
    source, events, _, _, seal = recorded_case
    generation = [e for e in events if e["kind"] == "phase_generation"][1]
    generation["held_text_after_failed_call"] = True
    events.insert(
        events.index(generation),
        {
            "kind": "interpolation_decision",
            "attempt_id": "astra_tli_2",
            "decision": {"accepted": False, "proposal": None},
        },
    )
    seal()
    row = plan_corpus([source])["windows"][1]
    assert (
        row["held_after_provider_error"] and row["accepted_decision_id"] == "accepted-0"
    )
