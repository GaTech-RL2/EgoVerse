"""Synthetic recorded pixels test selection, provenance and exported evidence."""

import json

import numpy as np
import pytest
from PIL import Image

from astra_reversal.image_donor_bank import DonorImage
from astra_reversal.image_perturbation_examples import ARMS, export_examples
from astra_reversal.image_perturbations import (
    CAMERAS,
    ImagePerturbationLimits,
    apply_image_perturbations,
)
from astra_reversal.records import Recorder, digest, file_sha256


class Library:
    library_id = "a" * 64

    def metadata(self):
        return {"library_id": self.library_id, "fixture": "synthetic recorded RGB"}

    def resolve(self, donor_id, camera):
        assert donor_id == "fixture-donor" and camera in CAMERAS
        pixels = np.full((224, 224, 3), 200 if camera == CAMERAS[0] else 240, np.uint8)
        return DonorImage(
            donor_id,
            camera,
            pixels,
            {
                "library_id": self.library_id,
                "sample_sha256": "b" * 64,
                "pixels_sha256": digest(pixels),
            },
        )


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_case(
    tmp_path,
    worker=0,
    *,
    success=False,
    phase="development",
    weights=(0, 0.5),
    accepted=True,
    no_rollout=False,
):
    """A small exporter fixture; full experiment auditing is tested separately."""
    parent = tmp_path / f"{phase}_worker_{worker}"
    recorder = Recorder(parent / "case_2_0")
    library = Library()
    runtime = {
        "worker": worker,
        "phase": phase,
        "workflow": "synthetic-" + phase,
        "payload_sha256": "c" * 64,
    }
    write_json(parent / "runtime.json", runtime)
    entry = {
        "episode_id": "fixture:task2:state0",
        "task_id": 2,
        "initial_state_id": 0,
        "instruction": "move the object",
        "seed": worker + (19 if phase == "development" else 37),
    }
    protocol = {"fixture": True}
    recorder.event("case", entry=entry, protocol=protocol)
    summary = {
        "episode_id": entry["episode_id"],
        "status": "complete",
        "protocol_sha256": digest(protocol),
        "image_library": library.metadata(),
        "arms": {},
    }
    bindings = []
    for arm in ARMS:
        name = arm + "_2"
        decisions = []
        if no_rollout:
            summary["arms"][arm] = {"attempts": [{"attempt_id": "baseline"}]}
            continue
        for index, weight in enumerate(weights):
            step = index * 25
            mode = arm.removeprefix("astra_")
            if mode == "demo_blend":
                operations = [
                    {
                        "kind": mode,
                        "camera": CAMERAS[0],
                        "donor_id": "fixture-donor",
                        "alpha": weight,
                    }
                ]
            else:
                operations = [
                    {
                        "kind": mode,
                        "camera": CAMERAS[0],
                        "box_xyxy": [10, 20, 110, 120],
                        "fill_rgb": [127, 127, 127],
                        "strength": weight,
                    }
                ]
            request = {
                "episode_id": entry["episode_id"],
                "attempt_id": name,
                "decision_index": index + 1,
                "observation_step": step,
                "image_mode": mode,
                "target_task": entry["instruction"],
            }
            request["request_fingerprint"] = digest(request)
            request["request_id"] = request["request_fingerprint"][:16]
            recorder.event("image_request", request=request)
            proposal = {
                key: request[key]
                for key in (
                    "episode_id",
                    "attempt_id",
                    "decision_index",
                    "observation_step",
                    "image_mode",
                    "request_id",
                    "request_fingerprint",
                )
            }
            proposal.update(
                decision_id=f"{name}_{index}", image_perturbations=operations
            )
            decision = {
                "decision_index": index + 1,
                "observation_step": step,
                "accepted": accepted,
                "proposal": proposal if accepted else None,
                "error": None if accepted else "synthetic rejected response",
            }
            decisions.append(decision)
            recorder.event(
                "image_decision",
                attempt_id=name,
                decision=decision,
                valid_until_step=step + 25,
            )
            raw = {
                CAMERAS[0]: np.full((224, 224, 3), 30 + index, np.uint8),
                CAMERAS[1]: np.full((224, 224, 3), 60 + index, np.uint8),
                "observation/state": np.arange(8, dtype=np.float32),
            }
            active = operations if accepted else []
            edited, audit = apply_image_perturbations(
                raw,
                active,
                library.resolve,
                limits=ImagePerturbationLimits(allowed_kinds=(mode,)),
            )
            recorder.event(
                "image_generation",
                mode=arm,
                iteration=2,
                attempt_id=name,
                observation_step=step,
                observation=raw,
                modified_observation=edited,
                observation_sha256=digest(raw),
                modified_observation_sha256=digest(edited),
                applied_accepted_decision_id=proposal["decision_id"]
                if accepted
                else None,
                image_perturbations=active,
                image_audit=audit,
                image_has_effect=audit["has_effect"],
                native_condition_fallback=not accepted,
                condition_id=digest({**edited, "prompt": entry["instruction"]}),
            )
            bindings.append(
                {
                    "kind": "provider",
                    "request_fingerprint": request["request_fingerprint"],
                    "decision_id": proposal["decision_id"] if accepted else None,
                    "provider_call": True,
                    "accepted": accepted,
                    "payload_sha256": "d" * 64,
                }
            )
        attempt = {
            "attempt_id": name,
            "mode": arm,
            "iteration": 2,
            "decisions": decisions,
            "success": success,
            "terminated": not success,
            "actions_executed": 30,
            "status": "success" if success else "terminated",
        }
        recorder.event("image_attempt", attempt=attempt)
        summary["arms"][arm] = {"attempts": [{"attempt_id": "baseline"}, attempt]}
    write_json(recorder.directory / "summary.json", summary)
    receipt = {
        "status": "passed",
        "complete": True,
        "episode_id": entry["episode_id"],
        "library_id": library.library_id,
        "summary_sha256": file_sha256(recorder.directory / "summary.json"),
        "events_sha256": file_sha256(recorder.directory / "events.jsonl"),
        "reset_entry_sha256": digest(entry),
        "protocol_sha256": digest(protocol),
        "all_arrays_verified": True,
        "provider_bindings_verified": True,
        "reset_pairing_verified": True,
        "decision_lifetimes_verified": True,
        "bindings": bindings,
        "input_file_sha256": {
            str(path.relative_to(recorder.directory)): file_sha256(path)
            for path in recorder.directory.rglob("*")
            if path.is_file()
        },
    }
    audit = {
        "schema_version": "image-perturbation-audit-1.0",
        "status": "passed",
        "complete": True,
        "library_id": library.library_id,
        **runtime,
        "source_metadata_sha256": {
            "runtime.json": file_sha256(parent / "runtime.json")
        },
        "cases": [receipt],
    }
    audit_path = parent / "audit.json"
    write_json(audit_path, audit)
    return recorder.directory, audit_path


def test_selection_uses_worker_and_event_order_not_success(tmp_path):
    first = make_case(tmp_path, 0, success=False)
    later = make_case(tmp_path, 1, success=True, weights=(1,))
    result = export_examples(
        [later[0], first[0]],
        audit_paths=[later[1], first[1]],
        library=Library(),
        output=tmp_path / "examples",
    )
    assert [row["worker"] for row in result["considered_cases"]] == [0, 1]
    for arm in ARMS:
        row = result["examples"][arm]
        assert row["case"]["worker"] == 0
        assert row["observation_step"] == 25
        assert row["recorded_outcome"]["success"] is False
        assert (
            row["request_sequence"]
            < row["decision_sequence"]
            < row["generation_sequence"]
            < row["outcome_sequence"]
        )
        assert (
            result["considered_cases"][0]["arms"][arm]["counts"][
                "accepted_explicit_noop_decisions"
            ]
            == 1
        )


def test_export_keeps_each_rgb_and_figure_pane_exact_and_source_unchanged(tmp_path):
    case, audit = make_case(tmp_path)
    before = {
        str(path): file_sha256(path) for path in case.rglob("*") if path.is_file()
    }
    output = tmp_path / "examples"
    result = export_examples(
        [case], audit_paths=[audit], library=Library(), output=output
    )
    for arm, row in result["examples"].items():
        assert set(row["cameras"]) == set(CAMERAS)
        with Image.open(output / row["figure"]["file"]) as image:
            figure = np.asarray(image)
            for pane in row["figure"]["panes"]:
                x, y, width, height = pane["xywh"]
                assert (
                    digest(figure[y : y + height, x : x + width])
                    == pane["pixels_sha256"]
                )
        for camera, data in row["cameras"].items():
            for role, image in data["images"].items():
                with Image.open(output / image["file"]) as loaded:
                    assert loaded.size == (224, 224)
                    assert digest(np.asarray(loaded)) == image["pixels_sha256"]
                assert file_sha256(output / image["file"]) == image["file_sha256"]
            assert data["pixel_metrics"]["has_effect"] is (camera == CAMERAS[0])
        assert row["cameras"][CAMERAS[1]]["donor_provenance"] is None
        if arm == "astra_demo_blend":
            assert row["cameras"][CAMERAS[0]]["images"]["donor"][
                "pixels_sha256"
            ] == digest(Library().resolve("fixture-donor", CAMERAS[0]).pixels)
        assert len(row["array_sources"]) == 6
    assert before == {
        str(path): file_sha256(path) for path in case.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"weights": (0, 1e-8)}, "accepted_without_pixel_change"),
        ({"accepted": False}, "no_accepted_decision"),
        ({"no_rollout": True}, "no_intervention_rollout"),
    ],
)
def test_missing_selection_is_explicit_and_never_fabricates_pixels(
    tmp_path, kwargs, reason
):
    case, audit = make_case(tmp_path, **kwargs)
    output = tmp_path / "examples"
    result = export_examples(
        [case], audit_paths=[audit], library=Library(), output=output
    )
    for arm in ARMS:
        assert result["examples"][arm]["status"] == "no_qualifying_generation"
        detail = result["considered_cases"][0]["arms"][arm]
        assert detail["status"] == reason
        if reason == "accepted_without_pixel_change":
            assert detail["counts"]["accepted_explicit_noop_decisions"] == 1
            assert (
                detail["counts"][
                    "accepted_nonzero_spec_generations_without_pixel_change"
                ]
                == 1
            )
    assert not list(output.rglob("*.png"))


def test_development_precedes_evaluation_and_falls_forward_only_when_absent(tmp_path):
    evaluation = make_case(tmp_path, 0, phase="evaluation", success=True)
    development = make_case(tmp_path, 2, phase="development", weights=(0,))
    result = export_examples(
        [evaluation[0], development[0]],
        audit_paths=[evaluation[1], development[1]],
        library=Library(),
        output=tmp_path / "examples",
    )
    assert [row["phase"] for row in result["considered_cases"]] == [
        "development",
        "evaluation",
    ]
    assert result["examples"][ARMS[0]]["case"]["phase"] == "evaluation"
    assert (
        result["considered_cases"][0]["arms"][ARMS[0]]["status"]
        == "accepted_without_pixel_change"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "audit_failed",
        "wrong_library",
        "summary",
        "events",
        "runtime",
        "pixels",
        "provider_binding",
    ],
)
def test_export_rejects_tampered_or_unaudited_evidence(tmp_path, mutation):
    case, audit_path = make_case(tmp_path)
    audit = json.loads(audit_path.read_text())
    if mutation == "audit_failed":
        audit["cases"][0]["all_arrays_verified"] = False
    elif mutation == "wrong_library":
        audit["library_id"] = "0" * 64
    elif mutation in ("summary", "events"):
        path = case / ("summary.json" if mutation == "summary" else "events.jsonl")
        path.write_text(path.read_text() + "\n")
    elif mutation == "runtime":
        (case.parent / "runtime.json").write_text("{}")
    elif mutation == "provider_binding":
        audit["cases"][0]["bindings"][1]["accepted"] = False
    else:
        event = next(
            json.loads(line)
            for line in (case / "events.jsonl").read_text().splitlines()
            if json.loads(line)["kind"] == "image_generation"
        )
        path = case / event["modified_observation"][CAMERAS[0]]["array"]
        pixels = np.load(path)
        pixels[0, 0, 0] ^= 1
        np.save(path, pixels, allow_pickle=False)
    write_json(audit_path, audit)
    with pytest.raises(ValueError):
        export_examples(
            [case],
            audit_paths=[audit_path],
            library=Library(),
            output=tmp_path / "examples",
        )
    assert not (tmp_path / "examples").exists()


def test_export_refuses_overwrite_source_write_or_duplicate_case(tmp_path):
    case, audit = make_case(tmp_path)
    with pytest.raises(ValueError, match="source case"):
        export_examples(
            [case], audit_paths=[audit], library=Library(), output=case / "exports"
        )
    with pytest.raises(ValueError, match="Duplicate"):
        export_examples(
            [case, case],
            audit_paths=[audit, audit],
            library=Library(),
            output=tmp_path / "duplicates",
        )
    output = tmp_path / "examples"
    output.mkdir()
    with pytest.raises(ValueError, match="replace"):
        export_examples([case], audit_paths=[audit], library=Library(), output=output)
    with pytest.raises(ValueError, match="pinned donor library"):
        export_examples(
            [case],
            audit_paths=[audit],
            library=tmp_path / "library",
            output=tmp_path / "library" / "exports",
        )


def test_rehashed_trace_still_requires_exact_request_fingerprint(tmp_path):
    case, audit_path = make_case(tmp_path)
    events_path = case / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    next(row for row in events if row["kind"] == "image_request")["request"][
        "target_task"
    ] = "changed request"
    events_path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    audit = json.loads(audit_path.read_text())
    receipt = audit["cases"][0]
    receipt["events_sha256"] = file_sha256(events_path)
    receipt["input_file_sha256"]["events.jsonl"] = receipt["events_sha256"]
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match="Request/decision/generation"):
        export_examples(
            [case],
            audit_paths=[audit_path],
            library=Library(),
            output=tmp_path / "examples",
        )


def test_rehashed_but_wrong_edited_pixels_fail_exact_replay(tmp_path):
    case, audit_path = make_case(tmp_path)
    events = [
        json.loads(line) for line in (case / "events.jsonl").read_text().splitlines()
    ]
    generation = next(
        row
        for row in events
        if row["kind"] == "image_generation" and row["observation_step"] == 25
    )
    descriptor = generation["modified_observation"][CAMERAS[0]]
    path = case / descriptor["array"]
    pixels = np.load(path)
    pixels[0, 0, 0] ^= 1
    np.save(path, pixels, allow_pickle=False)
    descriptor["sha256"] = digest(pixels)
    (case / "events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n"
    )
    audit = json.loads(audit_path.read_text())
    receipt = audit["cases"][0]
    receipt["events_sha256"] = file_sha256(case / "events.jsonl")
    receipt["input_file_sha256"]["events.jsonl"] = receipt["events_sha256"]
    receipt["input_file_sha256"][descriptor["array"]] = file_sha256(path)
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match="exact pinned pixel replay"):
        export_examples(
            [case],
            audit_paths=[audit_path],
            library=Library(),
            output=tmp_path / "examples",
        )
