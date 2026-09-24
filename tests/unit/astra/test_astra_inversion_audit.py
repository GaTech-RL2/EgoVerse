"""Synthetic recorded traces test audit logic, not real Astra performance."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from astra_reversal.action_adapter import ActionAdapter
from astra_reversal.agent import AstraAgent
from astra_reversal.audit_astra_inversions import analyze, audit_run
from astra_reversal.config import RunConfig
from astra_reversal.controller import Controller
from astra_reversal.records import Recorder, RunManifest, digest

from .conftest import SyntheticBackend, SyntheticEnvironment, SyntheticPolicy


@pytest.fixture
def recorded_reversal(tmp_path, spec):
    def create(*, mutate=None, stop_after=26):
        recorder = Recorder(tmp_path / "synthetic_reversal")
        config, policy = RunConfig(), SyntheticPolicy()
        recorder.manifest(
            RunManifest(
                asdict(config),
                {"synthetic": True},
                spec.as_dict(),
                policy.metadata,
                "synthetic-manifest",
                0,
            )
        )
        agent = AstraAgent(SyntheticBackend(mutate=mutate), config.agent, recorder)
        actions = ActionAdapter(spec, policy.input_transform, policy.output_transform)
        Controller(config, policy, actions, recorder, agent).run_episode(
            SyntheticEnvironment(stop_after=stop_after),
            episode_id="synthetic-episode",
            instruction="move cup",
            seed=1,
        )
        return recorder.directory

    return create


def events(directory):
    return [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text().splitlines()
    ]


def write_events(directory, rows):
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def change_flow_array(directory, step, key, delta, *, update_hash=True):
    rows = events(directory)
    flow = next(
        row
        for row in rows
        if row["kind"] == "flow"
        and row["role"] == "generation"
        and row["observation_step"] == step
    )
    reference = flow[key]
    path = directory / reference["array"]
    value = np.load(path, allow_pickle=False)
    value[0, 0, -1] += delta  # Padding must remain part of the full-tensor audit.
    np.save(path, value, allow_pickle=False)
    if update_hash:
        reference["sha256"] = digest(value)
        write_events(directory, rows)


def test_pairs_real_event_schema_and_separates_fresh_condition_reuse(recorded_reversal):
    report = audit_run(recorded_reversal())
    counts = report["counts"]
    assert report["status"] == "complete"
    assert counts["astra_inversions"] == counts["passing_roundtrips"] == 2
    assert counts["paired_on_first_generation"] == 2
    assert counts["generations"] == 6
    assert (
        counts["subsequent_generations"] == counts["changed_condition_generations"] == 4
    )
    assert (
        counts["subsequent_steps_advanced"]
        == counts["subsequent_observation_ids_changed"]
        == 4
    )
    assert counts["generations_with_wrong_recovered_latent"] == 0
    assert report["all_roundtrips_passed"] and report["all_latents_match_recovered"]
    assert all(plan["full_internal_shape"] == [1, 10, 32] for plan in report["plans"])
    assert report["recorded_model_versions"] == ["synthetic-test-fixture"]


def test_changed_condition_outputs_are_never_reconstruction_errors(recorded_reversal):
    directory = recorded_reversal()
    change_flow_array(directory, 5, "output", 100)
    report = analyze([directory])
    assert report["all_roundtrips_passed"]
    assert report["known_noise_recovery"] is None
    generation = report["runs"][0]["plans"][0]["generations"][1]
    assert generation["same_inverse_condition"] is False
    assert generation["reconstruction_measured"] is False
    assert report["maximum_full_internal_error"] < 1e-5


def test_padding_roundtrip_failure_counts_at_unchanged_tolerance(recorded_reversal):
    directory = recorded_reversal()
    change_flow_array(directory, 0, "output", 0.03)
    report = analyze([directory])
    assert report["action_atol"] == 0.02
    assert report["counts"]["failing_roundtrips"] == 1
    assert report["counts"]["passing_roundtrips"] == 1
    assert not report["all_roundtrips_passed"]
    pair = report["runs"][0]["plans"][0]["same_condition_roundtrip"]
    assert pair["full_internal_reconstruction"]["max_abs"] == pytest.approx(0.03)
    assert pair["full_internal_reconstruction"]["rmse"] == pytest.approx(
        0.03 / np.sqrt(320)
    )
    assert pair["action_channel_reconstruction"]["max_abs"] < 1e-5


def test_valid_hash_does_not_hide_latent_drift(recorded_reversal):
    directory = recorded_reversal()
    change_flow_array(directory, 5, "input", 1)
    report = analyze([directory])
    assert report["all_roundtrips_passed"]  # Initial same-condition replay is intact.
    assert not report["all_latents_match_recovered"]
    assert report["counts"]["generations_with_wrong_recovered_latent"] == 1
    assert report["counts"]["generations_with_latent_changed_within_plan"] == 1
    assert report["runs"][0]["plans"][0]["latent_reuse_verified"] is False


def test_missing_same_condition_replay_cannot_pass(recorded_reversal):
    directory = recorded_reversal(stop_after=6)
    rows = [
        row
        for row in events(directory)
        if not (
            row["kind"] == "flow"
            and row["role"] == "generation"
            and row["observation_step"] == 0
        )
    ]
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
    write_events(directory, rows)
    report = analyze([directory])
    assert report["counts"]["unpaired_inversions"] == 1
    assert report["counts"]["paired_same_condition_roundtrips"] == 0
    assert report["maximum_full_internal_error"] is None
    assert not report["all_roundtrips_passed"]


def test_tampered_array_fails_integrity_instead_of_reporting_metrics(recorded_reversal):
    directory = recorded_reversal()
    change_flow_array(directory, 0, "output", 1, update_hash=False)
    with pytest.raises(ValueError, match="integrity"):
        audit_run(directory)
    report = analyze([directory])
    assert report["status"] == "incomplete_or_invalid_recording"
    assert report["runs_with_verified_records"] == 0
    assert report["maximum_full_internal_error"] is None


def test_fallback_refresh_ends_recovered_latent_interval(recorded_reversal):
    def invalid_second_refresh(response, request):
        if request["observation_step"] >= 20:
            response["observation_step"] = -1

    report = audit_run(recorded_reversal(mutate=invalid_second_refresh))
    assert report["counts"]["astra_inversions"] == 1
    assert report["counts"]["generations"] == 4
    assert report["counts"]["fallback_flows"] == 2
    assert report["counts"]["orphan_generation_flows"] == 0
    assert report["all_roundtrips_passed"]
