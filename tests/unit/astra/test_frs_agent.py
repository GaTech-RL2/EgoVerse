"""CPU-only FRS request, response and transport tests; all evidence is synthetic."""

import base64
import copy
import hashlib
import io
import json
import urllib.error

import numpy as np
import pytest
from PIL import Image

from astra_reversal.agent import CAMERAS
from astra_reversal.astra_client import ClientError
from astra_reversal.frs_agent import (
    PROMPT_TEMPLATE_VERSION,
    ROLES,
    SCHEMA_VERSION,
    SYSTEM_PROMPTS,
    FRSClient,
    action_edit_request,
    build_payload,
    critique_request,
    direction_request,
    judge_request,
    parse_proposal,
    prompt_manifest,
    response_schema,
    summarize_calls,
)
from astra_reversal.records import digest

MODEL = "synthetic-provider/frs-test"
TEST_KEY = "synthetic-frs-test-key-not-a-real-credential"
IDENTITY = (
    "schema_version",
    "role",
    "episode_id",
    "attempt_id",
    "request_index",
    "observation_step",
    "request_id",
)


def observation(value=17):
    return {
        **{
            camera: np.full((16, 16, 3), (value + index) % 256, dtype=np.uint8)
            for index, camera in enumerate(CAMERAS)
        },
        "observation/state": np.array([0.3, 0.2, 0.1, 0, 0, 0, 0.02, -0.02]),
    }


def rollout(attempt_id="candidate-1", steps=10):
    return {
        "attempt_id": attempt_id,
        "snapshots": [
            {
                "label": "initial" if step == 0 else "final",
                "step": step,
                "observation": observation(step + 17),
            }
            for step in ([0, steps] if steps else [0])
        ],
        "executed_actions": np.full((steps, 7), 0.123456789123),
    }


def request_for(role="action_edit", **overrides):
    identity = {
        "episode_id": "synthetic-case",
        "attempt_id": "candidate-1",
        "request_index": 1,
        "observation_step": 10 if role in ("critique", "judge") else 0,
        "target_task": "Place the cup in the bowl.",
    }
    if role == "paper_direction":
        return direction_request(
            **(identity | {"external_image": observation()[CAMERAS[0]]} | overrides)
        )
    if role == "action_edit":
        return action_edit_request(
            **(
                identity
                | {"observation": observation(), "native_actions": np.zeros((10, 7))}
                | overrides
            )
        )
    if role == "critique":
        return critique_request(**(identity | {"rollout": rollout()} | overrides))
    return judge_request(
        **(
            identity
            | {"incumbent": rollout("incumbent-0"), "candidate": rollout()}
            | overrides
        )
    )


def proposal_for(request, **overrides):
    evidence = {
        "attempt_id": request["attempt_id"],
        "step": request["observation_step"],
        "camera": CAMERAS[0],
        "observation": "The gripper is visible beside the cup.",
    }
    choices = {
        "paper_direction": {
            "fine": False,
            "coords": [-1, 0, 1],
            "motion_amount": "less",
            "justification": "The gripper remains below the nearby rim.",
        },
        "action_edit": {
            "mode": "edit",
            "delta_xyz": [0.1, -0.5, 0.5],
            "gripper": "keep",
            "apply_steps": 4,
            "justification": "The hand remains beside the visible bowl.",
        },
        "critique": {
            "failure_assessment": "The cup remains visible; grasp is uncertain.",
            "rules": [
                {
                    "rule_id": "lift-if-held",
                    "trigger": "Cup visibly moves with the closing gripper.",
                    "action": "Lift before moving toward the bowl.",
                }
            ],
            "evidence": [evidence],
        },
        "judge": {"verdict": "uncertain", "evidence": [evidence]},
    }
    return {
        **{key: request[key] for key in IDENTITY},
        "response_id": f"response-{request['request_index']}",
        **choices[request["role"]],
        **overrides,
    }


def envelope_for(request, **proposal_changes):
    return {
        "id": "synthetic-provider-response",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(proposal_for(request, **proposal_changes)),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


class FakeResponse(io.BytesIO):
    status = 200


class FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(
            response if isinstance(response, bytes) else json.dumps(response).encode()
        )


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", TEST_KEY)

    def build(*responses, **kwargs):
        client = FRSClient(
            model=MODEL, response_log=tmp_path / "provider.jsonl", **kwargs
        )
        client.opener = FakeOpener(*responses)
        return client

    return build


def attached_arrays(payload):
    return [
        np.array(
            Image.open(
                io.BytesIO(base64.b64decode(row["image_url"]["url"].split(",", 1)[1]))
            )
        )
        for row in payload["messages"][1]["content"]
        if row["type"] == "image_url"
    ]


def test_direction_has_only_external_raw_plus_distinct_guide_and_paper_conventions():
    raw = observation()[CAMERAS[0]]
    guide = raw.copy()
    guide[:, 8] = [255, 0, 255]
    request = request_for("paper_direction", guide_image=guide)
    payload = build_payload(request, MODEL)
    images = attached_arrays(payload)
    assert len(images) == 2
    np.testing.assert_array_equal(images[0], raw)
    np.testing.assert_array_equal(images[1], guide)
    context = json.loads(payload["messages"][1]["content"][0]["text"])["request"]
    assert set(context["inputs"]) == {"external_image", "guide_image"}
    assert context["inputs"]["external_image"]["pixels_sha256"] == digest(raw)
    assert context["inputs"]["guide_image"]["pixels_sha256"] == digest(guide)
    assert (
        "x is camera\ndepth (-1 farther, +1 closer)"
        in SYSTEM_PROMPTS["paper_direction"]
    )
    assert (
        "y is image horizontal (-1 left, +1 right)" in SYSTEM_PROMPTS["paper_direction"]
    )
    assert (
        len(attached_arrays(build_payload(request_for("paper_direction"), MODEL))) == 1
    )
    with pytest.raises(ClientError, match="geometry"):
        request_for("paper_direction", guide_image=np.zeros((8, 8, 3), np.uint8))


def test_action_request_preserves_native_chunk_raw_pixels_and_eight_value_state():
    raw = observation()
    native = np.linspace(-1, 1, 70).reshape(10, 7)
    request = request_for(observation=raw, native_actions=native)
    payload = build_payload(request, MODEL)
    context = json.loads(payload["messages"][1]["content"][0]["text"])["request"]
    assert context["inputs"]["native_actions"] == native.tolist()
    assert (
        context["inputs"]["observation"]["observation/state"]
        == raw["observation/state"].tolist()
    )
    for camera, image in zip(CAMERAS, attached_arrays(payload), strict=True):
        np.testing.assert_array_equal(image, raw[camera])
        assert context["inputs"]["observation"][camera]["pixels_sha256"] == digest(
            image
        )
    raw[CAMERAS[0]][0, 0] = 0
    native[0, 0] = 0
    assert request["inputs"]["native_actions"][0][0] == -1
    assert attached_arrays(payload)[0][0, 0, 0] == 17


@pytest.mark.parametrize("role", ROLES)
def test_all_roles_have_exact_identity_and_local_full_fingerprint(role):
    request = request_for(role)
    proposal = parse_proposal(json.dumps(proposal_for(request)), request)
    assert proposal["schema_version"] == SCHEMA_VERSION
    assert proposal["request_fingerprint"] == request["request_fingerprint"]
    assert request["request_id"] == request["request_fingerprint"][:16]
    schema = response_schema(request)
    assert schema["additionalProperties"] is False
    assert "request_fingerprint" not in schema["properties"]
    assert set(proposal) - {"request_fingerprint"} == set(schema["required"])
    with pytest.raises(ClientError, match="exactly"):
        parse_proposal(
            proposal, request
        )  # Full hash is attached locally, never echoed.
    with pytest.raises(ClientError, match="request_id"):
        parse_proposal(proposal_for(request, request_id="0" * 16), request)


def test_critique_rules_can_feed_editor_without_transmitting_measured_outcome():
    request = request_for("critique")
    critique = parse_proposal(proposal_for(request), request)
    editor = request_for(rules=critique["rules"], attempt_id="candidate-2")
    assert editor["inputs"]["rules"] == critique["rules"]
    assert "failure_assessment" not in editor["inputs"]
    judged = request_for("judge")
    assert set(judged["inputs"]) == {"incumbent", "candidate"}
    assert (
        parse_proposal(proposal_for(judged, verdict="better"), judged)["verdict"]
        == "better"
    )


@pytest.mark.parametrize("role", ["critique", "judge"])
@pytest.mark.parametrize("field", ["success", "reward", "object_poses", "termination"])
def test_completed_rollout_rejects_privileged_or_outcome_metadata(role, field):
    observed = rollout()
    observed[field] = False
    with pytest.raises(ClientError, match="exactly"):
        request_for(
            role, **{("rollout" if role == "critique" else "candidate"): observed}
        )


def test_rollout_binding_requires_current_candidate_complete_action_range_and_raw_state():
    bad = rollout()
    bad["snapshots"] = bad["snapshots"][1:]
    with pytest.raises(ClientError, match="span"):
        request_for("critique", rollout=bad)
    bad = rollout()
    bad["snapshots"][-1]["step"] = 11
    with pytest.raises(ClientError, match="span"):
        request_for("critique", rollout=bad)
    bad = rollout()
    bad["snapshots"][0]["observation"]["object_poses"] = {}
    with pytest.raises(ClientError, match="exactly"):
        request_for("critique", rollout=bad)
    with pytest.raises(ClientError, match="identity"):
        request_for("critique", attempt_id="another-rollout")
    with pytest.raises(ClientError, match="distinct"):
        request_for("judge", incumbent=rollout())
    with pytest.raises(ClientError, match="identity"):
        request_for("judge", observation_step=0)
    empty = request_for("critique", rollout=rollout(steps=0), observation_step=0)
    assert empty["inputs"]["rollout"]["executed_actions"] == []


@pytest.mark.parametrize("role,expected_images", [("critique", 4), ("judge", 8)])
def test_completed_feedback_contains_all_actions_with_explicit_wire_rounding(
    role, expected_images
):
    request = request_for(role)
    original_hash = digest(request)
    payload = build_payload(request, MODEL)
    context = json.loads(payload["messages"][1]["content"][0]["text"])["request"]
    key = "rollout" if role == "critique" else "candidate"
    sent = context["inputs"][key]
    assert len(attached_arrays(payload)) == expected_images
    assert len(sent["executed_actions"]) == 10
    assert sent["executed_actions"][0][0] == 0.123457
    assert request["inputs"][key]["executed_actions"][0][0] == 0.123456789123
    assert "six decimal places" in sent["executed_action_display"]
    assert digest(request) == original_hash
    assert all("base64" not in json.dumps(row) for row in sent["snapshots"])


@pytest.mark.parametrize(
    "role,changes",
    [
        ("paper_direction", {"fine": True}),
        ("paper_direction", {"coords": [0, 0, 0]}),
        ("paper_direction", {"coords": [True, 0, 1]}),
        ("paper_direction", {"coords": [0.5, 0, 1]}),
        ("paper_direction", {"coords": [-2, 0, 1]}),
        ("paper_direction", {"motion_amount": "medium"}),
        ("action_edit", {"delta_xyz": [0, 0, 0.5001]}),
        ("action_edit", {"delta_xyz": [0, 0, True]}),
        ("action_edit", {"apply_steps": 0}),
        ("action_edit", {"apply_steps": 11}),
        ("action_edit", {"apply_steps": True}),
        ("action_edit", {"gripper": 1}),
        ("action_edit", {"mode": "defer"}),
        ("action_edit", {"rotation_offset": [0, 0, 0]}),
        ("judge", {"verdict": "success"}),
        ("judge", {"confidence": 0.9}),
        ("critique", {"rules": [{"rule_id": "r", "trigger": "x", "action": "y"}] * 2}),
    ],
)
def test_response_rejects_ambiguous_direction_unbounded_edits_and_extra_reward_fields(
    role, changes
):
    request = request_for(role)
    with pytest.raises(ClientError):
        parse_proposal(proposal_for(request, **changes), request)


def test_deferral_is_explicit_and_empty_critique_rule_set_is_valid():
    request = request_for("paper_direction")
    parsed = parse_proposal(proposal_for(request, fine=True, coords=[0, 0, 0]), request)
    assert parsed["fine"] is True
    request = request_for()
    parsed = parse_proposal(
        proposal_for(request, mode="defer", delta_xyz=[0, 0, 0]), request
    )
    assert parsed["gripper"] == "keep"
    with pytest.raises(ClientError, match="Deferral"):
        parse_proposal(
            proposal_for(request, mode="defer", delta_xyz=[0, 0, 0], gripper="close"),
            request,
        )
    request = request_for("critique")
    assert parse_proposal(proposal_for(request, rules=[]), request)["rules"] == []


@pytest.mark.parametrize("role", ["critique", "judge"])
@pytest.mark.parametrize("mutation", ["attempt_id", "step", "camera", "empty"])
def test_evidence_must_cite_actual_supplied_observations(role, mutation):
    request = request_for(role)
    proposal = proposal_for(request)
    if mutation == "empty":
        proposal["evidence"] = []
    else:
        proposal["evidence"][0][mutation] = {
            "attempt_id": "other",
            "step": 5,
            "camera": "hidden-camera",
        }[mutation]
    with pytest.raises(ClientError):
        parse_proposal(proposal, request)


def test_recent_history_contains_error_and_cannot_cross_attempts_or_reuse_response_ids():
    first = request_for()
    accepted = {
        "request_index": 1,
        "observation_step": 0,
        "proposal": parse_proposal(proposal_for(first), first),
        "accepted": True,
        "error": None,
    }
    failed = {
        "request_index": 2,
        "observation_step": 10,
        "proposal": None,
        "accepted": False,
        "error": "Inference endpoint returned HTTP 503",
    }
    third = request_for(
        request_index=3, observation_step=20, previous_decisions=[accepted, failed]
    )
    assert third["inputs"]["previous_decisions"][-1]["error"] == failed["error"]
    assert parse_proposal(proposal_for(third), third)["request_index"] == 3
    with pytest.raises(ClientError, match="fresh response_id"):
        parse_proposal(
            proposal_for(third, response_id=accepted["proposal"]["response_id"]), third
        )
    with pytest.raises(ClientError, match="attempt_id"):
        request_for(
            attempt_id="other-arm",
            request_index=3,
            observation_step=20,
            previous_decisions=[accepted, failed],
        )
    with pytest.raises(ClientError, match="increasing"):
        request_for(
            request_index=3, observation_step=20, previous_decisions=[failed, accepted]
        )
    bad = copy.deepcopy(accepted)
    bad["proposal"]["request_id"] = "0" * 16
    with pytest.raises(ClientError, match="fingerprint"):
        request_for(request_index=2, observation_step=10, previous_decisions=[bad])
    # The caller deliberately supplies only the most recent two rows, not all 29.
    request_for(request_index=30, observation_step=290, previous_decisions=[failed])


@pytest.mark.parametrize(
    "field",
    [
        "episode_id",
        "attempt_id",
        "observation_step",
        "request_index",
        "role",
        "schema_version",
    ],
)
def test_every_response_identity_field_is_exact(field):
    request = request_for()
    value = True if type(request[field]) is int else "another-identity"
    with pytest.raises(ClientError, match="echo"):
        parse_proposal(proposal_for(request, **{field: value}), request)


def test_complete_hash_detects_same_short_identity_with_changed_pixels_native_or_rules():
    original = request_for()
    for field in ("native_actions", "rules", "image"):
        changed = copy.deepcopy(original)
        if field == "native_actions":
            changed["inputs"][field][0][0] = 0.1
        elif field == "rules":
            changed["inputs"][field] = [
                {"rule_id": "r", "trigger": "visible tilt", "action": "lift"}
            ]
        else:
            changed["inputs"]["observation"][CAMERAS[0]] = request_for(
                observation=observation(18)
            )["inputs"]["observation"][CAMERAS[0]]
        with pytest.raises(ClientError, match="fingerprint"):
            build_payload(changed, MODEL)
    with pytest.raises(ClientError, match="request_id"):
        parse_proposal(proposal_for(original), request_for(observation=observation(18)))


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((9, 7)),
        np.zeros((10, 8)),
        np.full((10, 7), 1.001),
        [[False] * 7] * 10,
        np.full((10, 7), np.nan),
    ],
)
def test_native_actions_are_finite_controller_predictions(value):
    with pytest.raises(ClientError):
        request_for(native_actions=value)


def test_request_limits_and_action_cap_cannot_be_silently_relaxed():
    request = request_for()
    request["limits"]["max_delta_xyz"] = 1
    with pytest.raises(ClientError, match="limits"):
        build_payload(request, MODEL)
    with pytest.raises(ClientError):
        request_for("critique", rollout=rollout(steps=301), observation_step=300)
    request_for("critique", rollout=rollout(steps=300), observation_step=300)


def test_rule_budget_is_five_short_observable_trigger_action_pairs():
    rules = [
        {
            "rule_id": f"r-{index}",
            "trigger": "Visible cup slip.",
            "action": "Defer while occluded.",
        }
        for index in range(6)
    ]
    assert len(request_for(rules=rules[:5])["inputs"]["rules"]) == 5
    with pytest.raises(ClientError, match="five"):
        request_for(rules=rules)
    rules[0]["trigger"] = "x" * 161
    with pytest.raises(ClientError, match="160"):
        request_for(rules=rules[:1])
    request = request_for("critique")
    with pytest.raises(ClientError, match="five"):
        parse_proposal(proposal_for(request, rules=rules), request)


def test_four_snapshot_budget_cannot_hide_unsampled_intermediate_actions():
    observed = rollout(steps=40)
    observed["snapshots"] = [
        {"label": f"snapshot-{step}", "step": step, "observation": observation(step)}
        for step in (0, 10, 20, 40)
    ]
    request = request_for("critique", rollout=observed, observation_step=40)
    assert len(request["inputs"]["rollout"]["executed_actions"]) == 40
    assert len(attached_arrays(build_payload(request, MODEL))) == 8
    observed["snapshots"].insert(
        3, {"label": "extra", "step": 30, "observation": observation(30)}
    )
    with pytest.raises(ClientError, match="one to four"):
        request_for("critique", rollout=observed, observation_step=40)


def test_exact_original_prompts_and_settings_are_exported_for_reproducibility():
    manifest = prompt_manifest()
    assert manifest["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert set(manifest["roles"]) == set(ROLES)
    for role, item in manifest["roles"].items():
        assert item["system_prompt"] == SYSTEM_PROMPTS[role]
        assert (
            item["sha256"] == hashlib.sha256(item["system_prompt"].encode()).hexdigest()
        )
        payload = build_payload(request_for(role), MODEL)
        assert payload["messages"][0]["content"] == item["system_prompt"]
        assert "json" in payload["messages"][1]["content"][0]["text"]
        assert payload["reasoning_effort"] == "medium"
        assert payload["max_completion_tokens"] == 8192
        assert payload["cache"] == {"no-cache": True}
        assert payload["response_format"] == {"type": "json_object"}


def test_duplicate_response_fields_cannot_override_a_valid_identity():
    request = request_for()
    text = json.dumps(proposal_for(request))[:-1] + ',"request_id":"0000000000000000"}'
    with pytest.raises(ClientError, match="duplicate"):
        parse_proposal(text, request)


@pytest.mark.parametrize("role", ROLES)
def test_single_provider_call_binds_role_prompt_payload_and_actual_model(
    client_factory, role
):
    request = request_for(role)
    client = client_factory(
        envelope_for(request),
        reasoning_effort="medium",
        max_completion_tokens=8192,
        timeout=170,
    )
    proposal = client.propose(request)
    assert proposal["request_fingerprint"] == request["request_fingerprint"]
    assert len(client.opener.calls) == len(client.records) == 1
    http_request, timeout = client.opener.calls[0]
    assert timeout == 170
    row = client.records[0]
    assert row == json.loads(client.response_log.read_text())
    assert row["role"] == role
    assert row["payload_sha256"] == hashlib.sha256(http_request.data).hexdigest()
    assert (
        row["system_prompt_sha256"]
        == hashlib.sha256(SYSTEM_PROMPTS[role].encode()).hexdigest()
    )
    assert row["request_fingerprint"] == request["request_fingerprint"]
    assert row["accepted"] is True
    assert row["response_id"] == proposal["response_id"]
    assert TEST_KEY not in client.response_log.read_text()
    cost = summarize_calls(client.records)
    assert cost == summarize_calls(client.response_log)
    assert (
        cost["client_attempts"]
        == cost["provider_calls"]
        == cost["accepted_proposals"]
        == 1
    )
    assert cost["tokens"]["total_tokens"]["sum"] == 150
    assert cost["tokens"]["reasoning_tokens"]["sum"] == 20
    assert cost["reasoning_tokens_are_subset_of_output_tokens"] is True
    assert cost["actual_models"] == {MODEL: 1}
    assert cost["monetary_cost"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "stale",
        "truncated",
        "refusal",
        "extra_choices",
        "unsafe_delta",
        "wrong_role",
    ],
)
def test_paid_provider_or_contract_rejections_keep_usage_and_never_retry(
    client_factory, mutation
):
    request = request_for()
    envelope = envelope_for(request)
    choice = envelope["choices"][0]
    if mutation == "model":
        envelope["model"] = "synthetic-unexpected-model"
    elif mutation == "stale":
        choice["message"]["content"] = json.dumps(
            proposal_for(request, request_id="0" * 16)
        )
    elif mutation == "truncated":
        choice["finish_reason"] = "length"
    elif mutation == "refusal":
        choice["message"]["refusal"] = "Synthetic refusal"
    elif mutation == "extra_choices":
        envelope["choices"].append(copy.deepcopy(choice))
    else:
        changes = (
            {"delta_xyz": [0, 0, 0.6]}
            if mutation == "unsafe_delta"
            else {"role": "judge"}
        )
        choice["message"]["content"] = json.dumps(proposal_for(request, **changes))
    client = client_factory(envelope)
    with pytest.raises(ClientError):
        client.propose(request)
    assert len(client.opener.calls) == len(client.records) == 1
    assert client.records[0]["error_kind"] == "proposal_rejected"
    cost = summarize_calls(client.records)
    assert cost["failed_calls"] == cost["provider_calls"] == 1
    assert cost["accepted_proposals"] == 0
    assert cost["tokens"]["total_tokens"]["sum"] == 150


def test_http_failure_timeout_and_preflight_are_distinct_and_all_costs_survive(
    client_factory, monkeypatch
):
    request = request_for()
    body = {
        "model": MODEL,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "error": {
            "code": "synthetic",
            "type": "request_error",
            "message": f"Do not retain {TEST_KEY}",
            "debug": "private-debug",
        },
    }
    failure = urllib.error.HTTPError(
        "https://synthetic.invalid",
        400,
        "bad",
        {},
        io.BytesIO(json.dumps(body).encode()),
    )
    client = client_factory(
        envelope_for(request),
        envelope_for(request, request_id="0" * 16),
        failure,
        TimeoutError("private transport details"),
    )
    client.propose(request)
    for _ in range(3):
        with pytest.raises(ClientError):
            client.propose(request)
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError, match="not set"):
        client.propose(request)
    assert len(client.records) == 5
    assert len(client.opener.calls) == 4
    assert [row.get("error_kind") for row in client.records] == [
        None,
        "proposal_rejected",
        "http_error",
        "transport_error",
        "preflight_error",
    ]
    cost = summarize_calls(client.records)
    assert cost["provider_calls"] == 4
    assert cost["client_attempts"] == 5
    assert cost["preflight_failures"] == 1
    assert cost["accepted_proposals"] == 1
    assert cost["failed_calls"] == 3
    assert cost["tokens"]["total_tokens"] == {
        "sum": 315,
        "available_calls": 3,
        "missing_calls": 1,
        "complete": False,
    }
    assert cost["usage_unavailable_calls"] == 1
    assert summarize_calls(client.records[:2])["tokens"]["total_tokens"]["sum"] == 300
    assert client.records[2]["provider_error"]["message"] == "Do not retain [REDACTED]"
    logged = client.response_log.read_text()
    assert TEST_KEY not in logged
    assert "private-debug" not in logged
    assert "private transport details" not in logged


@pytest.mark.parametrize("bad_request", [None, {}, {"role": "unknown"}])
def test_malformed_preflight_is_still_ledgered_without_any_network_or_usage(
    client_factory, bad_request
):
    client = client_factory()
    with pytest.raises(ClientError):
        client.propose(bad_request)
    assert not client.opener.calls
    assert len(client.records) == 1
    cost = summarize_calls(client.records)
    assert cost["client_attempts"] == cost["preflight_failures"] == 1
    assert cost["provider_calls"] == 0
    assert cost["tokens"]["total_tokens"]["sum"] == 0


def test_ledger_rejects_forged_usage_physical_role_and_preflight_response(
    client_factory, monkeypatch
):
    request = request_for()
    client = client_factory(envelope_for(request))
    client.propose(request)
    forged = copy.deepcopy(client.records)
    forged[0]["token_usage"]["total_tokens"] += 1
    with pytest.raises(ClientError, match="usage"):
        summarize_calls(forged)
    forged = copy.deepcopy(client.records)
    forged[0]["role"] = "unlisted"
    with pytest.raises(ClientError, match="incompatible"):
        summarize_calls(forged)
    monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY")
    with pytest.raises(ClientError):
        client.propose(request)
    forged = copy.deepcopy(client.records[-1:])
    forged[0]["http_status"] = 200
    with pytest.raises(ClientError, match="Preflight"):
        summarize_calls(forged)


def test_constructor_sampling_cannot_silently_override_explicit_settings(
    client_factory,
):
    with pytest.raises(ClientError, match="conflicts"):
        client_factory(reasoning_effort="medium", sampling={"reasoning_effort": "low"})
    request = request_for()
    client = client_factory(
        envelope_for(request),
        reasoning_effort="high",
        max_completion_tokens=2048,
        sampling={"top_p": 1},
    )
    client.propose(request)
    payload = json.loads(client.opener.calls[0][0].data)
    assert payload["reasoning_effort"] == "high"
    assert payload["max_completion_tokens"] == 2048
    assert payload["top_p"] == 1
