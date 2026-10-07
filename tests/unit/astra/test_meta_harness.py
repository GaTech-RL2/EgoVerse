"""CPU contract and delayed-call tests; these are not robot success evidence."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from astra_reversal.meta_harness.astra_worker import AstraWorker
from astra_reversal.meta_harness.harness import ContractError, Harness
from astra_reversal.meta_harness.programs import Programs, public_observation
from astra_reversal.meta_harness.relay_agent import (
    build_payload,
    parse_proposal,
    wire_request,
)
from astra_reversal.meta_harness.runtime import Runtime, Trace
from astra_reversal.meta_harness.schema import Compiler, Limits, make_cards
from astra_reversal.meta_harness.search import (
    Archive,
    SearchView,
    prompt_only_change,
    split_plan,
)

INITIAL = (
    Path(__file__).parents[3] / "astra_reversal/meta_harness/initial_harness.py"
).read_text()


class Bank:
    bank_id = "a" * 64
    sources = {
        str(i): {"source_id": str(i), "prompt": prompt, "frame_count": 90}
        for i, prompt in enumerate(
            ("put the bowl in the basket", "put the cheese on the plate")
        )
    }

    def catalog(self):
        return list(self.sources.values())


def live():
    return {
        "observation/image": np.zeros((16, 16, 3), np.uint8),
        "observation/wrist_image": np.full((16, 16, 3), 25, np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "hidden_object_pose": [100, 200, 300],
        "success": True,
    }


def compiler():
    bank = Bank()
    return Compiler(bank, make_cards(bank))


def set_call(c, obs="obs0", stage="stage_0", **changes):
    return {
        "tool": "set_policy_program",
        "arguments": {
            "observation_id": obs,
            "expected_stage_id": stage,
            "skill_id": next(iter(c.cards)),
            "language": None,
            "vision": {"operator": "vei", "alpha": 0.25},
            "max_actions": 25,
            **changes,
        },
    }


def binding(c, **changes):
    return {
        "observation_id": "obs0",
        "action": 0,
        "captured_at": 100.0,
        "stage_id": "stage_0",
        "retrieved_ids": list(c.cards),
        **changes,
    }


def test_keep_does_not_restart_cursor_or_extend_expiry():
    c = compiler()
    p = Programs(c, Limits())
    p.boundary(live(), 0)
    receipt = p.apply(set_call(c), binding(c), live(), 101)
    assert receipt["validation_error"] is None
    assert p.choice["frame"] == 0
    original = p.program_id
    p.boundary(live(), 5)
    req = binding(c, stage_id=p.stage_id, action=5)
    p.apply(
        {
            "tool": "keep_policy_program",
            "arguments": {"observation_id": "obs0", "program_id": original},
        },
        req,
        live(),
        102,
    )
    assert p.choice["frame"] == 5 and p.expiry == 25 and p.started == 0
    for step in (10, 15, 20, 25):
        p.boundary(live(), step)
    assert p.choice is None and p.program_id is None
    assert p.termination == "expiry"


@pytest.mark.parametrize(
    "changes",
    [
        {"max_actions": True},
        {"max_actions": 12},
        {"max_actions": 105},
        {"skill_id": "invented"},
        {"vision": {"operator": "vli", "alpha": 0.25}},
        {"vision": {"operator": "vei", "alpha": float("nan")}},
        {"vision": {"operator": "vei", "alpha": 0.3}},
        {
            "language": {
                "operator": "tei",
                "source_a_id": "0",
                "source_b_id": "1",
                "alpha": 0.5,
            }
        },
        {
            "language": {
                "operator": "tli",
                "source_a_id": "0",
                "source_b_id": "1",
                "alpha": 0.5,
            },
            "vision": None,
        },
    ],
)
def test_invalid_or_unsupported_tools_are_rejected(changes):
    c = compiler()
    with pytest.raises(ValueError):
        c.compile(set_call(c, **changes), list(c.cards))


def test_tei_zero_preserves_source_a_semantics():
    c = compiler()
    language = {"operator": "tei", "source_a_id": "0", "source_b_id": "1", "alpha": 0}
    program = c.compile(set_call(c, language=language, vision=None), list(c.cards))
    assert program["stages"][0]["language"] == language
    assert program["stages"][0]["state_mode"] == "live"


@pytest.mark.parametrize(
    "req, now, expected",
    [
        ({"captured_at": 80}, 101, "stale_wall_time"),
        ({"stage_id": "stage_40"}, 101, "stale_stage"),
        ({"action": -25}, 101, "stale_action_age"),
    ],
)
def test_stale_calls_cannot_replace_active_program(req, now, expected):
    c = compiler()
    p = Programs(c, Limits())
    p.boundary(live(), 0)
    assert (
        p.apply(set_call(c), binding(c, **req), live(), now)["validation_error"]
        == expected
    )
    assert p.program_id is None


def test_public_view_excludes_privileged_fields_and_copies_arrays():
    source = live()
    raw, public = public_observation(
        source, step=0, now=100, stage_id="stage_0", episode_id="e"
    )
    assert set(raw) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
    }
    assert "success" not in public and "hidden_object_pose" not in public
    raw["observation/image"][:] = 99
    assert source["observation/image"].max() == 0


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef harness(observation, history, cards, memory):\n return {}",
        "def harness(observation, history, cards, memory):\n return open('/tmp/evaluator')",
        "def harness(observation, history, cards, memory):\n return observation.__class__",
        "def harness(observation, history, cards, memory):\n while True:\n  memory = {}",
        "def harness(observation, history, cards, memory):\n return eval('1')",
    ],
)
def test_candidate_access_contract_is_enforced(source):
    with pytest.raises(ContractError):
        Harness(source)


def test_initial_harness_and_prompt_only_control():
    c = compiler()
    result = Harness(INITIAL).run(
        {"action": 0, "original_goal": "put the bowl in the basket"},
        [],
        list(c.cards.values()),
        {},
    )
    assert result["request"] and len(result["card_ids"]) == 6
    assert result["memory"]["last_request"] == 0
    assert prompt_only_change(
        INITIAL,
        INITIAL.replace(
            "Preserve useful native behavior.", "Retain uncertainty about grasps."
        ),
    )
    assert not prompt_only_change(INITIAL, INITIAL.replace(">= 40", ">= 20"))


class FakeWorker:
    def __init__(self, *, wait=None, fail=False):
        self.records, self.requests = [], []
        self.wait, self.fail = wait, fail
        self.started = threading.Event()

    def __call__(self, request):
        self.requests.append(request)
        self.started.set()
        if self.wait is not None:
            assert self.wait.wait(5)
        call = {
            "tool": "clear_policy_program",
            "arguments": {"observation_id": request["binding"]["observation_id"]},
        }
        record = {
            "latency_seconds": 0.01,
            "error": "fixture" if self.fail else None,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        self.records.append(record)
        return {"call": None if self.fail else call, "record": record}


def runtime(tmp_path, worker, *, mode="synchronous_diagnostic", source=INITIAL):
    return Runtime(
        compiler=compiler(),
        limits=Limits(),
        harness=Harness(source),
        worker=worker,
        original_goal="put the bowl in the basket",
        episode_id="test",
        trace=Trace(tmp_path / "trace"),
        mode=mode,
        clock=lambda: 100.0,
    )


def test_failed_requests_consume_call_budget_and_never_leak_success(tmp_path):
    worker = FakeWorker(fail=True)
    r = runtime(tmp_path, worker, source=INITIAL.replace(">= 40", ">= 0"))
    for step in range(0, 300, 5):
        r.boundary(live(), step)
    result = r.close()
    assert result["runtime_requests"] == 8 and len(worker.requests) == 8
    assert len(result["tool_events"]) == 8
    assert not result["contract_errors"]
    for request in worker.requests:
        assert "success" not in json.dumps(request["context"])
        assert "hidden_object_pose" not in json.dumps(request["context"])


def test_environment_can_advance_while_one_request_is_pending(tmp_path):
    gate = threading.Event()
    worker = FakeWorker(wait=gate)
    r = runtime(tmp_path, worker, mode="async")
    try:
        r.boundary(live(), 0)
        assert worker.started.wait(1)
        for step in range(5, 35, 5):
            r.boundary(live(), step)
        assert len(worker.requests) == 1 and r.calls == 1
        future = r.pending[0]
        gate.set()
        future.result(timeout=1)
        r.boundary(live(), 35)
        assert r.receipts[0]["validation_error"] == "stale_action_age"
    finally:
        gate.set()
        r.close()


def request_fixture():
    c = compiler()
    raw, public = public_observation(
        live(), step=0, now=100, stage_id="stage_0", episode_id="e"
    )
    return {
        "request_id": "request0",
        "binding": {**public, "retrieved_ids": list(c.cards)},
        "context": {"current": public, "cards": list(c.cards.values())},
        "frames": [
            {
                "observation_id": public["observation_id"],
                "action": 0,
                "images": {
                    k: raw[k] for k in ("observation/image", "observation/wrist_image")
                },
            }
        ],
    }


def test_responses_counts_images_and_tools_then_generates_once():
    request = request_fixture()
    posts = []

    def post(path, body, timeout):
        posts.append((path, body))
        if path.endswith("input_tokens"):
            return {"input_tokens": 800}
        return {
            "model": "gpt-6-astra",
            "status": "completed",
            "usage": {"input_tokens": 800, "output_tokens": 120},
            "output": [
                {
                    "type": "function_call",
                    "name": "clear_policy_program",
                    "arguments": json.dumps(
                        {"observation_id": request["binding"]["observation_id"]}
                    ),
                }
            ],
        }

    worker = AstraWorker(post=post)
    result = worker(request)
    assert (
        result["record"]["accepted"]
        and result["call"]["tool"] == "clear_policy_program"
    )
    assert [p[0] for p in posts] == ["responses/input_tokens", "responses"]
    assert posts[0][1]["tools"] == posts[1][1]["tools"]
    assert posts[1][1]["max_output_tokens"] == 256


def test_over_budget_input_does_not_generate_or_retry():
    posts = []

    def post(path, body, timeout):
        posts.append(path)
        return {"input_tokens": 8193}

    result = AstraWorker(post=post)(request_fixture())
    assert result["call"] is None and posts == ["responses/input_tokens"]


def test_relay_profile_has_the_same_typed_call_and_live_images():
    request = wire_request(request_fixture())
    payload = build_payload(request, "gpt-6-astra")
    assert (
        len([x for x in payload["messages"][1]["content"] if x["type"] == "image_url"])
        == 2
    )
    parsed = parse_proposal(
        {
            "tool": "clear_policy_program",
            "arguments": {"observation_id": request["binding"]["observation_id"]},
        },
        request,
    )
    assert parsed["decision_id"] == request["request_id"]
    from astra_reversal.codex_relay import _proposal
    from astra_reversal.meta_harness import relay_agent

    # The executor returns a bound proposal; the relay must strip that binding
    # before applying the exact unbound policy-tool schema a second time.
    assert _proposal(parsed, request, relay_agent, "meta_harness_runtime") == parsed


def test_nested_growth_is_bounded_before_candidate_output():
    source = """def harness(observation, history, cards, memory):
    value = [0]
    for card in cards:
        value = append(value, value)
    return value
"""
    with pytest.raises(ContractError, match="128KB"):
        Harness(source).run({}, [], [{}] * 100, {})


def test_archive_preserves_rejected_candidates_and_blocks_final_reads(tmp_path):
    archive = Archive(tmp_path / "run", {"search_episode_ids": []})
    row = archive.register(
        "import os", "A forbidden change", kind="meta_harness", baseline=INITIAL
    )
    assert row["status"] == "rejected"
    assert (
        archive.root / "candidates" / row["candidate_id"] / "harness.py"
    ).read_text() == "import os"
    final = archive.root / "final_evaluation"
    final.mkdir()
    (final / "outcomes.json").write_text('{"secret": true}')
    view = SearchView(archive)
    with pytest.raises(ValueError):
        view.read("../final_evaluation/outcomes.json")
    link = archive.root / "candidates" / "leak.json"
    link.symlink_to(final / "outcomes.json")
    assert "leak.json" not in view.files()
    with pytest.raises(ValueError):
        view.read("leak.json")


def test_split_is_disjoint_and_does_not_claim_unseen_tasks():
    plan = split_plan()
    splits = [
        set(map(tuple, plan[key]))
        for key in ("development_tasks", "search_tasks", "final_tasks")
    ]
    assert [len(s) for s in splits] == [12, 4, 4]
    assert all(not a & b for i, a in enumerate(splits) for b in splits[i + 1 :])
    assert not set(plan["search_resets"]) & set(plan["final_resets"])
    assert "NOT unseen" in plan["generalization_claim"]


@pytest.mark.parametrize("ignore_tei", [False, True])
def test_weighted_gate_compares_tei_endpoints_with_the_target_layout(
    monkeypatch, ignore_tei
):
    from types import SimpleNamespace

    from astra_reversal.meta_harness import preflight

    bank = Bank()
    bank.observation = lambda *_: live()
    monkeypatch.setattr(
        preflight,
        "InputSkillConditioner",
        lambda *_: SimpleNamespace(_donor_pair=lambda *_: {}),
    )
    monkeypatch.setattr(
        preflight,
        "weighted_vision_probe",
        lambda *_: {"checks": {"native": True}, "velocity_evaluations": 90},
    )

    class Policy:
        def noise(self, _):
            return None

        def prepare(self, observation, identity, prompt):
            return prompt, prompt

        def prepare_interpolated(
            self, observation, identity, target, *, source_prompts, alpha, operator
        ):
            selected = (
                source_prompts[0] if operator == "tei" and not ignore_tei else target
            )
            return (target, selected), {
                "has_effect": selected != target,
                "text_mask_fixed": True,
                "protected_embedding_slots_unchanged": True,
                "vision_prefix_unchanged": True,
            }

        def sample(self, condition, noise, **kwargs):
            # Actions depend on both the protected target layout and selected
            # source. Equal A/B sources alone do not erase the target layout.
            target, selected = condition
            return SimpleNamespace(
                value=np.array([len(target), sum(map(ord, selected))], dtype=float),
                velocity_evaluations=10,
            )

    report = preflight.policy_gate(Policy(), bank)
    assert report["checks"]["tei_zero_selects_source_a_matching_masks"]
    assert report["tei_zero_endpoint"]["native_a_max_abs"] == 0
    assert report["velocity_evaluations"] == 140
    assert report["status"] == ("failed" if ignore_tei else "passed")
    assert report["checks"]["tei_zero_original_target_modified"] is not ignore_tei
