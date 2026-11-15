import json

import pytest

from reproductions.leact.astra_expert.client import MODEL
from reproductions.leact.astra_expert.data import solver_gate
from reproductions.leact.coldstart.collect_full import nested_states, collect_stage
from reproductions.leact.coldstart.data import request_for, validate_candidate
from reproductions.leact.coldstart.run import export_corpus
from reproductions.leact.core import policy_answer


def state(i=0):
    return {"state_id": str(i), "info_state": f"[Round {i % 2}][Private: 2c][Public: 3d]",
            "player": i % 2, "legal_actions": ["fold", "call"],
            "expert_policy": {"fold": 0.918273, "call": 0.081727},
            "opponent_card": "PRIVATE_CARD_MUST_NOT_ENTER_REQUEST"}


def envelope(reasoning="With 2c against public 3d, the visible hand is unpaired."):
    return {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps({"reasoning": reasoning,
        "action": "call", "policy": {"fold": 0.1, "call": 0.9}}),
        "reasoning_content": "PRIVATE_PROVIDER_REASONING"}}]}


def test_frontier_generation_is_solver_blind_and_validates_visible_cards():
    request = json.dumps(request_for(state(), "rules"))
    assert "0.918273" not in request and "PRIVATE_CARD_MUST_NOT_ENTER_REQUEST" not in request
    assert validate_candidate(envelope(), state())["action"] == "call"
    for bad in ["No named visible cards.", "With 2c but omitted the public card.",
                "<think>2c and 3d</think>", "2c 3d " + "word " * 201]:
        with pytest.raises(ValueError):
            validate_candidate(envelope(bad), state())
    assert "PRIVATE_PROVIDER_REASONING" not in json.dumps(validate_candidate(envelope(), state()))


def test_nested_state_order_is_stable_and_independent_of_solver_labels():
    states = [state(i) for i in range(12)]
    ordered = nested_states(states)
    assert {s["state_id"] for s in ordered} == {s["state_id"] for s in states}
    assert [s["state_id"] for s in ordered] == [s["state_id"] for s in nested_states(list(reversed(states)))]
    changed = [{**s, "expert_policy": {"fold": 0, "call": 1}} for s in states]
    assert [s["state_id"] for s in ordered] == [s["state_id"] for s in nested_states(changed)]
    assert len({s["player"] for s in ordered[:2]}) == 2


def test_astra_solver_gate_requires_full_coverage_correct_labels_and_at_most_point_one():
    metadata = {"num_infosets": 4032, "cfr_evaluation_oracle_sha256": "cfr",
                "quality": {"exploitability": 0.10}, "action_labels_sha256": "frozen-labels"}
    gate = {"passed": True, "maximum": 0.10, "exploitability": 0.10,
            "infosets": 4032, "oracle_sha256": "cfr", "action_labels_sha256": "frozen-labels"}
    assert solver_gate(metadata, gate, "cfr")["passed"]
    for change in [{"infosets": 128}, {"maximum": 1.82}, {"action_labels_sha256": "other"},
                   {"passed": False}, {"oracle_sha256": "other"}]:
        with pytest.raises(ValueError):
            solver_gate(metadata, {**gate, **change}, "cfr")
    for score in [0.100001, float("nan"), -0.01]:
        with pytest.raises(ValueError):
            solver_gate({**metadata, "quality": {"exploitability": score}}, {**gate, "exploitability": score}, "cfr")


def test_coldstart_keeps_poor_generated_policy_but_replaces_target_and_preserves_pairs(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(3000 if "OVERLONG" in messages[-1]["content"] else 100))
    s = state()
    pool = {s["state_id"]: [
        {"reasoning": "truncated", "finish_reason": "length", "fallback": None},
        {"reasoning": "OVERLONG", "finish_reason": "stop", "fallback": None},
        {"reasoning": "first valid candidate", "finish_reason": "stop", "fallback": None,
         "policy": {"fold": 0.0, "call": 1.0}},
        {"reasoning": "second valid candidate", "finish_reason": "stop", "fallback": None},
    ]}
    meta = {"rules": "rules", "sha256": "cfr"}
    a = export_corpus([s], meta, pool, tmp_path / "one", Tokenizer(), retained_per_state=1, source="self")
    b = export_corpus([s], meta, pool, tmp_path / "two", Tokenizer(), retained_per_state=2, source="self")
    assert a["sft_examples"] == 2 and b["sft_examples"] == 4
    one = json.loads((tmp_path / "one/data.json").read_text())
    two = json.loads((tmp_path / "two/data.json").read_text())
    assert one == two[:2]
    assert one[0]["conversations"][-1]["value"].endswith(policy_answer(s["expert_policy"]))
    assert "0.918273" not in one[1]["conversations"][0]["value"]
    assert "first valid candidate" in one[0]["conversations"][-1]["value"]
    assert export_corpus([s], meta, pool, tmp_path / "one", Tokenizer(), retained_per_state=1, source="self") == a
    with pytest.raises(ValueError, match="differs"):
        export_corpus([s], meta, pool, tmp_path / "one", Tokenizer(), retained_per_state=2, source="self")


def test_budget_recovery_preserves_attempts_and_accepted_responses(tmp_path, monkeypatch):
    from reproductions.leact.coldstart import collect_full
    s, meta = state(), {"rules": "rules", "sha256": "oracle"}
    attempts = tmp_path / "attempts/actions" / s["state_id"]
    attempts.mkdir(parents=True)
    denial = {"accepted": False, "http_status": 422,
              "provider_error": json.dumps({"error": {"type": "budget_exceeded"}})}
    for i in range(1, 6):
        (attempts / f"attempt-{i:02d}.json").write_text(json.dumps(denial))
    originals = {p: p.read_bytes() for p in attempts.iterdir()}
    invoked = []

    def call(s, rules, path, **kwargs):
        if path.exists():
            return json.loads(path.read_text())
        invoked.append(path.name)
        value = {"accepted": True, "response": {"usage": {"total_tokens": 10}}}
        path.write_text(json.dumps(value))
        return value

    monkeypatch.setattr(collect_full, "call", call)
    collect_stage("actions", [s], meta, tmp_path, workers=1, recover_budget=True)
    assert invoked == ["attempt-06.json"]
    assert all(p.read_bytes() == data for p, data in originals.items())
    accepted = (tmp_path / "actions/0.json").read_bytes()
    collect_stage("actions", [s], meta, tmp_path, workers=1, recover_budget=True)
    assert invoked == ["attempt-06.json"]
    assert (tmp_path / "actions/0.json").read_bytes() == accepted


def test_budget_recovery_does_not_reset_rejected_model_output_budget(tmp_path, monkeypatch):
    from reproductions.leact.coldstart import collect_full
    s, meta = state(), {"rules": "rules", "sha256": "oracle"}
    attempts = tmp_path / "attempts/actions" / s["state_id"]
    attempts.mkdir(parents=True)
    for i in range(1, 6):
        (attempts / f"attempt-{i:02d}.json").write_text(json.dumps({"accepted": False, "http_status": 200}))
    monkeypatch.setattr(collect_full, "call", lambda *a, **k: pytest.fail("Must not make another completion"))
    with pytest.raises(ValueError, match="attempt budget exhausted"):
        collect_stage("actions", [s], meta, tmp_path, workers=1, recover_budget=True)


def test_fresh_budget_denial_stops_immediately(tmp_path, monkeypatch):
    from reproductions.leact.coldstart import collect_full
    invoked = []

    def call(s, rules, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        invoked.append(path)
        path.write_text(json.dumps({"accepted": False, "http_status": 422,
            "provider_error": json.dumps({"error": {"type": "budget_exceeded"}})}))
        raise ValueError("Budget exceeded")

    monkeypatch.setattr(collect_full, "call", call)
    with pytest.raises(ValueError, match="Budget exceeded"):
        collect_stage("actions", [state()], {"rules": "rules", "sha256": "oracle"}, tmp_path, workers=1)
    assert len(invoked) == 1
