from copy import deepcopy
import json

import pytest

from reproductions.leact.astra_expert.client import (
    MODEL, decode, payload, public_response, strict_json, validate_policy, validate_summary,
)
from reproductions.leact.astra_expert.data import leduc_gate, pilot_states, read_call, export_teacher
from reproductions.leact.data import fingerprint


def state():
    return {"state_id": "id", "info_state": "[Round 0][Private: 2c][Public: ]", "player": 0,
            "legal_actions": ["fold", "call"], "expert_policy": {"fold": 0.918273, "call": 0.081727},
            "opponent_private_card": "NEVER_SEND_THIS", "action_ids": {"fold": 0, "call": 1}}


def envelope(result=None):
    return {"model": MODEL, "usage": {"completion_tokens": 100}, "choices": [{
        "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(result or {
            "action": "fold", "policy": {"fold": 0.8, "call": 0.2}}),
            "reasoning_content": "PRIVATE_PROVIDER_FIELD"}}]}


def test_request_excludes_cfr_and_hidden_state_and_freezes_targets():
    s = state()
    request = payload(s, "rules")
    serialized = json.dumps(request)
    assert "NEVER_SEND_THIS" not in serialized and "0.918273" not in serialized
    assert "json" in request["messages"][-1]["content"].lower()
    target = {"action": "call", "policy": {"fold": 0.1, "call": 0.9}}
    summary = payload(s, "rules", committed=target)
    assert "previously_committed_decision" in summary["messages"][-1]["content"]
    assert "0.9" in summary["messages"][-1]["content"]
    assert "PRIVATE_PROVIDER_FIELD" not in json.dumps(public_response(envelope()))
    assert public_response(envelope())["choices"][0]["message"]["content"] == envelope()["choices"][0]["message"]["content"]


@pytest.mark.parametrize("policy", [
    {"fold": True, "call": 0}, {"fold": "0.8", "call": 0.2},
    {"fold": 0.8, "call": 0.3}, {"fold": -0.1, "call": 1.1},
    {"fold": float("nan"), "call": 1}, {"fold": 1},
    {"fold": 1, "call": 0, "raise": 0},
])
def test_invalid_teacher_policies_are_rejected_without_repair(policy):
    with pytest.raises(ValueError):
        validate_policy({"action": "fold", "policy": policy}, ["fold", "call"])


@pytest.mark.parametrize("text", ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_strict_json(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_truncation_model_substitution_and_nonmodal_actions_are_rejected():
    for mutate in [lambda r: r.update(model="other-model"),
                   lambda r: r["choices"][0].update(finish_reason="length"),
                   lambda r: r["choices"][0]["message"].update(refusal="refused")]:
        r = envelope(); mutate(r)
        with pytest.raises(ValueError):
            decode(r, state())
    with pytest.raises(ValueError):
        validate_policy({"action": "call", "policy": {"fold": 0.8, "call": 0.2}}, ["fold", "call"])


def test_summary_has_no_action_override_or_unbounded_reasoning():
    for value in [{"decision_summary": "word " * 121}, {"decision_summary": "<think>anything</think>"},
                  {"decision_summary": "short", "action": "raise"}]:
        with pytest.raises(ValueError):
            validate_summary(value)
    assert validate_summary({"decision_summary": "Calling is uncertain because a stronger hand is possible."})


def complete_report():
    def epoch(name, mean):
        return {"checkpoint": name, "inference_seeds": [0, 1, 2],
                "single": {"exploitability": {"mean": mean, "std": 0.01}},
                "oracle_bon": {"exploitability": {"mean": mean / 2, "std": 0.01}}}
    return {"scope": "Leduc 6r2s independent reproduction",
            "oracle": {"num_infosets": 4032, "sha256": "cfr", "exploitability": 1e-5},
            "protocol": {"rounds": 3}, "coldstart_thinking": epoch("anchor", 1),
            "methods": {m: [{"round": i, "all_epochs": [epoch(f"{m}-{i}", 0.5)]}
                            for i in [1, 2, 3]] for m in ["leact", "exit"]}}


def test_full_experiment_requires_completed_improving_leduc():
    report = complete_report()
    assert leduc_gate(report, "cfr")["passed"]
    bad = deepcopy(report); bad["methods"]["exit"].pop()
    with pytest.raises(ValueError, match="not complete"):
        leduc_gate(bad, "cfr")
    bad = deepcopy(report); bad["coldstart_thinking"]["single"]["exploitability"]["mean"] = 0.1
    with pytest.raises(ValueError, match="not improved"):
        leduc_gate(bad, "cfr")
    with pytest.raises(ValueError, match="matching"):
        leduc_gate(report, "wrong-cfr")


def test_provenance_rejects_changed_request_and_changed_summary_target(tmp_path):
    from reproductions.leact.astra_expert.client import ENDPOINT, PROMPT_VERSION
    s, metadata = state(), {"sha256": "cfr", "rules": "rules"}
    request = payload(s, "rules")
    row = {"state_id": "id", "oracle_sha256": "cfr", "prompt_version": PROMPT_VERSION,
           "endpoint": ENDPOINT, "stage": "actions", "request_sha256": fingerprint(request),
           "committed_decision_sha256": None, "accepted": True, "request": request,
           "response": public_response(envelope()), "result": decode(envelope(), s)}
    path = tmp_path / "record.json"; path.write_text(json.dumps(row))
    assert read_call(path, s, metadata)["result"] == row["result"]
    row["request"]["messages"][0]["content"] = "changed"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="provenance"):
        read_call(path, s, metadata)
    with pytest.raises(ValueError):
        read_call(path, s, metadata, committed={"action": "call", "policy": {"fold": 0.1, "call": 0.9}})


def test_preflight_covers_rounds_players_but_cannot_be_exported_as_full_tree(tmp_path):
    states = [{**state(), "state_id": f"{r}{p}", "player": p, "info_state": f"[Round {r}]"}
              for r in [0, 1] for p in [0, 1]]
    assert len(pilot_states(states)) == 4
    with pytest.raises(ValueError, match="complete 4032"):
        export_teacher(states, {"num_infosets": 4032}, tmp_path, tmp_path / "teacher.jsonl")
    assert not (tmp_path / "teacher.jsonl").exists()


def test_expert_summary_only_enters_backward_hints_and_paired_length_audit(tmp_path):
    from reproductions.leact.astra_expert import leact
    from reproductions.leact.core import policy_answer
    from reproductions.leact.tests.test_pipeline import RecordingBackend
    s = {**state(), "expert_policy": {"fold": 0.8, "call": 0.2},
         "teacher_decision_summary": "EXPLANATION_MARKER A weak hand faces pressure."}
    metadata = {"rules": "rules", "sha256": "astra", "cfr_evaluation_oracle_sha256": "cfr"}
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(3000 if "EXPLANATION_MARKER" in messages[0]["content"] else 100))
    rows = {}
    for arm in leact.ARMS:
        backend = RecordingBackend()
        rows[arm] = leact.generate_batch([s], "rules", backend, arm=arm, n=2, m=2)
        assert ("EXPLANATION_MARKER" in backend.requests[0].prompt) == (arm == "astra_actions_summary")
        assert all("EXPLANATION_MARKER" not in r.prompt for r in backend.requests[1:])
    # The extra hint must participate in the context limit. An overlength
    # backward example excludes its forward partner too, without partial SFT.
    with pytest.raises(ValueError, match="No admitted"):
        leact.prepare_data([s], metadata, tmp_path / "too-long", arm="astra_actions_summary",
                           raw=rows["astra_actions_summary"], tokenizer=Tokenizer())
    assert not (tmp_path / "too-long").exists()
    for arm in leact.ARMS:
        result = leact.prepare_data([s], metadata, tmp_path / arm, arm=arm,
                                   raw=rows[arm], tokenizer=Tokenizer(), cutoff=4096)
        assert result["sft_examples"] == 4 and result["selected_traces"] == 2
        pairs = json.loads((tmp_path / arm / "data.json").read_text())
        assert pairs[0]["conversations"][1]["value"].endswith(policy_answer(s["expert_policy"]))
        assert "EXPLANATION_MARKER" not in pairs[0]["conversations"][0]["value"]
        assert "0.918273" not in json.dumps(pairs)
    with pytest.raises(ValueError, match="different teacher"):
        leact.prepare_data([s], metadata, tmp_path / "wrong-condition", arm="astra_actions",
                           raw=rows["astra_actions_summary"], tokenizer=Tokenizer())
