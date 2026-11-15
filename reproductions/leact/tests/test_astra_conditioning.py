from copy import deepcopy
import json

from reproductions.leact.astra_expert.conditioning import ARMS, admitted_pool, generate_batch, prepare_matched
from reproductions.leact.core import policy_answer
from reproductions.leact.tests.test_pipeline import RecordingBackend, state, generation


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(100))


def inputs():
    return [{**state(), "teacher_decision_summary": "EXPLANATION_MARKER A weak hand faces pressure."}], {
        "rules": "rules", "sha256": "astra", "cfr_evaluation_oracle_sha256": "cfr"}


def test_assisted_only_improvement_never_admits_a_trace():
    states, meta = inputs()
    rows = generate_batch(states, "rules", RecordingBackend(), arm=ARMS[1], n=2, m=2)
    for c in rows[0]["candidates"]:
        c["forward"] = generation(policy_answer({"fold": .1, "call": .9}))
    pool, audit = admitted_pool(states, meta, rows, Tokenizer(), arm=ARMS[1])
    assert not pool
    assert all(c["delta_0"] < 0 and c["delta_r"] > 0 for c in audit[0]["candidates"])


def test_shared_targets_no_hint_in_standalone_and_matched_updates(tmp_path):
    states, meta = inputs()
    pools, audits = {}, {}
    for arm in ARMS:
        rows = generate_batch(states, "rules", RecordingBackend(), arm=arm, n=2, m=2)
        pools[arm], audits[arm] = admitted_pool(states, meta, rows, Tokenizer(), arm=arm)
    directories = {arm: tmp_path / arm for arm in ARMS}
    manifests = prepare_matched(pools, audits, meta, directories, round_index=1)
    assert all(m["sft_examples"] == 4 for m in manifests.values())
    assert manifests[ARMS[1]]["task_counts"] == {"backward": 2, "assisted_forward": 1, "standalone_forward": 1}
    assert manifests[ARMS[0]]["task_counts"] == {"backward": 2, "assisted_forward": 0, "standalone_forward": 2}
    for line in (directories[ARMS[1]] / "all-views.jsonl").read_text().splitlines():
        views = json.loads(line)["views"]
        assert "EXPLANATION_MARKER" not in views["standalone_forward"]["conversations"][0]["value"]
        assert "EXPLANATION_MARKER" in views["assisted_forward"]["conversations"][0]["value"]
        assert views["standalone_forward"]["conversations"][1] == views["assisted_forward"]["conversations"][1]
        assert "<think>" in views["standalone_forward"]["conversations"][1]["value"]


def test_all_views_length_checked_and_invalid_assisted_diagnostic_is_not_a_gate():
    states, meta = inputs()
    rows = generate_batch(states, "rules", RecordingBackend(), arm=ARMS[1], n=2, m=2)
    rows[0]["assisted_baselines"][0]["finish_reason"] = "length"
    pool, audit = admitted_pool(states, meta, rows, Tokenizer(), arm=ARMS[1])
    assert len(pool) == 2
    assert all(c["delta_r"] is None for c in audit[0]["candidates"])
    class ShortContext:
        def apply_chat_template(self, messages, **kwargs):
            return list(range(3000 if "EXPLANATION_MARKER" in messages[0]["content"] else 100))
    pool, audit = admitted_pool(states, meta, rows, ShortContext(), arm=ARMS[1])
    assert not pool
    assert all(not c["admitted_to_sft"] for c in audit[0]["selected"])


def test_matched_subsampling_audits_unused_valid_traces(tmp_path):
    states, meta = inputs()
    pools, audits = {}, {}
    for arm in ARMS:
        rows = generate_batch(states, "rules", RecordingBackend(), arm=arm, n=2, m=2)
        pools[arm], audits[arm] = admitted_pool(states, meta, rows, Tokenizer(), arm=arm)
    extra = deepcopy(pools[ARMS[0]][0]); extra["state_id"] = "extra"
    pools[ARMS[0]].append(extra)
    result = prepare_matched(pools, audits, meta, {a: tmp_path / a for a in ARMS}, round_index=2)
    assert result[ARMS[0]]["eligible_traces_before_compute_matching"] == 3
    assert all(r["selected_traces"] == 2 for r in result.values())
