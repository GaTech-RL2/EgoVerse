import json

import pytest

from reproductions.leact.core import policy_answer
from reproductions.leact.data import load_oracle
from reproductions.leact.evaluate import evaluate_records
from reproductions.leact.oracle import build_oracle
from reproductions.leact.pipeline import generate_batch, generate_shard, load_shards, score_record
from reproductions.leact.training import prepare_data, supervised_pairs


def state():
    return {"state_id": "example", "info_state": "private=J, public=Q", "player": 0,
            "legal_actions": ["fold", "call"], "expert_policy": {"fold": 0.8, "call": 0.2},
            "action_ids": {"fold": 0, "call": 1}}


def generation(text):
    return {"text": text, "finish_reason": "stop", "generated_tokens": 30}


class RecordingBackend:
    def __init__(self):
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        outputs = []
        for request in requests:
            if "qualitative" in request.prompt:
                text = "<think>A weak hand faces pressure.</think>\nDONE"
            else:
                p = {"fold": 0.8, "call": 0.2} if request.reasoning else {"fold": 0.5, "call": 0.5}
                text = "<think>A weak hand faces pressure.</think>\n" + policy_answer(p)
            outputs.append([generation(text) for _ in range(request.n)])
        return outputs


def test_backward_hints_do_not_leak_to_forward_or_baseline():
    backend = RecordingBackend()
    rows = generate_batch([state()], "rules", backend, method="leact", n=3, m=2)
    assert len(backend.requests) == 5
    assert "qualitative" in backend.requests[0].prompt
    for request in backend.requests[1:]:
        assert "qualitative" not in request.prompt
        assert "usually fold" not in request.prompt
        assert "0.8" not in request.prompt
    assert backend.requests[1].reasoning is None
    assert all(r.reasoning == "A weak hand faces pressure." for r in backend.requests[2:])
    assert backend.requests[1].n == 2
    scored = score_record(state(), rows[0], method="leact", k=2)
    assert len(scored["selected"]) == 2
    assert all(c["score"] > 0 for c in scored["selected"])


def test_epf_and_exit_targets_are_different():
    chosen = {"reasoning": "The bet changes the odds.", "predicted_policy": {"fold": 0.6, "call": 0.4}}
    leact = supervised_pairs(state(), chosen, "rules", method="leact")
    exit_pairs = supervised_pairs(state(), chosen, "rules", method="exit")
    assert len(leact) == 2 and len(exit_pairs) == 1
    assert policy_answer(state()["expert_policy"]) in leact[0]["conversations"][1]["value"]
    assert policy_answer(chosen["predicted_policy"]) in exit_pairs[0]["conversations"][1]["value"]
    assert "Policy:" not in leact[1]["conversations"][1]["value"]


def test_prompt_ablation_keeps_expert_isolated_and_matches_training_prompts(tmp_path):
    backend = RecordingBackend()
    raw = generate_batch([state()], "rules", backend, method="leact", n=2, m=2,
                         prompt_variant="concise_v1")
    for request in backend.requests:
        assert "at most 200 words" in request.prompt
        assert "0.8" not in request.prompt
    assert "qualitative" in backend.requests[0].prompt
    assert all("qualitative" not in r.prompt for r in backend.requests[1:])
    chosen = score_record(state(), raw[0], method="leact")["selected"][0]
    pairs = supervised_pairs(state(), chosen, "rules", prompt_variant="concise_v1")
    assert pairs[0]["conversations"][0]["value"] == raw[0]["prompts"]["forward"]
    assert pairs[1]["conversations"][0]["value"] == raw[0]["prompts"]["backward"]
    metadata = {"rules": "rules", "sha256": "sha"}
    with pytest.raises(ValueError, match="prompt variant differs"):
        prepare_data([state()], metadata, tmp_path/"wrong", method="leact", raw=raw)
    assert not (tmp_path/"wrong").exists()
    result = prepare_data([state()], metadata, tmp_path/"correct", method="leact", raw=raw,
                          prompt_variant="concise_v1")
    assert result["prompt_variant"] == "concise_v1" and result["sft_examples"] == 4


def test_generation_resume_and_provenance_rejection(tmp_path):
    backend = RecordingBackend()
    path = tmp_path / "shard.jsonl"
    metadata = {"rules": "rules", "sha256": "oracle-checksum"}
    kwargs = dict(checkpoint="pinned-model", method="leact", n=2, m=2)
    generate_shard([state()], metadata, backend, path, **kwargs)
    before = len(backend.requests)
    generate_shard([state()], metadata, backend, path, **kwargs)
    assert len(backend.requests) == before
    assert len(load_shards([path], metadata, method="leact")) == 1
    with pytest.raises(ValueError, match="Resume configuration"):
        generate_shard([state()], metadata, backend, path, **(kwargs | {"n": 3}))
    with pytest.raises(ValueError, match="Resume configuration"):
        generate_shard([state()], metadata, backend, path, **kwargs, prompt_variant="concise_v1")
    with pytest.raises(ValueError, match="Wrong oracle"):
        load_shards([path], metadata | {"sha256": "another-oracle"}, method="leact")


def test_prompt_diagnostic_does_not_count_truncated_baseline_as_valid_improvement():
    from reproductions.leact.prompt_diagnostic import analyze

    class ShortTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1]*100

    raw = generate_batch([state()], "rules", RecordingBackend(), method="leact", n=2, m=2)
    metadata = {"rules": "rules", "sha256": "sha"}
    good = analyze([state()], metadata, raw, ShortTokenizer(), "original")
    assert good["positive_selected_with_clean_baselines"] == 2
    raw[0]["baselines"][0]["finish_reason"] = "length"
    invalid = analyze([state()], metadata, raw, ShortTokenizer(), "original")
    assert invalid["positive_selected"] == 2
    assert invalid["positive_selected_with_clean_baselines"] == 0
    assert invalid["admission"]["2048"]["selected_traces"] == 2
    assert invalid["admission"]["2048"]["selected_traces_with_clean_baselines"] == 0


def test_eval_seeds_share_loaded_model_but_keep_distinct_sampling_and_artifacts(tmp_path, monkeypatch):
    import sys
    from reproductions.leact import __main__ as cli, inference

    backend = RecordingBackend()
    initializations = []

    def initialize(*args, **kwargs):
        initializations.append(args)
        return backend

    metadata = {"rules": "rules", "sha256": "oracle-checksum"}
    monkeypatch.setattr(cli, "load_oracle", lambda path: ([state()], metadata))
    monkeypatch.setattr(inference, "VLLMBackend", initialize)
    monkeypatch.setattr(sys, "argv", ["leact", "generate", "--oracle", "fixture.jsonl",
                        "--checkpoint", "pinned", "--method", "eval", "--eval-seeds", "0", "1", "2",
                        "--output", str(tmp_path / "shard-00.jsonl")])
    cli.main()
    assert len(initializations) == 1
    assert len({request.seed for request in backend.requests}) == 3
    for seed in [0, 1, 2]:
        rows = load_shards([tmp_path / f"seed-{seed}" / "shard-00.jsonl"], metadata, method="eval")
        assert len(rows) == 1 and rows[0]["seed"] == seed


def test_no_positive_trace_means_no_training_fallback(tmp_path):
    s = state()
    raw = {"state_id": s["state_id"], "baselines": [generation(policy_answer(s["expert_policy"]))],
           "candidates": [{"reasoning": "weak reasoning", "forward": generation(policy_answer({"fold": 0.5, "call": 0.5}))}]}
    with pytest.raises(ValueError, match="No examples survived"):
        prepare_data([s], {"rules": "rules", "sha256": "sha"}, tmp_path / "sft", method="leact", raw=[raw])
    assert not (tmp_path / "sft").exists()


def test_coldstart_gate_records_and_rejects_an_all_truncated_pool(tmp_path, monkeypatch):
    from reproductions.leact import experiment

    rows = generate_batch([state()], "rules", RecordingBackend(), method="leact", n=2, m=2)
    for candidate in rows[0]["candidates"]:
        candidate["backward"]["finish_reason"] = "length"
    monkeypatch.setattr(experiment, "parallel_generate", lambda *args, **kwargs: [])
    monkeypatch.setattr(experiment, "load_shards", lambda *args, **kwargs: rows)
    with pytest.raises(RuntimeError, match="no complete, nonempty reasoning"):
        experiment.check_coldstart_generation(tmp_path / "oracle", tmp_path / "model", tmp_path / "diagnostic",
                                             states=[state()], metadata={}, gpus=1)
    result = json.loads((tmp_path / "diagnostic/result.json").read_text())
    assert result["truncated_candidates"] == 2
    assert result["usable_candidates"] == result["positive_selected"] == 0


@pytest.fixture(scope="module")
def real_oracle(tmp_path_factory):
    path = tmp_path_factory.mktemp("oracle") / "oracle.jsonl"
    build_oracle(path, ranks=2, suits=2, iterations=30, tolerance=0.01, check_every=30)
    return load_oracle(path)


def test_full_tree_evaluator_matches_independent_solver_residual(real_oracle):
    states, metadata = real_oracle
    rows = [{"state_id": s["state_id"], "seed": 0,
             "candidates": [{"forward": generation(policy_answer(s["expert_policy"]))}]}
            for s in states]
    result = evaluate_records(states, metadata, rows)
    assert result["single"]["exploitability"] == pytest.approx(metadata["exploitability"])
    assert result["single"]["mean_capped_kl"] == pytest.approx(0, abs=1e-12)
    assert result["oracle_bon_requires_teacher"] is True
    with pytest.raises(ValueError, match="exactly one record"):
        evaluate_records(states, metadata, rows[:-1])


def test_oracle_checksum_detects_tampering(tmp_path):
    path = tmp_path / "oracle.jsonl"
    build_oracle(path, ranks=2, suits=2, iterations=1, tolerance=0.5)
    path.write_text(path.read_text().replace('"player": 0', '"player": 1', 1))
    with pytest.raises(ValueError, match="checksum"):
        load_oracle(path)


def test_checkpoint_retention_keeps_selected_and_metrics(tmp_path):
    from reproductions.leact.experiment import retain_selected_weights

    root = tmp_path / "checkpoints"
    for directory in [root, root / "checkpoint-1", root / "checkpoint-2"]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors").write_text("test weights")
        (directory / "trainer_state.json").write_text('{}')
    with pytest.raises(ValueError, match="no saved weights"):
        retain_selected_weights(root, root / "checkpoint-3")
    assert (root / "model.safetensors").exists()
    retain_selected_weights(root, root / "checkpoint-1")
    assert (root / "checkpoint-1" / "model.safetensors").exists()
    assert not (root / "checkpoint-2" / "model.safetensors").exists()
    assert (root / "checkpoint-2" / "trainer_state.json").exists()
