from copy import deepcopy

import pytest

from reproductions.leact.compare import comparison


def test_comparison_selects_checkpoints_per_metric_not_individual_seeds():
    first = {"checkpoint": "epoch-1", "inference_seeds": [0, 1, 2],
             "single": {"exploitability": {"mean": 0.1, "std": 0.03}},
             "oracle_bon": {"exploitability": {"mean": 0.02, "std": 0.01}}}
    second = deepcopy(first)
    second["checkpoint"] = "epoch-2"
    second["single"]["exploitability"]["mean"] = 0.12
    second["oracle_bon"]["exploitability"]["mean"] = 0.01
    report = {"scope": "Leduc 6r2s independent reproduction",
              "oracle": {"num_infosets": 4032, "sha256": "fixture", "exploitability": 1e-5},
              "protocol": {"rounds": 3},
              "methods": {"leact": [{"round": 1, "all_epochs": [first, second]}]}}
    result = comparison(report)
    single, bon = result["rows"]
    assert single["checkpoint"] == "epoch-1"
    assert bon["checkpoint"] == "epoch-2"
    assert single["std"] == 0.03
    assert single["inference_seeds"] == [0, 1, 2]
    assert not result["all_planned_rounds_complete"]
    # A registered LeAct-only study can finish without claiming an ExIt run.
    report["protocol"].update(rounds=1, planned_methods=["leact"])
    assert comparison(report)["all_planned_rounds_complete"]
    report["protocol"]["planned_methods"] = ["leact", "exit"]
    assert not comparison(report)["all_planned_rounds_complete"]


def test_preflight_cannot_be_reported_as_a_benchmark():
    with pytest.raises(ValueError, match="not preflight"):
        comparison({"scope": "infrastructure preflight; no benchmark reproduction claim"})


def test_projection_ablation_keeps_its_scope_and_full_tree_requirement():
    scope = 'Leduc 6r2s independent reproduction with projection-sampled frontier warmstart'
    report = dict(scope=scope, oracle=dict(num_infosets=4032, sha256='fixture', exploitability=1e-5),
                  protocol=dict(rounds=2, planned_methods=['leact']), methods=dict(leact=[]))
    result = comparison(report)
    assert result['scope'] == scope
    assert result['all_planned_rounds_complete'] is False
    report['oracle']['num_infosets'] = 32
    with pytest.raises(ValueError, match='not preflight'):
        comparison(report)


def test_experiment_resume_rejects_changed_oracle_at_the_same_path(tmp_path):
    from reproductions.leact.experiment import audit_source

    args = {"oracle": "same-path.jsonl", "gpus": 8}
    audit_source(tmp_path, "full", args, "first-oracle")
    audit_source(tmp_path, "full", args, "first-oracle")
    with pytest.raises(ValueError, match="protocol differs"):
        audit_source(tmp_path, "full", args, "different-oracle")


def test_bc_evaluates_all_epochs_without_a_reasoning_gate(tmp_path, monkeypatch):
    from reproductions.leact import experiment
    root = tmp_path/'model'
    for step in [126, 252, 378]:
        p = root/f'checkpoint-{step}'
        p.mkdir(parents=True)
        (p/'config.json').write_text('{}')
        (p/'model.safetensors').write_text('fixture')
    calls = []
    def evaluate(oracle, checkpoint, directory, **kwargs):
        calls.append((checkpoint.name, kwargs['thinking']))
        epoch = int(checkpoint.name.split('-')[1])//126
        return {'checkpoint': str(checkpoint), 'inference_seeds': kwargs['seeds'],
                'single': {'exploitability': {'mean': epoch/10, 'std': 0.01}},
                'oracle_bon': {'exploitability': {'mean': (4-epoch)/100, 'std': 0.001}}}
    monkeypatch.setattr(experiment, 'evaluate_checkpoint', evaluate)
    monkeypatch.setattr(experiment, 'check_coldstart_generation', lambda *a, **k: pytest.fail('BC must not require reasoning'))
    report = experiment.evaluate_bc_epochs(tmp_path/'oracle', root, tmp_path/'report', states=[],
        metadata={'num_infosets': 4032, 'sha256': 'oracle', 'exploitability': 1e-5}, gpus=8, seeds=[0, 1, 2])
    assert calls == [('checkpoint-126', False), ('checkpoint-252', False), ('checkpoint-378', False)]
    result = comparison(report)
    assert result['rows'][0]['checkpoint'].endswith('checkpoint-126')
    assert result['rows'][1]['checkpoint'].endswith('checkpoint-378')
    assert not result['all_planned_rounds_complete']
    (root/'checkpoint-126/model.safetensors').unlink()
    with pytest.raises(ValueError, match='saved epoch weights'):
        experiment.epoch_checkpoints(root, expected_epochs=3)
    with pytest.raises(ValueError, match='Expected 4'):
        experiment.epoch_checkpoints(root, expected_epochs=4)


def test_reasoning_requires_explicit_provenance_without_assuming_a_variant(tmp_path):
    import json
    from reproductions.leact.experiment import validate_reasoning_anchor
    with pytest.raises(ValueError, match='explicit'):
        validate_reasoning_anchor(None, None)
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    (checkpoint/'model.safetensors').write_text('fixture')
    p = tmp_path/'provenance.json'
    for variant in ['nocot', 'cot', 'joint']:
        value = {'checkpoint': str(checkpoint), 'variant': variant,
                 'origin': 'independent_reconstruction', 'source': 'fixture corpus',
                 'dataset_sha256': 'a'*64, 'protocol_note': 'Explicit experimental assumption; not author verified.'}
        p.write_text(json.dumps(value))
        assert validate_reasoning_anchor(checkpoint, p) == value
    p.write_text(json.dumps({**value, 'dataset_sha256': 'not-a-hash'}))
    with pytest.raises(ValueError, match='checksum'):
        validate_reasoning_anchor(checkpoint, p)
