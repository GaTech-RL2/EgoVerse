import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reproductions.leact.coldstart import continue_leact as study
from reproductions.leact.data import file_sha256, write_json


def anchor_fixture(root):
    training = root / 'self_4032_n4/sft'
    checkpoint = training / 'checkpoints'
    checkpoint.mkdir(parents=True)
    write_json(training / 'data.json', [{'input': 'visible state', 'output': 'reasoning and CFR action'}])
    data_sha = file_sha256(training / 'data.json')
    write_json(training / 'manifest.json', dict(data_sha256=data_sha))
    (checkpoint / 'model-00001.safetensors').write_bytes(b'weight fixture')
    write_json(checkpoint / 'model.safetensors.index.json', dict(weight_map={'layer.weight':'model-00001.safetensors'}))
    write_json(checkpoint / 'weight_retention.json', dict(retained_checkpoint=str(checkpoint),
        weight_sha256={'model-00001.safetensors':file_sha256(checkpoint / 'model-00001.safetensors')}))
    write_json(root / 'results.json', dict(protocol={'role':'self'}, oracle={'sha256':'oracle'},
        coldstarts={'self_4032_n4':dict(selected_checkpoint=str(checkpoint), checkpoint_rule='fixed epoch 3')}))
    report_sha = file_sha256(root / 'results.json')
    write_json(root / 'complete.json', dict(results_sha256=report_sha))
    return dict(source_arm='self_4032_n4', source_results_sha256=report_sha,
        source_manifest_sha256=file_sha256(training / 'manifest.json'), source_data_sha256=data_sha)


def test_continuation_audits_the_completed_warmstart_and_retained_weights(tmp_path):
    registration = anchor_fixture(tmp_path)
    report, anchor, retention = study.verify_anchor(tmp_path, registration, 'oracle')
    assert report['protocol']['role'] == 'self'
    assert anchor['selected_checkpoint'] == retention['retained_checkpoint']
    with pytest.raises(ValueError, match='same CFR oracle'):
        study.verify_anchor(tmp_path, registration, 'different-oracle')


@pytest.mark.parametrize('relative,error', [
    ('results.json', 'report changed'),
    ('complete.json', 'not completely published'),
    ('self_4032_n4/sft/data.json', 'data changed'),
    ('self_4032_n4/sft/checkpoints/model-00001.safetensors', 'weight checksum mismatch')])
def test_continuation_rejects_changed_or_unfinished_parent(tmp_path, relative, error):
    registration = anchor_fixture(tmp_path)
    path = tmp_path / relative
    if relative == 'complete.json':
        write_json(path, dict(results_sha256='incomplete'))
    else:
        path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match=error):
        study.verify_anchor(tmp_path, registration, 'oracle')


def test_resume_uses_retained_checkpoint_and_does_not_repeat_completed_rounds(tmp_path, monkeypatch):
    args = SimpleNamespace(root=tmp_path, oracle=tmp_path/'oracle', scratch=tmp_path/'scratch')
    report = {'methods':{'leact':[dict(round=1, selected_checkpoint='/saved/selected-epoch',
                                     best={'checkpoint':'/different/final-epoch'})]}}
    calls = []
    def generate(oracle, checkpoint, directory, **settings):
        calls.append(('generate', str(checkpoint), settings))
        return []
    def prepare(states, metadata, directory, **settings):
        calls.append(('prepare', str(directory), settings))
        directory.mkdir(parents=True)
    def train(args, label, training, checkpoint, states, metadata, **settings):
        calls.append(('train', str(checkpoint), settings))
        return dict(selected_checkpoint='/saved/round-two-selected')
    monkeypatch.setattr(study, 'parallel_generate', generate)
    monkeypatch.setattr(study, 'load_shards', lambda *args, **kwargs: [])
    monkeypatch.setattr(study, 'prepare_data', prepare)
    monkeypatch.setattr(study, 'train_and_evaluate', train)
    monkeypatch.setattr(study, 'write_comparison', lambda *args: None)
    study.continue_rounds(args, report, Path('/warmstart'), [], {}, None)
    assert [c[0] for c in calls] == ['generate','prepare','train']
    assert calls[0][1] == calls[2][1] == '/saved/selected-epoch'
    assert calls[0][2] == dict(gpus=8, method='leact', seed=2, n=8, m=4, prompt_variant='concise_v1')
    assert calls[1][2]['require_valid_policies'] is True
    assert calls[2][2] == dict(lr=5e-6, all_epochs=True)
    assert [r['round'] for r in json.loads((tmp_path/'results.json').read_text())['methods']['leact']] == [1,2]
    study.continue_rounds(args, report, Path('/warmstart'), [], {}, None)
    assert len(calls) == 3
