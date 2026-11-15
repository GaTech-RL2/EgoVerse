import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.setitem(sys.modules, 'efe_native_pilot', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'efe_budget_gateway', SimpleNamespace(public_value=lambda v, _: v))
    path = Path(__file__).parents[1] / 'cluster/efe_astra_pilot.py'
    spec = importlib.util.spec_from_file_location('native_resume_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    spec = {'tasks': [{'task': str(i), 'suite': 'sim', 'partition': 'development'} for i in range(3)],
            'student_checkpoint': 'step32', 'sampling': 'frozen', 'max_output_tokens': 32768}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    trials = [{**row, 'status': 'graded', 'score': int(row['task'])} for row in spec['tasks'][:2]]
    trials.append({**spec['tasks'][2], 'status': 'infrastructure_failure', 'score': None})
    (tmp_path / 'results.json').write_text(json.dumps({'workflow': 'prior-1', 'trials': trials}))
    spec['resume_completed_trials'] = {'archive': str(tmp_path), 'workflow': 'prior-1',
        'source_spec_sha256': hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}
    return spec


def test_preserve_zero_score_and_skip_only_completed(pilot, tmp_path):
    spec = fixture(tmp_path)
    trials, receipt = pilot.resume_trials(spec)
    assert [row['score'] for row in trials] == [0, 1]
    assert all(row['source_workflow'] == 'prior-1' for row in trials)
    remaining = [r['task'] for r in spec['tasks'] if r['task'] not in {t['task'] for t in trials}]
    assert remaining == ['2'] and receipt['retained_graded_trials'] == 2


@pytest.mark.parametrize('change', ['workflow', 'duplicate', 'unknown', 'status', 'score'])
def test_reject_invalid_history(pilot, tmp_path, change):
    spec = fixture(tmp_path)
    path = tmp_path / 'results.json'
    result = json.loads(path.read_text())
    if change == 'workflow': result['workflow'] = 'different'
    elif change == 'duplicate': result['trials'].append(result['trials'][0])
    elif change == 'unknown': result['trials'][0]['task'] = 'foreign'
    elif change == 'status': result['trials'][0]['status'] = 'unknown'
    else: result['trials'][0]['score'] = None
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError): pilot.resume_trials(spec)


def test_reject_protocol_change(pilot, tmp_path):
    spec = fixture(tmp_path)
    spec['max_output_tokens'] = 65536
    with pytest.raises(ValueError, match='changed evaluation condition'):
        pilot.resume_trials(spec)


def test_no_resume_has_no_history(pilot):
    assert pilot.resume_trials({}) == ([], None)


def test_startup_preemption_preserves_older_completed_grades(pilot,tmp_path):
    spec=fixture(tmp_path)
    previous=spec['resume_completed_trials']
    spec['resume_completed_trials']={'archive':str(tmp_path/'never-started'),
        'workflow':'startup-preemption','fallback_sources':[previous]}
    trials,receipt=pilot.resume_trials(spec)
    assert [r['score'] for r in trials]==[0,1]
    assert receipt['source_workflow']=='prior-1'
