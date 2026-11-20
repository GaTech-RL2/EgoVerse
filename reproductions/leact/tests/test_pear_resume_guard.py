import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def worker(monkeypatch):
    cluster=Path(__file__).resolve().parents[1]/'cluster'
    monkeypatch.syspath_prepend(str(cluster))
    spec=importlib.util.spec_from_file_location('primary_resume_guard',cluster/'efe_pear_primary_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_new_run_without_optimizer_history_can_start_from_warmstart(worker,tmp_path):
    assert worker.checkpoint_for_resume(tmp_path)is None


def test_missing_checkpoint_cannot_silently_reset_existing_training(worker,tmp_path):
    steps=tmp_path/'telemetry/steps';steps.mkdir(parents=True)
    (steps/'00000003.json').write_text('{"step":3}')
    with pytest.raises(RuntimeError,match='explicit recovery or fresh-run fork'):
        worker.checkpoint_for_resume(tmp_path)
    assert json.loads((steps/'00000003.json').read_text())=={'step':3}


def test_verified_checkpoint_can_resume_an_existing_run(worker,tmp_path):
    pointer={'step':4,'archive':'s3://private/step4','readback_verified':True}
    (tmp_path/'checkpoint-latest.json').write_text(json.dumps(pointer))
    assert worker.checkpoint_for_resume(tmp_path)==pointer


@pytest.mark.parametrize('pointer',[
    {'step':4,'archive':'s3://private/step4','readback_verified':False},
    {'step':4,'readback_verified':True},
    {'step':True,'archive':'s3://private/step4','readback_verified':True},
])
def test_uncommitted_or_incomplete_checkpoint_is_never_a_resume_source(worker,tmp_path,pointer):
    (tmp_path/'checkpoint-latest.json').write_text(json.dumps(pointer))
    with pytest.raises(ValueError,match='verified resume checkpoint'):
        worker.checkpoint_for_resume(tmp_path)
