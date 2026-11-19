import json
import time

import pytest

from reproductions.leact.cluster import efe_pear_recipe_transition as transition


def cases():
    return[{'task':str(i),'model_revision':'warm496','status':'completed',
        'valid_actions':30,'invalid_actions':0,'official_success':int(i==0)}for i in range(4)]


def test_recipe_switch_requires_complete_native_success_and_valid_actions():
    rows=cases();tasks=[str(i)for i in range(4)]
    assert transition.replay_ready(rows,tasks,'warm496')
    assert not transition.replay_ready(rows[:-1],tasks,'warm496')
    assert not transition.replay_ready(rows,tasks,'other-weights')
    assert not transition.replay_ready([{**r,'official_success':0}for r in rows],tasks,'warm496')
    assert not transition.replay_ready([{**r,'invalid_actions':5}for r in rows],tasks,'warm496')
    rows[-1].update(status='infrastructure_or_runtime_failure',official_success=None,valid_actions=0)
    assert transition.replay_ready(rows,tasks,'warm496')


@pytest.mark.parametrize('updates,success,expected',[(1,True,'requires_checkpoint_handoff'),(0,False,'requires_harness_review')])
def test_recipe_switch_preserves_the_model_until_observed_readiness(tmp_path,monkeypatch,updates,success,expected):
    monkeypatch.setattr(transition,'DEST',tmp_path)
    state={'workflow':'current-model','run_id':'arm'}
    values={'recipe-transition':{'approved':True,'from_workflow':'current-model',
        'next_sampling_policy':transition.POLICY,'known_training_tasks':[str(i)for i in range(4)],'model_revision':'warm496'},
        'control':{'enabled_training_arms':['arm']},
        'matched-warmstart-smoke-h100-client':{'output':'not-read','attempt':3},
        'primary-driver-check-launch':{'name':'cpu-validation'}}
    for name,value in values.items():(tmp_path/(name+'.json')).write_text(json.dumps(value))
    rows=cases()
    if not success:rows=[{**r,'official_success':0}for r in rows]
    monkeypatch.setattr(transition,'capture',lambda *args:{'cases':rows,'completed_updates':updates,
        'head':{'workflow':'current-model','phase':'running_full_mixture_pear','updated_unix':time.time()},
        'driver_check':{'passed':True,'sampling_policies':[transition.POLICY]}})
    monkeypatch.setattr(transition,'command',lambda *args:{'status':'COMPLETED'})
    monkeypatch.setattr(transition.subprocess,'run',lambda *a,**k:pytest.fail('A live model was canceled before validation'))
    assert transition.tick(state)is None
    assert json.loads((tmp_path/'recipe-transition.json').read_text())['phase']==expected


def test_accepted_cancellation_is_reconciled_before_resubmitting_the_new_recipe(tmp_path,monkeypatch):
    from reproductions.leact.cluster import launch_efe_pear_primary as primary
    monkeypatch.setattr(transition,'DEST',tmp_path)
    state={'workflow':'owned-model','run_id':'arm','profile':12,'pool':'groot-h100-01'}
    (tmp_path/'control.json').write_text(json.dumps({'enabled_training_arms':['arm']}))
    (tmp_path/'recipe-transition.json').write_text(json.dumps({'approved':True,'from_workflow':'owned-model',
        'next_sampling_policy':transition.POLICY,'phase':'reallocating_for_validated_recipe'}))
    monkeypatch.setattr(transition,'command',lambda *a:{'status':'FAILED_CANCELED'})
    monkeypatch.setattr(transition.subprocess,'run',lambda *a,**k:pytest.fail('Cancellation was already accepted'))
    canceled=[];monkeypatch.setattr(primary,'cancel_native_workers',lambda name:canceled.append(name))
    monkeypatch.setattr(primary,'launch',lambda profile,pool:{'workflow':'new-model'})
    assert transition.tick(state)=={'workflow':'new-model'}
    assert canceled==['owned-model']
    assert json.loads((tmp_path/'primary-launch.json').read_text())['terminal_confirmed']is True
