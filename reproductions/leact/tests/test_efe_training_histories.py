import json

from reproductions.leact.embodied_first_exam.report import training_histories


def test_registered_run_without_updates_has_no_fabricated_loss(tmp_path):
    run=tmp_path/'training-pilot/comparison';run.mkdir(parents=True)
    (run/'scheduler.json').write_text(json.dumps({'status':'PENDING','name':'workflow'}))
    comparison={'arms':[{'id':'new','name':'New run','local_results_directory':'training-pilot/comparison','optimizer_steps':48}]}
    rows=training_histories(tmp_path,comparison)
    assert len(rows)==1 and rows[0]['history']==[]
    assert rows[0]['planned_steps']==48 and rows[0]['status']=='Queued'
    assert rows[0]['workflow']=='workflow' and not rows[0]['complete']


def test_only_complete_logged_updates_are_plotted_and_smoke_is_excluded(tmp_path):
    run=tmp_path/'training-pilot/run-v5';train=run/'train';train.mkdir(parents=True)
    row={'step':1,'steps':16,'forward_ce':1.,'backward_ce':3.,'forward_examples':16,'backward_examples':16,
         'gradient_norm':2.,'lr':1e-6,'max_allocated_bytes_per_gpu':2**30}
    (train/'metrics.jsonl').write_text(json.dumps(row)+'\n'+json.dumps({**row,'step':2}))
    smoke=run/'smoke';smoke.mkdir();(smoke/'metrics.jsonl').write_text(json.dumps({**row,'forward_ce':99.})+'\n')
    rows=training_histories(tmp_path,{})
    assert len(rows)==1 and len(rows[0]['history'])==1
    assert rows[0]['history'][0]['loss']==2. and rows[0]['history'][0]['gpu_memory_gib']==1.
    assert rows[0]['category']=='efe-warmstart' and rows[0]['planned_steps']==16


def test_scheduler_cancellation_overrides_stale_training_progress(tmp_path):
    run=tmp_path/'training-pilot/comparison';run.mkdir(parents=True)
    (run/'progress.json').write_text(json.dumps({'phase':'train','detail':'train: optimizer step 8/48'}))
    (run/'scheduler.json').write_text(json.dumps({'status':'FAILED_CANCELED','name':'workflow',
        'groups':[{'failure_message':'GPU quota enforcement: reclaims over-allocation'}]}))
    comparison={'arms':[{'id':'new','name':'New run','local_results_directory':'training-pilot/comparison','optimizer_steps':48}]}
    row=training_histories(tmp_path,comparison)[0]
    assert row['status']=='Canceled · GPU quota'
    assert row['scheduler_status']=='FAILED_CANCELED'
    assert 'No completed training receipt' in row['detail']
    assert not row['complete'] and not row['history']


def test_quality_hold_preserves_but_excludes_invalid_updates(tmp_path):
    run=tmp_path/'training-pilot/comparison';train=run/'train';train.mkdir(parents=True)
    metric={'step':21,'steps':737,'forward_ce':.916,'backward_ce':1.222,
            'forward_examples':16,'backward_examples':16,'gradient_norm':1.41,
            'lr':5.67e-5,'max_allocated_bytes_per_gpu':2**30}
    (train/'metrics.jsonl').write_text(json.dumps(metric)+'\n'+json.dumps(
        {**metric,'step':22,'forward_ce':11.223,'backward_ce':12.181})+'\n')
    (run/'scheduler.json').write_text(json.dumps({'status':'RUNNING'}))
    (run/'quality-hold.json').write_text(json.dumps({'active':True,'suspect_updates':[22,23,24]}))
    comparison={'arms':[{'id':'coverage','local_results_directory':'training-pilot/comparison'}]}
    row=training_histories(tmp_path,comparison)[0]
    assert row['status']=='Quality hold' and not row['complete']
    assert [m['step'] for m in row['history']]==[21]
    assert [m['step'] for m in row['rejected_history']]==[22]
    assert row['rejected_history'][0]['forward_ce']==11.223
