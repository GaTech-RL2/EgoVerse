import copy

import pytest

from reproductions.leact.cluster.efe_train_vl import training_plan
from reproductions.leact.cluster.efe_training_resume import validate_reuse
from reproductions.leact.cluster.launch_efe_schedule import comparison_spec, VARIANTS


def fixture():
    original={'base_model':'pinned-base','base_revision':'revision','context':32768,'seed':20261027,
              'effective_batch':32,'scene_audit_sha256':'audit','lr':1e-6}
    rows=[{'id':f'{i}-{direction}','state_id':str(i),'direction':direction,
           'suite':f'suite-{i%3}','task':f'task-{i%12}','episode':f'episode-{i%21}'}
          for i in range(256) for direction in ['forward','backward']]
    return original,rows


def test_recovery_does_not_silently_change_lr_or_training_budget():
    original,_=fixture()
    for change in [{'lr':1e-4},{'balanced_passes':3}]:
        with pytest.raises(ValueError,match='Recovery changed frozen'):
            validate_reuse({**original,**change},original)


def test_named_comparisons_share_exact_exposures_and_snapshot_budgets():
    original,rows=fixture();schedules=[]
    for variant in VARIANTS:
        spec,schedule=comparison_spec(original,variant,rows)
        validate_reuse(spec,original)
        steps,again,snapshots,digest=training_plan(rows,spec)
        assert steps==48 and snapshots==[16,32,48] and len(schedule)==1536
        assert schedule==again and digest==spec['expected_schedule_sha256']
        assert sum(r['direction']=='forward' for r in schedule)==768
        schedules.append(schedule)
    assert schedules[0]==schedules[1]==schedules[2]
    assert schedules[0][:512]==training_plan(rows,original)[1]


@pytest.mark.parametrize('field,value',[('base_model','other'),('seed',0),('context',8192),('scene_audit_sha256','other')])
def test_named_comparison_cannot_change_base_data_or_context(field,value):
    original,rows=fixture();spec,_=comparison_spec(original,VARIANTS[-1],rows)
    spec[field]=value
    with pytest.raises(ValueError,match='frozen condition'):validate_reuse(spec,original)


def test_undeclared_lr_change_and_tampered_budget_or_order_are_rejected():
    original,rows=fixture();spec,_=comparison_spec(original,VARIANTS[-1],rows)
    bad=copy.deepcopy(spec);bad['lr']=2e-4
    with pytest.raises(ValueError,match='declared exactly'):validate_reuse(bad,original)
    for patch in [{'expected_optimizer_steps':16},{'expected_schedule_sha256':'incorrect'}]:
        with pytest.raises(ValueError,match='frozen comparison'):training_plan(rows,{**spec,**patch})
