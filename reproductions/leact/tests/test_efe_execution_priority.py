from copy import deepcopy

import pytest

from reproductions.leact.cluster.launch_efe_coverage_eval import ordered_batches
from reproductions.leact.cluster.launch_efe_selection_workers import helper_assignments


def test_prioritizing_r1_preserves_frozen_panels_and_existing_results():
    batches=[{'model':model,'id':str(i),'tasks':[{'task':f'task-{i}'}],
              'attempts':[{'workflow':f'prior-{i}'}]}
             for i,model in enumerate(['warmstart','leact-from-step496-r1','base',
                                       'leact-from-step496-r1','warmstart-step496'])]
    before=deepcopy(batches)
    ordered=ordered_batches(batches,['leact-from-step496-r1','warmstart-step496'])
    assert [b['id'] for b in ordered]==['1','3','4','0','2']
    assert batches==before
    assert sorted(ordered,key=lambda b:b['id'])==batches


def test_helpers_cover_only_unfinished_states_exactly_once():
    missing=[1700,301,499,1500,700,2047,1]
    groups=helper_assignments(missing,2048,4)
    assert sorted(i for group in groups for i in group)==sorted(missing)
    assert max(map(len,groups))-min(map(len,groups))<=1
    assert helper_assignments([],2048,4)==[]
    assert helper_assignments([1],2048,4)==[[1]]


@pytest.mark.parametrize('indices',[[1,1],[-1],[2048],[True]])
def test_helpers_reject_changed_or_invalid_membership(indices):
    with pytest.raises(ValueError):helper_assignments(indices,2048,4)
