import pytest
from reproductions.leact.cluster.launch_efe_pear_update_qualification import training_successes


@pytest.mark.parametrize('partition,member,model,status,score,accepted', [
    ('training_evaluation',True,'base','graded',1,True),
    ('development',True,'base','graded',1,False),
    ('training_evaluation',False,'base','graded',1,False),
    ('training_evaluation',True,'astra','graded',1,False),
    ('training_evaluation',True,'base','infrastructure_failure',1,False),
    ('training_evaluation',True,'base','graded',0,False),
])
def test_optimizer_qualification_uses_only_admitted_training_successes(partition, member, model, status, score, accepted):
    task='test-task'
    evidence={'evaluations':{'leact-efe-coverage-eval-'+model+'-test':{'results':{'trials':[
        {'task':task,'partition':partition,'status':status,'score':score}]}}}}
    membership={'training_tasks':{task:[]} if member else {}}
    assert (task in training_successes(evidence,membership)) is accepted
