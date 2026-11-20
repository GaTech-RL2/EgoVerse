from copy import deepcopy

import pytest

from reproductions.leact.cluster.efe_coverage_retry import inference_cuda_unavailable_before_rollouts


def evidence():
    marker='Qwen serving exited; inspect logs/student-vllm.log'
    return {'progress':{'workflow':'allocation','benchmark_rollouts_started':False,
                        'phase':'qwen_baseline_needs_review'},
            'worker_complete':{'workflow':'allocation','success':False,'error':marker},
            'complete':{'completed_utc':'2026-11-17T18:30:19+00:00','completed_trials':0,'error':marker},
            'preflight':{'status':'failed','error':marker},
            'serving_failure':{'returncode':1,'log_tail':'torch.accelerator.set_device_index(self.device)\n'
                              'CUDA error: CUDA-capable device(s) is/are busy or unavailable'},
            'results':{},'public_manifest':{}}


def test_initial_device_failure_without_any_rollout_is_retryable():
    assert inference_cuda_unavailable_before_rollouts({'name':'allocation','status':'FAILED'},evidence())


@pytest.mark.parametrize('field,changes',[
    ('progress',{'workflow':'older-allocation'}),
    ('worker_complete',{'workflow':'older-allocation'}),
    ('progress',{'benchmark_rollouts_started':True}),
    ('complete',{'completed_trials':1}),
    ('complete',{'success':True}),
    ('results',{'trials':[{'status':'ungraded'}]}),
    ('public_manifest',{'episodes':[{'task':'existing-outcome'}]}),
    ('serving_failure',{'log_tail':'CUDA out of memory'}),
    ('serving_failure',{'log_tail':'unknown engine initialization failure'}),
    ('serving_failure',{'returncode':0}),
    ('preflight',{'status':'passed'}),
])
def test_unknown_errors_or_any_prior_rollout_remain_under_review(field,changes):
    row=deepcopy(evidence());row[field].update(changes)
    assert not inference_cuda_unavailable_before_rollouts({'name':'allocation','status':'FAILED'},row)


def test_running_or_other_allocation_is_not_retried():
    assert not inference_cuda_unavailable_before_rollouts({'name':'allocation','status':'RUNNING'},evidence())
    assert not inference_cuda_unavailable_before_rollouts({'name':'other','status':'FAILED'},evidence())
