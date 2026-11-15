import json

from reproductions.leact.cluster.launch_efe_pear_optimizer import result_from_logs


def receipt():
    return {'success':True,'optimizer_updates':1,'primary_rl_updates':0,
        'native_episodes':8,'generated_tokens':1418034,
        'exact_adam_and_rng_restored':True,'fresh_process_replay_max_error':0.0}


def test_final_receipt_requires_a_real_update_and_exact_fresh_process_reload():
    value=receipt()
    assert result_from_logs('2026/11/15 [preflight] PEAR_OPTIMIZER_RESULT '+json.dumps(value))==value
    for key,bad in [('success',False),('optimizer_updates',0),('primary_rl_updates',1),
                    ('native_episodes',7),('generated_tokens',1),
                    ('exact_adam_and_rng_restored',False),('fresh_process_replay_max_error',1.0)]:
        candidate={**value,key:bad}
        assert result_from_logs('PEAR_OPTIMIZER_RESULT '+json.dumps(candidate)) is None


def test_scheduler_completion_and_partial_progress_are_not_results():
    assert result_from_logs('COMPLETED\nPEAR_OPTIMIZER_PROGRESS '+json.dumps(receipt())) is None
    assert result_from_logs('PEAR_OPTIMIZER_RESULT {truncated') is None
