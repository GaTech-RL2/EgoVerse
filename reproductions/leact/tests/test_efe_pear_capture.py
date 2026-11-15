"""Native reward/token join must preserve failures and reject partial provenance."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import pytest

from reproductions.leact.cluster.efe_pear_sampler import final_answer, digest_body, write
from reproductions.leact.cluster.efe_pear_native_worker import capture_episode


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setenv('EFE_WORKFLOW_ID', 'test-native')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'cluster'))
    episode = tmp_path/'jobs/00/episode'
    for name in ['verifier', 'agent']:
        (episode/name).mkdir(parents=True)
    (episode/'verifier/episode.jsonl').write_text(json.dumps({'ev':'start', 'state_sha':'reset'})+'\n')
    (episode/'verifier/replay.mp4').write_bytes(b'test-fixture')
    write(episode/'agent/trajectory.json', {'steps':[]})
    rewards = {'success':0, 'resource_enforced':1}
    grade = {'verifier_result':{'rewards':rewards}}
    write(episode/'result.json', grade)
    trial = {'archive':str(episode.parent), 'task':'pick', 'suite':'metaworldplus',
        'partition':'train_candidates', 'status':'graded', 'resource_metrics':rewards,
        'model_requests':1, 'job':'00', 'score':0,
        'trial_result_sha256':hashlib.sha256(json.dumps(grade, sort_keys=True).encode()).hexdigest()}
    body = {'model':'qwen', 'instructions':'act', 'input':[]}
    requests = tmp_path/'requests/00';requests.mkdir(parents=True)
    write(requests/'request.request.json.gz', body, compressed=True)
    write(requests/'request.json', {'id':'request', 'started_utc':'2026-11-14', 'http_status':200,
        'status':'completed', 'public_output':[{'type':'message', 'id':'msg_pear_one'}],
        'usage':{'output_tokens':2}})
    samples = tmp_path/'sampler';samples.mkdir()
    record = {'request_sha256':digest_body(body), 'sampler_revision':'policy',
        'token_ids':[1,2], 'prompt_token_ids':[33,34], 'sampled_logprobs':[-math.log(64)]*2,
        'topk_token_ids':[list(range(32))]*2, 'topk_logprobs':[[-math.log(64)]*32]*2,
        'temperature':1., 'top_p':1., 'top_k':-1, 'model_seconds':2., 'finish_reason':'length'}
    write(samples/'pear_one.json.gz', record, compressed=True)
    spec = {'student_revision':'policy', 'pear':{'vocab_size':64, 'capture_top_k':32, 'role':'qualification'}}
    membership = {'training_tasks':{'pick':[{'scene_sha256':'scene', 'initial_state_sha256':'reset'}]}}
    return trial, spec, tmp_path, samples, membership, {'scene_sha256':'scene'}


def test_failed_truncated_episode_keeps_all_sampled_tokens(native):
    row = capture_episode(*native)
    assert row['reward'] == 0 and row['output_truncated']
    assert row['generated_tokens'] == 2 and row['model_calls'] == 1
    assert row['native_events_verified'] and row['sampling_capture_complete']
    assert len(row['captures']) == 1 and row['model_seconds'] == 2.


@pytest.mark.parametrize('change', ['policy', 'input', 'token_count', 'reset', 'native_grade'])
def test_native_capture_rejects_mismatched_evidence(native, change):
    trial, spec, archive, samples, membership, scene = native
    if change in {'policy', 'input'}:
        path = samples/'pear_one.json.gz'
        row = json.loads(gzip.decompress(path.read_bytes()))
        row['sampler_revision' if change=='policy' else 'request_sha256'] = 'wrong'
        write(path, row, compressed=True)
    elif change == 'token_count':
        path = archive/'requests/00/request.json';row = json.loads(path.read_text())
        row['usage']['output_tokens'] = 1;write(path, row)
    elif change == 'reset':
        membership['training_tasks']['pick'][0]['initial_state_sha256'] = 'heldout'
    else:
        trial['score'] = 1
    with pytest.raises(ValueError):
        capture_episode(*native)


def test_unfinished_student_thought_is_not_an_action():
    assert final_answer('I might emit {"action":[1]}') == ''
    assert final_answer('reasoning\n</think>\n{"stop":true}') == '{"stop":true}'
