import errno
import http.client
from http.server import ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from urllib.parse import urlparse

import pytest
import yaml

from reproductions.leact.cluster import efe_astra_transport_diagnostics as diagnostic
from reproductions.leact.cluster import stage_astra_transport_diagnostics as stage


def bridge_module():
    """A private module keeps instrumentation out of other tests and hero imports."""
    path = Path(diagnostic.__file__).with_name('efe_pear_bridge.py')
    spec = importlib.util.spec_from_file_location('private_teacher_bridge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured(tmp_path, *, instrument=True):
    module = bridge_module()
    rpc = SimpleNamespace(serve_episode=lambda folder, factory, args: factory())
    original = module.HarborEpisode
    if instrument:
        diagnostic.install(module, rpc)
    source = tmp_path/'source'
    source.mkdir()
    for name in ['efe_action_agent.py', 'efe_action_loop.py']:
        (source/name).write_text('# unchanged native source\n')
    config = {'episode_root': str(tmp_path/'episodes'), 'agent_source': str(source),
              'action_config': {}, 'reference': str(tmp_path), 'harbor': 'not-executed',
              'advertise_address': '127.0.0.1', 'retain_transport_observations': False}
    row = {'cutoff_seconds': 60, 'task': 'train-task', 'path': 'train-task'}
    return module, rpc, original, row, config


def summary(episode):
    return json.loads((episode.root/'teacher-transport-diagnostics.json').read_text())


def events(episode):
    return [json.loads(line) for line in (episode.root/'teacher-transport-events.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('instrument', [False, True])
def test_real_loopback_transport_and_native_verdict_are_unchanged(tmp_path, monkeypatch, instrument):
    module, _, _, row, config = configured(tmp_path, instrument=instrument)
    observed = {}
    episode = module.HarborEpisode(row, config, 'native-episode')
    observation = {'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': json.dumps({
        'mode': 'task', 'action_contract': {'native': 'preserved'},
        'objective': 'not retained in diagnostics'})}]}]}

    class HarborProcess:
        pid = 12345
        returncode = None

        def __init__(self, command, **kwargs):
            observed['command'] = command
            endpoint = urlparse(kwargs['env']['OPENAI_BASE_URL'])
            token = kwargs['env']['OPENAI_API_KEY']
            def native_agent():
                try:
                    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=5)
                    connection.request('POST', '/v1/responses', body=json.dumps(observation),
                                       headers={'Authorization': 'Bearer '+token})
                    response = connection.getresponse()
                    observed['status'] = response.status
                    observed['response'] = response.read().decode()
                    connection.close()
                    path = episode.root/'jobs/native-episode/trial/result.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({'task_name': 'train-task',
                        'verifier_result': {'rewards': {'success': 1, 'resource_enforced': 1}}}))
                except Exception as exc:
                    observed['error'] = exc
                finally:
                    self.returncode = 0
            self.thread = threading.Thread(target=native_agent)
            self.thread.start()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.thread.join(timeout)
            return self.returncode

    monkeypatch.setattr(module.subprocess, 'Popen', HarborProcess)
    try:
        first = episode.start()
        assert first == {'done': False, 'body': observation, 'reward': 0.}
        action = {'action_type': 'joint_position', 'action': [0.2], 'repeat': 3, 'summary': 'unchanged'}
        final = episode.step(action)
        assert final['done'] is True and final['reward'] == 1
        assert observed['status'] == 200 and 'error' not in observed
        payload = json.loads(observed['response'].split('\n')[0].removeprefix('data: '))
        content = payload['response']['output'][0]['content'][0]['text']
        assert json.loads(content) == action
    finally:
        episode.close()
    if instrument:
        report = summary(episode)
        assert report['after_close']['harbor_returncode'] == 0
        assert report['last_response']['http_status'] == 200
        assert report['last_response']['request_seconds'] >= 0
        names = [event['event'] for event in events(episode)]
        assert 'http_request_started' in names and 'actions_get' in names
        raw = json.dumps(report)+json.dumps(events(episode))
        assert episode.token not in raw
        assert 'not retained in diagnostics' not in raw and 'Authorization' not in raw
        assert not (episode.root/'teacher-transport-diagnostics.json').stat().st_mode & 0o077
    else:
        assert not (episode.root/'teacher-transport-diagnostics.json').exists()


def test_broken_pipe_records_timing_and_rethrows_same_exception(tmp_path):
    module, _, _, row, config = configured(tmp_path)
    episode = module.HarborEpisode(row, config, 'broken-pipe')
    failure = BrokenPipeError(errno.EPIPE, 'peer went away')
    class Writer:
        def write(self, data):
            raise failure
    writer = diagnostic._ObservedWriter(Writer(), episode.teacher_lifecycle)
    with pytest.raises(BrokenPipeError) as caught:
        writer.write(b'unchanged response')
    assert caught.value is failure
    record = summary(episode)['last_response_failure']
    assert record['errno'] == errno.EPIPE and record['bytes_attempted'] == 18
    assert record['write_seconds'] >= 0


def test_unfinalized_job_summary_stays_infrastructure_failure(tmp_path):
    module, _, _, row, config = configured(tmp_path)
    episode = module.HarborEpisode(row, config, 'missing-trial')
    episode.process = SimpleNamespace(pid=71, returncode=143, poll=lambda: 143)
    episode.calls = 1
    job = episode.root/'jobs/missing-trial'
    job.mkdir(parents=True)
    (job/'result.json').write_text('{"finished_at":null,"n_completed_trials":0}')
    with pytest.raises(module.NativeInfrastructureError, match='Missing or ambiguous native trial result'):
        episode.finish()
    before = summary(episode)['before_finish']
    assert before['harbor_returncode'] == 143
    assert before['job_summary_exists'] is True
    assert before['native_trial_result_paths'] == []


def test_cancel_origin_and_trial_exception_survive_before_archive(tmp_path):
    module, rpc, _, row, config = configured(tmp_path)
    transport = tmp_path/'transport-uid'
    transport.mkdir()
    (transport/'cancel.json').write_text('{"closed_unix":100}')
    (transport/'infrastructure-error.json').write_text('{"error":"connection lost"}')
    episode = rpc.serve_episode(transport, lambda: module.HarborEpisode(row, config, 'cancelled'), {})
    episode.process = SimpleNamespace(pid=72, poll=lambda: 0)
    trial = episode.root/'jobs/cancelled/trial'
    trial.mkdir(parents=True)
    (trial/'exception.txt').write_text('server error Bearer private-token sk-private-test-key glpat-private-token')
    episode.close()
    (trial/'exception.txt').unlink()  # The native archiver can now prune it.
    report = summary(episode)
    assert report['transport_episode_id'] == 'transport-uid'
    assert report['before_close']['cancellation_reason'] == 'client_closed'
    assert report['after_close']['harbor_returncode'] == 0
    text = report['after_close']['last_trial_exception']['tail']
    assert text.count('[REDACTED]') == 3 and 'private-token' not in text


def test_failed_diagnostic_write_never_masks_native_exception(tmp_path, monkeypatch):
    module, _, _, row, config = configured(tmp_path)
    episode = module.HarborEpisode(row, config, 'full-disk')
    def full_disk():
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(episode.teacher_lifecycle, '_persist', full_disk)
    with pytest.raises(module.NativeInfrastructureError, match='Missing or ambiguous native trial result'):
        episode.finish()
    assert episode.teacher_lifecycle.state['diagnostic_write_errors'] > 0


def test_unreadable_cancel_signal_does_not_hide_returncode_or_missing_trial(tmp_path):
    module, rpc, _, row, config = configured(tmp_path)
    transport = tmp_path/'transport-uid'
    transport.mkdir()
    (transport/'cancel.json').write_text('')
    episode = rpc.serve_episode(transport, lambda: module.HarborEpisode(row, config, 'bad-cancel'), {})
    episode.process = SimpleNamespace(pid=72, poll=lambda: 137)
    episode.close()
    after = summary(episode)['after_close']
    assert after['harbor_returncode'] == 137
    assert after['last_trial_exception'] is None
    assert after['transport_signals']['cancel.json']['read_error']['error_type'] == 'JSONDecodeError'


def test_instrumentation_is_explicit_idempotent_and_local_to_teacher_module(tmp_path):
    module, rpc, original, _, _ = configured(tmp_path, instrument=False)
    other = bridge_module()
    diagnostic.install(module, rpc)
    installed = module.HarborEpisode
    diagnostic.install(module, rpc)
    assert module.HarborEpisode is installed and installed is not original
    assert other.ThreadingHTTPServer is ThreadingHTTPServer
    assert other.HarborEpisode is not installed


def test_stage_preserves_frozen_recipe_sources_and_never_submits(tmp_path):
    source = tmp_path/'original'
    source.mkdir()
    spec = {'owner_model_workflow': stage.OWNER, 'teacher_profile_id': 'astra62-xhigh-v1',
            'teacher_native_entry': 'efe_pear_primary_native.py', 'teacher_native_only': True,
            'allocation_job_entry': 'efe_astra_wide_worker.py', 'teacher_root': 'separate-teacher',
            'max_calls': 96, 'wall_seconds': 1500, 'teacher_plan': {'scene': 'train'},
            'native_config': {'action_config': {'joint_only': True}}, 'transport_root': 'preserved'}
    for name in ['efe_pear_remote_worker.py', 'efe_pear_bridge.py', 'efe_pear_file_rpc.py',
                 'efe_astra_wide_worker.py', 'efe_pear_allocation_keeper.py']:
        (source/name).write_text('# pinned '+name+'\n')
    (source/'spec.json').write_text(json.dumps(spec))
    (source/'workflow.yaml').write_text(yaml.safe_dump({'workflow': {'name': 'original', 'tasks': [
        {'name': 'native', 'credentials': {'secret-reference': {'ENV': 'field'}}}], 'resources': {'default': {'gpu': 8}}}}))
    stage.freeze_worker(source, [p.name for p in source.iterdir()])
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    build = stage.prepare(source, build_root=tmp_path/'builds')
    changed = json.loads((build/'spec.json').read_text())
    assert {k: v for k, v in changed.items() if k != 'teacher_transport_diagnostics'} == spec
    assert (build/'efe_astra_original_remote_worker.py').read_bytes() == before['efe_pear_remote_worker.py']
    for name in ['efe_pear_bridge.py', 'efe_pear_file_rpc.py', 'efe_pear_allocation_keeper.py']:
        assert (build/name).read_bytes() == before[name]
    assert yaml.safe_load((build/'workflow.yaml').read_text())['workflow']['tasks'] == \
        yaml.safe_load(before['workflow.yaml'])['workflow']['tasks']
    assert stage.prepare(source, build_root=tmp_path/'builds') == build
    assert not (build/'submission.json').exists()
    assert not (build/'submission-intent.json').exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.parametrize('change', [
    {'owner_model_workflow': 'hero'}, {'teacher_profile_id': 'legacy-high'},
    {'teacher_native_entry': 'efe_pear_primary_rtx.py'}, {'teacher_native_only': False}])
def test_stage_rejects_non_teacher_or_unexpected_worker_before_writing(tmp_path, change):
    source = tmp_path/'source'
    source.mkdir()
    (source/'spec.json').write_text(json.dumps({'owner_model_workflow': stage.OWNER,
        'teacher_profile_id': 'astra62-xhigh-v1', 'teacher_native_entry': 'efe_pear_primary_native.py',
        'teacher_native_only': True, **change}))
    with pytest.raises(ValueError, match='independently owned'):
        stage.prepare(source, build_root=tmp_path/'must-not-exist')
    assert not (tmp_path/'must-not-exist').exists()
