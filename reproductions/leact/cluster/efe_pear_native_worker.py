"""Run one native eight-episode PEAR group without optimizer updates."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import os
import secrets
import signal
import subprocess
import time
import urllib.request

try:
    from .efe_pear_sampler import digest_body, write
    from .efe_pear_loss import validate_sampling
    from .efe_pear_groups import admit_group
except ImportError:
    from efe_pear_sampler import digest_body, write
    from efe_pear_loss import validate_sampling
    from efe_pear_groups import admit_group


def capture_episode(trial, spec, archive, samples, membership, scene):
    """Join every native request to its original sampler tokens, including failures."""
    from export_efe_viewer import export_episode
    task = trial['task']
    episode_id = os.environ['EFE_WORKFLOW_ID']+'/'+trial['job']
    if trial['status'] != 'graded':
        raise ValueError('Ungraded native episode; preserve it and stop the group')
    rewards = trial['resource_metrics']
    if rewards.get('success') not in (0,1) or rewards.get('resource_enforced') != 1 or any(
        rewards.get(k) for k in ['instance_mismatch','infra_error','grader_error','no_verdict',
            'sim_error','sim_crashed_on_call','resource_monitor_missing','resource_switch_error']):
        raise ValueError('Native grade/resource verification did not pass')
    directory = Path(trial['archive'])
    grades = list(directory.glob('*/result.json'))
    if len(grades) != 1:
        raise ValueError('Missing unique original native grade')
    native = json.loads(grades[0].read_text())
    if (hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest() != trial['trial_result_sha256']
        or native.get('verifier_result', {}).get('rewards') != rewards
        or trial['score'] != rewards['success']):
        raise ValueError('Original native grade differs from trial receipt')
    logs = list(directory.glob('*/verifier/episode.jsonl'))
    if len(logs) != 1:
        raise ValueError('Missing unique native episode log')
    events = [json.loads(line) for line in logs[0].read_text().splitlines() if line.strip()]
    starts = [event['state_sha'] for event in events if event.get('ev')=='start']
    if len(starts) != 1 or {'scene_sha256':scene['scene_sha256'], 'initial_state_sha256':starts[0]} not in membership['training_tasks'][task]:
        raise ValueError('Native scene/reset is outside frozen training membership')
    paths = sorted((archive/'requests'/trial['job']).glob('*.json'))
    if len(paths) != trial['model_requests'] or not paths:
        raise ValueError('Incomplete native request archive')
    records = sorted([(p,json.loads(p.read_text())) for p in paths], key=lambda r:r[1]['started_utc'])
    captures, generated, model_seconds, truncated = [], 0, 0., False
    for path, request in records:
        if request.get('http_status') != 200 or request.get('status') != 'completed' or request.get('error'):
            raise ValueError('Native request has missing or failed sampler transport')
        ids = [item['id'].removeprefix('msg_') for item in request['public_output']
               if item.get('type')=='message' and item.get('id','').startswith('msg_pear_')]
        if len(ids) != 1 or ids[0] in {r['id'] for r in captures}:
            raise ValueError('Missing or duplicate response-to-sampler join')
        sample_path = samples/(ids[0]+'.json.gz')
        record = json.loads(gzip.decompress(sample_path.read_bytes()))
        body = json.loads(gzip.decompress(path.with_name(request['id']+'.request.json.gz').read_bytes()))
        if record['request_sha256'] != digest_body(body) or record['sampler_revision'] != spec['student_revision']:
            raise ValueError('Sampler input or policy revision differs from native request')
        count = validate_sampling(record, spec['pear']['vocab_size'], spec['pear']['capture_top_k'])
        if count != request['usage']['output_tokens']:
            raise ValueError('Native token accounting differs from exact generation capture')
        generated += count
        model_seconds += record['model_seconds']
        truncated |= record['finish_reason'] == 'length'
        captures.append({'id':ids[0], 'file':str(sample_path),
            'sha256':hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            'tokens':count, 'request_record':str(path),
            'request_record_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    public = archive/'public-episodes';public.mkdir(exist_ok=True)
    exported = export_episode({**trial, 'role':'pear-native-qualification-'+trial['job']}, public)
    return {'episode_id':episode_id, 'task':task, 'suite':trial['suite'], 'partition':'train_candidates',
        'evaluation_only':False, 'status':'graded', 'reward':trial['score'],
        'scene_sha256':scene['scene_sha256'], 'initial_state_sha256':starts[0],
        'sampler_revision':spec['student_revision'], 'sampling_capture_complete':True,
        'generated_tokens':generated, 'model_seconds':model_seconds, 'model_calls':len(records),
        'output_truncated':truncated, 'native_events_verified':True,
        'native_event_sha256':hashlib.sha256(logs[0].read_bytes()).hexdigest(),
        'native_result_sha256':trial['trial_result_sha256'], 'captures':captures,
        'public_episode':exported, 'action_metrics':trial.get('action_metrics',{}),
        'role':spec['pear']['role'], 'optimizer_updates':0}


def main():
    import efe_native_pilot as runtime
    from efe_action_artifacts import retain_trials
    from scene_identity import scene_identity
    spec = json.loads(runtime.SPEC.read_text())
    lane = runtime.SPEC.parent
    membership_path = lane/'training-membership.json'
    if hashlib.sha256(membership_path.read_bytes()).hexdigest() != spec['pear']['membership_sha256']:
        raise ValueError('Frozen training membership checksum changed')
    membership = json.loads(membership_path.read_text())
    if len(spec['tasks']) != 8 or len({r['task'] for r in spec['tasks']}) != 1:
        raise ValueError('Native qualification requires exactly one complete group')
    if any(r['partition'] != 'train_candidates' or r['task'] not in membership['training_tasks'] for r in spec['tasks']):
        raise ValueError('Native group includes a task outside training membership')
    archive = Path(spec['archive_root'])/runtime.WORKFLOW
    samples = archive/'sampler'
    archive.mkdir(parents=True, exist_ok=True);(archive/'logs').mkdir(exist_ok=True)
    runtime.ARCHIVE = archive
    episodes, scenes = {}, {}
    original_write, original_setup = runtime.write, runtime.setup
    def setup(value):
        result = original_setup(value)
        for row in {r['task']:r for r in spec['tasks']}.values():
            scene = scene_identity(result[0], row)
            if scene['scene_sha256'] not in {x['scene_sha256'] for x in membership['training_tasks'][row['task']]}:
                raise ValueError('Live native scene differs from admitted warmstart')
            scenes[row['task']] = scene
        write(archive/'scene-receipts.json', scenes)
        return result
    def recorded_write(name, value):
        original_write(name, value)
        if name == 'results.json':
            retain_trials(archive, spec, value, runtime.WORKFLOW)
            for trial in value.get('trials', []):
                if trial['job'] not in episodes:
                    episodes[trial['job']] = capture_episode(trial, spec, archive, samples, membership, scenes[trial['task']])
            write(archive/'pear-episodes.json', {'episodes':list(episodes.values()), 'optimizer_updates':0})
            write(archive/'public-episodes/manifest.json', {'episodes':[r['public_episode'] for r in episodes.values()],
                'partition':'train_candidates', 'role':spec['pear']['role'], 'planned':8})
            print('PEAR_NATIVE_PROGRESS '+json.dumps({'workflow':runtime.WORKFLOW, 'graded':len(episodes),
                'successes':sum(r['reward'] for r in episodes.values()),
                'generated_tokens':sum(r['generated_tokens'] for r in episodes.values()),
                'optimizer_updates':0}), flush=True)
    runtime.setup, runtime.write = setup, recorded_write
    # This token is node-local only. It never leaves the allocation.
    os.environ['NVIDIA_INFERENCE_API_KEY'] = secrets.token_urlsafe(36)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7',
           'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'OMP_NUM_THREADS':'2', 'NCCL_P2P_DISABLE':'1'}
    env.pop('NCCL_ALGO', None);env.pop('NCCL_PROTO', None)
    log = (archive/'logs/pear-sampler.log').open('w')
    server = subprocess.Popen([os.environ['EFE_SERVING_PYTHON'], '-u', str(lane/'efe_pear_sampler.py'),
        '--spec', str(runtime.SPEC), '--root', str(samples)], env=env,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(archive/'worker-start.json', {'workflow':runtime.WORKFLOW, 'planned_rollouts':8,
        'role':spec['pear']['role'], 'optimizer_updates':0, 'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        for _ in range(240):
            if server.poll() is not None:
                raise RuntimeError('PEAR sampler exited during startup; inspect logs/pear-sampler.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:19880/health', timeout=5) as reply:
                    if reply.status == 200:break
            except Exception:
                time.sleep(5)
        else:
            raise TimeoutError('PEAR sampler startup exceeded 20 minutes')
        sampler_ready = json.loads((samples/'ready.json').read_text())
        if sampler_ready['revision'] != spec['student_revision']:
            raise ValueError('Sampler loaded an unexpected policy')
        spec['pear']['vocab_size'] = sampler_ready['vocab_size']
        module_spec = importlib.util.spec_from_file_location('efe_pear_pilot', lane/'pilot.py')
        pilot = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(pilot)
        pilot.main()
        result = json.loads((archive/'results.json').read_text())
        if result.get('setup_error') or len(episodes) != 8:
            raise ValueError('Native group incomplete: '+str(result.get('setup_error')))
        group = admit_group(list(episodes.values()), membership, spec['student_revision'])
        result = {'success':True, 'phase':'native_group_qualified', 'group':group,
            'sampler_revision':spec['student_revision'], 'role':spec['pear']['role'],
            'optimizer_updates':0, 'native_rollouts':8,
            'note':'Base-policy integration qualification only. Primary RL arms still require the common final warmstart.',
            'completed_utc':datetime.now(timezone.utc).isoformat()}
        write(archive/'pear-native-result.json', result)
        print('PEAR_NATIVE_RESULT '+json.dumps(result), flush=True)
    except Exception as error:
        result = {'success':False, 'error':repr(error), 'native_rollouts':len(episodes), 'optimizer_updates':0,
                  'completed_utc':datetime.now(timezone.utc).isoformat()}
        write(archive/'pear-native-error.json', result)
        print('PEAR_NATIVE_RESULT '+json.dumps(result), flush=True)
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL);server.wait()
        log.close()


if __name__ == '__main__':
    main()
