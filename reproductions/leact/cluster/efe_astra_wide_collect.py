"""Separate, resumable Astra demonstrations on the admitted PEAR training scenes.

This collector never writes learner queues or changes a native action. Only
provider-public output text is retained; hidden reasoning payloads are dropped.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

try:
    from .efe_action_loop import SYSTEM, parse_action
    from .efe_pear_action_json import final_action_json
    from .efe_pear_file_rpc import FileEpisodeClient, atomic_json
except ImportError:
    from efe_action_loop import SYSTEM, parse_action
    from efe_pear_action_json import final_action_json
    from efe_pear_file_rpc import FileEpisodeClient, atomic_json


MODEL = 'azure/openai/gpt-6-astra'
PROMPT = SYSTEM.replace('Return only one JSON object with action_type',
    'End your response with one json code block containing action_type') + (
    '\nGround your action in the supplied public camera, observation and action guide. '
    'The summary should briefly explain the observable reason for your action. '
    'Do not provide private deliberation or hidden reasoning.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def collection_plan(curriculum, *, attempts_per_task=2):
    if curriculum.get('partition') != 'train' or curriculum.get('qualification') != 'passed':
        raise ValueError('A qualified training curriculum is required')
    if type(attempts_per_task) is not int or attempts_per_task < 1:
        raise ValueError('A positive attempt count is required')
    grouped = {}
    for row in curriculum['configs']:
        if (row.get('partition') != 'train' or row.get('admitted') is not True
            or row.get('already_successful') or not row.get('observation_complete')
            or not row.get('scene_sha256') or not row.get('admission_receipt_sha256')):
            raise ValueError('Unqualified or protected teacher configuration')
        grouped.setdefault((row['suite'], row['task']), []).append(row)
    cases = []
    for (suite, task), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['config_id'])
        for attempt in range(attempts_per_task):
            # Multiple attempts on a single frozen EFE reset remain repeats.
            index = min(len(rows)-1, attempt*len(rows)//attempts_per_task)
            row = rows[index]
            case = {**row, 'collection_attempt': attempt,
                    'distinct_training_resets_available': len(rows),
                    'repeated_reset': attempt > 0 and index == 0}
            case['case_id'] = 'rp_jhh_' + digest([suite, task, row['config_id'], attempt])[:24]
            cases.append(case)
    return {'schema_version': 1, 'teacher_model': MODEL, 'partition': 'train',
            'curriculum_sha256': digest(curriculum), 'attempts_per_task': attempts_per_task,
            'cases': cases, 'task_count': len(grouped),
            'admission': 'Only finalized native successes with complete public responses, '
                         'unchanged action contracts and training-scene provenance are admitted.'}


def public_text(response):
    return '\n'.join(c['text'] for item in response.get('output', [])
        if item.get('type') == 'message' and item.get('channel') not in {'analysis', 'reasoning'}
        for c in item.get('content', []) if c.get('type') == 'output_text')


def api_response(body, *, endpoint, credential_file):
    key_path = Path(credential_file)
    if key_path.stat().st_mode & 0o077:
        raise ValueError('Teacher inference credential must be private')
    key = key_path.read_text().strip()
    request = urllib.request.Request(endpoint.rstrip('/') + '/responses',
        data=json.dumps(body, allow_nan=False).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # Store only public output, usage and identifiers, never the response's
    # provider-private reasoning entries, encrypted contents or key material.
    with urllib.request.urlopen(request, timeout=900) as stream:
        result = json.load(stream)
    if result.get('status') != 'completed':
        raise RuntimeError('Teacher response did not complete: ' + str(result.get('status')))
    text = public_text(result)
    if not text:
        raise RuntimeError('Teacher response has no public action text')
    return {'public_output': text, 'usage': result.get('usage', {}),
            'returned_model': result.get('model'), 'response_id': result.get('id'),
            'status': result.get('status')}


def public_observation(value, folder, turn):
    image_dir = folder/'frames'/f'{turn:04d}'
    image_dir.mkdir(parents=True, exist_ok=True)
    content, refs = [], []
    for block in value['public_blocks']:
        if block['type'] == 'text':
            content.append({'type': 'input_text', 'text': block['text']})
        elif block['type'] == 'image':
            name = f'view-{len(refs):02d}.png'; path = image_dir/name
            block['image'].convert('RGB').save(path)
            raw = path.read_bytes()
            refs.append({'file': str(path.relative_to(folder)),
                         'sha256': hashlib.sha256(raw).hexdigest()})
            content.append({'type': 'input_image', 'detail': 'auto',
                'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode()})
        else:
            raise ValueError('Unexpected observation block')
    return content, refs


def run_case(spec, case, *, infrastructure_attempt=0):
    if case['suite'] == 'procedural_reasoning':
        raise ValueError('Procedural scenes require their separately identified collector')
    folder = Path(spec['teacher_root'])/'episodes'/case['case_id']/f'infra-{infrastructure_attempt:02d}'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path = folder/'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get('status') in {'graded_success', 'graded_failure'}:
            return old
    kind = 'native_yam/'+case['task'] if case['suite'] == 'native_gr00t_yam' else 'efe/'+case['suite']
    client = FileEpisodeClient(spec['transport_root'], kind, case['env_args'],
                              timeout=14400 if case['suite'] == 'native_gr00t_yam' else 1800)
    result = {'case': case, 'teacher_model': MODEL, 'status': 'opening',
        'episode_id': client.uid, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'valid_actions': 0, 'invalid_actions': 0, 'generated_tokens': 0,
        'executed_control_steps': 0, 'model_seconds': 0., 'interface_sha256': None,
        'curriculum_sha256': spec['teacher_plan']['curriculum_sha256'],
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
        'infrastructure_attempt': infrastructure_attempt, 'sft_admitted': False,
        'source_model_job': spec.get('source_model_job'),
        'scope': 'Independent teacher collection; not learner rollouts or held-out evaluation.'}
    def save():
        result['updated_unix'] = time.time(); atomic_json(result_path, result)
    save()
    try:
        value = client.open(); result.update(status='running', opened_unix=time.time())
        if value.get('scene_sha256') and value['scene_sha256'] != case['scene_sha256']:
            raise ValueError('Teacher reset differs from the admitted training scene')
        contract = value['action_contract']; sha = digest(contract); result['interface_sha256'] = sha
        atomic_json(folder/'action-contract.json', contract); (folder/'system-prompt.txt').write_text(PROMPT)
        history = []; started = time.monotonic()
        for turn in range(spec.get('max_calls', 96)):
            if value['done']:
                break
            if time.monotonic()-started > spec.get('wall_seconds', 1500) or result['generated_tokens'] >= 160000:
                result['end_reason'] = 'teacher_budget'
                value = client.step({'stop': True, 'summary': 'Finalize at the declared collection budget.'})
                break
            if digest(value['action_contract']) != sha:
                raise ValueError('Native action contract changed within teacher episode')
            content, refs = public_observation(value, folder, turn)
            if history:
                content.insert(0, {'type': 'input_text', 'text': 'Recent public actions and feedback:\n'+json.dumps(history[-8:])})
            body = {'model': MODEL, 'instructions': PROMPT,
                'input': [{'role': 'user', 'content': content}], 'stream': False, 'store': False,
                'max_output_tokens': min(32768, 160000-result['generated_tokens']),
                'reasoning': {'effort': 'high'}}
            before = time.monotonic()
            reply = api_response(body, endpoint=spec['teacher_endpoint'], credential_file=spec['teacher_credential_file'])
            seconds = time.monotonic()-before
            result['generated_tokens'] += reply['usage'].get('output_tokens', 0)
            result['model_seconds'] += seconds
            row = {'turn': turn, 'observed_unix': time.time(), **reply,
                'model_seconds': seconds, 'request_sha256': digest(body),
                'action_contract_sha256': sha, 'frames': refs,
                'public_observation_text': [b['text'] for b in content if b['type'] == 'input_text']}
            try:
                action = parse_action(final_action_json(reply['public_output']), contract=contract, allow_stop=True)
                row['action'] = action
            except (ValueError, TypeError, KeyError) as error:
                result['invalid_actions'] += 1; row['validation_error'] = str(error)
                history.append({'validation_error': str(error), 'public_output': reply['public_output']})
                with (folder/'turns.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
                save(); continue
            before = time.monotonic(); value = client.step(action)
            result['valid_actions'] += int(not action.get('stop'))
            executed = value.get('executed_control_steps') or 0
            result['executed_control_steps'] += executed
            row.update(executed_control_steps=executed, native_seconds=time.monotonic()-before,
                       native_done=value['done'], native_reward=value['rewards'][-1])
            history.append({'action': action, 'executed_control_steps': executed,
                            'execution_completed': True})
            with (folder/'turns.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
            save()
        if not value['done']:
            value = client.step({'stop': True, 'summary': 'Finalize after the declared call budget.'})
            result['end_reason'] = 'teacher_call_budget'
        native = value.get('native_trial_path')
        if not value['done'] or value['rewards'][-1] not in [0, 1] or not native or not Path(native).is_file():
            raise RuntimeError('Teacher episode has no finalized native outcome')
        verdict = json.loads(Path(native).read_text()); atomic_json(folder/'native-verdict.json', verdict)
        success = bool(value['rewards'][-1]); result.update(status='graded_success' if success else 'graded_failure',
            native_success=success, native_trial_path=native, native_termination_terms=value.get('native_termination_terms'),
            finalized_native_outcome=True, native_success_candidate=success and result['invalid_actions'] == 0,
            qualitative_task_consistency='not_reviewed', sft_admitted=False)
    except Exception as error:
        # Never turn API/runtime failures into scored policy failures.
        result.update(status='infrastructure_failure', error_type=type(error).__name__, error=str(error)[:1200])
    finally:
        client.close(); result['completed_utc'] = datetime.now(timezone.utc).isoformat(); save()
    return result


def collect(spec):
    names = list(dict.fromkeys(spec['teacher_task_names']))
    cases = [r for r in spec['teacher_plan']['cases'] if r['task'] in names]
    def one_task(name):
        for case in [r for r in cases if r['task'] == name]:
            base = Path(spec['teacher_root'])/'episodes'/case['case_id'];base.mkdir(parents=True,exist_ok=True)
            with (base/'writer.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                for attempt in range(spec.get('max_infrastructure_attempts', 3)):
                    result = run_case(spec, case, infrastructure_attempt=attempt)
                    print('ASTRA_WIDE_EPISODE '+json.dumps({k:result.get(k)for k in
                        ['episode_id','status','native_success','valid_actions','invalid_actions','sft_admitted']}),flush=True)
                    if result['status'] != 'infrastructure_failure':break
                    time.sleep(10)
    with ThreadPoolExecutor(max_workers=spec.get('teacher_concurrency',2)) as pool:
        list(pool.map(one_task, names))

