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
import math
import os
from pathlib import Path
import re
import threading
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
LEGACY_PROFILE='astra6-high-v1'
ASTRA62_PROFILE='astra62-xhigh-v1'
PROFILES={
    LEGACY_PROFILE:{'id':LEGACY_PROFILE,'model':MODEL,'api_mode':'responses','effort':'high'},
    ASTRA62_PROFILE:{'id':ASTRA62_PROFILE,'model':'openai/openai/gpt-6.2-astra',
                     'api_mode':'chat_completions','effort':'xhigh'},
}
PROMPT = SYSTEM.replace('Return only one JSON object with action_type',
    'End your response with one json code block containing action_type') + (
    '\nGround your action in the supplied public camera, observation and action guide. '
    'The summary should briefly explain the observable reason for your action. '
    'Do not provide private deliberation or hidden reasoning.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def model_profile(spec):
    identifier=spec.get('teacher_profile_id',LEGACY_PROFILE)
    if identifier not in PROFILES:raise ValueError('Unapproved teacher model profile')
    return dict(PROFILES[identifier])


def request_body(profile, *, prompt, inputs, max_output_tokens):
    if profile['api_mode']=='responses':
        return {'model':profile['model'],'instructions':prompt,'input':inputs,
            'stream':False,'store':False,'max_output_tokens':max_output_tokens,
            'reasoning':{'effort':profile['effort']}}
    messages=[{'role':'system','content':prompt}]
    for item in inputs:
        content=item['content']
        if isinstance(content,list):
            converted=[]
            for block in content:
                if block['type']=='input_text':converted.append({'type':'text','text':block['text']})
                elif block['type']=='input_image':converted.append({'type':'image_url',
                    'image_url':{'url':block['image_url'],'detail':block.get('detail','auto')}})
                else:raise ValueError('Unsupported public chat input block')
            content=converted
        messages.append({'role':item['role'],'content':content})
    return {'model':profile['model'],'messages':messages,'stream':False,'store':False,
        'max_completion_tokens':max_output_tokens,'reasoning_effort':profile['effort']}


def collection_plan(curriculum, *, attempts_per_task=2, teacher_profile_id=LEGACY_PROFILE):
    profile=model_profile({'teacher_profile_id':teacher_profile_id})
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
            if suite == 'procedural_reasoning' and attempt == 1 and len(rows) > 1:
                # The pointing generator alternates points and boxes by seed.
                index = next((i for i,r in enumerate(rows)
                    if r['env_args']['seed'] % 2 != rows[0]['env_args']['seed'] % 2), index)
            row = rows[index]
            case = {**row, 'collection_attempt': attempt,
                    'distinct_training_resets_available': len(rows),
                    'repeated_reset': attempt > 0 and index == 0}
            case['case_id'] = 'rp_jhh_' + digest([suite, task, row['config_id'], attempt])[:24]
            cases.append(case)
    return {'schema_version': 1, 'teacher_model':profile['model'],'teacher_profile':profile,'partition':'train',
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
    chat='messages'in body
    request = urllib.request.Request(endpoint.rstrip('/') + ('/chat/completions'if chat else'/responses'),
        data=json.dumps(body, allow_nan=False).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    # Store only public output, usage and identifiers, never the response's
    # provider-private reasoning entries, encrypted contents or key material.
    with urllib.request.urlopen(request, timeout=900) as stream:
        result = json.load(stream)
    raw_usage=result.get('usage',{})
    if chat:
        choices=result.get('choices',[])
        if len(choices)!=1 or choices[0].get('finish_reason')!='stop':
            raise RuntimeError('Teacher chat response did not complete: '+str([c.get('finish_reason')for c in choices]))
        content=choices[0].get('message',{}).get('content')
        text=content if isinstance(content,str)else'\n'.join(b['text']for b in(content or[])if b.get('type')=='text')
        usage={'input_tokens':raw_usage.get('prompt_tokens',0),'output_tokens':raw_usage.get('completion_tokens',0),
               'total_tokens':raw_usage.get('total_tokens',0),
               'input_tokens_details':raw_usage.get('prompt_tokens_details',{}),
               'output_tokens_details':raw_usage.get('completion_tokens_details',{})}
    else:
        if result.get('status')!='completed':
            raise RuntimeError('Teacher response did not complete: '+str(result.get('status')))
        text=public_text(result);usage=raw_usage
    if not text:
        raise RuntimeError('Teacher response has no public action text')
    return {'public_output':text,'usage':usage,'provider_usage':raw_usage,
            'returned_model': result.get('model'), 'response_id': result.get('id'),
            'status':'completed','requested_model':body['model'],
            'requested_effort':body.get('reasoning_effort',body.get('reasoning',{}).get('effort')),
            'api_mode':'chat_completions'if chat else'responses','fallback_allowed':False}


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


def reasoning_prompt(schema):
    kind=schema['answer_type'];example={'answer_type':kind,'summary':'A short explanation grounded in the image.'}
    if kind=='bounding_box':example['bbox']=[.1,.2,.3,.4]
    elif kind=='point_set_count':example.update(points=[[.2,.3],[.6,.7]],count=2)
    else:example['point']=[.5,.5]
    return ('Solve the visual reasoning task using the supplied images, instruction, public camera guide '
        'and answer_schema. You may reason before answering. End each response with exactly one json code '
        'block. Copy the answer_type field exactly from answer_schema and return its other required fields '
        'at the top level. Do not nest the action under answer. Use normalized image coordinates with u increasing '
        'right and v down. If permitted, inspect a camera with {"action_type":"view","view":"overhead"} '
        'or another listed view before answering. Include a short summary. Every final answer is checked by '
        'the scene verifier. No code execution or other actions are available.\n'
        'Format example only; choose your own coordinates/count from the actual image:\n```json\n'+
        json.dumps(example)+'\n```')


def reasoning_action(payload, *, family, actions):
    """Apply the production public answer/view validation without repairing it."""
    action=json.loads(payload)
    if not isinstance(action,dict):raise ValueError('The action must be one JSON object')
    def point(value):
        if (not isinstance(value,list)or len(value)!=2 or
            any(type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<=1 for v in value)):
            raise ValueError('A point must contain two finite normalized image coordinates')
    if action.get('action_type')=='view':
        if family!='active_perception_memory':raise ValueError('This task does not permit camera actions')
        if sum(a.get('action_type')=='view'for a in actions)>=3:raise ValueError('The camera-action budget is exhausted')
        if action.get('view')not in {'front','side','overhead'}:raise ValueError('Unknown camera view')
    elif 'bbox'in action:
        box=action['bbox']
        if not isinstance(box,list)or len(box)!=4:raise ValueError('A box needs four coordinates')
        point(box[:2]);point(box[2:])
        if box[0]>=box[2]or box[1]>=box[3]:raise ValueError('Bounding box coordinates are inverted')
    elif 'points'in action:
        if not isinstance(action['points'],list)or type(action.get('count'))is not int or action['count']<0:
            raise ValueError('Point sets require a list of points and a nonnegative integer count')
        for value in action['points']:point(value)
    else:point(action['point'])
    return action


def previous_attempt(folder, spec, *, now=None):
    """Preserve terminal and interrupted evidence under its original attempt ID.

    The caller holds the case writer lock. A restart creates the next infra
    directory rather than appending a different episode into an existing trace.
    """
    folder=Path(folder);path=folder/'result.json'
    if not path.exists():return None
    old=json.loads(path.read_text())
    if old.get('status')in {'opening','running'}:
        old.update(status='infrastructure_failure',error_type='InterruptedCollector',
                   error='Previous collector stopped before a finalized native outcome.',
                   interrupted_recovery_unix=time.time()if now is None else now)
        atomic_json(path,old)
        uid=old.get('episode_id')
        if uid and old['case']['suite']!='procedural_reasoning':
            # Cancel only this recorded collection episode, never a worker or
            # allocation. Native transport preserves its prior evidence.
            episode=Path(spec['transport_root'])/'episodes'/uid
            if episode.is_dir():atomic_json(episode/'cancel.json',{'recovered_interrupted_attempt':True})
    return old


class ReasoningClient:
    """Use only the admitted primitive scene's public observation and verifier."""
    def __init__(self,case,folder):
        from uuid import uuid4
        try:from .efe_pear_reasoning_scenes import SceneEpisode
        except ImportError:from efe_pear_reasoning_scenes import SceneEpisode
        self.uid=uuid4().hex;self.folder=folder
        self.scene=SceneEpisode(**case['env_args'])
        if self.scene.scene_sha256!=case['scene_sha256']:
            self.scene.close();raise ValueError('Independent reasoning training scene changed')
    def result(self,observation,reward=0.,done=False):
        schema=observation['public']['answer_schema'] if observation else self.schema
        self.schema=schema;blocks=[]
        if observation:
            public=observation['public'];blocks=[{'type':'text','text':json.dumps(public)+'\nEnd with one json code block.'}]
            for name,image in zip(public['sensor_order'],observation['images'],strict=True):
                blocks.extend([{'type':'text','text':'Current camera: '+name},{'type':'image','image':image}])
        path=self.folder/'scene-verdict.json'
        if done:atomic_json(path,{'finalized':True,'success':bool(reward),'scene_sha256':self.scene.scene_sha256,
                                 'grader':'unchanged_independent_scene_verifier'})
        return {'done':done,'rewards':[float(reward)],'public_blocks':blocks,
            'scene_sha256':self.scene.scene_sha256,'action_contract':{'answer_schema':schema},
            'native_trial_path':str(path)if done else None,'executed_control_steps':0}
    def open(self):return self.result(self.scene.observation())
    def step(self,action):
        observation,reward,done=self.scene.step(action);return self.result(observation,reward,done)
    def close(self):self.scene.close()


def run_case(spec, case, *, infrastructure_attempt=0):
    reasoning=case['suite']=='procedural_reasoning'
    profile=model_profile(spec)
    folder = Path(spec['teacher_root'])/'episodes'/case['case_id']/f'infra-{infrastructure_attempt:02d}'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_path = folder/'result.json'
    old=previous_attempt(folder,spec)
    if old is not None:return old
    kind = 'native_yam/'+case['task'] if case['suite'] == 'native_gr00t_yam' else 'efe/'+case['suite']
    client=None
    result = {'case':case,'teacher_model':profile['model'],'teacher_profile':profile,
        'requested_model':profile['model'],'requested_effort':profile['effort'],
        'api_mode':profile['api_mode'],'fallback_allowed':False,'status':'opening',
        'episode_id': None, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'valid_actions': 0, 'invalid_actions': 0, 'generated_tokens': 0, 'input_tokens_sum':0,
        'maximum_input_tokens_per_call':0,
        'token_budget_definition':'Generated provider output tokens, including reported reasoning-token counts; context/input tokens recorded separately.',
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
        client=ReasoningClient(case,folder)if reasoning else FileEpisodeClient(spec['transport_root'],kind,case['env_args'],
                          timeout=14400 if case['suite']=='native_gr00t_yam'else 1800)
        result['episode_id']=client.uid;save()
        value = client.open(); result.update(status='running', opened_unix=time.time())
        if value.get('scene_sha256') and value['scene_sha256'] != case['scene_sha256']:
            raise ValueError('Teacher reset differs from the admitted training scene')
        contract = value['action_contract']; sha = digest(contract); result['interface_sha256'] = sha
        for block in value.get('public_blocks',[]):
            if block.get('type')!='text' or 'Public observation JSON:\n'not in block.get('text',''):continue
            visible=json.loads(block['text'].split('Public observation JSON:\n',1)[1])
            advertised=sorted(visible.get('spec',{}).get('action_types',{}))
            callable_types=sorted(contract.get('action_types',{}))
            result['interface_action_types_audit']={'public_spec_action_types':advertised,
                'callable_action_contract_types':callable_types,
                'advertised_but_not_callable':sorted(set(advertised)-set(callable_types)),
                'prompt_modified_for_audit':False}
            atomic_json(folder/'first-public-observation.json',visible)
            break
        prompt=reasoning_prompt(contract['answer_schema'])if reasoning else PROMPT
        result['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
        atomic_json(folder/'action-contract.json', contract); (folder/'system-prompt.txt').write_text(prompt)
        history = []; reasoning_history=[]; executed_actions=[]; started = time.monotonic()
        for turn in range(spec.get('max_calls', 96)):
            if value['done']:
                break
            if time.monotonic()-started > spec.get('wall_seconds', 1500) or result['generated_tokens'] >= 160000:
                result['end_reason'] = 'teacher_budget'
                if reasoning:break
                value = client.step({'stop': True, 'summary': 'Finalize at the declared collection budget.'})
                break
            if not reasoning and digest(value['action_contract']) != sha:
                raise ValueError('Native action contract changed within teacher episode')
            content, refs = public_observation(value, folder, turn)
            if history:
                content.insert(0, {'type': 'input_text', 'text': 'Recent public actions and feedback:\n'+json.dumps(history[-8:])})
            body=request_body(profile,prompt=prompt,inputs=reasoning_history+[{'role':'user','content':content}],
                              max_output_tokens=min(32768,160000-result['generated_tokens']))
            before = time.monotonic()
            reply = api_response(body, endpoint=spec['teacher_endpoint'], credential_file=spec['teacher_credential_file'])
            seconds = time.monotonic()-before
            result['generated_tokens'] += reply['usage'].get('output_tokens', 0)
            result['input_tokens_sum'] += reply['usage'].get('input_tokens',0)
            result['maximum_input_tokens_per_call']=max(result['maximum_input_tokens_per_call'],reply['usage'].get('input_tokens',0))
            if reasoning:
                reasoning_history.extend([{'role':'user','content':content},
                                          {'role':'assistant','content':reply['public_output']}])
            result['model_seconds'] += seconds
            row = {'turn': turn, 'observed_unix': time.time(), **reply,
                'teacher_profile':profile,'requested_model':profile['model'],'requested_effort':profile['effort'],
                'api_mode':profile['api_mode'],'fallback_allowed':False,
                'model_seconds': seconds, 'request_sha256': digest(body),
                'action_contract_sha256': sha, 'frames': refs,
                'public_observation_text': [b['text'] for b in content if b['type'] == 'input_text']}
            if reply.get('returned_model')!=profile['model']:
                row['model_identity_mismatch']=True
                with(folder/'turns.jsonl').open('a')as log:log.write(json.dumps(row)+'\n')
                raise RuntimeError('Teacher returned a different model; no fallback action executed')
            try:
                payload=final_action_json(reply['public_output'])
                action = reasoning_action(payload,family=case['env_args']['family'],actions=executed_actions)if reasoning else parse_action(payload, contract=contract, allow_stop=True)
                row['action'] = action
            except (ValueError, TypeError, KeyError) as error:
                result['invalid_actions'] += 1; row['validation_error'] = str(error)
                history.append({'validation_error': str(error), 'public_output': reply['public_output']})
                with (folder/'turns.jsonl').open('a') as log: log.write(json.dumps(row)+'\n')
                save(); continue
            before = time.monotonic(); value = client.step(action)
            executed_actions.append(action)
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
            if reasoning:
                result.update(status='ungraded_budget',end_reason=result.get('end_reason','teacher_call_budget'),
                              finalized_native_outcome=False,native_success_candidate=False)
                return result
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
        if client is not None:client.close()
        result['completed_utc'] = datetime.now(timezone.utc).isoformat(); save()
    return result


class LegacyCaseGuards:
    """Let existing legacy episodes finish, then retain their simulator lanes.

    Legacy collectors already skip a case when its writer lock is held. Taking
    every unstarted legacy case lock prevents another old-profile episode;
    waiting on existing locks respects each in-flight native episode boundary.
    """
    def __init__(self,spec,cases):
        self.spec=spec;self.handles={};self.pending={};self.mutex=threading.Lock()
        root=spec.get('legacy_teacher_root')
        if not root:return
        if Path(root).resolve()==Path(spec['teacher_root']).resolve():
            raise ValueError('A model-profile handoff needs a separate output directory')
        for case in cases:
            path=Path(root)/'episodes'/case['case_id']/'writer.lock';path.parent.mkdir(parents=True,exist_ok=True)
            handle=path.open('a');self.handles[path]=handle
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:self.pending.setdefault(case['task'],[]).append(path)
            else:self.preserve_orphans(path)
        self.save()
    def preserve_orphans(self,path):
        # Holding the legacy case's exclusive writer lock proves no legacy
        # collector still owns an unfinished record. Preserve it as interrupted
        # and close only its orphaned RPC, never an allocated native server.
        for result in path.parent.glob('infra-*/result.json'):
            previous_attempt(result.parent,self.spec)
    def save(self):
        root=Path(self.spec['teacher_root'])/'profile-handoffs';root.mkdir(mode=0o700,parents=True,exist_ok=True)
        atomic_json(root/(os.environ.get('EFE_WORKFLOW_ID','local')+'.json'),{
            'profile':model_profile(self.spec),'legacy_root':self.spec.get('legacy_teacher_root'),
            'legacy_inflight_tasks':sorted(k for k,v in self.pending.items()if v),
            'guarded_legacy_cases':len(self.handles)-sum(map(len,self.pending.values())),
            'updated_unix':time.time(),'pid':os.getpid(),'native_processes_preserved':True})
    def wait_task(self,name):
        while self.pending.get(name):
            with self.mutex:
                for path in list(self.pending[name]):
                    try:fcntl.flock(self.handles[path],fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        # A queued legacy open with no native owner can be
                        # retired atomically. Winning the pending-file rename
                        # proves no simulator episode was interrupted; a native
                        # claim that wins the race is left to finish normally.
                        for result_path in path.parent.glob('infra-*/result.json'):
                            result=json.loads(result_path.read_text());uid=result.get('episode_id')
                            if result.get('status')!='opening'or not uid or not re.fullmatch('[0-9a-f]{32}',uid):continue
                            episode=Path(self.spec['transport_root'])/'episodes'/uid
                            pending=Path(self.spec['transport_root'])/'pending'/(uid+'.json')
                            try:pending.rename(episode/'profile-superseded-pending.json')
                            except FileNotFoundError:continue
                            atomic_json(episode/'infrastructure-error.json',{
                                'error':'Legacy profile open superseded before native claim at authorized model handoff.',
                                'superseded_before_native_claim':True,'updated_unix':time.time()})
                        continue
                    self.pending[name].remove(path)
                    self.preserve_orphans(path)
                self.save()
            if self.pending[name]:time.sleep(1)
    def close(self):
        for handle in self.handles.values():handle.close()


def collect(spec, *, legacy_guards=None):
    names = list(dict.fromkeys(spec['teacher_task_names']))
    cases = [r for r in spec['teacher_plan']['cases'] if r['task'] in names]
    def one_task(name):
        if legacy_guards is not None:legacy_guards.wait_task(name)
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

