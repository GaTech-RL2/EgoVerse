"""Streaming admission of successful strict-harness episodes into a frozen corpus."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

try:
    from .efe_action_loop import parse_action
    from .efe_training_data import chat_inputs, sha, write
except ImportError:
    from efe_action_loop import parse_action
    from efe_training_data import chat_inputs, sha, write


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def episode_states(bundle, entry, workflow, task, scene):
    """Reject incomplete evidence; never turn a failed/invalid action into a label."""
    require(sha(bundle)==entry['retained_sha256'], 'Retained bundle checksum mismatch')
    require(entry['candidate_pending_scene_and_native_event_audit'], 'Episode did not pass candidate checks')
    require(scene['upstream_digest']==task['upstream_digest'], 'Scene/task digest mismatch')
    states = []
    with tarfile.open(bundle, 'r:gz') as tar:
        def content(name):
            raw = tar.extractfile(name).read()
            require(hashlib.sha256(raw).hexdigest()==entry['files'][name]['sha256'], 'Evidence member checksum mismatch')
            return raw
        def named(suffix):
            names = [n for n in entry['files'] if n.endswith(suffix)]
            require(len(names)==1, 'Missing/ambiguous evidence: '+suffix)
            return content(names[0])
        trajectory = json.loads(named('/agent/trajectory.json'))
        contract_record = json.loads(named('/agent/action-contract.json'))
        contract = contract_record['contract']
        events = [json.loads(l) for l in named('/verifier/episode.jsonl').splitlines() if l.strip()]
        initial = next(e['state_sha'] for e in events if e.get('ev')=='start')
        require(isinstance(initial,str) and len(initial)==64, 'Missing initial-state evidence')
        if task.get('teacher_initial_state_sha256'):
            require(initial==task['teacher_initial_state_sha256'], 'Reference initial state mismatch')
        native = [json.loads(content(n)) for n in entry['files'] if n.endswith('/result.json')]
        native = [r for r in native if 'verifier_result' in r]
        require(len(native)==1, 'Ambiguous native grade')
        native = native[0]
        reward = native['verifier_result']['rewards']
        require(reward['success']==1 and reward['resource_enforced']==1 and not native.get('exception_info'),
                'Episode lacks a clean enforced native success')
        require(not any(reward.get(k) for k in ['instance_mismatch','infra_error','grader_error','no_verdict',
            'sim_error','sim_crashed_on_call','resource_monitor_missing','resource_switch_error']), 'Native verification error')
        steps = {s['step_id']:s for s in trajectory['steps']}
        for target in entry['targets']:
            if not target['complete_request_and_execution']:
                continue
            step = steps[target['step_id']]
            ident = target['request_record_id']
            stem = 'requests/'+entry['job']+'/'+ident
            record = json.loads(content(stem+'.json'))
            body = json.loads(gzip.decompress(content(stem+'.request.json.gz')))
            require(record['status']=='completed' and record['http_status']==200 and not record.get('error'),
                    'Incomplete/truncated model response')
            require(record['request_sha256']==step['request_sha256']==target['request_sha256'], 'Request/trajectory mismatch')
            action = parse_action(step['public_output'], contract=contract, allow_stop=False)
            require(len(step['tool_calls'])==1 and step['tool_calls'][0]['arguments']==action, 'Decoded command differs')
            feedback = json.loads(step['observation']['results'][0]['content'])
            require(feedback['execution_completed'] and feedback['action']==action, 'Command did not execute')
            matches = [e for e in events if e.get('ev')=='submit' and e.get('control_steps')==feedback['control_steps_after']]
            require(len(matches)==1, 'Missing/ambiguous native submit')
            event = matches[0]
            require(event['rows']==action['repeat'] and event['type']==action['action_type'], 'Native command shape/type differs')
            require(event['dt']==contract['control_dt'] and event['n_clipped_rows']==0, 'Native command timing/clipping differs')
            require(feedback['control_steps_after']-feedback['control_steps_before']==action['repeat'], 'Native executed step count differs')
            visible = json.loads(body['input'][0]['content'][0]['text'])
            require(visible['mode']=='task' and visible['action_contract']==contract, 'Observed contract differs')
            require(visible['observation']['control_steps']==feedback['control_steps_before'], 'Input is not pre-action observation')
            # JSON member order has no action semantics. Freeze executable
            # fields before free text so the LeAct physical-command token mask
            # never depends on whether a teacher printed its summary first.
            original_order=list(action)
            action={k:action[k] for k in ['action_type','action','repeat','summary']}
            state_id = hashlib.sha256((workflow+'/'+entry['job']+'/'+ident).encode()).hexdigest()
            states.append({'id':state_id, 'task':task['task'], 'suite':task['suite'],
                'episode':workflow+'/'+entry['job'], 'messages':chat_inputs(body), 'tools':[],
                'action':action, 'public_commentary':action['summary'], 'scene_sha256':scene['scene_sha256'],
                'initial_state_sha256':initial, 'source_bundle_sha256':entry['retained_sha256'],
                'source_request_sha256':entry['files'][stem+'.request.json.gz']['sha256'],
                'native_submit':event, 'action_contract_sha256':contract_record['sha256'],
                'system_prompt_sha256':contract_record['system_prompt_sha256']})
            if original_order!=list(action):states[-1]['target_key_order_normalization']={'original_order':original_order,'values_unchanged':True}
    require(states, 'No executed targets')
    return states


def prepare(plan, root):
    root = Path(root)
    output = root/'corpus'
    output.mkdir(exist_ok=True)
    if (output/'admission.json').exists():
        receipt = json.loads((output/'admission.json').read_text())
        require(sha(output/'decisions.jsonl.gz')==receipt['decisions_sha256'], 'Frozen corpus changed')
        return receipt
    audit_path = root/'scene-audit/result.json'
    audit = json.loads(audit_path.read_text())
    require(audit['complete'] and not audit['errors'], 'Incomplete protected-scene audit')
    require(audit['upstream_revision']==plan['upstream_revision'], 'Audit revision differs')
    tasks = {r['task']:r for r in plan['tasks']}
    scenes = {r['task']:r for r in audit['rows'] if r['partition']=='train_candidates'}
    excluded_tasks = {r['task'] for r in audit['collisions']}
    excluded_tasks |= {r['task'] for r in plan['tasks'] if r['suite'] in audit.get('excluded_suites',[])}
    candidates, excluded, seen = [], [], set()
    for path in sorted((root/'collection').glob('*/retained-manifest.json')):
        manifest = json.loads(path.read_text())
        spec = json.loads((path.parent/'spec.json').read_text())
        by_task = {r['task']:r for r in spec['tasks']}
        for entry in manifest['episodes']:
            if entry['status']!='graded':
                continue
            task = entry['task']
            attempt = by_task[task]['collection_attempt']
            key = (task, attempt)
            require(key not in seen, 'Duplicate graded policy attempt')
            seen.add(key)
            if task not in tasks or task in excluded_tasks or not entry['candidate_pending_scene_and_native_event_audit']:
                excluded.append({'workflow':manifest['workflow'], 'task':task, 'attempt':attempt,
                    'reason':'Protected scene or noneligible native outcome', 'native_score':entry['score']})
                continue
            candidates.append((attempt, task, manifest['workflow'], entry, by_task[task]))
    candidates.sort(key=lambda r:(r[0],hashlib.sha256(r[1].encode()).hexdigest(),r[2]))
    episodes = []
    decision_count = 0
    temporary = output/'decisions.jsonl.gz.tmp'
    with temporary.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
        for attempt, task, workflow, entry, row in candidates:
            try:
                states = episode_states(Path(entry['durable_file']),entry,workflow,row,scenes[task])
            except Exception as error:
                excluded.append({'workflow':workflow, 'task':task, 'attempt':attempt, 'reason':str(error)})
                continue
            for state in states:
                stream.write((json.dumps(state,ensure_ascii=False)+'\n').encode())
            decision_count += len(states)
            episodes.append({'task':task,'suite':row['suite'],'workflow':workflow,'job':entry['job'],
                'collection_attempt':attempt,'decisions':len(states),'scene_sha256':scenes[task]['scene_sha256'],
                'initial_state_sha256':states[0]['initial_state_sha256'],'bundle_sha256':entry['retained_sha256']})
            write(output/'admission-progress.json',{'episodes':len(episodes),'decisions':decision_count,'excluded':excluded})
            if len(episodes)>=plan['target_successful_episodes'] and len({r['task'] for r in episodes})>=plan['minimum_distinct_successful_tasks']:
                break
    passed = len(episodes)>=plan['target_successful_episodes'] and len({r['task'] for r in episodes})>=plan['minimum_distinct_successful_tasks']
    receipt = {'gate_passed':passed, 'decisions':decision_count, 'episodes':episodes,
        'successful_episodes':len(episodes),'unique_tasks':len({r['task'] for r in episodes}),
        'unique_scenes':len({(r['suite'],r['scene_sha256']) for r in episodes}),
        'suite_episodes':dict(Counter(r['suite'] for r in episodes)), 'excluded':excluded,
        'scene_audit_sha256':sha(audit_path), 'training_partition_only':True,
        'native_execution_verified':True, 'target_truncation':False,
        'scope':'Executed physical commands from successful training episodes. Invalid commands and stop decisions remain evidence but are excluded from targets.'}
    if not passed:
        write(output/'admission-shortfall.json', receipt)
        return receipt
    temporary.replace(output/'decisions.jsonl.gz')
    receipt['decisions_sha256'] = sha(output/'decisions.jsonl.gz')
    write(output/'admission.json', receipt)
    return receipt


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    args = p.parse_args()
    result = prepare(json.loads(args.plan.read_text()), args.root)
    print(json.dumps({k:result[k] for k in ['gate_passed','decisions','successful_episodes','unique_tasks','unique_scenes']}),flush=True)
