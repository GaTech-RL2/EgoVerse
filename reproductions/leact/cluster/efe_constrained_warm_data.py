"""Admit exact successful constrained-harness decisions with native execution evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from .scale_efe import ROOT, BUILD_ROOT, read, write
from .efe_action_loop import parse_action
from .efe_training_data import chat_inputs, sha


def prepare(destination):
    destination.mkdir(parents=True,exist_ok=True)
    source=ROOT/'fixed-action-v1/runs/v3/astra-task'
    manifest=read(source/'retained-manifest.json')
    spec=read(BUILD_ROOT/'efe-fixed-action-20261103-v3-astra-task/spec.json')
    scene_path=ROOT/'direct-control-v1/scene-audit/result.json';scene=read(scene_path)
    assert scene['upstream_revision']==spec['revision']
    train=set(read(ROOT/'generalization-v2/splits.json')['train_candidates'])
    collisions={r['task'] for r in scene['collisions']}
    audited={r['task']:r for r in scene['rows'] if r['partition']=='train_candidates'}
    tasks={r['task']:r for r in spec['tasks']};states=[];episodes=[]
    for entry in manifest['episodes']:
        if not entry['candidate_pending_scene_and_native_event_audit']:continue
        task=entry['task'];assert task in train and task not in collisions and task in audited
        assert audited[task]['upstream_digest']==tasks[task]['upstream_digest']
        bundle=source/'retained'/entry['retained_file'];assert sha(bundle)==entry['retained_sha256']
        with tarfile.open(bundle,'r:gz') as tar:
            def content(name):
                raw=tar.extractfile(name).read()
                assert hashlib.sha256(raw).hexdigest()==entry['files'][name]['sha256']
                return raw
            def named(suffix):
                found=[n for n in entry['files'] if n.endswith(suffix)];assert len(found)==1
                return content(found[0])
            trajectory=json.loads(named('/agent/trajectory.json'))
            contract_record=json.loads(named('/agent/action-contract.json'));contract=contract_record['contract']
            events=[json.loads(l) for l in named('/verifier/episode.jsonl').splitlines() if l.strip()]
            initial=next(e['state_sha'] for e in events if e.get('ev')=='start')
            assert initial==tasks[task]['teacher_initial_state_sha256']
            native=[json.loads(content(n)) for n in entry['files'] if n.endswith('/result.json')]
            native=next(r for r in native if 'verifier_result' in r)
            reward=native['verifier_result']['rewards']
            assert reward['success']==1 and reward['resource_enforced']==1 and not native.get('exception_info')
            steps={s['step_id']:s for s in trajectory['steps']}
            admitted=0
            for target in entry['targets']:
                if not target['complete_request_and_execution']:continue
                step=steps[target['step_id']];ident=target['request_record_id']
                stem='requests/'+entry['job']+'/'+ident
                record=json.loads(content(stem+'.json'))
                body=json.loads(gzip.decompress(content(stem+'.request.json.gz')))
                assert record['status']=='completed' and record['http_status']==200 and not record.get('error')
                assert record['request_sha256']==step['request_sha256']==target['request_sha256']
                action=parse_action(step['public_output'],contract=contract,allow_stop=False)
                assert len(step['tool_calls'])==1 and step['tool_calls'][0]['arguments']==action
                feedback=json.loads(step['observation']['results'][0]['content'])
                assert feedback['execution_completed'] and feedback['action']==action
                matches=[e for e in events if e.get('ev')=='submit' and e.get('control_steps')==feedback['control_steps_after']]
                assert len(matches)==1
                event=matches[0]
                assert event['rows']==action['repeat'] and event['type']==action['action_type']
                assert event['dt']==contract['control_dt'] and event['n_clipped_rows']==0
                assert feedback['control_steps_after']-feedback['control_steps_before']==action['repeat']
                visible=json.loads(body['input'][0]['content'][0]['text'])
                assert visible['mode']=='task' and visible['action_contract']==contract
                assert visible['observation']['control_steps']==feedback['control_steps_before']
                state_id=hashlib.sha256((manifest['workflow']+'/'+entry['job']+'/'+ident).encode()).hexdigest()
                states.append({'id':state_id,'task':task,'suite':tasks[task]['suite'],
                    'episode':manifest['workflow']+'/'+entry['job'],'messages':chat_inputs(body),'tools':[],
                    'action':action,'public_commentary':action['summary'],'scene_sha256':audited[task]['scene_sha256'],
                    'initial_state_sha256':initial,'source_bundle_sha256':entry['retained_sha256'],
                    'source_request_sha256':entry['files'][stem+'.request.json.gz']['sha256'],
                    'native_submit':event,'action_contract_sha256':contract_record['sha256'],
                    'system_prompt_sha256':contract_record['system_prompt_sha256']})
                admitted+=1
            episodes.append({'task':task,'admitted':admitted,'native_success':1,'initial_state_sha256':initial,
                             'scene_sha256':audited[task]['scene_sha256'],'bundle_sha256':entry['retained_sha256']})
    assert len(states)==43 and len(episodes)==3
    payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in states).encode()
    (destination/'decisions.jsonl.gz').write_bytes(gzip.compress(payload,mtime=0))
    receipt={'decisions':len(states),'episodes':episodes,'decisions_sha256':sha(destination/'decisions.jsonl.gz'),
        'scene_audit_sha256':sha(scene_path),'training_partition_only':True,'native_execution_verified':True,
        'scope':'Three-task warmstart feasibility pilot. All targets are complete executed commands from native successful episodes; no movement diagnostic or future observation enters the inputs.'}
    write(destination/'admission.json',receipt)
    return receipt
