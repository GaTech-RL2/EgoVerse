"""Collect public replay, raw training inputs and per-action outcome evidence."""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tarfile

import efe_native_pilot as runtime
from export_efe_viewer import export_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def action_audit(trace):
    targets, violations = [], []
    exec_starts = []
    for step in trace['steps']:
        feedback = {r['source_call_id']:str(r.get('content',''))
                    for r in step.get('observation',{}).get('results',[]) if r.get('source_call_id')}
        for call in step['tool_calls']:
            name = call['function_name'].split('.')[-1]
            args = call.get('arguments') or {}; command = str(args.get('cmd',''))
            result = feedback.get(call.get('tool_call_id'))
            reason = []
            if result is None: reason.append('missing_execution_feedback')
            elif re.search(r'Process exited with code [1-9]\d*|Unknown process id|Traceback \(most recent call last\)', result):
                reason.append('error_returning_target')
            if name=='write_stdin':reason.append('interactive_process_tool');violations.append('write_stdin')
            if 'apply_patch' in command:reason.append('patch_or_setup');violations.append('apply_patch')
            if name=='exec_command':
                exec_starts.append(datetime.fromisoformat(step['timestamp'].replace('Z','+00:00')).timestamp())
                if re.search(r'(?:^|\n)\s*(?:async\s+)?def\s+|\bwhile\s+True\b', command):
                    violations.append('helper_or_controller_definition_requires_review')
                if args.get('yield_time_ms') != 30000:violations.append('short_shell_wait')
                if result is not None and 'Process exited with code 0' not in result:
                    reason.append('command_completion_not_verified')
            if name not in {'exec_command','view_image'}:reason.append('outside_direct_method')
            targets.append({'call_id':call.get('tool_call_id'), 'step_id':step['step_id'],
                'function':name, 'eligible_action_target':not reason, 'exclusion_reasons':reason})
    submits = [e for e in trace['simulator_events'] if e.get('ev')=='submit']
    for event in submits:
        if not isinstance(event.get('rows'), (int,float)) or not isinstance(event.get('dt'), (int,float)):
            violations.append('missing_action_horizon_evidence')
        elif event['rows']>128 or event['rows']*event['dt']>2.0+1e-6:
            violations.append('action_horizon_exceeded')
    for event in trace['simulator_events']:
        if event.get('ev')=='observe' and event.get('seconds',0)>2.0+1e-6:
            violations.append('observation_hold_exceeded')
    # Each observed submit must belong to a separate shell call, not a controller loop.
    blocks = {}
    for event in submits:
        starts = [s for s in exec_starts if s <= event.get('ts',0)]
        if not starts:violations.append('submit_without_recorded_shell_call');continue
        start=max(starts);blocks[start]=blocks.get(start,0)+1
    if any(count>1 for count in blocks.values()):violations.append('multiple_submits_per_shell_call')
    if not submits:violations.append('no_robot_action')
    return {'targets':targets, 'target_count':len(targets),
        'eligible_targets':sum(t['eligible_action_target'] for t in targets),
        'method_violations':sorted(set(violations)), 'direct_method_pass':not violations,
        'robot_submits':len(submits),
        'scope':'Executed-call and simulator-log checks; final training admission also requires scene isolation and code review. Error feedback remains context for subsequent valid recovery targets.'}


def main():
    module=importlib.util.spec_from_file_location('pilot','/tmp/efe-astra-pilot.py')
    pilot=importlib.util.module_from_spec(module);module.loader.exec_module(pilot)
    original_write=runtime.write
    spec=json.loads(runtime.SPEC.read_text())
    archive=Path(spec['archive_root'])/runtime.WORKFLOW
    public=archive/'public-episodes';public.mkdir(parents=True,exist_ok=True)
    retained=archive/'retained';retained.mkdir(exist_ok=True)
    durable=Path(spec['durable_root'])/runtime.WORKFLOW if spec.get('durable_root') else None
    if durable:durable.mkdir(parents=True,exist_ok=True)
    exports={}

    def publish(name, value):
        original_write(name,value)
        if durable and name in {'results.json','progress.json','complete.json','setup-error.json','native-config.json','spec.json'}:
            write(durable/name,value)
        if name!='results.json':return
        for trial in value.get('trials',[]):
            key=trial['job']
            if key in exports:continue
            entry={'task':trial['task'],'job':key,'status':trial['status'],'score':trial['score']}
            try:
                row=export_episode(dict(trial,role='direct_teacher'),public)
                trace=json.loads((public/row['id']/'trace.json').read_text())
                audit=action_audit(trace);write(public/row['id']/'action-audit.json',audit)
                entry.update(row,action_audit_sha256=sha(public/row['id']/'action-audit.json'),
                    direct_method_pass=audit['direct_method_pass'],method_violations=audit['method_violations'],
                    eligible_targets=audit['eligible_targets'],robot_submits=audit['robot_submits'])
                entry['candidate_episode']=trial['status']=='graded' and trial['score']==1 and not trial['model_errors'] and audit['direct_method_pass']
            except Exception as error:
                entry.update(export_error=str(error),candidate_episode=False)
            # Preserve exact pre-action inputs and public model outputs for later
            # data extraction, including unsuccessful attempts. Do not export
            # provider-private reasoning or raw Codex sessions to the dashboard.
            bundle=retained/(key+'.tar.gz')
            with tarfile.open(bundle,'w:gz') as tar:
                for p in sorted((archive/'requests'/key).glob('*')):
                    if p.is_file():tar.add(p,arcname=str(p.relative_to(archive)))
                for p in sorted((archive/'jobs'/key).rglob('*')):
                    relative=str(p.relative_to(archive))
                    if p.is_file() and (p.name=='result.json' or relative.endswith('/agent/trajectory.json')
                        or '/verifier/' in relative and p.suffix in {'.json','.jsonl'}):
                        tar.add(p,arcname=relative)
            entry.update(retained_file=bundle.name,retained_sha256=sha(bundle),retained_bytes=bundle.stat().st_size)
            if durable:
                target=durable/'retained'/bundle.name;target.parent.mkdir(exist_ok=True)
                temporary=target.with_suffix('.copy.tmp');shutil.copyfile(bundle,temporary)
                if sha(temporary)!=entry['retained_sha256']:raise ValueError('Durable teacher bundle differs')
                temporary.replace(target)
                if entry.get('id'):
                    shutil.copytree(public/entry['id'],durable/'public-episodes'/entry['id'],dirs_exist_ok=True)
            exports[key]=entry
        write(public/'manifest.json',{'workflow':runtime.WORKFLOW,'episodes':list(exports.values()),
            'protocol':'direct-control-v1','created_utc':runtime.now(),'planned':len(spec['tasks']),
            'provisional_eligible_episodes':sum(r.get('candidate_episode',False) for r in exports.values())})
        if durable:
            write(durable/'public-episodes/manifest.json',json.loads((public/'manifest.json').read_text()))

    runtime.write=publish
    pilot.main()


if __name__=='__main__':main()
