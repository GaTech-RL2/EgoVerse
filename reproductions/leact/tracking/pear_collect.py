"""Hosted, read-only OSMO log collector. Never submits or changes workflows."""
import argparse
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

try:
    from .server import connect,ingest_pear,latest,pack_payload
except ImportError:
    from server import connect,ingest_pear,latest,pack_payload


def command(binary,args,timeout=15):
    p=subprocess.Popen([binary,'workflow',*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:out,err=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill();out,err=p.communicate()
        if not out:raise TimeoutError('OSMO read timed out')
    if p.returncode not in (0,-9):raise RuntimeError('OSMO read failed: '+err[-180:])
    return out


def parse_records(logs):
    decoder=json.JSONDecoder()
    for line in logs.splitlines():
        for marker in ['PEAR_RUN ','PEAR_EVENT ','PEAR_HEARTBEAT ','PEAR_MODEL_CHECK ','PEAR_ADMISSION_CHECK ','PEAR_REASONING_CHECK ',
                       'PEAR_NATIVE_YAM_CHECK ','PEAR_ONPOLICY_CHECK ','PEAR_SERVER_CHECK ',
                       'PEAR_NATIVE_RTX_CHECK ','PEAR_NATIVE_IMAGE_PUBLISH ',
                       'EFE_RETENTION_PROGRESS ','EFE_RETENTION_RESULT ']:
            if marker not in line:continue
            try:row=decoder.raw_decode(line.split(marker,1)[1])[0]
            except (ValueError,TypeError):continue
            if isinstance(row,dict):yield marker.strip(),row


def fold(state,marker,row):
    if marker=='PEAR_RUN':state['metadata']=row
    elif marker=='PEAR_HEARTBEAT':
        state['heartbeat']={k:v for k,v in row.items()if k!='rollout_diagnostics'}
        state['rollout_diagnostics']=row.get('rollout_diagnostics',{})
    elif marker=='PEAR_EVENT':
        kind=row['kind']
        if kind=='step':
            values=row['metrics'];history={p['step']:p for p in state.get('history',[])}
            keep={k:v for k,v in values.items()if k.startswith((
                'reward_unfiltered_cumulative/family/','reward_unfiltered_cumulative/all',
                'reward_unfiltered_cumulative/macro_task','groups_cumulative/',
                'curriculum/retained_groups/','timing_s/','perf/'))}
            history[row['step']]={'step':row['step'],'updated_utc':row['updated_utc'],**row['headline'],'metrics':keep}
            state['history']=[v for _,v in sorted(history.items())][-512:]
            if row['step']>=state.get('optimizer_steps_observed',0):
                state['optimizer_steps_observed']=row['step'];state['latest_metrics']=values
                state['latest_task_rewards']={k.removeprefix('reward_unfiltered_cumulative/task/'):v for k,v in values.items()
                                             if k.startswith('reward_unfiltered_cumulative/task/')}
        elif kind in {'evaluation','checkpoint'}:
            key='evaluations'if kind=='evaluation'else'checkpoints'
            records={v['step']:v for v in state.get(key,[])};records[row['step']]=row
            state[key]=[v for _,v in sorted(records.items())][-512:]
        elif kind=='group':
            records={r['event_id']:r for r in state.get('recent_graded_groups',[])};records[row['event_id']]=row
            state['recent_graded_groups']=list(records.values())[-30:]
            rewards=row.get('rewards',[])
            if(row.get('partition')=='train'and row.get('infrastructure_ok')and len(rewards)==8
                and all(v in(0,1)for v in rewards)):
                seen=state.setdefault('graded_group_ids',{})
                if row['event_id']not in seen:
                    seen[row['event_id']]=True
                    totals=state.setdefault('group_reward_totals',{})
                    for scope in ['all','family/'+row['family'],'suite/'+row['suite'],
                        'task/'+row['suite']+'/'+row['task']]:
                        value=totals.setdefault(scope,{'episodes':0,'successes':0})
                        value['episodes']+=8;value['successes']+=sum(rewards)
                    state['latest_task_rewards']={key.removeprefix('task/'):v['successes']/v['episodes']
                        for key,v in totals.items()if key.startswith('task/')}
                    metrics=state.setdefault('latest_metrics',{})
                    for key,value in totals.items():
                        metrics['reward_unfiltered_cumulative/'+key]=value['successes']/value['episodes']
                        metrics['episodes_cumulative/'+key]=value['episodes']
                    metrics['groups_cumulative/judged']=len(seen)
                    metrics['reward_unfiltered_cumulative/macro_task']=sum(state['latest_task_rewards'].values())/len(state['latest_task_rewards'])
    return state


def fold_launch_check(state,workflow,records):
    """Keep qualification separate from primary optimizer metrics and history."""
    component=workflow['labels']['pear_phase']
    metadata=state.setdefault('metadata',{})
    checks=metadata.setdefault('launch_checks',{})
    previous=checks.get(component,{})
    if previous.get('submit_time','')>workflow['submit_time']:return False
    value=dict(previous)if previous.get('submit_time')==workflow['submit_time']else{'phase':component}
    value.update(submit_time=workflow['submit_time'],scheduler_status=workflow['status'],
                 observed_utc=datetime.now(timezone.utc).isoformat())
    for marker,row in records:
        if marker in {'PEAR_MODEL_CHECK','PEAR_ADMISSION_CHECK','PEAR_REASONING_CHECK',
                      'PEAR_NATIVE_YAM_CHECK','PEAR_ONPOLICY_CHECK','PEAR_SERVER_CHECK',
                      'PEAR_NATIVE_RTX_CHECK','PEAR_NATIVE_IMAGE_PUBLISH'}:
            value.update({key:row[key]for key in ['phase','updated_utc','checked','total','passed','counts','families']if key in row})
            if any(word in str(row.get('phase',''))for word in ['completed','passed','failed']):
                value['terminal_log_captured']=True
    checks[component]=value
    metadata['run_id']=workflow['labels']['pear_run_id']
    if not metadata.get('model_revision'):metadata['qualification_only']=True
    return True


def needs_qualification_poll(state,workflow):
    phase=workflow['labels']['pear_phase']
    previous=state.get('metadata',{}).get('launch_checks',{}).get(phase,{})
    if previous.get('submit_time','')>workflow['submit_time']:return False
    terminal=workflow['status']in {'COMPLETED','CANCELED','CANCELLED'}or workflow['status'].startswith('FAILED')
    return not(terminal and previous.get('submit_time')==workflow['submit_time']
               and previous.get('scheduler_status')==workflow['status']
               and(workflow['status']!='COMPLETED'or previous.get('terminal_log_captured')))


def tick(database,binary):
    workspace=latest(database).get('status',{}).get('reasoning_workspace',{})
    allowed={r['id']for r in workspace.get('pear',{}).get('arms',[])}
    rows=json.loads(command(binary,['list','--name','pear','--user','rpunamiya','--count','100',
                                   '--order','desc','--format-type','json']))['workflows']
    workflows={r['name']:r for r in rows if r.get('labels',{}).get('pear_run_id')in allowed
               or any(a in r['name']for a in allowed)}
    retention=json.loads(command(binary,['list','--name','warm496-realworldqa','--user','rpunamiya',
                                       '--count','1','--order','desc','--format-type','json']))['workflows']
    for r in retention:workflows[r['name']]=r
    states={}
    with closing(connect(database))as db:
        try:
            from .server import unpack_payload
        except ImportError:from server import unpack_payload
        for ident,payload in db.execute('SELECT run_id,payload FROM pear_runs'):
            states[ident]=json.loads(unpack_payload(payload))
    for name,workflow in workflows.items():
        phase=workflow.get('labels',{}).get('pear_phase','')
        if phase=='primary_native_workers':
            # Actions, reasoning, rewards and simulator errors are reported by
            # the primary tools. Avoid polling every Kit startup log here.
            continue
        qualification=(phase in {'router_optimizer_qualification','training_reset_admission',
                                  'reasoning_scene_and_transport_qualification',
                                  'native_yam_runtime_api_qualification',
                                  'ray_server_startup_qualification','native_robolab_rtx_api_check',
                                  'native_image_cpu_publication',
                                  'on_policy_score_centering_qualification'}or phase.startswith('all_tasks_')
                                  or phase.startswith('rtx_transport_'))
        if qualification and not needs_qualification_poll(states.get(workflow['labels'].get('pear_run_id'),{}),workflow):
            continue
        if not workflow.get('start_time'):
            logs=''
        else:
            try:logs=command(binary,['logs',name,'-n','80'if qualification else'2000'],timeout=4 if qualification else 6)
            except (TimeoutError,RuntimeError):logs=''
        if name in {r['name']for r in retention}:
            state=states.setdefault('__retention496__',{})
            source_id=workflow.get('workflow_uuid',workflow.get('uuid',hashlib.sha256(name.encode()).hexdigest()))
            if state.get('source_id')!=source_id:state.clear()
            labels=workflow.get('labels',{})
            state.update(source_id=source_id,scheduler_status=workflow['status'],benchmark='RealWorldQA',
                         questions_per_model=int(labels.get('reasoning_eval_questions',765)),
                         preliminary=labels.get('reasoning_eval_scope')=='preliminary',
                         observed_utc=datetime.now(timezone.utc).isoformat())
            for marker,row in parse_records(logs):
                if marker=='EFE_RETENTION_PROGRESS':state['progress']=row
                elif marker=='EFE_RETENTION_RESULT':state['result']=row
            # The private workflow name is only a local identity for restarts.
            public=dict(state)
            if 'progress'in public:public['progress']={k:v for k,v in public['progress'].items()if k!='workflow'}
            with closing(connect(database))as db,db:db.execute('INSERT OR REPLACE INTO pear_runs VALUES(?,?,?)',
                ('__retention496__',state['observed_utc'],pack_payload(json.dumps(public,allow_nan=False))))
            continue
        changed=set();records=list(parse_records(logs))
        ident=workflow.get('labels',{}).get('pear_run_id')
        if ident in allowed and qualification:
            state=states.setdefault(ident,{'history':[],'latest_task_rewards':{},'latest_metrics':{},
                                          'recent_graded_groups':[],'optimizer_steps_observed':0})
            if fold_launch_check(state,workflow,records):changed.add(ident)
        for marker,row in records:
            ident=row.get('run_id')
            if ident not in allowed:continue
            state=states.setdefault(ident,{'history':[],'latest_task_rewards':{},'latest_metrics':{},'recent_graded_groups':[]})
            fold(state,marker,row);changed.add(ident)
        for ident in changed:ingest_pear(database,{'schema_version':1,'run_id':ident,'data':states[ident]})
    return {'updated_utc':datetime.now(timezone.utc).isoformat(),'workflows_checked':len(workflows),'read_only':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--osmo',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        try:value={'success':True,**tick(args.db,args.osmo)}
        except Exception as exc:value={'success':False,'error':str(exc)[-240:],'updated_utc':datetime.now(timezone.utc).isoformat()}
        path=args.db.parent/'collector-status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value)+'\n');temp.replace(path)
        print(json.dumps(value),flush=True)
        if args.once:return
        time.sleep(20)


if __name__=='__main__':main()
