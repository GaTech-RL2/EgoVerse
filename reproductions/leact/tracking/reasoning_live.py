"""Read-only OSMO collector for the separate reasoning ITSS workspace."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from .reasoning import BASE,read,write


def command(args,timeout=15):
    p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:out,err=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill();out,err=p.communicate()
        if not out:raise TimeoutError('OSMO read timed out')
    if p.returncode not in (0,-9):raise RuntimeError(err[-400:])
    return out


def retention():
    folder=BASE/'artifacts/reasoning-workspace/retention-496'
    launch=read(folder/'launch.json',{})
    if not launch.get('name'):return {'status':'not submitted'}
    result=read(folder/'live.json',{})
    if result.get('workflow')!=launch['name']:result={}
    if result.get('scheduler_status')=='COMPLETED'and result.get('result'):return result
    state=json.loads(command(['osmo','workflow','query',launch['name'],'--format-type','json']))
    result.update(workflow=launch['name'],url=launch['overview'],scheduler_status=state['status'],
        observed_utc=datetime.now(timezone.utc).isoformat(),benchmark='RealWorldQA',
        questions_per_model=launch.get('questions_per_model',765),preliminary=launch.get('preliminary',False))
    if state.get('start_time'):
        logs=command(['osmo','workflow','logs',launch['name'],'-n','1000'],timeout=5)
        for line in logs.splitlines():
            for marker,key in [('EFE_RETENTION_PROGRESS ','progress'),('EFE_RETENTION_RESULT ','result')]:
                if marker in line:
                    try:result[key]=json.loads(line.split(marker,1)[1])
                    except json.JSONDecodeError:pass
        if state['status'].startswith('FAILED'):
            result['failure_message']=next((t.get('failure_message')for g in state.get('groups',[])
                for t in g.get('tasks',[])if t.get('failure_message')),None)
    write(folder/'live.json',result)
    return result


def compact_telemetry(rows,*,limit=512):
    """Keep all raw metrics on shared storage; send chart data and latest tasks."""
    steps={};groups={};evaluations={};checkpoints={}
    for row in rows:
        if row['kind']=='step':steps[row['step']]=row
        elif row['kind']=='group':groups[row['event_id']]=row
        elif row['kind']=='evaluation':evaluations[row['step']]=row
        elif row['kind']=='checkpoint':checkpoints[row['step']]=row
    history=[]
    for step,row in sorted(steps.items())[-limit:]:
        values=row['metrics']
        keep={k:v for k,v in values.items()if k.startswith((
            'reward_unfiltered_cumulative/family/','reward_unfiltered_cumulative/all',
            'reward_unfiltered_cumulative/macro_task','groups_cumulative/',
            'curriculum/retained_groups/','timing_s/','perf/'))}
        history.append({'step':step,'updated_utc':row['updated_utc'],**row['headline'],'metrics':keep})
    latest=steps[max(steps)]if steps else None
    tasks={k.removeprefix('reward_unfiltered_cumulative/task/'):v
           for k,v in (latest or {}).get('metrics',{}).items()if k.startswith('reward_unfiltered_cumulative/task/')}
    return {'history':history,'latest_task_rewards':tasks,'recent_graded_groups':list(groups.values())[-30:],
            'latest_metrics':latest['metrics']if latest else {},'optimizer_steps_observed':max(steps)if steps else 0,
            'evaluations':[r for _,r in sorted(evaluations.items())][-limit:],
            'checkpoints':[r for _,r in sorted(checkpoints.items())][-limit:]}


def pear():
    from ..cluster.monitor_efe_preflight import remote
    path=BASE/'artifacts/reasoning-workspace/live/pear.json';old=read(path,{})
    plan=read(BASE/'artifacts/embodied-first-exam/pear-scaled-v1/plan.json',{})
    arms=plan.get('arms',[])
    workflows=json.loads(command(['osmo','workflow','list','--name','leact','--status','RUNNING',
                                   '--count','200','--format-type','json']))['workflows']
    # Graphics workers on L40S-01 use LFS. Read model telemetry only through a
    # running worker that mounts AMLFS07; never allocate a GPU for collection.
    compatible={'groot-h100-01','groot-h100-02','groot-l40-02','groot-l40-03'}
    candidates=[w for w in workflows if('pear'in w['name']or'realworldqa'in w['name'])
        and w.get('pool')in compatible]
    primary=read(BASE/'artifacts/embodied-first-exam/pear-scaled-v1/primary-launch.json',{})
    candidates.sort(key=lambda w:w['name']!=primary.get('workflow'))
    if not candidates:return {**old,'collector_state':'waiting for a running PEAR or retention worker'}
    roots={a['id']:a.get('run_directory')for a in arms if a.get('run_directory')}
    if not roots:return {**old,'collector_state':'waiting for registered run directories'}
    script='''from pathlib import Path
import json
roots=json.loads(ROOTS)
result={}
for arm,root in roots.items():
 p=Path(root)/'telemetry';events=[];f=p/'events.jsonl'
 if f.is_file():
  with f.open('rb')as stream:
   size=f.stat().st_size;stream.seek(max(0,size-2000000));raw=stream.read()
  lines=raw.splitlines()
  if size>2000000:lines=lines[1:]
  for line in lines:
   try:events.append(json.loads(line))
   except (ValueError,UnicodeDecodeError):pass
 h=p/'heartbeat.json'
 try:heartbeat=json.loads(h.read_text())
 except (FileNotFoundError,ValueError):heartbeat=None
 diagnostics={}
 for worker in (p/'rollout-workers').glob('*.json'):
  try:value=json.loads(worker.read_text())
  except (FileNotFoundError,ValueError):continue
  for key,n in value['totals'].items():diagnostics[key]=diagnostics.get(key,0)+n
 try:metadata=json.loads((p/'run.json').read_text())
 except (FileNotFoundError,ValueError):metadata={}
 result[arm]={'events':events,'heartbeat':heartbeat,'rollout_diagnostics':diagnostics,'metadata':metadata}
print('EFE_CAPTURE '+json.dumps(result),flush=True)
'''.replace('ROOTS',repr(json.dumps(roots)))
    candidate=candidates[0];name=candidate['name']
    task='trainer'if'pear-primary-'in name else'native'if'pear-worker-'in name else'preflight'
    capture=remote(name,script,task=task)
    result={'observed_utc':datetime.now(timezone.utc).isoformat(),'collector_state':'connected','runs':{}}
    cache=path.parent;cache.mkdir(parents=True,exist_ok=True)
    for arm,value in capture.items():
        target=cache/(arm+'-events.json');saved=read(target,{})
        for row in value['events']:saved[row['event_id']]=row
        write(target,saved)
        result['runs'][arm]={**compact_telemetry(saved.values()),'heartbeat':value['heartbeat'],
                            'run_directory':roots[arm],'rollout_diagnostics':value['rollout_diagnostics'],
                            'metadata':value['metadata']}
    write(path,result);return result


def main():
    root=BASE/'artifacts/reasoning-workspace/live';root.mkdir(parents=True,exist_ok=True)
    for name,collect in [('retention',retention),('pear',pear)]:
        try:value=collect();write(root/(name+'-collector.json'),{'success':True,'updated_utc':datetime.now(timezone.utc).isoformat()})
        except Exception as exc:
            write(root/(name+'-collector.json'),{'success':False,'error':str(exc)[-500:],
                                              'updated_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
