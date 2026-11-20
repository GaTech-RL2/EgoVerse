"""Durable PEAR driver metrics, including groups discarded before training.

The driver receives already reduced actor metrics. Never average them again
over ranks or synthesize absent measurements (notably reference-policy KL).
"""
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import threading
import time
from uuid import uuid4


METRICS={
    'policy_loss':['actor/pg_loss','actor/policy_loss'],
    'grad_norm':['actor/grad_norm'], 'learning_rate':['actor/lr'],
    'entropy':['actor/entropy','actor/entropy_loss'],
    'rollout_policy_kl':['actor/ppo_kl','rollout_corr/kl'],
    'response_tokens':['response_length/mean'],
    'truncation_rate':['response_length/clip_ratio'],
    'step_seconds':['timing_s/step'],
    'gpu_memory_gb':['perf/max_memory_allocated_gb'],
}


def scalars(values):
    result={}
    for key,value in values.items():
        if hasattr(value,'item'):
            try:value=value.item()
            except (ValueError,RuntimeError):continue
        if isinstance(value,(int,float)) and math.isfinite(value):result[str(key)]=float(value)
    return result


def reward_metrics(groups,configs):
    """Cumulative, unfiltered binary reward; macro means weight tasks equally."""
    buckets=defaultdict(list);retained=0;count=0
    for g in groups.values():
        if not g['infrastructure_ok']:continue
        c=configs[g['config_id']];values=g['outcomes']
        if len(values)!=8 or any(v not in (0,1)for v in values):raise ValueError('Unjudged group in reward metrics')
        count+=1;retained+=int(min(values)!=max(values))
        for scope in ['all','family/'+c['family'],'suite/'+c['suite'],
                      'task/'+c['suite']+'/'+c['task'],'start/'+c['stratum']]:buckets[scope].extend(values)
    result={}
    for scope,values in buckets.items():
        result['reward_unfiltered_cumulative/'+scope]=sum(values)/len(values)
        result['episodes_cumulative/'+scope]=len(values)
    tasks=[value for key,value in result.items()if key.startswith('reward_unfiltered_cumulative/task/')]
    if tasks:result['reward_unfiltered_cumulative/macro_task']=sum(tasks)/len(tasks)
    result.update({'groups_cumulative/judged':count,'groups_cumulative/retained':retained,
                   'groups_cumulative/constant_reward':count-retained})
    if count:result['groups_cumulative/retained_fraction']=retained/count
    return result


class Telemetry:
    def __init__(self,root,run_id,*,heartbeat=False):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.run_id=run_id
        self.step=0;self.phase='initializing';self.started=time.monotonic()
        self.events=self.root/'events.jsonl';self.known=set()
        if self.events.exists():
            for line in self.events.read_text().splitlines():
                # A killed writer may leave only the final append incomplete.
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if row['run_id']!=run_id:raise ValueError('Telemetry directory belongs to another run')
                self.known.add(row['event_id'])
        if heartbeat:threading.Thread(target=self._heartbeat,daemon=True).start()

    def emit(self,kind,event_id,**payload):
        if event_id in self.known:return False
        row={'schema_version':1,'run_id':self.run_id,'kind':kind,'event_id':event_id,
             'updated_utc':datetime.now(timezone.utc).isoformat(),**payload}
        raw=json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'
        with self.events.open('a+')as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            # Separate complete events from an interrupted final append.
            stream.seek(0,2)
            if stream.tell():
                stream.seek(stream.tell()-1)
                if stream.read(1)!='\n':stream.write('\n')
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        self.known.add(event_id)
        if kind in {'step','evaluation','checkpoint','group'}:print('PEAR_EVENT '+raw.rstrip(),flush=True)
        return True

    def group(self,g,c):
        return self.emit('group','group/'+g['uid'],step=self.step,
            family=c['family'],suite=c['suite'],task=c['task'],stratum=c['stratum'],
            config_id=c['config_id'],rewards=g['outcomes'],
            infrastructure_ok=g['infrastructure_ok'],partition='train')

    def dispatch(self,uid,receipt):
        return self.emit('dispatch','dispatch/'+uid,step=self.step,**receipt)

    def metadata(self,value):
        path=self.root/'run.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps({'run_id':self.run_id,**value},allow_nan=False)+'\n');temp.replace(path)
        print('PEAR_RUN '+json.dumps({'run_id':self.run_id,**value},allow_nan=False),flush=True)

    def metrics(self,step,values):
        self.step=int(step);self.phase='training'
        nonfinite=[str(k)for k,v in values.items()if isinstance(v,(int,float))and not math.isfinite(v)]
        values=scalars(values)
        headline={name:next((values[k]for k in aliases if k in values),None)for name,aliases in METRICS.items()}
        headline['reference_policy_kl']=None
        folder=self.root/'steps';folder.mkdir(exist_ok=True)
        path=folder/(f'{int(step):08d}'+'.json');temp=path.with_suffix('.tmp')
        if not path.exists():
            temp.write_text(json.dumps({'step':int(step),'updated_utc':datetime.now(timezone.utc).isoformat(),
                                      'metrics':values,'headline':headline},allow_nan=False)+'\n');temp.replace(path)
        self.emit('step','step/'+str(step),step=int(step),metrics=values,headline=headline,nonfinite_metrics=nonfinite,
            definitions={'rollout_policy_kl':'Sampled rollout-versus-current policy estimate, not KL to initialization.',
                         'reference_policy_kl':'Not measured: PEAR has no frozen reference-policy forward or KL penalty.'})
        if nonfinite:raise FloatingPointError('Nonfinite training metrics: '+','.join(nonfinite))

    def _heartbeat(self):
        while True:
            value={'run_id':self.run_id,'step':self.step,'phase':self.phase,
                   'updated_utc':datetime.now(timezone.utc).isoformat(),
                   'process_uptime_seconds':time.monotonic()-self.started}
            path=self.root/'heartbeat.json';temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(value)+'\n');temp.replace(path)
            totals={}
            for worker in (self.root/'rollout-workers').glob('*.json'):
                try:record=json.loads(worker.read_text())
                except (FileNotFoundError,ValueError):continue
                for key,n in record['totals'].items():totals[key]=totals.get(key,0)+n
            print('PEAR_HEARTBEAT '+json.dumps({**value,'rollout_diagnostics':totals},allow_nan=False),flush=True)
            time.sleep(10)


class WorkerDiagnostics:
    """Each rollout process owns one cumulative file, across all its episodes."""
    def __init__(self,root):
        self.root=Path(root)/'rollout-workers';self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/(str(os.getpid())+'-'+uuid4().hex+'.json');self.totals={}

    def add(self,values):
        for key,value in scalars(values).items():self.totals[key]=self.totals.get(key,0)+value
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),
                                   'totals':self.totals},allow_nan=False)+'\n');temp.replace(self.path)
