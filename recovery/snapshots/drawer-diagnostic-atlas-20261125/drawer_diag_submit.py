import json,os,sys,subprocess,time
from pathlib import Path
P=Path('/coc/flash7/czhang883/warp-data/registry/cxin34-drawer-diagnostic-20261125')
sys.path.insert(0,str(P));import drawer_diag as d
s='/opt/slurm/Ubuntu-24.04/current/bin/'
lk=d.lock(P/'submission.lock')
records=sorted(set(e['recording_id'] for e in d.segments));cached=[];missing=[]
for rid in records:
 e=next(e for e in d.segments if e['recording_id']==rid)
 try:d.depth_for(e);cached.append(rid)
 except AssertionError:missing.append(rid)
def submit(name,args,script):
 receipt=P/(name+'-submit.json');intent=P/(name+'-intent.json')
 if receipt.exists():return json.loads(receipt.read_text())['job_id']
 assert not intent.exists(),'existing submission intent: audit first'
 active=subprocess.check_output([s+'squeue','-u','cxin34','-h','-o','%i|%j'],text=True);assert not any(line.split('|')[-1]==name for line in active.splitlines())
 scriptpath=P/(name+'.sbatch');scriptpath.write_text(script)
 cmd=[s+'sbatch','--parsable','--account=rl2-lab','--partition=rl2-lab','--qos=debug','--cpus-per-task=4','--mem=24G','--time=03:55:00','--job-name='+name,'--output='+str(P/'logs')+'/'+name+'-%A_%a.log','--chdir='+str(P)]+args+[str(scriptpath)]
 d.write(intent,dict(cmd=cmd,script_sha256=d.sha(scriptpath),time=time.time()))
 r=subprocess.run(cmd,text=True,capture_output=True);d.write(P/(name+'-submission-result.json'),dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
 jid=r.stdout.strip().split(';')[0];assert jid.isdigit();d.write(receipt,dict(job_id=jid,command=cmd));print(name,jid,flush=True);return jid
jobs=[]
for kind,items in [('cached',cached),('depth',missing)]:
 if not items:continue
 d.write(P/(kind+'-recordings.json'),items)
 script='''#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
P=/coc/flash7/czhang883/warp-data/registry/cxin34-drawer-diagnostic-20261125
exec /coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python -u "$P/run_recording.py" '''+kind+' "${SLURM_ARRAY_TASK_ID}"\n'
 args=['--array=0-'+str(len(items)-1)+'%2']
 if kind=='depth':args+=['--gres=gpu:a40:1','--exclude=brainiac,spot,bishop,dynamics,protocol']
 jobs.append(submit('drawer-viz-'+kind+'-1125',args,script))
script='''#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec /coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python -u /coc/flash7/czhang883/warp-data/registry/cxin34-drawer-diagnostic-20261125/finalize.py
'''
submit('drawer-viz-publish-1125',['--dependency=afterany:'+':'.join(jobs)],script)
d.write(P/'plan.json',dict(records=len(records),cached=len(cached),new_depth=len(missing),segments=len(d.segments),cpu_concurrency=2,gpu_concurrency=2,training_authorized=False,jobs=jobs))
