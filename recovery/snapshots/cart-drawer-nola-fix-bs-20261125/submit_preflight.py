"""One submission intent per independent CPU preflight; never blindly retry."""
import json,subprocess,fcntl,hashlib,datetime
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
S='/opt/slurm/Ubuntu-24.04/current/bin/'
lock=(R/'reports/submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name,py,script,array in [('drawer-ondevice-preflight','/coc/flash7/czhang883/miniconda3/envs/aria2/bin/python','drawer_stream_audit.py','0-3%4'),('cart-nola-preflight','/coc/flash7/czhang883/miniconda3/envs/rby1/bin/python','cart_input_audit.py',None)]:
    receipt=R/'reports'/(name+'-submit.json');intent=R/'reports'/(name+'.intent.json')
    if receipt.exists():print(receipt.read_text());continue
    assert not intent.exists(),'Existing intent requires Slurm reconciliation: '+str(intent)
    live=subprocess.run([S+'squeue','-h','-u','cxin34','-n',name,'-o','%i|%j'],text=True,capture_output=True,check=True)
    assert not live.stdout.strip(),live.stdout
    text='#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2\n'+py+' '+str(R/'scripts'/script)+(' "${SLURM_ARRAY_TASK_ID}"'if array else'')+'\n'
    batch=R/'scripts'/(name+'.sbatch');batch.write_text(text)
    cmd=[S+'sbatch','--parsable','--account=rl2-lab','--partition=rl2-lab','--qos=debug','--cpus-per-task=2','--mem=8G','--time=00:45:00','--job-name='+name,'--output='+str(R/'logs'/(name+'-%A_%a.log')),'--chdir='+str(R),'--exclude=brainiac,spot']
    if array:cmd+=['--array='+array]
    cmd+=[str(batch)]
    d=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,script_sha256=hashlib.sha256(text.encode()).hexdigest())
    with intent.open('x')as f:json.dump(d,f,indent=2)
    result=subprocess.run(cmd,text=True,capture_output=True)
    d.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
    token=result.stdout.strip().split(';')[0]
    if result.returncode or not token.isdigit():
        d['status']='BLOCKED_SUBMISSION_REQUIRES_AUDIT';receipt.write_text(json.dumps(d,indent=2));raise RuntimeError(d)
    d.update(job_id=token,status='SUBMITTED');receipt.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
