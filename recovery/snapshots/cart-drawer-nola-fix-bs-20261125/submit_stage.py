"""Single audited submission for an owned stage. Existing intent stops retries."""
import sys,os,json,subprocess,fcntl,hashlib,datetime
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
name=sys.argv[1];batch=Path(sys.argv[2]).resolve();assert batch.is_relative_to(R)and batch.is_file()
lock=(R/'reports/submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=R/'reports'/(name+'-submit.json');intent=R/'reports'/(name+'.intent.json')
if receipt.exists():print(receipt.read_text());raise SystemExit(0)
assert not intent.exists(),'Existing intent must be audited before retry'
q=subprocess.run([S+'squeue','-h','-u','cxin34','-n',name,'-o','%i|%j'],capture_output=True,text=True,check=True);assert not q.stdout.strip(),q.stdout
cmd=[S+'sbatch','--parsable','--account=rl2-lab','--partition=rl2-lab','--qos=debug','--cpus-per-task=4','--mem=24G','--time=03:55:00','--job-name='+name,'--output='+str(R/'logs'/(name+'-%j.log')),'--chdir='+str(R),'--exclude=brainiac,spot']
cmd+=sys.argv[3:];cmd+=[str(batch)]
d=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,script_sha256=hashlib.sha256(batch.read_bytes()).hexdigest())
with intent.open('x')as f:json.dump(d,f,indent=2)
p=subprocess.run(cmd,capture_output=True,text=True);d.update(stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)
token=p.stdout.strip().split(';')[0];good=p.returncode==0 and token.isdigit();d.update(status='SUBMITTED'if good else'BLOCKED_SUBMISSION_REQUIRES_AUDIT',job_id=token if good else None);receipt.write_text(json.dumps(d,indent=2));print(json.dumps(d));assert good
