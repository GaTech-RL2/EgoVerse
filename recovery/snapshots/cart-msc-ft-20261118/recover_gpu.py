import json,subprocess,fcntl
from pathlib import Path
R=Path('/coc/flash7/cxin34/cart-msc-ft-20261118');SL='/opt/slurm/Ubuntu-24.04/current/bin/'
lock=(R/'reports/submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=R/'reports/jobs.json';j=json.loads(p.read_text());assert j['audit_job']=='4794202'
assert not (R/'reports/gpu-recovery.intent').exists()
def call(args):return subprocess.check_output(args,text=True).strip()
assert 'FAILED' in call([SL+'sacct','-X','-n','-j','4794202','-o','State'])
(R/'reports/gpu-recovery.intent').write_text('Audit4794202 failed CUDA initialization on omgwth; fresh typedA40 audit required; existing report not accepted as job completion')
prior=R/'reports/photometric-audit.json'
if prior.exists():prior.rename(R/'reports/photometric-audit-before-gpu-recovery.json')
j['excluded']+=',omgwth'
for row in j['runs'].values():
    for key in ['smoke','formal']:
        subprocess.run([SL+'scontrol','update','JobId='+row[key],'Gres=gpu:a40:1','ExcNodeList='+j['excluded']],check=True)
args=[SL+'sbatch','--parsable','--partition=rl2-lab','--qos=debug','--account=rl2-lab','--gres=gpu:a40:1','--exclude='+j['excluded'],
 '--nice=30000','--cpus-per-task=4','--mem=64G','--time=01:00:00','--chdir='+str(R),'--job-name=cart-msc-contracts-a40',
 '--output='+str(R/'reports/audit-%j.log'),'--export=ALL,CAMPAIGN_ROOT='+str(R),str(R/'scripts/audit.sbatch')]
new=call(args);assert new.isdigit()
j['failed_audit_job']=j['audit_job'];j['audit_job']=new;p.write_text(json.dumps(j,indent=2))
for row in j['runs'].values():subprocess.run([SL+'scontrol','update','JobId='+row['smoke'],'Dependency=afterok:'+new],check=True)
print(json.dumps(j,indent=2))
