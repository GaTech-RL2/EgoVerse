import datetime,fcntl,json,re,subprocess
from pathlib import Path
R=Path('/coc/flash7/cxin34/cart-msc-ft-20261118');SL='/opt/slurm/Ubuntu-24.04/current/bin/'
lock=(R/'reports/submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'reports/submit.intent').exists(),'Audit existing ledger; do not duplicate'
def call(args):return subprocess.check_output(args,text=True).strip()
def save(x):
    p=R/'reports/jobs.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
inventory=call([SL+'sinfo','-p','rl2-lab','-N','-h','-o','%N %G'])
excluded=','.join(sorted({line.split()[0] for line in inventory.splitlines() if not re.search('gpu:(a40|l40s):',line)}|{'bishop','conroy','cheetah'}))
common=[SL+'sbatch','--parsable','--partition=rl2-lab','--qos=debug','--account=rl2-lab','--gres=gpu:1','--exclude='+excluded,
    '--nice=30000','--cpus-per-task=4','--mem=64G','--time=03:00:00','--signal=B:USR1@600','--requeue','--chdir='+str(R)]
ledger=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),excluded=excluded,runs={})
(R/'reports/submit.intent').write_text(ledger['created_utc']);save(ledger)
audit=call(common+['--job-name=cart-msc-contracts','--output='+str(R/'reports/audit-%j.log'),
    '--export=ALL,CAMPAIGN_ROOT='+str(R),str(R/'scripts/audit.sbatch')])
assert audit.isdigit();ledger['audit_job']=audit;save(ledger)
for c in json.loads((R/'reports/plan.json').read_text())['runs']:
    rid=c['id'];row=ledger['runs'][rid]={'smoke':'INTENT'};save(ledger)
    smoke=call(common+['--dependency=afterok:'+audit,'--job-name='+rid+'-smoke','--output='+str(R/'reports'/f'{rid}-smoke-%j.log'),
        '--export=ALL,CAMPAIGN_ROOT='+str(R)+',RUN_ID='+rid+',SMOKE=1',str(R/'scripts/train.sbatch')])
    assert smoke.isdigit();row['smoke']=smoke;save(ledger)
ledger['status']='AUDIT_AND_SMOKES_SUBMITTED_FORMAL_NOT_SUBMITTED';save(ledger)
print(json.dumps(ledger,indent=2))
