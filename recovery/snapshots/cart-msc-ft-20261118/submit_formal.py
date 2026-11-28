import fcntl,json,subprocess
from pathlib import Path
R=Path('/coc/flash7/cxin34/cart-msc-ft-20261118');SL='/opt/slurm/Ubuntu-24.04/current/bin/'
lock=(R/'reports/submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=R/'reports/jobs.json';j=json.loads(p.read_text())
assert not (R/'reports/formal-submit.intent').exists()
assert json.loads((R/'reports/photometric-audit.json').read_text())['passed']
(R/'reports/formal-submit.intent').write_text('Two authorized FT10k runs; automatic fail-closed independent smoke gates')
def save():
    t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
def call(args):
    x=subprocess.check_output(args,text=True).strip();assert x.isdigit();return x
common=[SL+'sbatch','--parsable','--partition=rl2-lab','--qos=debug','--account=rl2-lab','--nice=30000',
 '--cpus-per-task=4','--mem=64G','--time=03:00:00','--chdir='+str(R)]
for rid,row in j['runs'].items():
    row['gate']='INTENT';save()
    export='--export=ALL,CAMPAIGN_ROOT='+str(R)+',RUN_ID='+rid
    row['gate']=call(common+['--dependency=afterok:'+row['smoke'],'--job-name='+rid+'-audit',
        '--output='+str(R/'reports'/f'{rid}-gate-%j.log'),export,str(R/'scripts/gate.sbatch')]);save()
    row['formal']='INTENT';save()
    row['formal']=call(common+['--gres=gpu:1','--exclude='+j['excluded'],'--signal=B:USR1@600','--requeue',
        '--dependency=afterok:'+row['gate'],'--job-name='+rid,'--output='+str(R/'reports'/f'{rid}-%j.log'),
        export+',SMOKE=0',str(R/'scripts/formal.sbatch')]);save()
j['status']='TWO_FORMAL_CHAINS_SUBMITTED_BEHIND_INDEPENDENT_GATES';save();print(json.dumps(j,indent=2))
