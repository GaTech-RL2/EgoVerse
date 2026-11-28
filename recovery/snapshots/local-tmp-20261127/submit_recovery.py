import json,subprocess,time,fcntl
from pathlib import Path
R=Path('/coc/flash7/cxin34/cloth-valid-recovery-fix1124-20261124');S='/opt/slurm/Ubuntu-24.04/current/bin/'
f=(R/'reports/submit.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
def submit(name,opts,script):
 p=R/'reports'/(name+'.json');intent=R/'reports'/(name+'.intent')
 if p.exists():
  q=json.load(open(p));assert q['returncode']==0;return q['stdout'].strip().split(';')[0]
 assert not intent.exists()
 cmd=[S+'sbatch','--parsable','--account=rl2-lab','--qos=debug',*opts,'--job-name=cloth-recovery-'+name,'--output='+str(R/'logs'/(name+'-%A_%a.log')),str(R/'scripts'/script)]
 intent.write_text(json.dumps(dict(command=cmd,time=time.time()),indent=2));z=subprocess.run(cmd,text=True,capture_output=True);p.write_text(json.dumps(dict(command=cmd,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr),indent=2));assert z.returncode==0,(z.stdout,z.stderr);return z.stdout.strip().split(';')[0]
build=submit('build',['--partition=interactive-cpu','--exclude=hal,bmo','--cpus-per-task=2','--mem=12G','--time=01:00:00','--array=0-14%2'],'build.sbatch')
base=json.load(open('/coc/flash7/cxin34/egoa-target-retarget-fix-20261124/reports/bulk-retarget-submit.json'))['command'];exclude=next(x for x in base if x.startswith('--exclude='))
pilot=submit('pilot',['--partition=rl2-lab',exclude,'--gres=gpu:a40:1','--cpus-per-task=2','--mem=24G','--time=01:00:00','--array=0','--dependency=afterok:'+build+'_0'],'segment.sbatch')
rest=submit('segments',['--partition=rl2-lab',exclude,'--gres=gpu:a40:1','--cpus-per-task=2','--mem=24G','--time=01:00:00','--array=1-17%1','--dependency=afterok:'+build+':'+pilot],'segment.sbatch')
print(json.dumps(dict(build=build,pilot=pilot,segments=rest)))
