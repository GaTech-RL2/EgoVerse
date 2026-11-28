import pathlib,json,subprocess,datetime,hashlib,fcntl
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
lock=(R/'reports/drawer-controller-resume95.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
m=json.load(open(R/'drawer/segment-manifest.json'));p=pathlib.Path(m[95]['segment'])
assert json.load(open(p/'downstream/proof.json'))['passed']
s='/opt/slurm/Ubuntu-24.04/current/bin/'
a=subprocess.check_output([s+'sacct','-j','4872751_95','-n','-P','-o','JobID,State,ExitCode'],text=True)
assert '4872751_95|COMPLETED|0:0' in a,a
session='cxin34-drawer-downstream-20261125'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
assert not (R/'reports/drawer-controller-resume95.intent.json').exists()
q=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),recovery_job='4872751_95',sacct=a,proof_sha256=hashlib.sha256((p/'downstream/proof.json').read_bytes()).hexdigest(),session=session,reason='Unique failed CUDA initialization recovery passed; same locked dispatcher resumes unassigned eligible stages only.')
cmd=f'python3 -u {R}/scripts/drawer_downstream_controller.py > {R}/logs/drawer-downstream-controller-resume95.log 2>&1'
q['command']=cmd
with (R/'reports/drawer-controller-resume95.intent.json').open('x')as f:json.dump(q,f,indent=2)
x=subprocess.run(['tmux','new-session','-d','-s',session,cmd],capture_output=True,text=True)
q.update(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr)
with (R/'reports/drawer-controller-resume95.json').open('x')as f:json.dump(q,f,indent=2)
assert x.returncode==0,q
print(json.dumps(q))
