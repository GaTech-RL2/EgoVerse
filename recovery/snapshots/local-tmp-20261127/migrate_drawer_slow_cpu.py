import pathlib,json,subprocess,datetime,hashlib,fcntl
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
l=(R/'reports/drawer-slow-cpu-migration.lock').open('a');fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=R/'reports/drawer-slow-cpu-migration.intent.json';assert not out.exists()
m=json.load(open(R/'drawer/segment-manifest.json'));a=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Bounded migration of only four unfinished RGB extraction stages after 19-37min on slow nodes; all existing successful artifacts and training jobs untouched.',jobs=[])
for i in [21,97,137,138]:
 e=m[i];p=pathlib.Path(e['segment']);old=json.load(open(p/'retarget.intent.json'));job=old['job'];w=p/'retarget/work';tag=e['retarget_tag']
 for f in ['retarget-proof.json','retarget-proof-v2.json','retarget/fix.hdf5','retarget/fix_bs.hdf5']:assert not(p/f).exists(),(i,f)
 for ext in ['_rgb224.npz','_smpl.npz','_human.csv']:assert not(w/(tag+ext)).exists(),(i,ext)
 q=subprocess.check_output([S+'scontrol','show','job',job,'-o'],text=True);assert 'UserId=cxin34('in q and 'JobState=RUNNING'in q and ('NodeList=t1000 'in q or 'NodeList=hal 'in q)
 a['jobs'].append(dict(index=i,job=job,uuid=e['uuid'],before=q,original_intent_sha256=hashlib.sha256((p/'retarget.intent.json').read_bytes()).hexdigest()))
with out.open('x')as f:json.dump(a,f,indent=2)
subprocess.run([S+'scontrol','hold','4872386_21'],check=True)
for x in a['jobs']:
 p=subprocess.run([S+'scancel',x['job']],capture_output=True,text=True);x.update(cancel_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 (R/'reports/drawer-slow-cpu-migration-cancel.json').write_text(json.dumps(a,indent=2));assert p.returncode==0,p.stderr
print('OLD_ALLOCATIONS_CANCEL_REQUESTED; verify terminal before any recovery')
