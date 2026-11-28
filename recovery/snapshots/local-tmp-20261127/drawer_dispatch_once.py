"""Bounded unique downstream submissions after pilot acceptance and retarget QC."""
import pathlib,json,subprocess,fcntl,datetime,hashlib
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/';PY='/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python'
lock=(R/'reports/drawer-downstream-dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.load(open(R/'drawer/reports/downstream-pilot-review.json'))['passed']
m=json.load(open(R/'drawer/segment-manifest.json'));assigned=set();jobs=[]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p in (R/'reports').glob('drawer-downstream-*.intent.json'):
 assert p.with_name(p.name.replace('.intent.json','-submit.json')).exists(),('unresolved submission intent',str(p))
for p in sorted((R/'reports').glob('drawer-downstream-*-submit.json')):
 x=json.load(open(p));assert x['status']=='SUBMITTED',('submission uncertainty',str(p));jobs.append(x['job_id'])
 token=next(a.split('=',1)[1].split('%')[0]for a in x['command']if a.startswith('--array='))
 for v in token.split(','):
  a,b=map(int,v.split('-'))if '-'in v else(int(v),int(v))
  for i in range(a,b+1):assert i not in assigned,('duplicate index receipts',i);assigned.add(i)
unknown=[];outstanding=[];passed=[];eligible=[];quarantined=[]
for i,e in enumerate(m):
 p=pathlib.Path(e['segment']);d=p/'downstream'
 if (d/'proof.json').exists():assert json.load(open(d/'proof.json'))['passed'];passed.append(i);continue
 if(p/'downstream-geometry-quarantine.json').exists():quarantined.append(i);continue
 if(d/'failure.json').exists():unknown.append(i);continue
 if i in assigned:outstanding.append(i);continue
 if(d/'intent.json').exists():unknown.append(i);continue
 q=p/'retarget-proof-v2.json'
 if q.exists()and json.load(open(q))['passed']:eligible.append(i)
status=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),assigned=sorted(assigned),passed=passed,outstanding=outstanding,geometry_quarantined=quarantined,unresolved=unknown,eligible=eligible)
if unknown:
 status['status']='BLOCKED_WORKER_REQUIRES_AUDIT'
else:
 available=max(0,6-len(outstanding));selected=[]
 queue=subprocess.check_output([S+'squeue','-r','-h','-u','cxin34','-o','%j|%b'],text=True).splitlines()
 l40s_outstanding=sum(line.startswith('drawer-downstream-')and 'gpu:l40s:'in line for line in queue)
 gpu='l40s'if l40s_outstanding<3 else'a40'
 if gpu=='l40s':available=min(available,3-l40s_outstanding)
 status['next_gpu']=gpu
 # Validation first, then short segments, while keeping parent split frozen.
 eligible.sort(key=lambda i:(m[i]['split']!='valid',m[i]['rows_before_retarget']))
 for i in eligible:
  if len(selected)>=available:break
  reg=R/'drawer/reports'/f'registration-{i:03d}.json'
  if not reg.exists():subprocess.run([PY,str(R/'scripts/drawer_registration_audit.py'),str(i)],check=True)
  x=json.load(open(reg));p=pathlib.Path(m[i]['segment'])
  if not x['passed']:
   z=dict(status='GEOMETRY_QUARANTINED',passed=False,index=i,uuid=m[i]['uuid'],reason='Independent exact measured-knuckle/TCP registration fails unchanged 1cm/50%/5mm gate; no depth/cloud GPU submission.',registration_report=str(reg),registration_report_sha256=sha(reg),retarget_proof_sha256=sha(p/'retarget-proof-v2.json'),fix_sha256=sha(p/'retarget/fix.hdf5'),time=status['time'],training_ready=False)
   with(p/'downstream-geometry-quarantine.json').open('x')as f:json.dump(z,f,indent=2)
   continue
  selected.append(i)
 if selected:
  name='drawer-downstream-batch-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
  cmd=['python3',str(R/'scripts/submit_stage.py'),name,str(R/'scripts/drawer-downstream.sbatch'),'--array='+','.join(map(str,selected))+'%4','--gres=gpu:'+gpu+':1','--exclude=brainiac,spot,bishop,dynamics','--nice=0']
  subprocess.run(cmd,check=True);status['new_submission']=json.load(open(R/'reports'/(name+'-submit.json')));status['new_indices']=selected
 status['status']='ACTIVE'
(R/'reports/drawer-downstream-dispatch-live.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
