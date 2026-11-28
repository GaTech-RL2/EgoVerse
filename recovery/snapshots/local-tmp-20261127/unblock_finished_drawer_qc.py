import pathlib,json,subprocess,datetime
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
m=json.load(open(R/'drawer/segment-manifest.json'));indices=[2,3,4,5,7,8,9,10,11,12,13,15,17,18]
out=R/'reports/drawer-finished-qc-dependency-update.json';assert not out.exists()
a=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Existing pending QC tasks use already terminal retarget outputs; remove unrelated index21 dependency, no new jobs.',updates=[])
for i in indices:
 p=pathlib.Path(m[i]['segment'])
 for f in ['bs-gate-rgb-merge.json','retarget/fix.hdf5','retarget/fix_bs.hdf5']:assert(p/f).exists(),(i,f)
 job=f'4872386_{i}';before=subprocess.check_output([S+'scontrol','show','job',job,'-o'],text=True);assert 'UserId=cxin34('in before and 'JobState=PENDING'in before
 a['updates'].append(dict(job=job,before=before,status='INTENT'))
 out.write_text(json.dumps(a,indent=2))
 x=subprocess.run([S+'scontrol','update',f'JobId={job}','Dependency=','ExcNodeList=brainiac,spot,hal,t1000'],capture_output=True,text=True)
 a['updates'][-1].update(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,status='UPDATED'if x.returncode==0 else'BLOCKED')
 out.write_text(json.dumps(a,indent=2));assert x.returncode==0,x.stderr
print('UPDATED_EXISTING_QC',indices)
