"""Move only never-started owned GPU retarget tasks to tested CPU path."""
import pathlib,json,subprocess,datetime
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
report=R/'reports/drawer-pending-cpu-migration.json';assert not report.exists()
def call(args):return subprocess.run(args,capture_output=True,text=True,check=True).stdout
rows=call([S+'squeue','-r','-j','4872385','-h','-o','%i|%T|%u|%j']).splitlines()
ids=[]
for line in rows:
 jid,state,user,name=line.split('|');assert user=='cxin34' and name=='drawer-fix-bs-pilot-next20'
 if state=='PENDING':ids.append(jid)
assert ids
d=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pending_ids=ids,actions=[],status='INTENT')
report.write_text(json.dumps(d,indent=2))
for jid in ids:
 call([S+'scontrol','hold',jid]);d['actions'].append('hold '+jid);report.write_text(json.dumps(d,indent=2))
for jid in ids:
 text=call([S+'scontrol','show','job',jid]);assert 'JobState=PENDING' in text and 'UserId=cxin34' in text
indices=','.join(j.split('_')[1] for j in ids)
out=call(['python3',str(R/'scripts/submit_stage.py'),'drawer-cpu-remaining',str(R/'scripts/drawer-retarget-cpu.sbatch'),'--array='+indices+'%3','--mem=16G','--time=00:45:00','--exclude=brainiac,spot,hal','--dependency=afterany:4872385'])
receipt=json.loads((R/'reports/drawer-cpu-remaining-submit.json').read_text());assert receipt['status']=='SUBMITTED';new=receipt['job_id'];d['replacement_job']=new
call([S+'scontrol','update','JobId=4872386','Dependency=afterany:4872385:'+new]);d['actions'].append('QC dependency includes replacement '+new);report.write_text(json.dumps(d,indent=2))
for jid in ids:
 call([S+'scancel',jid]);d['actions'].append('cancel never-started '+jid);report.write_text(json.dumps(d,indent=2))
d['status']='MOVED_PENDING_ONLY';report.write_text(json.dumps(d,indent=2));print(json.dumps(d))
