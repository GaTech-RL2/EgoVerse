import pathlib,json,subprocess,hashlib,ast,datetime,sys
R=pathlib.Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125');S='/opt/slurm/Ubuntu-24.04/current/bin/'
i=int(sys.argv[1]);m=json.load(open(R/'drawer/segment-manifest.json'));e=m[i];p=pathlib.Path(e['segment']);old=json.load(open(p/'retarget.intent.json'));job=old['job']
state=subprocess.check_output([S+'sacct','-j',job,'-n','-X','-P','--format=JobID,State,ExitCode,NodeList'],text=True)
planned=False
if '|TIMEOUT|'not in state:
 migration=json.load(open(R/'reports/drawer-slow-cpu-migration.intent.json'))
 assert '|CANCELLED' in state and any(x['index']==i and x['job']==job for x in migration['jobs']),state
 planned=True
assert not subprocess.check_output([S+'squeue','-h','-j',job,'-o','%i'],text=True).strip()
for f in ['retarget-proof.json','retarget-proof-v2.json','retarget/fix.hdf5','retarget/fix_bs.hdf5']:assert not(p/f).exists(),f
work=p/'retarget/work';tag=e['retarget_tag']
for ext in ['_rgb224.npz','_smpl.npz','_human.csv']:assert not(work/(tag+ext)).exists(),ext
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
assert(work/(tag+'_aria60.npz')).exists()
attempt=p/'attempts'/('rgb-timeout-recovery-'+job);assert not attempt.exists();attempt.mkdir(parents=True);(attempt/'logs').mkdir()
audit=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),index=i,uuid=e['uuid'],original_intent=old,original_intent_sha256=sha(p/'retarget.intent.json'),slurm=state,planned_migration=planned,original_logs_sha256={x.name:sha(x)for x in(p/'logs').glob('*.log')},reused={str(work/(tag+'_aria60.npz')):sha(work/(tag+'_aria60.npz'))},reason='Verified terminal old CPU allocation with no RGB/SMPL/fix/BS outputs; recover only unfinished RGB and subsequent stages on another CPU node. Preserve original intent/logs and completed tracking input.',attempt=str(attempt))
ap=R/'reports'/f'drawer-rgb-timeout-recovery-{i}-audit.json'
with ap.open('x')as f:json.dump(audit,f,indent=2)
t=(R/'scripts/drawer_retarget_cpu.py').read_text()
guard=f"assert int(sys.argv[1])=={i}\nassert json.load(open(p/'retarget.intent.json'))['job']=={job!r}\nattempt=Path({str(attempt)!r})\nassert not(attempt/'retarget.intent.json').exists(),'Unresolved recovery'\nassert not(p/'retarget/fix.hdf5').exists()"
t=t.replace("assert not(p/'retarget.intent.json').exists(),'Unresolved attempt; audit required'",guard).replace("(p/'retarget.intent.json').write_text","(attempt/'retarget.intent.json').write_text").replace("with(p/'logs'/(name+'.log')).open('x')","with(attempt/'logs'/(name+'.log')).open('x')").replace("(p/'processing-error.json').write_text","(attempt/'processing-error.json').write_text")
stem=f'drawer_rgb_timeout_recovery_{i}'
ast.parse(t);(R/'scripts'/(stem+'.py')).write_text(t)
w=(R/'scripts/drawer_retarget_cpu_then_qc.py').read_text().replace('drawer_retarget_cpu.py',stem+'.py');ast.parse(w);(R/'scripts'/(stem+'_then_qc.py')).write_text(w)
b=(R/'scripts/drawer-retarget-cpu.sbatch').read_text().replace('drawer_retarget_cpu_then_qc.py',stem+'_then_qc.py');sp=R/'scripts'/(stem+'.sbatch');sp.write_text(b)
subprocess.run(['python3',str(R/'scripts/submit_stage.py'),f'drawer-rgb-timeout-recovery-{i}',str(sp),'--array='+str(i),'--mem=16G','--time=01:00:00','--exclude=brainiac,spot,hal,t1000,vicki'],check=True)
